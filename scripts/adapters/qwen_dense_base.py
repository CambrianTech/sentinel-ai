"""QwenDenseBase — shared base for the Qwen-family dense adapters.

Extracted from Qwen3DenseAdapter and Qwen2DenseAdapter once both siblings
existed with proven Tier 1 dispatch behavior. Per the OOP rule from
~/.claude/.../memory/feedback_adapters_not_branches.md and CLAUDE.md's
"outlier validation strategy": write two siblings first, prove they work,
THEN extract the base from the shared 80%.

What's shared (lives here in the base):
    - prune(): compute_head_importance → forge_model.prune (forward_hooks
      strategy) → immediate defrag_inline.defrag_live_model → per-layer
      importance bookkeeping. forge_model.prune is architecture-agnostic
      for dense Qwen-family models, so this body works for Qwen2.5,
      Qwen3.5, and any future dense Qwen variant.
    - train(): handles BOTH normal recovery LoRA (forge_model.train_lora)
      AND § 4.1.3.3 compensation distillation (compensation_lora.py).
      Internal dispatch on the presence of a 'teacher' field in stage
      params — that's how the alloy declares "this is a compensation
      distillation, not normal recovery training." Both flow through
      the same .train() method because the alloy uses the 'lora' stage
      type for both shapes.

What stays family-specific (overridden in subclasses):
    - architectures tuple — each subclass declares its source.architecture string
    - context_extend() — Qwen3DenseAdapter has the qwen3.5-4b-code-128k-forged
      YaRN path; Qwen2DenseAdapter doesn't ship a context-extended variant
      so it inherits the base's no-op default

The two adapters that inherit from this base:
    Qwen3DenseAdapter  → architectures = ("qwen3_5",)  (the 11 published Qwen3.5 catalog artifacts)
    Qwen2DenseAdapter  → architectures = ("qwen2",)    (the qwen2.5-{0.5b,1.5b,3b}-general + qwen2.5-coder-7b-compacted)

Future dense Qwen-family adapters (a hypothetical Qwen3.5-VL dense path,
a Qwen3.6 family if it ships) inherit from QwenDenseBase by default.
A new dense family that ISN'T Qwen (Llama, Mistral, etc) gets its own
base if its forge_model code path differs — but for the Qwen lineage
specifically, forge_model.prune handles all of them via the same code.

Tier 2 status:
    prune()              — REAL (lazy imports torch + forge_model)
    train() (recovery)   — REAL (lazy imports torch + forge_model)
    train() (compensation) — Tier 2 STUB (raises if ctx.model is non-None
                              with the right error pointing at compensation_lora.py).
                              The §4.1.3.3 flagship artifact (qwen2.5-coder-7b-compacted)
                              shipped via direct CLI invocation of compensation_lora.py;
                              wiring it through the adapter is roadmap step 3 of the
                              plugin sprint.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from .base import FamilyAdapter

if TYPE_CHECKING:
    from scripts.stages.base import ForgeContext


class QwenDenseBase(FamilyAdapter):
    """Shared base for the Qwen-family dense adapters.

    Subclasses MUST set the .architectures tuple. They MAY override prune,
    train, or context_extend if their family needs different behavior; in
    practice the inherited bodies handle every Qwen-family dense forge
    that's shipped to date.
    """

    # Subclasses set this. Empty here means this base is not directly registered
    # — it has no architecture string of its own and only gets used via inheritance.
    architectures: tuple[str, ...] = ()

    # ── prune ────────────────────────────────────────────────────────────────

    def prune(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Dense head pruning via forge_model.prune + immediate defrag.

        Param contract (from the alloy's prune stage):
            strategy: str             — entropy | magnitude | activation-magnitude | gradient
            level: float              — fraction of heads to remove (0.0..0.9)
            minHeadsPerLayer: int     — floor per layer (optional)
            minKvHeadsPerLayer: int   — KV-head floor (optional)
            analysisSteps: int        — importance computation steps (optional)
            perLayerNormalized: bool  — true for the §4.1.3.1 fix path (optional)
            defragMode: str           — defrag strategy (optional)
            notes: str                — methodology blockquote (optional)

        Tier 2 (live execution): imports torch + forge_model lazily so Tier 1
        dispatch resolution stays torch-free. If ctx.model is None
        (dispatch-only / dry-run), short-circuits without raising.
        """
        level = float(params.get("level", 0.3))
        strategy = params.get("strategy", "entropy")
        per_layer_normalized = params.get("perLayerNormalized")
        suffix = f" (perLayerNormalized={per_layer_normalized})" if per_layer_normalized is not None else ""
        self.log(f"Pruning {level:.0%} heads via {strategy}{suffix}")

        if ctx.model is None:
            self.log("  No model loaded — prune deferred (dispatch-only path)")
            return ctx

        # Lazy imports — Tier 1 dispatch must work without torch installed.
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from forge_model import prune, compute_head_importance

        # Compute importance BEFORE pruning — this is the data quant needs
        importance = compute_head_importance(ctx.model, ctx.info)

        heads, hooks = prune(ctx.model, level, ctx.info, "forward_hooks")
        ctx.hooks.extend(hooks)
        self.log(f"  Pruned {len(heads)} head groups")

        # CRITICAL: defrag IMMEDIATELY after prune, BEFORE training.
        # If we don't, LoRA trains on the full original projections with
        # hooks zeroing pruned outputs. The LoRA updates on pruned heads
        # become noise that destroys the model when hooks are removed.
        # Defragging now means the surviving smaller model is what trains.
        try:
            from defrag_inline import defrag_live_model
            self.log("  Defragging pruned heads into surviving structure...")
            freed = defrag_live_model(ctx.model, dead_heads=heads)
            self.log(f"  Freed {freed / 1e6:.0f}MB — model now operates on surviving heads only")
            # Hooks no longer needed — pruned heads are physically gone
            for h in ctx.hooks:
                h.remove()
            ctx.hooks.clear()
            ctx.dead_heads = heads
        except Exception as e:
            self.log(f"  WARNING: defrag failed ({e}) — falling back to hooks (LoRA may corrupt model)")

        # Save per-layer importance profile for variable quantization
        n_layers = importance.shape[0]
        layer_importance = []
        for li in range(n_layers):
            finite = importance[li][importance[li] < float("inf")]
            if len(finite) == 0:
                layer_importance.append({
                    "layer": li, "avgImportance": 0,
                    "survivingHeads": 0, "totalHeads": 0,
                })
                continue
            total_heads = len(finite)
            pruned_heads = len(heads.get(li, []))
            surviving = total_heads - pruned_heads
            layer_importance.append({
                "layer": li,
                "avgImportance": round(finite.mean().item(), 4),
                "minImportance": round(finite.min().item(), 4),
                "maxImportance": round(finite.max().item(), 4),
                "survivingHeads": surviving,
                "totalHeads": total_heads,
                "prunedHeads": pruned_heads,
            })

        # Store on context for quant stage to read
        if not hasattr(ctx, "layer_importance"):
            ctx.layer_importance = []
        ctx.layer_importance = layer_importance

        # Also save to alloy results for the card and verification
        if not hasattr(ctx, "alloy") or ctx.alloy is None:
            ctx.alloy = {}
        if not isinstance(ctx.alloy.get("results"), dict):
            ctx.alloy["results"] = {}
        ctx.alloy["results"]["layerImportance"] = layer_importance

        self.log(f"  Layer importance saved ({n_layers} layers)")
        importances = [l["avgImportance"] for l in layer_importance if l["avgImportance"] > 0]
        if importances:
            self.log(f"  Importance range: {min(importances):.3f} — {max(importances):.3f}")

        return ctx

    # ── train ────────────────────────────────────────────────────────────────

    def default_train_params(self, ctx: "ForgeContext") -> dict:
        """Qwen dense default training profile.

        Smart logic:
          - Coder source models (name contains 'coder' or 'code') get a
            code domain + code corpus
          - General models get wikitext text recovery
          - Step count and LR scale gently with model size

        Recipe authors override anything they want via stage params.
        Everything here is a SAFE default, not a hardcoded constant —
        the right value comes from the model + the source.architecture,
        not from the seeder.
        """
        base_model = (ctx.alloy.get("source") or {}).get("baseModel", "").lower()
        is_coder = "coder" in base_model or "code" in base_model

        # Total params from source.totalParamsB (set by HF-verified
        # geometry in the seeder), default to 7 if not declared
        total_b = (ctx.alloy.get("source") or {}).get("totalParamsB") or 7.0

        # Steps scale: small models need more recovery, big ones less
        # (the prune surface is bigger but each gradient is more expensive)
        if total_b < 5:
            steps = 300
        elif total_b < 15:
            steps = 200
        elif total_b < 35:
            steps = 150
        else:
            steps = 100

        # LR scales inverse to model size — bigger models need smaller LR
        if total_b < 5:
            lr = "1e-4"
        elif total_b < 15:
            lr = "5e-5"
        else:
            lr = "2e-5"

        if is_coder:
            return {
                "domain": "code",
                "dataset": "m-a-p/CodeFeedback-Filtered-Instruction",
                "steps": steps,
                "learningRate": lr,
                "batchSize": 4,
            }
        return {
            "domain": "wikitext",
            "dataset": "Salesforce/wikitext",
            "steps": steps,
            "learningRate": lr,
            "batchSize": 4,
        }

    def train(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """LoRA training — dispatches between recovery and compensation.

        The same .train() method handles both shapes because the alloy uses
        the 'lora' stage type for both. The discrimination is by content,
        not by stage name: presence of a 'teacher' field signals
        § 4.1.3.3 compensation distillation against an unmodified teacher;
        absence signals normal recovery LoRA on a domain dataset.

        Recovery LoRA param contract:
            domain: str              — code | general | math | ...
            dataset: str             — HF dataset id (e.g. 'Salesforce/wikitext')
            steps: int
            learningRate: str        — float-as-string per the alloy spec
            batchSize: int           — optional, defaults to 4

        Compensation distillation param contract (§4.1.3.3):
            teacher: str             — unmodified teacher model id
            calibrationDataset: str
            kdTemperature: float     — typically 2.0
            loraRank: int
            loraAlpha: int
            targetModules: list[str]
            lossType: str            — 'kl_logits' | 'mse_hidden' | 'both'
            mergedAtSave: bool
            teacherPrecision: str
            studentPrecision: str
            steps: int
            learningRate: str
            domain: str

        Tier 2:
            Recovery LoRA path is REAL (lazy-imports forge_model.train_lora).
            Compensation distillation path is a Tier 2 STUB (raises with a
            clear pointer to scripts/compensation_lora.py until adapter
            wiring lands per roadmap step 3).
        """
        teacher = params.get("teacher")

        if teacher:
            return self._train_compensation(ctx, **params)
        else:
            return self._train_recovery(ctx, **params)

    def _train_recovery(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Normal recovery LoRA path — forge_model.train_lora."""
        domain = params.get("domain", "code")
        steps = int(params.get("steps", 1000))
        lr = float(params.get("learningRate", "2e-4"))
        self.log(f"Training {steps} steps on {domain} data, lr={lr}")

        if ctx.model is None:
            self.log("  No model loaded — train deferred (dispatch-only path)")
            return ctx

        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import torch
        from forge_model import train_lora, make_dataloaders, ForgeConfig, evaluate

        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        cfg = ForgeConfig.auto(ctx.info.get("fp16_gb", 8), vram_gb, ctx.load_4bit)

        train_loader, eval_loader = make_dataloaders(ctx.tokenizer, cfg, domain)
        ctx.model = train_lora(ctx.model, train_loader, cfg, steps, lr, ctx.output_dir)

        post_train = evaluate(ctx.model, eval_loader, ctx.output_dir, "post-train")
        self.log(f"  Post-train perplexity: {post_train['perplexity']:.2f}")

        return ctx

    def _train_compensation(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """§ 4.1.3.3 KL-distillation-against-teacher compensation LoRA.

        Param contract (from the alloy's compensation lora stage):
            teacher:            HF id or path of the unmodified teacher
            teacherPrecision:   '8bit' | '4bit' (defaults to '8bit')
            calibrationDataset: path to the held-out calibration JSONL
            kdTemperature:      float (currently unused — temperature is fixed
                                in compute_distillation_loss; future work
                                threads it through)
            loraRank:           int
            loraAlpha:          int
            targetModules:      list[str] of LoRA target module names
            lossType:           'kl_logits' | 'mse_hidden' | 'both'
            steps:              int
            learningRate:       string (parsed to float)
            mergedAtSave:       bool (currently always merged via merge_and_unload)
            maxLength:          int (defaults 1024)

        Loads the teacher fresh in the requested quant tier (the adapter
        does NOT preload the teacher because it's only needed for the
        compensation step), runs the distillation training loop against
        ctx.model + ctx.tokenizer, merges the LoRA into the student
        weights, saves the compensated student to ctx.output_dir/compensated,
        reloads ctx.model from the compensated dir.
        """
        teacher_path = params.get("teacher")
        teacher_quant = params.get("teacherPrecision", "8bit")
        calibration_data = params.get("calibrationDataset")
        loss_type = params.get("lossType", "kl_logits")
        lora_rank = int(params.get("loraRank", 16))
        lora_alpha = int(params.get("loraAlpha", 32))
        target_modules = params.get("targetModules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ])
        steps = int(params.get("steps", 500))
        lr = float(params.get("learningRate", "1e-4"))
        max_length = int(params.get("maxLength", 1024))

        self.log(
            f"compensation-LoRA § 4.1.3.3 — teacher={teacher_path} ({teacher_quant}), "
            f"loss={loss_type}, loraRank={lora_rank}, steps={steps}"
        )

        if ctx.model is None:
            self.log("  No model loaded — compensation train deferred (dispatch-only path)")
            return ctx

        if not teacher_path:
            raise ValueError(
                f"{self.name}._train_compensation: alloy stage missing 'teacher' "
                f"field. The §4.1.3.3 compensation distillation requires a teacher "
                f"model id or path."
            )
        if not calibration_data:
            raise ValueError(
                f"{self.name}._train_compensation: alloy stage missing "
                f"'calibrationDataset' field. The §4.1.3.4.1 discipline gate "
                f"requires the held-out calibration corpus to be declared."
            )

        # Lazy import — Tier 1 dispatch must work without torch / transformers / peft.
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from compensation_lora import compensate_lora

        # Resolve calibration corpus relative to ctx.output_dir if not absolute.
        corpus_path = Path(calibration_data)
        if not corpus_path.is_absolute():
            corpus_path = (ctx.output_dir / corpus_path).resolve()

        compensated_out = (ctx.output_dir / "compensated").resolve()

        result = compensate_lora(
            student=ctx.model,
            student_tokenizer=ctx.tokenizer,
            teacher_path=teacher_path,
            teacher_quant=teacher_quant,
            calibration_data=corpus_path,
            output=compensated_out,
            steps=steps,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            learning_rate=lr,
            loss_type=loss_type,
            target_modules=list(target_modules),
            max_length=max_length,
        )

        # Reload ctx.model from the compensated dir so downstream stages see
        # the merged-and-unloaded compensated student rather than the
        # in-memory pre-compensation original.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.log(f"  reloading compensated student from {compensated_out}")
        del ctx.model
        torch.cuda.empty_cache()
        ctx.model = AutoModelForCausalLM.from_pretrained(
            str(compensated_out),
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        ctx.tokenizer = AutoTokenizer.from_pretrained(
            str(compensated_out), trust_remote_code=True,
        )
        ctx.compensated_model_dir = str(compensated_out)
        self.log(
            f"  compensation complete: {result['steps_completed']} steps, "
            f"final_loss={result['final_loss']:.6f}"
        )

        return ctx
