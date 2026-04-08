"""Qwen3DenseAdapter — family adapter for the Qwen3.5 dense catalog.

Handles every published continuum-ai/qwen3.5-*-forged* artifact whose source
declares architecture='qwen3_5' and is NOT a MoE variant. As of 2026-04-08
that's 11 published HuggingFace artifacts:

    qwen3.5-0.8b-general-forged
    qwen3.5-2b-general-forged
    qwen3.5-4b-general-forged
    qwen3.5-4b-code-forged          (+ -GGUF, -defragged variants)
    qwen3.5-4b-code-128k-forged
    qwen3.5-9b-general-forged       (highest downloads — 2.5K)
    qwen3.5-27b-code-forged         (+ -defragged, -mlx-4bit variants)

Reproducibility contract: this adapter MUST stay frozen. Methodology
improvements (new pruning metrics, new training schedules, new defrag
strategies) arrive as NEW adapters with NEW architecture strings or NEW
alloy field discriminators. The 11 artifacts above were forged with the
behavior wired here, and the reproducibility test is the proof — if a
future change to this file produces a different safetensors hash on any
of the 11 published alloys, the change is wrong by definition.

Tier 1 status (current): dispatch resolution only. The stage handler methods
report what they would execute but don't actually load a model or touch
tensors. Tier 2 (real forging) lights up after the dispatch contract is
proven and tested against all 11 artifacts.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from .base import FamilyAdapter
from .registry import register_family_adapter

if TYPE_CHECKING:
    from scripts.stages.base import ForgeContext


@register_family_adapter
class Qwen3DenseAdapter(FamilyAdapter):
    """Family adapter for Qwen3.5 dense models (the legacy catalog).

    Tensor layout: standard transformers Qwen3 dense — attention with
    grouped-query KV heads, MLP with gate/up/down projections, no MoE
    routing. Module tree: model.layers.{i}.{self_attn,mlp}.{...}.

    Forge concerns this family supports:
        - prune (entropy / magnitude / activation-magnitude head importance)
        - train (LoRA on dense projections)
        - context-extend (YaRN, used for the 4b-code-128k variant)

    Forge concerns this family does NOT support:
        - expert-prune (this is a dense family)
        - expert-activation-profile (MoE-only)
        - compensation-lora (the v2-7b dense compensated path uses qwen2,
          not qwen3_5; if a Qwen3.5 dense compensated alloy ever ships, it
          will be a separate adapter or this one gets the method overridden)
        - modality (the legacy Qwen3.5 catalog stripped vision; future
          VL-preserved Qwen3.5 forges will use a different adapter that
          consults vision_safety.py)
    """

    architectures = ("qwen3_5",)

    def prune(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Dense head pruning via forge_model.prune + immediate defrag.

        Param contract (from the alloy's prune stage):
            strategy: str    — entropy | magnitude | gradient | activation-magnitude
            level: float     — fraction of heads to remove (0.0..0.9)

        Body moved here from scripts/stages/transform_stages.py::PruneExecutor
        as part of the family-adapter refactor. PruneExecutor now delegates
        to the family adapter resolved from ctx.alloy['source']['architecture'].

        Tier 2 (live execution): imports torch + forge_model lazily so Tier 1
        dispatch resolution stays torch-free. If ctx.model is None (dispatch-
        only / dry-run), this method short-circuits without raising.
        """
        level = float(params.get("level", 0.3))
        strategy = params.get("strategy", "entropy")
        self.log(f"Pruning {level:.0%} heads via {strategy}")

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

    def train(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """LoRA training on a domain dataset.

        Param contract:
            domain: str        — code | general | math | ...
            dataset: str       — HF dataset id (e.g. 'Salesforce/wikitext')
            steps: int
            learningRate: str  — float-as-string per the alloy spec
            batchSize: int     — optional, defaults to 4

        Body moved here from PruneExecutor's sibling TrainExecutor.
        Same lazy-import pattern as prune() above.
        """
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

    def context_extend(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Context-window extension via YaRN — used by qwen3.5-4b-code-128k-forged.

        Param contract:
            targetLength: int   — e.g. 131072
            method: str         — yarn | ntk | linear | dynamic-ntk

        Currently a Tier 2 stub. The existing context-extend path lives in
        scripts/stages/input_stages.py::ContextExtendExecutor and is family-
        agnostic; this override exists so the dispatch test acknowledges
        Qwen3DenseAdapter handles the stage. Wiring to the real implementation
        lands when Tier 2 reproducibility for the 4b-code-128k variant runs.
        """
        target_length = params.get("targetLength")
        method = params.get("method", "yarn")
        self.log(f"context-extend → {target_length} via {method} (Tier 2 stub)")
        return ctx
