"""MoEUnfusedExpertsBase — shared base for MoE families with unfused experts.

Extracted from Qwen3MoEAdapter and OlmoeAdapter once both siblings existed
with proven Tier 1 dispatch behavior. Same OOP rule as QwenDenseBase: write
two siblings first, prove they work, THEN extract the base.

What "unfused experts" means: the expert weights live as separate parameter
tensors per expert per projection, in a module tree shaped like:

    model.layers.{i}.mlp.experts.{e}.{gate_proj,up_proj,down_proj}
    model.layers.{i}.mlp.gate     ← router

This is the layout Qwen3MoE and OLMoE both use. Other MoE families use
DIFFERENT layouts:

    Mixtral:        model.layers.{i}.block_sparse_moe.experts.{e}.w[123]
    Phi-MoE:        block_sparse_moe.experts.{e}.fc[12]
    Granite-MoE:    a "fused" layout where all experts share one big tensor
    DeepSeek-V2:    routed experts + shared expert in a hybrid layout

Each of those would need either its own base class OR an extension to the
expertTensorLayout dispatch inside this base. The current code path here
assumes the unfused layout that Qwen3MoE and OLMoE share. Adding Mixtral
support is roadmap-step-3 territory: extract a layout-dispatch helper or
add a Mixtral-specific subclass that overrides expert_prune entirely.

DO NOT bolt `if architectures[0] == "mixtral"` branches into this base.
That's the failure mode the never-branch rule exists to prevent.

What's shared (lives here in the base):
    - expert_activation_profile(): § 4.1.3.4 calibration-aware MoE expert
      importance profiling. Reads the calibration corpus + corpus file path
      from the alloy stage params, lazy-imports
      scripts/expert_activation_profile.py, runs the profile, writes the
      importance JSON, returns the mutated context. Tier 2 STUB today
      (raises with a clear pointer to the existing CLI script) until
      roadmap step 3 wires it.
    - expert_prune(): per-layer top-K removal keyed to the importance JSON
      from the upstream profiling stage. Lazy-imports
      scripts/cpu_expert_prune_v2.py with --importance-json, runs the
      prune, returns the mutated context. Tier 2 STUB today.

What stays family-specific (overridden in subclasses):
    - architectures tuple — each subclass declares its source.architecture string
    - Anything that requires tensor walks specific to a non-unfused layout.
      Today Qwen3MoE + OLMoE both inherit untouched. When a fused-layout
      family ships (e.g. GraniteMoEAdapter), its expert_prune() override
      handles the layout difference there, NOT here.

The two adapters that inherit from this base:
    Qwen3MoEAdapter  → architectures = ("qwen3_moe",)  (qwen3-coder-30b-a3b-compacted-19b-256k, the §4.1.3.4 anchor)
    OlmoeAdapter     → architectures = ("olmoe",)      (olmoe-1b-7b-compacted-5b, the §4.1.3.4 cross-arch anchor)

Tier 2 status:
    expert_activation_profile() — REAL (lazy-imports
                                    expert_activation_profile.profile_experts
                                    and calls it on the loaded model)
    expert_prune()              — REAL (lazy-imports
                                    cpu_expert_prune_v2.prune_experts and
                                    calls it on the model_dir on disk;
                                    reloads ctx.model from the pruned dir
                                    afterward so downstream stages see
                                    the smaller model)
    Both short-circuit cleanly when ctx.model is None (Tier 1 dispatch
    path stays working).

Reproducibility contract: this base + its subclasses MUST stay frozen
against the two published §4.1.3.4 anchor artifacts. Methodology
improvements ship as NEW adapters with NEW discriminators, never as
edits to the existing methods. The negative-baseline cells preserved
in priorMetricBaselines[] are what make the §4.1.3.4 claim falsifiable;
this base's behavior is the positive cell.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from .base import FamilyAdapter

if TYPE_CHECKING:
    from scripts.stages.base import ForgeContext


class MoEUnfusedExpertsBase(FamilyAdapter):
    """Shared base for MoE families with the unfused-experts module-tree
    layout. Subclasses MUST set the .architectures tuple."""

    architectures: tuple[str, ...] = ()  # subclass overrides

    # ── expert-activation-profile ────────────────────────────────────────────

    def expert_activation_profile(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """§ 4.1.3.4 calibration-aware MoE expert importance profiling.

        Param contract (from the alloy's expert-activation-profile stage):
            calibrationCorpus: str        — corpus identifier (e.g. 'heldout_code300')
            calibrationCorpusFile: str    — path to the .jsonl file
                                            (sha256 in the alloy's calibrationCorpora)
            calibrationExamples: int      — number of held-out examples
            calibrationTokens: int        — total token count across the corpus
            implementation: str           — script reference
            metricVersion: str            — '4.1.3.4' for the calibration-aware version
            metric: str                   — 'activation_count' (positive cell)
            notes: str                    — methodology blockquote that lands on the model card

        Output: an importance JSON written to ctx.output_dir, consumed by
        the downstream expert-prune stage via its importanceJson field.

        Tier 2 wires to scripts/expert_activation_profile.py. Tier 1: the
        method short-circuits cleanly when ctx.model is None (dispatch path).
        """
        corpus_file = params.get("calibrationCorpusFile") or params.get("calibrationCorpus")
        examples = params.get("calibrationExamples")
        tokens = params.get("calibrationTokens")
        metric = params.get("metric", "activation_count")
        max_length = int(params.get("maxLength", 2048))
        self.log(
            f"expert-activation-profile § 4.1.3.4 — metric={metric}, corpus={corpus_file}, "
            f"{examples} examples, {tokens} tokens"
        )

        if ctx.model is None:
            self.log("  No model loaded — profile deferred (dispatch-only path)")
            return ctx

        if not corpus_file:
            raise ValueError(
                f"{self.name}.expert_activation_profile: alloy stage missing "
                f"'calibrationCorpusFile' (or 'calibrationCorpus'). The §4.1.3.4 "
                f"calibration-aware metric requires a held-out corpus path. "
                f"Check the alloy's expert-activation-profile stage params."
            )

        # Lazy import — Tier 1 dispatch must work without torch installed.
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from expert_activation_profile import profile_experts

        # Resolve the corpus path. Recipes typically declare a relative path
        # like 'calibration/heldout_code300.jsonl'; resolve it against the
        # forge output dir if it's not absolute.
        corpus_path = Path(corpus_file)
        if not corpus_path.is_absolute():
            corpus_path = (ctx.output_dir / corpus_path).resolve()
        if not corpus_path.exists():
            raise FileNotFoundError(
                f"calibration corpus {corpus_path} does not exist. The §4.1.3.4.1 "
                f"discipline gate requires the corpus file to be present and "
                f"hash-pinned. Make sure publish_model.py uploaded it to the "
                f"forge output dir before this stage runs."
            )

        # Output the importance JSON next to the model so the downstream
        # expert-prune stage can find it without having to know its name in advance.
        importance_path = (ctx.output_dir / "importance.activation_count.json").resolve()

        # Determine the device the loaded model is on. ctx.device is the
        # cuda string used by alloy_executor when it loaded the model.
        device = ctx.device or "cuda:0"

        result = profile_experts(
            model=ctx.model,
            tokenizer=ctx.tokenizer,
            calibration_data=corpus_path,
            output=importance_path,
            max_length=max_length,
            device=device,
            model_label=ctx.model_name or type(ctx.model).__name__,
        )

        # Stash the importance JSON path on ctx so expert_prune() can find it.
        # Use a typed attribute so the contract is explicit.
        ctx.importance_json_path = str(importance_path)
        self.log(f"  importance written: {importance_path}")
        self.log(
            f"  profiled {result['calibration_examples']} examples, "
            f"{result['calibration_tokens']} tokens, "
            f"{result['num_hidden_layers']} layers × {result['num_experts']} experts"
        )

        return ctx

    # ── expert-prune ─────────────────────────────────────────────────────────

    def expert_prune(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Per-layer top-K MoE expert removal keyed to the §4.1.3.4 importance.

        Param contract (from the alloy's expert-prune stage):
            strategy: str                  — 'calibration-aware-activation-count' (the §4.1.3.4 metric)
                                             or 'router-gate-l2' (the negative-baseline anchor)
            metric: str                    — 'activation_count' (positive cell)
            metricSource: str              — script reference
            keepExpertsPerLayer: int       — survivors per layer
            originalExpertsPerLayer: int   — pre-prune count
            prunePct: float                — % removed
            expertsDropped: int            — total experts removed across all layers
            expertsRenamed: int            — total surviving experts renumbered
            routerSlicedLayers: int        — layers whose router gate was sliced
            perLayerNormalized: bool       — true for the §4.1.3.4 path
            implementation: str            — script reference
            rationale: str                 — methodology blockquote
            notes: str                     — empirical context for the model card
            expertTensorLayout: str        — layout discriminator (e.g. 'mlp-experts-unfused')

        Tier 2 wires to scripts/cpu_expert_prune_v2.py with --importance-json
        pointing at the JSON produced by the upstream expert-activation-profile
        stage. Tier 1: short-circuits cleanly when ctx.model is None.

        IMPORTANT: this base assumes the unfused experts layout that Qwen3MoE
        and OLMoE both use. Future MoE families with different layouts
        (Mixtral block_sparse_moe, Granite-MoE fused, DeepSeek-V2 routed+shared)
        either ship as their own family adapter that overrides this method
        OR extend the dispatch inside cpu_expert_prune_v2.py per the
        expertTensorLayout field. Do NOT branch on architectures here.
        """
        keep = params.get("keepExpertsPerLayer") or params.get("keepExperts")
        original = params.get("originalExpertsPerLayer")
        strategy = params.get("strategy", "calibration-aware-activation-count")
        prune_pct = params.get("prunePct")
        layout = params.get("expertTensorLayout", "mlp-experts-unfused")
        self.log(
            f"expert-prune {original or '?'}→{keep} per layer "
            f"({prune_pct or '?'}% removed) via {strategy} layout={layout}"
        )

        if ctx.model is None:
            self.log("  No model loaded — prune deferred (dispatch-only path)")
            return ctx

        if keep is None:
            raise ValueError(
                f"{self.name}.expert_prune: alloy stage missing "
                f"'keepExpertsPerLayer' (or legacy 'keepExperts'). Cannot "
                f"prune without a target survivor count."
            )
        if layout != "mlp-experts-unfused":
            raise ValueError(
                f"{self.name}.expert_prune: expertTensorLayout={layout!r} is "
                f"not handled by MoEUnfusedExpertsBase, which only knows the "
                f"unfused MoE module-tree layout (model.layers.{{i}}.mlp.experts). "
                f"For layouts like 'block_sparse_moe-unfused' (Mixtral), "
                f"'granite-moe-fused' (Granite-MoE), or 'deepseek-routed-shared' "
                f"(DeepSeek-V2), write a new family adapter that overrides "
                f"this method with a layout-specific tensor walk."
            )

        # Lazy import — Tier 1 dispatch must work without torch installed.
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from cpu_expert_prune_v2 import prune_experts

        # The pruner is a streaming disk-to-disk safetensors rewrite — it
        # reads the source model from a directory on disk and writes the
        # pruned shards to an output directory. The loaded ctx.model is
        # NOT touched (the streaming rewrite would be wasteful and risky
        # to do in memory for big models).
        #
        # We need:
        #   - source model directory (the original unmodified base)
        #   - output directory for the pruned shards
        #   - importance JSON path (set by the upstream expert_activation_profile
        #     stage on ctx.importance_json_path)
        src_model_dir = getattr(ctx, "source_model_dir", None) or ctx.model_name
        if not src_model_dir or not Path(src_model_dir).exists():
            raise ValueError(
                f"{self.name}.expert_prune: ctx.source_model_dir is not set "
                f"or does not exist. The pruner needs the original safetensors "
                f"shards on disk to do the streaming rewrite. alloy_executor "
                f"must populate ctx.source_model_dir with the local path to "
                f"the unmodified base model before this stage runs."
            )
        pruned_out = (ctx.output_dir / "pruned").resolve()
        importance_path = getattr(ctx, "importance_json_path", None)
        if not importance_path and strategy == "calibration-aware-activation-count":
            raise ValueError(
                f"{self.name}.expert_prune: alloy strategy is "
                f"'calibration-aware-activation-count' but ctx.importance_json_path "
                f"is not set. The expert-activation-profile stage must run "
                f"BEFORE expert-prune in the same alloy so the importance "
                f"JSON exists. Check the alloy stage ordering."
            )

        metadata = prune_experts(
            model_dir=src_model_dir,
            out_dir=pruned_out,
            keep_experts=int(keep),
            importance_json=importance_path,
        )

        # Reload ctx.model from the pruned dir so downstream stages (quant,
        # eval, package, publish) operate on the pruned model, not the
        # in-memory original.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.log(f"  reloading pruned model from {pruned_out}")
        # Free the original model's GPU memory before loading the pruned one.
        del ctx.model
        torch.cuda.empty_cache()
        ctx.model = AutoModelForCausalLM.from_pretrained(
            str(pruned_out),
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        ctx.tokenizer = AutoTokenizer.from_pretrained(
            str(pruned_out), trust_remote_code=True,
        )
        ctx.dead_heads = None  # not relevant for MoE
        ctx.pruned_model_dir = str(pruned_out)
        self.log(
            f"  prune complete: {metadata.get('total_bytes_out', 0) / 1e9:.1f} GB "
            f"surviving structure"
        )

        return ctx
