"""SOTA MoE family adapter stubs — Mixtral, Phi-MoE, GraniteMoE, DeepSeek-V2.

The MoEUnfusedExpertsBase from roadmap step 2 handles the unfused-experts
layout that Qwen3MoE and OLMoE share. The four families here use
DIFFERENT module-tree layouts and need their own adapters per the
never-branch rule:

  Mixtral 8x22B / Mixtral 8x7B
      arch:      mixtral
      param tree: model.layers.{i}.block_sparse_moe.experts.{e}.{w1, w2, w3}
                  + model.layers.{i}.block_sparse_moe.gate
      layout id: 'block_sparse_moe-unfused'

  Phi-3.5-MoE
      arch:      phimoe
      param tree: same block_sparse_moe-unfused as Mixtral but with
                  16 experts per layer (vs Mixtral's 8) and a different
                  router gate dimension. Could share a sub-base with
                  Mixtral if more block_sparse_moe-unfused families ship,
                  but we don't extract a base off two examples that
                  haven't both been forge-validated yet.
      layout id: 'block_sparse_moe-unfused'

  Granite-MoE / Granite-3.1-3b-a800m-instruct
      arch:      granitemoe
      param tree: model.layers.{i}.block_sparse_moe.input_linear
                  + model.layers.{i}.block_sparse_moe.output_linear
                  + model.layers.{i}.block_sparse_moe.router
                  (FUSED experts — all experts share one big tensor; the
                   per-expert weight slicing happens at runtime via
                   gather/scatter against the router output)
      layout id: 'granite-moe-fused'
      Structurally distinct from unfused: pruning experts here means
      slicing the FUSED tensor along an expert axis, not dropping
      named param entries.

  DeepSeek-V2 / DeepSeek-V2-Lite
      arch:      deepseek_v2
      param tree: model.layers.{i}.mlp.experts.{e}.{gate_proj, up_proj, down_proj}
                  PLUS model.layers.{i}.mlp.shared_experts.{gate_proj, up_proj, down_proj}
                  PLUS model.layers.{i}.mlp.gate
                  (routed experts AND a shared expert pathway that fires
                   on every token regardless of routing)
      layout id: 'deepseek-routed-shared'
      Pruning here means dropping routed experts BUT preserving the
      shared expert bit-exact (it carries the always-fires capability).

Each adapter today is a registered STUB whose expert_prune() raises
NotImplementedError with a layout-specific message. When the first
forge of a SOTA MoE family runs (Joel's stated frontier moonshot is
Mixtral 8x22B for the single-5090 tier and Qwen3-Coder-480B for the
multi-GPU grid tier), the corresponding stub gets a real body in a
focused commit. The architectural contract is: dispatch resolves the
right adapter, the implementer fills in the right tensor walk for that
family's layout. Adding a new MoE family is one new file in this
package or one new class in this file.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from .base import FamilyAdapter
from .registry import register_family_adapter

if TYPE_CHECKING:
    from scripts.stages.base import ForgeContext


def _stub_expert_prune_raise(adapter_name: str, arch: str, layout: str, params: dict) -> None:
    """Loud failure with a layout-specific message. Pointer at the file
    that needs the implementation when the first forge of this family runs."""
    raise NotImplementedError(
        f"{adapter_name}.expert_prune is a registered stub. The "
        f"architecture {arch!r} uses the {layout!r} layout, which is "
        f"structurally distinct from MoEUnfusedExpertsBase's unfused-Qwen "
        f"layout. To wire it: implement expert_prune() in "
        f"scripts/adapters/sota_moe.py with the family-specific tensor walk, "
        f"add a TDD test in tests/unit/adapters/test_sota_moe_adapters.py "
        f"asserting it scores a known fixture, and (when the first forge of "
        f"this family runs) verify the produced safetensors hash matches "
        f"the alloy modelHash. Called with params={sorted(params.keys())!r}."
    )


# ── Mixtral (block_sparse_moe-unfused) ──────────────────────────────────────


@register_family_adapter
class MixtralAdapter(FamilyAdapter):
    """Mixtral 8x7B / 8x22B — block_sparse_moe-unfused layout.

    The single-5090 prosumer headline play from Kash's frontier-target
    analysis: 8 experts per layer (2 active), 141B / ~39B active params.
    Sub-24 GB tier completely empty on HuggingFace today; the projected
    forge ships ~70B post-prune at ~22 GB Q4_K_M (fits a single RTX 5090
    or workstation 24 GB card).

    Tensor layout (verified against Mixtral-8x7B-Instruct-v0.1's published
    safetensors index):

        model.layers.{L}.block_sparse_moe.gate.weight
        model.layers.{L}.block_sparse_moe.experts.{K}.w1.weight   ← gate_proj
        model.layers.{L}.block_sparse_moe.experts.{K}.w2.weight   ← down_proj
        model.layers.{L}.block_sparse_moe.experts.{K}.w3.weight   ← up_proj

    Algorithm: identical to MoEUnfusedExpertsBase (the morning's
    qwen3-coder-30b-a3b §4.1.3.4 fix). Only the tensor name patterns
    differ. Implementation: cpu_expert_prune_v2.prune_experts(layout=MIXTRAL_LAYOUT)
    threads the layout spec through the same Pass 1 (read router gates) +
    Pass 2 (streaming rewrite) algorithm. The layout dispatch is in
    cpu_expert_prune_v2.py, NOT here — the adapter just chooses the right
    layout constant and delegates.
    """

    architectures = ("mixtral",)

    def expert_activation_profile(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """§ 4.1.3.4 calibration-aware MoE expert importance profiling for Mixtral.

        Same expert_activation_profile.profile_experts API as the
        unfused-Qwen path; the script's hook registration walks
        `model.layers.{i}.block_sparse_moe.gate` for Mixtral instead of
        `model.layers.{i}.mlp.gate` for Qwen3MoE. The cross-architecture
        portability fixes from sentinel-ai commit 488b740 made this
        layout-agnostic — same script, different config.json gives a
        different mlp module name, the hook attaches by walking
        named_modules() and matching the family-specific path.
        """
        corpus_file = params.get("calibrationCorpusFile") or params.get("calibrationCorpus")
        examples = params.get("calibrationExamples")
        tokens = params.get("calibrationTokens")
        max_length = int(params.get("maxLength", 2048))
        self.log(
            f"expert-activation-profile § 4.1.3.4 (Mixtral) — corpus={corpus_file}, "
            f"{examples} examples, {tokens} tokens"
        )

        if ctx.model is None:
            self.log("  No model loaded — Mixtral profile deferred (dispatch-only path)")
            return ctx

        if not corpus_file:
            raise ValueError(
                f"{self.name}.expert_activation_profile: alloy stage missing "
                f"'calibrationCorpusFile'. The §4.1.3.4 calibration-aware metric "
                f"requires a held-out corpus path."
            )

        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from expert_activation_profile import profile_experts

        corpus_path = Path(corpus_file)
        if not corpus_path.is_absolute():
            corpus_path = (ctx.output_dir / corpus_path).resolve()
        if not corpus_path.exists():
            raise FileNotFoundError(
                f"Mixtral calibration corpus {corpus_path} does not exist."
            )

        importance_path = (ctx.output_dir / "importance.activation_count.json").resolve()
        device = getattr(ctx, "device", None) or "cuda:0"

        result = profile_experts(
            model=ctx.model,
            tokenizer=ctx.tokenizer,
            calibration_data=corpus_path,
            output=importance_path,
            max_length=max_length,
            device=device,
            model_label=ctx.model_name or type(ctx.model).__name__,
        )
        ctx.importance_json_path = str(importance_path)
        self.log(
            f"  Mixtral importance written: {importance_path} "
            f"({result['num_hidden_layers']} layers × {result['num_experts']} experts)"
        )
        return ctx

    def expert_prune(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Per-layer top-K MoE expert removal for Mixtral via the layout-aware
        pruner. Real Tier 2 wiring — no stub. Calls
        cpu_expert_prune_v2.prune_experts(layout=MIXTRAL_LAYOUT) which threads
        the block_sparse_moe-unfused name patterns through the same
        algorithm the morning's qwen3-coder-30b-a3b flagship was forged
        through. The expertTensorLayout field on the alloy stage MUST be
        'block_sparse_moe-unfused' (the family discriminator); any other
        value means the alloy is misdeclaring its layout.
        """
        keep = params.get("keepExpertsPerLayer") or params.get("keepExperts")
        original = params.get("originalExpertsPerLayer")
        strategy = params.get("strategy", "calibration-aware-activation-count")
        prune_pct = params.get("prunePct")
        layout_id = params.get("expertTensorLayout", "block_sparse_moe-unfused")
        self.log(
            f"Mixtral expert-prune {original or '?'}→{keep} per layer "
            f"({prune_pct or '?'}% removed) via {strategy} layout={layout_id}"
        )

        if ctx.model is None:
            self.log(f"  No model loaded — Mixtral expert-prune deferred (dispatch-only path)")
            return ctx

        if keep is None:
            raise ValueError(
                f"{self.name}.expert_prune: alloy stage missing "
                f"'keepExpertsPerLayer'. Mixtral has 8 experts per layer in "
                f"its standard configurations; pick a target and re-declare."
            )
        if layout_id != "block_sparse_moe-unfused":
            raise ValueError(
                f"{self.name}.expert_prune: expertTensorLayout={layout_id!r} "
                f"is not the block_sparse_moe-unfused layout this adapter "
                f"handles. Either fix the alloy field or use a different "
                f"family adapter."
            )

        # Lazy import — Tier 1 dispatch must work without torch installed.
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from cpu_expert_prune_v2 import prune_experts, MIXTRAL_LAYOUT

        src_model_dir = getattr(ctx, "source_model_dir", None)
        if src_model_dir is None:
            raise ValueError(
                f"{self.name}.expert_prune: ctx.source_model_dir is not set. "
                f"The pruner needs the original Mixtral safetensors shards on "
                f"disk to do the streaming rewrite. alloy_executor must "
                f"populate ctx.source_model_dir before this stage runs."
            )
        if not Path(src_model_dir).exists():
            raise ValueError(
                f"{self.name}.expert_prune: ctx.source_model_dir={src_model_dir!r} "
                f"does not exist on disk."
            )

        pruned_out = (ctx.output_dir / "pruned").resolve()
        importance_path = getattr(ctx, "importance_json_path", None)
        if not importance_path and strategy == "calibration-aware-activation-count":
            raise ValueError(
                f"{self.name}.expert_prune: alloy strategy is "
                f"'calibration-aware-activation-count' but ctx.importance_json_path "
                f"is not set. The expert-activation-profile stage must run "
                f"BEFORE expert-prune in the same alloy."
            )

        metadata = prune_experts(
            model_dir=src_model_dir,
            out_dir=pruned_out,
            keep_experts=int(keep),
            importance_json=importance_path,
            layout=MIXTRAL_LAYOUT,
        )

        # Reload ctx.model from the pruned dir for downstream stages.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.log(f"  reloading pruned Mixtral from {pruned_out}")
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
        ctx.dead_heads = None
        ctx.pruned_model_dir = str(pruned_out)
        self.log(
            f"  Mixtral prune complete: "
            f"{metadata.get('total_bytes_out', 0) / 1e9:.1f} GB surviving structure"
        )
        return ctx


# ── Phi-MoE (block_sparse_moe-unfused, smaller experts) ─────────────────────


@register_family_adapter
class PhiMoEAdapter(FamilyAdapter):
    """Phi-3.5-MoE — same block_sparse_moe-unfused layout as Mixtral but
    with 16 experts/layer and a different router gate dimension."""

    architectures = ("phimoe",)

    def expert_activation_profile(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        if ctx.model is None:
            self.log("Phi-MoE expert-activation-profile (dispatch-only)")
            return ctx
        _stub_expert_prune_raise(self.name, "phimoe", "block_sparse_moe-unfused", params)

    def expert_prune(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        if ctx.model is None:
            self.log(f"Phi-MoE expert-prune (dispatch-only, layout=block_sparse_moe-unfused)")
            return ctx
        _stub_expert_prune_raise(self.name, "phimoe", "block_sparse_moe-unfused", params)


# ── GraniteMoE (granite-moe-fused) ──────────────────────────────────────────


@register_family_adapter
class GraniteMoEAdapter(FamilyAdapter):
    """IBM Granite-MoE — granite-moe-fused layout. All experts share one
    big input_linear + output_linear tensor; per-expert slicing happens
    at runtime against the router. Pruning means slicing the fused
    tensors along the expert axis, not dropping named param entries."""

    architectures = ("granitemoe",)

    def expert_activation_profile(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        if ctx.model is None:
            self.log("GraniteMoE expert-activation-profile (dispatch-only)")
            return ctx
        _stub_expert_prune_raise(self.name, "granitemoe", "granite-moe-fused", params)

    def expert_prune(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        if ctx.model is None:
            self.log(f"GraniteMoE expert-prune (dispatch-only, layout=granite-moe-fused)")
            return ctx
        _stub_expert_prune_raise(self.name, "granitemoe", "granite-moe-fused", params)


# ── DeepSeek-V2 (deepseek-routed-shared) ────────────────────────────────────


@register_family_adapter
class DeepSeekV2Adapter(FamilyAdapter):
    """DeepSeek-V2 / DeepSeek-V2-Lite — deepseek-routed-shared layout.
    Routed experts under .mlp.experts.{e}.* PLUS a shared expert pathway
    .mlp.shared_experts.* that fires on every token regardless of routing.
    Pruning the routed experts MUST preserve the shared expert bit-exact —
    it carries the always-fires capability that the model relies on."""

    architectures = ("deepseek_v2",)

    def expert_activation_profile(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        if ctx.model is None:
            self.log("DeepSeek-V2 expert-activation-profile (dispatch-only)")
            return ctx
        _stub_expert_prune_raise(self.name, "deepseek_v2", "deepseek-routed-shared", params)

    def expert_prune(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        if ctx.model is None:
            self.log(f"DeepSeek-V2 expert-prune (dispatch-only, layout=deepseek-routed-shared)")
            return ctx
        _stub_expert_prune_raise(self.name, "deepseek_v2", "deepseek-routed-shared", params)
