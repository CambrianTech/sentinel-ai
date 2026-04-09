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
    """Mixtral 8x7B / 8x22B — block_sparse_moe-unfused layout."""

    architectures = ("mixtral",)

    def expert_activation_profile(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        if ctx.model is None:
            self.log("Mixtral expert-activation-profile (dispatch-only)")
            return ctx
        _stub_expert_prune_raise(self.name, "mixtral", "block_sparse_moe-unfused", params)

    def expert_prune(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        if ctx.model is None:
            self.log(f"Mixtral expert-prune (dispatch-only, layout=block_sparse_moe-unfused)")
            return ctx
        _stub_expert_prune_raise(self.name, "mixtral", "block_sparse_moe-unfused", params)


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
