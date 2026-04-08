"""OlmoeAdapter — family adapter for the OLMoE architecture.

Handles continuum-ai/olmoe-1b-7b-compacted-5b (the §4.1.3.4 cross-architecture
anchor artifact, alloy hash bba0a92ff0c8bebb, 36.0 HumanEval). This is the
SECOND empirical anchor for the §4.1.3.4 calibration-aware MoE expert
importance methodology — paired with qwen3-coder-30b-a3b-compacted-19b-256k
on a structurally different MoE family to validate that the metric pattern
generalizes across architectures, not just across one family.

Tensor layout: OLMoE uses an unfused experts module-tree —
    model.layers.{i}.mlp.experts.{e}.{gate_proj,up_proj,down_proj}
    model.layers.{i}.mlp.gate
The naming convention is similar to Qwen3MoE on the surface but the layer
geometry is different (16 layers × 64 experts vs Qwen3MoE's 48 × 128) and
the source class is OlmoeForCausalLM vs Qwen3MoeForCausalLM. The shared
behavior between OLMoE and Qwen3MoE is what justifies extracting a future
MoEUnfusedExpertsBase — but per the outlier-validation rule, we write the
second sibling AS a sibling first, prove it works, THEN extract.

Forge concerns: same matrix as Qwen3MoEAdapter — expert-activation-profile,
expert-prune, plus family-agnostic quant/eval. No prune (alloys must use
expert-prune for MoE), no train (this artifact shipped without compensation
LoRA), no modality (text-only).

Reproducibility contract: this adapter MUST stay frozen against the published
artifact. The within-model A/B negative-baseline cell (broad-corpus calibration
vs code-corpus calibration on the same OLMoE base) is preserved in the alloy's
priorMetricBaselines[] as the §4.1.3.4 falsifiability anchor for OLMoE.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from .base import FamilyAdapter
from .registry import register_family_adapter

if TYPE_CHECKING:
    from scripts.stages.base import ForgeContext


@register_family_adapter
class OlmoeAdapter(FamilyAdapter):
    """Family adapter for OLMoE (the §4.1.3.4 cross-architecture anchor)."""

    architectures = ("olmoe",)

    def expert_activation_profile(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """§4.1.3.4 calibration-aware MoE expert importance profiling for OLMoE.

        Same param contract as Qwen3MoEAdapter.expert_activation_profile —
        the methodology is identical, only the tensor walk underneath differs.

        Tier 2 wires to scripts/expert_activation_profile.py with OLMoE's
        OlmoeForCausalLM module-tree probe. The cross-architecture portability
        fixes from sentinel-ai commit 488b740 are what made this work without
        rewriting the script per family.
        """
        corpus = params.get("calibrationCorpus")
        examples = params.get("calibrationExamples")
        tokens = params.get("calibrationTokens")
        self.log(
            f"expert-activation-profile § 4.1.3.4 (OLMoE) — corpus={corpus}, "
            f"{examples} examples, {tokens} tokens"
        )

        if ctx.model is None:
            self.log("  No model loaded — profile deferred (dispatch-only path)")
            return ctx

        raise NotImplementedError(
            "OlmoeAdapter.expert_activation_profile Tier 2 wiring not yet "
            "landed. Run scripts/expert_activation_profile.py directly until "
            "the Python API extraction lands."
        )

    def expert_prune(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Per-layer top-K MoE expert removal for OLMoE.

        Same param contract as Qwen3MoEAdapter.expert_prune. Tier 2 wires
        to scripts/cpu_expert_prune_v2.py --importance-json — same script,
        different model-family probe. The within-model A/B cells (broad-corpus
        calibration vs code-corpus calibration on the same OLMoE base) are
        what make this artifact the §4.1.3.4 cross-architecture validation.
        """
        keep = params.get("keepExpertsPerLayer") or params.get("keepExperts")
        original = params.get("originalExpertsPerLayer")
        strategy = params.get("strategy", "calibration-aware-activation-count")
        prune_pct = params.get("prunePct")
        self.log(
            f"expert-prune {original or '?'}→{keep} per layer "
            f"({prune_pct or '?'}% removed) via {strategy}"
        )

        if ctx.model is None:
            self.log("  No model loaded — prune deferred (dispatch-only path)")
            return ctx

        raise NotImplementedError(
            "OlmoeAdapter.expert_prune Tier 2 wiring not yet landed. Run "
            "scripts/cpu_expert_prune_v2.py --importance-json directly until "
            "the Python API extraction lands."
        )
