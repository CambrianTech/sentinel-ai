"""OlmoeAdapter — family adapter for the OLMoE architecture.

Inherits from MoEUnfusedExpertsBase. Pure inheritance — OLMoE uses the same
unfused experts layout that Qwen3MoE uses, just with different layer geometry
(16 layers × 64 experts vs Qwen3MoE's 48 × 128) and a different source
class (OlmoeForCausalLM vs Qwen3MoeForCausalLM). The cross-architecture
portability fixes from sentinel-ai commit 488b740 are what made the
underlying expert_activation_profile.py + cpu_expert_prune_v2.py scripts
handle both families without per-family forks.

Handles continuum-ai/olmoe-1b-7b-compacted-5b — the §4.1.3.4 cross-architecture
anchor artifact (alloy hash bba0a92ff0c8bebb, 36.0 HumanEval). This is the
SECOND empirical anchor for the §4.1.3.4 calibration-aware MoE expert
importance methodology, paired with qwen3-coder-30b-a3b on a structurally
different MoE family to validate that the metric pattern generalizes
across architectures, not just across one family.

Reproducibility contract: frozen against the published artifact. The
within-model A/B negative-baseline cell (broad-corpus calibration vs
code-corpus calibration on the same OLMoE base) is preserved in the
alloy's priorMetricBaselines[] as the §4.1.3.4 falsifiability anchor for
OLMoE.
"""

from __future__ import annotations
from .moe_unfused_base import MoEUnfusedExpertsBase
from .registry import register_family_adapter


@register_family_adapter
class OlmoeAdapter(MoEUnfusedExpertsBase):
    """Family adapter for OLMoE — pure inheritance from the unfused base."""

    architectures = ("olmoe",)
