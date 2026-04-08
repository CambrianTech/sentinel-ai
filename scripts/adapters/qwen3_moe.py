"""Qwen3MoEAdapter — family adapter for the Qwen3MoE architecture.

Inherits from MoEUnfusedExpertsBase. Pure inheritance — Qwen3MoE uses the
unfused experts layout the base assumes:

    model.layers.{i}.mlp.experts.{e}.{gate_proj,up_proj,down_proj}
    model.layers.{i}.mlp.gate

Handles continuum-ai/qwen3-coder-30b-a3b-compacted-19b-256k (the morning-of-
2026-04-08 §4.1.3.4 anchor artifact, alloyHash 011970c80c2f3429 after the
humaneval_plus correction in commit 1bc32d2, 88.4 HumanEval) and any future
Qwen3MoE forge — Qwen3-Coder-30B-A3B-Instruct, Qwen3-235B-A22B (when forged),
Qwen3-Coder-480B-A35B-Instruct (when forged on the grid).

The §4.1.3.4 methodology paper anchors here: 37.5% expert removal (128 → 80)
keyed to calibration-aware activation count, with the negative-baseline
router-gate-L2 control showing the metric swap closes +9.7 HumanEval points
on the same source / same K / same hardware / same eval. The negative-
baseline cell is preserved in the alloy's priorMetricBaselines[] as the
falsifiability anchor.

Reproducibility contract: this adapter MUST stay frozen against the morning's
artifact. If a future methodology change wants a different metric, that change
ships as a NEW adapter (Qwen3MoEAdapter_v2 or similar) registered against a
NEW discriminator, never as an edit to this file or to MoEUnfusedExpertsBase.
"""

from __future__ import annotations
from .moe_unfused_base import MoEUnfusedExpertsBase
from .registry import register_family_adapter


@register_family_adapter
class Qwen3MoEAdapter(MoEUnfusedExpertsBase):
    """Family adapter for Qwen3MoE — pure inheritance from the unfused base."""

    architectures = ("qwen3_moe",)
