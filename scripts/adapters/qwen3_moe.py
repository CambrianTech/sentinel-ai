"""Qwen3MoEAdapter — family adapter for the Qwen3MoE architecture.

Handles continuum-ai/qwen3-coder-30b-a3b-compacted-19b-256k (the morning-of-
2026-04-08 §4.1.3.4 anchor artifact, alloy hash aa61c4bdf463847c, 88.4
HumanEval) and any future Qwen3MoE forge — Qwen3-Coder-30B-A3B-Instruct,
Qwen3-235B-A22B (when forged), Qwen3-Coder-480B-A35B-Instruct (when forged
on the grid).

Tensor layout: Qwen3MoE uses an unfused experts module-tree —
    model.layers.{i}.mlp.experts.{e}.{gate_proj,up_proj,down_proj}
with a separate router gate at:
    model.layers.{i}.mlp.gate
Each layer has 128 experts of which 8 are activated per token. The §4.1.3.4
methodology paper anchors here: 37.5% expert removal (128 → 80) keyed to
calibration-aware activation count, with the negative-baseline router-gate-L2
control showing the metric swap closes +9.7 / +12.2 HumanEval / HumanEval+
points on the same source / same K / same hardware / same eval.

Forge concerns this family supports:
    - expert-activation-profile (§4.1.3.4 calibration-aware importance)
    - expert-prune (per-layer top-K removal keyed to the importance JSON)
    - quant (family-agnostic — handled by the base default + QuantExecutor)
    - eval (family-agnostic — handled by EvalExecutor + eval_with_calibration)

Forge concerns this family does NOT support:
    - prune (this is a MoE family — alloys must use 'expert-prune' not 'prune')
    - train / lora at this time (the morning's compaction shipped without
      compensation LoRA; if a Qwen3MoE compensation LoRA artifact ships
      later, train() gets overridden)
    - modality (the morning's artifact is text-only)

Reproducibility contract: this adapter MUST stay frozen against the morning's
artifact. The §4.1.3.4 metric ('calibration-aware-activation-count') is the
positive cell; the negative-baseline router-gate-L2 cell is preserved in the
alloy's priorMetricBaselines[] as the falsifiability anchor. If a future
methodology change wants to use a different metric, that change ships as a
NEW adapter (Qwen3MoEAdapter_v2 or similar) registered against a NEW
discriminator, never as an edit to this file's behavior.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from .base import FamilyAdapter
from .registry import register_family_adapter

if TYPE_CHECKING:
    from scripts.stages.base import ForgeContext


@register_family_adapter
class Qwen3MoEAdapter(FamilyAdapter):
    """Family adapter for Qwen3MoE (the morning's §4.1.3.4 anchor family)."""

    architectures = ("qwen3_moe",)

    def expert_activation_profile(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """§4.1.3.4 calibration-aware MoE expert importance profiling.

        Param contract (from the alloy's expert-activation-profile stage):
            calibrationCorpus: str        — corpus identifier (e.g. 'heldout_code300')
            calibrationCorpusFile: str    — path to the .jsonl file (sha256 in the alloy's calibrationCorpora)
            calibrationExamples: int      — number of held-out examples
            calibrationTokens: int        — total token count across the corpus
            implementation: str           — script reference (e.g. 'expert_activation_profile.py')
            metricVersion: str            — '4.1.3.4' for the calibration-aware version
            notes: str                    — methodology blockquote that lands on the model card

        Output: an importance JSON written to ctx.output_dir, consumed by
        the downstream expert-prune stage via its importanceJson field.

        Tier 2 wires to scripts/expert_activation_profile.py with the
        calibration corpus path and the cross-architecture portability
        fixes from commit 488b740. Tier 1 stub: short-circuit if no model.
        """
        corpus = params.get("calibrationCorpusFile") or params.get("calibrationCorpus")
        examples = params.get("calibrationExamples")
        tokens = params.get("calibrationTokens")
        self.log(
            f"expert-activation-profile § 4.1.3.4 — corpus={corpus}, "
            f"{examples} examples, {tokens} tokens"
        )

        if ctx.model is None:
            self.log("  No model loaded — profile deferred (dispatch-only path)")
            return ctx

        # Tier 2: lazy-import expert_activation_profile module and run.
        # The script is currently a CLI entry point; refactoring it to a
        # callable function lands as a follow-up commit gated on having
        # a 5090 to test against. For now this branch raises clearly if
        # somehow reached without that hookup.
        raise NotImplementedError(
            "Qwen3MoEAdapter.expert_activation_profile Tier 2 wiring not yet "
            "landed. Run scripts/expert_activation_profile.py directly until "
            "the Python API extraction lands. Tier 1 dispatch resolution "
            "is the current scope."
        )

    def expert_prune(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Per-layer top-K MoE expert removal keyed to the §4.1.3.4 importance.

        Param contract (from the alloy's expert-prune stage):
            strategy: str                  — 'calibration-aware-activation-count' (the §4.1.3.4 metric)
                                             or 'router-gate-l2' (the negative-baseline anchor)
            metric: str                    — 'activation_count' (positive cell)
            metricSource: str              — script reference (e.g. 'expert_activation_profile.py against ...')
            keepExpertsPerLayer: int       — survivors per layer (e.g. 80 of 128)
            originalExpertsPerLayer: int   — pre-prune count (128 for Qwen3-Coder-30B-A3B)
            prunePct: float                — % removed (e.g. 37.5)
            expertsDropped: int            — total experts removed across all layers
            expertsRenamed: int            — total surviving experts renumbered
            routerSlicedLayers: int        — layers whose router gate was sliced
            perLayerNormalized: bool       — true for the §4.1.3.4 path
            implementation: str            — 'scripts/cpu_expert_prune_v2.py --importance-json'
            rationale: str                 — methodology blockquote
            notes: str                     — empirical context for the model card

        Tier 2 wires to scripts/cpu_expert_prune_v2.py with --importance-json
        pointing at the JSON produced by the upstream expert-activation-profile
        stage. Tier 1: short-circuit if no model.
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
            "Qwen3MoEAdapter.expert_prune Tier 2 wiring not yet landed. Run "
            "scripts/cpu_expert_prune_v2.py --importance-json directly until "
            "the Python API extraction lands. The morning's qwen3-coder-30b-a3b-"
            "compacted-19b-256k artifact reproduces from the CLI today; the "
            "adapter wiring is what lets the reproducibility test run it as "
            "a single dispatch instead of a manual pipeline."
        )
