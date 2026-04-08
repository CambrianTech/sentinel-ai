"""Qwen2DenseAdapter — family adapter for the Qwen2.5 dense catalog.

Handles continuum-ai/qwen2.5-coder-7b-compacted (the v2-7b-coder-compensated
artifact, the §4.1.3.3 anchor for compensation-LoRA). The alloy declares
architecture='qwen2' against base Qwen/Qwen2.5-Coder-7B, dense (not MoE).

Forge concerns this family supports:
    - prune (dense head pruning, identical pattern to Qwen3DenseAdapter at
      this layer — both use forge_model.prune which is architecture-agnostic
      for dense Qwen-family models)
    - train (normal LoRA recovery training AND the §4.1.3.3 compensation
      LoRA — both flow through .train() because the alloy uses 'lora' stage
      type for both, and the adapter dispatches internally on the presence
      of a 'teacher' field which signals KL-distillation against an unmodified
      teacher)

Forge concerns this family does NOT support:
    - expert-prune (dense family — alloys for this family use 'prune')
    - expert-activation-profile (MoE-only)
    - modality (text-only)

Reproducibility contract: this adapter MUST stay frozen against the v2-7b
artifact. The compensation LoRA stage's params (teacher, kdTemperature,
loraRank, loraAlpha, lossType, mergedAtSave) ARE the §4.1.3.3 methodology;
changes to those values would change the published artifact.

NOTE: this adapter overlaps significantly with Qwen3DenseAdapter at the
prune layer — both call forge_model.prune the same way. That's the
duplication that justifies extracting a QwenDenseBase. Per the OOP rule,
the extraction lands as its own commit AFTER both siblings exist with
working behavior, not before. Don't extract a base off one example.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from .base import FamilyAdapter
from .registry import register_family_adapter

if TYPE_CHECKING:
    from scripts.stages.base import ForgeContext


@register_family_adapter
class Qwen2DenseAdapter(FamilyAdapter):
    """Family adapter for Qwen2 / Qwen2.5 dense models."""

    architectures = ("qwen2",)

    def prune(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Dense head pruning. Same pattern as Qwen3DenseAdapter.prune.

        Param contract (from the alloy's prune stage, v2-7b-coder shape):
            strategy: str             — typically 'activation-magnitude' (the §4.1.3.1 fix)
            level: float              — fraction of heads to remove
            minHeadsPerLayer: int     — floor per layer
            minKvHeadsPerLayer: int   — KV-head floor
            analysisSteps: int        — importance computation steps
            perLayerNormalized: bool  — true for the §4.1.3.1 fix path
            defragMode: str           — defrag strategy
            notes: str
        """
        level = float(params.get("level", 0.3))
        strategy = params.get("strategy", "activation-magnitude")
        per_layer_normalized = params.get("perLayerNormalized", True)
        self.log(
            f"Pruning {level:.0%} heads via {strategy} "
            f"(perLayerNormalized={per_layer_normalized})"
        )

        if ctx.model is None:
            self.log("  No model loaded — prune deferred (dispatch-only path)")
            return ctx

        # Tier 2 wiring intentionally deferred — same pattern as the MoE
        # adapters. The forge_model.prune call works identically here as in
        # Qwen3DenseAdapter, but landing both Tier 2 light-ups together will
        # happen in a single commit gated on having a 5090 to validate against.
        raise NotImplementedError(
            "Qwen2DenseAdapter.prune Tier 2 wiring lands together with all "
            "other adapters' Tier 2 wiring once the reproducibility test "
            "moves from dispatch-only to actual byte-equivalent re-forge. "
            "The v2-7b-coder-compensated artifact reproduces from the existing "
            "scripts/forge_model.py + scripts/compensation_lora.py CLI today."
        )

    def train(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """LoRA training — handles both normal recovery LoRA AND the §4.1.3.3
        compensation distillation LoRA. Dispatches internally on the presence
        of a 'teacher' field, which signals KL-distillation against an
        unmodified teacher per the §4.1.3.3 methodology.

        Param contract (recovery LoRA — first lora stage):
            domain: str
            dataset: str
            steps: int
            learningRate: str
            batchSize: int
            scheduler: str
            sequenceLength: int
            gradientAccumulation: int
            precision: str
            calibrationSource: str

        Param contract (compensation distillation — second lora stage):
            teacher: str               — unmodified teacher model id
            teacherPrecision: str
            studentPrecision: str
            calibrationDataset: str
            kdTemperature: float       — typically 2.0
            loraRank: int
            loraAlpha: int
            targetModules: list[str]
            lossType: str              — 'kl_logits' | 'mse_hidden' | 'both'
            mergedAtSave: bool
            trainableParamsPct: float
            domain: str
            steps: int
            learningRate: str
            name: str                  — stage label
            notes: str
        """
        teacher = params.get("teacher")
        if teacher:
            # § 4.1.3.3 compensation distillation
            kd_t = params.get("kdTemperature", 2.0)
            loss = params.get("lossType", "kl_logits")
            rank = params.get("loraRank")
            self.log(
                f"compensation-LoRA § 4.1.3.3 — teacher={teacher}, "
                f"kdTemperature={kd_t}, loss={loss}, loraRank={rank}"
            )
        else:
            # Normal recovery LoRA
            domain = params.get("domain", "code")
            steps = int(params.get("steps", 1000))
            self.log(f"Training {steps} steps on {domain} data")

        if ctx.model is None:
            self.log("  No model loaded — train deferred (dispatch-only path)")
            return ctx

        raise NotImplementedError(
            "Qwen2DenseAdapter.train Tier 2 wiring deferred. The compensation "
            "distillation path runs through scripts/compensation_lora.py "
            "today; Tier 2 hookup wires it via the adapter so the alloy is "
            "the single entry point."
        )
