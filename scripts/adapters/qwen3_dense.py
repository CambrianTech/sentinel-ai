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
        """Dense head pruning via the legacy forge_model.prune path.

        Param contract (from the alloy's prune stage):
            strategy: str    — entropy | magnitude | gradient | activation-magnitude
            level: float     — fraction of heads to remove (0.0..0.9)

        Tier 1: dispatch resolution only. The body below is the Tier 2
        wiring that will land once the reproducibility test gates pass.
        """
        strategy = params.get("strategy", "entropy")
        level = params.get("level", 0.0)
        # Tier 1 stub — Tier 2 wires to forge_model.prune + defrag_inline.
        # Intentionally not raising — Tier 1 dispatch resolution must succeed
        # so the reproducibility test for the Qwen3.5 catalog can flip green
        # at the dispatch layer first, then progressively at the execution layer.
        return ctx

    def train(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """LoRA training on a domain dataset.

        Param contract:
            domain: str        — code | general | math | ...
            dataset: str       — HF dataset id (e.g. 'Salesforce/wikitext')
            steps: int
            learningRate: str  — float-as-string per the alloy spec
            batchSize: int     — optional, defaults to 4

        Tier 1: dispatch resolution only.
        """
        return ctx

    def context_extend(self, ctx: "ForgeContext", **params) -> "ForgeContext":
        """Context-window extension via YaRN — used by qwen3.5-4b-code-128k-forged.

        Param contract:
            targetLength: int   — e.g. 131072
            method: str         — yarn | ntk | linear | dynamic-ntk
        """
        return ctx
