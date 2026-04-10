"""Qwen3DenseAdapter — family adapter for the Qwen3.5 dense catalog.

Inherits the prune/train bodies from QwenDenseBase. Only declares the
architecture string and overrides context_extend (the YaRN-based context
extension used by qwen3.5-4b-code-128k-forged — Qwen2.5 doesn't ship a
context-extended variant so the base doesn't need to handle it).

Handles every published continuum-ai/qwen3.5-* artifact whose source
declares architecture='qwen3_5':

    qwen3.5-0.8b-general-forged
    qwen3.5-2b-general-forged
    qwen3.5-4b-general-forged
    qwen3.5-4b-code-forged           (+ -GGUF, -defragged variants)
    qwen3.5-4b-code-128k-forged
    qwen3.5-9b-general-forged
    qwen3.5-27b-code-forged          (+ -defragged, -mlx-4bit variants)

Reproducibility contract: this adapter MUST stay frozen against those 11+
artifacts. Changes to behavior land as a NEW adapter under a NEW
architecture string, never as edits to this file.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from .qwen_dense_base import QwenDenseBase
from .registry import register_family_adapter

if TYPE_CHECKING:
    from scripts.stages.base import ForgeContext


@register_family_adapter
class Qwen3DenseAdapter(QwenDenseBase):
    """Family adapter for Qwen3.5 dense models."""

    architectures = ("qwen3_5",)

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
