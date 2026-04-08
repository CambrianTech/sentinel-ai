"""Qwen2DenseAdapter — family adapter for the Qwen2 / Qwen2.5 dense lineage.

Pure inheritance from QwenDenseBase. The Qwen2 lineage forge_model code path
is identical to Qwen3 at the layer this adapter operates on (forge_model.prune
+ forge_model.train_lora are both architecture-agnostic for dense
Qwen-family models), and the §4.1.3.3 compensation distillation flow lives
in QwenDenseBase.train()'s teacher-field dispatch which both families share.

Handles every published continuum-ai/qwen2.5-* artifact whose source
declares architecture='qwen2':

    qwen2.5-0.5b-general-forged       (alloy backfilled 2026-04-08)
    qwen2.5-1.5b-general-forged       (alloy backfilled 2026-04-08)
    qwen2.5-3b-general-forged         (alloy backfilled 2026-04-08)
    qwen2.5-coder-7b-compacted        (the v2-7b §4.1.3.3 compensation anchor)

Reproducibility contract: this adapter MUST stay frozen against those 4
artifacts. Changes to behavior land as a NEW adapter under a NEW
architecture string, never as edits to this file or to QwenDenseBase.
"""

from __future__ import annotations
from .qwen_dense_base import QwenDenseBase
from .registry import register_family_adapter


@register_family_adapter
class Qwen2DenseAdapter(QwenDenseBase):
    """Family adapter for Qwen2 / Qwen2.5 dense models — pure inheritance."""

    architectures = ("qwen2",)
