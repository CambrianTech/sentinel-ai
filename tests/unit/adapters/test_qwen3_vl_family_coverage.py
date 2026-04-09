"""TDD spec: QwenVLAdapter must cover the Qwen3-VL family.

Surfaced by reading HF config.json for the seed-catalog candidates:

    Qwen/Qwen3-VL-8B-Instruct          → model_type: 'qwen3_vl'
    Qwen/Qwen3-VL-30B-A3B-Instruct     → model_type: 'qwen3_vl_moe'

Neither is in QwenVLAdapter.architectures today (only qwen2_5_vl and
qwen3_5_vl). Two of the seed-catalog targets would dispatch-fail at
intake without this.

Fix: add 'qwen3_vl' and 'qwen3_vl_moe' to QwenVLAdapter.architectures.
The forge code path is unchanged — Qwen3-VL is the same vision-tower +
LLM split as Qwen2.5-VL, just with newer weights and (for the MoE
variant) the same Qwen3MoE layout the existing pruner already handles.
The adapter just needs to claim the dispatch tags.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def test_qwen_vl_adapter_registers_qwen3_vl():
    from adapters import resolve_family_adapter
    from adapters.qwen_vl import QwenVLAdapter
    a = resolve_family_adapter("qwen3_vl")
    assert isinstance(a, QwenVLAdapter)


def test_qwen_vl_adapter_registers_qwen3_vl_moe():
    from adapters import resolve_family_adapter
    from adapters.qwen_vl import QwenVLAdapter
    a = resolve_family_adapter("qwen3_vl_moe")
    assert isinstance(a, QwenVLAdapter)


def test_qwen_vl_adapter_architectures_tuple_complete():
    from adapters.qwen_vl import QwenVLAdapter
    expected = {"qwen2_5_vl", "qwen3_5_vl", "qwen3_vl", "qwen3_vl_moe"}
    actual = set(QwenVLAdapter.architectures)
    missing = expected - actual
    assert not missing, f"QwenVLAdapter missing architectures: {missing}"
