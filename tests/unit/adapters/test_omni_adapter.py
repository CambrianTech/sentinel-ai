"""TDD spec for QwenOmniAdapter — text+vision+audio in, text+speech out.

Roadmap "fill the gaps" follow-up: Kash's frontier-target mapping
identifies Qwen2.5-Omni-7B as the **only clean-license native omni
model** on HuggingFace (Apache-2.0, ~15GB fp16, ~7GB at 3B variant).
It's the obvious foundation for the lab's first native omni forge AND
fills the existing 'Qwen3-Omni' product agent slot in Continuum.

The convo-with-kash analysis (read 2026-04-08) flagged this as
Priority 1 of the multimodal forge roadmap:

    Qwen2.5-Omni-7B
    text+vision+video+audio IN, text+speech OUT in a single inference loop
    Apache-2.0, no commercial restrictions
    Forge target: 15 GB fp16 → projected 5-7 GB Q4_K_M
    Estimated 3-5 days from existing methodology + vision_safety scaffolding

This TDD test gates the adapter that will execute that forge once it
runs on a 5090. The adapter inherits from QwenDenseBase (text-decoder
layer is dense) and overrides modality() to handle the multi-encoder
omni shape (vision + audio + speech-decoder), with bit-exact
preservation of all three encoder towers via vision_safety-style
whitelisting.

Written test-first per TDD/TDValidation discipline.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def test_qwen_omni_adapter_module_is_importable():
    import importlib
    mod = importlib.import_module("adapters.qwen_omni")
    assert mod is not None
    assert hasattr(mod, "QwenOmniAdapter")


def test_qwen_omni_adapter_registers_qwen2_5_omni():
    """Resolving an alloy with source.architecture='qwen2_5_omni' MUST
    yield a QwenOmniAdapter — the Priority 1 multimodal forge target."""
    from adapters import resolve_family_adapter
    from adapters.qwen_omni import QwenOmniAdapter
    a = resolve_family_adapter("qwen2_5_omni")
    assert isinstance(a, QwenOmniAdapter)


def test_qwen_omni_adapter_inherits_from_dense_base():
    """The text-decoder layer of Qwen2.5-Omni is dense Qwen2.5 architecture
    + multiple encoders bolted on. The text-side prune / train work
    inherits from QwenDenseBase; the encoder-side preservation is added
    via overrides on top, mirroring the QwenVLAdapter pattern."""
    from adapters.qwen_omni import QwenOmniAdapter
    from adapters.qwen_dense_base import QwenDenseBase
    assert issubclass(QwenOmniAdapter, QwenDenseBase)


def test_modality_handles_omni_shape():
    """The omni modality stage MUST handle vision, audio, AND speech
    decoder simultaneously — they're three separate encoder/decoder
    towers attached to the same text-decoder. The modality() body
    references all three pathways and refuses to forge if any of the
    three is missing/damaged.
    """
    from adapters.qwen_omni import QwenOmniAdapter
    src = inspect.getsource(QwenOmniAdapter.modality)
    # Body must reference vision + audio + speech as the three preserved
    # towers. (No specific implementation enforcement here — just that
    # the adapter is aware of the omni shape, not just inheriting a
    # generic modality() that only handles vision.)
    msg = src.lower()
    assert "vision" in msg, "QwenOmniAdapter.modality must handle vision tower"
    assert "audio" in msg, "QwenOmniAdapter.modality must handle audio encoder"
    assert "speech" in msg or "talker" in msg, (
        "QwenOmniAdapter.modality must handle speech decoder / talker"
    )


def test_prune_consults_omni_safety():
    """The omni prune MUST consult a vision-safety-style whitelist that
    covers ALL THREE encoder towers (vision, audio, speech decoder), not
    just the vision one. Same loud-failure principle: if any of the three
    encoder towers' params would be touched, the prune halts.
    """
    from adapters.qwen_omni import QwenOmniAdapter
    src = inspect.getsource(QwenOmniAdapter.prune)
    # The prune body must reference some form of preservation check.
    # Either it lazy-imports vision_safety (covering vision + audio merger)
    # OR it implements its own omni_safety helper. Either is acceptable;
    # the contract is that prune is NOT just inheriting QwenDenseBase.prune.
    assert "vision_safety" in src or "omni_safety" in src or "preserve" in src.lower(), (
        "QwenOmniAdapter.prune must reference an encoder-tower preservation "
        "check — inheriting QwenDenseBase.prune unchanged would silently "
        "destroy the audio encoder + speech decoder of an omni forge."
    )


def test_dispatch_resolves_synthetic_omni_alloy():
    """A synthetic Qwen2.5-Omni alloy MUST resolve through QwenOmniAdapter."""
    from adapters import resolve_adapter_chain
    from adapters.qwen_omni import QwenOmniAdapter

    synthetic_omni = {
        "name": "synthetic-qwen2.5-omni-test",
        "version": "0.0.1-test",
        "source": {
            "baseModel": "Qwen/Qwen2.5-Omni-7B",
            "architecture": "qwen2_5_omni",
        },
        "stages": [
            {
                "type": "modality",
                "modality": "multimodal",
                "encoderModel": "qwen-2.5-omni-built-in",
                "freezeBase": True,
                "freezeEncoder": True,
            },
            {
                "type": "prune",
                "strategy": "activation-magnitude",
                "level": 0.2,
            },
            {
                "type": "quant",
                "format": "gguf",
                "quantTypes": ["Q4_K_M", "Q5_K_M"],
                "deviceTargets": ["macbook-air-16gb", "rtx4070"],
            },
        ],
        "cycles": 1,
    }
    chain = resolve_adapter_chain(synthetic_omni)
    assert len(chain) == 3
    assert isinstance(chain[0].family_adapter, QwenOmniAdapter)
    assert [c.method_name for c in chain] == ["modality", "prune", "quant"]
