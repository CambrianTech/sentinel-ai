"""TDD spec for QwenVLAdapter — vision-safety integration via the family
adapter set.

Roadmap step 6 from docs/PLUGIN-SPRINT.md: the vision_safety.py whitelist
module (committed in f82773b as part of the VL forge scaffolding) is
load-bearing for any future Qwen3.5-VL re-forge — it identifies the vision
tower + merger params + vision token vocab indices as untouchable so the
prune / train / quant stages don't silently destroy the vision pathway.
This step wires that module into a new family adapter so the dispatch
test routes VL alloys through a path that consults the whitelist.

Written test-first per TDD/TDValidation discipline. The contract this
test asserts IS the spec the adapter must satisfy. The test does NOT
load a real VL model — it verifies dispatch + import-time integration
+ the contract that prune() / train() consult vision_safety before
touching tensors.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ── Adapter is registered ───────────────────────────────────────────────────


def test_qwen_vl_adapter_module_is_importable():
    """The qwen_vl module exists in the adapter package and is importable
    on a Mac without torch (lazy-imports vision_safety inside methods)."""
    import importlib
    mod = importlib.import_module("adapters.qwen_vl")
    assert mod is not None
    assert hasattr(mod, "QwenVLAdapter")


def test_qwen_vl_adapter_registers_qwen2_5_vl():
    """Resolving an alloy with source.architecture='qwen2_5_vl' MUST yield
    a QwenVLAdapter instance. This is the morning's-flagship-class
    Qwen2.5-VL family that the existing vision_safety.py module was
    written against."""
    from adapters import resolve_family_adapter
    from adapters.qwen_vl import QwenVLAdapter
    a = resolve_family_adapter("qwen2_5_vl")
    assert isinstance(a, QwenVLAdapter)


def test_qwen_vl_adapter_registers_qwen3_5_vl():
    """Same family pattern, different release. When Qwen3.5-VL ships, the
    existing adapter handles it without code changes — same vision tower
    layout, same vision_safety whitelist applies. Adapter declares both
    architecture strings in its tuple."""
    from adapters import resolve_family_adapter
    from adapters.qwen_vl import QwenVLAdapter
    a = resolve_family_adapter("qwen3_5_vl")
    assert isinstance(a, QwenVLAdapter)


def test_qwen_vl_adapter_inherits_from_qwen_dense_base():
    """The VL adapter is a dense Qwen variant with a vision tower attached;
    its tensor walks for prune / train at the text-decoder layer are
    identical to QwenDenseBase. The vision_safety integration is a
    decorator pattern on top of the inherited bodies, not a separate
    code path."""
    from adapters.qwen_vl import QwenVLAdapter
    from adapters.qwen_dense_base import QwenDenseBase
    assert issubclass(QwenVLAdapter, QwenDenseBase)


# ── prune / train consult vision_safety ─────────────────────────────────────


def test_prune_method_references_vision_safety():
    """QwenVLAdapter.prune MUST consult vision_safety.py in its body before
    invoking the inherited prune. The contract: vision-tower params are
    untouchable, so the adapter computes the whitelist from ctx.model and
    uses it to guard the prune path. If this assertion fails, prune is
    just inheriting the base unchanged and would silently destroy the
    vision tower the next time it ran on a real VL model."""
    from adapters.qwen_vl import QwenVLAdapter
    src = inspect.getsource(QwenVLAdapter.prune)
    assert "vision_safety" in src, (
        "QwenVLAdapter.prune body must lazy-import scripts.vision_safety "
        "and use the whitelist to guard against modifying vision-tower params. "
        "If the body is empty / inherits from base, vision tower will be "
        "silently corrupted on a real VL forge."
    )


def test_train_method_references_vision_safety():
    """QwenVLAdapter.train MUST also consult vision_safety — the LoRA
    target_modules pass needs filter_target_modules() to drop any vision-
    side projection that happens to share a name with text-side ones.
    Without this, a recovery LoRA could attach to vision_proj layers
    and the merge_and_unload step would corrupt the vision tower."""
    from adapters.qwen_vl import QwenVLAdapter
    src = inspect.getsource(QwenVLAdapter.train)
    assert "vision_safety" in src or "filter_target_modules" in src, (
        "QwenVLAdapter.train body must lazy-import vision_safety and use "
        "filter_target_modules to drop vision-side projections from the "
        "LoRA target list."
    )


def test_modality_method_is_overridden():
    """The modality stage handler MUST be a real override (not the
    NotImplementedError stub from FamilyAdapter base). VL alloys
    declare modality stages to attach vision encoders; if the adapter
    doesn't handle them, the dispatch path fails immediately."""
    from adapters.qwen_vl import QwenVLAdapter
    from adapters.base import FamilyAdapter
    assert QwenVLAdapter.modality is not FamilyAdapter.modality, (
        "QwenVLAdapter.modality must be overridden — VL alloys carry "
        "modality stages and the base default raises NotImplementedError."
    )


# ── Dispatch test on a synthetic VL alloy ───────────────────────────────────


def test_dispatch_resolves_synthetic_vl_alloy():
    """A synthetic VL alloy MUST resolve through QwenVLAdapter without any
    branches in the dispatch layer. Uses an in-memory dict (not a file)
    so the test is hermetic and doesn't depend on a real published VL
    artifact existing yet."""
    from adapters import resolve_adapter_chain
    from adapters.qwen_vl import QwenVLAdapter

    synthetic_vl_alloy = {
        "name": "synthetic-qwen2.5-vl-test",
        "version": "0.0.1-test",
        "source": {
            "baseModel": "Qwen/Qwen2.5-VL-3B-Instruct",
            "architecture": "qwen2_5_vl",
        },
        "stages": [
            {
                "type": "modality",
                "modality": "vision",
                "encoderModel": "google/siglip-so400m-patch14-384",
                "freezeBase": True,
                "freezeEncoder": True,
            },
            {
                "type": "prune",
                "strategy": "activation-magnitude",
                "level": 0.2,
            },
            {
                "type": "train",
                "domain": "general",
                "dataset": "Salesforce/wikitext",
                "steps": 100,
                "learningRate": "1e-4",
            },
        ],
        "cycles": 1,
    }
    chain = resolve_adapter_chain(synthetic_vl_alloy)
    assert len(chain) == 3
    family = chain[0].family_adapter
    assert isinstance(family, QwenVLAdapter)
    # Each stage must resolve to the right method on QwenVLAdapter
    method_names = [c.method_name for c in chain]
    assert method_names == ["modality", "prune", "train"]


# ── vision_safety integration smoke (does NOT load a real VL model) ─────────


def test_vision_safety_module_is_importable():
    """vision_safety.py itself must remain importable from the scripts
    directory — the adapter lazy-imports it inside the methods, so this
    is a sanity check that the import path is correct."""
    sys_path = list(sys.path)
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import vision_safety
        assert hasattr(vision_safety, "build_whitelist_from_model")
        assert hasattr(vision_safety, "filter_target_modules")
        assert hasattr(vision_safety, "assert_vl_config")
    finally:
        sys.path[:] = sys_path
