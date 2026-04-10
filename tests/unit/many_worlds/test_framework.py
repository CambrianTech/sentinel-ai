"""Unit tests for scripts/many_worlds/framework.py.

Verifies ManyWorldsFramework orchestration in isolation. Uses
fabricated in-memory base models (small random nn.Module stand-ins)
rather than real HF models so the tests stay fast and don't require
network access or heavy weight files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

torch = pytest.importorskip("torch")

from many_worlds.framework import FrameworkConfig, ManyWorldsFramework, PopulationMember
from many_worlds.project_read import AdapterConfig, AdapterPair
from many_worlds.substrate import SubstrateConfig, SubstrateVectorSpace


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def small_substrate():
    return SubstrateVectorSpace(SubstrateConfig(dimensionality=16, num_bases=32))


@pytest.fixture
def framework(small_substrate):
    return ManyWorldsFramework(
        config=FrameworkConfig(
            name="test-framework",
            substrate_dim=16,
        ),
        substrate=small_substrate,
    )


@pytest.fixture
def adapter_a():
    return AdapterPair(
        config=AdapterConfig(residual_hidden_size=64, substrate_dim=16, lora_rank=8),
        base_model_name="model-a",
    )


@pytest.fixture
def adapter_b():
    return AdapterPair(
        config=AdapterConfig(residual_hidden_size=128, substrate_dim=16, lora_rank=8),
        base_model_name="model-b",
    )


# ── FrameworkConfig ────────────────────────────────────────────────────


def test_framework_config_defaults():
    cfg = FrameworkConfig()
    assert cfg.name == "many-worlds-v0"
    assert cfg.substrate_dim == 128
    assert cfg.default_layer_fraction == pytest.approx(2.0 / 3.0)
    assert cfg.query_face_routing == "learned_gating"


def test_framework_config_serialization_roundtrip():
    cfg = FrameworkConfig(name="test", substrate_dim=256)
    cfg2 = FrameworkConfig.from_dict(cfg.to_dict())
    assert cfg2.name == "test"
    assert cfg2.substrate_dim == 256


# ── Population management ──────────────────────────────────────────────


def test_framework_starts_with_empty_population(framework):
    assert len(framework.population) == 0
    summary = framework.population_summary()
    assert summary["num_members"] == 0
    assert summary["ready_for_inference"] is True  # empty population is trivially ready


def test_framework_add_member(framework):
    member = framework.add_member(
        name="m1",
        base_model_repo="fake/model-1",
        architecture="fake_dense",
        residual_hidden_size=64,
        num_hidden_layers=12,
    )
    assert member.name == "m1"
    assert member.base_model_repo == "fake/model-1"
    assert member.architecture == "fake_dense"
    assert member.residual_hidden_size == 64
    assert member.num_hidden_layers == 12
    # Default layer_idx = 2/3 * 12 = 8
    assert member.layer_idx == 8
    assert member.adapter is None
    assert len(framework.population) == 1


def test_framework_add_member_explicit_layer_idx(framework):
    member = framework.add_member(
        name="m1",
        base_model_repo="fake/model-1",
        architecture="fake_dense",
        residual_hidden_size=64,
        num_hidden_layers=12,
        layer_idx=6,
    )
    assert member.layer_idx == 6


def test_framework_add_member_duplicate_name_raises(framework):
    framework.add_member(
        name="m1", base_model_repo="fake/m1", architecture="f",
        residual_hidden_size=64, num_hidden_layers=12,
    )
    with pytest.raises(ValueError, match="already exists"):
        framework.add_member(
            name="m1", base_model_repo="fake/other", architecture="f",
            residual_hidden_size=64, num_hidden_layers=12,
        )


def test_framework_get_member(framework):
    framework.add_member(
        name="m1", base_model_repo="fake/m1", architecture="f",
        residual_hidden_size=64, num_hidden_layers=12,
    )
    member = framework.get_member("m1")
    assert member.name == "m1"


def test_framework_get_member_missing_raises(framework):
    with pytest.raises(KeyError):
        framework.get_member("does-not-exist")


# ── Adapter attachment ─────────────────────────────────────────────────


def test_framework_attach_adapter(framework, adapter_a):
    framework.add_member(
        name="a", base_model_repo="fake/a", architecture="f",
        residual_hidden_size=64, num_hidden_layers=12,
    )
    framework.attach_adapter("a", adapter_a)
    member = framework.get_member("a")
    assert member.adapter is adapter_a


def test_framework_attach_adapter_residual_mismatch_raises(framework):
    framework.add_member(
        name="a", base_model_repo="fake/a", architecture="f",
        residual_hidden_size=64, num_hidden_layers=12,
    )
    wrong_adapter = AdapterPair(
        config=AdapterConfig(residual_hidden_size=128, substrate_dim=16, lora_rank=8),
        base_model_name="wrong",
    )
    with pytest.raises(ValueError, match="residual_hidden_size"):
        framework.attach_adapter("a", wrong_adapter)


def test_framework_attach_adapter_substrate_dim_mismatch_raises(framework):
    framework.add_member(
        name="a", base_model_repo="fake/a", architecture="f",
        residual_hidden_size=64, num_hidden_layers=12,
    )
    wrong_adapter = AdapterPair(
        config=AdapterConfig(residual_hidden_size=64, substrate_dim=64, lora_rank=8),
        base_model_name="wrong",
    )
    with pytest.raises(ValueError, match="substrate_dim"):
        framework.attach_adapter("a", wrong_adapter)


# ── Enable/disable ─────────────────────────────────────────────────────


def test_framework_disable_all_adapters(framework, adapter_a, adapter_b):
    framework.add_member(
        name="a", base_model_repo="fake/a", architecture="f",
        residual_hidden_size=64, num_hidden_layers=12,
    )
    framework.add_member(
        name="b", base_model_repo="fake/b", architecture="f",
        residual_hidden_size=128, num_hidden_layers=24,
    )
    framework.attach_adapter("a", adapter_a)
    framework.attach_adapter("b", adapter_b)

    framework.disable_all_adapters()
    assert framework.get_member("a").adapter.config.enabled is False
    assert framework.get_member("b").adapter.config.enabled is False


def test_framework_enable_all_adapters(framework, adapter_a, adapter_b):
    framework.add_member(
        name="a", base_model_repo="fake/a", architecture="f",
        residual_hidden_size=64, num_hidden_layers=12,
    )
    framework.add_member(
        name="b", base_model_repo="fake/b", architecture="f",
        residual_hidden_size=128, num_hidden_layers=24,
    )
    framework.attach_adapter("a", adapter_a)
    framework.attach_adapter("b", adapter_b)

    framework.disable_all_adapters()
    framework.enable_all_adapters()
    assert framework.get_member("a").adapter.config.enabled is True
    assert framework.get_member("b").adapter.config.enabled is True


# ── Core operations ────────────────────────────────────────────────────


def test_framework_cross_project_shape_contract(framework, adapter_a, adapter_b):
    """cross_project should produce a residual with the TARGET member's size."""
    framework.add_member(
        name="a", base_model_repo="fake/a", architecture="f",
        residual_hidden_size=64, num_hidden_layers=12,
    )
    framework.add_member(
        name="b", base_model_repo="fake/b", architecture="f",
        residual_hidden_size=128, num_hidden_layers=24,
    )
    framework.attach_adapter("a", adapter_a)
    framework.attach_adapter("b", adapter_b)

    source_residual = torch.randn(2, 5, 64)  # model-a's residual size
    target_residual = framework.cross_project("a", "b", source_residual)
    assert target_residual.shape == (2, 5, 128)  # model-b's residual size


def test_framework_project_residual_without_adapter_raises(framework):
    framework.add_member(
        name="a", base_model_repo="fake/a", architecture="f",
        residual_hidden_size=64, num_hidden_layers=12,
    )
    residual = torch.randn(1, 3, 64)
    with pytest.raises(ValueError, match="no adapter attached"):
        framework.project_residual("a", residual)


# ── Parameters ─────────────────────────────────────────────────────────


def test_framework_substrate_parameters_yields_substrate_params(framework):
    params = list(framework.substrate_parameters())
    assert len(params) > 0


def test_framework_adapter_parameters_empty_when_no_adapters(framework):
    framework.add_member(
        name="a", base_model_repo="fake/a", architecture="f",
        residual_hidden_size=64, num_hidden_layers=12,
    )
    params = list(framework.adapter_parameters())
    assert len(params) == 0


def test_framework_adapter_parameters_yields_after_attach(framework, adapter_a):
    framework.add_member(
        name="a", base_model_repo="fake/a", architecture="f",
        residual_hidden_size=64, num_hidden_layers=12,
    )
    framework.attach_adapter("a", adapter_a)
    params = list(framework.adapter_parameters())
    assert len(params) > 0


def test_framework_adapter_parameters_scoped_by_member(framework, adapter_a, adapter_b):
    framework.add_member(
        name="a", base_model_repo="fake/a", architecture="f",
        residual_hidden_size=64, num_hidden_layers=12,
    )
    framework.add_member(
        name="b", base_model_repo="fake/b", architecture="f",
        residual_hidden_size=128, num_hidden_layers=24,
    )
    framework.attach_adapter("a", adapter_a)
    framework.attach_adapter("b", adapter_b)

    a_params = list(framework.adapter_parameters("a"))
    b_params = list(framework.adapter_parameters("b"))
    all_params = list(framework.adapter_parameters())
    assert len(a_params) > 0
    assert len(b_params) > 0
    assert len(all_params) == len(a_params) + len(b_params)


# ── Save / load ────────────────────────────────────────────────────────


def test_framework_save_load_roundtrip(tmp_path, framework, adapter_a):
    framework.add_member(
        name="a", base_model_repo="fake/a", architecture="f",
        residual_hidden_size=64, num_hidden_layers=12,
    )
    framework.attach_adapter("a", adapter_a)

    save_dir = tmp_path / "framework-snapshot"
    framework.save(str(save_dir))

    # Verify expected files
    assert (save_dir / "manifest.json").exists()
    assert (save_dir / "substrate.pt").exists()
    assert (save_dir / "adapters" / "a.pt").exists()

    # Load and verify
    loaded = ManyWorldsFramework.load(str(save_dir))
    assert loaded.config.name == "test-framework"
    assert loaded.config.substrate_dim == 16
    assert len(loaded.population) == 1
    assert loaded.get_member("a").adapter is not None
    assert loaded.get_member("a").residual_hidden_size == 64


def test_framework_save_load_empty_population(tmp_path, framework):
    save_dir = tmp_path / "empty-framework"
    framework.save(str(save_dir))

    loaded = ManyWorldsFramework.load(str(save_dir))
    assert len(loaded.population) == 0
