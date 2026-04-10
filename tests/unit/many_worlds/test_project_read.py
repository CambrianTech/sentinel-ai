"""Unit tests for scripts/many_worlds/project_read.py.

Verifies ProjectModule, ReadModule, and AdapterPair primitives in
isolation. Pure torch tests — no real base models required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

torch = pytest.importorskip("torch")

from many_worlds.project_read import AdapterConfig, AdapterPair, ProjectModule, ReadModule


# ── AdapterConfig ──────────────────────────────────────────────────────


def test_adapter_config_defaults():
    cfg = AdapterConfig(residual_hidden_size=2048)
    assert cfg.residual_hidden_size == 2048
    assert cfg.substrate_dim == 128
    assert cfg.lora_rank == 64
    assert cfg.enabled is True
    assert cfg.output_scale_init == 0.01


def test_adapter_config_serialization_roundtrip():
    cfg = AdapterConfig(
        residual_hidden_size=4096,
        substrate_dim=512,
        lora_rank=128,
        layer_idx=24,
        output_scale_init=0.05,
    )
    cfg2 = AdapterConfig.from_dict(cfg.to_dict())
    assert cfg2.residual_hidden_size == 4096
    assert cfg2.substrate_dim == 512
    assert cfg2.lora_rank == 128
    assert cfg2.layer_idx == 24
    assert cfg2.output_scale_init == 0.05


# ── ProjectModule ──────────────────────────────────────────────────────


def test_project_module_construction():
    cfg = AdapterConfig(residual_hidden_size=128, substrate_dim=32, lora_rank=16)
    proj = ProjectModule(cfg)
    _ = proj.module  # trigger lazy build
    params = list(proj.parameters())
    assert len(params) > 0


def test_project_module_output_shape():
    """Project produces (mu, log_var) of shape (B, S, substrate_dim) each."""
    cfg = AdapterConfig(residual_hidden_size=128, substrate_dim=32, lora_rank=16)
    proj = ProjectModule(cfg)

    residual = torch.randn(2, 5, 128)
    mu, log_var = proj(residual)

    assert mu.shape == (2, 5, 32)
    assert log_var.shape == (2, 5, 32)
    assert torch.isfinite(mu).all()
    assert torch.isfinite(log_var).all()


def test_project_module_zero_init_produces_near_zero_output():
    """Freshly-initialized Project should produce near-zero mu output.

    The mean_head is zero-initialized so the initial projection
    contribution to substrate coordinates is effectively zero. The
    log_var_head is also zero-initialized (so initial log_var ≈ 0,
    meaning σ² ≈ 1, unit variance).
    """
    cfg = AdapterConfig(residual_hidden_size=128, substrate_dim=32, log_var_init=0.0)
    proj = ProjectModule(cfg)

    residual = torch.randn(2, 5, 128)
    mu, log_var = proj(residual)

    # mu should be near zero (zero-init mean head + small output scale)
    assert mu.abs().mean() < 1e-3
    # log_var should be at the init value (~0)
    assert log_var.abs().mean() < 1e-3


def test_project_module_disabled_returns_zeros():
    """When enabled=False, Project returns zero tensors."""
    cfg = AdapterConfig(residual_hidden_size=128, substrate_dim=32, enabled=False)
    proj = ProjectModule(cfg)

    residual = torch.randn(2, 5, 128)
    mu, log_var = proj(residual)

    assert torch.all(mu == 0)
    assert torch.all(log_var == 0)


def test_project_module_set_enabled_toggles_behavior():
    cfg = AdapterConfig(residual_hidden_size=128, substrate_dim=32)
    proj = ProjectModule(cfg)

    # Start enabled
    proj.set_enabled(True)
    residual = torch.randn(2, 5, 128)
    mu1, _ = proj(residual)

    # Disable
    proj.set_enabled(False)
    mu2, _ = proj(residual)
    assert torch.all(mu2 == 0)

    # Re-enable
    proj.set_enabled(True)
    mu3, _ = proj(residual)
    # Should match mu1 (same deterministic input + weights)
    assert torch.allclose(mu1, mu3)


# ── ReadModule ─────────────────────────────────────────────────────────


def test_read_module_output_shape():
    """Read produces (B, S, residual_hidden_size) from (B, S, substrate_dim)."""
    cfg = AdapterConfig(residual_hidden_size=128, substrate_dim=32, lora_rank=16)
    read = ReadModule(cfg)

    substrate_vector = torch.randn(2, 5, 32)
    residual_delta = read(substrate_vector)

    assert residual_delta.shape == (2, 5, 128)
    assert torch.isfinite(residual_delta).all()


def test_read_module_zero_init_no_op():
    """Freshly-initialized Read produces near-zero output (zero-init out_proj)."""
    cfg = AdapterConfig(residual_hidden_size=128, substrate_dim=32)
    read = ReadModule(cfg)

    substrate_vector = torch.randn(2, 5, 32)
    residual_delta = read(substrate_vector)

    assert residual_delta.abs().mean() < 1e-6


def test_read_module_disabled_returns_zeros():
    cfg = AdapterConfig(residual_hidden_size=128, substrate_dim=32, enabled=False)
    read = ReadModule(cfg)

    substrate_vector = torch.randn(2, 5, 32)
    residual_delta = read(substrate_vector)

    assert torch.all(residual_delta == 0)


# ── AdapterPair ────────────────────────────────────────────────────────


def test_adapter_pair_construction():
    cfg = AdapterConfig(residual_hidden_size=128, substrate_dim=32, lora_rank=16)
    pair = AdapterPair(cfg, base_model_name="test-model-1")
    assert pair.base_model_name == "test-model-1"
    assert pair.project is not None
    assert pair.read is not None


def test_adapter_pair_yields_both_sets_of_parameters():
    cfg = AdapterConfig(residual_hidden_size=128, substrate_dim=32, lora_rank=16)
    pair = AdapterPair(cfg, base_model_name="test-model-1")

    # Trigger lazy build
    _ = pair.project.module
    _ = pair.read.module

    pair_params = list(pair.parameters())
    project_params = list(pair.project.parameters())
    read_params = list(pair.read.parameters())

    assert len(pair_params) == len(project_params) + len(read_params)


def test_adapter_pair_set_enabled_affects_both():
    cfg = AdapterConfig(residual_hidden_size=128, substrate_dim=32)
    pair = AdapterPair(cfg, base_model_name="test-model-1")

    pair.set_enabled(False)
    residual = torch.randn(1, 3, 128)
    mu, log_var = pair.project(residual)
    assert torch.all(mu == 0)
    assert torch.all(log_var == 0)

    substrate_vec = torch.randn(1, 3, 32)
    out = pair.read(substrate_vec)
    assert torch.all(out == 0)

    pair.set_enabled(True)
    mu, _ = pair.project(residual)
    # After enable, output should be trainable (may still be small due
    # to zero-init, but the call path is active)
    assert mu.shape == (1, 3, 32)


def test_adapter_pair_save_load_roundtrip(tmp_path):
    cfg = AdapterConfig(residual_hidden_size=128, substrate_dim=32, lora_rank=16)
    pair = AdapterPair(cfg, base_model_name="test-model-xyz")

    # Trigger lazy build and record initial state
    _ = pair.project.module
    _ = pair.read.module
    # Modify some weights so we have a non-default state to compare
    with torch.no_grad():
        for p in pair.project.module.mean_head.parameters():
            p.fill_(0.5)
    original_mean_head = pair.project.module.mean_head.weight.detach().clone()

    path = tmp_path / "adapter.pt"
    pair.save(str(path))
    assert path.exists()

    loaded = AdapterPair.load(str(path))
    assert loaded.base_model_name == "test-model-xyz"
    assert loaded.config.residual_hidden_size == 128

    loaded_mean_head = loaded.project.module.mean_head.weight.detach().clone()
    assert torch.allclose(original_mean_head, loaded_mean_head)


# ── Differentiability ──────────────────────────────────────────────────


def test_project_module_is_differentiable():
    cfg = AdapterConfig(residual_hidden_size=64, substrate_dim=16)
    proj = ProjectModule(cfg)

    # Need to unblock the zero-init output heads by modifying them
    with torch.no_grad():
        proj.module.mean_head.weight.fill_(0.01)

    residual = torch.randn(1, 3, 64)
    mu, log_var = proj(residual)
    loss = mu.sum() + log_var.sum()
    loss.backward()

    # At least one parameter should have a non-zero gradient
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in proj.parameters()
    )
    assert has_grad


def test_read_module_is_differentiable():
    cfg = AdapterConfig(residual_hidden_size=64, substrate_dim=16)
    read = ReadModule(cfg)

    # Unblock the zero-init out_proj
    with torch.no_grad():
        read.module.out_proj.weight.fill_(0.01)

    substrate_vec = torch.randn(1, 3, 16)
    out = read(substrate_vec)
    loss = out.sum()
    loss.backward()

    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in read.parameters()
    )
    assert has_grad
