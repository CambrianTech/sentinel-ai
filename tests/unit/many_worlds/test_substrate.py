"""Unit tests for scripts/many_worlds/substrate.py.

These tests verify the SubstrateVectorSpace primitive in isolation
without loading any real base models. They cover:
  - Construction from SubstrateConfig
  - write() and read() tensor shape contracts
  - Round-trip self-consistency (write then read with same query)
  - Log-variance clamping
  - Save/load roundtrip
  - is_trained flag behavior
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make scripts/ importable
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

torch = pytest.importorskip("torch")

from many_worlds.substrate import SubstrateConfig, SubstrateVectorSpace


# ── SubstrateConfig ────────────────────────────────────────────────────


def test_substrate_config_defaults():
    cfg = SubstrateConfig()
    assert cfg.dimensionality == 128
    assert cfg.num_bases == 1024
    assert cfg.init_strategy == "orthogonal"
    assert cfg.seed == 42


def test_substrate_config_serialization_roundtrip():
    cfg = SubstrateConfig(dimensionality=256, num_bases=512, init_strategy="normal", seed=7)
    d = cfg.to_dict()
    cfg2 = SubstrateConfig.from_dict(d)
    assert cfg2.dimensionality == 256
    assert cfg2.num_bases == 512
    assert cfg2.init_strategy == "normal"
    assert cfg2.seed == 7


# ── SubstrateVectorSpace construction ──────────────────────────────────


def test_substrate_construction_defaults():
    substrate = SubstrateVectorSpace(SubstrateConfig(dimensionality=16, num_bases=32))
    assert substrate.config.dimensionality == 16
    assert substrate.config.num_bases == 32
    assert not substrate.is_trained
    assert substrate.device == "cpu"


def test_substrate_module_lazy_build():
    """The torch module is only constructed on first access via .module"""
    substrate = SubstrateVectorSpace(SubstrateConfig(dimensionality=16, num_bases=32))
    # Before access, internal _module is None
    assert substrate._module is None
    # Access triggers build
    module = substrate.module
    assert module is not None
    # Second access returns the same module (not rebuilt)
    assert substrate.module is module


def test_substrate_parameters_available():
    """The optimizer needs to be able to iterate over parameters."""
    substrate = SubstrateVectorSpace(SubstrateConfig(dimensionality=16, num_bases=32))
    params = list(substrate.parameters())
    assert len(params) > 0  # at least the bases parameter
    for p in params:
        assert p.requires_grad


@pytest.mark.parametrize("init_strategy", ["orthogonal", "xavier", "normal"])
def test_substrate_init_strategies(init_strategy):
    """All three init strategies should produce valid bases."""
    cfg = SubstrateConfig(
        dimensionality=16,
        num_bases=32,
        init_strategy=init_strategy,
        use_weight_norm=False,  # weight_norm breaks orthogonality
    )
    substrate = SubstrateVectorSpace(cfg)
    bases = substrate.module.bases
    assert bases.shape == (32, 16)
    assert bases.dtype == torch.float32
    # No NaN/Inf
    assert torch.isfinite(bases).all()


# ── write() tensor shapes ──────────────────────────────────────────────


def test_substrate_write_shape_contract():
    """write() produces per-token field assignments over bases."""
    cfg = SubstrateConfig(dimensionality=16, num_bases=32)
    substrate = SubstrateVectorSpace(cfg)

    batch, seq, d = 2, 5, 16
    mu = torch.randn(batch, seq, d)
    log_var = torch.zeros(batch, seq, d)
    field = substrate.write(mu, log_var)

    assert field.shape == (batch, seq, cfg.num_bases)
    # Field is a softmax output → each token's row sums to 1
    row_sums = field.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    # All values in [0, 1]
    assert (field >= 0).all()
    assert (field <= 1).all()


def test_substrate_write_handles_different_seq_lengths():
    substrate = SubstrateVectorSpace(SubstrateConfig(dimensionality=8, num_bases=16))
    for seq_len in [1, 5, 50]:
        mu = torch.randn(1, seq_len, 8)
        log_var = torch.zeros(1, seq_len, 8)
        field = substrate.write(mu, log_var)
        assert field.shape == (1, seq_len, 16)


def test_substrate_write_clamps_log_var():
    """Extreme log_var values should be clamped to config range."""
    cfg = SubstrateConfig(
        dimensionality=8, num_bases=16, log_var_min=-2.0, log_var_max=2.0,
    )
    substrate = SubstrateVectorSpace(cfg)

    # Extreme log_var values
    mu = torch.randn(1, 3, 8)
    log_var = torch.tensor([[[100.0] * 8, [-100.0] * 8, [0.0] * 8]])

    # Should not crash, should produce valid field
    field = substrate.write(mu, log_var)
    assert torch.isfinite(field).all()
    assert field.shape == (1, 3, 16)


# ── read() tensor shapes ───────────────────────────────────────────────


def test_substrate_read_shape_contract():
    """read() produces per-token dense vectors in substrate coordinates."""
    cfg = SubstrateConfig(dimensionality=16, num_bases=32)
    substrate = SubstrateVectorSpace(cfg)

    batch, seq, d = 2, 5, 16
    query_mu = torch.randn(batch, seq, d)
    query_log_var = torch.zeros(batch, seq, d)
    read_vec = substrate.read(query_mu, query_log_var)

    assert read_vec.shape == (batch, seq, d)
    assert torch.isfinite(read_vec).all()


def test_substrate_read_is_weighted_basis_combination():
    """The read output should be a convex combination of bases."""
    cfg = SubstrateConfig(dimensionality=8, num_bases=4)
    substrate = SubstrateVectorSpace(cfg)

    # A single-token query
    query_mu = torch.zeros(1, 1, 8)
    query_log_var = torch.zeros(1, 1, 8)
    read_vec = substrate.read(query_mu, query_log_var)

    # The read vector should be expressible as field @ bases
    field = substrate.write(query_mu, query_log_var)
    expected = field @ substrate.module.bases
    assert torch.allclose(read_vec, expected, atol=1e-5)


# ── Save / load ────────────────────────────────────────────────────────


def test_substrate_save_load_roundtrip(tmp_path):
    """Save a substrate, load it, verify weights match."""
    cfg = SubstrateConfig(dimensionality=16, num_bases=32, seed=123)
    substrate = SubstrateVectorSpace(cfg)
    original_bases = substrate.module.bases.detach().clone()

    path = tmp_path / "substrate.pt"
    substrate.save(str(path))
    assert path.exists()

    loaded = SubstrateVectorSpace.load(str(path))
    loaded_bases = loaded.module.bases.detach().clone()

    assert torch.allclose(original_bases, loaded_bases)
    assert loaded.config.dimensionality == 16
    assert loaded.config.num_bases == 32
    assert loaded.config.seed == 123


def test_substrate_save_load_preserves_trained_flag(tmp_path):
    substrate = SubstrateVectorSpace(SubstrateConfig(dimensionality=8, num_bases=16))
    substrate.mark_trained()
    assert substrate.is_trained

    path = tmp_path / "substrate.pt"
    substrate.save(str(path))

    loaded = SubstrateVectorSpace.load(str(path))
    assert loaded.is_trained


# ── Differentiability ──────────────────────────────────────────────────


def test_substrate_write_is_differentiable():
    """Gradients flow through write() back to the input tensors."""
    substrate = SubstrateVectorSpace(SubstrateConfig(dimensionality=8, num_bases=16))

    mu = torch.randn(1, 3, 8, requires_grad=True)
    log_var = torch.zeros(1, 3, 8, requires_grad=True)
    field = substrate.write(mu, log_var)
    loss = field.sum()
    loss.backward()

    assert mu.grad is not None
    assert log_var.grad is not None
    assert torch.isfinite(mu.grad).all()
    assert torch.isfinite(log_var.grad).all()


def test_substrate_bases_are_learnable():
    """The substrate's bases receive gradients during training."""
    substrate = SubstrateVectorSpace(SubstrateConfig(dimensionality=8, num_bases=16))

    mu = torch.randn(1, 3, 8)
    log_var = torch.zeros(1, 3, 8)
    read_vec = substrate.read(mu, log_var)
    loss = read_vec.pow(2).sum()
    loss.backward()

    # The bases parameter should have a non-zero gradient
    # (with weight_norm, the gradient lives on bases_g and bases_v)
    has_grad = False
    for name, p in substrate.module.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "no parameter received a gradient from read()"
