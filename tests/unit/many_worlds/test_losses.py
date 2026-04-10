"""Unit tests for scripts/many_worlds/losses.py.

Verifies the two-term Phase A + Phase B loss functions on fabricated
tensor inputs. No real models or training — just checking the math
produces the right shapes, values are finite, and gradients flow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

torch = pytest.importorskip("torch")

from many_worlds.losses import (
    PhaseALossConfig,
    PhaseBLossConfig,
    contrastive_alignment_loss,
    native_preservation_loss,
    phase_a_loss,
    phase_b_loss,
    round_trip_reconstruction_loss,
)


# ── Contrastive alignment ──────────────────────────────────────────────


def test_contrastive_alignment_two_members():
    """Two-member population produces a finite scalar loss."""
    projections = {
        "a": torch.randn(8, 16),  # (batch, substrate_dim)
        "b": torch.randn(8, 16),
    }
    loss = contrastive_alignment_loss(projections, temperature=0.07)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert loss.item() > 0  # InfoNCE is positive


def test_contrastive_alignment_perfect_alignment_yields_low_loss():
    """If two members project to identical points, loss should be minimal."""
    # Same vectors for both members → perfect positive pairs
    base = torch.randn(8, 16)
    projections = {"a": base.clone(), "b": base.clone()}
    loss = contrastive_alignment_loss(projections, temperature=0.07)

    # Random vectors for the same inputs across members
    rand_projections = {
        "a": torch.randn(8, 16),
        "b": torch.randn(8, 16),
    }
    random_loss = contrastive_alignment_loss(rand_projections, temperature=0.07)

    # Identical projections should have lower loss than random
    assert loss.item() < random_loss.item()


def test_contrastive_alignment_single_member_returns_zero():
    """Single-member populations have no contrastive signal."""
    projections = {"a": torch.randn(8, 16)}
    loss = contrastive_alignment_loss(projections)
    assert loss.item() == 0.0


def test_contrastive_alignment_three_members():
    """Three-member populations compute pairwise loss over all 3 pairs."""
    projections = {
        "a": torch.randn(4, 8),
        "b": torch.randn(4, 8),
        "c": torch.randn(4, 8),
    }
    loss = contrastive_alignment_loss(projections, temperature=0.07)
    assert torch.isfinite(loss)
    assert loss.item() > 0


# ── Round-trip reconstruction ──────────────────────────────────────────


def test_round_trip_mse_loss():
    original = torch.randn(2, 5, 16)
    reconstructed = original + 0.1 * torch.randn(2, 5, 16)
    loss = round_trip_reconstruction_loss(original, reconstructed, loss_type="mse")
    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_round_trip_mse_is_zero_for_identical():
    x = torch.randn(2, 5, 16)
    loss = round_trip_reconstruction_loss(x, x.clone(), loss_type="mse")
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_round_trip_cosine_loss():
    original = torch.randn(2, 5, 16)
    reconstructed = original.clone()
    loss = round_trip_reconstruction_loss(original, reconstructed, loss_type="cosine")
    assert loss.item() == pytest.approx(0.0, abs=1e-5)

    # Anti-correlated reconstruction should have high cosine loss
    anti = -original
    loss_anti = round_trip_reconstruction_loss(original, anti, loss_type="cosine")
    assert loss_anti.item() > loss.item()


def test_round_trip_l1_loss():
    original = torch.randn(2, 5, 16)
    reconstructed = original + 0.5
    loss = round_trip_reconstruction_loss(original, reconstructed, loss_type="l1")
    assert loss.item() == pytest.approx(0.5, abs=1e-5)


def test_round_trip_unknown_loss_type_raises():
    with pytest.raises(ValueError, match="unknown loss_type"):
        round_trip_reconstruction_loss(
            torch.randn(1, 1, 4), torch.randn(1, 1, 4), loss_type="bogus"
        )


# ── Native preservation ────────────────────────────────────────────────


def test_native_preservation_zero_for_small_scale():
    """Within max_scale, the regularizer is zero."""
    scale = torch.tensor(0.5)
    loss = native_preservation_loss(scale, max_scale=1.0)
    assert loss.item() == 0.0


def test_native_preservation_nonzero_for_large_scale():
    """Beyond max_scale, quadratic penalty applies."""
    scale = torch.tensor(2.0)
    loss = native_preservation_loss(scale, max_scale=1.0)
    # excess = |2.0| - 1.0 = 1.0, penalty = 1.0^2 = 1.0
    assert loss.item() == pytest.approx(1.0)


def test_native_preservation_handles_negative_scale():
    """Absolute value means negative scale is also penalized."""
    scale = torch.tensor(-3.0)
    loss = native_preservation_loss(scale, max_scale=1.0)
    # excess = |-3.0| - 1.0 = 2.0, penalty = 2.0^2 = 4.0
    assert loss.item() == pytest.approx(4.0)


# ── Phase A full loss ──────────────────────────────────────────────────


def test_phase_a_loss_structure():
    projections = {
        "a": torch.randn(4, 16),
        "b": torch.randn(4, 16),
    }
    original_residuals = {
        "a": torch.randn(4, 10, 64),
        "b": torch.randn(4, 10, 128),
    }
    reconstructed_residuals = {
        "a": torch.randn(4, 10, 64),
        "b": torch.randn(4, 10, 128),
    }
    cfg = PhaseALossConfig(alpha_contrastive=1.0, beta_round_trip=1.0)
    loss, metrics = phase_a_loss(
        projections, original_residuals, reconstructed_residuals, cfg
    )

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert "phase_a/contrastive" in metrics
    assert "phase_a/round_trip" in metrics
    assert "phase_a/total" in metrics


def test_phase_a_loss_weights_affect_total():
    """Setting α=0 should zero out the contrastive contribution."""
    projections = {
        "a": torch.randn(4, 16),
        "b": torch.randn(4, 16),
    }
    res_orig = {
        "a": torch.randn(4, 10, 64),
        "b": torch.randn(4, 10, 64),
    }
    res_recon = {
        "a": res_orig["a"].clone(),  # perfect reconstruction → zero rt loss
        "b": res_orig["b"].clone(),
    }
    cfg_contrastive_only = PhaseALossConfig(alpha_contrastive=1.0, beta_round_trip=0.0)
    loss_contrastive, _ = phase_a_loss(projections, res_orig, res_recon, cfg_contrastive_only)
    cfg_round_trip_only = PhaseALossConfig(alpha_contrastive=0.0, beta_round_trip=1.0)
    loss_round_trip, _ = phase_a_loss(projections, res_orig, res_recon, cfg_round_trip_only)

    # With perfect reconstruction, round-trip-only loss should be ~0
    assert loss_round_trip.item() == pytest.approx(0.0, abs=1e-5)
    # Contrastive-only loss should be > 0 (non-degenerate)
    assert loss_contrastive.item() > 0


# ── Phase B full loss ──────────────────────────────────────────────────


def test_phase_b_loss_structure():
    residual = torch.randn(2, 5, 64)
    round_trip = residual + 0.1 * torch.randn(2, 5, 64)
    cross_model_loss = torch.tensor(0.5)
    output_scale = torch.tensor(0.3)

    cfg = PhaseBLossConfig()
    loss, metrics = phase_b_loss(
        round_trip, residual, cross_model_loss, output_scale, cfg
    )

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert "phase_b/round_trip" in metrics
    assert "phase_b/cross_model" in metrics
    assert "phase_b/native_reg" in metrics
    assert "phase_b/total" in metrics


def test_phase_b_loss_respects_weights():
    """All weights at zero should produce exactly zero loss."""
    residual = torch.randn(2, 5, 64)
    round_trip = residual + 0.1 * torch.randn(2, 5, 64)
    cross_model_loss = torch.tensor(0.5)
    output_scale = torch.tensor(2.0)  # above max_scale, so native_reg > 0

    cfg = PhaseBLossConfig(
        gamma_round_trip=0.0,
        delta_cross_model=0.0,
        epsilon_native_preservation=0.0,
    )
    loss, _ = phase_b_loss(round_trip, residual, cross_model_loss, output_scale, cfg)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_phase_b_loss_is_differentiable():
    """Gradients flow through the phase B loss."""
    residual = torch.randn(2, 5, 64)
    round_trip = torch.randn(2, 5, 64, requires_grad=True)
    cross_model_loss = torch.tensor(0.5, requires_grad=True)
    output_scale = torch.tensor(0.3, requires_grad=True)

    cfg = PhaseBLossConfig()
    loss, _ = phase_b_loss(round_trip, residual, cross_model_loss, output_scale, cfg)
    loss.backward()

    assert round_trip.grad is not None
    assert cross_model_loss.grad is not None
    # output_scale has zero gradient because 0.3 < max_scale=1.0,
    # so native_preservation is zero (and its derivative at 0.3 is 0)
