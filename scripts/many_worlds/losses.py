"""losses.py — the two-term Many-Worlds training objective.

Implements the Phase A (substrate) and Phase B (per-model adapter)
loss functions for training the Many-Worlds framework. Per Kash's
review (§VI.9 of MANY-WORLDS-ABSTRACT.md):

> Round-trip fidelity must be in the loss function, not just contrastive
> alignment. Contrastive learning produces a substrate that distinguishes
> inputs from each other, not necessarily one that supports task transfer.
> The loss must include both terms.

So both Phase A and Phase B use two-term objectives:

**Phase A (substrate training)** — trains the substrate's bases and
temperature against the frozen population:

  L_A = α · L_contrastive_alignment + β · L_round_trip_reconstruction

  - L_contrastive_alignment: semantically equivalent residuals from
    different base models (same input, different member) should land
    at nearby substrate coordinates; semantically different residuals
    (different inputs, same member) should land at distant coordinates.
    Uses InfoNCE-style softmax over a batch.

  - L_round_trip_reconstruction: residual → Project → substrate.write
    → substrate.read → Read should reconstruct the original residual
    with minimal loss. MSE between original and reconstructed residual.
    This is the term Kash insisted on — without it, the substrate is
    just a similarity space, not a task-transfer medium.

**Phase B (per-model adapter training)** — trains each base model's
Project + Read adapter against a frozen substrate and (optionally)
frozen other adapters:

  L_B = γ · L_round_trip_fidelity + δ · L_cross_model_transfer + ε · L_native_preservation

  - L_round_trip_fidelity: same as Phase A's round-trip but computed
    per-adapter against the now-frozen substrate. Each adapter must
    learn to Project and Read its own model's residuals losslessly.

  - L_cross_model_transfer: Project from model A, Read into model B,
    then continue model B's inference for N tokens; the resulting
    continuation should be coherent with model A's intended thought.
    Measured by perplexity of the continuation under a reference
    model, OR by task-specific metrics on held-out downstream tasks.
    This is the term that forces the adapter to produce SUBSTRATE
    FIELDS THAT OTHER MODELS CAN CONSUME, not just ones that round-trip
    within the same model.

  - L_native_preservation: with the adapter DISABLED, the base model's
    behavior must be bit-identical to its frozen original. This is
    a penalty term that activates only when the adapter has drifted
    the base model's behavior (which shouldn't happen since adapters
    are additive, but the loss catches any drift as a safety check).
    For v0, implemented as a regularization on the adapter's
    output_scale parameter to keep it from growing too large too fast.

All losses are computed per-batch and averaged. The training loop in
scripts/stages/many_worlds_stages.py calls these loss functions and
backpropagates.

Loss weights (α, β, γ, δ, ε) are hyperparameters that need tuning
during the v0 validation. Default values below are reasonable
starting points but should be ablated in §VII.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import Tensor


@dataclass
class PhaseALossConfig:
    """Weights for the Phase A (substrate training) two-term loss."""

    alpha_contrastive: float = 1.0   # L_contrastive_alignment weight
    beta_round_trip: float = 1.0     # L_round_trip_reconstruction weight
    contrastive_temperature: float = 0.07  # InfoNCE temperature
    round_trip_loss_type: str = "mse"  # "mse" | "cosine" | "l1"


@dataclass
class PhaseBLossConfig:
    """Weights for the Phase B (adapter training) three-term loss."""

    gamma_round_trip: float = 1.0          # L_round_trip_fidelity weight
    delta_cross_model: float = 1.0         # L_cross_model_transfer weight
    epsilon_native_preservation: float = 0.1  # L_native_preservation weight
    transfer_rollout_length: int = 50      # tokens to continue during cross-model eval


def contrastive_alignment_loss(
    projections: dict[str, "Tensor"],
    temperature: float = 0.07,
) -> "Tensor":
    """InfoNCE-style contrastive alignment over a population.

    Args:
        projections: {member_name: (batch, substrate_dim)} — each
            population member's projected representation of the SAME
            batch of inputs. The keys are member names; the values
            are per-sequence pooled substrate coordinates (one vector
            per input sequence, averaged across tokens).
        temperature: InfoNCE temperature (default 0.07 per SimCLR).

    Returns:
        scalar loss tensor

    Math:
        For every pair (member_A, member_B) in the population, the
        positive pair is (A[i], B[i]) — same input, different member.
        Negatives are (A[i], B[j]) for i != j. Loss is symmetric
        InfoNCE over these pairs.

        Intuition: if two models see the same input, their substrate
        projections should land at nearby coordinates (positive pair).
        If they see different inputs, the projections should be
        distinguishable (negatives). Training this pulls the substrate
        coordinate system into alignment with semantic content rather
        than model-specific quirks.

    Returns the mean loss across all (A, B) pairs in the population.
    If there are N members, this is N*(N-1)/2 pair losses averaged.
    """
    import torch
    import torch.nn.functional as F

    members = sorted(projections.keys())
    if len(members) < 2:
        # Single-member populations have no contrastive signal.
        # Return a zero loss — this is a valid case for edge-case
        # testing but shouldn't happen in real training.
        return torch.tensor(0.0, device=next(iter(projections.values())).device)

    total_loss = torch.tensor(0.0, device=projections[members[0]].device)
    pair_count = 0

    for i, a_name in enumerate(members):
        for b_name in members[i + 1:]:
            a = F.normalize(projections[a_name], dim=-1)  # (B, d)
            b = F.normalize(projections[b_name], dim=-1)  # (B, d)

            # Similarity matrix: (B, B) where entry [i,j] is the
            # cosine similarity between a[i] and b[j]. Positives are
            # on the diagonal (same input index, different member).
            sim = a @ b.t() / temperature  # (B, B)

            # Symmetric InfoNCE: row-wise (a→b) + column-wise (b→a)
            labels = torch.arange(sim.shape[0], device=sim.device)
            loss_a_to_b = F.cross_entropy(sim, labels)
            loss_b_to_a = F.cross_entropy(sim.t(), labels)
            pair_loss = 0.5 * (loss_a_to_b + loss_b_to_a)

            total_loss = total_loss + pair_loss
            pair_count += 1

    return total_loss / max(pair_count, 1)


def round_trip_reconstruction_loss(
    original_residual: "Tensor",
    reconstructed_residual: "Tensor",
    loss_type: str = "mse",
) -> "Tensor":
    """Round-trip reconstruction loss: Project → write → read → Read.

    Args:
        original_residual: (batch, seq, d) — the input residual at the
            target layer before Project
        reconstructed_residual: (batch, seq, d) — the output of the
            full round-trip: Project → substrate.write → substrate.read
            → Read on the same adapter and same substrate
        loss_type: "mse" | "cosine" | "l1"

    Returns:
        scalar loss

    This is the Kash-insisted term: without it, the substrate learns
    to distinguish inputs (via contrastive) but doesn't necessarily
    preserve the information downstream layers need to consume. MSE
    reconstruction forces the round-trip to be approximately the
    identity, which means the substrate + adapters together learn a
    lossless-enough representation of the model's cognitive state.

    For Phase A this is computed across all members (one per member
    per batch) and averaged. For Phase B this is computed per-adapter
    during that adapter's training step.
    """
    import torch
    import torch.nn.functional as F

    if loss_type == "mse":
        return F.mse_loss(reconstructed_residual, original_residual)
    elif loss_type == "cosine":
        # 1 - cosine similarity, averaged per token
        sim = F.cosine_similarity(reconstructed_residual, original_residual, dim=-1)
        return (1.0 - sim).mean()
    elif loss_type == "l1":
        return F.l1_loss(reconstructed_residual, original_residual)
    else:
        raise ValueError(f"unknown loss_type: {loss_type!r}")


def native_preservation_loss(
    output_scale: "Tensor",
    max_scale: float = 1.0,
) -> "Tensor":
    """Regularization penalty on the adapter's output_scale parameter.

    This is the v0 implementation of the native-preservation constraint.
    A more rigorous version would check that the base model's behavior
    is bit-identical with the adapter disabled, but that's expensive
    to compute per-step. Instead, we penalize the adapter's output_scale
    from growing too large — keeping the adapter's contribution small
    ensures the base model's behavior is approximately preserved.

    Args:
        output_scale: the adapter's learned output_scale parameter
            (scalar tensor)
        max_scale: maximum scale before penalty kicks in. Default 1.0
            means "the adapter can contribute up to ~unit-scale delta
            to the residual stream without penalty; any larger and
            the loss grows quadratically."

    Returns:
        scalar regularization loss
    """
    import torch

    excess = torch.clamp(output_scale.abs() - max_scale, min=0.0)
    return excess ** 2


def phase_a_loss(
    projections: dict[str, "Tensor"],
    original_residuals: dict[str, "Tensor"],
    reconstructed_residuals: dict[str, "Tensor"],
    config: PhaseALossConfig,
) -> tuple["Tensor", dict[str, float]]:
    """Full Phase A (substrate training) loss.

    Args:
        projections: {member_name: (batch, substrate_dim)} pooled
            per-sequence projections, for the contrastive term
        original_residuals: {member_name: (batch, seq, d_member)}
            original per-token residuals from each member
        reconstructed_residuals: {member_name: (batch, seq, d_member)}
            round-trip reconstructions from each member's adapter
        config: PhaseALossConfig with the term weights

    Returns:
        total_loss: scalar tensor
        metrics: dict of per-term float values for logging
    """
    import torch

    contrastive = contrastive_alignment_loss(
        projections, temperature=config.contrastive_temperature
    )

    # Round-trip: averaged across all members
    per_member_rt = []
    for name in original_residuals:
        orig = original_residuals[name]
        recon = reconstructed_residuals[name]
        rt = round_trip_reconstruction_loss(orig, recon, loss_type=config.round_trip_loss_type)
        per_member_rt.append(rt)
    round_trip = torch.stack(per_member_rt).mean() if per_member_rt else torch.tensor(0.0)

    total = (
        config.alpha_contrastive * contrastive
        + config.beta_round_trip * round_trip
    )

    metrics = {
        "phase_a/contrastive": float(contrastive.detach().item()),
        "phase_a/round_trip": float(round_trip.detach().item()),
        "phase_a/total": float(total.detach().item()),
    }
    return total, metrics


def phase_b_loss(
    round_trip_residual: "Tensor",
    original_residual: "Tensor",
    cross_model_loss: "Tensor",
    output_scale: "Tensor",
    config: PhaseBLossConfig,
) -> tuple["Tensor", dict[str, float]]:
    """Full Phase B (per-adapter training) loss.

    Args:
        round_trip_residual: (batch, seq, d) — the round-trip output
            for THIS member's adapter (Project → write → read → Read
            on its own substrate coordinates)
        original_residual: (batch, seq, d) — the input residual
        cross_model_loss: scalar tensor — pre-computed cross-model
            transfer loss for this adapter (Project this member's
            residual, Read into a peer member, measure peer's output
            quality). Cross-model loss is computed by the training
            loop which has access to peer models; this function just
            takes the scalar.
        output_scale: the adapter's output_scale parameter for the
            native-preservation regularizer
        config: PhaseBLossConfig with term weights

    Returns:
        total_loss, metrics_dict
    """
    import torch

    round_trip = round_trip_reconstruction_loss(
        original_residual, round_trip_residual
    )
    native_reg = native_preservation_loss(output_scale)

    total = (
        config.gamma_round_trip * round_trip
        + config.delta_cross_model * cross_model_loss
        + config.epsilon_native_preservation * native_reg
    )

    metrics = {
        "phase_b/round_trip": float(round_trip.detach().item()),
        "phase_b/cross_model": float(cross_model_loss.detach().item()),
        "phase_b/native_reg": float(native_reg.detach().item()),
        "phase_b/total": float(total.detach().item()),
    }
    return total, metrics
