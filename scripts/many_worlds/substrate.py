"""substrate.py — the Many-Worlds shared continuous coordinate space.

This module defines the SubstrateVectorSpace class, which represents the
learned coordinate system that all per-base-model adapters project into
and read from. The substrate is the shared medium through which cognition
crosses between independently-trained LLMs.

Design per MANY-WORLDS-ABSTRACT.md §III.1 and Kash's review (§VI.9 of the
abstract document):

- The substrate is a REAL-VALUED vector space of learned dimensionality d
  (default d=128 for v0 tiny-scale validation; d=512 for v1 production;
  d=1024+ for larger populations and more expressive needs).
- Each token's representation in the substrate is parameterized as a
  DIAGONAL GAUSSIAN DISTRIBUTION over coordinates: N(μ, diag(σ²)).
  This is Kash's correction to the "metaphorical Gaussian" framing —
  the substrate uses literal Gaussian parameterization per token, not
  just "smooth and continuous" as a vague hand-wave. Each token's
  projection into the substrate is a (mean, log-variance) pair.
- The substrate itself is small (~100-500 MB for d=1024 with a few
  hundred basis vectors), distributable alongside the per-model
  adapters, and STABLE once trained. New base models joining the
  population train their adapter against the existing substrate and
  accept the lossy-join cost rather than forcing substrate retraining.
  This preserves the flywheel: adding a new family is one adapter's
  worth of cost, not a population-wide retraining.

The substrate is NOT a model. It is a coordinate system — like a
tokenizer for continuous semantic fields rather than discrete tokens.
One substrate per population; shared by every base model in that
population.

Key design invariants:

1. The substrate's forward pass is simple arithmetic + a learned
   projection head. No attention, no MLPs inside the substrate
   itself — the substrate is a COORDINATE SYSTEM, not a model.
   All the cognitive work happens in the base models and their
   per-model adapters.

2. The Gaussian parameterization is per-token, not per-sequence.
   A sequence of N tokens produces N independent Gaussian
   distributions over substrate coordinates, one per token.

3. The substrate supports "field queries" — reading a Gaussian
   region back by specifying (μ, σ²) and getting a weighted
   combination of the substrate's learned basis vectors that
   best matches that region.

4. All operations are differentiable end-to-end so the substrate
   can be trained via backpropagation against the contrastive +
   round-trip loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import Tensor, nn


@dataclass
class SubstrateConfig:
    """Configuration for a SubstrateVectorSpace.

    This dataclass is used to construct the substrate and is also
    serialized alongside the substrate weights so that loading code
    can reconstruct the exact geometry. Fields match the language
    used in MANY-WORLDS-ABSTRACT.md §V.2 and §VII.2.
    """

    # The substrate's dimensionality. d=128 for v0 tiny validation,
    # d=512 for v1 production, d=1024+ for larger populations. Should
    # be large enough to preserve the semantic structure the base
    # models encode, but small enough that per-model adapters stay
    # small and training converges fast. Ablate in §VII.
    dimensionality: int = 128

    # Number of learned basis vectors in the substrate. Each basis
    # vector is a d-dimensional anchor point; Gaussian-field reads
    # are computed as weighted combinations over these bases. More
    # bases = finer-grained substrate representation but more params.
    # Rule of thumb: ~10-50 bases per "concept cluster" the population
    # collectively knows about; start at 1024 for v0 and ablate.
    num_bases: int = 1024

    # Log-variance clamp — the Gaussian σ² can drift to absurd values
    # during training without a clamp. Clamp to [min, max] in log space
    # so σ lives in [e^min, e^max]. Default range gives σ ∈ [0.05, 20]
    # which covers "confident point estimate" to "broad uncertainty"
    # without collapse or divergence.
    log_var_min: float = -6.0  # σ² ≥ e^-6 ≈ 0.0025  (σ ≥ 0.05)
    log_var_max: float = 6.0   # σ² ≤ e^6 ≈ 403      (σ ≤ 20.0)

    # Temperature for the basis-softmax in Gaussian-field reads. Lower
    # temperature makes the read sharper (picks fewer bases); higher
    # makes it smoother (spreads across more bases). Learned as a
    # parameter in the substrate rather than hardcoded, but this is
    # the initialization value.
    read_temperature_init: float = 1.0

    # Initialization strategy for the substrate basis vectors.
    #   "xavier" — Xavier uniform, standard default
    #   "orthogonal" — orthogonal init, good for low-dimensional
    #                  substrates where we want bases to maximally
    #                  span the space from the start
    #   "normal" — N(0, 1/sqrt(d)) init, standard transformer style
    init_strategy: str = "orthogonal"

    # Whether to use weight normalization on the basis vectors. If True,
    # each basis vector has its norm learned separately from its
    # direction, which can stabilize early training.
    use_weight_norm: bool = True

    # Seed for deterministic initialization. Important for reproducibility
    # of the v0 validation — the substrate init must be bit-identical
    # across runs so the three-way comparison (Conditions A/B/C) is fair.
    seed: int = 42

    def to_dict(self) -> dict:
        """Serialize for persistence alongside the substrate weights."""
        return {
            "dimensionality": self.dimensionality,
            "num_bases": self.num_bases,
            "log_var_min": self.log_var_min,
            "log_var_max": self.log_var_max,
            "read_temperature_init": self.read_temperature_init,
            "init_strategy": self.init_strategy,
            "use_weight_norm": self.use_weight_norm,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SubstrateConfig":
        return cls(**d)


class SubstrateVectorSpace:
    """The Many-Worlds shared continuous coordinate space.

    ONE instance per population. Holds:
      - A learned basis matrix B of shape (num_bases, dimensionality)
      - A learned read-temperature scalar (for basis-softmax in reads)
      - Configuration metadata for reconstruction

    Operations:
      - `write(mu, log_var) → field`: given a per-token Gaussian (μ,σ²)
        produced by a base model's Project module, return a dense field
        representation that can be stored in the substrate stream.
      - `read(query_mu, query_log_var, field) → read_vector`: given a
        field (written by write) and a query Gaussian, return the
        substrate's learned interpretation of that region as a dense
        vector suitable for the calling base model's Read module to
        consume.
      - `save(path)` / `load(path)`: persistence for forge pipeline
        handoff and cross-adapter sharing.

    The substrate's forward pass is deliberately simple — it is a
    coordinate system, not a model. All the heavy cognitive work lives
    in the base models and their per-model adapters (see project_read.py
    and framework.py).

    Thread safety: the substrate's weights are read-only once trained;
    inference is thread-safe. During training the substrate is updated
    via backpropagation and should not be read from other threads
    concurrently.
    """

    def __init__(self, config: SubstrateConfig, device: str = "cpu"):
        """Construct a new substrate with freshly initialized weights.

        Use SubstrateVectorSpace.load(path) to reconstruct a previously
        trained substrate from disk. Both paths produce objects that
        can be used for write/read operations immediately.
        """
        self.config = config
        self.device = device
        self._module: Optional["nn.Module"] = None  # lazy-built on first use
        self._is_trained = False  # set True after training or load()

    def _build_module(self) -> "nn.Module":
        """Lazy-construct the underlying torch.nn.Module holding the
        learned parameters. Deferred until first use so that importing
        the package doesn't require torch.

        The module is simple:
          - `bases`: (num_bases, dimensionality) — the learned basis vectors
          - `log_temperature`: scalar — learned temperature for basis softmax

        Nothing else. The Gaussian parameterization of per-token reads
        is computed via arithmetic on the input (μ, σ²) pairs from the
        Project modules; the substrate itself does not store token-level
        state. The substrate is STATELESS across forward passes — it
        only holds the bases.
        """
        import torch
        import torch.nn as nn

        class _SubstrateModule(nn.Module):
            def __init__(self, num_bases: int, d: int, init: str, use_wn: bool, seed: int, temp_init: float):
                super().__init__()
                g = torch.Generator().manual_seed(seed)

                if init == "orthogonal":
                    # Orthogonal init gives us bases that span the substrate
                    # space maximally from the start, which helps early
                    # training stability on small substrates (d=128).
                    W = torch.empty(num_bases, d)
                    nn.init.orthogonal_(W, gain=1.0, generator=g)
                elif init == "xavier":
                    W = torch.empty(num_bases, d)
                    nn.init.xavier_uniform_(W, gain=1.0, generator=g)
                elif init == "normal":
                    W = torch.randn(num_bases, d, generator=g) / (d ** 0.5)
                else:
                    raise ValueError(f"unknown init_strategy: {init!r}")

                self.bases = nn.Parameter(W)
                if use_wn:
                    # Weight normalization: separate norm and direction
                    # parameters. The 'g' parameter (per-basis scalar norm)
                    # is initialized to 1.0 so the initial substrate is
                    # exactly the init values above.
                    nn.utils.weight_norm(self, name="bases")

                # Learned read temperature (scalar). log-parameterized
                # so it stays positive.
                import math
                self.log_temperature = nn.Parameter(torch.tensor(math.log(temp_init), dtype=torch.float32))

            @property
            def temperature(self) -> "Tensor":
                return self.log_temperature.exp()

            def forward(self):
                raise RuntimeError(
                    "SubstrateModule has no generic forward() — use write() "
                    "and read() on the parent SubstrateVectorSpace instead."
                )

        module = _SubstrateModule(
            num_bases=self.config.num_bases,
            d=self.config.dimensionality,
            init=self.config.init_strategy,
            use_wn=self.config.use_weight_norm,
            seed=self.config.seed,
            temp_init=self.config.read_temperature_init,
        ).to(self.device)
        return module

    @property
    def module(self) -> "nn.Module":
        """The underlying torch module. Built lazily on first access."""
        if self._module is None:
            self._module = self._build_module()
        return self._module

    def parameters(self):
        """Expose the substrate's learnable parameters for the optimizer."""
        return self.module.parameters()

    def train(self):
        self.module.train()
        return self

    def eval(self):
        self.module.eval()
        return self

    # ── Core operations: write and read ─────────────────────────────────

    def write(self, mu: "Tensor", log_var: "Tensor") -> "Tensor":
        """Write a per-token Gaussian into the substrate.

        Args:
            mu: (batch, seq, d) — per-token mean in substrate coordinates
            log_var: (batch, seq, d) — per-token log-variance (diagonal)

        Returns:
            field: (batch, seq, num_bases) — per-token soft assignment
                to basis vectors. Each token's row is a probability
                distribution over basis vectors weighted by how much
                the Gaussian (μ, σ²) overlaps each basis's region.

        The field representation is what's actually stored and
        transmitted in the substrate. A Project module produces (μ, σ²)
        pairs from a base model's residual stream; write() converts
        them into basis-space field assignments; read() reverses the
        process to produce residual-space vectors for a different
        base model's Read module.

        The math:
          field[b, s, k] = softmax_k(-0.5 * ||μ[b,s] - B[k]||² / (σ²[b,s] + eps))
                           / temperature

        Where B is the (num_bases, d) basis matrix. This is a
        probabilistic nearest-neighbor soft assignment: each basis k
        gets weighted by how likely it is that the Gaussian at (μ, σ²)
        would "generate" basis B[k]. The result is a probability
        distribution over bases per token.
        """
        import torch

        # Clamp log_var to the configured range so variance stays bounded.
        log_var_clamped = log_var.clamp(
            min=self.config.log_var_min, max=self.config.log_var_max
        )
        var = log_var_clamped.exp()  # (batch, seq, d)

        # Compute squared distance from each μ to each basis vector.
        # mu: (B, S, d)
        # bases: (K, d)
        # diff: (B, S, K, d)
        # Uses broadcasting to avoid materializing the full tensor
        # when possible.
        bases = self.module.bases  # (K, d)
        # (B, S, 1, d) - (1, 1, K, d) = (B, S, K, d)
        diff = mu.unsqueeze(-2) - bases.unsqueeze(0).unsqueeze(0)
        # Sum over d: (B, S, K)
        sq_dist = (diff ** 2).sum(dim=-1)

        # Per-token variance (mean across d for the scalar denominator).
        # Keeping this as a per-token scalar rather than per-dim keeps
        # the write operation simple; per-dim Gaussian distance is an
        # ablation option in §VII but not the v0 default.
        eps = 1e-6
        scalar_var = var.mean(dim=-1, keepdim=True) + eps  # (B, S, 1)

        # Log-likelihood of each basis under the per-token Gaussian,
        # up to a constant that drops out in softmax.
        neg_log_lik = 0.5 * sq_dist / scalar_var  # (B, S, K)

        # Softmax over bases with learned temperature.
        logits = -neg_log_lik / self.module.temperature
        field = torch.softmax(logits, dim=-1)  # (B, S, K)

        return field

    def read(self, query_mu: "Tensor", query_log_var: "Tensor") -> "Tensor":
        """Read from the substrate via a query Gaussian.

        Args:
            query_mu: (batch, seq, d) — per-token query mean (from a
                target base model's Read module, specifying "what
                region of the substrate to pull into my representation")
            query_log_var: (batch, seq, d) — per-token query log-variance

        Returns:
            read_vector: (batch, seq, d) — the substrate's interpretation
                of the queried region, as a dense vector in substrate
                coordinates. This is what gets passed back to the Read
                module, which projects it into the target base model's
                native residual form.

        The read is a Gaussian-weighted combination of the learned
        basis vectors — essentially a soft attention over the bases
        where the attention weights are determined by how much each
        basis overlaps the query region.

        Math: for each token,
          weights[k] = softmax_k(-0.5 * ||query_μ - B[k]||² / query_σ²) / T
          read_vec = Σ_k weights[k] * B[k]

        Note that this is SYMMETRIC to write — reading with the same
        (μ, σ²) you wrote with should recover an approximation of the
        original query point, up to substrate quantization.
        """
        field = self.write(query_mu, query_log_var)  # (B, S, K)
        # Weighted combination of bases: (B, S, K) @ (K, d) = (B, S, d)
        read_vector = field @ self.module.bases
        return read_vector

    # ── Persistence ─────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the substrate to disk as a .pt bundle.

        The bundle contains the module state_dict + the SubstrateConfig
        so that `SubstrateVectorSpace.load(path)` can fully reconstruct
        the object.

        This is the sharing mechanism: after Phase A training in the
        forge pipeline produces a trained substrate, it's saved here,
        and Phase B per-model adapter training loads it as a frozen
        reference. The same saved substrate is also what ships
        alongside each per-model adapter as the shared coordinate
        system that makes the adapters interoperable.
        """
        import torch
        from pathlib import Path

        bundle = {
            "config": self.config.to_dict(),
            "state_dict": self.module.state_dict(),
            "is_trained": self._is_trained,
            "format_version": 1,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SubstrateVectorSpace":
        """Reconstruct a substrate from a previously saved bundle."""
        import torch

        bundle = torch.load(path, map_location=device, weights_only=False)
        config = SubstrateConfig.from_dict(bundle["config"])
        substrate = cls(config=config, device=device)
        # Build the module to initialize the parameter shapes, then
        # load the saved state_dict into it.
        _ = substrate.module
        substrate.module.load_state_dict(bundle["state_dict"])
        substrate._is_trained = bundle.get("is_trained", True)
        return substrate

    def mark_trained(self) -> None:
        """Flag the substrate as having completed its training phase.

        Called by SubstrateTrainExecutor after Phase A converges. Used
        by downstream consumers (adapter training, inference) to verify
        they're operating against a trained substrate and not a freshly
        initialized one.
        """
        self._is_trained = True

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def __repr__(self) -> str:
        return (
            f"SubstrateVectorSpace(d={self.config.dimensionality}, "
            f"num_bases={self.config.num_bases}, "
            f"trained={self._is_trained}, device={self.device})"
        )
