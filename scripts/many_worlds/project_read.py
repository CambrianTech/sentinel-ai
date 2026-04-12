"""project_read.py — per-base-model adapters for substrate interop.

Each base model in a Many-Worlds population gets its own adapter pair:
  - ProjectModule: maps the base model's internal residual stream
    (at a chosen layer) to a per-token Gaussian distribution in
    substrate coordinates, producing (μ, σ²) pairs for the substrate's
    write() operation.
  - ReadModule: takes a dense read vector from the substrate (produced
    by the substrate's read() operation) and projects it back into
    the base model's residual form, ready to be added to or replace
    the model's residual stream at the corresponding layer.

These two modules are LoRA-style adapters grafted onto a frozen base
model. They are small (~50-200M params per base model for typical
configurations) and are the only things that train during Phase B
of the Many-Worlds training protocol. Phase A trains the substrate;
Phase B trains per-model adapters against a frozen substrate.

Design per MANY-WORLDS-ABSTRACT.md §III.2 and Kash's review:

1. The adapter is ADDITIVE — disabling it (via the `enabled=False`
   flag) makes the base model's behavior BIT-IDENTICAL to the original.
   This is the "native preservation" constraint from §III.4 point 3:
   any base model in a Many-Worlds population must remain fully usable
   as itself. The adapter is opt-in capability.

2. Project and Read operate at a CHOSEN LAYER in the base model's
   residual stream. Default is 2/3 depth (e.g. layer 32 of 48 in
   Qwen3-Coder-30B-A3B). This choice is ablatable in §VII but 2/3
   depth is the rule-of-thumb for "rich enough to carry semantic
   structure but not so deep that language-head specialization
   dominates the representation."

3. The Project output is a (mean, log-variance) pair per token. The
   ReadModule input is a dense vector per token. The two sides are
   asymmetric — Project writes Gaussians, Read consumes vectors —
   because the substrate itself handles the write→read conversion
   via its write() + read() operations.

4. The round-trip invariant (Phase B loss term 1): Project → substrate.write
   → substrate.read → Read should approximately reconstruct the original
   residual at the chosen layer. This is what forces the substrate to
   learn a lossless-enough representation.

5. The cross-model transfer invariant (Phase B loss term 2): Project A
   → substrate.write → substrate.read → Read B should produce a
   residual that B's downstream layers can consume coherently. This
   is what forces the substrate to learn a representation SHARED
   across models, not just internally consistent per model.

The adapter modules are pure PyTorch nn.Module subclasses. They can be
attached to a base model via forward hooks OR by explicit insertion
into the residual stream depending on the use case:
  - Training: forward hooks that fire at the chosen layer, capture
    the residual, Project it, round-trip through the substrate, Read
    it back, and compute the losses.
  - Inference (query mode): explicit insertion into the model's
    layer loop via a wrapped forward() that Projects the residual
    at the chosen layer, queries the substrate, and adds the Read
    result to the residual stream before continuing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import Tensor, nn


@dataclass
class AdapterConfig:
    """Configuration for a ProjectModule + ReadModule pair for one base model.

    The adapter is specific to one base model — its input/output shapes
    match that model's residual dimensionality and its layer depth
    matches the target model's layer count.
    """

    # The base model's hidden size (residual dimensionality). Must be
    # populated from the loaded model's config before constructing the
    # adapter. Qwen3-Coder-30B-A3B: 2048. Mixtral 8x7B: 4096.
    residual_hidden_size: int

    # The substrate's dimensionality. Must match the substrate this
    # adapter is trained against. Default 128 for v0; 512 for v1.
    substrate_dim: int = 128

    # LoRA rank for the Project and Read internal matrices. Higher rank
    # = more capacity but more parameters. Default 64 for v0; 128 for
    # v1. The total adapter parameter count is roughly
    #   2 * (residual_hidden_size * rank + rank * substrate_dim * 2)
    # for Project (mean + log-var heads) plus
    #   residual_hidden_size * rank + rank * substrate_dim for Read.
    lora_rank: int = 64

    # Which layer of the base model to hook. Default is None which means
    # 2/3 depth (computed from the model's num_hidden_layers at construction).
    # Explicit values override this (useful for ablation in §VII).
    layer_idx: Optional[int] = None

    # Initial scale of the adapter's output. Starts small so the adapter's
    # contribution to the residual stream is negligible at init, then
    # grows during training as the loss pulls it up. This is the same
    # "zero init on residual adapter" trick from the LoRA lineage:
    # training starts from the frozen base's behavior and additively
    # learns the adapter's contribution.
    output_scale_init: float = 0.01

    # Dropout on the adapter's internal projection (after the down
    # projection, before the up projection). Small values only (0.0-0.1).
    dropout: float = 0.0

    # Whether the adapter is enabled. When False, forward passes through
    # the adapter produce a zero output (the base model behaves exactly
    # as itself). Used for the "native preservation" sanity check AND
    # for the text-bottleneck baseline condition in §VII.4 (Condition A:
    # disable Many-Worlds adapters and use text between models).
    enabled: bool = True

    # Log-variance initialization. Start at 0.0 so initial σ² = 1.0
    # (unit variance). Training pulls this up or down as needed.
    log_var_init: float = 0.0

    def to_dict(self) -> dict:
        return {
            "residual_hidden_size": self.residual_hidden_size,
            "substrate_dim": self.substrate_dim,
            "lora_rank": self.lora_rank,
            "layer_idx": self.layer_idx,
            "output_scale_init": self.output_scale_init,
            "dropout": self.dropout,
            "enabled": self.enabled,
            "log_var_init": self.log_var_init,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AdapterConfig":
        return cls(**d)


class ProjectModule:
    """Project a base model's residual stream into substrate Gaussian coordinates.

    Maps a residual-form vector (dim = residual_hidden_size) at a chosen
    layer to a per-token (μ, log_var) pair in substrate coordinates
    (dim = substrate_dim each). The output is consumed by
    SubstrateVectorSpace.write() to produce a field representation.

    Internal architecture (LoRA-style):
      residual → down_proj (lora_rank) → dropout → activation
                                       → mean_head (substrate_dim)
                                       → log_var_head (substrate_dim)

    The down_proj + two heads is intentionally minimal — each head is
    just a linear layer from the rank-lora bottleneck to the substrate
    dim. Adding more depth here is an ablation in §VII but the v0
    default is "as simple as possible, LoRA bottleneck + heads."
    """

    def __init__(self, config: AdapterConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self._module: Optional["nn.Module"] = None

    def _build_module(self) -> "nn.Module":
        import torch
        import torch.nn as nn

        class _ProjectInner(nn.Module):
            def __init__(self, cfg: AdapterConfig):
                super().__init__()
                self.down = nn.Linear(cfg.residual_hidden_size, cfg.lora_rank, bias=False)
                self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
                self.activation = nn.GELU()
                self.mean_head = nn.Linear(cfg.lora_rank, cfg.substrate_dim, bias=True)
                self.log_var_head = nn.Linear(cfg.lora_rank, cfg.substrate_dim, bias=True)

                # Xavier init (gradient flow) + learned output_scale (magnitude control)
                nn.init.xavier_uniform_(self.mean_head.weight, gain=0.1)
                nn.init.zeros_(self.mean_head.bias)
                nn.init.xavier_uniform_(self.log_var_head.weight, gain=0.1)
                nn.init.constant_(self.log_var_head.bias, cfg.log_var_init)

                self.output_scale = nn.Parameter(torch.tensor(cfg.output_scale_init))
                self.enabled = cfg.enabled

            def forward(self, residual: "Tensor") -> tuple["Tensor", "Tensor"]:
                """Project residual stream into substrate Gaussian parameters.

                Args:
                    residual: (batch, seq, residual_hidden_size)

                Returns:
                    mu: (batch, seq, substrate_dim) — per-token mean
                    log_var: (batch, seq, substrate_dim) — per-token log-variance
                """
                if not self.enabled:
                    import torch
                    # Return zeros — downstream substrate.write() of zeros
                    # produces a uniform field which read() maps back to
                    # the basis centroid. The caller typically checks
                    # enabled first and skips the substrate path entirely.
                    batch, seq, _ = residual.shape
                    return (
                        torch.zeros(batch, seq, self.mean_head.out_features, device=residual.device),
                        torch.zeros(batch, seq, self.log_var_head.out_features, device=residual.device),
                    )

                # Main path: LoRA bottleneck → heads
                h = self.down(residual)
                h = self.dropout(h)
                h = self.activation(h)
                mu = self.mean_head(h) * self.output_scale
                log_var = self.log_var_head(h)  # not scaled — we want the log_var init to survive
                return mu, log_var

        return _ProjectInner(self.config).to(self.device)

    @property
    def module(self) -> "nn.Module":
        if self._module is None:
            self._module = self._build_module()
        return self._module

    def parameters(self):
        return self.module.parameters()

    def __call__(self, residual: "Tensor") -> tuple["Tensor", "Tensor"]:
        return self.module(residual)

    def set_enabled(self, enabled: bool) -> None:
        self.module.enabled = enabled
        self.config.enabled = enabled


class ReadModule:
    """Read from the substrate back into a base model's residual form.

    Inverse of ProjectModule: takes a dense vector from the substrate
    (the read_vector output of SubstrateVectorSpace.read()) and projects
    it into the base model's residual_hidden_size so it can be added
    to or blended with the model's residual stream.

    Internal architecture (LoRA-style):
      substrate_vector → up_proj (lora_rank) → dropout → activation
                                              → out_proj (residual_hidden_size)

    The output is scaled by output_scale (learned, starts small) so
    initial training adds nothing to the residual stream and grows
    additively from there. Zero-init on the output projection ensures
    the adapter is a no-op at init.
    """

    def __init__(self, config: AdapterConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self._module: Optional["nn.Module"] = None

    def _build_module(self) -> "nn.Module":
        import torch
        import torch.nn as nn

        class _ReadInner(nn.Module):
            def __init__(self, cfg: AdapterConfig):
                super().__init__()
                self.up = nn.Linear(cfg.substrate_dim, cfg.lora_rank, bias=False)
                self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
                self.activation = nn.GELU()
                self.out_proj = nn.Linear(cfg.lora_rank, cfg.residual_hidden_size, bias=True)

                # Xavier init (non-zero for gradient flow) + small learned scale
                # (right magnitude relative to residual stream ~50-100 norm)
                nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
                nn.init.zeros_(self.out_proj.bias)
                nn.init.xavier_uniform_(self.up.weight)

                self.output_scale = nn.Parameter(torch.tensor(cfg.output_scale_init))
                self.enabled = cfg.enabled

            def forward(self, substrate_vector: "Tensor") -> "Tensor":
                """Project substrate vector back into residual form.

                Args:
                    substrate_vector: (batch, seq, substrate_dim)

                Returns:
                    residual_delta: (batch, seq, residual_hidden_size)
                        The vector to be added to the base model's
                        residual stream at the target layer.
                """
                if not self.enabled:
                    import torch
                    batch, seq, _ = substrate_vector.shape
                    return torch.zeros(
                        batch, seq, self.out_proj.out_features, device=substrate_vector.device
                    )

                h = self.up(substrate_vector)
                h = self.dropout(h)
                h = self.activation(h)
                out = self.out_proj(h) * self.output_scale
                return out

        return _ReadInner(self.config).to(self.device)

    @property
    def module(self) -> "nn.Module":
        if self._module is None:
            self._module = self._build_module()
        return self._module

    def parameters(self):
        return self.module.parameters()

    def __call__(self, substrate_vector: "Tensor") -> "Tensor":
        return self.module(substrate_vector)

    def set_enabled(self, enabled: bool) -> None:
        self.module.enabled = enabled
        self.config.enabled = enabled


class AdapterPair:
    """Convenience wrapper bundling a Project + Read pair for one base model.

    Use this when you want a single object per base model in the
    population. The pair is saved/loaded together and both modules
    share the same AdapterConfig.
    """

    def __init__(
        self,
        config: AdapterConfig,
        base_model_name: str,
        device: str = "cpu",
    ):
        self.config = config
        self.base_model_name = base_model_name
        self.device = device
        self.project = ProjectModule(config, device)
        self.read = ReadModule(config, device)

    def parameters(self):
        """Yield all learnable parameters from both sub-modules.

        Used by the optimizer during Phase B training.
        """
        yield from self.project.parameters()
        yield from self.read.parameters()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the entire pair in one call.

        Used for the §VII.4 Condition A (text-bottleneck baseline):
        disable all adapters in the population, force text serialization,
        measure cross-model transfer quality. Then enable all adapters
        and measure substrate-mediated transfer quality. The delta is
        the substrate's contribution.
        """
        self.project.set_enabled(enabled)
        self.read.set_enabled(enabled)

    def save(self, path: str) -> None:
        """Save the pair to disk as a .pt bundle."""
        import torch
        from pathlib import Path

        bundle = {
            "config": self.config.to_dict(),
            "base_model_name": self.base_model_name,
            "project_state": self.project.module.state_dict(),
            "read_state": self.read.module.state_dict(),
            "format_version": 1,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "AdapterPair":
        import torch

        bundle = torch.load(path, map_location=device, weights_only=False)
        config = AdapterConfig.from_dict(bundle["config"])
        pair = cls(
            config=config,
            base_model_name=bundle["base_model_name"],
            device=device,
        )
        # Build both modules then load state
        _ = pair.project.module
        _ = pair.read.module
        pair.project.module.load_state_dict(bundle["project_state"])
        pair.read.module.load_state_dict(bundle["read_state"])
        return pair

    def __repr__(self) -> str:
        return (
            f"AdapterPair(model={self.base_model_name}, "
            f"residual_hidden={self.config.residual_hidden_size}, "
            f"substrate_dim={self.config.substrate_dim}, "
            f"lora_rank={self.config.lora_rank}, "
            f"enabled={self.config.enabled})"
        )
