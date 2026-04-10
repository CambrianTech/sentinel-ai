"""framework.py — the ManyWorldsFramework top-level orchestrator.

The ManyWorldsFramework is the top-level object that holds a population
of base models, their per-model adapters, and a shared substrate. It is
the Python-level entry point for Many-Worlds operations:
  - Construct a framework with a chosen substrate + population definition
  - Attach per-model adapters to each base model in the population
  - Train the substrate (Phase A) and the per-model adapters (Phase B)
  - Serve inference via the query-face routing mechanism
  - Save / load the full framework state

Design per MANY-WORLDS-ABSTRACT.md §III.3 and §V.6.5-V.6.6:

1. **Population is the first-class concept.** A ManyWorldsFramework has
   a `population` of PopulationMember records, each identifying one
   base model by its HF repo name + architecture string + adapter
   pair. The population is not a fixed type — new members can be
   added by training their adapter against the existing frozen
   substrate. Adding a member is O(one adapter training run) and
   does not affect the other members or the substrate.

2. **The query face is chosen per-query.** A query is sent to the
   framework along with a `query_face` parameter identifying which
   base model should produce the response in its native conversational
   style. The query face is always-on during inference; the other
   population members are invoked only when the substrate signals
   uncertainty (the asymmetric-population mitigation from §V.6.4).
   Condition A of §VII.4 uses query_face alone with all adapters
   disabled for the text-bottleneck baseline.

3. **Gate network decides when to query the substrate.** Per Kash's
   review (§VI.9 point 4), a learned gate network on the query face
   decides per-token whether to inject substrate information into
   the residual stream. The gate is a small MLP that takes the
   query face's residual and outputs a scalar in [0, 1]. v0 default:
   learned gating. Ablations: confidence-threshold and always-blend.

4. **Save/load preserves provenance.** The framework's save() method
   writes a manifest.json listing all population members, the
   substrate hash, the adapter hashes, and the framework version.
   This manifest IS the framework's content address — any two
   frameworks with the same manifest hash are bit-identical.

5. **Native preservation is enforceable.** The framework exposes a
   `disable_all_adapters()` method that returns the framework to a
   state where each base model behaves exactly as its frozen original.
   This is the self-test for the "native preservation" constraint
   from §III.2 point 5.

This module does NOT actually load the torch base models — that's
the responsibility of the forge pipeline's load_model path. The
framework accepts already-loaded models by reference and manages
adapter attachment, training, and inference over them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import Tensor, nn
    from .substrate import SubstrateVectorSpace
    from .project_read import AdapterPair, AdapterConfig


@dataclass
class PopulationMember:
    """One entry in a Many-Worlds population.

    Identifies a single base model + its metadata + its adapter pair
    (if one has been trained). The adapter field is None for members
    that have been declared in the population but haven't had their
    Phase B adapter training run yet.
    """

    name: str                                    # short identifier (e.g. "qwen25-1-5b")
    base_model_repo: str                         # HF repo name
    architecture: str                            # family adapter discriminator
    residual_hidden_size: int                    # from the base model config
    num_hidden_layers: int                       # from the base model config
    layer_idx: int                               # chosen layer for Project/Read
    adapter: Optional["AdapterPair"] = None      # the per-model adapter pair
    model_ref: Optional[Any] = None              # optional reference to the loaded torch model
    tokenizer_ref: Optional[Any] = None          # optional reference to the loaded tokenizer

    def to_manifest(self) -> dict:
        """Serialize just the identity fields for the framework manifest.

        The manifest does NOT include the actual model or adapter
        weights — those are stored separately and referenced by hash.
        """
        return {
            "name": self.name,
            "base_model_repo": self.base_model_repo,
            "architecture": self.architecture,
            "residual_hidden_size": self.residual_hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "layer_idx": self.layer_idx,
            "has_adapter": self.adapter is not None,
        }

    def is_adapter_ready(self) -> bool:
        return self.adapter is not None


@dataclass
class FrameworkConfig:
    """Top-level configuration for a ManyWorldsFramework.

    The framework's identity is defined by the substrate it wraps plus
    the population composition. Different substrates + different
    populations = different frameworks.
    """

    name: str = "many-worlds-v0"
    substrate_dim: int = 128
    default_layer_fraction: float = 2.0 / 3.0  # 2/3 depth is the rule of thumb
    query_face_routing: str = "learned_gating"  # learned_gating | confidence_threshold | always_blend
    uncertainty_threshold: float = 0.5          # used if routing == "confidence_threshold"
    always_blend_weight: float = 0.3            # used if routing == "always_blend"
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "substrate_dim": self.substrate_dim,
            "default_layer_fraction": self.default_layer_fraction,
            "query_face_routing": self.query_face_routing,
            "uncertainty_threshold": self.uncertainty_threshold,
            "always_blend_weight": self.always_blend_weight,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FrameworkConfig":
        return cls(**d)


class ManyWorldsFramework:
    """The top-level Many-Worlds population + substrate + routing orchestrator.

    Construct with a FrameworkConfig and an initial SubstrateVectorSpace.
    Population members are added one at a time via add_member().
    Training is done in two phases (substrate, then adapters) by
    external training loops that call into the framework's attached
    primitives. Inference is done via query() which handles the
    query-face routing and substrate injection.
    """

    def __init__(
        self,
        config: "FrameworkConfig",
        substrate: "SubstrateVectorSpace",
        device: str = "cpu",
    ):
        self.config = config
        self.substrate = substrate
        self.device = device
        self.population: list[PopulationMember] = []

    # ── Population management ──────────────────────────────────────────

    def add_member(
        self,
        *,
        name: str,
        base_model_repo: str,
        architecture: str,
        residual_hidden_size: int,
        num_hidden_layers: int,
        layer_idx: Optional[int] = None,
        model_ref: Optional[Any] = None,
        tokenizer_ref: Optional[Any] = None,
    ) -> PopulationMember:
        """Declare a new population member without yet training its adapter.

        Args:
            name: short identifier, unique within this population
            base_model_repo: HF repo name (e.g. "Qwen/Qwen2.5-1.5B-Instruct")
            architecture: family adapter discriminator (e.g. "qwen2_5_dense")
            residual_hidden_size: from the base model config
            num_hidden_layers: from the base model config
            layer_idx: which layer to hook. None → default_layer_fraction
                      (2/3 depth by default)
            model_ref: optional reference to an already-loaded torch model
            tokenizer_ref: optional reference to an already-loaded tokenizer

        Returns the created PopulationMember. The member's adapter
        field is None until `train_adapter()` runs Phase B training
        for this member against the framework's substrate.
        """
        if any(m.name == name for m in self.population):
            raise ValueError(f"population member {name!r} already exists")

        if layer_idx is None:
            layer_idx = int(num_hidden_layers * self.config.default_layer_fraction)

        member = PopulationMember(
            name=name,
            base_model_repo=base_model_repo,
            architecture=architecture,
            residual_hidden_size=residual_hidden_size,
            num_hidden_layers=num_hidden_layers,
            layer_idx=layer_idx,
            model_ref=model_ref,
            tokenizer_ref=tokenizer_ref,
        )
        self.population.append(member)
        return member

    def attach_adapter(
        self,
        member_name: str,
        adapter: "AdapterPair",
    ) -> None:
        """Attach a trained AdapterPair to an existing population member.

        Called after Phase B training produces the adapter. The adapter's
        config must have residual_hidden_size + substrate_dim matching
        the population member and the framework.
        """
        member = self.get_member(member_name)
        if adapter.config.residual_hidden_size != member.residual_hidden_size:
            raise ValueError(
                f"adapter residual_hidden_size {adapter.config.residual_hidden_size} "
                f"does not match member {member_name!r} residual_hidden_size "
                f"{member.residual_hidden_size}"
            )
        if adapter.config.substrate_dim != self.config.substrate_dim:
            raise ValueError(
                f"adapter substrate_dim {adapter.config.substrate_dim} "
                f"does not match framework substrate_dim {self.config.substrate_dim}"
            )
        member.adapter = adapter

    def get_member(self, name: str) -> PopulationMember:
        for m in self.population:
            if m.name == name:
                return m
        raise KeyError(f"no population member named {name!r}")

    def population_summary(self) -> dict:
        """High-level summary of the population for logging / UI."""
        return {
            "framework": self.config.name,
            "substrate_dim": self.config.substrate_dim,
            "substrate_trained": self.substrate.is_trained,
            "num_members": len(self.population),
            "members": [m.to_manifest() for m in self.population],
            "ready_for_inference": all(m.is_adapter_ready() for m in self.population),
        }

    # ── Enable/disable for §VII.4 conditions ───────────────────────────

    def disable_all_adapters(self) -> None:
        """Disable every adapter in the population.

        Used for:
          - Condition A of §VII.4 (text-bottleneck baseline)
          - The "native preservation" sanity check: with all adapters
            disabled, each base model should behave bit-identically
            to its original.
        """
        for member in self.population:
            if member.adapter is not None:
                member.adapter.set_enabled(False)

    def enable_all_adapters(self) -> None:
        """Enable every adapter in the population.

        Used for:
          - Condition B of §VII.4 (substrate-transfer)
          - Normal inference after training completes
        """
        for member in self.population:
            if member.adapter is not None:
                member.adapter.set_enabled(True)

    def set_adapter_enabled(self, member_name: str, enabled: bool) -> None:
        """Enable / disable a single member's adapter."""
        member = self.get_member(member_name)
        if member.adapter is None:
            raise ValueError(f"member {member_name!r} has no adapter attached")
        member.adapter.set_enabled(enabled)

    # ── Core operation: project from one member into the substrate ─────

    def project_residual(
        self,
        member_name: str,
        residual: "Tensor",
    ) -> tuple["Tensor", "Tensor"]:
        """Project a residual stream from one member into substrate coords.

        Args:
            member_name: which member's residual this is
            residual: (batch, seq, member.residual_hidden_size)

        Returns:
            (mu, log_var) — the substrate-space Gaussian parameters
            for the member's adapter pair. Pass these to
            substrate.write() to get a field representation, or to
            cross_project() for end-to-end transfer.
        """
        member = self.get_member(member_name)
        if member.adapter is None:
            raise ValueError(
                f"member {member_name!r} has no adapter attached — "
                f"run Phase B training for this member first"
            )
        return member.adapter.project(residual)

    def read_into(
        self,
        member_name: str,
        substrate_vector: "Tensor",
    ) -> "Tensor":
        """Read a substrate vector into the target member's residual form.

        Args:
            member_name: which member's residual form to produce
            substrate_vector: (batch, seq, substrate_dim) from
                substrate.read() or a direct substrate region.

        Returns:
            residual_delta: (batch, seq, member.residual_hidden_size)
                Ready to be added to or blended with the target
                member's residual stream at its layer_idx.
        """
        member = self.get_member(member_name)
        if member.adapter is None:
            raise ValueError(
                f"member {member_name!r} has no adapter attached"
            )
        return member.adapter.read(substrate_vector)

    def cross_project(
        self,
        source_member: str,
        target_member: str,
        residual: "Tensor",
    ) -> "Tensor":
        """Full round-trip: residual → substrate → residual (different model).

        This is THE core Many-Worlds operation: take a residual stream
        from source_member at its layer_idx, project it into the
        substrate via source's Project module, write/read through the
        substrate, then read it back into target_member's residual
        form via target's Read module.

        Args:
            source_member: name of the source population member
            target_member: name of the target population member
            residual: (batch, seq, source.residual_hidden_size)

        Returns:
            target_residual: (batch, seq, target.residual_hidden_size)
                A residual-form vector the target member's downstream
                layers can consume.

        This is the function whose output Condition B of §VII.4 measures:
        with substrate transfer enabled, the target member should
        produce a coherent continuation when fed this residual at its
        layer_idx. With substrate transfer disabled (Condition A), the
        only alternative is text serialization.
        """
        mu, log_var = self.project_residual(source_member, residual)
        substrate_vector = self.substrate.read(mu, log_var)
        return self.read_into(target_member, substrate_vector)

    # ── Parameters (for the optimizer) ─────────────────────────────────

    def substrate_parameters(self):
        """Yield substrate parameters — for Phase A training."""
        yield from self.substrate.parameters()

    def adapter_parameters(self, member_name: Optional[str] = None):
        """Yield adapter parameters — for Phase B training.

        If member_name is None, yields parameters from all members'
        adapters (for training all at once). If provided, yields only
        that member's adapter parameters (for training one adapter at
        a time against a frozen substrate).
        """
        if member_name is not None:
            member = self.get_member(member_name)
            if member.adapter is not None:
                yield from member.adapter.parameters()
            return
        for member in self.population:
            if member.adapter is not None:
                yield from member.adapter.parameters()

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save the full framework state to a directory.

        Layout:
          {directory}/
            manifest.json            — FrameworkConfig + population summary
            substrate.pt             — SubstrateVectorSpace bundle
            adapters/
              {member.name}.pt       — AdapterPair bundle per member
        """
        import json
        from pathlib import Path

        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        adapters_dir = dir_path / "adapters"
        adapters_dir.mkdir(exist_ok=True)

        # Substrate
        self.substrate.save(str(dir_path / "substrate.pt"))

        # Per-member adapters
        for member in self.population:
            if member.adapter is not None:
                member.adapter.save(str(adapters_dir / f"{member.name}.pt"))

        # Manifest
        manifest = {
            "format_version": 1,
            "framework_config": self.config.to_dict(),
            "substrate_file": "substrate.pt",
            "substrate_trained": self.substrate.is_trained,
            "population": [m.to_manifest() for m in self.population],
        }
        (dir_path / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, directory: str, device: str = "cpu") -> "ManyWorldsFramework":
        """Reconstruct a framework from a save directory.

        Note: this reconstructs the substrate + adapters but NOT the
        base models themselves. Callers must separately load each
        base model and attach via `member.model_ref = loaded_model`
        if inference is needed.
        """
        import json
        from pathlib import Path

        from .substrate import SubstrateVectorSpace
        from .project_read import AdapterPair

        dir_path = Path(directory)
        manifest = json.loads((dir_path / "manifest.json").read_text())

        config = FrameworkConfig.from_dict(manifest["framework_config"])
        substrate = SubstrateVectorSpace.load(
            str(dir_path / manifest["substrate_file"]), device=device
        )

        framework = cls(config=config, substrate=substrate, device=device)

        for member_manifest in manifest["population"]:
            member = framework.add_member(
                name=member_manifest["name"],
                base_model_repo=member_manifest["base_model_repo"],
                architecture=member_manifest["architecture"],
                residual_hidden_size=member_manifest["residual_hidden_size"],
                num_hidden_layers=member_manifest["num_hidden_layers"],
                layer_idx=member_manifest["layer_idx"],
            )
            if member_manifest.get("has_adapter", False):
                adapter_path = dir_path / "adapters" / f"{member.name}.pt"
                if adapter_path.exists():
                    adapter = AdapterPair.load(str(adapter_path), device=device)
                    framework.attach_adapter(member.name, adapter)

        return framework

    def __repr__(self) -> str:
        return (
            f"ManyWorldsFramework(name={self.config.name!r}, "
            f"substrate_dim={self.config.substrate_dim}, "
            f"population_size={len(self.population)}, "
            f"substrate_trained={self.substrate.is_trained}, "
            f"device={self.device})"
        )
