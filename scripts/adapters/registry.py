"""Adapter registry — maps source.architecture strings to FamilyAdapter classes.

Lookup is by the alloy's source.architecture field, which is the canonical
identifier for "which model family forged this." Examples seen in published
continuum-ai/* alloys:

    "qwen3_5"             — Qwen3.5 dense (the legacy Qwen3.5 catalog)
    "qwen3_moe"           — Qwen3MoE (the morning's qwen3-coder-30b-a3b-compacted)
    "olmoe"               — OLMoE (the §4.1.3.4 cross-architecture anchor)
    "qwen2"               — Qwen2/2.5 dense (v2-7b-coder-compensated)
    (future) "mixtral", "phi_moe", "granite_moe", "deepseek_v2", ...

The lookup MUST be exact-match on the architecture string. No fuzzy matching,
no fallback to a default adapter, no isinstance() probing. If an architecture
isn't registered, raise a clear error so the gap is visible.

Backwards compatibility guarantee: once an adapter is registered for an
architecture, it stays registered forever and its behavior never changes.
Methodology improvements arrive as NEW adapters with NEW architecture strings
or NEW alloy field discriminators — never as edits to a frozen adapter.
"""

from __future__ import annotations
from .base import FamilyAdapter


class AdapterRegistry:
    """Architecture string → FamilyAdapter class lookup."""

    def __init__(self) -> None:
        self._adapters: dict[str, type[FamilyAdapter]] = {}

    def register(self, adapter_class: type[FamilyAdapter]) -> type[FamilyAdapter]:
        """Register a FamilyAdapter subclass under every architecture in its
        .architectures tuple. Idempotent — re-registering the same class is
        a no-op; registering a DIFFERENT class against an existing arch
        raises (silent override would let one adapter shadow another)."""
        if not adapter_class.architectures:
            raise ValueError(
                f"{adapter_class.__name__} has no .architectures — set the class attribute "
                f"to a tuple of source.architecture strings this adapter handles."
            )
        for arch in adapter_class.architectures:
            existing = self._adapters.get(arch)
            if existing is not None and existing is not adapter_class:
                raise ValueError(
                    f"Architecture '{arch}' is already registered to {existing.__name__}; "
                    f"cannot also register {adapter_class.__name__}. If this is a methodology "
                    f"upgrade, register under a NEW architecture string — old alloys must keep "
                    f"resolving to the original adapter for reproducibility."
                )
            self._adapters[arch] = adapter_class
        return adapter_class

    def resolve(self, architecture: str) -> FamilyAdapter:
        """Look up the adapter for an architecture string and instantiate it."""
        adapter_class = self._adapters.get(architecture)
        if adapter_class is None:
            registered = sorted(self._adapters.keys())
            raise KeyError(
                f"No FamilyAdapter registered for source.architecture='{architecture}'. "
                f"Registered architectures: {registered}. "
                f"To add support: create scripts/adapters/<family>.py with a FamilyAdapter "
                f"subclass that sets architectures = ('{architecture}',), then import it from "
                f"scripts/adapters/__init__.py to register it. NEVER add a branch to an "
                f"existing adapter to make this work — write a new one."
            )
        return adapter_class()

    def architectures(self) -> list[str]:
        """All registered architecture strings, sorted."""
        return sorted(self._adapters.keys())


# Module-level singleton — the canonical registry the dispatcher reads from.
_REGISTRY = AdapterRegistry()


def register_family_adapter(adapter_class: type[FamilyAdapter]) -> type[FamilyAdapter]:
    """Module-level helper / decorator. Use as @register_family_adapter on a
    FamilyAdapter subclass to register it with the singleton registry."""
    return _REGISTRY.register(adapter_class)


def resolve_family_adapter(architecture: str) -> FamilyAdapter:
    """Module-level helper. Look up and instantiate the adapter for an
    architecture string against the singleton registry."""
    return _REGISTRY.resolve(architecture)


def registered_architectures() -> list[str]:
    """All architecture strings the registry currently knows about."""
    return _REGISTRY.architectures()
