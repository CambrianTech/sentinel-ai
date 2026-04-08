"""Family adapters — per-model-architecture implementations of forge concerns.

Each adapter handles ONE model family's tensor layout and module-tree quirks.
The StageExecutors in scripts/stages/ are domain-agnostic dispatchers; they
look up the family adapter for the current model and delegate the work.

Architectural rule: NEVER add an isinstance/architectures branch to a shared
path to make a new model work. New family = new adapter file in this directory,
registered in registry.py. Old adapters stay frozen so older alloys keep
reproducing bit-identically. See ~/.claude/projects/.../memory/feedback_adapters_not_branches.md.
"""

from .base import FamilyAdapter, AdapterCall
from .registry import (
    resolve_family_adapter,
    register_family_adapter,
    registered_architectures,
    AdapterRegistry,
)
from .dispatch import resolve_adapter_chain, DispatchError

# Importing each concrete adapter module triggers its @register_family_adapter
# decorator. Order doesn't matter — registry is keyed by architecture string,
# not by import order. NEW family = new module here.
from . import qwen3_dense  # noqa: F401  — registers Qwen3DenseAdapter for qwen3_5

__all__ = [
    "FamilyAdapter",
    "AdapterCall",
    "AdapterRegistry",
    "DispatchError",
    "resolve_adapter_chain",
    "resolve_family_adapter",
    "register_family_adapter",
    "registered_architectures",
]
