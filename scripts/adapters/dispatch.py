"""Adapter dispatch — resolve an alloy to its concrete adapter chain.

This is the Tier 1 entry point: load an alloy, look up the family adapter for
its source.architecture, and produce an ordered list of AdapterCall records
describing exactly which adapter method handles each stage. No torch import,
no model load, no GPU touch — pure dispatch resolution.

The reproducibility test for the published continuum-ai/* catalog uses this
function as its first gate. Every published alloy MUST resolve to a non-empty
adapter chain without error. Alloys that don't resolve are the ones that
expose missing adapters or missing stage handlers — those are the gaps the
plugin sprint exists to close.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Union

from .base import AdapterCall, FamilyAdapter
from .registry import resolve_family_adapter


class DispatchError(ValueError):
    """Raised when an alloy cannot be resolved to an adapter chain.

    Caught by the reproducibility test as a structured failure mode rather
    than an opaque KeyError or AttributeError. The message names the alloy,
    the failing stage (if any), and the missing piece (architecture or
    stage type).
    """


def _load_alloy_dict(alloy: Union[str, Path, dict]) -> dict:
    """Accept a path, an open file's contents, or an already-parsed dict."""
    if isinstance(alloy, dict):
        return alloy
    path = Path(alloy)
    if not path.exists():
        raise DispatchError(f"Alloy file does not exist: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise DispatchError(f"Alloy {path} is not valid JSON: {e}") from e


def resolve_adapter_chain(alloy: Union[str, Path, dict]) -> list[AdapterCall]:
    """Resolve an alloy to its concrete adapter chain.

    Args:
        alloy: Either a path to a .alloy.json file, a Path object, or an
               already-parsed dict.

    Returns:
        A list of AdapterCall records, one per stage in alloy.stages, in
        order. Each AdapterCall names the family adapter, the method on it
        that handles the stage type, the stage's params, and the stage's
        position in the alloy's stage list.

    Raises:
        DispatchError: if the alloy lacks source.architecture, or the
            architecture isn't registered, or any stage type doesn't have
            a handler method on the resolved adapter.

    Tier 1: this function does not load any model, does not import torch,
    and does not touch the GPU. It is safe to call on any platform with
    only the standard library installed.
    """
    data = _load_alloy_dict(alloy)
    name = data.get("name", "<unnamed>")

    source = data.get("source")
    if not isinstance(source, dict):
        raise DispatchError(f"Alloy '{name}' has no source dict (got {type(source).__name__})")

    architecture = source.get("architecture")
    if not architecture:
        raise DispatchError(
            f"Alloy '{name}' has no source.architecture field — cannot resolve "
            f"a family adapter without it. baseModel='{source.get('baseModel')}'."
        )

    try:
        family_adapter = resolve_family_adapter(architecture)
    except KeyError as e:
        # KeyError → DispatchError so the test catches a single error type.
        raise DispatchError(str(e)) from e

    stages = data.get("stages", [])
    if not isinstance(stages, list):
        raise DispatchError(
            f"Alloy '{name}' has stages that aren't a list (got {type(stages).__name__})"
        )
    if not stages:
        raise DispatchError(f"Alloy '{name}' has zero stages — nothing to dispatch")

    chain: list[AdapterCall] = []
    for i, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise DispatchError(
                f"Alloy '{name}' stages[{i}] is not a dict (got {type(stage).__name__})"
            )
        stage_type = stage.get("type")
        if not stage_type:
            raise DispatchError(f"Alloy '{name}' stages[{i}] has no 'type' field")

        method_name = FamilyAdapter.STAGE_METHOD_MAP.get(stage_type)
        if method_name is None:
            raise DispatchError(
                f"Alloy '{name}' stages[{i}] has unknown stage type '{stage_type}'. "
                f"Known stage types: {sorted(FamilyAdapter.STAGE_METHOD_MAP.keys())}. "
                f"To add support: add a method on FamilyAdapter, register the mapping in "
                f"FamilyAdapter.STAGE_METHOD_MAP, then override on the family adapters that handle it."
            )

        # The method must EXIST on the resolved family adapter (every method
        # is defined on FamilyAdapter base, even if it just raises NotImplementedError).
        # We check it's callable here so the dispatch failure is loud and clear,
        # not deferred until execution.
        method = getattr(family_adapter, method_name, None)
        if not callable(method):
            raise DispatchError(
                f"Alloy '{name}' stages[{i}] type '{stage_type}' maps to method "
                f"'{method_name}' but {family_adapter.name} has no such callable. "
                f"This is a base-class bug — every stage in STAGE_METHOD_MAP must have "
                f"a default method on FamilyAdapter."
            )

        # Capture the stage's own params (everything except 'type') as the kwargs
        # the adapter method will receive in Tier 2 execution.
        params = {k: v for k, v in stage.items() if k != "type"}

        chain.append(AdapterCall(
            stage_type=stage_type,
            stage_index=i,
            family_adapter=family_adapter,
            method_name=method_name,
            params=params,
        ))

    return chain


def describe_chain(chain: list[AdapterCall]) -> str:
    """Format a resolved chain as a human-readable dispatch report. Used by
    the reproducibility test for failure messages and by the CLI dry-run."""
    if not chain:
        return "<empty chain>"
    family = type(chain[0].family_adapter).__name__
    lines = [f"Family adapter: {family}"]
    for call in chain:
        lines.append(
            f"  [{call.stage_index}] {call.stage_type:30s} → "
            f"{family}.{call.method_name}({', '.join(sorted(call.params.keys()))})"
        )
    return "\n".join(lines)
