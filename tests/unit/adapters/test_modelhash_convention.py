"""TDD spec for the unified modelHash convention.

Roadmap step 7 from docs/PLUGIN-SPRINT.md: publish_model.py and the
backfill tools (backfill_alloy_from_results.py + derive_alloy_from_parent.py)
currently use DIFFERENT modelHash conventions over the same underlying
bytes. Both attest the published model weights, but a verifier has to
know which convention each alloy uses to reproduce the hash. Step 7
unifies them on the per-shard-list convention because:

  1. It's reproducible from HF metadata alone (LFS sha256 per shard,
     no downloads required even for 50GB MoE artifacts)
  2. It preserves per-shard attestation in integrity.fileHashes[] so
     verifiers who want to check individual shards can
  3. It's the convention all 8 backfilled alloys already use; only
     the 3 freshly-forged ones need a one-shot migration

Written test-first per TDD/TDValidation discipline.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ── Shared hashing module ───────────────────────────────────────────────────


def test_alloy_hashing_module_is_importable():
    """scripts/alloy_hashing.py is the new shared module that exposes the
    canonical hash composition function. Both publish_model.py and the
    backfill tools import it — there is exactly ONE function across the
    codebase that turns per-shard sha256s into the alloy's modelHash."""
    import alloy_hashing
    assert hasattr(alloy_hashing, "compose_model_hash")
    assert hasattr(alloy_hashing, "fetch_shard_hashes_from_hf")


def test_compose_model_hash_signature():
    """compose_model_hash takes a list of shard hash dicts and returns a
    single sha256: prefixed string. The dicts have 'filename' and 'sha256'
    fields at minimum (size is preserved if present)."""
    from alloy_hashing import compose_model_hash
    sig = inspect.signature(compose_model_hash)
    assert "shard_hashes" in sig.parameters


def test_compose_model_hash_is_deterministic():
    """Same input → same output. The function MUST be a pure function
    of the per-shard hash list, sorted by filename. Order of input is
    irrelevant; the function sorts internally."""
    from alloy_hashing import compose_model_hash

    shards_a = [
        {"filename": "model-00001-of-00002.safetensors", "sha256": "a" * 64},
        {"filename": "model-00002-of-00002.safetensors", "sha256": "b" * 64},
    ]
    shards_b = [
        {"filename": "model-00002-of-00002.safetensors", "sha256": "b" * 64},
        {"filename": "model-00001-of-00002.safetensors", "sha256": "a" * 64},
    ]
    h_a = compose_model_hash(shards_a)
    h_b = compose_model_hash(shards_b)
    assert h_a == h_b, "compose_model_hash must be order-independent"
    assert h_a.startswith("sha256:")


def test_compose_model_hash_changes_when_any_shard_changes():
    """Sensitivity property: changing any shard's hash MUST change the
    composed modelHash. This is the security guarantee — no shard can
    be silently swapped."""
    from alloy_hashing import compose_model_hash
    shards_a = [{"filename": "shard.safetensors", "sha256": "a" * 64}]
    shards_b = [{"filename": "shard.safetensors", "sha256": "b" * 64}]
    assert compose_model_hash(shards_a) != compose_model_hash(shards_b)


def test_compose_model_hash_empty_list_raises():
    """Empty input is a contract violation, not a thing to silently
    return a sentinel for. Loud failure."""
    from alloy_hashing import compose_model_hash
    with pytest.raises(ValueError):
        compose_model_hash([])


# ── publish_model.py uses the shared function ──────────────────────────────


def test_publish_model_uses_shared_compose():
    """publish_model.py MUST import compose_model_hash from alloy_hashing
    rather than rolling its own per-shard hashing or the legacy
    concat-and-hash convention. Same source of truth across the
    publish path and the backfill path."""
    src = (REPO_ROOT / "scripts" / "publish_model.py").read_text()
    assert "from alloy_hashing import" in src or "import alloy_hashing" in src, (
        "publish_model.py must import the shared alloy_hashing module so "
        "the modelHash convention is unified across publish and backfill paths."
    )
    assert "compose_model_hash" in src, (
        "publish_model.py must call alloy_hashing.compose_model_hash to "
        "produce the modelHash field."
    )


def test_backfill_uses_shared_compose():
    """The backfill scripts also import the shared module. Same convention
    end-to-end."""
    backfill_src = (REPO_ROOT / "scripts" / "backfill_alloy_from_results.py").read_text()
    assert "alloy_hashing" in backfill_src, (
        "backfill_alloy_from_results.py must use the shared alloy_hashing module."
    )
    derive_src = (REPO_ROOT / "scripts" / "derive_alloy_from_parent.py").read_text()
    assert "alloy_hashing" in derive_src, (
        "derive_alloy_from_parent.py must use the shared alloy_hashing module."
    )


# ── Every cached alloy is migrated ──────────────────────────────────────────


def _cached_alloys():
    """All cached alloys with their parsed contents."""
    cache = REPO_ROOT / "tests" / "reproducibility" / "_cache"
    out = []
    for cf in sorted(cache.glob("continuum-ai_*.json")):
        try:
            data = json.loads(cf.read_text())
        except json.JSONDecodeError:
            continue
        out.append((cf.name, data))
    return out


def test_every_cached_alloy_has_file_hashes():
    """Every cached alloy MUST carry integrity.fileHashes[] post-migration.
    Without per-shard hashes, the verifier cannot recompute modelHash
    from HF metadata alone — they'd have to download every shard. The
    backfilled alloys already have this; the 3 freshly-forged ones need
    the migration to populate it."""
    failures = []
    for name, alloy in _cached_alloys():
        integrity = (alloy.get("results") or {}).get("integrity") or {}
        file_hashes = integrity.get("fileHashes")
        if not file_hashes:
            failures.append(name)
    assert not failures, (
        f"{len(failures)} cached alloys missing integrity.fileHashes:\n  "
        + "\n  ".join(failures[:5])
        + ("\n  ..." if len(failures) > 5 else "")
        + "\n\nRun the migration: python scripts/migrate_modelhash_convention.py"
    )


def test_every_cached_alloy_modelhash_matches_composed():
    """The modelHash field MUST equal compose_model_hash(fileHashes) for
    every cached alloy. This is the gate that ensures the convention is
    actually unified — if any alloy carries a modelHash that doesn't
    reproduce from its own fileHashes, that alloy is using a different
    (or stale) hash convention."""
    from alloy_hashing import compose_model_hash
    failures = []
    for name, alloy in _cached_alloys():
        integrity = (alloy.get("results") or {}).get("integrity") or {}
        recorded = integrity.get("modelHash")
        file_hashes = integrity.get("fileHashes")
        if not file_hashes or not recorded:
            continue  # skipped by the previous test
        composed = compose_model_hash(file_hashes)
        if composed != recorded:
            failures.append(f"{name}: recorded={recorded[:30]}... composed={composed[:30]}...")
    assert not failures, (
        f"{len(failures)} cached alloys have modelHash that does NOT match "
        f"compose_model_hash(fileHashes):\n  " + "\n  ".join(failures[:5])
    )


# ── Migration tool exists ───────────────────────────────────────────────────


def test_migration_script_exists():
    """scripts/migrate_modelhash_convention.py is the one-shot tool that
    walks every cached alloy, populates integrity.fileHashes[] from HF's
    LFS metadata, and re-stamps integrity.modelHash via the shared
    convention. Idempotent — running it twice produces the same output."""
    migration = REPO_ROOT / "scripts" / "migrate_modelhash_convention.py"
    assert migration.exists(), (
        "scripts/migrate_modelhash_convention.py must exist as the one-shot "
        "tool for migrating cached alloys to the unified modelHash convention."
    )
