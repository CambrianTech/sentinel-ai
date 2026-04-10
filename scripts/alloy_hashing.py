"""alloy_hashing — single source of truth for the canonical alloy modelHash.

This module is the unified hashing layer that publish_model.py, the
backfill tools (backfill_alloy_from_results.py, derive_alloy_from_parent.py),
and the migration tool (migrate_modelhash_convention.py) all import from.
There is exactly ONE function that turns a per-shard sha256 list into the
alloy's modelHash field across the entire codebase.

== The convention

    modelHash = "sha256:" + sha256(canonical_json([
        {"filename": "model-00001-of-00011.safetensors", "sha256": "..."},
        {"filename": "model-00002-of-00011.safetensors", "sha256": "..."},
        ...
    ]))

where the list is sorted by filename and the canonical JSON uses
separators=(",", ":") and sort_keys=True so the bytes the sha256 walks
are deterministic.

Why this convention vs. the legacy "sha256(concat(shard_bytes))":
    1. Reproducible from HF metadata alone — HuggingFace's API exposes
       LFS sha256 per shard via ?blobs=true, so a verifier can recompute
       modelHash without downloading any of the shards (a 50GB MoE
       artifact gets verified in milliseconds, not hours)
    2. Per-shard attestation is preserved in integrity.fileHashes[] so
       a verifier can also check individual shards
    3. Same security guarantee — any change to any shard's bytes
       changes its LFS sha256, which changes the canonical JSON, which
       changes the composed sha256

== Usage

    from alloy_hashing import compose_model_hash, fetch_shard_hashes_from_hf

    # From local files (publish path)
    shard_hashes = [
        {"filename": s.name, "sha256": hash_local_file(s)}
        for s in sorted(model_dir.glob("*.safetensors"))
    ]
    model_hash = compose_model_hash(shard_hashes)

    # From HF metadata (backfill / migration path — no downloads)
    shard_hashes = fetch_shard_hashes_from_hf(
        repo="continuum-ai/qwen3.5-27b-code-forged",
        extensions=(".safetensors",),
    )
    model_hash = compose_model_hash(shard_hashes)
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from typing import Any


def compose_model_hash(shard_hashes: list[dict[str, Any]]) -> str:
    """Compose a single deterministic modelHash from a per-shard sha256 list.

    The composition is sha256 of the canonical JSON of the per-shard list,
    sorted by filename. Order-independent (the function sorts internally).
    Pure function — no side effects, no IO.

    Args:
        shard_hashes: list of dicts, each with at minimum 'filename' and
                      'sha256' fields. Extra fields (e.g. 'size') are
                      ignored for the hash composition but should be
                      preserved in the alloy's integrity.fileHashes[] for
                      auditing.

    Returns:
        A 'sha256:' prefixed hex string of length 7 + 64 = 71.

    Raises:
        ValueError: if shard_hashes is empty (a model with zero shards
                    is a contract violation, not a thing to silently
                    return a sentinel for).
    """
    if not shard_hashes:
        raise ValueError(
            "compose_model_hash: shard_hashes is empty. A model with zero "
            "safetensors shards is a contract violation — the upstream "
            "step that produced the list should have raised before reaching "
            "this function."
        )
    # Sort the LIST by filename so the composition is order-independent
    # at the input layer (json.dumps's sort_keys only sorts dict keys
    # WITHIN each item, not the items themselves).
    sorted_shards = sorted(shard_hashes, key=lambda s: s["filename"])
    canonical = json.dumps(
        [{"filename": s["filename"], "sha256": s["sha256"]} for s in sorted_shards],
        separators=(",", ":"),
        sort_keys=True,
    )
    h = hashlib.sha256(canonical.encode()).hexdigest()
    return f"sha256:{h}"


def fetch_shard_hashes_from_hf(
    repo: str,
    extensions: tuple[str, ...] = (".safetensors",),
) -> list[dict[str, Any]]:
    """Pull per-shard sha256 from HuggingFace's LFS metadata API.

    No downloads — uses the ?blobs=true query parameter on the model
    metadata endpoint, which returns siblings[].lfs.sha256 for every
    LFS-tracked file in the repo. The 11×5GB shards of a 27B model
    get hashed in seconds because no shard bytes are fetched.

    Args:
        repo: HF repo id (e.g. 'continuum-ai/qwen3-coder-30b-a3b-compacted-19b-256k')
        extensions: tuple of file extensions to include. Defaults to
                    ('.safetensors',). Use ('.gguf',) for GGUF artifacts,
                    ('.safetensors',) for MLX too (MLX writes .safetensors).

    Returns:
        Sorted-by-filename list of {'filename', 'sha256', 'size'} dicts.
        Empty list if the repo has no matching files (caller decides
        whether that's an error).
    """
    url = f"https://huggingface.co/api/models/{repo}?blobs=true"
    with urllib.request.urlopen(url, timeout=30) as resp:
        meta = json.loads(resp.read())
    out: list[dict[str, Any]] = []
    for s in meta.get("siblings", []):
        fn = s.get("rfilename", "")
        if not any(fn.endswith(ext) for ext in extensions):
            continue
        lfs = s.get("lfs") or {}
        sha = lfs.get("sha256")
        size = lfs.get("size")
        if sha:
            out.append({"filename": fn, "sha256": sha, "size": size})
    out.sort(key=lambda d: d["filename"])
    return out


def hash_local_safetensors_dir(model_dir) -> list[dict[str, Any]]:
    """Hash every *.safetensors shard in a local directory and return
    the per-shard list in the same shape as fetch_shard_hashes_from_hf.

    Used by publish_model.py for freshly-forged artifacts where the
    weights are on disk locally and there's no HF repo yet to query.
    """
    from pathlib import Path
    model_dir = Path(model_dir)
    shards = sorted(model_dir.glob("*.safetensors"))
    out: list[dict[str, Any]] = []
    for shard in shards:
        h = hashlib.sha256()
        with open(shard, "rb") as f:
            while True:
                chunk = f.read(1 << 20)  # 1 MB chunks
                if not chunk:
                    break
                h.update(chunk)
        out.append({
            "filename": shard.name,
            "sha256": h.hexdigest(),
            "size": shard.stat().st_size,
        })
    return out
