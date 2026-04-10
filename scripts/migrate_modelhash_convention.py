#!/usr/bin/env python3
"""migrate_modelhash_convention.py — one-shot migration to the unified
modelHash convention.

Roadmap step 7 from docs/PLUGIN-SPRINT.md. Walks every cached
forge-alloy under tests/reproducibility/_cache/ and re-stamps:

  integrity.fileHashes  (populated from HF's LFS metadata API; no downloads)
  integrity.modelHash   (composed via the canonical alloy_hashing.compose_model_hash)

Idempotent — running it twice produces the same output. The 8 backfilled
alloys (qwen2.5-{0.5b,1.5b,3b}-general-forged, qwen3.5-27b-code-forged
and its 4 derivatives) already carry fileHashes from the backfill /
derivation tools, so they're no-ops on this migration. The 3 freshly-
forged ones (qwen3-coder-30b-a3b-compacted-19b-256k, olmoe-1b-7b-compacted-5b,
qwen2.5-coder-7b-compacted) get their fileHashes populated for the first
time and their modelHash re-stamped to the canonical convention.

The local cache is the source of truth — this script does NOT push to HF.
After the migration runs and the test passes, run republish_alloy_only.py
to push the corrected alloys to HF with the new modelHash field.

Usage:
    # Dry-run (default — print the diff, no writes)
    python scripts/migrate_modelhash_convention.py

    # Actually rewrite the cached alloys
    python scripts/migrate_modelhash_convention.py --confirm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from alloy_hashing import compose_model_hash, fetch_shard_hashes_from_hf

CACHE_DIR = REPO_ROOT / "tests" / "reproducibility" / "_cache"


def repo_id_from_cache_filename(filename: str) -> str:
    """Cache filename convention: 'continuum-ai_<slug>__<alloy-name>.json'.
    The first underscore is the org/slug separator."""
    stem = filename.split("__", 1)[0]
    return stem.replace("_", "/", 1)


def needs_migration(alloy: dict) -> tuple[bool, str]:
    """Return (yes_no, reason) for whether this alloy needs migrating.

    The contract this enforces:
        - Every alloy with results.integrity MUST have integrity.fileHashes
          populated AND integrity.modelHash equal to compose_model_hash(fileHashes).
        - Recipe-only alloys (no results dict at all) are not migrated.
        - Alloys missing modelHash entirely (the morning's 3 freshly-forged
          ones) need migration to populate BOTH fields.
    """
    results = alloy.get("results")
    if results is None:
        return False, "no results field — recipe alloy, not a results alloy"
    integrity = results.get("integrity") or {}
    file_hashes = integrity.get("fileHashes")
    recorded_hash = integrity.get("modelHash")

    if not file_hashes:
        if recorded_hash:
            return True, "no fileHashes field (modelHash present, will replace)"
        return True, "no fileHashes field AND no modelHash field"

    if not recorded_hash:
        return True, "fileHashes present but no modelHash — recompose"

    composed = compose_model_hash(file_hashes)
    if composed != recorded_hash:
        return True, f"modelHash mismatch (recorded={recorded_hash[:24]}... composed={composed[:24]}...)"
    return False, "already canonical"


def migrate_one(cache_path: Path, confirm: bool) -> dict:
    """Process one cached alloy. Returns a status dict."""
    alloy = json.loads(cache_path.read_text())
    repo = repo_id_from_cache_filename(cache_path.name)

    needs, reason = needs_migration(alloy)
    if not needs:
        return {"file": cache_path.name, "repo": repo, "status": "skipped", "reason": reason}

    print(f"\n  {cache_path.name}")
    print(f"    repo: {repo}")
    print(f"    reason: {reason}")

    # Determine the file extension to scan based on what's in the repo.
    # GGUF artifacts use .gguf; everything else uses .safetensors.
    # Fall back to safetensors if neither is found (the alloy_hashing
    # function returns an empty list and the migration raises loudly).
    extensions: tuple[str, ...]
    if "GGUF" in repo or "gguf" in repo:
        extensions = (".gguf",)
    else:
        extensions = (".safetensors",)

    print(f"    pulling per-shard LFS sha256s ({extensions[0]}) from HF...")
    shard_hashes = fetch_shard_hashes_from_hf(repo, extensions=extensions)
    if not shard_hashes:
        # Try the other extension as a probe — some repos have both
        # .safetensors and .gguf, and the heuristic above might be wrong.
        other = (".safetensors",) if extensions == (".gguf",) else (".gguf",)
        shard_hashes = fetch_shard_hashes_from_hf(repo, extensions=other)
        if shard_hashes:
            extensions = other
    if not shard_hashes:
        return {
            "file": cache_path.name,
            "repo": repo,
            "status": "ERROR",
            "reason": f"no shards found for extensions {extensions} on HF",
        }

    print(f"    shards: {len(shard_hashes)}")
    new_model_hash = compose_model_hash(shard_hashes)
    print(f"    new modelHash: {new_model_hash[:30]}...")
    print(f"    new fileHashes: {len(shard_hashes)} entries")

    integrity = (alloy.get("results") or {}).get("integrity") or {}
    old_model_hash = integrity.get("modelHash", "")

    integrity["fileHashes"] = shard_hashes
    integrity["modelHash"] = new_model_hash
    if "results" not in alloy:
        alloy["results"] = {}
    alloy["results"]["integrity"] = integrity

    if confirm:
        cache_path.write_text(json.dumps(alloy, indent=2) + "\n")
        print(f"    WROTE {cache_path}")
    else:
        print(f"    DRY-RUN: would write {cache_path}")

    return {
        "file": cache_path.name,
        "repo": repo,
        "status": "migrated",
        "old_model_hash": old_model_hash,
        "new_model_hash": new_model_hash,
        "shard_count": len(shard_hashes),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--confirm",
        action="store_true",
        help="Actually rewrite the cached alloys (default is dry-run)",
    )
    args = ap.parse_args()

    cache_files = sorted(CACHE_DIR.glob("continuum-ai_*.json"))
    print(f"Walking {len(cache_files)} cached alloys under {CACHE_DIR}")
    if not args.confirm:
        print("DRY RUN — re-run with --confirm to actually rewrite\n")

    results: list[dict] = []
    for cf in cache_files:
        try:
            r = migrate_one(cf, confirm=args.confirm)
        except Exception as e:
            r = {"file": cf.name, "status": "EXCEPTION", "reason": str(e)}
            print(f"  EXCEPTION on {cf.name}: {e}")
        results.append(r)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    for status, count in sorted(by_status.items()):
        print(f"  {status:<10}  {count}")

    if any(r["status"] in ("ERROR", "EXCEPTION") for r in results):
        sys.exit(1)
    if not args.confirm:
        print("\nDRY RUN complete. Re-run with --confirm to rewrite the cached alloys.")


if __name__ == "__main__":
    main()
