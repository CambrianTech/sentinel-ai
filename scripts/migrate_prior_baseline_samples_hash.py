#!/usr/bin/env python3
"""migrate_prior_baseline_samples_hash.py — populate samplesHash on every
priorMetricBaselines cell that has a samplesPath but no hash.

Roadmap step 8 from docs/PLUGIN-SPRINT.md. The §4.1.3.4 falsifiability
anchors (priorMetricBaselines[]) publish their negative-baseline JSONLs
to HF but the alloy schema today has no samplesHash field on the
evaluation block. This means anyone with HF write access could swap
the negative-baseline JSONL after publish and the published Δ wouldn't
be byte-verifiable.

This tool walks every cached alloy under tests/reproducibility/_cache/,
finds every priorMetricBaselines cell with a samplesPath, downloads the
samples file from HuggingFace if not already cached locally, computes
sha256 of the bytes, and writes 'sha256:<hex>' into the cell's
evaluation.samplesHash field. Idempotent — running it twice produces
the same output.

Affected cells (today):
    qwen3-coder-30b-a3b-compacted-19b-256k priorMetricBaselines[router-gate-l2-norm-2026-04-08]
        eval/humaneval/student_samples_router_l2_baseline.jsonl
    olmoe-1b-7b-compacted-5b priorMetricBaselines[olmoe-broad-corpus-2026-04-08]
        eval/humaneval/student_samples_broad_calibration.jsonl

The local cache is the source of truth — this script does NOT push to HF.
After the migration runs and the test passes, run republish_alloy_only.py
to push the corrected alloys to HF with the new samplesHash field.

Usage:
    # Dry-run (default)
    python scripts/migrate_prior_baseline_samples_hash.py

    # Actually rewrite the cached alloys
    python scripts/migrate_prior_baseline_samples_hash.py --confirm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "tests" / "reproducibility" / "_cache"
SAMPLES_CACHE = CACHE_DIR / "samples"


def repo_id_from_cache_filename(filename: str) -> str:
    """Cache filename convention: 'continuum-ai_<slug>__<alloy-name>.json'."""
    stem = filename.split("__", 1)[0]
    return stem.replace("_", "/", 1)


def fetch_samples_bytes(repo: str, samples_path: str) -> bytes:
    """Fetch the samples JSONL bytes from HuggingFace, caching locally."""
    SAMPLES_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = SAMPLES_CACHE / f"{repo.replace('/', '_')}__{samples_path.replace('/', '_')}"
    if not cache_path.exists():
        url = f"https://huggingface.co/{repo}/resolve/main/{samples_path}"
        print(f"      fetching {url}")
        with urllib.request.urlopen(url, timeout=60) as resp:
            cache_path.write_bytes(resp.read())
    return cache_path.read_bytes()


def migrate_one(cache_path: Path, confirm: bool) -> dict:
    """Process one cached alloy. Returns a status dict."""
    alloy = json.loads(cache_path.read_text())
    repo = repo_id_from_cache_filename(cache_path.name)

    pmbs = alloy.get("priorMetricBaselines", [])
    if not pmbs:
        return {"file": cache_path.name, "status": "skipped", "reason": "no priorMetricBaselines"}

    pinned_now = []
    already_pinned = []
    skipped_no_path = []

    for pmb in pmbs:
        ev = pmb.get("evaluation") or {}
        samples_path = ev.get("samplesPath")
        existing_hash = ev.get("samplesHash")
        if not samples_path:
            skipped_no_path.append(pmb.get("id", "?"))
            continue
        if existing_hash:
            already_pinned.append(pmb.get("id", "?"))
            continue

        # Download (or load from cache) and hash
        data = fetch_samples_bytes(repo, samples_path)
        sha = hashlib.sha256(data).hexdigest()
        new_hash = f"sha256:{sha}"
        ev["samplesHash"] = new_hash
        pmb["evaluation"] = ev
        pinned_now.append((pmb.get("id", "?"), new_hash[:30] + "...", len(data)))

    if not pinned_now:
        return {
            "file": cache_path.name,
            "status": "skipped",
            "reason": (
                f"no work to do: {len(already_pinned)} already pinned, "
                f"{len(skipped_no_path)} have no samplesPath"
            ),
        }

    print(f"\n  {cache_path.name}")
    print(f"    repo: {repo}")
    for pmb_id, hash_preview, size in pinned_now:
        print(f"    + {pmb_id}: {hash_preview} ({size} bytes)")

    if confirm:
        cache_path.write_text(json.dumps(alloy, indent=2) + "\n")
        print(f"    WROTE {cache_path.name}")
    else:
        print(f"    DRY-RUN: would write {cache_path.name}")

    return {
        "file": cache_path.name,
        "status": "migrated",
        "pinned": [p[0] for p in pinned_now],
        "already_pinned": already_pinned,
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
    print(f"Walking {len(cache_files)} cached alloys for unpinned priorMetricBaselines")
    if not args.confirm:
        print("DRY RUN — re-run with --confirm to actually rewrite")

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
