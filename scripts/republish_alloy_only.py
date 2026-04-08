#!/usr/bin/env python3
"""republish_alloy_only.py — Re-upload an alloy + its derived QR + card to HF
WITHOUT touching the model weights.

Use case: a published alloy has a wrong field (e.g. the qwen3-coder-30b-a3b-
compacted-19b-256k humaneval_plus values that were authored with a non-
canonical pass@1 convention and overstated by 0.6 pp). The model weights
are still correct — only the alloy text needs fixing. Re-uploading the
weights is wasteful (multiple GB) and risky (could touch the modelHash
chain). This script touches only the metadata files.

What it does:
    1. Read a corrected local alloy file (the source of truth)
    2. Compute its sha256 → that's the new alloyHash
    3. Diff against the alloy currently published on HF — refuses to
       proceed if there's no actual change (defensive against re-publish
       loops)
    4. Generate a new alloy-qr.png that encodes the new verify URL
    5. Generate a new README.md via alloy_to_card with the new hash
    6. Upload all three files atomically (alloy.json + alloy-qr.png + README.md)
    7. Print the verify URL change

Refuses to upload unless --confirm is passed. Default is dry-run that
prints exactly what would change.

Usage:
    # Dry-run (default — no upload)
    python scripts/republish_alloy_only.py \\
        --repo continuum-ai/qwen3-coder-30b-a3b-compacted-19b-256k \\
        --alloy tests/reproducibility/_cache/.../qwen3-coder-30b-a3b-compacted-19b-256k.alloy.json

    # Actually push
    python scripts/republish_alloy_only.py \\
        --repo continuum-ai/qwen3-coder-30b-a3b-compacted-19b-256k \\
        --alloy <path> \\
        --confirm

Safety:
    - Refuses if local alloy bytes are identical to current HF bytes (no diff to push).
    - Refuses if local alloy lacks results.benchmarks or results.integrity
      (defensive against publishing a recipe-only alloy as a results alloy).
    - Refuses if results.integrity.modelHash differs from the current HF
      alloy's modelHash (the model weights would have to be re-published
      separately — that's a different operation, use publish_model.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

# Add scripts/ to path so we can import alloy_to_card and reuse its renderer
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _verify_url(alloy_hash_hex: str) -> str:
    return f"https://cambriantech.github.io/forge-alloy/verify/#{alloy_hash_hex[:16]}"


def _diff_summary(old: dict, new: dict) -> list[str]:
    """Return a human-readable list of meaningful field changes."""
    changes: list[str] = []

    # Version
    if old.get("version") != new.get("version"):
        changes.append(f"version: {old.get('version')!r} → {new.get('version')!r}")

    # Benchmarks (the most likely thing to be corrected)
    old_b = {b.get("name"): b for b in (old.get("results") or {}).get("benchmarks", [])}
    new_b = {b.get("name"): b for b in (new.get("results") or {}).get("benchmarks", [])}
    for name in sorted(set(old_b) | set(new_b)):
        ob = old_b.get(name) or {}
        nb = new_b.get(name) or {}
        for field in ("score", "baseScore", "delta"):
            if ob.get(field) != nb.get(field):
                changes.append(
                    f"results.benchmarks[{name}].{field}: "
                    f"{ob.get(field)!r} → {nb.get(field)!r}"
                )
        if "scoreCorrection" in nb and "scoreCorrection" not in ob:
            changes.append(f"results.benchmarks[{name}].scoreCorrection: ADDED")

    # priorMetricBaselines deltas
    old_pmb_count = len((old.get("priorMetricBaselines") or []))
    new_pmb_count = len((new.get("priorMetricBaselines") or []))
    if old_pmb_count != new_pmb_count:
        changes.append(f"priorMetricBaselines count: {old_pmb_count} → {new_pmb_count}")

    return changes


def republish(repo: str, local_alloy_path: Path, confirm: bool) -> int:
    """Returns 0 on success, non-zero on refusal."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()

    # 1. Read local corrected alloy
    if not local_alloy_path.exists():
        print(f"ERROR: local alloy not found: {local_alloy_path}", file=sys.stderr)
        return 2
    new_bytes = local_alloy_path.read_bytes()
    try:
        new_alloy = json.loads(new_bytes)
    except json.JSONDecodeError as e:
        print(f"ERROR: local alloy is not valid JSON: {e}", file=sys.stderr)
        return 2
    new_hash = _hash_bytes(new_bytes)

    # 2. Find the alloy filename in the HF repo (different repos use different names)
    print(f"Listing files in {repo} ...")
    files = api.list_repo_files(repo)
    alloy_filenames = [f for f in files if f.endswith(".alloy.json")]
    if not alloy_filenames:
        # Fallback: some legacy repos use forge-alloy.json
        alloy_filenames = [f for f in files if f == "forge-alloy.json"]

    # NEW: backfill mode — repo has NO alloy at all. Use the local file's
    # basename as the in-repo path. Skip the diff-against-current-HF check
    # because there's nothing to diff.
    backfill_mode = not alloy_filenames
    if backfill_mode:
        alloy_filename = local_alloy_path.name
        print(f"  no alloy on HF — backfill mode, will upload as: {alloy_filename}")
        old_alloy: dict = {}
        old_hash = "0" * 64
    else:
        alloy_filename = alloy_filenames[0]
        print(f"  alloy filename: {alloy_filename}")
        # 3. Download current alloy + diff
        old_local = Path(hf_hub_download(repo, alloy_filename))
        old_bytes = old_local.read_bytes()
        old_alloy = json.loads(old_bytes)
        old_hash = _hash_bytes(old_bytes)

        if new_bytes == old_bytes:
            print("REFUSED: local alloy bytes are identical to current HF alloy.")
            print("         Nothing to upload.")
            return 3

    print(f"\n  current alloyHash: {old_hash[:16]}  ({_verify_url(old_hash)})")
    print(f"  new     alloyHash: {new_hash[:16]}  ({_verify_url(new_hash)})")

    # 4. Defensive: refuse if modelHash changes (unless we're backfilling — in
    # which case the old alloy doesn't exist so there's no old modelHash to
    # compare against)
    if not backfill_mode:
        old_mh = ((old_alloy.get("results") or {}).get("integrity") or {}).get("modelHash")
        new_mh = ((new_alloy.get("results") or {}).get("integrity") or {}).get("modelHash")
        if old_mh and new_mh and old_mh != new_mh:
            print(f"\nREFUSED: results.integrity.modelHash changed:")
            print(f"  old: {old_mh}")
            print(f"  new: {new_mh}")
            print(f"  Use publish_model.py for a full re-publish that includes weights.")
            return 4

    # 5. Show meaningful field diff (or full benchmark list in backfill mode)
    if backfill_mode:
        print("\nBackfill — fields that will land on HF:")
        for b in (new_alloy.get("results") or {}).get("benchmarks", []):
            print(f"  results.benchmarks[{b.get('name')}]: {b.get('metrics', {})}")
        print(f"  source.architecture: {((new_alloy.get('source') or {}).get('architecture'))}")
        print(f"  source.baseModel:    {((new_alloy.get('source') or {}).get('baseModel'))}")
    else:
        print("\nDiff summary (corrected fields):")
        changes = _diff_summary(old_alloy, new_alloy)
        if not changes:
            print("  (no recognizable field changes — bytes differ on whitespace/order only)")
        else:
            for c in changes:
                print(f"  {c}")

    # 6. Generate the new card via alloy_to_card
    from alloy_to_card import alloy_to_card
    new_card = alloy_to_card(new_alloy, new_hash)
    print(f"\n  generated README.md: {len(new_card)} chars")

    # 7. Generate the new QR
    qr_bytes: bytes | None = None
    try:
        import qrcode  # type: ignore
        from io import BytesIO
        qr = qrcode.make(_verify_url(new_hash))
        buf = BytesIO()
        qr.save(buf)
        qr_bytes = buf.getvalue()
        print(f"  generated alloy-qr.png: {len(qr_bytes)} bytes")
    except ImportError:
        print(f"  alloy-qr.png: SKIPPED (pip install 'qrcode[pil]')")

    # 8. Confirm
    if not confirm:
        print("\nDRY RUN — no upload. Re-run with --confirm to push the changes.")
        print("\nFiles that WOULD be uploaded:")
        print(f"  - {alloy_filename}  ({len(new_bytes)} bytes)")
        print(f"  - README.md          ({len(new_card)} chars)")
        if qr_bytes:
            print(f"  - alloy-qr.png       ({len(qr_bytes)} bytes)")
        print(f"\nFiles NOT touched: model weights, eval/*.jsonl, calibration/*, tokenizer*, config*")
        return 0

    # 9. Upload atomically — alloy first (so its hash is canonical), then QR + README
    print("\nUploading...")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        alloy_tmp = tmp / alloy_filename
        alloy_tmp.write_bytes(new_bytes)
        api.upload_file(
            path_or_fileobj=str(alloy_tmp),
            path_in_repo=alloy_filename,
            repo_id=repo,
            commit_message=f"Correct {alloy_filename} pass@1 to canonical evalplus convention (v{new_alloy.get('version','?')})",
        )
        print(f"  ✓ uploaded {alloy_filename}")

        readme_tmp = tmp / "README.md"
        readme_tmp.write_text(new_card)
        api.upload_file(
            path_or_fileobj=str(readme_tmp),
            path_in_repo="README.md",
            repo_id=repo,
            commit_message=f"Regenerate model card from corrected alloy (alloyHash {new_hash[:16]})",
        )
        print(f"  ✓ uploaded README.md")

        if qr_bytes:
            qr_tmp = tmp / "alloy-qr.png"
            qr_tmp.write_bytes(qr_bytes)
            api.upload_file(
                path_or_fileobj=str(qr_tmp),
                path_in_repo="alloy-qr.png",
                repo_id=repo,
                commit_message=f"Regenerate QR for new verify URL ({new_hash[:16]})",
            )
            print(f"  ✓ uploaded alloy-qr.png")

    print(f"\nDone. New verify URL: {_verify_url(new_hash)}")
    print(f"Old verify URL stays cached at: {_verify_url(old_hash)} (orphaned, will not resolve)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. continuum-ai/qwen3-coder-30b-a3b-compacted-19b-256k")
    ap.add_argument("--alloy", required=True, type=Path, help="Path to the corrected local .alloy.json")
    ap.add_argument("--confirm", action="store_true", help="Actually push (default is dry-run)")
    args = ap.parse_args()
    sys.exit(republish(args.repo, args.alloy, args.confirm))


if __name__ == "__main__":
    main()
