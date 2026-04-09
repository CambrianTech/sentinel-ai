#!/usr/bin/env python3
"""backfill_alloy_from_results.py — Synthesize a forge-alloy from a legacy
forging_results.json that pre-dates the alloy schema.

Use case: Several continuum-ai/* artifacts (qwen2.5-{0.5b,1.5b,3b}-general-forged,
qwen3.5-27b-code-forged) shipped before the forge-alloy schema existed. Their
provenance lives in a `forging_results.json` blob with the old Sentinel
result fields (model, strategy, pruning_level, baseline_ppl, final_ppl,
training_data, hardware_targets, etc).

The dispatch test catalog can't include these artifacts at Tier 1 because
they have no .alloy.json — and the brand-integrity story breaks because the
verifier has no Merkle envelope to walk. The fix is to synthesize a valid
alloy from the legacy results blob + the repo's config.json + README, then
publish the alloy alongside the existing weights via republish_alloy_only.py.

The backfilled alloy:
    - Carries the same forge stages the legacy run actually executed
      (prune → train, with the legacy strategy preserved as a free-form
      string in stage notes if it doesn't match the canonical enum)
    - Carries the legacy perplexity numbers in results.baselinePerplexity /
      results.finalPerplexity / results.improvementPct
    - Carries the legacy hardware_targets in results.hardwareVerified[]
    - Carries the model architecture from the actual config.json
    - Stamps a backfill marker in the alloy's notes so the audit trail is
      explicit ("this alloy was synthesized 2026-04-08 from legacy
      forging_results.json — the forge run itself happened on the date
      in results.completedAt")
    - Computes integrity.modelHash from the actual safetensors files in
      the repo (downloaded only enough to hash, not full weights)

Refuses to backfill if the repo already has a .alloy.json (use
republish_alloy_only.py for corrections, not this).

Usage:
    # Generate locally (writes to ./backfill_alloys/<repo-slug>.alloy.json)
    python scripts/backfill_alloy_from_results.py continuum-ai/qwen2.5-0.5b-general-forged

    # Generate locally + immediately upload to HF
    python scripts/backfill_alloy_from_results.py continuum-ai/qwen2.5-0.5b-general-forged --upload
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

# Reuse the existing publish_model.hash_file for safetensors hashing
sys.path.insert(0, str(Path(__file__).resolve().parent))


# Map common legacy strategy strings to canonical-ish enum values where there's
# a clear correspondence. Anything we don't recognize is preserved verbatim
# in stage notes and the strategy field gets a defensible default.
LEGACY_STRATEGY_MAP = {
    "combined":                "magnitude",       # legacy "combined" was magnitude+entropy hybrid
    "magnitude":               "magnitude",
    "entropy":                 "entropy",
    "gradient":                "gradient",
    "experiential_plasticity": "magnitude",       # legacy umbrella name
    "experiential-plasticity": "magnitude",
}


def _hf_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def _http_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())


def _http_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def _list_repo_files(repo: str) -> list[str]:
    meta = _http_json(f"https://huggingface.co/api/models/{repo}")
    return [s["rfilename"] for s in meta.get("siblings", [])]


# Per-shard hashing + composition lives in scripts/alloy_hashing.py — the
# single source of truth for the modelHash convention across publish and
# backfill paths. Roadmap step 7 unified the two paths there; the local
# duplicates were removed in this commit.
from alloy_hashing import compose_model_hash, fetch_shard_hashes_from_hf


def _detect_architecture(repo: str) -> str:
    """Read config.json from the repo and map model_type → alloy architecture string."""
    config = _http_json(_hf_url(repo, "config.json"))
    mt = config.get("model_type", "")
    archs = config.get("architectures", [])
    # Canonical mapping. Add cases as new families ship.
    if mt == "qwen2":
        return "qwen2"
    if mt in ("qwen3", "qwen3_5"):
        return "qwen3_5"
    if mt == "qwen3_moe":
        return "qwen3_moe"
    if mt == "olmoe":
        return "olmoe"
    if mt == "llama":
        return "llama"
    if mt == "mistral":
        return "mistral"
    if mt == "mixtral":
        return "mixtral"
    # Fall back to model_type as-is — the dispatch test will surface unknown
    # architectures via DispatchError, which is the right place for the gap.
    return mt or (archs[0] if archs else "unknown")


def _hash_safetensors_in_repo(repo: str, files: list[str]) -> str:
    """Hash all *.safetensors shards in the repo, in sorted order. Mirrors
    publish_model.hash_model_weights's convention so a backfilled alloy
    produces a modelHash that an independent verifier could reproduce."""
    shards = sorted(f for f in files if f.endswith(".safetensors"))
    if not shards:
        return ""
    h = hashlib.sha256()
    for shard in shards:
        print(f"  hashing {shard} ...")
        try:
            data = _http_bytes(_hf_url(repo, shard))
            h.update(data)
        except Exception as e:
            print(f"    WARNING: could not fetch {shard}: {e}")
            return ""  # Refuse partial hashing
    return f"sha256:{h.hexdigest()}"


def backfill(repo: str) -> dict:
    """Build a forge-alloy dict from the artifact's legacy provenance files."""
    print(f"Backfilling alloy for {repo}")
    files = _list_repo_files(repo)

    if any(f.endswith(".alloy.json") or f == "forge-alloy.json" for f in files):
        raise ValueError(
            f"{repo} already has an alloy file. Use republish_alloy_only.py "
            f"for corrections instead of backfill."
        )

    if "forging_results.json" not in files:
        raise ValueError(
            f"{repo} has no forging_results.json — nothing to backfill from. "
            f"Files: {files[:10]}..."
        )

    # 1. Pull the legacy results blob
    fr = _http_json(_hf_url(repo, "forging_results.json"))
    print(f"  legacy results: model={fr.get('model')} strategy={fr.get('strategy')} "
          f"level={fr.get('pruning_level')} ppl={fr.get('baseline_ppl')}→{fr.get('final_ppl')}")

    # 2. Detect architecture from config.json
    arch = _detect_architecture(repo)
    print(f"  architecture:   {arch}")

    # 3. Compose stages from legacy fields
    legacy_strategy = (fr.get("strategy") or "magnitude").lower()
    canonical_strategy = LEGACY_STRATEGY_MAP.get(legacy_strategy, "magnitude")
    prune_stage = {
        "type": "prune",
        "strategy": canonical_strategy,
        "level": float(fr.get("pruning_level", 0.3)),
    }
    if legacy_strategy != canonical_strategy:
        prune_stage["notes"] = (
            f"Legacy strategy name from forging_results.json: {legacy_strategy!r}. "
            f"Mapped to canonical {canonical_strategy!r} for the alloy schema. "
            f"The actual forge run used the legacy code path; this alloy is "
            f"a retroactive provenance record, not a re-execution recipe."
        )

    train_stage = {
        "type": "train",
        "domain": fr.get("domain", "general"),
        "steps": int(fr.get("training_steps", 0)) or 1,
        "learningRate": str(fr.get("learning_rate", "2e-4")),
    }
    if fr.get("training_data"):
        train_stage["dataset"] = fr["training_data"]
    if fr.get("training_method"):
        train_stage["notes"] = f"Legacy training method: {fr['training_method']}"

    stages = [prune_stage, train_stage]

    # 4. Hash the safetensors via HF's LFS metadata (no downloads).
    # Pulls per-shard sha256s from the shared alloy_hashing module and
    # composes a deterministic modelHash. Same convention as publish_model.py
    # post-roadmap-step-7 unification.
    print(f"  pulling per-shard LFS sha256s from HF metadata...")
    shard_hashes = fetch_shard_hashes_from_hf(repo, extensions=(".safetensors",))
    print(f"  shards: {len(shard_hashes)}")
    for s in shard_hashes[:3]:
        print(f"    {s['filename']}: {s['sha256'][:16]}... ({s.get('size','?')} bytes)")
    if len(shard_hashes) > 3:
        print(f"    ... ({len(shard_hashes) - 3} more)")
    model_hash = compose_model_hash(shard_hashes) if shard_hashes else "sha256:no-shards"
    print(f"  composed modelHash: {model_hash[:30]}...")

    # 5. Compose the alloy
    name = repo.split("/")[-1]
    base_model = fr.get("model", "unknown")
    alloy = {
        "name": name,
        "version": "1.0.0",
        "description": (
            f"Forged {base_model.split('/')[-1]} for {fr.get('domain','general')} "
            f"domain via per-layer head pruning + LoRA recovery training. "
            f"This alloy was retroactively synthesized from forging_results.json "
            f"on 2026-04-08 — the forge run itself executed at "
            f"{fr.get('forged_at', 'unknown date')}. The published model weights "
            f"are unchanged; this alloy adds the missing forge-alloy provenance "
            f"envelope so the artifact participates in the chain-of-custody system."
        ),
        "author": "continuum-ai",
        "tags": [
            fr.get("domain", "general"),
            "forged",
            "experiential-plasticity",
            "forge-alloy",
            "alloy-backfilled",
        ],
        "license": "apache-2.0",
        "source": {
            "baseModel": base_model,
            "architecture": arch,
        },
        "stages": stages,
        "cycles": int(fr.get("cycles", 1)),
        "results": {
            "completedAt": fr.get("forged_at"),
            "baselinePerplexity": fr.get("baseline_ppl"),
            "finalPerplexity": fr.get("final_ppl"),
            "improvementPct": fr.get("improvement_pct"),
            "benchmarks": [
                {
                    "name": "perplexity",
                    "metrics": {
                        "baseline": fr.get("baseline_ppl"),
                        "final": fr.get("final_ppl"),
                        "improvement_pct": fr.get("improvement_pct"),
                        "dataset": fr.get("training_data", "wikitext-2"),
                    },
                },
            ],
            "hardwareVerified": [
                {
                    "device": ht.get("device", ""),
                    "format": ht.get("format", "fp16"),
                    "verified": bool(ht.get("verified", False)),
                }
                for ht in fr.get("hardware_targets", [])
            ],
            "samples": [],
            "integrity": {
                "trustLevel": "self-attested",
                "code": {
                    "runner": "sentinel-ai/forge_model (legacy pre-§4.1.3.1 path)",
                    "version": "2.x",
                    "binaryHash": "sha256:legacy-pre-alloy-schema",
                },
                "modelHash": model_hash or "sha256:not-computed",
                "fileHashes": shard_hashes,  # per-shard attestation list
                "datasets": [
                    {
                        "name": fr.get("training_data", "Salesforce/wikitext"),
                        "hash": "sha256:not-pinned-legacy",
                    },
                ],
                "attestedAt": fr.get("forged_at"),
            },
        },
    }
    return alloy


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", help="HF repo id (e.g. continuum-ai/qwen2.5-0.5b-general-forged)")
    ap.add_argument("--out-dir", type=Path, default=Path("backfill_alloys"),
                    help="Where to write the backfilled alloy (default: ./backfill_alloys/)")
    ap.add_argument("--upload", action="store_true",
                    help="After writing locally, immediately upload to HF via republish_alloy_only.py "
                         "(uses --confirm).")
    args = ap.parse_args()

    try:
        alloy = backfill(args.repo)
    except ValueError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.repo.split('/')[-1]}.alloy.json"
    out_path.write_text(json.dumps(alloy, indent=2) + "\n")
    print(f"\nWrote backfilled alloy: {out_path} ({out_path.stat().st_size} bytes)")
    print(f"  benchmarks: {len(alloy['results']['benchmarks'])}")
    print(f"  stages:     {' → '.join(s['type'] for s in alloy['stages'])}")

    if args.upload:
        print("\nUploading via huggingface_hub...")
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo=out_path.name,
            repo_id=args.repo,
            commit_message=f"Backfill forge-alloy from forging_results.json (provenance envelope, weights unchanged)",
        )
        print(f"  ✓ uploaded {out_path.name} to {args.repo}")
        print(f"\nNOTE: this only uploads the alloy.json. Re-running model card +")
        print(f"      QR generation requires running scripts/republish_alloy_only.py:")
        print(f"      python scripts/republish_alloy_only.py --repo {args.repo} \\")
        print(f"          --alloy {out_path} --confirm")


if __name__ == "__main__":
    main()
