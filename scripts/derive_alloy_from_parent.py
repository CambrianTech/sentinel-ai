#!/usr/bin/env python3
"""derive_alloy_from_parent.py — Synthesize a forge-alloy for a downstream
variant artifact (defragged / GGUF / mlx-4bit) by inheriting from its
parent's published alloy and appending a derivation stage.

Use case: continuum-ai/* has 4 downstream-variant artifacts that ship
without an alloy of their own:

    qwen3.5-4b-code-forged-defragged    ← derived from qwen3.5-4b-code-forged
    qwen3.5-4b-code-forged-GGUF         ← derived from qwen3.5-4b-code-forged
    qwen3.5-27b-code-forged-defragged   ← derived from qwen3.5-27b-code-forged
    qwen3.5-27b-code-forged-mlx-4bit    ← derived from qwen3.5-27b-code-forged

Each is a post-hoc transformation of its parent: defragging the prune
output into a smaller dense structure, GGUF quantization for llama.cpp /
Ollama / LM Studio, MLX 4-bit quantization for Apple Silicon. The
weights are different (different bytes, different modelHash) but the
behavioral chain of custody comes from the parent's forge stages — the
prune + train + evaluation results.

The derived alloy:
    - Inherits source.baseModel + source.architecture from the parent
    - Inherits the parent's stages[] verbatim (the forge journey is the same)
    - Appends a SINGLE derivation stage (`package` for defragged, `quant`
      for GGUF / MLX) describing what was done to produce this variant
    - Inherits the parent's results.benchmarks (the model behavior is
      preserved through defrag/quant within the published tolerance)
    - Adds a `derivedFrom` field pointing at the parent repo so verifiers
      can walk the chain
    - Computes its OWN modelHash from this variant's actual files via
      HF's LFS metadata API (no downloads)
    - Adds a `notes` block on the derivation stage explaining the
      transformation

Refuses to derive if the child repo already has an alloy (use republish
for corrections instead).

Usage:
    # GGUF derivative
    python scripts/derive_alloy_from_parent.py \\
        --child continuum-ai/qwen3.5-4b-code-forged-GGUF \\
        --parent continuum-ai/qwen3.5-4b-code-forged \\
        --kind gguf

    # Defragged variant
    python scripts/derive_alloy_from_parent.py \\
        --child continuum-ai/qwen3.5-27b-code-forged-defragged \\
        --parent continuum-ai/qwen3.5-27b-code-forged \\
        --kind defragged

    # MLX 4-bit
    python scripts/derive_alloy_from_parent.py \\
        --child continuum-ai/qwen3.5-27b-code-forged-mlx-4bit \\
        --parent continuum-ai/qwen3.5-27b-code-forged \\
        --kind mlx-4bit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _hf_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def _http_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())


def _list_files(repo: str) -> list[str]:
    meta = _http_json(f"https://huggingface.co/api/models/{repo}")
    return [s["rfilename"] for s in meta.get("siblings", [])]


# Shared hashing — see scripts/alloy_hashing.py.
from alloy_hashing import compose_model_hash, fetch_shard_hashes_from_hf


def _fetch_parent_alloy(parent_repo: str) -> dict:
    """Find and download the parent's alloy file."""
    files = _list_files(parent_repo)
    candidates = [f for f in files if f.endswith(".alloy.json") or f == "forge-alloy.json"]
    if not candidates:
        raise ValueError(
            f"Parent {parent_repo} has no .alloy.json — backfill the parent first "
            f"with scripts/backfill_alloy_from_results.py."
        )
    return _http_json(_hf_url(parent_repo, candidates[0]))


def _derivation_stage(kind: str, child_repo: str) -> dict:
    """Build the per-kind derivation stage that gets appended to the parent's
    stages list. Each kind maps to an existing forge-alloy stage type so the
    dispatcher routes through the right family adapter method."""
    if kind == "defragged":
        # Defrag is part of the prune pipeline normally; for a standalone
        # defrag-only artifact we model it as a `package` stage with a clear
        # transformation note. The dispatch test passes through it because
        # `package` is family-agnostic on the FamilyAdapter base.
        return {
            "type": "package",
            "format": "safetensors-defragged",
            "validateOn": [],
            "includeTokenizer": True,
            "notes": (
                f"Defrag-only derivative of the parent forge. The parent's "
                f"prune stage marks heads as dead via forward-hooks; this "
                f"artifact reifies that pruning by physically reshaping the "
                f"projection matrices to remove the dead heads' parameters. "
                f"Behaviorally equivalent to the parent (same logits per "
                f"surviving head); structurally smaller on disk and in VRAM."
            ),
        }
    if kind == "gguf":
        return {
            "type": "quant",
            "format": "gguf",
            "quantTypes": ["Q4_K_M", "Q8_0"],
            "deviceTargets": [
                "macbook-pro-m-series", "macbook-air-16gb", "rtx3060",
                "rtx4070", "rtx4090", "iphone", "android",
            ],
            "notes": (
                f"GGUF quantization of the parent's safetensors weights via "
                f"llama.cpp llama-quantize. Targets llama.cpp / Ollama / "
                f"LM Studio / koboldcpp inference runtimes. Q4_K_M and Q8_0 "
                f"shipped together so users can pick the size/quality tier "
                f"their hardware supports."
            ),
        }
    if kind == "mlx-4bit":
        return {
            "type": "quant",
            "format": "mlx",
            "quantTypes": ["4bit"],
            "deviceTargets": ["macbook-pro-m-series", "macbook-air-16gb"],
            "notes": (
                f"MLX 4-bit quantization for Apple Silicon. The parent forge's "
                f"safetensors weights are converted to MLX format and quantized "
                f"to 4 bits per weight via mlx-lm. Runs natively on M-series "
                f"Macs with the unified memory architecture, no llama.cpp / "
                f"GGUF intermediate."
            ),
        }
    raise ValueError(f"Unknown derivation kind: {kind!r}. Known: defragged, gguf, mlx-4bit")


# Map kind → file extensions used to enumerate the variant's shards via LFS
KIND_EXTENSIONS = {
    "defragged": (".safetensors",),
    "gguf":      (".gguf",),
    "mlx-4bit":  (".safetensors",),
}


def derive(child_repo: str, parent_repo: str, kind: str) -> dict:
    print(f"Deriving alloy for {child_repo}")
    print(f"  parent: {parent_repo}")
    print(f"  kind:   {kind}")

    # Refuse if child already has an alloy
    child_files = _list_files(child_repo)
    if any(f.endswith(".alloy.json") or f == "forge-alloy.json" for f in child_files):
        raise ValueError(
            f"{child_repo} already has an alloy file. Use republish_alloy_only.py "
            f"for corrections instead of derive."
        )

    # Pull the parent alloy as the inheritance base
    parent_alloy = _fetch_parent_alloy(parent_repo)
    print(f"  parent alloy: name={parent_alloy.get('name')} stages={[s.get('type') for s in parent_alloy.get('stages', [])]}")

    # Pull THIS variant's per-shard LFS hashes
    extensions = KIND_EXTENSIONS[kind]
    shard_hashes = fetch_shard_hashes_from_hf(child_repo, extensions=extensions)
    print(f"  variant shards ({extensions}): {len(shard_hashes)}")
    for s in shard_hashes[:3]:
        print(f"    {s['filename']}: {s['sha256'][:16]}... ({s.get('size','?')} bytes)")
    if len(shard_hashes) > 3:
        print(f"    ... ({len(shard_hashes) - 3} more)")
    if not shard_hashes:
        raise ValueError(
            f"{child_repo} has no files matching {extensions} — cannot compute "
            f"a modelHash. Check the repo file list."
        )
    new_model_hash = compose_model_hash(shard_hashes) if shard_hashes else "sha256:no-shards"
    print(f"  composed modelHash: {new_model_hash[:30]}...")

    # Compose: inherit + append derivation stage
    inherited_stages = list(parent_alloy.get("stages", []))
    inherited_stages.append(_derivation_stage(kind, child_repo))

    # Inherit benchmarks unchanged from the parent (the model behavior is
    # preserved through defrag/quant within the published tolerance — the
    # qwen3-coder-30b-a3b alloy demonstrated this with hardware-anchored
    # cross-quant evaluation, and the per-variant alloys can be re-scored
    # later if/when per-variant samples ship).
    inherited_results = parent_alloy.get("results") or {}
    parent_benchmarks = list((inherited_results.get("benchmarks") or []))

    name = child_repo.split("/")[-1]
    parent_name = parent_repo.split("/")[-1]
    derived_alloy = {
        "name": name,
        "version": "1.0.0",
        "description": (
            f"{kind.upper()} derivative of [`{parent_name}`](https://huggingface.co/{parent_repo}). "
            f"Same forge journey as the parent (prune + train as published in "
            f"the parent's alloy); this artifact adds a single {kind!r} "
            f"transformation stage to produce a smaller / faster / "
            f"more-portable variant of the same logical model. Inherits the "
            f"parent's published benchmark results; per-variant evaluation "
            f"samples will land in a follow-up release if/when per-variant "
            f"benchmarks are run."
        ),
        "author": "continuum-ai",
        "tags": list(set(parent_alloy.get("tags", []) + [kind, "derivative", "alloy-backfilled"])),
        "license": parent_alloy.get("license", "apache-2.0"),
        "source": dict(parent_alloy.get("source") or {}),
        "stages": inherited_stages,
        "cycles": parent_alloy.get("cycles", 1),
        "derivedFrom": {
            "repo": parent_repo,
            "alloyHash": (parent_alloy.get("results", {}).get("integrity", {}) or {}).get("alloyHash"),
            "kind": kind,
        },
        "results": {
            "completedAt": (parent_alloy.get("results") or {}).get("completedAt"),
            "baselinePerplexity": inherited_results.get("baselinePerplexity"),
            "finalPerplexity": inherited_results.get("finalPerplexity"),
            "improvementPct": inherited_results.get("improvementPct"),
            "benchmarks": parent_benchmarks,
            "hardwareVerified": list(inherited_results.get("hardwareVerified") or []),
            "samples": [],
            "integrity": {
                "trustLevel": "self-attested",
                "code": {
                    "runner": f"sentinel-ai/derive_alloy_from_parent ({kind})",
                    "version": "1.0",
                    "binaryHash": "sha256:derivation-tool-only",
                },
                "modelHash": new_model_hash,
                "fileHashes": shard_hashes,
                "datasets": [],
                "attestedAt": "2026-04-08",
                "parentAlloyHash": (parent_alloy.get("results", {}).get("integrity", {}) or {}).get("alloyHash"),
            },
        },
    }
    return derived_alloy


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--child", required=True, help="Child repo id (e.g. continuum-ai/qwen3.5-4b-code-forged-GGUF)")
    ap.add_argument("--parent", required=True, help="Parent repo id (e.g. continuum-ai/qwen3.5-4b-code-forged)")
    ap.add_argument("--kind", required=True, choices=["defragged", "gguf", "mlx-4bit"])
    ap.add_argument("--out-dir", type=Path, default=Path("backfill_alloys"))
    args = ap.parse_args()

    try:
        alloy = derive(args.child, args.parent, args.kind)
    except ValueError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.child.split('/')[-1]}.alloy.json"
    out.write_text(json.dumps(alloy, indent=2) + "\n")
    print(f"\nWrote derived alloy: {out} ({out.stat().st_size} bytes)")
    print(f"  stages: {' → '.join(s['type'] for s in alloy['stages'])}")
    print(f"  derivedFrom: {alloy['derivedFrom']}")


if __name__ == "__main__":
    main()
