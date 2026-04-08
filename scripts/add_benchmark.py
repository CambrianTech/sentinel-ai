#!/usr/bin/env python3
"""
add_benchmark.py — Add benchmark results to an executed alloy and update HuggingFace.

Appends benchmark results to the alloy's benchmarks array without modifying
model weights. Regenerates QR, updates model card, re-uploads to HF.
The model hash stays the same. The evidence grows. The chain proves it.

Usage:
    python scripts/add_benchmark.py continuum-ai/qwen3.5-4b-code-forged humaneval --score 74.1 --passing 63 --total 85
    python scripts/add_benchmark.py continuum-ai/qwen3.5-4b-code-forged mmlu --score 68.5 --subset mmlu-pro --nshot 5
    python scripts/add_benchmark.py continuum-ai/qwen3.5-4b-code-forged --from-evalplus /tmp/evalplus_results/humaneval/
"""

import argparse
import hashlib
import json
import time
from pathlib import Path


def add_benchmark(repo_id: str, benchmark_name: str, metrics: dict,
                  subset: str = None, from_evalplus: str = None,
                  dry_run: bool = False):
    """Add a benchmark result to an alloy on HuggingFace."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()

    # Find the alloy file in the repo
    files = api.list_repo_files(repo_id)
    alloy_files = [f for f in files if f.endswith('.alloy.json')]
    if not alloy_files:
        print(f"ERROR: No .alloy.json found in {repo_id}")
        return

    alloy_filename = alloy_files[0]
    print(f"Alloy: {alloy_filename}")

    # Download current alloy
    alloy_path = hf_hub_download(repo_id, alloy_filename)
    alloy = json.loads(Path(alloy_path).read_text())

    # Check model hash hasn't changed (weights unchanged)
    if alloy.get("results", {}).get("integrity", {}).get("modelHash"):
        print(f"Model hash: {alloy['results']['integrity']['modelHash'][:40]}...")

    # Load evalplus results if specified
    if from_evalplus:
        metrics = _load_evalplus_results(from_evalplus, benchmark_name)
        if not metrics:
            print(f"ERROR: Could not load evalplus results from {from_evalplus}")
            return

    # Build benchmark entry
    entry = {
        "name": benchmark_name,
        "metrics": metrics,
        "submittedToLeaderboard": False,
    }
    if subset:
        entry["subset"] = subset

    # Add timestamp to metrics
    metrics["evaluatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    # Append to benchmarks
    if "results" not in alloy:
        alloy["results"] = {}
    if "benchmarks" not in alloy["results"]:
        alloy["results"]["benchmarks"] = []

    # Check if this benchmark already exists — update if so
    existing = [b for b in alloy["results"]["benchmarks"] if b["name"] == benchmark_name]
    if existing:
        print(f"Updating existing {benchmark_name} benchmark")
        existing[0]["metrics"] = metrics
    else:
        print(f"Adding new {benchmark_name} benchmark")
        alloy["results"]["benchmarks"].append(entry)

    print(f"  Metrics: {metrics}")

    if dry_run:
        print("DRY RUN — would update:")
        print(f"  Benchmarks: {len(alloy['results']['benchmarks'])}")
        return

    # Write updated alloy
    updated_path = Path(f"/tmp/{alloy_filename}")
    updated_path.write_text(json.dumps(alloy, indent=2))

    # Upload updated alloy
    api.upload_file(path_or_fileobj=str(updated_path), path_in_repo=alloy_filename, repo_id=repo_id)
    print(f"Alloy updated on HF")

    # Regenerate QR from the uploaded file (hash from delivery, not local)
    delivered_path = hf_hub_download(repo_id, alloy_filename)
    delivered_hash = hashlib.sha256(Path(delivered_path).read_bytes()).hexdigest()
    verify_url = f"https://cambriantech.github.io/forge-alloy/verify#{delivered_hash[:16]}"

    try:
        import qrcode
        qr = qrcode.make(verify_url)
        qr_path = Path("/tmp/alloy-qr-updated.png")
        qr.save(str(qr_path))
        api.upload_file(path_or_fileobj=str(qr_path), path_in_repo="alloy-qr.png", repo_id=repo_id)
        print(f"QR updated: {verify_url}")
    except ImportError:
        print(f"QR skipped (pip install qrcode[pil])")

    print(f"\nDONE: {benchmark_name} added to {repo_id}")
    print(f"Total benchmarks: {len(alloy['results']['benchmarks'])}")
    print(f"Verify: {verify_url}")


def _load_evalplus_results(result_dir: str, benchmark_name: str) -> dict:
    """Score an evalplus output directory or JSONL using the canonical
    evalplus pass@1 (the same number evalplus's CLI prints, the same
    number `eval_with_calibration.py` parses, the same number any third-
    party verifier will compute against the published JSONL).

    Previous version had two failure modes:
      1. The eval_results.json branch read keys (`pass@1.n_correct`)
         that don't exist in evalplus's actual output schema and always
         returned 0/164.
      2. The JSONL fallback counted `is_passing` / `passed` fields that
         the published JSONLs don't carry — always returned 0/164.
    Both meant `add_benchmark.py --from-evalplus` was a no-op that
    silently wrote 0% to the alloy.

    Fix: delegate to tests/reproducibility/_humaneval_scorer.py which
    wraps evalplus's official CLI with the macOS reliability_guard +
    fork-multiprocessing patches and returns the canonical pass@1 for
    BOTH humaneval (base) and humaneval_plus (base AND plus passes
    convention).

    Args:
        result_dir: path to a directory containing one or more *.jsonl
                    sample files, or to a single .jsonl file directly
        benchmark_name: 'humaneval' or 'humaneval_plus' (selects which
                    of the two pass@1 numbers to return)
    """
    result_dir = Path(result_dir)

    # Find the samples JSONL. Accept either a direct file or a directory.
    if result_dir.is_file() and result_dir.suffix == ".jsonl":
        samples_path = result_dir
    else:
        # Prefer a sanitized JSONL if present (evalplus.sanitize convention),
        # otherwise the first .jsonl in the directory or its humaneval/ subdir.
        candidates = []
        for d in (result_dir, result_dir / "humaneval"):
            if d.exists():
                candidates.extend(sorted(d.glob("*-sanitized.jsonl")))
                candidates.extend(sorted(d.glob("*.jsonl")))
        candidates = [c for c in candidates if not c.name.endswith("_eval_results.json")]
        if not candidates:
            print(f"  No JSONL samples found under {result_dir}")
            return None
        samples_path = candidates[0]

    # Delegate to the canonical scorer (uses evalplus's official CLI).
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "reproducibility"))
    from _humaneval_scorer import score_jsonl

    try:
        result = score_jsonl(samples_path)
    except Exception as e:
        print(f"  evalplus scoring failed: {e}")
        return None

    # Select the benchmark we were asked for.
    key = benchmark_name
    if key not in result:
        print(f"  scorer returned no key for benchmark {key!r}; got {sorted(result.keys())}")
        return None
    section = result[key]

    return {
        "passing": section.get("passed"),
        "total": section.get("total"),
        # Canonical pass@1 percentage. round to 2 dp NEVER 1 dp — 1 dp
        # rounding loses the third digit and creates the publish-pipeline
        # discrepancies the Tier 4 reproducibility test catches.
        "score": round(section["pass_at_1"] * 100, 2),
        "metric": "pass@1",
        "samplesPath": str(samples_path),
        "datasetHash": result.get("dataset_hash"),
    }


def main():
    parser = argparse.ArgumentParser(description="Add benchmark results to a published alloy")
    parser.add_argument("repo_id", help="HuggingFace repo ID (e.g., continuum-ai/qwen3.5-4b-code-forged)")
    parser.add_argument("benchmark", nargs="?", help="Benchmark name (e.g., humaneval, mmlu)")
    parser.add_argument("--score", type=float, help="Primary score")
    parser.add_argument("--passing", type=int, help="Number passing")
    parser.add_argument("--total", type=int, help="Total problems")
    parser.add_argument("--subset", type=str, help="Benchmark subset")
    parser.add_argument("--nshot", type=int, help="N-shot setting")
    parser.add_argument("--from-evalplus", type=str, help="Load from evalplus results directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    metrics = {}
    if args.score is not None:
        metrics["score"] = args.score
    if args.passing is not None:
        metrics["passing"] = args.passing
    if args.total is not None:
        metrics["total"] = args.total
    if args.nshot is not None:
        metrics["nShot"] = args.nshot

    if args.from_evalplus:
        benchmark_name = args.benchmark or "humaneval"
    elif not args.benchmark:
        parser.error("benchmark name is required (or use --from-evalplus)")
        return
    else:
        benchmark_name = args.benchmark

    add_benchmark(args.repo_id, benchmark_name, metrics,
                  subset=args.subset, from_evalplus=args.from_evalplus,
                  dry_run=args.dry_run)


if __name__ == "__main__":
    main()
