#!/usr/bin/env python3
"""
alloy_executor.py — Execute ForgeAlloy pipelines stage by stage.

Reads an .alloy.json, executes each stage in order, writes results back.
Stage executors live in scripts/stages/ — one file per stage group.

Usage:
    python scripts/alloy_executor.py path/to/recipe.alloy.json
    python scripts/alloy_executor.py path/to/recipe.alloy.json --output-dir output/custom
    python scripts/alloy_executor.py path/to/recipe.alloy.json --dry-run
"""

import argparse
import hashlib

# Use forge-alloy SDK for type-safe alloy handling
try:
    from forge_alloy import ForgeAlloy
    HAS_SDK = True
except ImportError:
    HAS_SDK = False
import json
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

# Add scripts dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from stages import ForgeContext, create_executor, STAGE_EXECUTORS


# Stage position classification
INPUT_TYPES = {"source-config", "context-extend", "modality"}
OUTPUT_TYPES = {"quant", "package", "eval", "deliver", "publish", "deploy"}


def _resolve_local_model(model_name: str) -> str:
    """Check if a HF model exists locally from a previous forge, avoiding re-download.

    Searches output/forged/*/model/ for a config.json that matches.
    Prefers the most specific match (longest candidate name that matches slug).
    Returns local path if found, otherwise the original HF model name.
    """
    # Only try for org/name format (HF repos)
    if "/" not in model_name:
        return model_name

    slug = model_name.split("/")[-1].lower()
    search_dirs = [
        Path("output/forged"),
        Path.home() / "sentinel-ai" / "output" / "forged",
    ]
    best_match = None
    best_score = -1

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for candidate in search_dir.iterdir():
            model_dir = candidate / "model"
            config = model_dir / "config.json"
            if not (config.exists() and (model_dir / "model.safetensors").exists()):
                continue
            candidate_name = candidate.name.lower()
            # Exact slug match is best
            if candidate_name == slug:
                return str(model_dir)
            # Score by common prefix length (split on '-' to compare segments)
            # e.g., "qwen3.5-4b-code-128k-final" vs "qwen3.5-4b-code-128k-forged"
            #   → 4 matching segments = score 4 (very good)
            # vs "qwen3.5-4b" → 2 matching segments = score 2 (less specific)
            slug_parts = slug.split("-")
            cand_parts = candidate_name.split("-")
            common = 0
            for s, c in zip(slug_parts, cand_parts):
                if s == c:
                    common += 1
                else:
                    break
            # Require at least 2 matching segments to avoid false positives
            if common >= 2 and common > best_score:
                best_score = common
                best_match = str(model_dir)

    return best_match or model_name


def execute_alloy(alloy_path: str, output_dir: str = None, dry_run: bool = False):
    """Execute a complete ForgeAlloy pipeline."""
    # Load via SDK if available (validates types), fall back to raw JSON
    if HAS_SDK:
        alloy_obj = ForgeAlloy.from_file(alloy_path)
        alloy = json.loads(alloy_obj.model_dump_json(by_alias=True))
        print(f"  Loaded via forge-alloy SDK v{__import__('forge_alloy').__version__}")
    else:
        alloy = json.loads(Path(alloy_path).read_text())
        print(f"  Loaded raw JSON (install forge-alloy SDK for validation)")
    stages = alloy.get("stages", [])
    cycles = alloy.get("cycles", 1)
    model_name = alloy["source"]["baseModel"]
    name = alloy.get("name", "unnamed")

    print(f"\n{'='*60}")
    print(f"  ALLOY EXECUTOR: {name} v{alloy.get('version', '?')}")
    print(f"  Model: {model_name}")
    print(f"  Stages: {len(stages)}, Cycles: {cycles}")
    print(f"  Pipeline: {' → '.join(s['type'] for s in stages)}")
    print(f"{'='*60}\n")

    if dry_run:
        _dry_run(stages, cycles)
        return

    # Classify stages
    input_stages = [s for s in stages if s["type"] in INPUT_TYPES]
    transform_stages = [s for s in stages if s["type"] not in INPUT_TYPES and s["type"] not in OUTPUT_TYPES]
    output_stages = [s for s in stages if s["type"] in OUTPUT_TYPES]

    # Setup
    slug = model_name.split("/")[-1].lower()
    out = Path(output_dir or f"output/forged/{slug}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    (out / "benchmark").mkdir(exist_ok=True)

    ctx = ForgeContext(model_name=model_name, output_dir=out, alloy=alloy)

    # Load model — check local output dirs before downloading from HF
    print("[1] Loading model...")
    import torch
    from forge_model import load_model, get_model_info, evaluate, make_dataloaders, ForgeConfig, generate_samples

    load_path = _resolve_local_model(model_name)
    if load_path != model_name:
        print(f"  Using local: {load_path}")
    ctx.info = get_model_info(load_path)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    cfg = ForgeConfig.auto(ctx.info["fp16_gb"], vram_gb)
    ctx.tier = cfg.tier
    ctx.load_4bit = cfg.load_4bit
    ctx.device = torch.cuda.get_device_name(0)
    ctx.model, ctx.tokenizer = load_model(load_path, cfg.load_4bit)

    # Populate ctx.source_model_dir — the absolute on-disk path to the
    # source model files. Family adapter expert_prune methods need this
    # for the streaming CPU pruner (it walks safetensors shards directly,
    # not via the in-memory model object). For HF-cached models, resolve
    # the snapshot path via huggingface_hub.snapshot_download with
    # local_files_only=True (no network — just returns the cache path
    # if the model is already loaded).
    try:
        from pathlib import Path as _P
        if _P(load_path).exists():
            # Already a local path
            ctx.source_model_dir = str(_P(load_path).resolve())
        else:
            # HF id — resolve snapshot path from the cache
            from huggingface_hub import snapshot_download
            ctx.source_model_dir = snapshot_download(
                repo_id=load_path, local_files_only=True,
            )
        print(f"  source_model_dir: {ctx.source_model_dir}")
    except Exception as e:
        print(f"  WARN: could not resolve source_model_dir: {e}")
        ctx.source_model_dir = None

    # Input stages
    print("\n[2] Input stages...")
    for stage in input_stages:
        ctx = create_executor(stage).execute(ctx)

    # Baseline
    print("\n[3] Baseline evaluation...")
    domain = _find_domain(transform_stages)
    _, eval_loader = make_dataloaders(ctx.tokenizer, cfg, domain)
    baseline = evaluate(ctx.model, eval_loader, out, "baseline")
    ctx.baseline_ppl = baseline["perplexity"]
    print(f"  Baseline perplexity: {ctx.baseline_ppl:.2f}")

    # Transform stages (cycled)
    # Layer 6 invariant: NO SILENT REGRESSION ACROSS CYCLES.
    # After each cycle, eval and compare to the previous cycle.
    # If perplexity regresses by more than the noise floor, halt and dump state.
    # This is the structural fix that makes the LoRA-on-pruned-hooks bug class
    # impossible to ship silently. See continuum #842 and kash-feedback.md.
    cycle_ppls = [ctx.baseline_ppl]
    REGRESSION_THRESHOLD_RATIO = 1.10   # >10% worse than previous cycle = regression
    REGRESSION_THRESHOLD_ABS = 1.0      # AND >1.0 PPL absolute (avoid false alarms on tiny baselines)
    for cycle in range(1, cycles + 1):
        print(f"\n[4.{cycle}] Cycle {cycle}/{cycles}")
        for stage in transform_stages:
            ctx = create_executor(stage).execute(ctx)

        # Per-cycle eval — same loader as baseline so the comparison is apples-to-apples
        # Note: at this point hooks are still active, so this is the "training-side" view
        cycle_eval = evaluate(ctx.model, eval_loader, out, f"cycle-{cycle}-eval")
        cycle_ppl = cycle_eval["perplexity"]
        cycle_ppls.append(cycle_ppl)
        prev_ppl = cycle_ppls[-2]
        print(f"  Cycle {cycle} eval: {cycle_ppl:.2f} (prev: {prev_ppl:.2f})")

        # Check for silent regression
        regressed_ratio = cycle_ppl > prev_ppl * REGRESSION_THRESHOLD_RATIO
        regressed_abs = cycle_ppl - prev_ppl > REGRESSION_THRESHOLD_ABS
        if regressed_ratio and regressed_abs:
            print()
            print("=" * 70)
            print(f"  HALT: Layer 6 invariant violated at cycle {cycle}")
            print("=" * 70)
            print(f"  Previous cycle PPL: {prev_ppl:.2f}")
            print(f"  Current cycle PPL:  {cycle_ppl:.2f}")
            print(f"  Regression: {((cycle_ppl - prev_ppl) / prev_ppl * 100):.1f}%")
            print(f"  Threshold: >{(REGRESSION_THRESHOLD_RATIO - 1) * 100:.0f}% AND >{REGRESSION_THRESHOLD_ABS} PPL absolute")
            print()
            print("  This is a SILENT REGRESSION HALT — the harness has detected that")
            print("  this cycle made the model worse than the previous cycle and refuses")
            print("  to advance. Investigate before re-running. Per-cycle data:")
            for i, ppl in enumerate(cycle_ppls):
                marker = "  ←" if i == len(cycle_ppls) - 1 else ""
                print(f"    Cycle {i}: {ppl:.2f}{marker}")
            print()
            print("  References:")
            print("    - continuum #842 (Layer 6 invariant)")
            print("    - sentinel-ai #152 (LoRA-on-pruned-hooks: the bug class this catches)")
            print("    - continuum kash-feedback.md")
            print("=" * 70)

            # Dump state to a file for post-mortem
            import json as _json
            halt_state = {
                "halted_at_cycle": cycle,
                "cycle_ppls": cycle_ppls,
                "regression_ratio": cycle_ppl / prev_ppl,
                "regression_absolute": cycle_ppl - prev_ppl,
                "threshold_ratio": REGRESSION_THRESHOLD_RATIO,
                "threshold_absolute": REGRESSION_THRESHOLD_ABS,
                "model": model_name,
                "domain": domain,
            }
            (out / "REGRESSION_HALT.json").write_text(_json.dumps(halt_state, indent=2))
            raise RuntimeError(
                f"Layer 6 silent-regression invariant: cycle {cycle} PPL {cycle_ppl:.2f} "
                f"is >{(REGRESSION_THRESHOLD_RATIO - 1) * 100:.0f}% worse than previous cycle "
                f"({prev_ppl:.2f}). Halting. State dumped to {out / 'REGRESSION_HALT.json'}."
            )

    # Cleanup hooks
    for h in ctx.hooks:
        h.remove()
    ctx.hooks.clear()

    # Final eval
    print("\n[5] Final evaluation...")
    final = evaluate(ctx.model, eval_loader)
    ctx.final_ppl = final["perplexity"]
    imp = (ctx.baseline_ppl - ctx.final_ppl) / ctx.baseline_ppl * 100
    print(f"  Final: {ctx.baseline_ppl:.2f} → {ctx.final_ppl:.2f} ({imp:+.1f}%)")

    # Save model
    print("\n[6] Saving model...")
    model_dir = out / "model"
    model_dir.mkdir(exist_ok=True)
    ctx.model.save_pretrained(str(model_dir))
    ctx.tokenizer.save_pretrained(str(model_dir))

    # Generate samples
    print("\n[7] Generating samples...")
    ctx.samples = generate_samples(ctx.model, ctx.tokenizer, domain)
    for sname, text in ctx.samples.items():
        (out / "benchmark" / f"{sname}.txt").write_text(text)

    # Output stages
    print("\n[8] Output stages...")
    for stage in output_stages:
        ctx = create_executor(stage).execute(ctx)

    # Results
    results = {
        "model": model_name,
        "domain": domain,
        "baseline_ppl": round(ctx.baseline_ppl, 4),
        "final_ppl": round(ctx.final_ppl, 4),
        "improvement_pct": round(imp, 2),
        "forged_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": ctx.device,
        "tier": ctx.tier,
        "cycles": cycles,
        "stages": [s["type"] for s in stages],
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))

    # Executed alloy
    _write_executed_alloy(ctx, results, out)

    print(f"\n{'='*60}")
    print(f"  {model_name}: {ctx.baseline_ppl:.2f} → {ctx.final_ppl:.2f} ({imp:+.1f}%)")
    print(f"  Output: {out}")
    print(f"{'='*60}")


def _find_domain(transform_stages: list) -> str:
    """Extract domain from train stages."""
    for s in transform_stages:
        if s["type"] in ("train", "lora") and "domain" in s:
            return s["domain"]
    return "general"


def _dry_run(stages: list, cycles: int):
    """Show what would execute without running."""
    print("DRY RUN — showing what would execute:\n")
    for i, stage in enumerate(stages):
        stype = stage["type"]
        status = "READY" if stype in STAGE_EXECUTORS else "NOT IMPLEMENTED"
        position = "INPUT" if stype in INPUT_TYPES else "OUTPUT" if stype in OUTPUT_TYPES else "TRANSFORM"
        print(f"  Stage {i+1}: {stype} [{status}] ({position})")
        for k, v in stage.items():
            if k != "type":
                print(f"    {k}: {v}")
    print(f"\n  Cycles: {cycles} (transform stages repeat)")


def _write_executed_alloy(ctx: ForgeContext, results: dict, out: Path):
    """Write the executed alloy with full results and attestation."""
    alloy = {k: v for k, v in ctx.alloy.items() if not k.startswith("_")}

    if "forge-alloy" not in alloy.get("tags", []):
        alloy.setdefault("tags", []).append("forge-alloy")

    # Samples
    alloy_samples = []
    for sname, text in ctx.samples.items():
        label = sname.replace(".txt", "").replace("_", " ").title()
        alloy_samples.append({
            "label": label,
            "prompt": "(generation sample)",
            "completion": text.strip()[:2000],
        })

    # Model hash (full file, not partial)
    model_hash = ""
    model_dir = out / "model"
    if model_dir.exists():
        safetensors = sorted(model_dir.glob("*.safetensors"))
        if safetensors:
            h = hashlib.sha256()
            for sf in safetensors:
                with open(sf, 'rb') as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        h.update(chunk)
            model_hash = f"sha256:{h.hexdigest()}"

    script_hash = f"sha256:{hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()}"

    # Capture git commit of the runner repo
    import subprocess
    try:
        repo_dir = str(Path(__file__).resolve().parent.parent)
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=repo_dir, text=True
        ).strip()
        git_remote = subprocess.check_output(
            ['git', 'remote', 'get-url', 'origin'], cwd=repo_dir, text=True
        ).strip()
    except Exception:
        git_commit = None
        git_remote = None

    alloy["results"] = {
        "completedAt": results.get("forged_at", ""),
        "baselinePerplexity": results.get("baseline_ppl"),
        "finalPerplexity": results.get("final_ppl"),
        "improvementPct": results.get("improvement_pct"),
        "benchmarks": [
            {"name": "perplexity", "metrics": {
                "baseline": results.get("baseline_ppl", 0),
                "final": results.get("final_ppl", 0),
                "improvement": results.get("improvement_pct", 0),
            }}
        ] + ctx.eval_results,
        "hardwareVerified": [{
            "device": results.get("device", "unknown"),
            "format": "fp16" if not ctx.load_4bit else "4-bit",
            "verified": True,
        }],
        "samples": alloy_samples,
        "integrity": {
            "trustLevel": "self-attested",
            "code": {
                "runner": "sentinel-ai/alloy_executor",
                "version": "1.0.0",
                "binaryHash": script_hash,
                **({"commit": git_commit} if git_commit else {}),
                **({"sourceRepo": git_remote} if git_remote else {}),
            },
            "modelHash": model_hash,
            "datasets": [],
            "attestedAt": results.get("forged_at", ""),
        },
    }

    alloy_path = out / f"{alloy.get('name', 'unnamed')}.alloy.json"
    alloy_path.write_text(json.dumps(alloy, indent=2))
    print(f"  Alloy: {alloy_path}")

    # Generate QR code linking to verification URL
    alloy_hash = hashlib.sha256(alloy_path.read_bytes()).hexdigest()
    # URL carries the hash — verifier checks alloy file matches the hash in the QR
    verify_url = f"https://huggingface.co/continuum-ai/{alloy.get('name', 'model')}/blob/main/{alloy_path.name}#sha256:{alloy_hash[:16]}"
    try:
        import qrcode
        qr = qrcode.make(verify_url)
        qr_path = out / "alloy-qr.png"
        qr.save(str(qr_path))
        print(f"  QR: {qr_path} → {verify_url}")
    except ImportError:
        print(f"  QR: skipped (pip install qrcode[pil])")


def main():
    parser = argparse.ArgumentParser(description="Execute a ForgeAlloy pipeline")
    parser.add_argument("alloy", help="Path to .alloy.json file")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    execute_alloy(args.alloy, args.output_dir, args.dry_run)


if __name__ == "__main__":
    main()
