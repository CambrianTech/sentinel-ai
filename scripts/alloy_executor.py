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
OUTPUT_TYPES = {"quant", "package", "eval", "publish", "deploy"}


def execute_alloy(alloy_path: str, output_dir: str = None, dry_run: bool = False):
    """Execute a complete ForgeAlloy pipeline."""
    alloy = json.loads(Path(alloy_path).read_text())
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

    # Load model
    print("[1] Loading model...")
    import torch
    from forge_model import load_model, get_model_info, evaluate, make_dataloaders, ForgeConfig, generate_samples

    ctx.info = get_model_info(model_name)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    cfg = ForgeConfig.auto(ctx.info["fp16_gb"], vram_gb)
    ctx.tier = cfg.tier
    ctx.load_4bit = cfg.load_4bit
    ctx.device = torch.cuda.get_device_name(0)
    ctx.model, ctx.tokenizer = load_model(model_name, cfg.load_4bit)

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
    for cycle in range(1, cycles + 1):
        print(f"\n[4.{cycle}] Cycle {cycle}/{cycles}")
        for stage in transform_stages:
            ctx = create_executor(stage).execute(ctx)

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
            },
            "modelHash": model_hash,
            "datasets": [],
            "attestedAt": results.get("forged_at", ""),
        },
    }

    alloy_path = out / f"{alloy.get('name', 'unnamed')}.alloy.json"
    alloy_path.write_text(json.dumps(alloy, indent=2))
    print(f"  Alloy: {alloy_path}")


def main():
    parser = argparse.ArgumentParser(description="Execute a ForgeAlloy pipeline")
    parser.add_argument("alloy", help="Path to .alloy.json file")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    execute_alloy(args.alloy, args.output_dir, args.dry_run)


if __name__ == "__main__":
    main()
