#!/usr/bin/env python3
"""
forge_pipeline.py — One command: base model → forged + defragged + quantized + published.

Usage:
    python scripts/forge_pipeline.py Qwen/Qwen3.5-4B --domain code --target macbook-32gb
    python scripts/forge_pipeline.py Qwen/Qwen3.5-27B --domain code --target macbook-32gb --publish
    python scripts/forge_pipeline.py Qwen/Qwen3.5-27B --domain code --target rtx-3090
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

# Target hardware profiles
TARGETS = {
    "macbook-air-16gb": {"max_gb": 8, "quantize": "mlx-4bit", "format": "MLX"},
    "macbook-32gb": {"max_gb": 15, "quantize": "mlx-4bit", "format": "MLX"},
    "rtx-3090": {"max_gb": 20, "quantize": None, "format": "bf16"},
    "rtx-5090": {"max_gb": 28, "quantize": None, "format": "bf16"},
    "any-gpu": {"max_gb": 999, "quantize": None, "format": "fp16"},
}


def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"FAILED: {desc}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Full forge pipeline: train → defrag → quantize → publish")
    parser.add_argument("model", help="HuggingFace model ID")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--target", default="macbook-32gb", choices=list(TARGETS.keys()))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--prune-groups", type=int, default=1, help="GQA groups to remove per self_attn layer")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--org", default="continuum-ai")
    args = parser.parse_args()

    slug = args.model.split("/")[-1].lower()
    target = TARGETS[args.target]
    base_dir = Path(f"output/pipeline/{slug}-{args.domain}")
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  FORGE PIPELINE")
    print(f"  Model: {args.model}")
    print(f"  Domain: {args.domain}")
    print(f"  Target: {args.target} ({target['max_gb']}GB, {target['format']})")
    print(f"  Prune: {args.prune_groups} GQA group(s) per layer")
    print(f"{'#'*60}")

    scripts = Path(__file__).parent

    # Step 1: Forge (LoRA train)
    forge_dir = base_dir / "forged"
    run(f"python3 -u {scripts}/forge_model.py {args.model} --domain {args.domain} "
        f"--prune-level 0.0 --steps {args.steps} --cycles 1 "
        f"--output-dir {forge_dir}",
        f"Step 1: LoRA training on {args.domain} data ({args.steps} steps)")

    # Step 2: Uniform defrag
    run(f"python3 -u {scripts}/defrag_forged.py {forge_dir}/ --prune-groups {args.prune_groups}",
        f"Step 2: Structural defrag ({args.prune_groups} group(s) per layer)")

    # Step 3: Quantize if needed
    defrag_dir = base_dir / "defragged"
    if target["quantize"] == "mlx-4bit":
        run(f"python3 -c \"from mlx_lm import convert; convert('{defrag_dir}/model', '{base_dir}/mlx-4bit', quantize=True, q_bits=4, q_group_size=64)\"",
            f"Step 3: MLX 4-bit quantization")

    # Step 4: Publish
    if args.publish:
        repo = f"{args.org}/{slug}-{args.domain}-forged-defragged"
        # TODO: publish_forged.py with the right paths
        print(f"\nReady to publish: {repo}")

    print(f"\n{'#'*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Output: {base_dir}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
