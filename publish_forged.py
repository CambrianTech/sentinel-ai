#!/usr/bin/env python3
"""
Publish a forged model to HuggingFace.

Usage:
    python publish_forged.py output/forged/qwen-qwen2.5-7b/
    python publish_forged.py output/forged/qwen-qwen2.5-7b/ --org continuum-ai
    python publish_forged.py --all --org continuum-ai
"""

import argparse
import json
import os
import sys
from pathlib import Path


def publish_model(forged_dir: Path, org: str, dry_run: bool = False):
    """Publish a single forged model to HuggingFace."""
    results_path = forged_dir / "results.json"
    card_path = forged_dir / "model_card.md"

    if not results_path.exists():
        print(f"  SKIP: {forged_dir} — no results.json")
        return False

    results = json.load(open(results_path))
    model_name = results.get("model", "")
    slug = model_name.replace("/", "-").lower()
    repo_id = f"{org}/{slug}-forged"

    improvement = results.get("improvement_pct", 0)
    baseline = results.get("baseline_ppl", 0)
    final = results.get("final_ppl", 0)

    print(f"\n  Model: {model_name}")
    print(f"  Repo: {repo_id}")
    print(f"  Baseline PPL: {baseline:.2f} → Final: {final:.2f} ({improvement:+.1f}%)")

    if dry_run:
        print(f"  DRY RUN: would upload to {repo_id}")
        return True

    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    # Create repo
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"  Created/verified repo: {repo_id}")
    except Exception as e:
        print(f"  ERROR creating repo: {e}")
        return False

    # Upload model card
    if card_path.exists():
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
        )
        print(f"  Uploaded model card")

    # Upload results
    api.upload_file(
        path_or_fileobj=str(results_path),
        path_in_repo="forging_results.json",
        repo_id=repo_id,
    )
    print(f"  Uploaded results.json")

    # Upload figures
    figures_dir = forged_dir / "figures"
    if figures_dir.exists():
        for fig in figures_dir.glob("*.png"):
            api.upload_file(
                path_or_fileobj=str(fig),
                path_in_repo=f"figures/{fig.name}",
                repo_id=repo_id,
            )
        print(f"  Uploaded {len(list(figures_dir.glob('*.png')))} figures")

    # Upload benchmark samples
    bench_dir = forged_dir / "benchmark"
    if bench_dir.exists():
        for txt in bench_dir.glob("*.txt"):
            api.upload_file(
                path_or_fileobj=str(txt),
                path_in_repo=f"benchmark/{txt.name}",
                repo_id=repo_id,
            )
        print(f"  Uploaded benchmark samples")

    print(f"  PUBLISHED: https://huggingface.co/{repo_id}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Publish forged models to HuggingFace")
    parser.add_argument("forged_dir", nargs="?", help="Path to forged model output dir")
    parser.add_argument("--org", default="continuum-ai", help="HuggingFace org")
    parser.add_argument("--all", action="store_true", help="Publish all forged models")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be published")
    args = parser.parse_args()

    if args.all:
        forged_root = Path("output/forged")
        if not forged_root.exists():
            print("No output/forged/ directory found")
            sys.exit(1)

        published = 0
        for d in sorted(forged_root.iterdir()):
            if d.is_dir() and (d / "results.json").exists():
                if publish_model(d, args.org, args.dry_run):
                    published += 1

        print(f"\n{'=' * 50}")
        print(f"Published {published} models to {args.org}")
    elif args.forged_dir:
        publish_model(Path(args.forged_dir), args.org, args.dry_run)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
