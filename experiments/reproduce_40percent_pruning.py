#!/usr/bin/env python
"""
Reproduce the 40% Pruning Experiment from April 2025

This script reproduces the key finding that GPT-2 can be pruned by 40%
with minimal quality impact, demonstrating Sentinel-AI's neural plasticity.

Original experiment showed:
- 30-40% of attention heads can be pruned
- Perplexity: 975 → 211 (78% improvement)
- Maintained performance with significant parameter reduction

This reproduction uses:
- Model: GPT-2 (124M parameters)
- Dataset: TinyShakespeare (simpler) or WikiText-2 (original)
- Strategy: Entropy-based pruning
- Cycles: 3 (warmup → prune → regrow)
"""

import os
import sys
import json
import torch
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_experiment_config(args):
    """Create experiment configuration dictionary."""
    return {
        "experiment_name": "40_percent_pruning_reproduction",
        "date": datetime.now().isoformat(),
        "model": {
            "name": args.model,
            "type": "gpt2",
            "parameters": "124M"
        },
        "dataset": {
            "name": args.dataset,
            "split": "train/validation"
        },
        "pruning": {
            "strategy": "entropy",
            "target_level": 0.4,
            "cycles": args.cycles,
            "regrowth_enabled": True
        },
        "training": {
            "steps_per_cycle": args.training_steps,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_length": args.max_length
        },
        "expected_results": {
            "heads_pruned_percent": "30-40%",
            "perplexity_improvement": "significant",
            "parameter_reduction": "30-40%"
        }
    }


def run_experiment(args):
    """Run the 40% pruning experiment."""

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"experiments/results/pruning_40percent_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("Sentinel-AI: 40% Pruning Reproduction Experiment")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Started: {datetime.now()}")
    logger.info("")

    # Save experiment configuration
    config = create_experiment_config(args)
    config_path = output_dir / "experiment_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration saved: {config_path}")

    # Import and run the neural plasticity experiment
    try:
        # Try to import from the Sentinel-AI utils
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from utils.neural_plasticity.experiment import NeuralPlasticityExperiment

        logger.info("Creating NeuralPlasticityExperiment instance...")

        experiment = NeuralPlasticityExperiment(
            model_name=args.model,
            dataset_name=args.dataset,
            output_dir=str(output_dir),
            pruning_strategy="entropy",
            pruning_level=0.4,
            training_steps=args.training_steps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_length=args.max_length,
            save_model=True,
            verbose=True
        )

        logger.info("")
        logger.info("Phase 1: Baseline Training")
        logger.info("-" * 40)
        baseline_results = experiment.run_baseline()

        logger.info("")
        logger.info("Phase 2: Progressive Pruning (0% → 40%)")
        logger.info("-" * 40)

        results = []
        for cycle in range(args.cycles):
            logger.info(f"\nCycle {cycle + 1}/{args.cycles}")
            logger.info("-" * 40)

            # Prune heads
            logger.info(f"Pruning {(cycle + 1) * 0.4 / args.cycles * 100:.1f}% of heads...")
            prune_results = experiment.prune_heads()

            # Measure impact
            logger.info("Evaluating pruned model...")
            pruned_metrics = experiment.evaluate()

            # Regrow heads
            logger.info("Regrowing heads strategically...")
            regrow_results = experiment.regrow_heads()

            # Final evaluation
            logger.info("Final evaluation after regrowth...")
            final_metrics = experiment.evaluate()

            # Record cycle results
            cycle_results = {
                "cycle": cycle + 1,
                "pruned_perplexity": pruned_metrics.get("perplexity"),
                "regrown_perplexity": final_metrics.get("perplexity"),
                "active_heads": final_metrics.get("active_heads"),
                "total_heads": final_metrics.get("total_heads"),
                "pruning_percent": (1 - final_metrics.get("active_heads", 0) / final_metrics.get("total_heads", 1)) * 100
            }
            results.append(cycle_results)

            logger.info(f"Cycle {cycle + 1} Results:")
            logger.info(f"  Perplexity: {cycle_results['pruned_perplexity']:.2f} → {cycle_results['regrown_perplexity']:.2f}")
            logger.info(f"  Active heads: {cycle_results['active_heads']}/{cycle_results['total_heads']} ({100 - cycle_results['pruning_percent']:.1f}% active)")

        # Save results
        results_path = output_dir / "pruning_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                "baseline": baseline_results,
                "cycles": results,
                "config": config
            }, f, indent=2)

        logger.info("")
        logger.info("="*60)
        logger.info("Experiment Complete!")
        logger.info("="*60)
        logger.info(f"Completed: {datetime.now()}")
        logger.info(f"Results saved: {results_path}")

        # Generate summary
        final_result = results[-1]
        logger.info("")
        logger.info("Summary:")
        logger.info(f"  Baseline perplexity: {baseline_results.get('perplexity', 'N/A')}")
        logger.info(f"  Final perplexity: {final_result['regrown_perplexity']:.2f}")
        logger.info(f"  Heads pruned: {final_result['pruning_percent']:.1f}%")
        logger.info(f"  Parameter reduction: ~{final_result['pruning_percent']:.1f}%")

        # Compare to April 2025 results
        logger.info("")
        logger.info("Comparison to April 2025:")
        logger.info("  Expected: 30-40% prunable, perplexity 975 → 211")
        logger.info(f"  Actual: {final_result['pruning_percent']:.1f}% pruned, perplexity {baseline_results.get('perplexity', '?')} → {final_result['regrown_perplexity']:.2f}")

        return results

    except ImportError as e:
        logger.error(f"Failed to import NeuralPlasticityExperiment: {e}")
        logger.error("Make sure you're running this from the sentinel-ai directory")
        logger.error("and all dependencies are installed:")
        logger.error("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the 40% pruning experiment from April 2025",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--model", type=str, default="gpt2",
                       help="Model to use (gpt2, distilgpt2)")
    parser.add_argument("--dataset", type=str, default="tiny_shakespeare",
                       help="Dataset to use (tiny_shakespeare, wikitext)")
    parser.add_argument("--cycles", type=int, default=3,
                       help="Number of pruning/regrowth cycles")
    parser.add_argument("--training_steps", type=int, default=500,
                       help="Training steps per cycle")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Maximum sequence length")

    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
