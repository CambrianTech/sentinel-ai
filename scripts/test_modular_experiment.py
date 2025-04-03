#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the modular experiment framework for pruning and fine-tuning.

This script verifies that the new modular experiment framework works correctly
and is compatible with the existing codebase.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import time
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import custom modules
from utils.pruning.experiment import PruningExperiment, PruningFineTuningExperiment
from utils.pruning.environment import Environment


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test modular experiment framework")
    
    # Model selection
    parser.add_argument("--model", type=str, default="distilgpt2",
                       help="Model to test (default: distilgpt2)")
                       
    # Experiment parameters
    parser.add_argument("--strategy", type=str, default="random",
                       help="Pruning strategy to use (default: random)")
    parser.add_argument("--pruning_level", type=float, default=0.3,
                       help="Pruning level to use (default: 0.3)")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of fine-tuning epochs")
    
    # Test options
    parser.add_argument("--full_experiment", action="store_true",
                       help="Run a full multi-model, multi-strategy experiment")
    parser.add_argument("--output_dir", type=str, default="test_output",
                       help="Directory to save results and plots")
    parser.add_argument("--prompt", type=str, 
                        default="Artificial intelligence will transform society by",
                        help="Prompt to use for evaluation")
    
    return parser.parse_args()


def test_single_experiment(args):
    """Test a single experiment with the PruningExperiment class"""
    logger.info(f"Testing single experiment with model: {args.model}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create experiment
    experiment = PruningExperiment(
        results_dir=str(output_dir),
        use_improved_fine_tuner=True,
        detect_environment=True,
        optimize_memory=True
    )
    
    # Print environment details
    logger.info(f"Hardware detection: GPU memory = {experiment.gpu_memory_gb:.1f} GB")
    
    # Run experiment
    result = experiment.run_single_experiment(
        model=args.model,
        strategy=args.strategy,
        pruning_level=args.pruning_level,
        prompt=args.prompt,
        fine_tuning_epochs=args.epochs,
        save_results=True
    )
    
    # Print summary
    logger.info("\nExperiment Summary:")
    
    baseline_perplexity = result["stages"]["baseline"]["perplexity"]
    logger.info(f"Baseline perplexity: {baseline_perplexity:.4f}")
    
    pruned_perplexity = result["stages"]["pruned"]["perplexity"]
    logger.info(f"Pruned perplexity: {pruned_perplexity:.4f}")
    
    if "fine_tuned" in result["stages"]:
        fine_tuned_perplexity = result["stages"]["fine_tuned"]["perplexity"]
        logger.info(f"Fine-tuned perplexity: {fine_tuned_perplexity:.4f}")
        
        # Calculate improvement
        if "recovery_percentage" in result["stages"]["fine_tuned"]:
            recovery = result["stages"]["fine_tuned"]["recovery_percentage"]
            logger.info(f"Recovery percentage: {recovery:.2f}%")
        elif "improvement_percentage" in result["stages"]["fine_tuned"]:
            improvement = result["stages"]["fine_tuned"]["improvement_percentage"]
            logger.info(f"Improvement percentage: {improvement:.2f}%")
    
    # Plot results
    logger.info("Plotting results...")
    fig = experiment.plot_results()
    
    # Save the plot
    plot_path = output_dir / f"single_experiment_{args.model.replace('/', '_')}_{args.strategy}_{args.pruning_level}.png"
    fig.savefig(plot_path)
    logger.info(f"Plot saved to {plot_path}")
    
    return result


def test_multi_experiment(args):
    """Test a multi-experiment run with the PruningFineTuningExperiment class"""
    logger.info("Testing multi-experiment framework")
    
    # Create output directory
    output_dir = Path(args.output_dir) / "multi_experiment"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create experiment
    experiment = PruningFineTuningExperiment(
        results_dir=str(output_dir),
        use_improved_fine_tuner=True,
        detect_environment=True,
        optimize_memory=True
    )
    
    # Define models, strategies and pruning levels
    models = ["distilgpt2", "gpt2"]
    if experiment.gpu_memory_gb >= 8:
        models.append("facebook/opt-350m")
    
    strategies = ["random", "attention"]
    pruning_levels = [0.1, 0.3, 0.5]
    
    # Update model size limits (add more if needed)
    experiment.model_size_limits.update({
        "distilgpt2": 1.0,
        "gpt2": 1.0,
        "facebook/opt-350m": 0.5,
    })
    
    # Set max runtime to 20 minutes
    max_runtime = 1200  # 20 minutes
    
    # Run experiments
    logger.info(f"Running {len(models) * len(strategies) * len(pruning_levels)} experiments")
    logger.info(f"Models: {models}")
    logger.info(f"Strategies: {strategies}")
    logger.info(f"Pruning levels: {pruning_levels}")
    
    results = experiment.run_experiment(
        strategies=strategies,
        pruning_levels=pruning_levels,
        prompt=args.prompt,
        fine_tuning_epochs=args.epochs,
        max_runtime=max_runtime,
        models=models
    )
    
    # Plot final results
    logger.info("Plotting final results...")
    fig = experiment.plot_results(figsize=(16, 14))
    
    # Save the plot
    plot_path = output_dir / "multi_experiment_results.png"
    plt.savefig(plot_path)
    logger.info(f"Plot saved to {plot_path}")
    
    return results


def main():
    """Main entry point"""
    args = parse_args()
    
    # Print system information
    env = Environment()
    env.print_info()
    
    start_time = time.time()
    
    if args.full_experiment:
        # Test multiple experiments
        results = test_multi_experiment(args)
        logger.info(f"Completed {len(results)} experiments")
    else:
        # Test single experiment
        result = test_single_experiment(args)
        
    elapsed_time = time.time() - start_time
    logger.info(f"Total runtime: {elapsed_time/60:.2f} minutes")


if __name__ == "__main__":
    main()