#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone program to run pruning and fine-tuning experiments using the modular framework.

This script can be run from the command line with various arguments to configure
the experiment. It uses the PruningFineTuningExperiment class to handle all the
experiment logic.

Example usage:
    python run_experiment.py --model distilgpt2 --strategies random entropy --pruning_levels 0.1 0.3 0.5
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).parents[3].absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import experiment framework
from utils.pruning import PruningFineTuningExperiment, Environment


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run pruning and fine-tuning experiments")
    
    # Model selection
    parser.add_argument("--model", type=str, default=None,
                       help="Specific model to test, or None to automatically select based on environment")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                       help="List of models to test, or None to automatically select based on environment")
                       
    # Experiment parameters
    parser.add_argument("--strategies", type=str, nargs="+", default=["random", "entropy"],
                       help="List of pruning strategies to use (default: random entropy)")
    parser.add_argument("--pruning_levels", type=float, nargs="+", default=[0.1, 0.3, 0.5],
                       help="List of pruning levels to use (default: 0.1 0.3 0.5)")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of fine-tuning epochs (default: 1)")
    parser.add_argument("--prompt", type=str, 
                        default="Artificial intelligence will transform society by",
                        help="Prompt to use for evaluation")
    parser.add_argument("--runtime", type=int, default=3600,
                        help="Maximum runtime in seconds (default: 3600 - 1 hour)")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="experiment_results",
                       help="Directory to save results and plots")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Start timing
    start_time = time.time()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Print system information
    env = Environment()
    env.print_info()
    
    # Create experiment
    experiment = PruningFineTuningExperiment(
        results_dir=str(output_dir),
        use_improved_fine_tuner=True,  # Use improved fine-tuner for better stability
        detect_environment=True,       # Automatically detect hardware capabilities
        optimize_memory=True           # Optimize based on model size and hardware
    )
    
    # Set model(s) to use
    models_to_use = None
    if args.model:
        models_to_use = [args.model]
    elif args.models:
        models_to_use = args.models
    # If neither is specified, use automatically detected models
    
    # Run the experiment
    logger.info(f"Starting experiment with:")
    logger.info(f"- Strategies: {args.strategies}")
    logger.info(f"- Pruning levels: {args.pruning_levels}")
    logger.info(f"- Fine-tuning epochs: {args.epochs}")
    if models_to_use:
        logger.info(f"- Models: {models_to_use}")
    else:
        logger.info("- Using automatically detected models based on hardware")
    logger.info(f"- Maximum runtime: {args.runtime} seconds")
    
    results = experiment.run_experiment(
        strategies=args.strategies,
        pruning_levels=args.pruning_levels,
        prompt=args.prompt,
        fine_tuning_epochs=args.epochs,
        max_runtime=args.runtime,
        models=models_to_use
    )
    
    # Print summary
    elapsed_time = time.time() - start_time
    logger.info(f"Experiment completed in {elapsed_time/60:.2f} minutes")
    logger.info(f"Completed {len(results)} experiment configurations")
    
    # Create summary table
    try:
        import pandas as pd
        
        if not experiment.results_df.empty:
            # Get data for different stages
            baseline_df = experiment.results_df[experiment.results_df["stage"] == "baseline"][
                ["model", "strategy", "pruning_level", "perplexity"]]
            baseline_df = baseline_df.rename(columns={"perplexity": "baseline_perplexity"})
            
            pruned_df = experiment.results_df[experiment.results_df["stage"] == "pruned"][
                ["model", "strategy", "pruning_level", "perplexity"]]
            pruned_df = pruned_df.rename(columns={"perplexity": "pruned_perplexity"})
            
            fine_tuned_df = experiment.results_df[experiment.results_df["stage"] == "fine_tuned"][
                ["model", "strategy", "pruning_level", "perplexity"]]
            fine_tuned_df = fine_tuned_df.rename(columns={"perplexity": "fine_tuned_perplexity"})
            
            # Merge dataframes
            summary = pd.merge(baseline_df, pruned_df, on=["model", "strategy", "pruning_level"], how="outer")
            summary = pd.merge(summary, fine_tuned_df, on=["model", "strategy", "pruning_level"], how="outer")
            
            # Calculate changes
            summary["pruning_effect"] = summary["pruned_perplexity"] - summary["baseline_perplexity"]
            summary["fine_tuning_effect"] = summary["fine_tuned_perplexity"] - summary["pruned_perplexity"]
            summary["net_change"] = summary["fine_tuned_perplexity"] - summary["baseline_perplexity"]
            
            # Save summary to CSV
            summary_path = output_dir / "experiment_summary.csv"
            summary.to_csv(summary_path, index=False)
            logger.info(f"Summary saved to {summary_path}")
            
            # Print summary statistics
            logger.info("\nExperiment Summary Statistics:")
            logger.info(f"Total experiments completed: {len(summary)}")
            
            # Count improvements vs. degradations
            improved = summary[summary["net_change"] < 0]
            degraded = summary[summary["net_change"] > 0]
            logger.info(f"Experiments with improved perplexity: {len(improved)}")
            logger.info(f"Experiments with degraded perplexity: {len(degraded)}")
            
            # Calculate average recovery
            recovery_df = experiment.results_df[experiment.results_df["stage"] == "fine_tuned"]
            if "recovery_percentage" in recovery_df.columns and not recovery_df["recovery_percentage"].empty:
                avg_recovery = recovery_df["recovery_percentage"].mean()
                logger.info(f"Average recovery percentage: {avg_recovery:.2f}%")
    except Exception as e:
        logger.error(f"Error creating summary table: {e}")
    
    # Generate and save final plots
    try:
        import matplotlib.pyplot as plt
        
        # Plot results
        fig = experiment.plot_results(figsize=(12, 10))
        
        # Save the plot
        plot_path = output_dir / "experiment_results.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {plot_path}")
    except Exception as e:
        logger.error(f"Error generating plot: {e}")
    
    logger.info("Experiment complete!")


if __name__ == "__main__":
    main()