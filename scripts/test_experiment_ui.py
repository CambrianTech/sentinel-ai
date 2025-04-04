#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the modular experiment UI framework.

This script verifies that the ModularExperimentRunner works correctly
and can run experiments programmatically.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the module to test
from utils.colab.experiment_ui import ModularExperimentRunner


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test modular experiment UI framework")
    
    # Model selection
    parser.add_argument("--model", type=str, default="distilgpt2",
                       help="Model to test (default: distilgpt2)")
    parser.add_argument("--model_size", type=str, default="tiny",
                       choices=["tiny", "small", "medium", "large", "xl"],
                       help="Model size category")
                       
    # Experiment parameters
    parser.add_argument("--pruning_strategy", type=str, default="entropy",
                       help="Pruning strategy to use (default: entropy)")
    parser.add_argument("--pruning_level", type=float, default=0.3,
                       help="Pruning level to use (default: 0.3)")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of fine-tuning epochs")
    
    # Adaptive options
    parser.add_argument("--adaptive", action="store_true",
                       help="Test adaptive plasticity")
    parser.add_argument("--plasticity_level", type=float, default=0.2,
                       help="Initial plasticity level for adaptive experiments")
    parser.add_argument("--max_cycles", type=int, default=2,
                       help="Maximum number of neural plasticity cycles")
    
    # Test options
    parser.add_argument("--output_dir", type=str, default="experiments/results/test_ui",
                       help="Directory to save results and plots")
    parser.add_argument("--prompt", type=str, 
                        default="Artificial intelligence will transform society by",
                        help="Prompt to use for evaluation")
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Creating ModularExperimentRunner")
    # Create experiment runner
    runner = ModularExperimentRunner()
    
    # Configure experiment
    config = {
        "model": args.model,
        "model_size": args.model_size,
        "enable_pruning": True,
        "pruning_strategy": args.pruning_strategy,
        "pruning_level": args.pruning_level,
        "enable_fine_tuning": args.epochs > 0,
        "fine_tuning_epochs": args.epochs,
        "enable_adaptive_plasticity": args.adaptive,
        "plasticity_level": args.plasticity_level,
        "max_cycles": args.max_cycles,
        "prompt": args.prompt,
        "results_dir": str(output_dir),
        "enable_visualization": True,
        "save_results": True
    }
    
    # Update runner configuration
    logger.info(f"Configuring experiment with: {config}")
    runner.update_config(**config)
    
    # Create experiment instance
    logger.info("Creating experiment")
    experiment = runner.create_experiment()
    
    # Run experiment
    logger.info("Running experiment")
    results = runner.run_experiment()
    
    # Print results summary
    logger.info("\nExperiment Results:")
    
    if isinstance(results, dict) and "success" in results:
        # Adaptive plasticity experiment
        if results["success"]:
            logger.info(f"Experiment completed successfully")
            if "final_perplexity" in results:
                logger.info(f"Final perplexity: {results['final_perplexity']:.4f}")
            if "final_head_count" in results:
                logger.info(f"Final head count: {results['final_head_count']}")
            if "cycles_completed" in results:
                logger.info(f"Cycles completed: {results['cycles_completed']}")
        else:
            logger.error(f"Experiment failed: {results.get('error', 'Unknown error')}")
            
    elif isinstance(results, dict) and "pruning_results" in results:
        # Standard pruning/fine-tuning experiment
        for model, model_results in results["pruning_results"].items():
            for strategy, strategy_results in model_results.items():
                for level, level_results in strategy_results.items():
                    logger.info(f"Model: {model}, Strategy: {strategy}, Level: {level}")
                    if "initial_perplexity" in level_results:
                        logger.info(f"  Initial perplexity: {level_results['initial_perplexity']:.4f}")
                    if "pruned_perplexity" in level_results:
                        logger.info(f"  Pruned perplexity: {level_results['pruned_perplexity']:.4f}")
                    if "fine_tuned_perplexity" in level_results:
                        logger.info(f"  Fine-tuned perplexity: {level_results['fine_tuned_perplexity']:.4f}")
                        
    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()