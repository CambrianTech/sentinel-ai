#!/usr/bin/env python
"""
Run Transformer Pruning Experiments

This script provides a command-line interface for running pruning and optimization
experiments with transformer models, using the sentinel.upgrayedd package.

Examples:
    Run with default settings:
    $ python run_experiment.py
    
    Run with custom model and dataset:
    $ python run_experiment.py --model_name distilgpt2 --dataset wikitext --pruning_ratio 0.3
    
    Run with config file:
    $ python run_experiment.py --config configs/entropy_pruning.json
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import torch

# Add parent directory to path to ensure imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import the adaptive optimizer
from sentinel.upgrayedd.optimizer.adaptive_optimizer import AdaptiveOptimizer, AdaptiveOptimizerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run transformer pruning experiments")
    
    # Config file
    parser.add_argument(
        "--config", type=str,
        help="Path to JSON config file"
    )
    
    # Model parameters
    parser.add_argument(
        "--model_name", type=str, default="distilgpt2",
        help="Model name or path (default: distilgpt2)"
    )
    parser.add_argument(
        "--cache_dir", type=str,
        help="Directory to cache models"
    )
    
    # Pruning parameters
    parser.add_argument(
        "--pruning_ratio", type=float, default=0.3,
        help="Ratio of heads to prune (default: 0.3)"
    )
    parser.add_argument(
        "--strategy", type=str, default="entropy", choices=["entropy", "magnitude", "random"],
        help="Pruning strategy to use (default: entropy)"
    )
    parser.add_argument(
        "--growth_ratio", type=float, default=0.1,
        help="Ratio of pruned heads to regrow (default: 0.1)"
    )
    
    # Training parameters
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5,
        help="Learning rate for fine-tuning (default: 5e-5)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for training (default: 4)"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of fine-tuning epochs per cycle (default: 3)"
    )
    parser.add_argument(
        "--max_cycles", type=int, default=1,
        help="Maximum number of optimization cycles (default: 1)"
    )
    parser.add_argument(
        "--grad_accum", type=int, default=1,
        help="Gradient accumulation steps (default: 1)"
    )
    
    # Dataset parameters
    parser.add_argument(
        "--dataset", type=str, default="wikitext",
        help="Dataset to use (default: wikitext)"
    )
    parser.add_argument(
        "--dataset_path", type=str,
        help="Path to custom dataset"
    )
    
    # Output parameters
    parser.add_argument(
        "--output_dir", type=str,
        help="Directory to save output (default: auto-generated based on model and date)"
    )
    parser.add_argument(
        "--save_freq", type=int, default=1,
        help="Save checkpoints every N cycles (default: 1)"
    )
    
    # Other parameters
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available, otherwise cpu)"
    )
    parser.add_argument(
        "--seed", type=int,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--differential_lr", action="store_true",
        help="Use differential learning rates for different parts of the model"
    )
    
    return parser.parse_args()

def load_config_from_file(config_path):
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def create_output_dir(args):
    """Create and return an output directory based on model name and date."""
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Create a directory name based on model and timestamp
        model_name = args.model_name.split('/')[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("./experiments", f"{model_name}_{args.strategy}_{timestamp}")
    
    # Create the directory
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def save_config(config, output_dir):
    """Save configuration to the output directory."""
    config_path = os.path.join(output_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved configuration to {config_path}")

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Create output directory
    output_dir = create_output_dir(args)
    logger.info(f"Output will be saved to: {output_dir}")
    
    # Load config from file or create from args
    if args.config:
        config_dict = load_config_from_file(args.config)
    else:
        # Create config from args
        config_dict = {
            "model_name": args.model_name,
            "cache_dir": args.cache_dir,
            "pruning_ratio": args.pruning_ratio,
            "strategy": args.strategy,
            "growth_ratio": args.growth_ratio,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs_per_cycle": args.epochs,
            "max_cycles": args.max_cycles,
            "gradient_accumulation": args.grad_accum,
            "dataset": args.dataset,
            "dataset_path": args.dataset_path,
            "output_dir": output_dir,
            "save_frequency": args.save_freq,
            "device": args.device,
            "use_differential_lr": args.differential_lr
        }
    
    # Override output_dir in config
    config_dict["output_dir"] = output_dir
    
    # Save config
    save_config(config_dict, output_dir)
    
    # Create optimizer config
    config = AdaptiveOptimizerConfig(**config_dict)
    
    # Create optimizer
    optimizer = AdaptiveOptimizer(config)
    
    # Run optimization
    logger.info("Starting optimization...")
    results = optimizer.run_continuous_optimization(max_cycles=config.max_cycles)
    
    # Log results summary
    if results["improvement"] is not None:
        logger.info(f"Optimization complete. Improvement: {results['improvement']:.2f}%")
    else:
        logger.info("Optimization complete.")
    
    logger.info(f"Results saved to: {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())