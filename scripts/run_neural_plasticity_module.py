#!/usr/bin/env python
"""
Neural Plasticity Experiment Runner

This script runs the neural plasticity experiment using our shared modular implementation.
It can be used in both Colab and local environments with the same code base.

Version: v0.0.56 (2025-04-19 23:30:00)
"""

import os
import sys
import argparse
import torch
import platform
from pathlib import Path
from datetime import datetime

# Set environment variables for improved stability
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for Apple Silicon
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.processor() == "arm"

# Import our experiment module
from utils.neural_plasticity.experiment import run_neural_plasticity_experiment

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Neural Plasticity Experiment')
    
    parser.add_argument('--model_name', type=str, default='distilgpt2',
                        help='Name of the model to use')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (default: timestamped directory)')
    parser.add_argument('--pruning_strategy', type=str, default='entropy',
                        choices=['entropy', 'magnitude', 'random', 'combined'],
                        help='Strategy for pruning')
    parser.add_argument('--prune_percent', type=float, default=0.2,
                        help='Percentage of heads to prune (0-1)')
    parser.add_argument('--warmup_steps', type=int, default=10,
                        help='Number of warmup steps for baseline')
    parser.add_argument('--training_steps', type=int, default=50,
                        help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--minimal', action='store_true',
                        help='Run with minimal settings for testing')
    parser.add_argument('--colab', action='store_true',
                        help='Running in Colab environment')
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        if IS_APPLE_SILICON:
            device = 'cpu'
            print(f"🍎 Apple Silicon detected, forcing CPU usage")
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Override settings for minimal run
    if args.minimal:
        print("Running with minimal settings for testing...")
        args.warmup_steps = 5
        args.training_steps = 20
        args.batch_size = 2
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.output_dir is None:
        args.output_dir = f"test_output/neural_plasticity_{args.model_name}_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print experiment settings
    print(f"\n=== Neural Plasticity Experiment ===")
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Pruning strategy: {args.pruning_strategy}")
    print(f"Pruning level: {args.prune_percent:.2f}")
    print(f"Training steps: {args.training_steps}")
    print(f"Output directory: {args.output_dir}")
    
    # Run the experiment
    results = run_neural_plasticity_experiment(
        model_name=args.model_name,
        device=device,
        output_dir=args.output_dir,
        pruning_strategy=args.pruning_strategy,
        prune_percent=args.prune_percent,
        warmup_steps=args.warmup_steps,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        colab_mode=args.colab
    )
    
    print(f"\nNeural Plasticity Experiment completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())