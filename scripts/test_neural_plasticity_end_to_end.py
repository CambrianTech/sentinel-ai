#!/usr/bin/env python
"""
Neural Plasticity End-to-End Test

This script runs a full neural plasticity experiment end-to-end with minimal
configuration to verify that all components work correctly together.

Usage:
    python scripts/test_neural_plasticity_end_to_end.py [--output_dir OUTPUT_DIR] [--steps STEPS]

Version: v0.0.61 (2025-04-20 15:00:00)
"""

import os
import sys
import shutil
import argparse
import platform
from datetime import datetime

# Add parent directory to path to ensure imports work when script is run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set environment variables for improved stability
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['TORCH_USE_MKL_FFT'] = '0'

IS_APPLE_SILICON = platform.system() == "Darwin" and platform.processor() == "arm"

try:
    import torch
    import numpy as np
    from utils.neural_plasticity.experiment import NeuralPlasticityExperiment
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Error importing neural plasticity modules: {e}")
    print("This script requires PyTorch and the neural plasticity modules to run.")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run neural plasticity end-to-end test")
    
    parser.add_argument('--output_dir', type=str, default='test_output/neural_plasticity_test',
                        help='Directory to save test outputs')
    parser.add_argument('--steps', type=int, default=5,
                        help='Number of training steps per phase (warmup, pruning)')
    parser.add_argument('--model', type=str, default='distilgpt2',
                        help='Model to use for testing')
    parser.add_argument('--clean', action='store_true',
                        help='Clean output directory before starting')
    parser.add_argument('--dataset', type=str, default='wikitext',
                        help='Dataset to use')
    parser.add_argument('--dataset_config', type=str, default='wikitext-2-raw-v1',
                        help='Dataset configuration')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training and evaluation')
    parser.add_argument('--max_length', type=int, default=64,
                        help='Maximum sequence length')
    parser.add_argument('--pruning_level', type=float, default=0.2,
                        help='Pruning level (0-1)')
    parser.add_argument('--pruning_strategy', type=str, default='combined',
                        choices=['entropy', 'gradient', 'random', 'combined'],
                        help='Pruning strategy to use')
    parser.add_argument('--show_samples', action='store_true',
                        help='Show sample text and predictions during training')
    parser.add_argument('--sample_interval', type=int, default=10,
                        help='Interval for showing sample predictions')
    
    return parser.parse_args()


def run_test(args):
    """Run the neural plasticity end-to-end test with the given arguments."""
    # Print test configuration
    print(f"Neural Plasticity End-to-End Test")
    print(f"--------------------------------")
    print(f"Platform: {platform.system()} {platform.processor()}")
    print(f"Apple Silicon detected: {IS_APPLE_SILICON}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}/{args.dataset_config}")
    print(f"Steps: {args.steps}")
    print(f"Output directory: {args.output_dir}")
    print(f"Pruning: {args.pruning_level*100:.1f}% with {args.pruning_strategy} strategy")
    print(f"Sample display: {'Enabled' if args.show_samples else 'Disabled'}{f' (interval: {args.sample_interval})' if args.show_samples else ''}")
    print(f"--------------------------------")
    
    # Create or clean output directory
    if args.clean and os.path.exists(args.output_dir):
        print(f"Cleaning output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer first if we're showing samples
    tokenizer = None
    if args.show_samples:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer for sample display...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize experiment
    print(f"Initializing experiment...")
    experiment = NeuralPlasticityExperiment(
        model_name=args.model,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pruning_level=args.pruning_level,
        pruning_strategy=args.pruning_strategy,
        learning_rate=5e-5,
        verbose=True,
        save_results=True,
        show_samples=args.show_samples,
        sample_interval=args.sample_interval,
        tokenizer=tokenizer
    )
    
    # Run experiment phases
    print(f"Setting up experiment...")
    experiment.setup()
    
    print(f"Running warmup phase...")
    warmup_results = experiment.run_warmup(max_epochs=1, max_steps=args.steps)
    
    print(f"Analyzing attention patterns...")
    attention_analysis = experiment.analyze_attention()
    
    print(f"Running pruning cycle...")
    pruning_results = experiment.run_pruning_cycle(training_steps=args.steps)
    
    print(f"Evaluating model...")
    eval_metrics = experiment.evaluate()
    
    # Display key metrics
    print(f"\nFinal Results:")
    print(f"Baseline perplexity: {experiment.baseline_perplexity:.2f}")
    print(f"Final perplexity: {experiment.final_perplexity:.2f}")
    print(f"Improvement: {eval_metrics['improvement_percent']:.2f}%")
    print(f"Pruned heads: {len(experiment.pruned_heads)} out of {attention_analysis['model_structure'][0] * attention_analysis['model_structure'][1]}")
    
    # List generated visualizations
    print(f"\nGenerated visualization directories:")
    subdirs = [d for d in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, d))]
    for subdir in sorted(subdirs):
        png_count = len([f for f in os.listdir(os.path.join(args.output_dir, subdir)) if f.endswith('.png')])
        if png_count > 0:
            print(f"- {subdir}/: {png_count} visualizations")
    
    print("\nTest completed successfully!")
    return eval_metrics


def main():
    """Main function to run the test."""
    args = parse_args()
    start_time = datetime.now()
    
    try:
        metrics = run_test(args)
        
        # Write success summary
        duration = datetime.now() - start_time
        with open(os.path.join(args.output_dir, "test_summary.txt"), "w") as f:
            f.write(f"Neural Plasticity End-to-End Test\n")
            f.write(f"--------------------------------\n")
            f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {duration.total_seconds():.1f} seconds\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Dataset: {args.dataset}/{args.dataset_config}\n")
            f.write(f"Pruning: {args.pruning_level*100:.1f}% with {args.pruning_strategy} strategy\n")
            f.write(f"Sample display: {'Enabled' if args.show_samples else 'Disabled'}{f' (interval: {args.sample_interval})' if args.show_samples else ''}\n\n")
            f.write(f"Results:\n")
            f.write(f"- Baseline perplexity: {metrics['baseline_perplexity']:.2f}\n")
            f.write(f"- Final perplexity: {metrics['perplexity']:.2f}\n")
            f.write(f"- Improvement: {metrics['improvement_percent']:.2f}%\n")
            f.write(f"- Pruned heads: {metrics['num_pruned_heads']}\n")
            
        print(f"Test summary written to: {os.path.join(args.output_dir, 'test_summary.txt')}")
        print(f"Total duration: {duration.total_seconds():.1f} seconds")
        
        return 0
    except Exception as e:
        # Write error information
        with open(os.path.join(args.output_dir, "test_error.log"), "w") as f:
            f.write(f"Neural Plasticity Test Error\n")
            f.write(f"------------------------\n")
            f.write(f"Error occurred at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Error message: {str(e)}\n\n")
            
            import traceback
            f.write(traceback.format_exc())
        
        print(f"ERROR: Test failed with exception: {e}")
        print(f"Error details written to: {os.path.join(args.output_dir, 'test_error.log')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())