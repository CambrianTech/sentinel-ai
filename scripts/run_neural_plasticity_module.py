#!/usr/bin/env python
"""
Neural Plasticity Module Runner

This script runs the modular neural plasticity pipeline that works cross-platform:
- Apple Silicon (M1/M2/M3) 
- Standard CPUs
- GPU environments including Colab

The script provides all the functionality of the original NeuralPlasticityDemo
notebook but with proper modularization, better performance, and cross-platform 
compatibility.

Version: v0.0.60 (2025-04-20)
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Set environment variables for improved stability
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Neural Plasticity Module')
    
    parser.add_argument('--model', type=str, default='distilgpt2',
                        help='Name of the model to use (default: distilgpt2)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use (auto, cpu, cuda) (default: auto)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (default: timestamped directory)')
    parser.add_argument('--strategy', type=str, default='combined',
                        choices=['entropy', 'gradient', 'random', 'combined'],
                        help='Pruning strategy (default: combined)')
    parser.add_argument('--prune_percent', type=float, default=0.2,
                        help='Percentage of heads to prune (0-1) (default: 0.2)')
    parser.add_argument('--warmup_steps', type=int, default=10,
                        help='Number of warmup steps for baseline (default: 10)')
    parser.add_argument('--training_steps', type=int, default=100,
                        help='Number of training steps (default: 100)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training (default: 4)')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate for fine-tuning (default: 5e-5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--minimal', action='store_true',
                        help='Run with minimal settings for quick testing')
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualizations during execution')
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Import neural plasticity modules with improved API
    try:
        from utils.neural_plasticity import NeuralPlasticity
        from utils.neural_plasticity import (
            IS_APPLE_SILICON, IS_COLAB, HAS_GPU,
            run_neural_plasticity_experiment,
            visualize_head_entropy,
            visualize_head_gradients,
            visualize_pruning_decisions,
            visualize_training_metrics
        )
        print("✅ Successfully imported neural plasticity modules")
    except ImportError as e:
        print(f"❌ Error importing modules: {e}")
        sys.exit(1)
    
    # Get environment info
    env_info = NeuralPlasticity.get_environment_info()
    print("=== Environment Information ===")
    print(f"Platform: {env_info['platform']}")
    print(f"Python version: {env_info['python_version']}")
    print(f"PyTorch version: {env_info['pytorch_version']}")
    print(f"Apple Silicon detected: {env_info['is_apple_silicon']}")
    print(f"Colab environment: {env_info['is_colab']}")
    print(f"GPU available: {env_info['has_gpu']}")
    print(f"Default device: {env_info['device']}")
    
    # Set device
    if args.device == 'auto':
        device = env_info['device']
    else:
        device = args.device
        
    # Force CPU on Apple Silicon regardless of requested device
    if IS_APPLE_SILICON and device != 'cpu':
        print(f"🍎 Apple Silicon detected, forcing CPU usage")
        device = 'cpu'
    
    # Override settings for minimal run
    if args.minimal:
        print("Running with minimal settings for testing...")
        args.warmup_steps = 5
        args.training_steps = 20
        args.batch_size = 2
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.output_dir is None:
        args.output_dir = f"test_output/neural_plasticity_{args.model}_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print experiment settings
    print(f"\n=== Neural Plasticity Experiment ===")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Pruning strategy: {args.strategy}")
    print(f"Pruning level: {args.prune_percent:.2f}")
    print(f"Training steps: {args.training_steps}")
    print(f"Output directory: {args.output_dir}")
    
    # Create results tracking callback
    def experiment_callback(stage, step, metrics):
        """Callback for tracking experiment progress."""
        print(f"Stage: {stage}, Step: {step}")
        
        if stage == "pruning" and args.visualize:
            # Create and display pruning visualization if requested
            if "pruning_mask" in metrics and "grad_norm_values" in metrics:
                fig = visualize_pruning_decisions(
                    grad_norm_values=metrics["grad_norm_values"],
                    pruning_mask=metrics["pruning_mask"],
                    title=f"Pruning Decisions ({args.strategy} strategy)",
                    figsize=(12, 8)
                )
                plt.savefig(os.path.join(args.output_dir, f"pruning_decisions.png"))
                plt.show()
                plt.close()
                
        if stage == "training" and args.visualize and step % 20 == 0:
            # Show training metrics periodically if requested
            metrics_data = {
                "train_loss": metrics.get("train_loss", 0),
                "eval_loss": metrics.get("eval_loss", 0),
                "perplexity": metrics.get("perplexity", 0),
            }
            print(f"  Train loss: {metrics_data['train_loss']:.4f}")
            print(f"  Eval loss: {metrics_data['eval_loss']:.4f}")
            print(f"  Perplexity: {metrics_data['perplexity']:.2f}")
        
        if stage == "final":
            # Show final perplexity improvement
            if "perplexity" in metrics:
                print(f"Final evaluation completed: Perplexity = {metrics['perplexity']:.4f}")
    
    # Run the experiment with the modular API
    try:
        results = run_neural_plasticity_experiment(
            model_name=args.model,
            device=device,
            output_dir=args.output_dir,
            pruning_strategy=args.strategy,
            prune_percent=args.prune_percent,
            warmup_steps=args.warmup_steps,
            training_steps=args.training_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            colab_mode=IS_COLAB
        )
        
        # Print summary
        if results:
            baseline_ppl = results["metrics"]["baseline"]["perplexity"]
            pruned_ppl = results["metrics"]["post_pruning"]["perplexity"]
            final_ppl = results["metrics"]["final"]["perplexity"]
            pruned_heads = results["metrics"]["post_pruning"]["pruned_heads"]
            
            recovery_rate = results.get("recovery_rate", 0) * 100
            
            print("\n=== Neural Plasticity Experiment Summary ===")
            print(f"Model: {args.model}")
            print(f"Pruning: {args.strategy} at {args.prune_percent*100:.1f}% level")
            print(f"Pruned {pruned_heads} heads")
            print(f"Perplexity:")
            print(f"- Baseline: {baseline_ppl:.4f}")
            
            pruned_change = ((pruned_ppl / baseline_ppl) - 1) * 100
            print(f"- After Pruning: {pruned_ppl:.4f} ({pruned_change:+.1f}%)")
            
            final_change = ((final_ppl / baseline_ppl) - 1) * 100
            print(f"- After Fine-tuning: {final_ppl:.4f} ({final_change:+.1f}%)")
            
            print(f"Recovery rate: {recovery_rate:.1f}%")
            
            # Print success level
            if final_ppl <= baseline_ppl * 1.05:
                success_message = "SUCCESS! Model recovered fully after pruning."
            elif final_ppl <= baseline_ppl * 1.15:
                success_message = "PARTIAL SUCCESS. Model recovered moderately well after pruning."
            else:
                success_message = "LIMITED SUCCESS. Model showed limited recovery after pruning."
            
            print(f"\nConclusion: {success_message}")
    
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print(f"\nNeural Plasticity Experiment completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())