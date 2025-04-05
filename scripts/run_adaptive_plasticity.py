#!/usr/bin/env python
"""
Run Adaptive Neural Plasticity Optimization

This script runs the adaptive plasticity system which:
1. Uses degeneration detection as a quality control mechanism
2. Automatically finds optimal pruning and growth strategies
3. Self-improves by learning from successful transformations
4. Adapts strategies based on historical performance
5. Finds efficient network structures while maintaining quality

Example usage:
    python scripts/run_adaptive_plasticity.py --model_name distilgpt2 --dataset tiny_shakespeare
    python scripts/run_adaptive_plasticity.py --model_name gpt2 --max_cycles 15 --patience 5
    python scripts/run_adaptive_plasticity.py --model_name facebook/opt-125m --training_steps 200
"""

import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Sentinel-AI modules
from sdata.dataset_loader import load_dataset
from utils.adaptive.adaptive_plasticity import run_adaptive_system

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run Adaptive Neural Plasticity Optimization")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                      help="Model name (default: distilgpt2)")
    parser.add_argument("--dataset", type=str, default="tiny_shakespeare",
                      help="Dataset name (default: tiny_shakespeare)")
    parser.add_argument("--output_dir", type=str, default="./output/adaptive_plasticity",
                      help="Directory to save results (default: ./output/adaptive_plasticity)")
    
    # Optimization parameters
    parser.add_argument("--max_cycles", type=int, default=10,
                      help="Maximum number of plasticity cycles to run (default: 10)")
    parser.add_argument("--initial_pruning", type=float, default=0.2,
                      help="Initial pruning level (default: 0.2 = 20%% of heads)")
    parser.add_argument("--initial_growth", type=float, default=0.5,
                      help="Initial growth ratio (default: 0.5 = 50%% of pruned heads)")
    parser.add_argument("--training_steps", type=int, default=100,
                      help="Initial training steps per cycle (default: 100)")
    parser.add_argument("--patience", type=int, default=3,
                      help="Number of cycles without improvement before stopping (default: 3)")
    
    # Quality control parameters
    parser.add_argument("--max_degeneration", type=float, default=3.0,
                      help="Maximum acceptable degeneration score (default: 3.0)")
    parser.add_argument("--max_perplexity_increase", type=float, default=0.15,
                      help="Maximum acceptable perplexity increase (default: 0.15 = 15%% worse)")
    
    # System parameters
    parser.add_argument("--device", type=str, default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
                      help="Device to use (default: cuda if available, else cpu)")
    parser.add_argument("--sequence_length", type=int, default=128,
                      help="Sequence length for training (default: 128)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                      help="Base learning rate (default: 5e-5)")
    parser.add_argument("--memory_capacity", type=int, default=10,
                      help="Number of past successful transformations to remember (default: 10)")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose output")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed (default: 42)")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"=== Adaptive Neural Plasticity System ===")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Max cycles: {args.max_cycles}")
    print(f"Device: {args.device}")
    
    # Save start time
    start_time = datetime.now()
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load tokenizer first
    print(f"Loading tokenizer for {args.model_name}...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load the dataset
    print(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        max_length=args.sequence_length
    )
    
    # Run the adaptive system
    system = run_adaptive_system(
        model_name=args.model_name,
        dataset=dataset,
        output_dir=args.output_dir,
        max_cycles=args.max_cycles,
        device=args.device,
        initial_pruning_level=args.initial_pruning,
        initial_growth_ratio=args.initial_growth,
        initial_training_steps=args.training_steps,
        patience=args.patience,
        verbose=args.verbose
    )
    
    # Print total time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n=== Optimization Complete ===")
    print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Results saved to: {system.run_dir}")

if __name__ == "__main__":
    main()