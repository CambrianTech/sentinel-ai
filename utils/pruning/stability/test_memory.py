#!/usr/bin/env python
"""
Test script for memory management utilities.

This script demonstrates and tests the memory management tools,
which help optimize training parameters for different models and hardware.

Usage:
    python -m utils.pruning.stability.test_memory

Options:
    --model MODEL       Model to test with (default: gpt2)
    --gpu_memory N      GPU memory in GB to simulate (default: auto-detect)
    --test_all          Test all common models
"""

import argparse
import logging
import sys
from utils.pruning.stability import (
    estimate_model_memory, optimize_training_parameters,
    get_default_gpu_memory
)

# Available models for testing
TEST_MODELS = [
    "gpt2", 
    "gpt2-medium", 
    "gpt2-large", 
    "gpt2-xl",
    "distilgpt2",
    "facebook/opt-125m", 
    "facebook/opt-350m", 
    "facebook/opt-1.3b", 
    "facebook/opt-2.7b",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b"
]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test memory management functions")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model to test with (default: gpt2)")
    parser.add_argument("--gpu_memory", type=float, default=None,
                        help="GPU memory in GB to simulate (default: auto-detect)")
    parser.add_argument("--test_all", action="store_true",
                        help="Test all common models")
    return parser.parse_args()

def main():
    """Run memory management tests."""
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    
    # Get GPU memory
    if args.gpu_memory is None:
        gpu_memory = get_default_gpu_memory()
        logging.info(f"Auto-detected GPU memory: {gpu_memory:.1f}GB")
    else:
        gpu_memory = args.gpu_memory
        logging.info(f"Using specified GPU memory: {gpu_memory:.1f}GB")
    
    # Test models
    if args.test_all:
        models_to_test = TEST_MODELS
    else:
        models_to_test = [args.model]
    
    print("\n{:<20} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}".format(
        "Model", "Size (M)", "Batch", "Seq Len", "Mem Est", "Memory Use %"
    ))
    print("-" * 80)
    
    for model in models_to_test:
        # Get optimized parameters
        params = optimize_training_parameters(model, gpu_memory)
        
        # Calculate memory usage percentage
        memory_percentage = (params["memory_estimate_gb"] / gpu_memory) * 100
        
        # Print results
        print("{:<20} | {:<10.1f} | {:<10} | {:<10} | {:<10.2f} | {:<10.1f}%".format(
            model,
            params["model_size_millions"],
            params["batch_size"],
            params["sequence_length"],
            params["memory_estimate_gb"],
            memory_percentage
        ))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())