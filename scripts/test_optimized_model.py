#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized Model Comparison Test Script

This script tests the performance improvements of the optimized UNet transformer
with baseline integration compared to the original implementation.

Features:
- Loads both original and optimized models
- Compares inference speed and quality
- Tests with different pruning levels
- Reports speedup metrics

Usage:
    python scripts/test_optimized_model.py [--options]
"""

import os
import sys
import time
import argparse
import torch
from pathlib import Path

# Add root directory to path 
try:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
except NameError:
    # In Colab or interactive environments where __file__ isn't defined
    if os.path.exists("models") and os.path.exists("utils"):
        sys.path.insert(0, os.path.abspath("."))
    elif os.path.exists("../models") and os.path.exists("../utils"):
        sys.path.insert(0, os.path.abspath(".."))
    else:
        print("Warning: Could not determine repository root path. Import errors may occur.")

from models.loaders.loader import load_baseline_model, load_adaptive_model
from scripts.pruning_comparison.pruning_agency_comparison import apply_pruning, evaluate_model
from transformers import AutoTokenizer


def setup_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare original vs optimized model performance")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="gpt2", 
                        help="Base model to use (default: gpt2)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="Device to run on (default: cuda)")
    parser.add_argument("--precision", type=str, default="float16", 
                        choices=["float32", "float16", "bfloat16"],
                        help="Precision for model weights (default: float16)")
    
    # Generation parameters
    parser.add_argument("--num_tokens", type=int, default=50,
                        help="Number of tokens to generate (default: 50)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation (default: 0.7)")
    parser.add_argument("--max_prompts", type=int, default=2,
                        help="Maximum number of prompts to test (default: 2)")
    
    # Pruning parameters
    parser.add_argument("--pruning_levels", type=str, default="0,30,50,70",
                        help="Comma-separated pruning percentages (default: 0,30,50,70)")
    
    # Test configuration
    parser.add_argument("--iterations", type=int, default=2,
                        help="Iterations for each test (default: 2)")
    parser.add_argument("--warmup", action="store_true",
                        help="Perform warmup runs before timing")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    
    return parser.parse_args()


def load_prompts():
    """Load or create test prompts."""
    test_prompts = [
        "Write a short summary of artificial intelligence and its applications in modern technology.",
        "Explain how transformer neural networks function in simple terms.",
        "What are the key ethical implications of large language models?",
        "Describe the concept of attention in neural networks and why it's important.",
        "Write a function to calculate the Fibonacci sequence in Python."
    ]
    return test_prompts


def format_results(results, title):
    """Format results table for display."""
    output = f"\n=== {title} ===\n"
    output += f"{'Metric':<20} {'Value':<15} {'StdDev':<10}\n"
    output += f"{'-'*45}\n"
    
    for key, value in results.items():
        if key != "outputs" and not isinstance(value, dict):
            output += f"{key:<20} {value:<15.4f} {0:<10.4f}\n"
    
    return output


def compare_models(args):
    """Run comparison between original and optimized models."""
    print("\n===== Optimized Model Performance Comparison =====\n")
    
    # Load prompts
    prompts = load_prompts()
    if args.max_prompts and args.max_prompts < len(prompts):
        prompts = prompts[:args.max_prompts]
    
    print(f"Using {len(prompts)} prompts for evaluation")
    
    # Parse pruning levels
    pruning_levels = [int(x) for x in args.pruning_levels.split(",")]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Results dictionary to store all measurements
    results = {
        "original": {},
        "optimized": {},
        "speedup": {},
        "metadata": {
            "model_name": args.model_name,
            "device": args.device,
            "precision": args.precision,
            "pruning_levels": pruning_levels,
            "num_tokens": args.num_tokens,
            "temperature": args.temperature,
            "iterations": args.iterations
        }
    }
    
    # For each pruning level
    for level in pruning_levels:
        print(f"\n{'-'*50}")
        print(f"Testing pruning level: {level}%")
        print(f"{'-'*50}")
        
        results["original"][level] = {"times": []}
        results["optimized"][level] = {"times": []}
        
        # Run multiple iterations for statistical significance
        for iteration in range(args.iterations):
            print(f"\nIteration {iteration+1}/{args.iterations}")
            
            # Test original model
            print("\nTesting original model...")
            # Force use of original model
            os.environ["USE_OPTIMIZED_MODEL"] = "0"
            
            # Load baseline model first
            baseline_model = load_baseline_model(args.model_name, args.device)
            
            # Then load original adaptive model
            start_time = time.time()
            original_model, _ = load_adaptive_model(
                args.model_name, 
                baseline_model, 
                args.device,
                debug=False,
                quiet=not args.verbose
            )
            load_time_original = time.time() - start_time
            
            # Apply pruning
            original_model, pruned_count, _ = apply_pruning(
                original_model, 
                level, 
                verbose=args.verbose,
                quiet=not args.verbose
            )
            
            # Warmup run if requested
            if args.warmup:
                print("Performing warmup run...")
                _ = evaluate_model(
                    original_model,
                    tokenizer,
                    prompts[:1],
                    args.num_tokens,
                    temperature=args.temperature,
                    device=args.device,
                    quiet=True
                )
            
            # Time evaluation
            start_time = time.time()
            original_results = evaluate_model(
                original_model,
                tokenizer,
                prompts,
                args.num_tokens,
                temperature=args.temperature,
                device=args.device,
                quiet=not args.verbose
            )
            eval_time_original = time.time() - start_time
            
            # Free memory
            del baseline_model
            del original_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Test optimized model
            print("\nTesting optimized model...")
            # Force use of optimized model
            os.environ["USE_OPTIMIZED_MODEL"] = "1"
            
            # Load baseline model first
            baseline_model = load_baseline_model(args.model_name, args.device)
            
            # Then load optimized adaptive model
            start_time = time.time()
            optimized_model, _ = load_adaptive_model(
                args.model_name, 
                baseline_model, 
                args.device,
                debug=False,
                quiet=not args.verbose
            )
            load_time_optimized = time.time() - start_time
            
            # Apply pruning
            optimized_model, pruned_count, _ = apply_pruning(
                optimized_model, 
                level, 
                verbose=args.verbose,
                quiet=not args.verbose
            )
            
            # Warmup run if requested
            if args.warmup:
                print("Performing warmup run...")
                _ = evaluate_model(
                    optimized_model,
                    tokenizer,
                    prompts[:1],
                    args.num_tokens,
                    temperature=args.temperature,
                    device=args.device,
                    quiet=True
                )
            
            # Time evaluation
            start_time = time.time()
            optimized_results = evaluate_model(
                optimized_model,
                tokenizer,
                prompts,
                args.num_tokens,
                temperature=args.temperature,
                device=args.device,
                quiet=not args.verbose
            )
            eval_time_optimized = time.time() - start_time
            
            # Free memory
            del baseline_model
            del optimized_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Store iteration results
            results["original"][level]["times"].append(eval_time_original)
            results["optimized"][level]["times"].append(eval_time_optimized)
            
            if iteration == 0:
                # Store first iteration metrics
                results["original"][level].update({
                    "load_time": load_time_original,
                    "tokens_per_second": original_results["tokens_per_second"],
                    "perplexity": original_results["perplexity"],
                    "diversity": original_results["diversity"]
                })
                
                results["optimized"][level].update({
                    "load_time": load_time_optimized,
                    "tokens_per_second": optimized_results["tokens_per_second"],
                    "perplexity": optimized_results["perplexity"],
                    "diversity": optimized_results["diversity"]
                })
            
            # Calculate speedup for this iteration
            speedup = eval_time_original / eval_time_optimized
            tokens_speedup = optimized_results["tokens_per_second"] / original_results["tokens_per_second"]
            
            print(f"\nResults for iteration {iteration+1}:")
            print(f"  Original model: {eval_time_original:.2f}s ({original_results['tokens_per_second']:.2f} tokens/sec)")
            print(f"  Optimized model: {eval_time_optimized:.2f}s ({optimized_results['tokens_per_second']:.2f} tokens/sec)")
            print(f"  Speedup: {speedup:.2f}x (raw), {tokens_speedup:.2f}x (tokens/sec)")
        
        # Calculate average speedup
        avg_time_original = sum(results["original"][level]["times"]) / len(results["original"][level]["times"])
        avg_time_optimized = sum(results["optimized"][level]["times"]) / len(results["optimized"][level]["times"])
        avg_speedup = avg_time_original / avg_time_optimized
        
        results["speedup"][level] = {
            "time_speedup": avg_speedup,
            "tokens_speedup": results["optimized"][level]["tokens_per_second"] / results["original"][level]["tokens_per_second"],
            "quality_ratio": results["original"][level]["perplexity"] / results["optimized"][level]["perplexity"]
        }
        
        print(f"\nAverage speedup at {level}% pruning: {avg_speedup:.2f}x")
        print(f"Tokens per second: {results['optimized'][level]['tokens_per_second']:.2f} vs {results['original'][level]['tokens_per_second']:.2f}")
        print(f"Perplexity: {results['optimized'][level]['perplexity']:.2f} vs {results['original'][level]['perplexity']:.2f}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF OPTIMIZED MODEL PERFORMANCE")
    print("="*60)
    
    print("\nPerformance speedup by pruning level:")
    for level in pruning_levels:
        speedup = results["speedup"][level]["tokens_speedup"]
        quality = results["speedup"][level]["quality_ratio"]
        quality_str = "better" if quality > 1.0 else "worse"
        
        print(f"  Level {level}%: {speedup:.2f}x faster, {abs(1-quality):.2f}x {quality_str} quality")
    
    return results


def main():
    """Main function."""
    args = setup_args()
    
    # Run the comparison
    results = compare_models(args)
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()