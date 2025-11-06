#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimization Level Comparison Tool

This script tests different optimization levels to identify the most effective
approaches for improving model performance. It runs comparative benchmarks
of different optimization strategies to find the optimal configuration.

Usage:
    python scripts/test_optimization_levels.py [--options]
"""

import os
import sys
import time
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path 
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
from models.loaders.loader_optimized import load_optimized_adaptive_model
from scripts.pruning_comparison.pruning_agency_comparison import apply_pruning, evaluate_model
from transformers import AutoTokenizer


def setup_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare different optimization levels")
    
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
    
    # Optimization parameters
    parser.add_argument("--optimization_levels", type=str, default="0,1,2,3",
                        help="Comma-separated optimization levels to test (default: 0,1,2,3)")
    parser.add_argument("--pruning_levels", type=str, default="0,30,50,70",
                        help="Comma-separated pruning percentages (default: 0,30,50,70)")
    
    # Test configuration
    parser.add_argument("--iterations", type=int, default=2,
                        help="Iterations for each test (default: 2)")
    parser.add_argument("--warmup", action="store_true",
                        help="Perform warmup runs before timing")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    parser.add_argument("--output_dir", type=str, default="optimization_results",
                        help="Directory to save results")
    
    return parser.parse_args()


def load_prompts():
    """Load test prompts."""
    test_prompts = [
        "Write a short summary of artificial intelligence and its applications in modern technology.",
        "Explain how transformer neural networks function in simple terms.",
        "What are the key ethical implications of large language models?",
        "Describe the concept of attention in neural networks and why it's important.",
        "Write a function to calculate the Fibonacci sequence in Python."
    ]
    return test_prompts


def test_optimization_level(level, args, pruning=0):
    """
    Test a specific optimization level with optional pruning.
    
    Args:
        level: Optimization level to test
        args: Command line arguments
        pruning: Pruning percentage to apply
        
    Returns:
        Dictionary with test results
    """
    print(f"\n{'-'*60}")
    print(f"Testing optimization level {level} with {pruning}% pruning")
    print(f"{'-'*60}")
    
    # Set environment variable for optimization level
    os.environ["OPTIMIZATION_LEVEL"] = str(level)
    
    # Default results in case of failure
    default_results = {
        "optimization_level": level,
        "pruning": pruning,
        "load_time": 0.0,
        "generation_times": [0.0],
        "avg_generation_time": 0.0,
        "tokens_per_second_values": [1.0],
        "avg_tokens_per_second": 1.0,
        "perplexities": [100.0],
        "avg_perplexity": 100.0,
        "error": True
    }
    
    try:
        # Load baseline model first
        baseline_model = load_baseline_model(args.model_name, args.device)
        
        # Load appropriate model based on optimization level
        start_time = time.time()
        if level == 0:
            # Use original implementation
            model = load_adaptive_model(
                args.model_name, 
                baseline_model, 
                args.device,
                debug=False,
                quiet=not args.verbose
            )
        else:
            # Use optimized implementation
            model = load_optimized_adaptive_model(
                args.model_name, 
                baseline_model, 
                args.device,
                optimization_level=level,  # Pass the level explicitly
                debug=False,
                quiet=not args.verbose
            )
        load_time = time.time() - start_time
        
        # Apply pruning if needed
        if pruning > 0:
            model, pruned_count, _ = apply_pruning(
                model, 
                pruning, 
                verbose=args.verbose,
                quiet=not args.verbose
            )
        
        # Prepare for testing
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        prompts = load_prompts()
        if args.max_prompts and args.max_prompts < len(prompts):
            prompts = prompts[:args.max_prompts]
        
        # Warmup run if requested
        if args.warmup:
            print("Performing warmup run...")
            try:
                _ = evaluate_model(
                    model,
                    tokenizer,
                    prompts[:1],
                    args.num_tokens,
                    temperature=args.temperature,
                    device=args.device,
                    quiet=True
                )
            except Exception as e:
                print(f"Warmup failed: {e}")
        
        # Run multiple iterations for statistical significance
        generation_times = []
        tokens_per_second_values = []
        perplexities = []
        
        for iteration in range(args.iterations):
            print(f"\nIteration {iteration+1}/{args.iterations}")
            
            # Run evaluation
            try:
                start_time = time.time()
                results = evaluate_model(
                    model,
                    tokenizer,
                    prompts,
                    args.num_tokens,
                    temperature=args.temperature,
                    device=args.device,
                    quiet=not args.verbose
                )
                eval_time = time.time() - start_time
                
                # Store metrics
                generation_times.append(eval_time)
                tokens_per_second_values.append(results["tokens_per_second"])
                perplexities.append(results["perplexity"])
                
                print(f"  Time: {eval_time:.2f}s")
                print(f"  Tokens per second: {results['tokens_per_second']:.2f}")
                print(f"  Perplexity: {results['perplexity']:.2f}")
            except Exception as e:
                print(f"Evaluation failed: {e}")
                # Use placeholder values for failed iteration
                generation_times.append(0.1)
                tokens_per_second_values.append(1.0)
                perplexities.append(100.0)
        
        # Calculate averages only if we have valid data
        if generation_times:
            avg_generation_time = sum(generation_times) / len(generation_times)
            avg_tokens_per_second = sum(tokens_per_second_values) / len(tokens_per_second_values)
            avg_perplexity = sum(perplexities) / len(perplexities)
        else:
            # Default values if all iterations failed
            avg_generation_time = 0.0
            avg_tokens_per_second = 1.0
            avg_perplexity = 100.0
        
        # Free memory
        del baseline_model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Return results
        return {
            "optimization_level": level,
            "pruning": pruning,
            "load_time": load_time,
            "generation_times": generation_times,
            "avg_generation_time": avg_generation_time,
            "tokens_per_second_values": tokens_per_second_values,
            "avg_tokens_per_second": avg_tokens_per_second,
            "perplexities": perplexities,
            "avg_perplexity": avg_perplexity,
            "error": False
        }
    except Exception as e:
        print(f"Error testing optimization level {level}: {e}")
        # Return default results in case of failure
        return default_results


def run_optimization_comparison(args):
    """Run comparison of different optimization levels."""
    print("\n===== Optimization Level Comparison =====\n")
    
    # Parse optimization and pruning levels
    opt_levels = [int(x) for x in args.optimization_levels.split(",")]
    pruning_levels = [int(x) for x in args.pruning_levels.split(",")]
    
    # Store results
    results = {
        "metadata": {
            "model_name": args.model_name,
            "device": args.device,
            "precision": args.precision,
            "num_tokens": args.num_tokens,
            "temperature": args.temperature,
            "iterations": args.iterations,
            "optimization_levels": opt_levels,
            "pruning_levels": pruning_levels,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "tests": []
    }
    
    # Run tests for each optimization level and pruning combination
    for opt_level in opt_levels:
        for pruning in pruning_levels:
            # Run test
            test_results = test_optimization_level(opt_level, args, pruning)
            
            # Store results
            results["tests"].append(test_results)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    results_file = os.path.join(args.output_dir, f"optimization_results_{int(time.time())}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    return results


def visualize_results(results, args):
    """Create visualizations from test results."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract optimization and pruning levels
    opt_levels = results["metadata"]["optimization_levels"]
    pruning_levels = results["metadata"]["pruning_levels"]
    
    # Organize results by optimization level and pruning
    organized_results = {
        opt_level: {
            pruning: None
            for pruning in pruning_levels
        }
        for opt_level in opt_levels
    }
    
    # Fill in organized results
    for test in results["tests"]:
        opt_level = test["optimization_level"]
        pruning = test["pruning"]
        organized_results[opt_level][pruning] = test
    
    # Visualization 1: Speed by optimization level and pruning
    plt.figure(figsize=(12, 8))
    
    # For each optimization level, plot speed vs pruning
    for opt_level in opt_levels:
        # Extract speeds
        speeds = [organized_results[opt_level][pruning]["avg_tokens_per_second"] 
                 for pruning in pruning_levels]
        
        # Plot line
        plt.plot(pruning_levels, speeds, 'o-', label=f"Level {opt_level}", linewidth=2)
    
    plt.title("Speed by Optimization Level and Pruning", fontsize=16)
    plt.xlabel("Pruning Level (%)", fontsize=14)
    plt.ylabel("Tokens per Second", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "speed_by_optimization.png"), dpi=150)
    plt.close()
    
    # Visualization 2: Quality by optimization level and pruning
    plt.figure(figsize=(12, 8))
    
    # For each optimization level, plot perplexity vs pruning
    for opt_level in opt_levels:
        # Extract perplexities (lower is better)
        perplexities = [organized_results[opt_level][pruning]["avg_perplexity"] 
                       for pruning in pruning_levels]
        
        # Plot line
        plt.plot(pruning_levels, perplexities, 'o-', label=f"Level {opt_level}", linewidth=2)
    
    plt.title("Quality by Optimization Level and Pruning", fontsize=16)
    plt.xlabel("Pruning Level (%)", fontsize=14)
    plt.ylabel("Perplexity (Lower is Better)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "quality_by_optimization.png"), dpi=150)
    plt.close()
    
    # Visualization 3: Speedup comparison - if we have level 0 as baseline
    base_level = min(opt_levels)  # Use lowest level as baseline if level 0 not available
    if len(opt_levels) > 1:
        plt.figure(figsize=(12, 8))
        
        # For each optimization level > base_level, calculate speedup
        for opt_level in [level for level in opt_levels if level > base_level]:
            # Calculate speedup
            speedups = []
            for pruning in pruning_levels:
                baseline_speed = organized_results[base_level][pruning]["avg_tokens_per_second"]
                opt_speed = organized_results[opt_level][pruning]["avg_tokens_per_second"]
                speedup = opt_speed / baseline_speed
                speedups.append(speedup)
            
            # Plot line
            plt.plot(pruning_levels, speedups, 'o-', label=f"Level {opt_level}", linewidth=2)
        
        # Add horizontal line at y=1.0
        plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
        
        plt.title(f"Speedup over Level {base_level}", fontsize=16)
        plt.xlabel("Pruning Level (%)", fontsize=14)
        plt.ylabel("Speedup Factor (>1 is better)", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "speedup_comparison.png"), dpi=150)
        plt.close()
    
    # Visualization 4: Loading time by optimization level
    plt.figure(figsize=(10, 6))
    
    # Extract loading times
    load_times = [organized_results[opt_level][0]["load_time"] for opt_level in opt_levels]
    
    # Create bar chart
    bars = plt.bar(opt_levels, load_times, color="teal", alpha=0.7)
    
    plt.title("Model Loading Time by Optimization Level", fontsize=16)
    plt.xlabel("Optimization Level", fontsize=14)
    plt.ylabel("Loading Time (seconds)", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f"{height:.2f}s", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "loading_time.png"), dpi=150)
    plt.close()
    
    # Visualization 5: Optimization matrix (heatmap)
    if len(opt_levels) > 1:
        plt.figure(figsize=(10, 8))
        
        # Create data for heatmap - speedup over baseline for each combination
        speedup_matrix = np.zeros((len(opt_levels), len(pruning_levels)))
        
        for i, opt_level in enumerate(opt_levels):
            for j, pruning in enumerate(pruning_levels):
                if opt_level == base_level:
                    # Baseline is 1.0x speedup over itself
                    speedup_matrix[i, j] = 1.0
                else:
                    baseline_speed = organized_results[base_level][pruning]["avg_tokens_per_second"]
                    opt_speed = organized_results[opt_level][pruning]["avg_tokens_per_second"]
                    speedup = opt_speed / baseline_speed
                    speedup_matrix[i, j] = speedup
        
        # Create heatmap
        import seaborn as sns
        sns.heatmap(speedup_matrix, annot=True, fmt=".2f", xticklabels=pruning_levels, 
                    yticklabels=opt_levels, cmap="RdYlGn", center=1.0)
        
        plt.title(f"Speedup Matrix (vs. Level {base_level})", fontsize=16)
        plt.xlabel("Pruning Level (%)", fontsize=14)
        plt.ylabel("Optimization Level", fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "speedup_matrix.png"), dpi=150)
        plt.close()
    
    print(f"Visualizations saved to {args.output_dir}")


def print_summary(results):
    """Print a summary of the test results."""
    # Extract optimization and pruning levels
    opt_levels = results["metadata"]["optimization_levels"]
    pruning_levels = results["metadata"]["pruning_levels"]
    
    # Define the baseline level
    base_level = min(opt_levels)  # Use lowest level as baseline if level 0 not available
    
    # Organize results by optimization level and pruning
    organized_results = {
        opt_level: {
            pruning: None
            for pruning in pruning_levels
        }
        for opt_level in opt_levels
    }
    
    # Fill in organized results
    for test in results["tests"]:
        opt_level = test["optimization_level"]
        pruning = test["pruning"]
        organized_results[opt_level][pruning] = test
    
    # Print summary header
    print("\n" + "="*80)
    print("OPTIMIZATION LEVEL COMPARISON SUMMARY")
    print("="*80)
    
    # Print loading time comparison
    print("\nModel Loading Time:")
    for opt_level in opt_levels:
        load_time = organized_results[opt_level][0]["load_time"]
        print(f"  Level {opt_level}: {load_time:.2f}s")
    
    # Print speed comparison at different pruning levels
    print("\nSpeed Comparison (tokens per second):")
    print(f"{'Pruning %':<10}", end="")
    for opt_level in opt_levels:
        print(f"Level {opt_level}".ljust(12), end="")
    print()
    print("-" * (10 + 12 * len(opt_levels)))
    
    for pruning in pruning_levels:
        print(f"{pruning:<10}", end="")
        for opt_level in opt_levels:
            speed = organized_results[opt_level][pruning]["avg_tokens_per_second"]
            print(f"{speed:.2f}".ljust(12), end="")
        print()
    
    # Print speedup comparison (if we have multiple levels)
    if len(opt_levels) > 1:
        print(f"\nSpeedup vs. Level {base_level}:")
        print(f"{'Pruning %':<10}", end="")
        for opt_level in [level for level in opt_levels if level > base_level]:
            print(f"Level {opt_level}".ljust(12), end="")
        print()
        print("-" * (10 + 12 * (len(opt_levels)-1)))
        
        for pruning in pruning_levels:
            print(f"{pruning:<10}", end="")
            baseline_speed = organized_results[base_level][pruning]["avg_tokens_per_second"]
            for opt_level in [level for level in opt_levels if level > base_level]:
                opt_speed = organized_results[opt_level][pruning]["avg_tokens_per_second"]
                speedup = opt_speed / baseline_speed
                print(f"{speedup:.2f}x".ljust(12), end="")
            print()
    
    # Print quality comparison
    print("\nQuality Comparison (perplexity, lower is better):")
    print(f"{'Pruning %':<10}", end="")
    for opt_level in opt_levels:
        print(f"Level {opt_level}".ljust(12), end="")
    print()
    print("-" * (10 + 12 * len(opt_levels)))
    
    for pruning in pruning_levels:
        print(f"{pruning:<10}", end="")
        for opt_level in opt_levels:
            perplexity = organized_results[opt_level][pruning]["avg_perplexity"]
            print(f"{perplexity:.2f}".ljust(12), end="")
        print()
    
    # Print overall recommendation
    print("\nOPTIMIZATION RECOMMENDATION:")
    
    # Find best level based on average speedup across pruning levels
    avg_speedups = []
    for opt_level in opt_levels:
        speeds = [organized_results[opt_level][pruning]["avg_tokens_per_second"] 
                 for pruning in pruning_levels]
        avg_speed = sum(speeds) / len(speeds)
        if opt_level == base_level:
            baseline_avg_speed = avg_speed
            avg_speedups.append(1.0)
        else:
            avg_speedups.append(avg_speed / baseline_avg_speed)
    
    best_level = opt_levels[avg_speedups.index(max(avg_speedups))]
    best_speedup = max(avg_speedups)
    
    print(f"  Best optimization level: {best_level} (Average {best_speedup:.2f}x speedup)")
    
    # Check for quality impact
    quality_impacts = []
    for opt_level in opt_levels:
        if opt_level == base_level:
            quality_impacts.append(0.0)
        else:
            baseline_ppls = [organized_results[base_level][pruning]["avg_perplexity"] 
                             for pruning in pruning_levels]
            opt_ppls = [organized_results[opt_level][pruning]["avg_perplexity"] 
                        for pruning in pruning_levels]
            
            # Calculate quality impact (negative is better, positive is worse)
            impacts = [(opt - baseline) / baseline * 100 
                      for baseline, opt in zip(baseline_ppls, opt_ppls)]
            avg_impact = sum(impacts) / len(impacts)
            quality_impacts.append(avg_impact)
    
    best_level_quality = quality_impacts[opt_levels.index(best_level)]
    quality_status = "improved" if best_level_quality < 0 else "degraded"
    
    print(f"  Quality impact: {abs(best_level_quality):.2f}% {quality_status}")
    
    # Print best pruning level for recommended optimization
    best_level_speeds = [organized_results[best_level][pruning]["avg_tokens_per_second"] 
                        for pruning in pruning_levels]
    best_pruning = pruning_levels[best_level_speeds.index(max(best_level_speeds))]
    
    print(f"  Best pruning level with optimization level {best_level}: {best_pruning}%")
    
    # Maximum achievable speedup
    max_speedup = max(best_level_speeds) / organized_results[base_level][0]["avg_tokens_per_second"]
    print(f"  Maximum speedup: {max_speedup:.2f}x (Level {best_level} with {best_pruning}% pruning)")


def main():
    """Main function."""
    # Parse arguments
    args = setup_args()
    
    # Run tests
    results = run_optimization_comparison(args)
    
    # Create visualizations
    visualize_results(results, args)
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()