#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimization Benchmark Suite

This script provides a complete suite of benchmarks for the optimization levels
in the Sentinel AI model. It combines the testing of optimization levels with
multi-model comparisons and pruning effectiveness to provide comprehensive performance
insights.

Features:
- Tests all optimization levels (0-3) with multiple models
- Benchmarks performance across different pruning percentages
- Generates visualizations comparing speed, quality, and memory usage
- Produces detailed reports with recommendations for production deployment

Usage:
    python scripts/benchmark_optimization.py [--options]
    
Example:
    python scripts/benchmark_optimization.py --models gpt2,gpt2-medium --pruning 0,30,50,70
"""

import os
import sys
import time
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

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
from scripts.pruning_comparison.pruning_agency_comparison import apply_pruning
from transformers import AutoTokenizer

# For typing hints
from typing import Dict, List, Optional, Tuple, Union, Any


def setup_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Comprehensive optimization benchmark suite")
    
    # Model configuration
    parser.add_argument("--models", type=str, default="gpt2",
                        help="Comma-separated list of models to benchmark (default: gpt2)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (default: cuda if available, else cpu)")
    parser.add_argument("--precision", type=str, default="float16", 
                        choices=["float32", "float16", "bfloat16"],
                        help="Precision for model weights (default: float16)")
    
    # Test parameters
    parser.add_argument("--optimization_levels", type=str, default="0,1,2,3",
                        help="Comma-separated optimization levels to test (default: 0,1,2,3)")
    parser.add_argument("--pruning", type=str, default="0,30,50,70",
                        help="Comma-separated pruning percentages to test (default: 0,30,50,70)")
    parser.add_argument("--sequence_length", type=int, default=64,
                        help="Input sequence length for benchmarking (default: 64)")
    parser.add_argument("--generation_length", type=int, default=50,
                        help="Number of tokens to generate (default: 50)")
    parser.add_argument("--prompts", type=str, default="default",
                        help="Comma-separated prompts or 'default' to use built-in (default: default)")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Iterations for each benchmark (default: 3)")
    
    # Performance measurement
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warmup runs before measurement (default: 1)")
    parser.add_argument("--memory_profiling", action="store_true",
                        help="Enable memory profiling (may slow down benchmarks)")
    parser.add_argument("--profile_components", action="store_true",
                        help="Profile individual components (attention, FFN, etc.)")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                        help="Directory to save benchmark results")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations of benchmark results")
    parser.add_argument("--save_generated_text", action="store_true",
                        help="Save generated text samples from each model configuration")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed benchmark information")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress non-essential output")
    
    # Advanced options
    parser.add_argument("--baseline_only", action="store_true",
                        help="Run baseline models only (no adaptive model)")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip baseline model benchmarks")
    parser.add_argument("--unet_configuration", type=str, default="auto",
                        choices=["auto", "enabled", "disabled"],
                        help="UNet skip connection configuration (default: auto)")
    parser.add_argument("--controller_config", type=str, default=None,
                        help="Path to controller configuration file")
    
    return parser.parse_args()


def get_test_prompts():
    """Return a list of default test prompts."""
    return [
        "The transformer model architecture revolutionized natural language processing by",
        "Explain how self-attention mechanisms work in neural networks in simple terms.",
        "What are the key advantages of large language models compared to traditional ML approaches?",
        "Write a short paragraph about the future of artificial intelligence.",
        "Summarize the main concepts behind reinforcement learning with human feedback."
    ]


def format_benchmark_name(model_name, opt_level, pruning, is_baseline=False):
    """Create a formatted name for the benchmark configuration."""
    if is_baseline:
        return f"{model_name} (baseline)"
    
    opt_names = {
        0: "Original",
        1: "Optimized Attention",
        2: "Optimized UNet",
        3: "Integration Optimized"
    }
    
    opt_name = opt_names.get(opt_level, f"Level {opt_level}")
    if pruning > 0:
        return f"{model_name} - {opt_name} (Pruned {pruning}%)"
    else:
        return f"{model_name} - {opt_name}"


def run_single_benchmark(
    model_name: str,
    opt_level: int,
    pruning: int,
    args,
    tokenizer=None,
    prompts=None
) -> Dict[str, Any]:
    """
    Run a benchmark for a specific configuration.
    
    Args:
        model_name: The name of the model to benchmark
        opt_level: Optimization level (0-3)
        pruning: Pruning percentage (0-100)
        args: Command line arguments
        tokenizer: Optional pre-loaded tokenizer
        prompts: Optional list of prompts
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'-'*80}")
    benchmark_name = format_benchmark_name(model_name, opt_level, pruning)
    print(f"Running benchmark: {benchmark_name}")
    print(f"{'-'*80}")
    
    # Set environment variable for optimization level
    os.environ["OPTIMIZATION_LEVEL"] = str(opt_level)
    
    # Default benchmark results in case of failure
    default_results = {
        "model_name": model_name,
        "optimization_level": opt_level,
        "pruning": pruning,
        "load_time": 0.0,
        "peak_memory": 0,
        "initialization_error": True,
        "benchmark_error": True,
        "generation_times": [],
        "tokens_per_second": 0.0,
        "first_token_latency": 0.0,
        "perplexity": 100.0,
        "generated_samples": []
    }
    
    try:
        # Start memory tracking if enabled
        if args.memory_profiling and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated()
        else:
            start_memory = 0
        
        # Load the baseline model
        load_start = time.time()
        baseline_model = load_baseline_model(model_name, args.device)
        
        # Load the appropriate model based on optimization level
        if opt_level == 0:
            # Use original implementation
            model = load_adaptive_model(
                model_name, 
                baseline_model, 
                args.device,
                debug=args.verbose,
                quiet=args.quiet
            )
        else:
            # Use optimized implementation
            model = load_optimized_adaptive_model(
                model_name, 
                baseline_model, 
                args.device,
                optimization_level=opt_level,
                debug=args.verbose,
                quiet=args.quiet
            )
        
        load_time = time.time() - load_start
        
        # Measure peak memory after model loading
        if args.memory_profiling and torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() - start_memory
        else:
            peak_memory = 0
        
        # Apply pruning if needed
        if pruning > 0:
            model, pruned_count, _ = apply_pruning(
                model, 
                pruning, 
                verbose=args.verbose,
                quiet=args.quiet
            )
            
            if not args.quiet:
                print(f"Applied {pruning}% pruning ({pruned_count} heads pruned)")
        
        # Load tokenizer if not provided
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
        
        # Use provided prompts or load defaults
        if prompts is None:
            if args.prompts == "default":
                prompts = get_test_prompts()
            else:
                prompts = args.prompts.split(",")
        
        # Prepare the benchmark results
        benchmark_results = {
            "model_name": model_name,
            "optimization_level": opt_level,
            "pruning": pruning,
            "load_time": load_time,
            "peak_memory": peak_memory,
            "initialization_error": False,
            "benchmark_error": False,
            "generation_times": [],
            "tokens_per_second_values": [],
            "first_token_latencies": [],
            "perplexities": [],
            "generated_samples": []
        }
        
        # Warmup run if requested
        if args.warmup > 0:
            if not args.quiet:
                print(f"Performing {args.warmup} warmup runs...")
                
            with torch.no_grad():
                for _ in range(args.warmup):
                    # Encode warmup prompt
                    warmup_prompt = prompts[0]
                    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(args.device)
                    
                    # Generate text
                    _ = model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=len(inputs.input_ids[0]) + 20,
                        do_sample=True,
                        temperature=0.7
                    )
        
        # Run component profiling if requested
        if args.profile_components and hasattr(model, "get_agency_report"):
            model.profile_time = True
            
            # Enable debug/stats tracking
            if hasattr(model, "debug"):
                model.debug = True
            if hasattr(model, "reset_stats"):
                model.reset_stats()
                
            # Ensure CUDA synchronization if available
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            # Run a single generation with profiling enabled
            with torch.no_grad():
                inputs = tokenizer(prompts[0], return_tensors="pt").to(args.device)
                _ = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=len(inputs.input_ids[0]) + 20,
                    do_sample=True,
                    temperature=0.7
                )
                
            # Get profiling stats
            if hasattr(model, "stats"):
                benchmark_results["component_stats"] = model.stats.copy()
            
            # Get agency report
            if hasattr(model, "get_agency_report"):
                benchmark_results["agency_report"] = model.get_agency_report()
                
            # Reset after profiling
            if hasattr(model, "reset_stats"):
                model.reset_stats()
        
        # Run the actual benchmark iterations
        for iteration in range(args.iterations):
            iteration_results = []
            
            for prompt_idx, prompt in enumerate(prompts):
                if not args.quiet:
                    print(f"Running iteration {iteration+1}/{args.iterations}, prompt {prompt_idx+1}/{len(prompts)}")
                
                # Encode prompt
                inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
                
                # Optional cache clearing
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Measure first token latency and total generation time
                start_time = time.time()
                
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=len(inputs.input_ids[0]) + args.generation_length,
                        do_sample=True,
                        temperature=0.7
                    )
                
                # Ensure CUDA operations are completed
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Record total generation time
                generation_time = time.time() - start_time
                
                # Calculate tokens per second
                tokens_generated = len(output_ids[0]) - len(inputs.input_ids[0])
                tokens_per_second = tokens_generated / generation_time
                
                # Decode generated text
                generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # Calculate perplexity (using a simplistic approach)
                try:
                    with torch.no_grad():
                        # Get input IDs for the generated text
                        eval_inputs = tokenizer(generated_text, return_tensors="pt").to(args.device)
                        
                        # Forward pass through the model
                        if hasattr(model, "forward") and not hasattr(model, "transformer"):
                            # Our custom model
                            outputs = model(eval_inputs.input_ids, labels=eval_inputs.input_ids)
                        else:
                            # HuggingFace model
                            outputs = model(input_ids=eval_inputs.input_ids, labels=eval_inputs.input_ids)
                            
                        # Get loss
                        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                        perplexity = torch.exp(loss).item()
                except Exception as e:
                    if args.verbose:
                        print(f"Perplexity calculation failed: {e}")
                    perplexity = 100.0
                
                # Store results for this prompt
                prompt_result = {
                    "prompt": prompt,
                    "generation_time": generation_time,
                    "tokens_generated": tokens_generated,
                    "tokens_per_second": tokens_per_second,
                    "perplexity": perplexity,
                    "generated_text": generated_text
                }
                
                iteration_results.append(prompt_result)
                
                if not args.quiet:
                    print(f"  Time: {generation_time:.2f}s, "
                          f"Tokens: {tokens_generated}, "
                          f"Tokens/sec: {tokens_per_second:.2f}, "
                          f"Perplexity: {perplexity:.2f}")
            
            # Calculate aggregate metrics for this iteration
            iteration_generation_time = sum(r["generation_time"] for r in iteration_results) / len(iteration_results)
            iteration_tokens_per_second = sum(r["tokens_per_second"] for r in iteration_results) / len(iteration_results)
            iteration_perplexity = sum(r["perplexity"] for r in iteration_results) / len(iteration_results)
            
            # Record first token latency (approximated as total_time / tokens)
            first_token_latency = iteration_results[0]["generation_time"] / iteration_results[0]["tokens_generated"]
            
            # Add to benchmark results
            benchmark_results["generation_times"].append(iteration_generation_time)
            benchmark_results["tokens_per_second_values"].append(iteration_tokens_per_second)
            benchmark_results["first_token_latencies"].append(first_token_latency)
            benchmark_results["perplexities"].append(iteration_perplexity)
            
            # Store generated text samples if requested
            if args.save_generated_text:
                benchmark_results["generated_samples"].extend([
                    {"prompt": r["prompt"], "text": r["generated_text"]}
                    for r in iteration_results
                ])
        
        # Calculate average metrics
        benchmark_results["avg_generation_time"] = sum(benchmark_results["generation_times"]) / len(benchmark_results["generation_times"])
        benchmark_results["avg_tokens_per_second"] = sum(benchmark_results["tokens_per_second_values"]) / len(benchmark_results["tokens_per_second_values"])
        benchmark_results["avg_first_token_latency"] = sum(benchmark_results["first_token_latencies"]) / len(benchmark_results["first_token_latencies"])
        benchmark_results["avg_perplexity"] = sum(benchmark_results["perplexities"]) / len(benchmark_results["perplexities"])
        
        # Count parameters
        benchmark_results["parameter_count"] = sum(p.numel() for p in model.parameters())
        
        # Free memory
        del model
        del baseline_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Return results
        return benchmark_results
    
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return default_results


def run_all_benchmarks(args):
    """Run all requested benchmarks."""
    print("\n===== Starting Comprehensive Optimization Benchmark =====\n")
    
    # Parse optimization and pruning levels
    model_names = [model.strip() for model in args.models.split(",")]
    opt_levels = [int(level) for level in args.optimization_levels.split(",")]
    pruning_levels = [int(pruning) for pruning in args.pruning.split(",")]
    
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Store benchmark metadata
    benchmark_metadata = {
        "timestamp": datetime.now().isoformat(),
        "device": args.device,
        "precision": args.precision,
        "models": model_names,
        "optimization_levels": opt_levels,
        "pruning_levels": pruning_levels,
        "sequence_length": args.sequence_length,
        "generation_length": args.generation_length,
        "iterations": args.iterations,
        "warmup_runs": args.warmup,
        "args": vars(args)
    }
    
    # Store all benchmark results
    all_results = {
        "metadata": benchmark_metadata,
        "benchmarks": []
    }
    
    # Run benchmarks for each configuration
    total_benchmarks = len(model_names) * len(opt_levels) * len(pruning_levels)
    print(f"Running {total_benchmarks} benchmark configurations")
    
    # For tracking progress
    completed = 0
    
    # Load the prompts
    if args.prompts == "default":
        prompts = get_test_prompts()
    else:
        prompts = args.prompts.split(",")
    
    # Run benchmarks for each model
    for model_name in model_names:
        # Preload the tokenizer (shared across configurations)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Run benchmark for each optimization level
        for opt_level in opt_levels:
            # Run benchmark for each pruning level
            for pruning in pruning_levels:
                # Update progress
                completed += 1
                print(f"\nProgress: {completed}/{total_benchmarks} benchmarks")
                
                # Run the benchmark
                benchmark_results = run_single_benchmark(
                    model_name=model_name,
                    opt_level=opt_level,
                    pruning=pruning,
                    args=args,
                    tokenizer=tokenizer,
                    prompts=prompts
                )
                
                # Add to results
                all_results["benchmarks"].append(benchmark_results)
                
                # Save intermediate results
                results_file = os.path.join(args.output_dir, f"benchmark_results_{int(time.time())}.json")
                with open(results_file, "w") as f:
                    json.dump(all_results, f, indent=2)
                
                # Optional pause to allow system cooling
                time.sleep(1)
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results_file = os.path.join(args.output_dir, f"optimization_benchmark_{timestamp}.json")
    with open(final_results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {final_results_file}")
    
    return all_results


def create_visualizations(results, args):
    """Create visualizations from benchmark results."""
    if not args.visualize:
        return
    
    print("\n===== Creating Visualizations =====\n")
    
    # Create output directory for visualizations
    viz_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Set common plot style
    plt.style.use("ggplot")
    
    # Extract metadata
    metadata = results["metadata"]
    model_names = metadata["models"]
    opt_levels = metadata["optimization_levels"]
    pruning_levels = metadata["pruning_levels"]
    
    # Get all benchmark results
    benchmarks = results["benchmarks"]
    
    # Create lookup table for easy access to benchmark results
    benchmark_lookup = {}
    for benchmark in benchmarks:
        # Skip benchmarks with errors
        if benchmark["benchmark_error"] or benchmark["initialization_error"]:
            continue
            
        key = (benchmark["model_name"], benchmark["optimization_level"], benchmark["pruning"])
        benchmark_lookup[key] = benchmark
    
    # Generate heatmap data per model
    for model_name in model_names:
        # Create optimization level vs pruning level heatmaps for tokens per second
        plt.figure(figsize=(10, 8))
        
        # Prepare data for heatmap
        heatmap_data = np.zeros((len(opt_levels), len(pruning_levels)))
        
        for i, opt_level in enumerate(opt_levels):
            for j, pruning in enumerate(pruning_levels):
                key = (model_name, opt_level, pruning)
                if key in benchmark_lookup:
                    heatmap_data[i, j] = benchmark_lookup[key]["avg_tokens_per_second"]
        
        # Create heatmap
        ax = sns.heatmap(
            heatmap_data, 
            annot=True, 
            fmt=".2f", 
            xticklabels=pruning_levels,
            yticklabels=opt_levels, 
            cmap="viridis"
        )
        
        plt.title(f"{model_name}: Performance (tokens/sec)", fontsize=16)
        plt.xlabel("Pruning Level (%)", fontsize=14)
        plt.ylabel("Optimization Level", fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"{model_name}_performance_heatmap.png"), dpi=150)
        plt.close()
    
    # Create optimization level comparison across models (at 0% pruning)
    plt.figure(figsize=(12, 8))
    
    bar_width = 0.15
    index = np.arange(len(model_names))
    
    for i, opt_level in enumerate(opt_levels):
        tokens_per_second = []
        
        for model_name in model_names:
            key = (model_name, opt_level, 0)  # 0% pruning
            if key in benchmark_lookup:
                tokens_per_second.append(benchmark_lookup[key]["avg_tokens_per_second"])
            else:
                tokens_per_second.append(0)
        
        plt.bar(
            index + i * bar_width, 
            tokens_per_second, 
            bar_width,
            label=f"Level {opt_level}"
        )
    
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Tokens per Second", fontsize=14)
    plt.title("Performance by Optimization Level (0% Pruning)", fontsize=16)
    plt.xticks(index + bar_width * (len(opt_levels) - 1) / 2, model_names)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "optimization_level_comparison.png"), dpi=150)
    plt.close()
    
    # Create pruning effectiveness chart (for optimization level 3)
    plt.figure(figsize=(12, 8))
    
    for model_name in model_names:
        tokens_per_second = []
        
        for pruning in pruning_levels:
            key = (model_name, 3, pruning)  # Level 3 optimization
            if key in benchmark_lookup:
                tokens_per_second.append(benchmark_lookup[key]["avg_tokens_per_second"])
            else:
                tokens_per_second.append(0)
        
        plt.plot(pruning_levels, tokens_per_second, marker='o', linewidth=2, label=model_name)
    
    plt.xlabel("Pruning Level (%)", fontsize=14)
    plt.ylabel("Tokens per Second", fontsize=14)
    plt.title("Pruning Effectiveness (Level 3 Optimization)", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "pruning_effectiveness.png"), dpi=150)
    plt.close()
    
    # Create perplexity impact chart (quality vs. speed tradeoff)
    plt.figure(figsize=(12, 8))
    
    for model_name in model_names:
        tokens_per_second = []
        perplexities = []
        
        for pruning in pruning_levels:
            key = (model_name, 3, pruning)  # Level 3 optimization
            if key in benchmark_lookup:
                tokens_per_second.append(benchmark_lookup[key]["avg_tokens_per_second"])
                perplexities.append(benchmark_lookup[key]["avg_perplexity"])
        
        plt.scatter(tokens_per_second, perplexities, label=model_name, s=100)
        
        # Add labels for pruning levels
        for i, pruning in enumerate(pruning_levels):
            if i < len(tokens_per_second):
                plt.annotate(
                    f"{pruning}%", 
                    (tokens_per_second[i], perplexities[i]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center'
                )
    
    plt.xlabel("Tokens per Second", fontsize=14)
    plt.ylabel("Perplexity (lower is better)", fontsize=14)
    plt.title("Quality vs. Speed Tradeoff", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "quality_vs_speed.png"), dpi=150)
    plt.close()
    
    # Create first token latency comparison
    plt.figure(figsize=(12, 8))
    
    for i, opt_level in enumerate(opt_levels):
        first_token_latencies = []
        
        for model_name in model_names:
            key = (model_name, opt_level, 0)  # 0% pruning
            if key in benchmark_lookup:
                first_token_latencies.append(benchmark_lookup[key]["avg_first_token_latency"])
            else:
                first_token_latencies.append(0)
        
        plt.bar(
            index + i * bar_width, 
            first_token_latencies, 
            bar_width,
            label=f"Level {opt_level}"
        )
    
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("First Token Latency (seconds)", fontsize=14)
    plt.title("First Token Latency by Optimization Level", fontsize=16)
    plt.xticks(index + bar_width * (len(opt_levels) - 1) / 2, model_names)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "first_token_latency.png"), dpi=150)
    plt.close()
    
    # Create memory usage comparison
    plt.figure(figsize=(12, 8))
    
    for i, opt_level in enumerate(opt_levels):
        memory_usages = []
        
        for model_name in model_names:
            key = (model_name, opt_level, 0)  # 0% pruning
            if key in benchmark_lookup:
                # Convert to MB for more readable values
                memory_usages.append(benchmark_lookup[key]["peak_memory"] / (1024 * 1024))
            else:
                memory_usages.append(0)
        
        plt.bar(
            index + i * bar_width, 
            memory_usages, 
            bar_width,
            label=f"Level {opt_level}"
        )
    
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Peak Memory Usage (MB)", fontsize=14)
    plt.title("Memory Usage by Optimization Level", fontsize=16)
    plt.xticks(index + bar_width * (len(opt_levels) - 1) / 2, model_names)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "memory_usage.png"), dpi=150)
    plt.close()
    
    print(f"Visualizations saved to {viz_dir}")


def generate_benchmark_report(results, args):
    """Generate a detailed benchmark report with recommendations."""
    print("\n===== Generating Benchmark Report =====\n")
    
    # Extract metadata
    metadata = results["metadata"]
    model_names = metadata["models"]
    opt_levels = metadata["optimization_levels"]
    pruning_levels = metadata["pruning_levels"]
    
    # Get all benchmark results
    benchmarks = results["benchmarks"]
    
    # Create lookup table for easy access to benchmark results
    benchmark_lookup = {}
    for benchmark in benchmarks:
        # Skip benchmarks with errors
        if benchmark["benchmark_error"] or benchmark["initialization_error"]:
            continue
            
        key = (benchmark["model_name"], benchmark["optimization_level"], benchmark["pruning"])
        benchmark_lookup[key] = benchmark
    
    # Calculate improvements over baseline (opt_level 0)
    improvements = {}
    
    for model_name in model_names:
        improvements[model_name] = {}
        
        for opt_level in opt_levels:
            if opt_level == 0:
                continue
                
            improvements[model_name][opt_level] = {}
            
            for pruning in pruning_levels:
                baseline_key = (model_name, 0, pruning)
                current_key = (model_name, opt_level, pruning)
                
                if baseline_key in benchmark_lookup and current_key in benchmark_lookup:
                    baseline_tps = benchmark_lookup[baseline_key]["avg_tokens_per_second"]
                    current_tps = benchmark_lookup[current_key]["avg_tokens_per_second"]
                    
                    if baseline_tps > 0:
                        speedup = current_tps / baseline_tps
                    else:
                        speedup = 1.0
                    
                    improvements[model_name][opt_level][pruning] = {
                        "speedup": speedup,
                        "baseline_tps": baseline_tps,
                        "current_tps": current_tps
                    }
    
    # Find the best configuration for each model
    best_configs = {}
    
    for model_name in model_names:
        best_tps = 0
        best_config = None
        best_quality_adjusted = 0
        best_quality_config = None
        
        for opt_level in opt_levels:
            for pruning in pruning_levels:
                key = (model_name, opt_level, pruning)
                
                if key in benchmark_lookup:
                    # Pure speed
                    current_tps = benchmark_lookup[key]["avg_tokens_per_second"]
                    if current_tps > best_tps:
                        best_tps = current_tps
                        best_config = (opt_level, pruning)
                    
                    # Quality-adjusted (tokens per second / perplexity)
                    # Lower perplexity is better, so invert it
                    current_quality = current_tps / benchmark_lookup[key]["avg_perplexity"]
                    if current_quality > best_quality_adjusted:
                        best_quality_adjusted = current_quality
                        best_quality_config = (opt_level, pruning)
        
        best_configs[model_name] = {
            "best_speed": best_config,
            "best_speed_tps": best_tps,
            "best_quality": best_quality_config,
            "best_quality_adjusted": best_quality_adjusted
        }
    
    # Create the report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_file = os.path.join(args.output_dir, "benchmark_report.md")
    
    with open(report_file, "w") as f:
        f.write(f"# Sentinel AI Optimization Benchmark Report\n\n")
        f.write(f"Generated: {timestamp}\n\n")
        
        f.write("## Benchmark Configuration\n\n")
        f.write(f"- Device: {args.device}\n")
        f.write(f"- Precision: {args.precision}\n")
        f.write(f"- Models tested: {', '.join(model_names)}\n")
        f.write(f"- Optimization levels: {', '.join(map(str, opt_levels))}\n")
        f.write(f"- Pruning levels: {', '.join(map(str, pruning_levels))}%\n")
        f.write(f"- Iterations per benchmark: {args.iterations}\n")
        f.write(f"- Generation length: {args.generation_length} tokens\n\n")
        
        f.write("## Performance Summary\n\n")
        
        # Create performance table
        f.write("### Tokens per Second by Configuration\n\n")
        
        # Table header
        f.write("| Model | Optimization Level | Pruning | Tokens/sec | Speedup vs. Level 0 | Perplexity |\n")
        f.write("|-------|-------------------|---------|------------|---------------------|------------|\n")
        
        for model_name in model_names:
            for opt_level in opt_levels:
                for pruning in pruning_levels:
                    key = (model_name, opt_level, pruning)
                    
                    if key in benchmark_lookup:
                        benchmark = benchmark_lookup[key]
                        
                        # Calculate speedup
                        speedup = "-"
                        baseline_key = (model_name, 0, pruning)
                        if opt_level > 0 and baseline_key in benchmark_lookup:
                            baseline_tps = benchmark_lookup[baseline_key]["avg_tokens_per_second"]
                            if baseline_tps > 0:
                                speedup = f"{benchmark['avg_tokens_per_second'] / baseline_tps:.2f}x"
                        
                        f.write(f"| {model_name} | Level {opt_level} | {pruning}% | {benchmark['avg_tokens_per_second']:.2f} | {speedup} | {benchmark['avg_perplexity']:.2f} |\n")
        
        f.write("\n### Best Configurations\n\n")
        
        # Table header
        f.write("| Model | Best Speed Config | Tokens/sec | Best Quality-Adjusted Config | Quality-Adjusted Score |\n")
        f.write("|-------|-------------------|------------|-----------------------------|------------------------|\n")
        
        for model_name, configs in best_configs.items():
            best_speed = configs["best_speed"]
            best_quality = configs["best_quality"]
            
            if best_speed and best_quality:
                best_speed_desc = f"Level {best_speed[0]}, {best_speed[1]}% pruning"
                best_quality_desc = f"Level {best_quality[0]}, {best_quality[1]}% pruning"
                
                f.write(f"| {model_name} | {best_speed_desc} | {configs['best_speed_tps']:.2f} | {best_quality_desc} | {configs['best_quality_adjusted']:.4f} |\n")
        
        f.write("\n## Detailed Analysis\n\n")
        
        for model_name in model_names:
            f.write(f"### {model_name}\n\n")
            
            # Get best configurations
            if model_name in best_configs:
                best_speed = best_configs[model_name]["best_speed"]
                best_quality = best_configs[model_name]["best_quality"]
                
                if best_speed:
                    best_speed_key = (model_name, best_speed[0], best_speed[1])
                    if best_speed_key in benchmark_lookup:
                        best_benchmark = benchmark_lookup[best_speed_key]
                        
                        f.write(f"**Best Speed Configuration:** Level {best_speed[0]}, {best_speed[1]}% pruning\n\n")
                        f.write(f"- Tokens per second: {best_benchmark['avg_tokens_per_second']:.2f}\n")
                        f.write(f"- First token latency: {best_benchmark['avg_first_token_latency']:.4f} seconds\n")
                        f.write(f"- Perplexity: {best_benchmark['avg_perplexity']:.2f}\n")
                        f.write(f"- Peak memory usage: {best_benchmark['peak_memory'] / (1024 * 1024):.2f} MB\n\n")
            
            # Optimization level impact
            f.write("**Optimization Level Impact (0% pruning):**\n\n")
            
            # Table header
            f.write("| Optimization Level | Tokens/sec | Speedup vs. Level 0 | Perplexity | First Token Latency |\n")
            f.write("|---------------------|------------|---------------------|------------|--------------------|\n")
            
            for opt_level in opt_levels:
                key = (model_name, opt_level, 0)  # 0% pruning
                
                if key in benchmark_lookup:
                    benchmark = benchmark_lookup[key]
                    
                    # Calculate speedup
                    speedup = "-"
                    baseline_key = (model_name, 0, 0)
                    if opt_level > 0 and baseline_key in benchmark_lookup:
                        baseline_tps = benchmark_lookup[baseline_key]["avg_tokens_per_second"]
                        if baseline_tps > 0:
                            speedup = f"{benchmark['avg_tokens_per_second'] / baseline_tps:.2f}x"
                    
                    f.write(f"| Level {opt_level} | {benchmark['avg_tokens_per_second']:.2f} | {speedup} | {benchmark['avg_perplexity']:.2f} | {benchmark['avg_first_token_latency']:.4f}s |\n")
            
            # Pruning impact (Level 3)
            f.write("\n**Pruning Impact (Level 3 optimization):**\n\n")
            
            # Table header
            f.write("| Pruning Level | Tokens/sec | Relative Speed | Perplexity | Memory Usage |\n")
            f.write("|---------------|------------|----------------|------------|-------------|\n")
            
            baseline_key = (model_name, 3, 0)  # 0% pruning with Level 3
            baseline_tps = 0
            if baseline_key in benchmark_lookup:
                baseline_tps = benchmark_lookup[baseline_key]["avg_tokens_per_second"]
            
            for pruning in pruning_levels:
                key = (model_name, 3, pruning)
                
                if key in benchmark_lookup:
                    benchmark = benchmark_lookup[key]
                    
                    # Calculate relative speed
                    relative_speed = "-"
                    if baseline_tps > 0:
                        relative_speed = f"{benchmark['avg_tokens_per_second'] / baseline_tps:.2f}x"
                    
                    memory_mb = benchmark['peak_memory'] / (1024 * 1024)
                    f.write(f"| {pruning}% | {benchmark['avg_tokens_per_second']:.2f} | {relative_speed} | {benchmark['avg_perplexity']:.2f} | {memory_mb:.2f} MB |\n")
            
            f.write("\n")
        
        f.write("## Recommendations\n\n")
        
        # Overall best configuration
        overall_best_model = None
        overall_best_config = None
        overall_best_tps = 0
        
        for model_name, configs in best_configs.items():
            if configs["best_speed_tps"] > overall_best_tps:
                overall_best_tps = configs["best_speed_tps"]
                overall_best_model = model_name
                overall_best_config = configs["best_speed"]
        
        if overall_best_model and overall_best_config:
            f.write(f"### Best Overall Configuration for Speed\n\n")
            f.write(f"- **Model:** {overall_best_model}\n")
            f.write(f"- **Optimization Level:** {overall_best_config[0]}\n")
            f.write(f"- **Pruning Level:** {overall_best_config[1]}%\n")
            f.write(f"- **Performance:** {overall_best_tps:.2f} tokens/sec\n\n")
        
        # Model-specific recommendations
        f.write("### Model-Specific Recommendations\n\n")
        
        for model_name in model_names:
            if model_name in best_configs and best_configs[model_name]["best_speed"]:
                best_speed = best_configs[model_name]["best_speed"]
                best_quality = best_configs[model_name]["best_quality"]
                
                f.write(f"**{model_name}:**\n\n")
                
                if best_speed == best_quality:
                    f.write(f"- **Recommended configuration:** Level {best_speed[0]}, {best_speed[1]}% pruning\n")
                    f.write(f"- This configuration provides the best balance of speed and quality\n\n")
                else:
                    f.write(f"- **For maximum speed:** Level {best_speed[0]}, {best_speed[1]}% pruning\n")
                    f.write(f"- **For best quality/speed ratio:** Level {best_quality[0]}, {best_quality[1]}% pruning\n\n")
        
        # General insights
        f.write("### General Insights\n\n")
        
        # Average speedup by optimization level
        avg_speedups = {opt_level: [] for opt_level in opt_levels if opt_level > 0}
        
        for model_name in model_names:
            if model_name in improvements:
                for opt_level in opt_levels:
                    if opt_level > 0 and opt_level in improvements[model_name]:
                        for pruning in pruning_levels:
                            if pruning in improvements[model_name][opt_level]:
                                avg_speedups[opt_level].append(
                                    improvements[model_name][opt_level][pruning]["speedup"]
                                )
        
        # Calculate averages
        for opt_level in avg_speedups:
            if avg_speedups[opt_level]:
                avg = sum(avg_speedups[opt_level]) / len(avg_speedups[opt_level])
                f.write(f"- **Optimization Level {opt_level}** provides an average {avg:.2f}x speedup across all models and pruning levels\n")
        
        f.write("\n- **Pruning Impact:** ")
        if pruning_levels:
            # Analyze average impact of pruning across all models and optimization levels
            pruning_impacts = {pruning: [] for pruning in pruning_levels if pruning > 0}
            
            for model_name in model_names:
                for opt_level in opt_levels:
                    # Use the highest optimization level as reference
                    if opt_level == max(opt_levels):
                        baseline_key = (model_name, opt_level, 0)  # 0% pruning
                        
                        if baseline_key in benchmark_lookup:
                            baseline_tps = benchmark_lookup[baseline_key]["avg_tokens_per_second"]
                            
                            for pruning in pruning_levels:
                                if pruning > 0:
                                    key = (model_name, opt_level, pruning)
                                    
                                    if key in benchmark_lookup:
                                        current_tps = benchmark_lookup[key]["avg_tokens_per_second"]
                                        
                                        if baseline_tps > 0:
                                            impact = current_tps / baseline_tps
                                            pruning_impacts[pruning].append(impact)
            
            # Calculate average impact
            avg_impacts = {}
            for pruning, impacts in pruning_impacts.items():
                if impacts:
                    avg_impacts[pruning] = sum(impacts) / len(impacts)
            
            if avg_impacts:
                impact_text = ", ".join([f"{pruning}% pruning: {avg_impacts[pruning]:.2f}x" for pruning in sorted(avg_impacts.keys())])
                f.write(f"Average speedup from pruning (relative to 0%): {impact_text}\n")
        
        f.write("\n### Command-Line Usage\n\n")
        
        if overall_best_model and overall_best_config:
            f.write("To use the best overall configuration:\n\n")
            f.write("```bash\n")
            f.write(f"python main.py --model_name {overall_best_model} --optimization_level {overall_best_config[0]} --prompt \"Your text here\"\n")
            f.write("```\n\n")
            
            # Add pruning command if applicable
            if overall_best_config[1] > 0:
                f.write("To enable pruning:\n\n")
                f.write("```bash\n")
                f.write(f"python scripts/inference_with_pruning.py --model_name {overall_best_model} --optimization_level {overall_best_config[0]} --pruning_level {overall_best_config[1]/100} --prompt \"Your text here\"\n")
                f.write("```\n\n")
        
        f.write("## Component Profiling\n\n")
        
        # Add component profiling data if available
        for model_name in model_names:
            key = (model_name, 3, 0)  # Level 3, 0% pruning
            
            if key in benchmark_lookup and "component_stats" in benchmark_lookup[key]:
                f.write(f"### {model_name} Component Breakdown\n\n")
                
                stats = benchmark_lookup[key]["component_stats"]
                total_time = stats.get("total_time", 1.0)  # Default to 1.0 to avoid division by zero
                
                f.write("| Component | Time (s) | Percentage |\n")
                f.write("|-----------|----------|------------|\n")
                
                for component, time_value in stats.items():
                    if component != "total_time":
                        percentage = (time_value / total_time) * 100
                        f.write(f"| {component.replace('_', ' ').title()} | {time_value:.4f} | {percentage:.2f}% |\n")
                
                f.write("\n")
    
    print(f"Benchmark report saved to {report_file}")
    
    # Return path to the report
    return report_file


def main():
    # Parse arguments
    args = setup_args()
    
    # Run benchmarks
    benchmark_results = run_all_benchmarks(args)
    
    # Create visualizations
    if args.visualize:
        create_visualizations(benchmark_results, args)
    
    # Generate report
    report_path = generate_benchmark_report(benchmark_results, args)
    
    print(f"\nBenchmark completed successfully. Report saved to: {report_path}")


if __name__ == "__main__":
    main()