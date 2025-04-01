#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Attention Optimization Profiling Tool

This script provides comprehensive profiling of the optimized attention mechanism
compared to the original implementation. It analyzes performance, memory usage,
and computational patterns to identify bottlenecks and verify optimizations.

Features:
- Detailed profiling of tensor operations
- GPU utilization analysis
- Memory usage tracking
- Isolated component testing
- Visualization of execution patterns
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
from torch.profiler import profile, record_function, ProfilerActivity

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from models.adaptive_transformer import GatedMultiHeadAttention
from models.optimized_attention import OptimizedGatedMultiHeadAttention
from models.loaders.loader import load_baseline_model, load_adaptive_model

def setup_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Profile attention optimization effectiveness")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="gpt2", 
                        help="Base model to use (default: gpt2)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (default: cuda if available, else cpu)")
    
    # Test parameters
    parser.add_argument("--seq_lengths", type=str, default="32,64,128,256,512",
                        help="Comma-separated sequence lengths to test")
    parser.add_argument("--batch_sizes", type=str, default="1,2,4,8", 
                        help="Comma-separated batch sizes to test")
    parser.add_argument("--pruning_levels", type=str, default="0,30,50,70",
                        help="Comma-separated pruning percentages to test")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of iterations for each test")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup iterations before measurement")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="profiling_results",
                        help="Directory to save profiling results")
    parser.add_argument("--detailed", action="store_true",
                        help="Run detailed profiling (slower but more comprehensive)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations of profiling results")
    
    return parser.parse_args()

def test_attention_modules():
    """
    Create and test attention modules directly to verify optimizations.
    """
    print("\n==== Testing Attention Modules Directly ====")
    
    # Configuration
    embed_dim = 768
    num_heads = 12
    head_dim = embed_dim // num_heads
    seq_lengths = [32, 64, 128, 256, 512]
    batch_size = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create modules
    original_attn = GatedMultiHeadAttention(embed_dim, num_heads).to(device)
    optimized_attn = OptimizedGatedMultiHeadAttention(embed_dim, num_heads).to(device)
    
    # Set to evaluation mode
    original_attn.eval()
    optimized_attn.eval()
    
    results = {
        "operation": "direct_attention",
        "original": {},
        "optimized": {},
        "speedup": {}
    }
    
    # Test with different sequence lengths
    for seq_len in seq_lengths:
        print(f"\nTesting with sequence length: {seq_len}")
        
        # Create input tensor (matching dimensions expected by the attention modules)
        hidden_states = torch.randn(batch_size, seq_len, embed_dim, device=device)
        
        # Warm up
        for _ in range(5):
            with torch.no_grad():
                _ = original_attn(hidden_states)
                _ = optimized_attn(hidden_states)
        
        # Benchmark original attention
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = original_attn(hidden_states)
        torch.cuda.synchronize()
        original_time = time.time() - start_time
        
        # Benchmark optimized attention
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = optimized_attn(hidden_states)
        torch.cuda.synchronize()
        optimized_time = time.time() - start_time
        
        # Calculate speedup
        speedup = original_time / optimized_time
        
        print(f"  Original:  {original_time:.6f}s")
        print(f"  Optimized: {optimized_time:.6f}s")
        print(f"  Speedup:   {speedup:.2f}x")
        
        # Store results
        results["original"][seq_len] = original_time
        results["optimized"][seq_len] = optimized_time
        results["speedup"][seq_len] = speedup
    
    return results

def run_detailed_profiling(args):
    """
    Run detailed profiling of attention mechanisms using PyTorch Profiler.
    """
    print("\n==== Running Detailed Profiling ====")
    device = args.device
    
    # Create modules
    embed_dim = 768
    num_heads = 12
    seq_len = 128
    batch_size = 2
    
    original_attn = GatedMultiHeadAttention(embed_dim, num_heads).to(device)
    optimized_attn = OptimizedGatedMultiHeadAttention(embed_dim, num_heads).to(device)
    
    # Set to evaluation mode
    original_attn.eval()
    optimized_attn.eval()
    
    # Create input tensor
    hidden_states = torch.randn(batch_size, seq_len, embed_dim, device=device)
    
    # Warm up
    for _ in range(5):
        with torch.no_grad():
            _ = original_attn(hidden_states)
            _ = optimized_attn(hidden_states)
    
    # Profile original attention
    print("Profiling original attention...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        profile_memory=True
    ) as prof_original:
        with record_function("original_attention"):
            with torch.no_grad():
                for _ in range(10):
                    _ = original_attn(hidden_states)
    
    # Profile optimized attention
    print("Profiling optimized attention...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        profile_memory=True
    ) as prof_optimized:
        with record_function("optimized_attention"):
            with torch.no_grad():
                for _ in range(10):
                    _ = optimized_attn(hidden_states)
    
    # Save profiling results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Export and save profiling results
    original_trace_path = os.path.join(args.output_dir, "original_attention_trace.json")
    optimized_trace_path = os.path.join(args.output_dir, "optimized_attention_trace.json")
    
    # Print summary tables
    print("\nOriginal Attention Profile:")
    print(prof_original.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    print("\nOptimized Attention Profile:")
    print(prof_optimized.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Export traces
    print(f"Exporting trace to {original_trace_path}")
    prof_original.export_chrome_trace(original_trace_path)
    
    print(f"Exporting trace to {optimized_trace_path}")
    prof_optimized.export_chrome_trace(optimized_trace_path)
    
    return {
        "original_profile": prof_original.key_averages().table(sort_by="cuda_time_total", row_limit=20),
        "optimized_profile": prof_optimized.key_averages().table(sort_by="cuda_time_total", row_limit=20),
        "original_trace_path": original_trace_path,
        "optimized_trace_path": optimized_trace_path
    }

def test_scaling_behavior(args):
    """
    Test how both implementations scale with sequence length and batch size.
    """
    print("\n==== Testing Scaling Behavior ====")
    device = args.device
    
    # Parse parameters
    seq_lengths = [int(x) for x in args.seq_lengths.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    
    # Configuration
    embed_dim = 768
    num_heads = 12
    
    # Create modules
    original_attn = GatedMultiHeadAttention(embed_dim, num_heads).to(device)
    optimized_attn = OptimizedGatedMultiHeadAttention(embed_dim, num_heads).to(device)
    
    # Set to evaluation mode
    original_attn.eval()
    optimized_attn.eval()
    
    results = {
        "operation": "scaling_test",
        "seq_length_scaling": {},
        "batch_size_scaling": {}
    }
    
    # Test scaling with sequence length
    print("\nTesting scaling with sequence length:")
    batch_size = 2  # Fixed batch size
    for seq_len in seq_lengths:
        print(f"  Sequence length: {seq_len}")
        
        # Create input tensor
        hidden_states = torch.randn(batch_size, seq_len, embed_dim, device=device)
        
        # Warm up
        for _ in range(args.warmup):
            with torch.no_grad():
                _ = original_attn(hidden_states)
                _ = optimized_attn(hidden_states)
        
        # Benchmark original attention
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(args.iterations):
                _ = original_attn(hidden_states)
        torch.cuda.synchronize()
        original_time = time.time() - start_time
        
        # Benchmark optimized attention
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(args.iterations):
                _ = optimized_attn(hidden_states)
        torch.cuda.synchronize()
        optimized_time = time.time() - start_time
        
        # Calculate speedup
        speedup = original_time / optimized_time
        
        print(f"    Original:  {original_time:.6f}s")
        print(f"    Optimized: {optimized_time:.6f}s")
        print(f"    Speedup:   {speedup:.2f}x")
        
        # Store results
        results["seq_length_scaling"][seq_len] = {
            "original": original_time,
            "optimized": optimized_time,
            "speedup": speedup
        }
    
    # Test scaling with batch size
    print("\nTesting scaling with batch size:")
    seq_len = 128  # Fixed sequence length
    for batch_size in batch_sizes:
        print(f"  Batch size: {batch_size}")
        
        # Create input tensor
        hidden_states = torch.randn(batch_size, seq_len, embed_dim, device=device)
        
        # Warm up
        for _ in range(args.warmup):
            with torch.no_grad():
                _ = original_attn(hidden_states)
                _ = optimized_attn(hidden_states)
        
        # Benchmark original attention
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(args.iterations):
                _ = original_attn(hidden_states)
        torch.cuda.synchronize()
        original_time = time.time() - start_time
        
        # Benchmark optimized attention
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(args.iterations):
                _ = optimized_attn(hidden_states)
        torch.cuda.synchronize()
        optimized_time = time.time() - start_time
        
        # Calculate speedup
        speedup = original_time / optimized_time
        
        print(f"    Original:  {original_time:.6f}s")
        print(f"    Optimized: {optimized_time:.6f}s")
        print(f"    Speedup:   {speedup:.2f}x")
        
        # Store results
        results["batch_size_scaling"][batch_size] = {
            "original": original_time,
            "optimized": optimized_time,
            "speedup": speedup
        }
    
    return results

def analyze_memory_usage(args):
    """
    Analyze memory usage patterns of both implementations.
    """
    print("\n==== Analyzing Memory Usage ====")
    device = args.device
    
    if device != "cuda":
        print("Memory usage analysis requires CUDA. Skipping...")
        return {"error": "Memory usage analysis requires CUDA"}
    
    # Configuration
    embed_dim = 768
    num_heads = 12
    seq_len = 256
    batch_size = 4
    
    # Create modules
    original_attn = GatedMultiHeadAttention(embed_dim, num_heads).to(device)
    optimized_attn = OptimizedGatedMultiHeadAttention(embed_dim, num_heads).to(device)
    
    # Set to evaluation mode
    original_attn.eval()
    optimized_attn.eval()
    
    # Create input tensor
    hidden_states = torch.randn(batch_size, seq_len, embed_dim, device=device)
    
    # Measure baseline memory usage
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    baseline_memory = torch.cuda.memory_allocated()
    
    # Measure memory for original attention
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    with torch.no_grad():
        _ = original_attn(hidden_states)
    original_memory = torch.cuda.max_memory_allocated() - baseline_memory
    
    # Measure memory for optimized attention
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    with torch.no_grad():
        _ = optimized_attn(hidden_states)
    optimized_memory = torch.cuda.max_memory_allocated() - baseline_memory
    
    # Calculate memory efficiency
    memory_ratio = optimized_memory / original_memory
    
    print(f"Original attention memory usage:  {original_memory / (1024**2):.2f} MB")
    print(f"Optimized attention memory usage: {optimized_memory / (1024**2):.2f} MB")
    print(f"Memory ratio (optimized/original): {memory_ratio:.2f}x")
    
    return {
        "operation": "memory_usage",
        "original_memory": float(original_memory),
        "optimized_memory": float(optimized_memory),
        "memory_ratio": float(memory_ratio)
    }

def visualize_results(results, args):
    """
    Create visualizations from profiling results.
    """
    print("\n==== Creating Visualizations ====")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualization 1: Direct Attention Test
    if "operation" in results["direct_attention"] and results["direct_attention"]["operation"] == "direct_attention":
        data = results["direct_attention"]
        seq_lengths = list(data["original"].keys())
        
        # Sort sequence lengths
        seq_lengths = sorted([int(x) for x in seq_lengths])
        
        # Extract values
        original_times = [data["original"][length] for length in seq_lengths]
        optimized_times = [data["optimized"][length] for length in seq_lengths]
        speedups = [data["speedup"][length] for length in seq_lengths]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot times
        plt.subplot(1, 2, 1)
        plt.plot(seq_lengths, original_times, '-o', label="Original")
        plt.plot(seq_lengths, optimized_times, '-o', label="Optimized")
        plt.xlabel("Sequence Length")
        plt.ylabel("Time (seconds)")
        plt.title("Attention Execution Time")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Plot speedup
        plt.subplot(1, 2, 2)
        plt.bar(seq_lengths, speedups)
        plt.axhline(y=1.0, color='r', linestyle='--')
        plt.xlabel("Sequence Length")
        plt.ylabel("Speedup (x)")
        plt.title("Attention Optimization Speedup")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add annotations
        for i, v in enumerate(speedups):
            plt.text(seq_lengths[i], v + 0.05, f"{v:.2f}x", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "direct_attention_test.png"))
        plt.close()
    
    # Visualization 2: Scaling Behavior
    if "operation" in results["scaling"] and results["scaling"]["operation"] == "scaling_test":
        data = results["scaling"]
        
        # Extract scaling data
        seq_lengths = sorted([int(x) for x in data["seq_length_scaling"].keys()])
        batch_sizes = sorted([int(x) for x in data["batch_size_scaling"].keys()])
        
        # Sequence length scaling data
        seq_original = [data["seq_length_scaling"][length]["original"] for length in seq_lengths]
        seq_optimized = [data["seq_length_scaling"][length]["optimized"] for length in seq_lengths]
        seq_speedups = [data["seq_length_scaling"][length]["speedup"] for length in seq_lengths]
        
        # Batch size scaling data
        batch_original = [data["batch_size_scaling"][size]["original"] for size in batch_sizes]
        batch_optimized = [data["batch_size_scaling"][size]["optimized"] for size in batch_sizes]
        batch_speedups = [data["batch_size_scaling"][size]["speedup"] for size in batch_sizes]
        
        # Create sequence length scaling plot
        plt.figure(figsize=(12, 10))
        
        # Plot sequence length times
        plt.subplot(2, 2, 1)
        plt.plot(seq_lengths, seq_original, '-o', label="Original")
        plt.plot(seq_lengths, seq_optimized, '-o', label="Optimized")
        plt.xlabel("Sequence Length")
        plt.ylabel("Time (seconds)")
        plt.title("Scaling with Sequence Length")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Plot sequence length speedup
        plt.subplot(2, 2, 2)
        plt.bar(seq_lengths, seq_speedups)
        plt.axhline(y=1.0, color='r', linestyle='--')
        plt.xlabel("Sequence Length")
        plt.ylabel("Speedup (x)")
        plt.title("Speedup vs. Sequence Length")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add annotations
        for i, v in enumerate(seq_speedups):
            plt.text(seq_lengths[i], v + 0.05, f"{v:.2f}x", ha='center')
        
        # Plot batch size times
        plt.subplot(2, 2, 3)
        plt.plot(batch_sizes, batch_original, '-o', label="Original")
        plt.plot(batch_sizes, batch_optimized, '-o', label="Optimized")
        plt.xlabel("Batch Size")
        plt.ylabel("Time (seconds)")
        plt.title("Scaling with Batch Size")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Plot batch size speedup
        plt.subplot(2, 2, 4)
        plt.bar(batch_sizes, batch_speedups)
        plt.axhline(y=1.0, color='r', linestyle='--')
        plt.xlabel("Batch Size")
        plt.ylabel("Speedup (x)")
        plt.title("Speedup vs. Batch Size")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add annotations
        for i, v in enumerate(batch_speedups):
            plt.text(batch_sizes[i], v + 0.05, f"{v:.2f}x", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "scaling_behavior.png"))
        plt.close()
    
    # Visualization 3: Memory Usage
    if "operation" in results["memory"] and results["memory"]["operation"] == "memory_usage":
        data = results["memory"]
        
        # Extract memory data
        original_memory = data["original_memory"] / (1024**2)
        optimized_memory = data["optimized_memory"] / (1024**2)
        memory_ratio = data["memory_ratio"]
        
        # Create memory usage plot
        plt.figure(figsize=(8, 6))
        
        # Plot memory usage
        bars = plt.bar(["Original", "Optimized"], [original_memory, optimized_memory])
        plt.ylabel("Memory Usage (MB)")
        plt.title("Attention Memory Usage Comparison")
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f"{height:.1f} MB", ha='center', va='bottom')
        
        # Add ratio annotation
        ratio_text = f"Memory Ratio: {memory_ratio:.2f}x"
        if memory_ratio < 1.0:
            ratio_text += f" ({(1-memory_ratio)*100:.1f}% less memory)"
        else:
            ratio_text += f" ({(memory_ratio-1)*100:.1f}% more memory)"
        
        plt.annotate(ratio_text, xy=(0.5, 0.02), xycoords='figure fraction',
                    ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "memory_usage.png"))
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def main():
    """Main function."""
    args = setup_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Store all results
    results = {}
    
    # Run direct attention tests
    results["direct_attention"] = test_attention_modules()
    
    # Run scaling behavior tests
    results["scaling"] = test_scaling_behavior(args)
    
    # Analyze memory usage
    results["memory"] = analyze_memory_usage(args)
    
    # Run detailed profiling if requested
    if args.detailed:
        results["profiling"] = run_detailed_profiling(args)
    
    # Save results
    results_file = os.path.join(args.output_dir, "profiling_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Create visualizations if requested
    if args.visualize:
        visualize_results(results, args)
    
    # Print summary
    print("\n===== Profiling Summary =====")
    print(f"Direct Attention Tests:")
    for seq_len, speedup in results["direct_attention"]["speedup"].items():
        print(f"  Sequence Length {seq_len}: {speedup:.2f}x speedup")
    
    print("\nScaling with Sequence Length:")
    for seq_len, data in results["scaling"]["seq_length_scaling"].items():
        print(f"  Sequence Length {seq_len}: {data['speedup']:.2f}x speedup")
    
    print("\nScaling with Batch Size:")
    for batch_size, data in results["scaling"]["batch_size_scaling"].items():
        print(f"  Batch Size {batch_size}: {data['speedup']:.2f}x speedup")
    
    print("\nMemory Usage:")
    memory_ratio = results["memory"]["memory_ratio"]
    if memory_ratio < 1.0:
        print(f"  Optimized model uses {(1-memory_ratio)*100:.1f}% less memory")
    else:
        print(f"  Optimized model uses {(memory_ratio-1)*100:.1f}% more memory")

if __name__ == "__main__":
    main()