#!/usr/bin/env python
"""
Benchmark the Adaptive Transformer against baseline models.

This script benchmarks the performance of the adaptive transformer model against
baseline models in terms of:
1. Generation quality (perplexity, BLEU, etc.)
2. Inference speed (tokens per second)
3. Memory usage
4. Parameter efficiency

Usage:
    python scripts/benchmark.py --model_path /path/to/checkpoint.pth \
                              --baseline_model gpt2 \
                              --dataset wikitext \
                              --batch_sizes 1 4 8 16

Features:
- Comprehensive benchmarking across multiple metrics
- Support for different batch sizes
- Memory profiling
- Generation quality assessment
"""

import os
import sys
import argparse
import torch
import json
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from models.loaders.loader import load_baseline_model, load_adaptive_model
from utils.checkpoint import load_checkpoint
from utils.generation_wrapper import GenerationWrapper


def measure_perplexity(model, tokenizer, dataset, device, max_samples=1000):
    """
    Measure perplexity of the model on a dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        dataset: HuggingFace dataset
        device: Device to run evaluation on
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        Mean perplexity over the dataset
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset, desc="Evaluating perplexity")):
            if i >= max_samples:
                break
                
            # Truncate to a reasonable length
            input_text = sample['text'][:1024]
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            
            # Forward pass
            outputs = model(**inputs) if hasattr(model, "forward") and not hasattr(model, "blocks") else model(inputs.input_ids)
            
            # Calculate loss
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs
                
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs.input_ids[:, 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
    
    # Calculate perplexity
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return perplexity


def benchmark_generation_speed(model, tokenizer, prompts, device, batch_sizes=[1, 2, 4, 8], 
                             max_length=50, num_runs=3):
    """
    Benchmark generation speed across different batch sizes.
    
    Args:
        model: Model to benchmark
        tokenizer: Tokenizer for the model
        prompts: List of prompts to use for generation
        device: Device to run benchmarking on
        batch_sizes: List of batch sizes to benchmark
        max_length: Maximum length of generated sequence
        num_runs: Number of runs to average over
        
    Returns:
        Dictionary with benchmarking results
    """
    results = {}
    
    # Create generation wrapper
    wrapper = GenerationWrapper(model=model, tokenizer=tokenizer, device=device)
    
    for batch_size in batch_sizes:
        # Check if batch size exceeds available prompts
        if batch_size > len(prompts):
            continue
            
        batch_times = []
        
        for _ in range(num_runs):
            # Sample batch_size prompts
            batch_prompts = np.random.choice(prompts, size=batch_size, replace=False)
            
            # Time generation
            start_time = time.time()
            
            _ = wrapper.run_inference(
                batch_prompts,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                num_return_sequences=1
            )
            
            end_time = time.time()
            batch_times.append(end_time - start_time)
        
        # Calculate statistics
        mean_time = np.mean(batch_times)
        tokens_per_second = batch_size * max_length / mean_time
        
        results[batch_size] = {
            "mean_time_seconds": float(mean_time),
            "tokens_per_second": float(tokens_per_second),
            "samples_per_second": float(batch_size / mean_time)
        }
    
    return results


def profile_memory_usage(model, tokenizer, prompts, device, batch_sizes=[1, 2, 4, 8],
                       max_length=50):
    """
    Profile memory usage during generation.
    
    Args:
        model: Model to profile
        tokenizer: Tokenizer for the model
        prompts: List of prompts to use for generation
        device: Device to run profiling on
        batch_sizes: List of batch sizes to profile
        max_length: Maximum length of generated sequence
        
    Returns:
        Dictionary with memory usage results
    """
    if device == "cpu" or not torch.cuda.is_available():
        return {"error": "Memory profiling only available with CUDA devices"}
    
    results = {}
    wrapper = GenerationWrapper(model=model, tokenizer=tokenizer, device=device)
    
    for batch_size in batch_sizes:
        # Check if batch size exceeds available prompts
        if batch_size > len(prompts):
            continue
            
        # Sample batch_size prompts
        batch_prompts = np.random.choice(prompts, size=batch_size, replace=False)
        
        # Reset GPU memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Run generation
        _ = wrapper.run_inference(
            batch_prompts,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            num_return_sequences=1
        )
        
        # Record memory stats
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        
        results[batch_size] = {
            "peak_memory_mb": float(peak_memory),
            "memory_per_batch_item_mb": float(peak_memory / batch_size)
        }
    
    return results


def count_parameters(model):
    """
    Count parameters in a model, distinguishing between active and inactive.
    
    Args:
        model: Model to count parameters for
        
    Returns:
        Dictionary with parameter counts
    """
    if not hasattr(model, "blocks"):
        # Standard HuggingFace model
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "frozen_params": int(total_params - trainable_params)
        }
    else:
        # Adaptive model
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Count active heads and parameters
        active_heads = 0
        active_head_params = 0
        
        for block in model.blocks:
            attn_module = block["attn"]
            
            for head_idx in range(attn_module.num_heads):
                if attn_module.gate[head_idx].item() > 0.1:
                    active_heads += 1
                    
                    # Count params in this head
                    for param in list(attn_module.W_q[head_idx].parameters()) + \
                               list(attn_module.W_k[head_idx].parameters()) + \
                               list(attn_module.W_v[head_idx].parameters()) + \
                               list(attn_module.W_o[head_idx].parameters()):
                        active_head_params += param.numel()
        
        return {
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "frozen_params": int(total_params - trainable_params),
            "active_heads": int(active_heads),
            "active_head_params": int(active_head_params)
        }


def plot_benchmark_results(adaptive_results, baseline_results, output_dir):
    """
    Plot benchmark results for comparison.
    
    Args:
        adaptive_results: Dictionary with adaptive model results
        baseline_results: Dictionary with baseline model results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot generation speed comparison
    if "generation_speed" in adaptive_results and "generation_speed" in baseline_results:
        plt.figure(figsize=(10, 6))
        
        # Extract batch sizes and tokens per second
        batch_sizes_adaptive = list(adaptive_results["generation_speed"].keys())
        tokens_per_second_adaptive = [adaptive_results["generation_speed"][bs]["tokens_per_second"] 
                                    for bs in batch_sizes_adaptive]
        
        batch_sizes_baseline = list(baseline_results["generation_speed"].keys())
        tokens_per_second_baseline = [baseline_results["generation_speed"][bs]["tokens_per_second"] 
                                    for bs in batch_sizes_baseline]
        
        plt.plot(batch_sizes_adaptive, tokens_per_second_adaptive, 'o-', label="Adaptive Model")
        plt.plot(batch_sizes_baseline, tokens_per_second_baseline, 's-', label="Baseline Model")
        
        plt.title("Generation Speed Comparison")
        plt.xlabel("Batch Size")
        plt.ylabel("Tokens per Second")
        plt.legend()
        plt.grid()
        
        plt.savefig(os.path.join(output_dir, "generation_speed.png"))
        plt.close()
    
    # Plot memory usage comparison
    if "memory_usage" in adaptive_results and "memory_usage" in baseline_results:
        plt.figure(figsize=(10, 6))
        
        # Extract batch sizes and memory usage
        batch_sizes_adaptive = list(adaptive_results["memory_usage"].keys())
        memory_usage_adaptive = [adaptive_results["memory_usage"][bs]["peak_memory_mb"] 
                               for bs in batch_sizes_adaptive]
        
        batch_sizes_baseline = list(baseline_results["memory_usage"].keys())
        memory_usage_baseline = [baseline_results["memory_usage"][bs]["peak_memory_mb"] 
                               for bs in batch_sizes_baseline]
        
        plt.plot(batch_sizes_adaptive, memory_usage_adaptive, 'o-', label="Adaptive Model")
        plt.plot(batch_sizes_baseline, memory_usage_baseline, 's-', label="Baseline Model")
        
        plt.title("Memory Usage Comparison")
        plt.xlabel("Batch Size")
        plt.ylabel("Peak Memory (MB)")
        plt.legend()
        plt.grid()
        
        plt.savefig(os.path.join(output_dir, "memory_usage.png"))
        plt.close()
    
    # Plot parameter efficiency
    if "parameter_counts" in adaptive_results:
        plt.figure(figsize=(10, 6))
        
        params_data = [
            adaptive_results["parameter_counts"]["active_head_params"],
            adaptive_results["parameter_counts"]["total_params"] - adaptive_results["parameter_counts"]["active_head_params"],
            baseline_results["parameter_counts"]["total_params"]
        ]
        
        labels = [
            "Adaptive: Active Head Parameters",
            "Adaptive: Other Parameters",
            "Baseline: Total Parameters"
        ]
        
        colors = ['#2ca02c', '#1f77b4', '#d62728']
        
        plt.bar(range(len(params_data)), params_data, color=colors)
        plt.xticks(range(len(params_data)), labels, rotation=45, ha="right")
        
        plt.title("Parameter Efficiency Comparison")
        plt.ylabel("Number of Parameters")
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "parameter_efficiency.png"))
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Adaptive Transformer vs Baseline")
    parser.add_argument("--model_path", type=str, required=True, help="Path to adaptive model checkpoint")
    parser.add_argument("--baseline_model", type=str, default="gpt2", help="Baseline model name")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset name for evaluation")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset configuration")
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4, 8], help="Batch sizes to benchmark")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results/", help="Output directory for results")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum generation length")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum dataset samples to use")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of runs to average over")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None,
                      help="Device to use (defaults to CUDA if available)")
    args = parser.parse_args()
    
    # Determine device
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset {args.dataset} ({args.dataset_config})...")
    try:
        dataset = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split)
        print(f"Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using a simple text dataset instead")
        # Create a simple dataset with prompts
        prompts = [
            "Once upon a time in a land far away",
            "The scientist made a discovery that would change",
            "The most important thing to remember about artificial intelligence is",
            "When I look back on my life, I realize",
            "The future of technology depends on our ability to",
            "In a world where everything is connected"
        ]
        dataset = [{"text": prompt} for prompt in prompts]
    
    # Load baseline model
    print(f"Loading baseline model {args.baseline_model}...")
    baseline_tokenizer = AutoTokenizer.from_pretrained(args.baseline_model)
    baseline_model = load_baseline_model(args.baseline_model, device)
    
    # Load adaptive model
    print(f"Loading adaptive model from {args.model_path}...")
    adaptive_model = load_adaptive_model(args.baseline_model, baseline_model, device)
    
    if os.path.exists(args.model_path):
        optimizer = torch.optim.AdamW(adaptive_model.parameters())
        head_lr_multipliers = {}
        adaptive_model, _, _, _, _ = load_checkpoint(
            adaptive_model, optimizer, head_lr_multipliers, args.model_path, device)
        print(f"Loaded checkpoint from {args.model_path}")
    
    # Set up prompts for generation benchmarks
    prompts = [sample["text"][:100] for sample in dataset[:50]]
    
    # Initialize results dictionaries
    adaptive_results = {}
    baseline_results = {}
    
    # Benchmark 1: Perplexity
    print("\nMeasuring perplexity...")
    
    try:
        adaptive_perplexity = measure_perplexity(
            adaptive_model, baseline_tokenizer, dataset, device, max_samples=args.max_samples)
        adaptive_results["perplexity"] = adaptive_perplexity
        print(f"Adaptive model perplexity: {adaptive_perplexity:.2f}")
        
        baseline_perplexity = measure_perplexity(
            baseline_model, baseline_tokenizer, dataset, device, max_samples=args.max_samples)
        baseline_results["perplexity"] = baseline_perplexity
        print(f"Baseline model perplexity: {baseline_perplexity:.2f}")
    except Exception as e:
        print(f"Error measuring perplexity: {e}")
    
    # Benchmark 2: Generation Speed
    print("\nBenchmarking generation speed...")
    
    try:
        adaptive_speed = benchmark_generation_speed(
            adaptive_model, baseline_tokenizer, prompts, device, 
            batch_sizes=args.batch_sizes, max_length=args.max_length, num_runs=args.num_runs)
        adaptive_results["generation_speed"] = adaptive_speed
        
        baseline_speed = benchmark_generation_speed(
            baseline_model, baseline_tokenizer, prompts, device,
            batch_sizes=args.batch_sizes, max_length=args.max_length, num_runs=args.num_runs)
        baseline_results["generation_speed"] = baseline_speed
        
        print("Generation speed results:")
        for batch_size in args.batch_sizes:
            if batch_size in adaptive_speed and batch_size in baseline_speed:
                adaptive_tps = adaptive_speed[batch_size]["tokens_per_second"]
                baseline_tps = baseline_speed[batch_size]["tokens_per_second"]
                speedup = (adaptive_tps / baseline_tps - 1) * 100
                
                print(f"  Batch size {batch_size}:")
                print(f"    Adaptive: {adaptive_tps:.2f} tokens/sec")
                print(f"    Baseline: {baseline_tps:.2f} tokens/sec")
                print(f"    {'Speedup' if speedup > 0 else 'Slowdown'}: {abs(speedup):.2f}%")
    except Exception as e:
        print(f"Error benchmarking generation speed: {e}")
    
    # Benchmark 3: Memory Usage
    if device.type == "cuda":
        print("\nProfiling memory usage...")
        
        try:
            adaptive_memory = profile_memory_usage(
                adaptive_model, baseline_tokenizer, prompts, device,
                batch_sizes=args.batch_sizes, max_length=args.max_length)
            adaptive_results["memory_usage"] = adaptive_memory
            
            baseline_memory = profile_memory_usage(
                baseline_model, baseline_tokenizer, prompts, device,
                batch_sizes=args.batch_sizes, max_length=args.max_length)
            baseline_results["memory_usage"] = baseline_memory
            
            print("Memory usage results:")
            for batch_size in args.batch_sizes:
                if batch_size in adaptive_memory and batch_size in baseline_memory:
                    adaptive_mem = adaptive_memory[batch_size]["peak_memory_mb"]
                    baseline_mem = baseline_memory[batch_size]["peak_memory_mb"]
                    memory_saved = (baseline_mem - adaptive_mem) / baseline_mem * 100
                    
                    print(f"  Batch size {batch_size}:")
                    print(f"    Adaptive: {adaptive_mem:.2f} MB")
                    print(f"    Baseline: {baseline_mem:.2f} MB")
                    print(f"    Memory saved: {memory_saved:.2f}%")
        except Exception as e:
            print(f"Error profiling memory usage: {e}")
    
    # Benchmark 4: Parameter Counts
    print("\nCounting parameters...")
    
    adaptive_params = count_parameters(adaptive_model)
    baseline_params = count_parameters(baseline_model)
    
    adaptive_results["parameter_counts"] = adaptive_params
    baseline_results["parameter_counts"] = baseline_params
    
    print("Parameter counts:")
    print(f"  Baseline model: {baseline_params['total_params']:,} parameters")
    print(f"  Adaptive model: {adaptive_params['total_params']:,} parameters")
    print(f"    Active heads: {adaptive_params.get('active_heads', 'N/A')}")
    print(f"    Active head parameters: {adaptive_params.get('active_head_params', 'N/A'):,}")
    
    # Save results
    print("\nSaving results...")
    
    with open(os.path.join(args.output_dir, "adaptive_results.json"), "w") as f:
        json.dump(adaptive_results, f, indent=2)
    
    with open(os.path.join(args.output_dir, "baseline_results.json"), "w") as f:
        json.dump(baseline_results, f, indent=2)
    
    # Plot results
    print("\nPlotting comparison results...")
    plot_benchmark_results(adaptive_results, baseline_results, args.output_dir)
    
    print(f"\nBenchmarking complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()