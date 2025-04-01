#!/usr/bin/env python

"""
Sentinel AI Improvement Demonstration

This script demonstrates the concrete improvements of the Sentinel AI model
over the base GPT-2 model in three key areas:
1. Speed: Show faster inference with optimized pathways
2. Efficiency: Demonstrate comparable quality with fewer resources
3. Adaptability: Show how pruning and optimization can be tuned

The demonstrations use real-world prompts and measure actual performance
improvements from the optimizations we've implemented.
"""

import os
import sys
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel

# Add project root to the path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Import model loaders
from models.loaders.loader import load_baseline_model, load_adaptive_model
from models.loaders.loader_optimized import load_optimized_adaptive_model
from scripts.pruning_comparison.pruning_agency_comparison import apply_pruning


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Demonstrate improvements over base GPT-2")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="gpt2", 
                       help="Name of the model to test (default: gpt2)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run on (default: cpu; use cuda for GPU)")
    
    # Test configuration
    parser.add_argument("--tokens", type=int, default=50,
                       help="Number of tokens to generate for each test")
    parser.add_argument("--iterations", type=int, default=3,
                       help="Number of iterations to run for consistent results")
    parser.add_argument("--pruning", type=float, default=0.5,
                       help="Percentage of heads to prune (0.0-1.0)")
    parser.add_argument("--optimization_level", type=int, default=3,
                       help="Optimization level to use (0-3)")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="demo_results",
                       help="Directory to save results")
    
    return parser.parse_args()


def ensure_output_dir(directory):
    """Ensure the output directory exists."""
    os.makedirs(directory, exist_ok=True)
    return directory


def load_test_prompts():
    """Load test prompts for the demonstration."""
    return [
        "Write a paragraph explaining artificial intelligence in simple terms.",
        "Explain how transformer neural networks function.",
        "Write a poem about machine learning and creativity.",
        "Describe the key challenges of natural language processing.",
        "Summarize the history of deep learning in a few sentences."
    ]


def measure_baseline_performance(model_name, device, tokens, iterations, prompts):
    """Measure performance of the baseline GPT-2 model."""
    print("\n=== Measuring Baseline Performance ===")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    
    results = []
    
    # Run through all prompts
    for prompt in prompts:
        prompt_results = []
        
        # Run multiple iterations
        for _ in range(iterations):
            # Tokenize prompt
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            # Time the generation
            start_time = time.time()
            
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_length=input_ids.size(1) + tokens,
                    do_sample=True,
                    temperature=0.7
                )
            
            end_time = time.time()
            
            # Calculate metrics
            generation_time = end_time - start_time
            tokens_generated = output.size(1) - input_ids.size(1)
            tokens_per_second = tokens_generated / generation_time
            
            # Calculate perplexity
            with torch.no_grad():
                outputs = model(output)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = output[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                perplexity = torch.exp(loss).item()
            
            # Get generated text
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            prompt_results.append({
                "time": generation_time,
                "tokens": tokens_generated,
                "tokens_per_second": tokens_per_second,
                "perplexity": perplexity,
                "text": generated_text
            })
        
        # Average the results for this prompt
        avg_time = sum(r["time"] for r in prompt_results) / len(prompt_results)
        avg_tokens_per_second = sum(r["tokens_per_second"] for r in prompt_results) / len(prompt_results)
        avg_perplexity = sum(r["perplexity"] for r in prompt_results) / len(prompt_results)
        
        results.append({
            "prompt": prompt,
            "avg_time": avg_time,
            "avg_tokens_per_second": avg_tokens_per_second,
            "avg_perplexity": avg_perplexity,
            "iterations": prompt_results
        })
        
        print(f"Prompt: {prompt[:30]}...")
        print(f"  Time: {avg_time:.2f}s")
        print(f"  Speed: {avg_tokens_per_second:.2f} tokens/sec")
        print(f"  Perplexity: {avg_perplexity:.2f}")
    
    # Calculate overall averages
    overall_time = sum(r["avg_time"] for r in results) / len(results)
    overall_speed = sum(r["avg_tokens_per_second"] for r in results) / len(results)
    overall_perplexity = sum(r["avg_perplexity"] for r in results) / len(results)
    
    print("\nBaseline Overall Performance:")
    print(f"  Avg. Time: {overall_time:.2f}s")
    print(f"  Avg. Speed: {overall_speed:.2f} tokens/sec")
    print(f"  Avg. Perplexity: {overall_perplexity:.2f}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return {
        "model": "baseline",
        "overall": {
            "time": overall_time,
            "speed": overall_speed,
            "perplexity": overall_perplexity
        },
        "prompt_results": results
    }


def measure_optimized_performance(model_name, device, tokens, iterations, prompts, pruning_percentage, opt_level):
    """Measure performance of the optimized model with pruning."""
    print(f"\n=== Measuring Optimized Performance (Pruning: {pruning_percentage*100}%, Opt Level: {opt_level}) ===")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load baseline model first
    baseline_model = load_baseline_model(model_name, device)
    
    # Load optimized model
    print(f"Loading optimized model with level {opt_level}...")
    if opt_level > 0:
        model = load_optimized_adaptive_model(
            model_name, 
            baseline_model, 
            device,
            optimization_level=opt_level
        )
    else:
        model = load_adaptive_model(model_name, baseline_model, device)
    
    # Apply pruning if needed
    if pruning_percentage > 0:
        print(f"Applying {pruning_percentage*100}% pruning...")
        model, pruned_count, _ = apply_pruning(model, pruning_percentage, verbose=False, quiet=True)
    
    results = []
    
    # Run through all prompts
    for prompt in prompts:
        prompt_results = []
        
        # Run multiple iterations
        for _ in range(iterations):
            # Tokenize prompt
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            attention_mask = torch.ones_like(input_ids)
            
            # Time the generation
            start_time = time.time()
            
            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=input_ids.size(1) + tokens,
                    do_sample=True,
                    temperature=0.7
                )
            
            end_time = time.time()
            
            # Calculate metrics
            generation_time = end_time - start_time
            tokens_generated = output.size(1) - input_ids.size(1)
            tokens_per_second = tokens_generated / generation_time
            
            # Calculate perplexity
            try:
                with torch.no_grad():
                    outputs = model(output)
                    # Handle different output formats
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = output[..., 1:].contiguous()
                    
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    perplexity = torch.exp(loss).item()
            except Exception as e:
                print(f"Error calculating perplexity: {e}")
                perplexity = float('nan')
            
            # Get generated text
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            prompt_results.append({
                "time": generation_time,
                "tokens": tokens_generated,
                "tokens_per_second": tokens_per_second,
                "perplexity": perplexity,
                "text": generated_text
            })
        
        # Average the results for this prompt
        avg_time = sum(r["time"] for r in prompt_results) / len(prompt_results)
        avg_tokens_per_second = sum(r["tokens_per_second"] for r in prompt_results) / len(prompt_results)
        perplexities = [r["perplexity"] for r in prompt_results if not np.isnan(r["perplexity"])]
        avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else float('nan')
        
        results.append({
            "prompt": prompt,
            "avg_time": avg_time,
            "avg_tokens_per_second": avg_tokens_per_second,
            "avg_perplexity": avg_perplexity,
            "iterations": prompt_results
        })
        
        print(f"Prompt: {prompt[:30]}...")
        print(f"  Time: {avg_time:.2f}s")
        print(f"  Speed: {avg_tokens_per_second:.2f} tokens/sec")
        print(f"  Perplexity: {avg_perplexity:.2f}")
    
    # Calculate overall averages
    overall_time = sum(r["avg_time"] for r in results) / len(results)
    overall_speed = sum(r["avg_tokens_per_second"] for r in results) / len(results)
    perplexities = [r["avg_perplexity"] for r in results if not np.isnan(r["avg_perplexity"])]
    overall_perplexity = sum(perplexities) / len(perplexities) if perplexities else float('nan')
    
    print("\nOptimized Overall Performance:")
    print(f"  Avg. Time: {overall_time:.2f}s")
    print(f"  Avg. Speed: {overall_speed:.2f} tokens/sec")
    print(f"  Avg. Perplexity: {overall_perplexity:.2f}")
    
    # Calculate parameter counts and memory usage
    param_count = sum(p.numel() for p in model.parameters())
    if pruning_percentage > 0:
        # Estimate effective parameter count after pruning
        effective_params = param_count * (1 - pruning_percentage * 0.5)  # Rough estimate as pruning doesn't actually remove params
    else:
        effective_params = param_count
    
    # Cleanup
    del model
    del baseline_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return {
        "model": f"optimized_l{opt_level}_p{int(pruning_percentage*100)}",
        "overall": {
            "time": overall_time,
            "speed": overall_speed,
            "perplexity": overall_perplexity,
            "parameters": param_count,
            "effective_parameters": effective_params
        },
        "prompt_results": results
    }


def run_end_to_end_comparison(args):
    """Run a comprehensive comparison between baseline and optimized models."""
    results = {}
    
    # Create the output directory
    output_dir = ensure_output_dir(args.output_dir)
    
    # Load test prompts
    prompts = load_test_prompts()
    
    # Measure baseline performance
    baseline_results = measure_baseline_performance(
        args.model_name, 
        args.device, 
        args.tokens, 
        args.iterations, 
        prompts
    )
    results["baseline"] = baseline_results
    
    # Measure optimized performance
    optimized_results = measure_optimized_performance(
        args.model_name, 
        args.device, 
        args.tokens, 
        args.iterations, 
        prompts, 
        args.pruning, 
        args.optimization_level
    )
    results["optimized"] = optimized_results
    
    # Calculate improvement metrics
    speedup = optimized_results["overall"]["speed"] / baseline_results["overall"]["speed"]
    perplexity_ratio = optimized_results["overall"]["perplexity"] / baseline_results["overall"]["perplexity"]
    
    print("\n=== Improvement Summary ===")
    print(f"Speed: {speedup:.2f}x faster")
    print(f"Quality (Perplexity): {perplexity_ratio:.2f}x baseline (lower is better)")
    
    # Create visualizations
    create_visualizations(baseline_results, optimized_results, output_dir)
    
    return results


def create_visualizations(baseline_results, optimized_results, output_dir):
    """Create visualizations of the performance comparison."""
    # Extract data for plotting
    baseline_speed = baseline_results["overall"]["speed"]
    optimized_speed = optimized_results["overall"]["speed"]
    
    baseline_perplexity = baseline_results["overall"]["perplexity"]
    optimized_perplexity = optimized_results["overall"]["perplexity"]
    
    # Extract per-prompt data
    prompts = [r["prompt"][:20] + "..." for r in baseline_results["prompt_results"]]
    baseline_prompt_speeds = [r["avg_tokens_per_second"] for r in baseline_results["prompt_results"]]
    optimized_prompt_speeds = [r["avg_tokens_per_second"] for r in optimized_results["prompt_results"]]
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # 1. Overall Speed Comparison
    plt.subplot(2, 2, 1)
    bars = plt.bar(["GPT-2 Baseline", "Sentinel AI"], 
                   [baseline_speed, optimized_speed],
                   color=['lightblue', 'green'])
    plt.title('Generation Speed Comparison')
    plt.ylabel('Tokens per Second (higher is better)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f"{height:.2f}", ha='center', va='bottom')
    
    # 2. Per-prompt Speed Comparison
    plt.subplot(2, 2, 2)
    x = range(len(prompts))
    plt.bar([i - 0.2 for i in x], baseline_prompt_speeds, width=0.4, color='lightblue', label='GPT-2 Baseline')
    plt.bar([i + 0.2 for i in x], optimized_prompt_speeds, width=0.4, color='green', label='Sentinel AI')
    plt.title('Speed by Prompt')
    plt.ylabel('Tokens per Second')
    plt.xticks(x, prompts, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. Quality Comparison
    plt.subplot(2, 2, 3)
    if not np.isnan(baseline_perplexity) and not np.isnan(optimized_perplexity):
        bars = plt.bar(["GPT-2 Baseline", "Sentinel AI"], 
                      [baseline_perplexity, optimized_perplexity],
                      color=['lightblue', 'green'])
        plt.title('Output Quality (Perplexity)')
        plt.ylabel('Perplexity (lower is better)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f"{height:.2f}", ha='center', va='bottom')
    else:
        plt.text(0.5, 0.5, "Perplexity calculation failed", ha='center', va='center')
    
    # 4. Speedup & Efficiency
    plt.subplot(2, 2, 4)
    speedup = optimized_speed / baseline_speed
    quality_ratio = optimized_perplexity / baseline_perplexity if not np.isnan(optimized_perplexity) and not np.isnan(baseline_perplexity) else 1.0
    
    bars = plt.bar(["Speed Improvement", "Quality Ratio"],
                  [speedup, quality_ratio],
                  color=['coral', 'purple'])
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    plt.title('Performance Ratios (Sentinel AI / GPT-2)')
    plt.ylabel('Ratio (>1 is better for speed, <1 is better for quality)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f"{height:.2f}x", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'))
    print(f"Visualization saved to {os.path.join(output_dir, 'performance_comparison.png')}")


def main():
    """Main function."""
    args = parse_args()
    
    # Print system info
    device_name = "GPU" if args.device == "cuda" else "CPU"
    print(f"Running tests on {device_name}")
    print(f"Torch version: {torch.__version__}")
    
    if args.device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    
    # Run the end-to-end comparison
    run_end_to_end_comparison(args)


if __name__ == "__main__":
    main()