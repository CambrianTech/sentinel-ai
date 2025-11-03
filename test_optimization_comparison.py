#!/usr/bin/env python

"""
Optimization Comparison Test Script

This script provides a direct comparison between standard and optimized models
at different pruning levels. It measures performance in terms of speed and
perplexity to validate that optimizations are effective.
"""

import os
import time
import torch
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to the path
project_root = Path(__file__).parent.absolute()
import sys
sys.path.insert(0, str(project_root))

# Import model loaders
from models.loaders.loader import load_baseline_model, load_adaptive_model
from models.loaders.loader_optimized import load_optimized_adaptive_model
from scripts.pruning_comparison.pruning_agency_comparison import apply_pruning


def setup_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare standard vs. optimized models at different pruning levels")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="gpt2", 
                       help="Name of the model to test (default: gpt2)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run on (default: cpu)")
    
    # Test settings
    parser.add_argument("--pruning_levels", type=str, default="0,50,70",
                       help="Comma-separated pruning levels to test (default: 0,50,70)")
    parser.add_argument("--prompt", type=str, 
                       default="Write a short story about artificial intelligence", 
                       help="Prompt for text generation")
    parser.add_argument("--max_tokens", type=int, default=100,
                       help="Number of tokens to generate (default: 100)")
    parser.add_argument("--iterations", type=int, default=3,
                       help="Number of iterations for each test configuration (default: 3)")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualizations of results")
    
    return parser.parse_args()


def generate_text(model, tokenizer, prompt, max_tokens, device):
    """Generate text and measure performance metrics."""
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    
    # Time generation
    start_time = time.time()
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.size(1) + max_tokens,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )
    
    end_time = time.time()
    
    # Calculate metrics
    generation_time = end_time - start_time
    tokens_generated = output.size(1) - input_ids.size(1)
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
    
    # Calculate perplexity (quality metric)
    try:
        with torch.no_grad():
            # Get logits for the generated sequence
            outputs = model(output)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = output[..., 1:].contiguous()
            
            # Calculate loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                            shift_labels.view(-1))
            
            # Perplexity = exp(loss)
            perplexity = torch.exp(loss).item()
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        perplexity = float('nan')
    
    # Decode generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return {
        "time": generation_time,
        "tokens": tokens_generated,
        "tokens_per_second": tokens_per_second,
        "perplexity": perplexity,
        "text": generated_text
    }


def test_model(model_type, pruning_level, args):
    """Test a model with specified configuration."""
    print(f"\n===== Testing {model_type} model with {pruning_level}% pruning =====\n")
    
    # Load baseline model
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None
    
    baseline_model = load_baseline_model(args.model_name, args.device)
    
    # Set optimization level - use only level 1 for CPU tests to reduce overhead
    # Level 3 is intended for GPU and increases parameter count substantially
    optimization_level = 1 if args.device == "cpu" else 3
    if model_type == "standard":
        model = load_adaptive_model(args.model_name, baseline_model, args.device)
    else:  # optimized
        model = load_optimized_adaptive_model(
            args.model_name, 
            baseline_model, 
            args.device,
            optimization_level=optimization_level
        )
        print(f"Using optimization level {optimization_level} for {args.device}")
    
    # Apply pruning if needed
    if pruning_level > 0:
        pruning_ratio = pruning_level / 100.0
        print(f"Applying {pruning_level}% pruning...")
        model, _, _ = apply_pruning(model, pruning_ratio, verbose=False, quiet=True)
    
    # Run iterations
    results = []
    for i in range(args.iterations):
        print(f"Running iteration {i+1}/{args.iterations}")
        result = generate_text(
            model, 
            tokenizer, 
            args.prompt, 
            args.max_tokens,
            args.device
        )
        results.append(result)
        print(f"  Generated {result['tokens']} tokens in {result['time']:.2f}s "
              f"({result['tokens_per_second']:.2f} tokens/sec)")
        print(f"  Perplexity: {result['perplexity']:.2f}")
    
    # Calculate averages
    avg_time = sum(r["time"] for r in results) / len(results)
    avg_tokens_per_second = sum(r["tokens_per_second"] for r in results) / len(results)
    perplexities = [r["perplexity"] for r in results if not np.isnan(r["perplexity"])]
    avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else float('nan')
    
    print(f"\nAverage results for {model_type} model with {pruning_level}% pruning:")
    print(f"  Time: {avg_time:.2f}s")
    print(f"  Tokens per second: {avg_tokens_per_second:.2f}")
    print(f"  Perplexity: {avg_perplexity:.2f}")
    
    # Clean up to free memory
    del model
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "model_type": model_type,
        "pruning_level": pruning_level,
        "avg_time": avg_time,
        "avg_tokens_per_second": avg_tokens_per_second,
        "avg_perplexity": avg_perplexity,
        "iterations": results
    }


def visualize_results(results, args):
    """Create visualizations from the test results."""
    if not args.visualize:
        return
    
    print("\n===== Creating visualizations =====\n")
    
    # Organize results by model type and pruning level
    standard_results = [r for r in results if r["model_type"] == "standard"]
    optimized_results = [r for r in results if r["model_type"] == "optimized"]
    
    pruning_levels = [r["pruning_level"] for r in standard_results]
    
    # Extract metrics
    standard_speed = [r["avg_tokens_per_second"] for r in standard_results]
    optimized_speed = [r["avg_tokens_per_second"] for r in optimized_results]
    
    standard_perplexity = [r["avg_perplexity"] for r in standard_results]
    optimized_perplexity = [r["avg_perplexity"] for r in optimized_results]
    
    # Calculate speedup ratio
    speedup = [opt/std if std > 0 else 0 for opt, std in zip(optimized_speed, standard_speed)]
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot tokens per second
    plt.subplot(2, 2, 1)
    plt.plot(pruning_levels, standard_speed, 'o-', label="Standard", color="dodgerblue", linewidth=2)
    plt.plot(pruning_levels, optimized_speed, 'o-', label="Optimized", color="green", linewidth=2)
    plt.title('Generation Speed vs. Pruning Level')
    plt.xlabel('Pruning Level (%)')
    plt.ylabel('Tokens per Second')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add annotations
    for i, level in enumerate(pruning_levels):
        plt.annotate(f"{standard_speed[i]:.1f}",
                    xy=(level, standard_speed[i]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)
        plt.annotate(f"{optimized_speed[i]:.1f}",
                    xy=(level, optimized_speed[i]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)
    
    # Plot speedup
    plt.subplot(2, 2, 2)
    bars = plt.bar(pruning_levels, speedup, color="coral")
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    plt.title('Optimization Speedup by Pruning Level')
    plt.xlabel('Pruning Level (%)')
    plt.ylabel('Speedup Factor (>1 is better)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f"{height:.2f}x", ha='center', va='bottom')
    
    # Plot perplexity (quality)
    plt.subplot(2, 2, 3)
    plt.plot(pruning_levels, standard_perplexity, 'o-', label="Standard", color="dodgerblue", linewidth=2)
    plt.plot(pruning_levels, optimized_perplexity, 'o-', label="Optimized", color="green", linewidth=2)
    plt.title('Output Quality (Perplexity) vs. Pruning Level')
    plt.xlabel('Pruning Level (%)')
    plt.ylabel('Perplexity (lower is better)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig("optimization_comparison.png")
    print("Visualization saved to 'optimization_comparison.png'")


def main():
    """Main function."""
    args = setup_args()
    
    # Parse pruning levels
    pruning_levels = [int(level) for level in args.pruning_levels.split(",")]
    
    # Run tests
    results = []
    
    # Test standard model at each pruning level
    for level in pruning_levels:
        result = test_model("standard", level, args)
        if result:
            results.append(result)
    
    # Test optimized model at each pruning level
    for level in pruning_levels:
        result = test_model("optimized", level, args)
        if result:
            results.append(result)
    
    # Create visualizations
    visualize_results(results, args)
    
    # Final summary
    print("\n===== Test Summary =====\n")
    for level in pruning_levels:
        std_result = next((r for r in results if r["model_type"] == "standard" and r["pruning_level"] == level), None)
        opt_result = next((r for r in results if r["model_type"] == "optimized" and r["pruning_level"] == level), None)
        
        if std_result and opt_result:
            std_speed = std_result["avg_tokens_per_second"]
            opt_speed = opt_result["avg_tokens_per_second"]
            speedup = opt_speed / std_speed if std_speed > 0 else 0
            
            print(f"Pruning Level {level}%:")
            print(f"  Standard: {std_speed:.2f} tokens/sec")
            print(f"  Optimized: {opt_speed:.2f} tokens/sec")
            print(f"  Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()