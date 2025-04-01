#!/usr/bin/env python
"""
Simple pruning benchmark script that compares different pruning strategies.

This script extends the simple_pruning_test.py to compare different pruning strategies:
1. Random pruning (baseline)
2. Entropy-based pruning (prunes heads with high attention entropy)
3. Gradient-based pruning (prunes heads with low gradient norm)

For each strategy, it measures:
- Inference speed
- Text generation quality

This provides evidence that strategic pruning is better than random pruning.
"""

import os
import sys
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from tqdm import tqdm

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.loaders.loader import load_baseline_model, load_adaptive_model
from utils.generation_wrapper import generate_text
from controller.metrics.head_metrics import collect_head_metrics

def measure_inference_speed(model, tokenizer, prompt, num_tokens=50, num_runs=3):
    """Measure inference speed in tokens per second."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Warmup run
    _ = model.generate(**inputs, max_length=len(inputs.input_ids[0]) + 10, do_sample=False)
    
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = model.generate(
            **inputs, 
            max_length=len(inputs.input_ids[0]) + num_tokens,
            do_sample=True,
            temperature=0.7
        )
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    tokens_per_second = num_tokens / avg_time
    
    return tokens_per_second

def collect_metrics_for_pruning(model, tokenizer, test_inputs):
    """Collect metrics for determining which heads to prune."""
    model.eval()
    
    # Create batch for metrics collection
    inputs = tokenizer(test_inputs, return_tensors="pt", padding=True).to(model.device)
    
    # Collect head metrics
    metrics = collect_head_metrics(model, batch=inputs)
    
    return metrics

def apply_random_pruning(model, pruning_level):
    """Apply random pruning to a percentage of attention heads."""
    if not hasattr(model, "blocks"):
        print("Model doesn't support pruning")
        return model
    
    num_layers = len(model.blocks)
    num_heads = model.blocks[0]["attn"].num_heads
    total_heads = num_layers * num_heads
    heads_to_prune = int(total_heads * pruning_level)
    
    print(f"Randomly pruning {heads_to_prune} of {total_heads} heads ({pruning_level*100:.1f}%)")
    
    # Get a flattened list of (layer, head) tuples
    all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    
    # Randomly select heads to prune
    pruned_head_indices = np.random.choice(len(all_heads), heads_to_prune, replace=False)
    
    # Set gates to near-zero for pruned heads
    with torch.no_grad():
        for idx in pruned_head_indices:
            layer_idx, head_idx = all_heads[idx]
            model.blocks[layer_idx]["attn"].gate[head_idx] = torch.tensor(0.001, device=model.device)
    
    return model

def apply_entropy_pruning(model, pruning_level, metrics):
    """Apply entropy-based pruning (prune heads with highest entropy)."""
    if not hasattr(model, "blocks"):
        print("Model doesn't support pruning")
        return model
    
    num_layers = len(model.blocks)
    num_heads = model.blocks[0]["attn"].num_heads
    total_heads = num_layers * num_heads
    heads_to_prune = int(total_heads * pruning_level)
    
    print(f"Entropy-based pruning of {heads_to_prune} of {total_heads} heads ({pruning_level*100:.1f}%)")
    
    # Get entropy from metrics
    if "entropy" not in metrics:
        print("Entropy metrics not available, defaulting to random pruning")
        return apply_random_pruning(model, pruning_level)
    
    entropy = metrics["entropy"]
    if not isinstance(entropy, torch.Tensor):
        entropy = torch.tensor(entropy)
    
    # Higher entropy = less focused attention = more likely to be pruned
    # Reshape into [layers, heads] if needed
    if len(entropy.shape) < 2:
        entropy = entropy.reshape(num_layers, num_heads)
        
    # Flatten entropy values
    flat_entropy = entropy.view(-1)
    
    # Get indices of the highest entropy heads
    _, indices = torch.sort(flat_entropy, descending=True)
    indices_to_prune = indices[:heads_to_prune]
    
    # Map flat indices back to (layer, head) tuples
    with torch.no_grad():
        for idx in indices_to_prune:
            layer_idx = idx.item() // num_heads
            head_idx = idx.item() % num_heads
            model.blocks[layer_idx]["attn"].gate[head_idx] = torch.tensor(0.001, device=model.device)
    
    return model

def apply_gradient_pruning(model, pruning_level, metrics):
    """Apply gradient-based pruning (prune heads with lowest gradient norm)."""
    if not hasattr(model, "blocks"):
        print("Model doesn't support pruning")
        return model
    
    num_layers = len(model.blocks)
    num_heads = model.blocks[0]["attn"].num_heads
    total_heads = num_layers * num_heads
    heads_to_prune = int(total_heads * pruning_level)
    
    print(f"Gradient-based pruning of {heads_to_prune} of {total_heads} heads ({pruning_level*100:.1f}%)")
    
    # Get gradient norms from metrics
    if "grad_norm" not in metrics:
        print("Gradient norm metrics not available, defaulting to random pruning")
        return apply_random_pruning(model, pruning_level)
    
    grad_norm = metrics["grad_norm"]
    if not isinstance(grad_norm, torch.Tensor):
        grad_norm = torch.tensor(grad_norm)
    
    # Lower gradient norm = less important head = more likely to be pruned
    # Reshape into [layers, heads] if needed
    if len(grad_norm.shape) < 2:
        grad_norm = grad_norm.reshape(num_layers, num_heads)
        
    # Flatten gradient norms
    flat_grad_norm = grad_norm.view(-1)
    
    # Get indices of the lowest gradient norm heads
    _, indices = torch.sort(flat_grad_norm)
    indices_to_prune = indices[:heads_to_prune]
    
    # Map flat indices back to (layer, head) tuples
    with torch.no_grad():
        for idx in indices_to_prune:
            layer_idx = idx.item() // num_heads
            head_idx = idx.item() % num_heads
            model.blocks[layer_idx]["attn"].gate[head_idx] = torch.tensor(0.001, device=model.device)
    
    return model

def test_generation_quality(model, tokenizer, prompt):
    """Test generation quality with a given prompt."""
    print(f"\nPrompt: {prompt}")
    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=100,
        temperature=0.7,
        device=model.device
    )
    print(f"Generated: {output}\n")
    return output

def compute_text_quality_metrics(text):
    """Compute basic quality metrics for generated text."""
    words = text.split()
    if len(words) <= 1:
        return {"repetition_score": 0, "lexical_diversity": 0}
    
    # Repetition score (lower is better)
    window_size = min(50, len(words))
    repeats = 0
    for i in range(len(words) - 1):
        end_idx = min(i + window_size, len(words))
        if words[i] in words[i+1:end_idx]:
            repeats += 1
    
    repetition_score = repeats / (len(words) - 1)
    
    # Lexical diversity (higher is better)
    unique_words = len(set(words))
    total_words = len(words)
    diversity = unique_words / total_words
    
    return {
        "repetition_score": repetition_score,
        "lexical_diversity": diversity
    }

def plot_results(results, output_file="pruning_strategy_comparison.png"):
    """Plot comparative results of different pruning strategies."""
    pruning_levels = sorted(list(set([level for strategy, level, _, _ in results])))
    strategies = sorted(list(set([strategy for strategy, _, _, _ in results])))
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Inference Speed
    for strategy in strategies:
        x_values = []
        y_values = []
        
        for s, level, speed, _ in results:
            if s == strategy:
                x_values.append(level)
                y_values.append(speed)
        
        # Sort by pruning level for proper line plots
        points = sorted(zip(x_values, y_values))
        x_values = [p[0] for p in points]
        y_values = [p[1] for p in points]
        
        axs[0].plot(x_values, y_values, 'o-', linewidth=2, label=strategy.capitalize())
    
    axs[0].set_xlabel("Pruning Level")
    axs[0].set_ylabel("Tokens per Second")
    axs[0].set_title("Inference Speed vs Pruning Level")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    
    # Plot 2: Lexical Diversity
    for strategy in strategies:
        x_values = []
        y_values = []
        
        for s, level, _, metrics in results:
            if s == strategy and metrics:
                x_values.append(level)
                y_values.append(metrics.get("lexical_diversity", 0))
        
        # Sort by pruning level for proper line plots
        points = sorted(zip(x_values, y_values))
        x_values = [p[0] for p in points]
        y_values = [p[1] for p in points]
        
        axs[1].plot(x_values, y_values, 'o-', linewidth=2, label=strategy.capitalize())
    
    axs[1].set_xlabel("Pruning Level")
    axs[1].set_ylabel("Lexical Diversity")
    axs[1].set_title("Text Quality vs Pruning Level")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Results visualization saved to {output_file}")

def main():
    # Parameters - use reduced set for faster testing
    model_name = "distilgpt2"  # Smaller model for faster testing
    pruning_levels = [0.0, 0.3, 0.7]  # Reduced set of levels
    pruning_strategies = ["random", "entropy"]  # Just test random vs entropy for quick proof
    test_prompts = [
        "Once upon a time in a land far away,"
    ]
    
    # For metrics collection - simplified
    collection_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can process large amounts of data."
    ]
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token for batch processing
    
    # Load model once
    print(f"Loading model: {model_name}")
    baseline_model = load_baseline_model(model_name, device)
    
    # Create output directory for results
    os.makedirs("pruning_results", exist_ok=True)
    
    # Collect metrics for pruning decisions
    print("\nCollecting head metrics for pruning decisions...")
    adaptive_model = load_adaptive_model(model_name, baseline_model, device)
    metrics = collect_metrics_for_pruning(adaptive_model, tokenizer, collection_prompts)
    del adaptive_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Store results
    all_results = []
    
    # Test each pruning strategy
    for strategy in pruning_strategies:
        print(f"\n=== Testing {strategy} pruning strategy ===")
        
        for level in pruning_levels:
            print(f"\n--- Pruning level: {level*100:.1f}% ---")
            
            # Create fresh model copy
            model = load_adaptive_model(model_name, baseline_model, device)
            
            # Apply pruning based on strategy
            if strategy == "random":
                pruned_model = apply_random_pruning(model, level)
            elif strategy == "entropy":
                pruned_model = apply_entropy_pruning(model, level, metrics)
            elif strategy == "gradient":
                pruned_model = apply_gradient_pruning(model, level, metrics)
            else:
                raise ValueError(f"Unknown pruning strategy: {strategy}")
            
            # Measure inference speed
            prompt_idx = 0
            speed = measure_inference_speed(pruned_model, tokenizer, test_prompts[prompt_idx])
            print(f"Inference speed: {speed:.2f} tokens/sec")
            
            # Test generation quality
            generated_text = test_generation_quality(pruned_model, tokenizer, test_prompts[prompt_idx])
            quality_metrics = compute_text_quality_metrics(generated_text)
            print(f"Repetition score: {quality_metrics['repetition_score']:.3f}")
            print(f"Lexical diversity: {quality_metrics['lexical_diversity']:.3f}")
            
            # Store results
            all_results.append((strategy, level, speed, quality_metrics))
            
            # Clean up
            del model
            del pruned_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Plot results
    plot_results(all_results, output_file="pruning_results/pruning_strategy_comparison.png")
    
    print("\nBenchmark complete!")

if __name__ == "__main__":
    main()