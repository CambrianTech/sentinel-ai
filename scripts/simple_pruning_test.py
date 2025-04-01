#!/usr/bin/env python
"""
Simple pruning test script that demonstrates pruning effectiveness.

This script performs a basic test of pruning by:
1. Loading a model (e.g., GPT-2)
2. Running inference with the full model
3. Pruning a percentage of attention heads
4. Running inference again with the pruned model
5. Comparing performance metrics

This shows whether pruning degrades performance and by how much.
"""

import os
import sys
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.loaders.loader import load_baseline_model, load_adaptive_model
from utils.generation_wrapper import generate_text

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

def apply_pruning(model, pruning_level):
    """Apply pruning to a percentage of attention heads."""
    if not hasattr(model, "blocks"):
        print("Model doesn't support pruning")
        return model
    
    num_layers = len(model.blocks)
    num_heads = model.blocks[0]["attn"].num_heads
    total_heads = num_layers * num_heads
    heads_to_prune = int(total_heads * pruning_level)
    
    print(f"Pruning {heads_to_prune} of {total_heads} attention heads ({pruning_level*100:.1f}%)")
    
    # Get a flattened list of (layer, head) tuples
    all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    
    # For this test, we'll just prune random heads
    pruned_heads = np.random.choice(len(all_heads), heads_to_prune, replace=False)
    
    # Set gates to near-zero for pruned heads
    with torch.no_grad():
        for idx in pruned_heads:
            layer_idx, head_idx = all_heads[idx]
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

def plot_results(results):
    """Plot the pruning results."""
    pruning_levels = [level for level, _ in results]
    speeds = [speed for _, speed in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(pruning_levels, speeds, 'o-', linewidth=2)
    plt.xlabel("Pruning Level")
    plt.ylabel("Tokens per Second")
    plt.title("Inference Speed vs Pruning Level")
    plt.grid(True, alpha=0.3)
    plt.savefig("pruning_speed_results.png")
    plt.close()

def main():
    # Parameters
    model_name = "distilgpt2"  # Smaller model for faster testing
    pruning_levels = [0.0, 0.1, 0.3, 0.5, 0.7]
    prompts = [
        "Once upon a time in a land far away,",
        "The future of artificial intelligence depends on",
        "Scientists have recently discovered that"
    ]
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    baseline_model = load_baseline_model(model_name, device)
    
    # Convert to adaptive model
    print("Converting to adaptive model")
    model = load_adaptive_model(model_name, baseline_model, device)
    
    # Test generation with full model
    print("\n=== Testing full model (0% pruning) ===")
    test_generation_quality(model, tokenizer, prompts[0])
    
    # Test different pruning levels
    results = []
    for level in pruning_levels:
        print(f"\n=== Testing pruning level: {level*100:.1f}% ===")
        
        # Create a fresh model copy for each pruning level
        model_copy = load_adaptive_model(model_name, baseline_model, device)
        
        # Apply pruning
        pruned_model = apply_pruning(model_copy, level)
        
        # Measure inference speed
        speed = measure_inference_speed(pruned_model, tokenizer, prompts[0])
        print(f"Inference speed: {speed:.2f} tokens/sec")
        
        # Test generation quality
        test_generation_quality(pruned_model, tokenizer, prompts[0])
        
        # Store results
        results.append((level, speed))
        
        # Clean up
        del model_copy
        del pruned_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Plot results
    plot_results(results)
    print("\nTest complete. Results visualized in pruning_speed_results.png")

if __name__ == "__main__":
    main()