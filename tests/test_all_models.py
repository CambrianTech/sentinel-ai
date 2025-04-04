#!/usr/bin/env python
"""
Test script to verify all model loaders work properly with the adaptive transformer.

This simplified script tests text generation with:
1. Original baseline models
2. Adaptive models without pruning
3. Adaptive models with pruning
"""

import os
import sys
import argparse
import torch
import time
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Python path:", sys.path)
print("Current directory:", os.getcwd())

try:
    # Try the new sentinel namespace
    from sentinel.models.loaders.loader import load_baseline_model, load_adaptive_model
    print("Successfully imported from sentinel namespace")
except ImportError:
    # Fallback to old import paths
    print("Falling back to old import paths")
    from models.loaders.loader import load_baseline_model, load_adaptive_model

def measure_inference_speed(model, tokenizer, prompt, num_tokens=20):
    """Measure inference speed in tokens per second."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Warmup run
    _ = model.generate(**inputs, max_length=len(inputs.input_ids[0]) + 5, do_sample=False)
    
    # Actual timed run
    start_time = time.time()
    outputs = model.generate(
        **inputs, 
        max_length=len(inputs.input_ids[0]) + num_tokens,
        do_sample=True,
        temperature=0.7
    )
    end_time = time.time()
    
    # Calculate tokens per second
    total_time = end_time - start_time
    tokens_per_second = num_tokens / total_time
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text, tokens_per_second

def apply_pruning(model, pruning_level=0.5):
    """Apply random pruning to a percentage of attention heads."""
    if not hasattr(model, "blocks"):
        print("Model doesn't support pruning")
        return model
    
    num_layers = len(model.blocks)
    num_heads = model.blocks[0]["attn"].num_heads
    total_heads = num_layers * num_heads
    heads_to_prune = int(total_heads * pruning_level)
    
    print(f"\nRandomly pruning {heads_to_prune} of {total_heads} heads ({pruning_level*100:.1f}%)")
    
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

def test_model(model_name, prompt, device):
    """Test a model with baseline, adaptive, and pruned configurations."""
    print(f"\n{'-'*80}")
    print(f"Testing model: {model_name}")
    print(f"Prompt: {prompt}")
    print(f"{'-'*80}")
    
    # Step 1: Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Step 2: Load and test baseline model
    print(f"Loading baseline model: {model_name}")
    baseline_model = load_baseline_model(model_name, device)
    
    print("Generating with baseline model...")
    baseline_text, baseline_speed = measure_inference_speed(baseline_model, tokenizer, prompt)
    
    # Step 3: Load and test adaptive model
    print("\nConverting to adaptive model...")
    adaptive_model = load_adaptive_model(model_name, baseline_model, device)
    
    print("Generating with adaptive model...")
    adaptive_text, adaptive_speed = measure_inference_speed(adaptive_model, tokenizer, prompt)
    
    # Step 4: Test pruning
    print("\nApplying pruning to adaptive model...")
    pruned_model = apply_pruning(adaptive_model, pruning_level=0.5)
    
    print("Generating with pruned model...")
    pruned_text, pruned_speed = measure_inference_speed(pruned_model, tokenizer, prompt)
    
    # Report results
    print(f"\n{'-'*80}")
    print(f"Results for {model_name}")
    print(f"{'-'*80}")
    
    print(f"\nBaseline model ({baseline_speed:.2f} tokens/sec):")
    print(f"  {baseline_text}")
    
    print(f"\nAdaptive model ({adaptive_speed:.2f} tokens/sec):")
    print(f"  {adaptive_text}")
    
    print(f"\nPruned model ({pruned_speed:.2f} tokens/sec):")
    print(f"  {pruned_text}")
    
    # Assess quality
    baseline_quality = "✅ Good" if len(baseline_text) > len(prompt) + 10 else "❌ Poor"
    adaptive_quality = "✅ Good" if len(adaptive_text) > len(prompt) + 10 and not adaptive_text.endswith(prompt) else "❌ Poor"
    pruned_quality = "✅ Good" if len(pruned_text) > len(prompt) + 10 and not pruned_text.endswith(prompt) else "❌ Poor"
    
    print(f"\nQuality Assessment:")
    print(f"  Baseline: {baseline_quality}")
    print(f"  Adaptive: {adaptive_quality}")
    print(f"  Pruned:   {pruned_quality}")
    
    # Performance comparison
    print(f"\nPerformance Comparison:")
    print(f"  Adaptive vs Baseline: {(adaptive_speed/baseline_speed - 1)*100:.1f}% change")
    print(f"  Pruned vs Adaptive:   {(pruned_speed/adaptive_speed - 1)*100:.1f}% change")
    print(f"  Pruned vs Baseline:   {(pruned_speed/baseline_speed - 1)*100:.1f}% change")
    
    return {
        "model_name": model_name,
        "baseline_quality": baseline_quality,
        "adaptive_quality": adaptive_quality,
        "pruned_quality": pruned_quality,
        "baseline_speed": baseline_speed,
        "adaptive_speed": adaptive_speed,
        "pruned_speed": pruned_speed
    }

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test configuration
    test_configs = [
        {
            "model_name": "distilgpt2",
            "prompt": "The future of artificial intelligence is"
        },
        {
            "model_name": "gpt2",
            "prompt": "Once upon a time there was a"
        },
        {
            "model_name": "facebook/opt-125m",
            "prompt": "The meaning of life is"
        },
        {
            "model_name": "EleutherAI/pythia-70m",
            "prompt": "In a world where robots have become"
        }
    ]
    
    # Run tests
    results = []
    for config in test_configs:
        result = test_model(config["model_name"], config["prompt"], device)
        results.append(result)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)
    
    for result in results:
        print(f"\nModel: {result['model_name']}")
        print(f"  Baseline Quality: {result['baseline_quality']}")
        print(f"  Adaptive Quality: {result['adaptive_quality']}")
        print(f"  Pruned Quality:   {result['pruned_quality']}")
        print(f"  Speed Improvement: {(result['pruned_speed']/result['baseline_speed'] - 1)*100:.1f}%")

if __name__ == "__main__":
    main()