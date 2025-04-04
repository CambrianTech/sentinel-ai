#!/usr/bin/env python
"""
Minimal test script for model pruning and generation.
This script tests the core pruning functionality without relying on 
circular imports or complex experiment frameworks.
"""

import os
import sys
import argparse
import time
import torch
from transformers import AutoTokenizer
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use direct sentinel imports to avoid circular issues
from sentinel.models.loaders.loader import load_baseline_model, load_adaptive_model


def prune_random_heads(model, pruning_level=0.3):
    """Apply random pruning to model attention heads."""
    # Check if model is adaptive
    if not hasattr(model, "blocks"):
        print("Model is not an adaptive transformer with attention gates.")
        return model
    
    # Count total attention heads
    num_layers = len(model.blocks)
    num_heads = model.blocks[0]["attn"].num_heads
    total_heads = num_layers * num_heads
    
    # Calculate number of heads to prune
    num_to_prune = int(total_heads * pruning_level)
    
    # Track which heads we've pruned
    pruned_heads = []
    
    # Generate candidates for pruning (all possible heads)
    candidates = []
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            candidates.append((layer_idx, head_idx))
    
    # Randomly select heads to prune
    np.random.shuffle(candidates)
    
    # Apply pruning
    with torch.no_grad():
        for i in range(num_to_prune):
            if i < len(candidates):
                layer_idx, head_idx = candidates[i]
                model.blocks[layer_idx]["attn"].gate[head_idx] = 0.0
                pruned_heads.append((layer_idx, head_idx))
    
    print(f"Pruned {len(pruned_heads)} of {total_heads} attention heads ({pruning_level*100}%)")
    return model


def benchmark_generation(model, tokenizer, prompt, max_tokens=50, num_runs=3):
    """Benchmark generation speed."""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Warm-up run
    with torch.no_grad():
        _ = model.generate(
            **inputs, 
            max_length=len(inputs[0]) + 1,
            do_sample=True
        )
    
    # Timed runs
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_length=len(inputs[0]) + max_tokens,
                do_sample=True,
                temperature=0.8,
                top_k=40,
                top_p=0.9
            )
    end_time = time.time()
    
    # Calculate tokens per second
    elapsed = end_time - start_time
    tokens_generated = (output.shape[1] - inputs["input_ids"].shape[1]) * num_runs
    tokens_per_sec = tokens_generated / elapsed
    
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return output_text, tokens_per_sec


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test pruning with different models")
    parser.add_argument("--model", type=str, default="distilgpt2", 
                        help="Model name (default: distilgpt2)")
    parser.add_argument("--pruning_levels", type=str, default="0.0,0.1,0.3,0.5,0.7",
                       help="Comma-separated pruning levels to test")
    parser.add_argument("--prompt", type=str, default="Once upon a time in a land far away,",
                       help="Prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=30,
                       help="Maximum tokens to generate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    return parser.parse_args()


def main():
    """Run the experiment."""
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    baseline_model = load_baseline_model(args.model, device)
    
    # Get pruning levels to test
    pruning_levels = [float(level) for level in args.pruning_levels.split(",")]
    
    # Test the full model (no pruning) first
    print("\n=== Testing full model (0% pruning) ===\n")
    with torch.no_grad():
        output = baseline_model.generate(
            **tokenizer(args.prompt, return_tensors="pt").to(device),
            max_length=len(tokenizer.encode(args.prompt)) + args.max_tokens,
            do_sample=True,
            temperature=0.8,
            top_k=40,
            top_p=0.9
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Prompt: {args.prompt}")
    print(f"Generated: {generated_text}")
    
    # Test each pruning level
    for pruning_level in pruning_levels:
        print(f"\n=== Testing pruning level: {pruning_level * 100}% ===")
        
        # Create adaptive model
        adaptive_model = load_adaptive_model(args.model, baseline_model, device)
        
        # Apply pruning
        adaptive_model = prune_random_heads(adaptive_model, pruning_level)
        
        # Test generation
        generated, speed = benchmark_generation(
            adaptive_model, tokenizer, args.prompt, args.max_tokens
        )
        
        print(f"Inference speed: {speed:.2f} tokens/sec")
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {generated}")
    
    print("\n=== Experiment complete ===")


if __name__ == "__main__":
    main()