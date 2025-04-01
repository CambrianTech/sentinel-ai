#!/usr/bin/env python
"""
Demonstrate inference with different pruning strategies.

This script lets users test text generation with different pruning strategies:
1. No pruning (baseline)
2. Random pruning
3. Entropy-based pruning
4. Gradient-based pruning

This provides an interactive way to observe pruning effects.
"""

import os
import sys
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer
import time

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.loaders.loader import load_baseline_model, load_adaptive_model
from utils.generation_wrapper import generate_text
from controller.metrics.head_metrics import collect_head_metrics

def apply_random_pruning(model, pruning_level):
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

def apply_entropy_pruning(model, pruning_level, metrics):
    """Apply entropy-based pruning (prune heads with highest entropy)."""
    if not hasattr(model, "blocks"):
        print("Model doesn't support pruning")
        return model
    
    num_layers = len(model.blocks)
    num_heads = model.blocks[0]["attn"].num_heads
    total_heads = num_layers * num_heads
    heads_to_prune = int(total_heads * pruning_level)
    
    print(f"\nEntropy-based pruning of {heads_to_prune} of {total_heads} heads ({pruning_level*100:.1f}%)")
    
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
    
    print(f"\nGradient-based pruning of {heads_to_prune} of {total_heads} heads ({pruning_level*100:.1f}%)")
    
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

def parse_args():
    parser = argparse.ArgumentParser(description="Test inference with different pruning strategies")
    parser.add_argument("--model", type=str, default="distilgpt2", help="Model name (e.g., 'distilgpt2', 'gpt2')")
    parser.add_argument("--pruning_level", type=float, default=0.5, help="Percentage of heads to prune (0.0-1.0)")
    parser.add_argument("--strategy", type=str, default="entropy", 
                        choices=["none", "random", "entropy", "gradient"], 
                        help="Pruning strategy to use")
    parser.add_argument("--prompt", type=str, 
                        default="Once upon a time in a land far away,", 
                        help="Prompt for text generation")
    parser.add_argument("--num_generate", type=int, default=100, 
                        help="Number of tokens to generate")
    parser.add_argument("--interactive", action="store_true", 
                        help="Run in interactive mode")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token for batch processing
    
    # Load model
    print(f"Loading model: {args.model}")
    baseline_model = load_baseline_model(args.model, device)
    
    # Collection prompts for metrics collection
    collection_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can process large amounts of data.",
        "The architecture of neural networks is inspired by the human brain.",
        "Scientists are working to understand the nature of consciousness.",
        "Language models generate text by predicting the next word in a sequence."
    ]
    
    # Convert to adaptive model
    print("Converting to adaptive model")
    adaptive_model = load_adaptive_model(args.model, baseline_model, device)
    
    # Collect metrics for metric-driven pruning
    if args.strategy in ["entropy", "gradient"]:
        print("\nCollecting head metrics for pruning decisions...")
        # Create input batch
        inputs = tokenizer(collection_prompts, return_tensors="pt", padding=True).to(device)
        metrics = collect_head_metrics(adaptive_model, batch=inputs)
    else:
        metrics = {}
    
    # Apply pruning
    if args.strategy == "none":
        print("\nNo pruning applied (baseline model)")
        model = adaptive_model
    elif args.strategy == "random":
        model = apply_random_pruning(adaptive_model, args.pruning_level)
    elif args.strategy == "entropy":
        model = apply_entropy_pruning(adaptive_model, args.pruning_level, metrics)
    elif args.strategy == "gradient":
        model = apply_gradient_pruning(adaptive_model, args.pruning_level, metrics)
    
    # Measure inference speed
    speed = measure_inference_speed(model, tokenizer, args.prompt)
    print(f"\nInference speed: {speed:.2f} tokens/sec")
    
    if args.interactive:
        print("\n=== Interactive Text Generation ===")
        print("Enter prompts (or 'q' to quit):")
        
        while True:
            try:
                prompt = input("\nPrompt> ")
                if prompt.lower() in ["q", "quit", "exit"]:
                    break
                    
                if not prompt:
                    continue
                
                print("Generating...")
                output = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_length=args.num_generate,
                    temperature=0.7,
                    device=device
                )
                print(f"\nGenerated: {output}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
                
            except Exception as e:
                print(f"Error: {e}")
    else:
        # Generate text with the given prompt
        print(f"\nPrompt: {args.prompt}")
        print("Generating...")
        output = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.num_generate,
            temperature=0.7,
            device=device
        )
        print(f"\nGenerated: {output}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()