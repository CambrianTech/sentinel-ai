#!/usr/bin/env python3
"""
Run pruning benchmark locally with BLAS fixes for M1/M2 Macs
"""

import os
import sys
import argparse

# Set environment variables first, before importing any numerical libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Now import libraries
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datetime import datetime
import json
import platform

# Set PyTorch threading
torch.set_num_threads(1)
if hasattr(torch, 'set_num_interop_threads'):
    torch.set_num_interop_threads(1)

def main():
    parser = argparse.ArgumentParser(description="Run pruning benchmark")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="Model name")
    parser.add_argument("--strategy", type=str, default="random", choices=["random", "entropy"], 
                        help="Pruning strategy")
    parser.add_argument("--pruning_level", type=float, default=0.3, 
                        help="Pruning level (0.0 to 1.0)")
    parser.add_argument("--prompt", type=str, default="Artificial intelligence is", 
                        help="Prompt for text generation")
    parser.add_argument("--output_dir", type=str, default="pruning_results", 
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Print system info
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nRunning pruning benchmark with:")
    print(f"  Model: {args.model_name}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Pruning level: {args.pruning_level}")
    print(f"  Prompt: {args.prompt}")
    
    # Load model and tokenizer
    print(f"\nLoading {args.model_name}...")
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    model.eval()
    
    # Get model info
    num_layers = len(model.transformer.h)
    num_heads = model.transformer.h[0].attn.num_heads
    print(f"Model has {num_layers} layers with {num_heads} heads each")
    
    # Simple evaluation function
    def evaluate_perplexity(text):
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"])
        return torch.exp(outputs.loss).item()
    
    # Text generation function
    def generate_text(text, max_length=50):
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Zero-out heads function
    def zero_out_heads(layer_idx, head_idxs):
        attn = model.transformer.h[layer_idx].attn
        head_size = attn.head_dim
        
        with torch.no_grad():
            for head_idx in head_idxs:
                start_idx = head_idx * head_size
                end_idx = (head_idx + 1) * head_size
                attn.c_proj.weight[:, start_idx:end_idx] = 0
                if hasattr(attn.c_proj, 'bias') and attn.c_proj.bias is not None:
                    attn.c_proj.bias[start_idx:end_idx] = 0
        
        print(f"Zero'd out heads {head_idxs} in layer {layer_idx}")
    
    # Calculate head importance
    def get_random_importance(layer_idx):
        num_heads = model.transformer.h[layer_idx].attn.num_heads
        return np.random.rand(num_heads)
    
    def get_entropy_importance(layer_idx):
        attn = model.transformer.h[layer_idx].attn
        num_heads = attn.num_heads
        head_importance = np.zeros(num_heads)
        
        for head_idx in range(num_heads):
            start_idx = head_idx * attn.head_dim
            end_idx = (head_idx + 1) * attn.head_dim
            
            # Get L2 norm of weights as proxy for importance
            weight_norm = torch.norm(attn.c_proj.weight[:, start_idx:end_idx]).item()
            head_importance[head_idx] = weight_norm
        
        if np.sum(head_importance) > 0:
            head_importance = head_importance / np.sum(head_importance)
        
        return head_importance
    
    # Evaluate before pruning
    print("\nEvaluating before pruning...")
    perplexity_before = evaluate_perplexity(args.prompt)
    print(f"Perplexity before pruning: {perplexity_before:.4f}")
    
    generated_before = generate_text(args.prompt)
    print(f"Generated (before pruning): {generated_before}")
    
    # Get head importance for each layer
    print("\nCalculating head importance...")
    all_head_importance = []
    
    for layer_idx in range(num_layers):
        if args.strategy == "random":
            importance = get_random_importance(layer_idx)
        else:  # entropy
            importance = get_entropy_importance(layer_idx)
            
        for head_idx, score in enumerate(importance):
            all_head_importance.append((layer_idx, head_idx, score))
    
    # Sort by importance (ascending)
    all_head_importance.sort(key=lambda x: x[2])
    
    # Calculate number of heads to prune
    total_heads = num_layers * num_heads
    heads_to_prune = int(total_heads * args.pruning_level)
    print(f"Pruning {heads_to_prune} out of {total_heads} heads")
    
    # Get heads to prune
    pruned_heads = all_head_importance[:heads_to_prune]
    
    # Group by layer
    pruned_by_layer = {}
    for layer_idx, head_idx, _ in pruned_heads:
        if layer_idx not in pruned_by_layer:
            pruned_by_layer[layer_idx] = []
        pruned_by_layer[layer_idx].append(head_idx)
    
    # Prune the heads
    print("\nPruning heads...")
    for layer_idx, head_idxs in pruned_by_layer.items():
        zero_out_heads(layer_idx, head_idxs)
    
    # Evaluate after pruning
    print("\nEvaluating after pruning...")
    perplexity_after = evaluate_perplexity(args.prompt)
    print(f"Perplexity after pruning: {perplexity_after:.4f}")
    print(f"Perplexity change: {perplexity_after - perplexity_before:.4f}")
    
    generated_after = generate_text(args.prompt)
    print(f"Generated (after pruning): {generated_after}")
    
    # Save results
    results = {
        "model": args.model_name,
        "strategy": args.strategy,
        "pruning_level": args.pruning_level,
        "pruned_heads": heads_to_prune,
        "total_heads": total_heads,
        "prompt": args.prompt,
        "perplexity_before": perplexity_before,
        "perplexity_after": perplexity_after,
        "perplexity_change": perplexity_after - perplexity_before,
        "generated_before": generated_before,
        "generated_after": generated_after,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{args.strategy}_{args.pruning_level}_{timestamp}.json"
    filepath = os.path.join(args.output_dir, filename)
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {filepath}")
    print("\nBenchmark completed successfully!")

if __name__ == "__main__":
    main()