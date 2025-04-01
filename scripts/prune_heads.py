#!/usr/bin/env python
"""
Dynamically prune attention heads in an adaptive transformer model.

This script implements the pruning component of the dynamic architecture adjustment
capabilities of the adaptive transformer. It can prune heads based on metrics like
attention entropy, head importance, and gradient norms.

Usage:
    python scripts/prune_heads.py --model_path /path/to/checkpoint.pth \
                               --auto_prune \
                               --output_path /path/to/pruned_model.pth
    
    # Or specify heads manually:
    python scripts/prune_heads.py --model_path /path/to/checkpoint.pth \
                               --layer_heads "0:1,2 2:0,3" \
                               --output_path /path/to/pruned_model.pth
"""

import os
import argparse
import torch
import re
from transformers import AutoTokenizer
from models.loaders.loader import load_baseline_model, load_adaptive_model
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.head_metrics import (
    compute_attention_entropy,
    compute_head_importance,
    compute_gradient_norms,
    recommend_pruning_growth
)
from datasets.dataset_loader import load_and_tokenize_dataset
from utils.training import compute_loss


def prune_attention_heads(model, prune_spec):
    """
    Prune attention heads in specified layers.
    
    Args:
        model: The adaptive transformer model
        prune_spec: Dictionary mapping layer indices to lists of head indices to prune
        
    Returns:
        Modified model with pruned heads
    """
    print("Pruning heads:")
    
    heads_pruned = 0
    for layer_idx, head_indices in prune_spec.items():
        if layer_idx >= len(model.blocks):
            print(f"  Warning: Layer {layer_idx} does not exist, skipping")
            continue
            
        # Get the block to modify
        block = model.blocks[layer_idx]
        attn_module = block["attn"]
        
        for head_idx in head_indices:
            if head_idx >= attn_module.num_heads:
                print(f"  Warning: Head {head_idx} in layer {layer_idx} does not exist, skipping")
                continue
                
            # Set gate value to zero
            attn_module.gate.data[head_idx] = 0.0
            
            # Freeze parameters for this head
            for param in attn_module.W_q[head_idx].parameters():
                param.requires_grad = False
            for param in attn_module.W_k[head_idx].parameters():
                param.requires_grad = False
            for param in attn_module.W_v[head_idx].parameters():
                param.requires_grad = False
            for param in attn_module.W_o[head_idx].parameters():
                param.requires_grad = False
                
            print(f"  Pruned layer {layer_idx}, head {head_idx}")
            heads_pruned += 1
    
    print(f"Total heads pruned: {heads_pruned}")
    return model


def parse_layer_head_spec(spec_str):
    """
    Parse a string specification of layers and heads to prune.
    Format: "layer1:head1,head2 layer2:head1,head2"
    
    Example: "0:1,2 2:0,3" means prune layer 0 heads 1,2 and layer 2 heads 0,3
    
    Args:
        spec_str: String specification of layers and heads
        
    Returns:
        Dictionary mapping layer indices to lists of head indices
    """
    prune_spec = {}
    
    if not spec_str:
        return prune_spec
        
    # Split by whitespace to get layer:head pairs
    layer_head_pairs = spec_str.strip().split()
    
    for pair in layer_head_pairs:
        # Check format
        if ":" not in pair:
            print(f"Warning: Invalid format for layer-head pair '{pair}', skipping")
            continue
            
        # Split by colon to get layer and heads
        layer_str, heads_str = pair.split(":")
        
        try:
            layer_idx = int(layer_str)
        except ValueError:
            print(f"Warning: Invalid layer index '{layer_str}', skipping")
            continue
            
        # Split heads by comma
        head_indices = []
        for head_str in heads_str.split(","):
            try:
                head_idx = int(head_str)
                head_indices.append(head_idx)
            except ValueError:
                print(f"Warning: Invalid head index '{head_str}', skipping")
                
        if head_indices:
            prune_spec[layer_idx] = head_indices
    
    return prune_spec


def auto_select_heads_to_prune(model, dataset_loader, device, 
                             max_heads_per_layer=1, max_total_heads=None,
                             importance_threshold=0.05, entropy_threshold=1.5):
    """
    Automatically select heads to prune based on metrics.
    
    Args:
        model: The adaptive transformer model
        dataset_loader: DataLoader for computing metrics
        device: Device to run computation on
        max_heads_per_layer: Maximum number of heads to prune per layer
        max_total_heads: Maximum total number of heads to prune (None for no limit)
        importance_threshold: Maximum importance for prunable heads
        entropy_threshold: Minimum entropy for prunable heads
        
    Returns:
        Dictionary mapping layer indices to lists of head indices to prune
    """
    print("Computing metrics for automatic head selection...")
    
    # Compute metrics
    entropy_dict = compute_attention_entropy(model, device=device)
    importance_dict = compute_head_importance(model, dataset_loader, compute_loss, device=device)
    
    # Create optimizer for gradient computation
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    grad_norms_dict = compute_gradient_norms(model, dataset_loader, compute_loss, optimizer, device=device)
    
    # Get pruning recommendations
    prune_candidates, _ = recommend_pruning_growth(
        entropy_dict, importance_dict, grad_norms_dict,
        entropy_threshold=entropy_threshold,
        importance_threshold=importance_threshold
    )
    
    # Organize candidates by layer
    candidates_by_layer = {}
    for layer_idx, head_idx in prune_candidates:
        if layer_idx not in candidates_by_layer:
            candidates_by_layer[layer_idx] = []
        candidates_by_layer[layer_idx].append(head_idx)
    
    # Apply limits
    prune_spec = {}
    total_heads = 0
    
    for layer_idx, head_indices in candidates_by_layer.items():
        # Sort heads by importance (lowest first)
        head_indices.sort(key=lambda h: importance_dict[layer_idx][h].item())
        
        # Apply per-layer limit
        layer_heads = head_indices[:max_heads_per_layer]
        prune_spec[layer_idx] = layer_heads
        total_heads += len(layer_heads)
        
        # Check total limit
        if max_total_heads is not None and total_heads >= max_total_heads:
            # Trim the last layer if needed
            excess = total_heads - max_total_heads
            if excess > 0:
                prune_spec[layer_idx] = layer_heads[:-excess]
                total_heads = max_total_heads
            break
    
    print(f"Selected {total_heads} heads for pruning across {len(prune_spec)} layers")
    return prune_spec


def update_learning_rate_multipliers(head_lr_multipliers, prune_spec):
    """
    Update head learning rate multipliers after pruning.
    Remove entries for pruned heads and reset the rest.
    
    Args:
        head_lr_multipliers: Dictionary of learning rate multipliers
        prune_spec: Dictionary mapping layer indices to lists of head indices to prune
        
    Returns:
        Updated head_lr_multipliers dictionary
    """
    # Create a new dictionary
    new_multipliers = {}
    
    # Copy all entries except for pruned heads
    for key, value in head_lr_multipliers.items():
        layer_idx, head_idx = key
        if layer_idx in prune_spec and head_idx in prune_spec[layer_idx]:
            # Pruned head, skip
            continue
        new_multipliers[key] = value
    
    return new_multipliers


def main():
    parser = argparse.ArgumentParser(description="Prune attention heads in adaptive transformer")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Base model name")
    parser.add_argument("--layer_heads", type=str, default=None, 
                      help="Specification of layers and heads to prune, e.g., '0:1,2 2:0,3'")
    parser.add_argument("--auto_prune", action="store_true", 
                      help="Automatically select heads to prune based on metrics")
    parser.add_argument("--max_heads_per_layer", type=int, default=1,
                      help="Maximum number of heads to prune per layer in auto mode")
    parser.add_argument("--max_total_heads", type=int, default=None,
                      help="Maximum total number of heads to prune in auto mode")
    parser.add_argument("--dataset", type=str, default="tiny_shakespeare",
                      help="Dataset name for computing metrics in auto mode")
    parser.add_argument("--importance_threshold", type=float, default=0.05,
                      help="Maximum importance for prunable heads in auto mode")
    parser.add_argument("--entropy_threshold", type=float, default=1.5,
                      help="Minimum entropy for prunable heads in auto mode")
    parser.add_argument("--output_path", type=str, required=True, help="Output checkpoint path")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None,
                      help="Device to use (defaults to CUDA if available)")
    args = parser.parse_args()
    
    # Determine device
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Load models
    baseline_model = load_baseline_model(args.model_name, device)
    model = load_adaptive_model(args.model_name, baseline_model, device)
    
    # Load checkpoint
    if os.path.exists(args.model_path):
        optimizer = torch.optim.AdamW(model.parameters())
        head_lr_multipliers = {}
        model, optimizer, head_lr_multipliers, epoch, step = load_checkpoint(
            model, optimizer, head_lr_multipliers, args.model_path, device)
        print(f"Loaded checkpoint from {args.model_path} (epoch {epoch}, step {step})")
    else:
        print(f"Warning: Checkpoint {args.model_path} not found, using freshly initialized model")
        optimizer = torch.optim.AdamW(model.parameters())
        head_lr_multipliers = {}
        epoch, step = 0, 0
    
    # Determine which heads to prune
    prune_spec = {}
    if args.auto_prune:
        # Load dataset for metrics computation
        train_ids, val_ids = load_and_tokenize_dataset(args.model_name, dataset_name=args.dataset)
        val_dataset = torch.utils.data.TensorDataset(torch.tensor(val_ids))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8)
        
        prune_spec = auto_select_heads_to_prune(
            model, val_loader, device,
            max_heads_per_layer=args.max_heads_per_layer,
            max_total_heads=args.max_total_heads,
            importance_threshold=args.importance_threshold,
            entropy_threshold=args.entropy_threshold
        )
    elif args.layer_heads:
        prune_spec = parse_layer_head_spec(args.layer_heads)
    
    if not prune_spec:
        print("No heads selected for pruning. Exiting.")
        return
    
    # Prune heads
    pruned_model = prune_attention_heads(model, prune_spec)
    
    # Update learning rate multipliers
    new_multipliers = update_learning_rate_multipliers(head_lr_multipliers, prune_spec)
    
    # Save pruned model
    print(f"Saving pruned model to {args.output_path}")
    save_checkpoint(args.output_path, pruned_model, optimizer, new_multipliers, epoch, step)
    
    # Print summary
    print("\nPruning summary:")
    total_params_before = sum(p.numel() for p in model.parameters())
    active_params_after = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)
    
    print(f"  Original model:  {total_params_before:,} parameters")
    print(f"  After pruning: {active_params_after:,} trainable parameters")
    
    saved_params = total_params_before - active_params_after
    print(f"  Parameters frozen: {saved_params:,} ({saved_params / total_params_before * 100:.2f}%)")
    
    print("\nActive head counts by layer:")
    for layer_idx, block in enumerate(pruned_model.blocks):
        attn_module = block["attn"]
        active_heads = sum(float(g) > 0.01 for g in attn_module.gate)
        print(f"  Layer {layer_idx}: {active_heads}/{attn_module.num_heads} active heads")
    

if __name__ == "__main__":
    main()