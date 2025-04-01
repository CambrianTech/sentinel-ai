#!/usr/bin/env python
"""
Dynamically expand attention heads in an adaptive transformer model.

This script demonstrates the dynamic architecture adjustment capabilities
of the adaptive transformer, allowing it to expand capacity in specific layers
based on metrics like attention entropy and gradient activity.

Usage:
    python scripts/expand_heads.py --model_path /path/to/checkpoint.pth \
                                 --layers 3 6 9 \
                                 --heads_per_layer 2 \
                                 --output_path /path/to/expanded_model.pth
"""

import os
import argparse
import torch
import torch.nn as nn
import copy
from transformers import AutoTokenizer
from models.loaders.loader import load_baseline_model, load_adaptive_model
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.head_metrics import compute_attention_entropy


def expand_attention_heads(model, layer_indices, heads_per_layer):
    """
    Expand attention heads in specified layers.
    
    Args:
        model: The adaptive transformer model
        layer_indices: List of layer indices to expand
        heads_per_layer: Number of heads to add per layer
        
    Returns:
        Modified model with expanded heads
    """
    for layer_idx in layer_indices:
        if layer_idx >= len(model.blocks):
            print(f"Warning: Layer {layer_idx} does not exist, skipping")
            continue
            
        # Get the block to modify
        block = model.blocks[layer_idx]
        attn_module = block["attn"]
        
        # Current number of heads and dimensions
        current_heads = attn_module.num_heads
        embed_dim = attn_module.embed_dim
        head_dim = attn_module.head_dim
        
        # Target number of heads
        new_heads = current_heads + heads_per_layer
        
        print(f"Expanding layer {layer_idx} from {current_heads} to {new_heads} heads")
        
        # Create new projection modules for each component
        new_W_q = nn.ModuleList([nn.Linear(embed_dim, head_dim, bias=True) for _ in range(new_heads)])
        new_W_k = nn.ModuleList([nn.Linear(embed_dim, head_dim, bias=True) for _ in range(new_heads)])
        new_W_v = nn.ModuleList([nn.Linear(embed_dim, head_dim, bias=True) for _ in range(new_heads)])
        new_W_o = nn.ModuleList([nn.Linear(head_dim, embed_dim, bias=True) for _ in range(new_heads)])
        
        # Copy existing weights
        for i in range(current_heads):
            new_W_q[i].weight.data.copy_(attn_module.W_q[i].weight.data)
            new_W_q[i].bias.data.copy_(attn_module.W_q[i].bias.data)
            
            new_W_k[i].weight.data.copy_(attn_module.W_k[i].weight.data)
            new_W_k[i].bias.data.copy_(attn_module.W_k[i].bias.data)
            
            new_W_v[i].weight.data.copy_(attn_module.W_v[i].weight.data)
            new_W_v[i].bias.data.copy_(attn_module.W_v[i].bias.data)
            
            new_W_o[i].weight.data.copy_(attn_module.W_o[i].weight.data)
            new_W_o[i].bias.data.copy_(attn_module.W_o[i].bias.data)
        
        # Initialize new heads
        # We'll use a combination of:
        # 1. Clone and slightly perturb existing heads (for stability)
        # 2. Initialize new heads with small random values (for diversity)
        for i in range(current_heads, new_heads):
            # Choose a source head to clone (pick heads with highest activity)
            gate_values = attn_module.gate.detach().cpu().numpy()
            source_idx = gate_values.argmax().item()
            
            # Q, K, V projections - clone with random perturbation
            new_W_q[i].weight.data.copy_(attn_module.W_q[source_idx].weight.data)
            new_W_q[i].weight.data += torch.randn_like(new_W_q[i].weight.data) * 0.01
            new_W_q[i].bias.data.copy_(attn_module.W_q[source_idx].bias.data)
            new_W_q[i].bias.data += torch.randn_like(new_W_q[i].bias.data) * 0.01
            
            new_W_k[i].weight.data.copy_(attn_module.W_k[source_idx].weight.data)
            new_W_k[i].weight.data += torch.randn_like(new_W_k[i].weight.data) * 0.01
            new_W_k[i].bias.data.copy_(attn_module.W_k[source_idx].bias.data)
            new_W_k[i].bias.data += torch.randn_like(new_W_k[i].bias.data) * 0.01
            
            new_W_v[i].weight.data.copy_(attn_module.W_v[source_idx].weight.data)
            new_W_v[i].weight.data += torch.randn_like(new_W_v[i].weight.data) * 0.01
            new_W_v[i].bias.data.copy_(attn_module.W_v[source_idx].bias.data)
            new_W_v[i].bias.data += torch.randn_like(new_W_v[i].bias.data) * 0.01
            
            # Output projection - initialize with small values for stability
            new_W_o[i].weight.data.normal_(mean=0.0, std=0.01)
            new_W_o[i].bias.data.zero_()
            
        # Create new gate values tensor
        new_gate = nn.Parameter(torch.zeros(new_heads))
        # Copy existing gates
        new_gate.data[:current_heads].copy_(attn_module.gate.data)
        # Initialize new gates with slightly conservative values
        new_gate.data[current_heads:].fill_(0.5)  # Start at 0.5 (moderate contribution)
        
        # Replace the module components
        attn_module.W_q = new_W_q
        attn_module.W_k = new_W_k
        attn_module.W_v = new_W_v
        attn_module.W_o = new_W_o
        attn_module.gate = new_gate
        attn_module.num_heads = new_heads
        
    return model


def get_expansion_candidates(model, entropy_dict, importance_dict=None, num_layers=3):
    """
    Determine which layers would benefit most from head expansion.
    
    Args:
        model: The adaptive transformer model
        entropy_dict: Dictionary of entropy values per head
        importance_dict: Optional dictionary of importance values per head
        num_layers: Number of layers to recommend for expansion
        
    Returns:
        List of layer indices recommended for expansion
    """
    layer_metrics = {}
    
    # Calculate metrics per layer
    for layer_idx in range(len(model.blocks)):
        # Get gate values for this layer
        gate_values = model.blocks[layer_idx]["attn"].gate.detach().cpu()
        
        # Count active heads
        active_heads = (gate_values > 0.2).sum().item()
        
        # Get average entropy for active heads
        layer_entropy = entropy_dict[layer_idx]
        active_entropy = layer_entropy[gate_values > 0.2].mean().item() if active_heads > 0 else 0
        
        # Calculate capacity utilization (% of heads active)
        capacity_utilization = active_heads / len(gate_values)
        
        # Calculate importance score if available
        importance_score = 0
        if importance_dict is not None:
            importance_score = importance_dict[layer_idx].mean().item()
        
        # Compute expansion score:
        # - High if many heads are active (high utilization)
        # - High if active heads have low entropy (specialized) 
        # - High if layer has high importance to the model
        expansion_score = (
            capacity_utilization * 0.5 +  # Weight for utilization
            (1.0 - active_entropy / 5.0) * 0.3 +  # Weight for entropy (inverted)
            importance_score * 0.2  # Weight for importance
        )
        
        layer_metrics[layer_idx] = {
            'active_heads': active_heads,
            'total_heads': len(gate_values),
            'utilization': capacity_utilization,
            'avg_entropy': active_entropy,
            'importance': importance_score,
            'expansion_score': expansion_score
        }
    
    # Sort layers by expansion score
    sorted_layers = sorted(layer_metrics.items(), 
                         key=lambda x: x[1]['expansion_score'], 
                         reverse=True)
    
    # Return top layers
    return [layer_idx for layer_idx, _ in sorted_layers[:num_layers]]


def main():
    parser = argparse.ArgumentParser(description="Expand attention heads in adaptive transformer")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Base model name")
    parser.add_argument("--layers", type=int, nargs="+", help="Layer indices to expand")
    parser.add_argument("--heads_per_layer", type=int, default=2, help="Heads to add per layer")
    parser.add_argument("--auto_select", action="store_true", help="Automatically select layers to expand")
    parser.add_argument("--num_auto_layers", type=int, default=3, help="Number of layers to auto-select")
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
    
    # Determine which layers to expand
    if args.auto_select:
        print("Computing metrics for automatic layer selection...")
        entropy_dict = compute_attention_entropy(model, device=device)
        
        print("Determining layers to expand...")
        layer_indices = get_expansion_candidates(model, entropy_dict, num_layers=args.num_auto_layers)
        print(f"Selected layers for expansion: {layer_indices}")
    else:
        if not args.layers:
            raise ValueError("Must specify --layers or --auto_select")
        layer_indices = args.layers
    
    # Expand heads
    print(f"Expanding {len(layer_indices)} layers with {args.heads_per_layer} heads each")
    expanded_model = expand_attention_heads(model, layer_indices, args.heads_per_layer)
    
    # Update head learning rate multipliers for the new heads
    new_multipliers = {}
    for layer_idx, block in enumerate(expanded_model.blocks):
        attn_module = block["attn"]
        for head_idx in range(attn_module.num_heads):
            key = (layer_idx, head_idx)
            if key in head_lr_multipliers:
                new_multipliers[key] = head_lr_multipliers[key]
            else:
                # New head gets standard learning rate
                new_multipliers[key] = 1.0
    
    # Save expanded model
    print(f"Saving expanded model to {args.output_path}")
    save_checkpoint(args.output_path, expanded_model, optimizer, new_multipliers, epoch, step)
    
    # Print summary
    print("\nExpansion summary:")
    print(f"  Original model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Expanded model: {sum(p.numel() for p in expanded_model.parameters()):,} parameters")
    
    added_params = sum(p.numel() for p in expanded_model.parameters()) - sum(p.numel() for p in model.parameters())
    print(f"  Added parameters: {added_params:,} ({added_params / sum(p.numel() for p in model.parameters()) * 100:.2f}%)")
    
    print("\nExpanded model head counts:")
    for layer_idx, block in enumerate(expanded_model.blocks):
        attn_module = block["attn"]
        print(f"  Layer {layer_idx}: {attn_module.num_heads} heads")
    

if __name__ == "__main__":
    main()