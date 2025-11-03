"""
Magnitude-based pruning strategy.

This module provides functions for pruning attention heads based on
the magnitude of their weights, with smaller weights being pruned first.
"""

import torch
import torch.nn as nn
import logging
from typing import List, Tuple, Dict, Optional, Union

logger = logging.getLogger(__name__)

def collect_weight_magnitudes(model: nn.Module) -> torch.Tensor:
    """
    Collect weight magnitudes for each attention head.
    
    Args:
        model: The transformer model
        
    Returns:
        Tensor with magnitude scores for each head
    """
    magnitudes = {}
    num_layers = 0
    num_heads = 0
    
    # Try to get layers and heads info from config
    if hasattr(model, "config"):
        config = model.config
        if hasattr(config, "n_layer") and hasattr(config, "n_head"):
            # GPT-2 style
            num_layers = config.n_layer
            num_heads = config.n_head
        elif hasattr(config, "num_hidden_layers") and hasattr(config, "num_attention_heads"):
            # BERT style
            num_layers = config.num_hidden_layers
            num_heads = config.num_attention_heads
    
    # Calculate magnitude for each attention head
    for name, param in model.named_parameters():
        # Check if this is an attention weight
        is_q_weight = "query" in name.lower() and "weight" in name.lower()
        is_k_weight = "key" in name.lower() and "weight" in name.lower()
        is_v_weight = "value" in name.lower() and "weight" in name.lower()
        
        # For GPT-2 style attention (combined QKV)
        is_attn_weight = "attn" in name.lower() and "weight" in name.lower()
        
        if is_q_weight or is_k_weight or is_v_weight or is_attn_weight:
            # Try to extract layer number from parameter name
            layer_parts = name.split(".")
            layer_idx = None
            for part in layer_parts:
                if part.isdigit():
                    layer_idx = int(part)
                    break
            
            if layer_idx is None:
                continue
                
            # Update max layer count
            num_layers = max(num_layers, layer_idx + 1)
            
            # For GPT-2, we need to split the weights into heads
            if is_attn_weight and param.dim() > 1:
                # Get embedding dimension
                embed_dim = param.size(0)
                
                # Try to get number of heads from config
                if num_heads == 0:
                    # This is a guess - assume embedding dim is a multiple of num_heads
                    for candidate in [8, 12, 16, 24, 32]:
                        if embed_dim % candidate == 0:
                            num_heads = candidate
                            break
                    
                    if num_heads == 0:
                        # Default fallback
                        num_heads = 12
                
                # Calculate head dimension
                head_dim = embed_dim // num_heads
                
                # Split weights by head
                for head_idx in range(num_heads):
                    start_idx = head_idx * head_dim
                    end_idx = (head_idx + 1) * head_dim
                    
                    # Get weights for this head
                    head_weights = param[start_idx:end_idx]
                    
                    # Calculate magnitude (Frobenius norm)
                    magnitude = head_weights.norm()
                    
                    # Store magnitude
                    if layer_idx not in magnitudes:
                        magnitudes[layer_idx] = {}
                    magnitudes[layer_idx][head_idx] = magnitude.item()
            
            # For models with separate Q, K, V weights
            elif (is_q_weight or is_k_weight or is_v_weight) and param.dim() > 1:
                # Get output dimension
                out_dim = param.size(0)
                
                # Try to get number of heads from config
                if num_heads == 0:
                    # This is a guess - assume output dim is a multiple of num_heads
                    for candidate in [8, 12, 16, 24, 32]:
                        if out_dim % candidate == 0:
                            num_heads = candidate
                            break
                    
                    if num_heads == 0:
                        # Default fallback
                        num_heads = 12
                
                # Calculate head dimension
                head_dim = out_dim // num_heads
                
                # Split weights by head
                for head_idx in range(num_heads):
                    start_idx = head_idx * head_dim
                    end_idx = (head_idx + 1) * head_dim
                    
                    # Get weights for this head
                    head_weights = param[start_idx:end_idx]
                    
                    # Calculate magnitude (Frobenius norm)
                    magnitude = head_weights.norm()
                    
                    # Store magnitude
                    if layer_idx not in magnitudes:
                        magnitudes[layer_idx] = {}
                    if head_idx not in magnitudes[layer_idx]:
                        magnitudes[layer_idx][head_idx] = 0
                    
                    # Accumulate magnitude across Q, K, V
                    magnitudes[layer_idx][head_idx] += magnitude.item()
    
    # Create a tensor with magnitudes
    if not magnitudes:
        logger.warning("Could not compute head magnitudes, using random values")
        return torch.rand(12, 12)
    
    # Find max layer and head from magnitudes
    max_layer = max(magnitudes.keys()) + 1
    max_head = max(max(heads.keys()) for heads in magnitudes.values()) + 1
    
    # Create magnitude tensor
    magnitude_tensor = torch.zeros(max_layer, max_head)
    for layer_idx, heads in magnitudes.items():
        for head_idx, magnitude in heads.items():
            magnitude_tensor[layer_idx, head_idx] = magnitude
    
    return magnitude_tensor

def magnitude_based_pruning(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader,
    prune_ratio: float = 0.3,
    device: str = "cuda"
) -> List[Tuple[int, int]]:
    """
    Prune attention heads based on weight magnitudes.
    
    Args:
        model: The transformer model
        dataloader: DataLoader (not used but kept for API consistency)
        prune_ratio: Fraction of heads to prune (0-1)
        device: Device (not used but kept for API consistency)
        
    Returns:
        List of (layer_idx, head_idx) tuples for pruned heads
    """
    logger.info("Computing head importance using magnitude strategy")
    
    # Compute head magnitudes
    magnitudes = collect_weight_magnitudes(model)
    
    # Flatten magnitudes
    magnitudes_flat = magnitudes.view(-1)
    num_heads_total = magnitudes_flat.size(0)
    num_heads_to_prune = int(num_heads_total * prune_ratio)
    
    # Get indices of heads to prune (smallest magnitudes)
    _, indices = torch.topk(magnitudes_flat, num_heads_to_prune, largest=False)
    
    # Convert flat indices to (layer, head) tuples
    num_heads_per_layer = magnitudes.size(1)
    pruned_heads = [
        (int(idx // num_heads_per_layer), int(idx % num_heads_per_layer))
        for idx in indices
    ]
    
    logger.info(f"Selected {len(pruned_heads)} heads for pruning using magnitude strategy")
    
    # Apply pruning masks to the model
    _apply_pruning_mask(model, pruned_heads)
    
    return pruned_heads

def _apply_pruning_mask(model: nn.Module, heads_to_prune: List[Tuple[int, int]]) -> None:
    """
    Apply pruning mask to the model by zeroing out specified attention heads.
    
    Args:
        model: The transformer model
        heads_to_prune: List of (layer_idx, head_idx) tuples to prune
    """
    # Group heads by layer for efficiency
    heads_by_layer = {}
    for layer, head in heads_to_prune:
        if layer not in heads_by_layer:
            heads_by_layer[layer] = []
        heads_by_layer[layer].append(head)
    
    # Apply pruning to each layer
    for name, module in model.named_modules():
        # Check if this is an attention layer
        if "attention" in name.lower() and hasattr(module, "pruned_heads"):
            # Extract layer index from the name
            layer_parts = name.split(".")
            layer_idx = None
            for part in layer_parts:
                if part.isdigit():
                    layer_idx = int(part)
                    break
            
            # Skip if we couldn't determine the layer index
            if layer_idx is None:
                continue
                
            # Skip if this layer doesn't have heads to prune
            if layer_idx not in heads_by_layer:
                continue
                
            # Get heads to prune for this layer
            heads = heads_by_layer[layer_idx]
            
            # Apply pruning
            module.pruned_heads = set(heads)
            logger.info(f"Pruned heads {heads} in layer {layer_idx}")
    
    # If the model doesn't support pruned_heads attribute, apply attention mask
    # by creating a mask that zeros out the pruned heads during inference
    if not any(hasattr(module, "pruned_heads") for name, module in model.named_modules()):
        logger.info("Model doesn't have pruned_heads attribute, using head masks")
        
        # Create hook to apply masks during forward pass
        def create_head_mask_hook(layer_idx, head_indices):
            def hook(module, input, output):
                # If output is a tuple with attentions
                if isinstance(output, tuple) and len(output) > 1:
                    output_tensor, attentions = output[0], output[1]
                    
                    # Create mask for pruned heads
                    mask = torch.ones_like(attentions)
                    for head_idx in head_indices:
                        mask[:, head_idx] = 0
                        
                    # Apply mask to attentions
                    masked_attentions = attentions * mask
                    
                    # Return modified output
                    return (output_tensor, masked_attentions)
                return output
            return hook
        
        # Register hooks for attention layers
        hooks = []
        for name, module in model.named_modules():
            if "attention" in name.lower() and "output" not in name.lower():
                layer_parts = name.split(".")
                layer_idx = None
                for part in layer_parts:
                    if part.isdigit():
                        layer_idx = int(part)
                        break
                
                if layer_idx is not None and layer_idx in heads_by_layer:
                    hook = module.register_forward_hook(
                        create_head_mask_hook(layer_idx, heads_by_layer[layer_idx])
                    )
                    hooks.append(hook)
        
        # Save hooks in model for later removal if needed
        if not hasattr(model, "_pruning_hooks"):
            model._pruning_hooks = []
        model._pruning_hooks.extend(hooks)