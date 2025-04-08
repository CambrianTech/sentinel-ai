"""
Random pruning strategy.

This module provides functions for randomly selecting attention heads to prune,
which is useful as a baseline for comparison with more sophisticated strategies.
"""

import torch
import torch.nn as nn
import random
import logging
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

def random_pruning(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader,
    prune_ratio: float = 0.3,
    seed: Optional[int] = None,
    device: str = "cuda"
) -> List[Tuple[int, int]]:
    """
    Randomly prune attention heads.
    
    Args:
        model: The transformer model
        dataloader: DataLoader (not used, kept for API consistency)
        prune_ratio: Fraction of heads to prune (0-1)
        seed: Random seed for reproducibility
        device: Device to run on (not used, kept for API consistency)
        
    Returns:
        List of (layer_idx, head_idx) tuples for pruned heads
    """
    # Set random seed if specified
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    
    # Count layers and heads in the model
    num_layers = 0
    num_heads = 0
    
    # Try to get this from the config first
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
    
    # If config doesn't have what we need, try to count attention modules
    if num_layers == 0 or num_heads == 0:
        attention_layers = {}
        for name, module in model.named_modules():
            if "attention" in name.lower() and "output" not in name.lower():
                # Extract layer number from name
                layer_parts = name.split(".")
                layer_idx = None
                for part in layer_parts:
                    if part.isdigit():
                        layer_idx = int(part)
                        break
                
                if layer_idx is not None:
                    attention_layers[layer_idx] = module
        
        if attention_layers:
            num_layers = max(attention_layers.keys()) + 1
            
            # Try to get num_heads from a module attribute
            for module in attention_layers.values():
                if hasattr(module, "num_heads"):
                    num_heads = module.num_heads
                    break
                elif hasattr(module, "num_attention_heads"):
                    num_heads = module.num_attention_heads
                    break
                # For GPT-2 attention
                elif hasattr(module, "split_size"):
                    num_heads = module.split_size
                    break
    
    # Default values if we couldn't determine from the model
    if num_layers == 0:
        logger.warning("Could not determine number of layers, defaulting to 12")
        num_layers = 12
    
    if num_heads == 0:
        logger.warning("Could not determine number of heads per layer, defaulting to 12")
        num_heads = 12
    
    logger.info(f"Model has {num_layers} layers with {num_heads} heads per layer")
    
    # Calculate number of heads to prune
    total_heads = num_layers * num_heads
    num_to_prune = int(total_heads * prune_ratio)
    
    # Generate all possible (layer, head) pairs
    all_heads = [(layer, head) for layer in range(num_layers) for head in range(num_heads)]
    
    # Randomly select heads to prune
    pruned_heads = random.sample(all_heads, num_to_prune)
    
    logger.info(f"Randomly selected {len(pruned_heads)} heads for pruning")
    
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