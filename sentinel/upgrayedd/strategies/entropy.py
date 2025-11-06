"""
Entropy-based pruning strategy.

This module provides functions for computing head importance based on
attention entropy and pruning the least important heads.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Union

logger = logging.getLogger(__name__)

def compute_attention_entropy(attn_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute entropy of attention distributions.
    
    Higher entropy means more uniform attention (less focused),
    which indicates the head might be less important.
    
    Args:
        attn_probs: Attention probabilities tensor [batch, heads, seq_len, seq_len]
        eps: Small constant to avoid log(0)
        
    Returns:
        Tensor containing entropy for each head
    """
    # Add epsilon to avoid log(0)
    log_attn = torch.log(attn_probs + eps)
    
    # Compute entropy: -sum(p * log(p))
    entropy = -torch.sum(attn_probs * log_attn, dim=-1)  # [batch, heads, seq_len]
    
    # Average over batch and sequence dimension
    return entropy.mean(dim=(0, 2))  # [heads]

def collect_attention_distributions(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader,
    num_batches: int = 10,
    device: str = "cuda"
) -> Dict[int, torch.Tensor]:
    """
    Collect attention distributions from the model.
    
    Args:
        model: The transformer model
        dataloader: DataLoader with evaluation data
        num_batches: Number of batches to process
        device: Device to run on
        
    Returns:
        Dictionary of layer index to attention distribution tensor
    """
    device_obj = torch.device(device)
    model.eval()
    attention_distributions = {}
    
    # Register hooks for attention layers
    hooks = []
    
    def get_attention_hook(layer_idx):
        def hook(module, input, output):
            if hasattr(output, "attentions") and output.attentions is not None:
                attn_probs = output.attentions
                if isinstance(attn_probs, tuple):
                    attn_probs = attn_probs[0]
                    
                # Store attention probabilities
                if layer_idx in attention_distributions:
                    attention_distributions[layer_idx] = torch.cat(
                        [attention_distributions[layer_idx], attn_probs.detach()], dim=0
                    )
                else:
                    attention_distributions[layer_idx] = attn_probs.detach()
        return hook
    
    # Add hooks to all attention layers
    for name, module in model.named_modules():
        if "attention" in name.lower() and "output" not in name.lower():
            layer_idx = int(name.split(".")[-2]) if "." in name else 0
            hook = module.register_forward_hook(get_attention_hook(layer_idx))
            hooks.append((layer_idx, hook))
    
    # Run forward passes to collect attention
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
                
            # Prepare input batch
            if isinstance(batch, dict):
                inputs = {k: v.to(device_obj) for k, v in batch.items()}
            else:
                # Assume the batch is a tuple with input_ids first
                input_ids = batch[0].to(device_obj)
                attention_mask = batch[1].to(device_obj) if len(batch) > 1 else None
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }
            
            # Forward pass with attention output
            outputs = model(**inputs, output_attentions=True)
    
    # Remove hooks
    for _, hook in hooks:
        hook.remove()
    
    return attention_distributions

def compute_head_importance(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader,
    num_batches: int = 10,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Compute importance scores for each attention head based on entropy.
    
    Args:
        model: The transformer model
        dataloader: DataLoader with evaluation data
        num_batches: Number of batches to process
        device: Device to run on
        
    Returns:
        Tensor with importance scores for each head
    """
    logger.info("Computing head importance using entropy strategy")
    
    # Collect attention distributions
    attention_distributions = collect_attention_distributions(
        model, dataloader, num_batches, device
    )
    
    # Compute entropy for each head
    head_entropies = {}
    for layer_idx, attn_probs in attention_distributions.items():
        # Compute entropy
        entropy = compute_attention_entropy(attn_probs)
        head_entropies[layer_idx] = entropy
    
    # Number of layers and heads
    num_layers = max(head_entropies.keys()) + 1
    num_heads = max(len(e) for e in head_entropies.values())
    
    # Create importance tensor (higher entropy = lower importance)
    importance = torch.zeros(num_layers, num_heads, device=device)
    for layer_idx, entropy in head_entropies.items():
        importance[layer_idx, :len(entropy)] = entropy
    
    return importance

def entropy_based_pruning(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader,
    prune_ratio: float = 0.3,
    num_batches: int = 10,
    device: str = "cuda"
) -> List[Tuple[int, int]]:
    """
    Prune attention heads based on entropy scores.
    
    Args:
        model: The transformer model
        dataloader: DataLoader with evaluation data
        prune_ratio: Fraction of heads to prune (0-1)
        num_batches: Number of batches to use for importance calculation
        device: Device to run on
        
    Returns:
        List of (layer_idx, head_idx) tuples for pruned heads
    """
    # Compute head importance
    importance = compute_head_importance(model, dataloader, num_batches, device)
    
    # Sort heads by entropy (higher entropy = less important)
    importance_flat = importance.view(-1)
    num_heads_total = importance_flat.size(0)
    num_heads_to_prune = int(num_heads_total * prune_ratio)
    
    # Get indices of heads to prune (highest entropy)
    _, indices = torch.topk(importance_flat, num_heads_to_prune)
    
    # Convert flat indices to (layer, head) tuples
    num_heads_per_layer = importance.size(1)
    pruned_heads = [
        (int(idx // num_heads_per_layer), int(idx % num_heads_per_layer))
        for idx in indices
    ]
    
    logger.info(f"Selected {len(pruned_heads)} heads for pruning using entropy strategy")
    
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