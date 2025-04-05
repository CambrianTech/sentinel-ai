"""
Advanced pruning strategies based on attention entropy and weight magnitude.

This module implements scientifically rigorous pruning strategies that analyze
attention distributions and weight magnitudes to identify the least important
attention heads for pruning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def compute_attention_entropy(attn_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes entropy per attention head from the attention probability tensor.
    
    Args:
        attn_probs: Attention probability tensor with shape (batch_size, num_heads, seq_len, seq_len)
        eps: Small constant to avoid log(0)
        
    Returns:
        Tensor of shape (num_heads) containing the average entropy for each head
    """
    log_attn = torch.log(attn_probs + eps)
    entropy = -torch.sum(attn_probs * log_attn, dim=-1)  # shape: (batch_size, num_heads, seq_len)
    return entropy.mean(dim=(0, 2))  # average over batch and sequence


def collect_attention_distributions(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader,
    num_batches: int = 10,
    device: Optional[torch.device] = None
) -> Dict[int, torch.Tensor]:
    """
    Collects attention probability distributions from the model for multiple batches.
    
    Args:
        model: The model to collect attention distributions from
        dataloader: DataLoader containing evaluation data
        num_batches: Number of batches to collect distributions from
        device: Device to run on (defaults to model's device)
        
    Returns:
        Dictionary mapping layer indices to attention distributions
    """
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    distributions = {}
    
    # For adaptive transformer in Sentinel-AI, the blocks are accessed differently
    if hasattr(model, 'blocks'):
        # For blocks, we'll collect attention from the blocks directly
        return _collect_attention_from_blocks(model, dataloader, num_batches, device)
        
    # Try standard HuggingFace model structures
    transformer = None
    if hasattr(model, 'transformer'):
        transformer = model.transformer
    elif hasattr(model, 'model') and hasattr(model.model, 'transformer'):
        transformer = model.model.transformer
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'transformer'):
        transformer = model.base_model.transformer
    else:
        logger.warning(f"Could not find transformer attribute in model. Model attributes: {dir(model)}")
        return {}
    
    # Check if transformer has layers
    if not hasattr(transformer, 'h'):
        logger.warning(f"Transformer does not have 'h' attribute. Transformer attributes: {dir(transformer)}")
        return {}
    
    # Hook to capture attention distributions
    attention_distributions = {}
    
    def hook_fn(layer_idx):
        def hook(module, input, output):
            # Get attention scores - implementation depends on model architecture
            if hasattr(output, "attentions") and output.attentions is not None:
                # HuggingFace transformer with output_attentions=True
                attention_distributions[layer_idx] = output.attentions.detach()
            elif isinstance(output, tuple) and len(output) > 1:
                # Some models return a tuple with attentions as second element
                attention_distributions[layer_idx] = output[1].detach()
            else:
                # Standard case - output is the attention probs directly
                attention_distributions[layer_idx] = output.detach()
        return hook
    
    # Register hooks for each attention layer
    hooks = []
    for i, layer in enumerate(transformer.h):
        hook = layer.attn.register_forward_hook(hook_fn(i))
        hooks.append(hook)
    
    # Collect distributions
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            # Prepare inputs
            if isinstance(batch, dict):
                # Dataloader returns dict of tensors
                inputs = {k: v.to(device) for k, v in batch.items()}
            else:
                # Dataloader returns tuple of tensors
                inputs = {
                    "input_ids": batch[0].to(device),
                    "attention_mask": batch[1].to(device) if len(batch) > 1 else None
                }
            
            # Forward pass
            outputs = model(**inputs, output_attentions=True)
            
            # Accumulate distributions
            for layer_idx, attn_probs in attention_distributions.items():
                if layer_idx not in distributions:
                    distributions[layer_idx] = attn_probs
                else:
                    distributions[layer_idx] = torch.cat([distributions[layer_idx], attn_probs], dim=0)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return distributions


def entropy_based_pruning(
    model: nn.Module, 
    attn_distributions: Dict[int, torch.Tensor], 
    prune_ratio: float,
    safe_update_tensor_fn=None
) -> List[Tuple[int, int, float]]:
    """
    Prunes heads with highest entropy (least focused attention).
    
    Args:
        model: The model to prune
        attn_distributions: Dict mapping layer indices to attention probability tensors
        prune_ratio: Ratio of heads to prune (0.0 to 1.0)
        safe_update_tensor_fn: Function to safely update tensor without breaking gradients
        
    Returns:
        List of (layer_idx, head_idx, score) tuples for pruned heads
    """
    # Check if distributions is empty
    if not attn_distributions:
        logger.warning("No attention distributions provided for entropy-based pruning")
        return []
        
    entropy_scores = []  # [(layer_idx, head_idx, score)]
    
    # Compute entropy for each head
    for layer_idx, probs in attn_distributions.items():
        ent = compute_attention_entropy(probs)
        for head_idx, score in enumerate(ent):
            entropy_scores.append((layer_idx, head_idx, score.item()))
    
    # Sort by entropy (highest first - these are least focused)
    entropy_scores.sort(key=lambda x: x[2], reverse=True)
    
    # Calculate number of heads to prune
    num_heads_to_prune = int(len(entropy_scores) * prune_ratio)
    to_prune = entropy_scores[:num_heads_to_prune]
    
    logger.info(f"Entropy-based pruning: pruning {num_heads_to_prune} of {len(entropy_scores)} heads")
    
    # For blocks-based model, apply pruning directly
    if hasattr(model, 'blocks'):
        blocks = model.blocks
        for layer_idx, head_idx, _ in to_prune:
            if 0 <= layer_idx < len(blocks) and hasattr(blocks[layer_idx].attn, 'gate'):
                gate = blocks[layer_idx].attn.gate
                if 0 <= head_idx < len(gate):
                    # Safely update gate
                    if safe_update_tensor_fn is not None:
                        safe_update_tensor_fn(gate, 0.0, index=head_idx)
                    else:
                        with torch.no_grad():
                            gate[head_idx] = 0.0
    else:
        # Apply pruning for standard models
        _apply_pruning(model, to_prune, safe_update_tensor_fn)
    
    return to_prune


def _collect_attention_from_blocks(model, dataloader, num_batches, device=None):
    """
    Collect attention distributions from a model with blocks structure.
    
    Args:
        model: The model with blocks attribute
        dataloader: DataLoader containing evaluation data
        num_batches: Number of batches to collect distributions from
        device: Device to run on
        
    Returns:
        Dictionary mapping layer indices to attention distributions
    """
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    distributions = {}
    
    # Add hooks to the attention modules in blocks
    hooks = []
    attention_outputs = {}
    
    def hook_fn(block_idx):
        def hook(module, input, output):
            # Try to extract attention probabilities
            if isinstance(output, tuple) and len(output) > 1:
                # Some modules return a tuple with attention as second element
                attention_outputs[block_idx] = output[1].detach()
            else:
                # Direct attention output
                attention_outputs[block_idx] = output.detach()
        return hook
    
    # Register hooks for each block's attention module
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'attn'):
            hook = block.attn.register_forward_hook(hook_fn(i))
            hooks.append(hook)
    
    # Process batches
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            # Prepare inputs
            if isinstance(batch, dict):
                # Dataloader returns dict of tensors
                inputs = {k: v.to(device) for k, v in batch.items()}
            else:
                # Dataloader returns tuple of tensors
                inputs = {
                    "input_ids": batch[0].to(device),
                    "attention_mask": batch[1].to(device) if len(batch) > 1 else None
                }
            
            # Forward pass
            model(**inputs, output_attentions=True)
            
            # Process collected attention outputs
            for block_idx, attn_output in attention_outputs.items():
                # Add to distributions
                if block_idx not in distributions:
                    distributions[block_idx] = attn_output
                else:
                    distributions[block_idx] = torch.cat([distributions[block_idx], attn_output], dim=0)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return distributions


def magnitude_based_pruning(
    model: nn.Module, 
    prune_ratio: float,
    safe_update_tensor_fn=None
) -> List[Tuple[int, int, float]]:
    """
    Prunes heads with lowest Q/K/V/O weight magnitude.
    
    Args:
        model: The model to prune
        prune_ratio: Ratio of heads to prune (0.0 to 1.0)
        safe_update_tensor_fn: Function to safely update tensor without breaking gradients
        
    Returns:
        List of (layer_idx, head_idx, score) tuples for pruned heads
    """
    magnitude_scores = []  # [(layer_idx, head_idx, magnitude)]
    
    # Get model layers, handling different model structures
    # For adaptive transformer in Sentinel-AI, the blocks are accessed differently
    if hasattr(model, 'blocks'):
        # Direct access to blocks
        return _compute_magnitude_from_blocks(model.blocks, prune_ratio, safe_update_tensor_fn)
    
    # Try standard HuggingFace model structures
    transformer = None
    if hasattr(model, 'transformer'):
        transformer = model.transformer
    elif hasattr(model, 'model') and hasattr(model.model, 'transformer'):
        transformer = model.model.transformer
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'transformer'):
        transformer = model.base_model.transformer
    else:
        logger.warning(f"Could not find transformer attribute in model. Model attributes: {dir(model)}")
        return []
    
    # Check if transformer has layers
    if not hasattr(transformer, 'h'):
        logger.warning(f"Transformer does not have 'h' attribute. Transformer attributes: {dir(transformer)}")
        return []
    
    # Compute magnitude for each head
    for i, layer in enumerate(transformer.h):
        # Handle different model architectures for weight access
        if hasattr(layer.attn, 'c_attn'):
            # GPT-2 style
            qkv_weight = layer.attn.c_attn.weight  # shape: (3 * embed_dim, embed_dim)
            proj_weight = layer.attn.c_proj.weight
            num_heads = layer.attn.num_heads
            head_dim = qkv_weight.shape[0] // 3 // num_heads
            
            # Compute magnitude for each head
            for h in range(num_heads):
                # Calculate offsets for Q, K, V
                q_start = h * head_dim
                q_end = (h + 1) * head_dim
                k_start = q_start + num_heads * head_dim
                k_end = q_end + num_heads * head_dim
                v_start = k_start + num_heads * head_dim
                v_end = k_end + num_heads * head_dim
                
                # Extract weights for this head
                q = qkv_weight[q_start:q_end]
                k = qkv_weight[k_start:k_end]
                v = qkv_weight[v_start:v_end]
                o = proj_weight[:, q_start:q_end]
                
                # Compute magnitude as norm of all weights
                magnitude = (q.norm() + k.norm() + v.norm() + o.norm())
                magnitude_scores.append((i, h, magnitude.item()))
        
        elif hasattr(layer.attn, 'q_proj'):
            # BERT/RoBERTa style
            q_weight = layer.attn.q_proj.weight  # shape: (embed_dim, embed_dim)
            k_weight = layer.attn.k_proj.weight
            v_weight = layer.attn.v_proj.weight
            o_weight = layer.attn.o_proj.weight
            num_heads = layer.attn.num_attention_heads
            head_dim = q_weight.shape[0] // num_heads
            
            # Compute magnitude for each head
            for h in range(num_heads):
                start_idx = h * head_dim
                end_idx = (h + 1) * head_dim
                
                q = q_weight[start_idx:end_idx]
                k = k_weight[start_idx:end_idx]
                v = v_weight[start_idx:end_idx]
                o = o_weight[:, start_idx:end_idx]
                
                magnitude = (q.norm() + k.norm() + v.norm() + o.norm())
                magnitude_scores.append((i, h, magnitude.item()))
        
        else:
            # Generic fallback - less accurate but tries to handle unknown architectures
            logger.warning(f"Unknown attention architecture for layer {i}. Using fallback magnitude calculation.")
            for param_name, param in layer.attn.named_parameters():
                if "weight" in param_name and param.dim() > 1:
                    # Use any weight matrix we can find as a proxy for head importance
                    if hasattr(layer.attn, 'num_heads'):
                        num_heads = layer.attn.num_heads
                    elif hasattr(layer.attn, 'num_attention_heads'):
                        num_heads = layer.attn.num_attention_heads
                    else:
                        num_heads = 12  # Default fallback
                        
                    # Split the weight matrix by heads and compute magnitude
                    head_dim = param.shape[0] // num_heads
                    for h in range(num_heads):
                        start_idx = h * head_dim
                        end_idx = (h + 1) * head_dim
                        magnitude = param[start_idx:end_idx].norm().item()
                        magnitude_scores.append((i, h, magnitude))
    
    # Sort by magnitude (lowest first)
    magnitude_scores.sort(key=lambda x: x[2])
    
    # Calculate number of heads to prune
    num_heads_to_prune = int(len(magnitude_scores) * prune_ratio)
    to_prune = magnitude_scores[:num_heads_to_prune]
    
    logger.info(f"Magnitude-based pruning: pruning {num_heads_to_prune} of {len(magnitude_scores)} heads")
    
    # Apply pruning
    _apply_pruning(model, to_prune, safe_update_tensor_fn)
    
    return to_prune


def _compute_magnitude_from_blocks(blocks, prune_ratio, safe_update_tensor_fn=None):
    """
    Computes magnitude-based pruning for Sentinel-AI's adaptive transformer blocks.
    
    Args:
        blocks: The transformer blocks
        prune_ratio: Ratio of heads to prune (0.0 to 1.0)
        safe_update_tensor_fn: Function to safely update tensor without breaking gradients
        
    Returns:
        List of (layer_idx, head_idx, score) tuples for pruned heads
    """
    magnitude_scores = []  # [(layer_idx, head_idx, magnitude)]
    
    # Compute magnitude for each head in each block
    for i, block in enumerate(blocks):
        if not hasattr(block, 'attn'):
            logger.warning(f"Block {i} does not have attention module")
            continue
            
        attn = block.attn
        
        # Check for gate attribute
        if not hasattr(attn, 'gate'):
            logger.warning(f"Attention module in block {i} does not have gate attribute")
            continue
            
        # Count heads
        num_heads = len(attn.gate)
        
        # Process each head
        for h in range(num_heads):
            # For adaptive transformer, we check all weight matrices that might contribute to this head
            magnitude = 0.0
            
            # Get all parameters that might be associated with the head
            for name, param in attn.named_parameters():
                if 'weight' in name:
                    # This is a parameter that might contribute to the head's function
                    try:
                        if param.dim() > 1:
                            head_dim = param.size(0) // num_heads
                            start_idx = h * head_dim
                            end_idx = (h + 1) * head_dim
                            
                            # Try to extract relevant part of the parameter
                            if param.size(0) % num_heads == 0:
                                head_param = param[start_idx:end_idx]
                                magnitude += head_param.norm().item()
                    except Exception as e:
                        # Skip if we can't get a norm for this parameter
                        pass
            
            # Add to list
            magnitude_scores.append((i, h, magnitude))
    
    # Sort by magnitude (lowest first for pruning)
    magnitude_scores.sort(key=lambda x: x[2])
    
    # Calculate number of heads to prune
    num_heads_to_prune = int(len(magnitude_scores) * prune_ratio)
    to_prune = magnitude_scores[:num_heads_to_prune]
    
    logger.info(f"Magnitude-based pruning: pruning {num_heads_to_prune} of {len(magnitude_scores)} heads from blocks")
    
    # Apply pruning directly to the blocks
    for layer_idx, head_idx, _ in to_prune:
        if 0 <= layer_idx < len(blocks) and hasattr(blocks[layer_idx].attn, 'gate'):
            gate = blocks[layer_idx].attn.gate
            if 0 <= head_idx < len(gate):
                # Safely update gate
                if safe_update_tensor_fn is not None:
                    safe_update_tensor_fn(gate, 0.0, index=head_idx)
                else:
                    with torch.no_grad():
                        gate[head_idx] = 0.0
    
    return to_prune


def _apply_pruning(
    model: nn.Module, 
    heads_to_prune: List[Tuple[int, int, float]],
    safe_update_tensor_fn=None
):
    """
    Apply pruning to the specified heads.
    
    Args:
        model: The model to prune
        heads_to_prune: List of (layer_idx, head_idx, score) tuples
        safe_update_tensor_fn: Function to safely update tensor without breaking gradients
    """
    # Group by layer for efficiency
    pruning_by_layer = {}
    for layer_idx, head_idx, _ in heads_to_prune:
        if layer_idx not in pruning_by_layer:
            pruning_by_layer[layer_idx] = []
        pruning_by_layer[layer_idx].append(head_idx)
    
    # Apply pruning
    for layer_idx, head_indices in pruning_by_layer.items():
        # Get model layers, handling different model structures
        transformer = None
        if hasattr(model, 'transformer'):
            transformer = model.transformer
        elif hasattr(model, 'model') and hasattr(model.model, 'transformer'):
            transformer = model.model.transformer
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'transformer'):
            transformer = model.base_model.transformer
        else:
            logger.warning(f"Could not find transformer attribute in model. Model attributes: {dir(model)}")
            continue
        
        # Get layer
        if not hasattr(transformer, 'h'):
            logger.warning(f"Transformer does not have 'h' attribute. Transformer attributes: {dir(transformer)}")
            continue
            
        if layer_idx >= len(transformer.h):
            logger.warning(f"Layer index {layer_idx} out of range for model with {len(transformer.h)} layers")
            continue
            
        layer = transformer.h[layer_idx]
        
        # Check for mask_heads method (HuggingFace compatible)
        if hasattr(layer.attn, 'mask_heads'):
            layer.attn.mask_heads(head_indices)
            
        # Check for head_mask attribute
        elif hasattr(layer.attn, 'head_mask'):
            update_mask(layer.attn.head_mask, head_indices, 0.0, safe_update_tensor_fn)
            
        # Check for pruning_mask
        elif hasattr(layer.attn, 'pruning_mask'):
            update_mask(layer.attn.pruning_mask, head_indices, 0.0, safe_update_tensor_fn)
            
        # Check for gate parameter
        elif hasattr(layer.attn, 'gate'):
            update_mask(layer.attn.gate, head_indices, 0.0, safe_update_tensor_fn)
            
        # Generic fallback - add a head mask if none exists
        else:
            logger.warning(f"No pruning mechanism found for layer {layer_idx}. Creating a head mask.")
            if hasattr(layer.attn, 'num_heads'):
                num_heads = layer.attn.num_heads
            elif hasattr(layer.attn, 'num_attention_heads'):
                num_heads = layer.attn.num_attention_heads
            else:
                num_heads = 12  # Default fallback
                
            head_mask = torch.ones(num_heads, device=next(model.parameters()).device)
            layer.attn.register_buffer("head_mask", head_mask)
            update_mask(layer.attn.head_mask, head_indices, 0.0, safe_update_tensor_fn)


def update_mask(
    mask: torch.Tensor, 
    indices: List[int], 
    value: float, 
    safe_update_fn=None
):
    """
    Safely update a mask tensor without breaking gradients.
    
    Args:
        mask: The mask tensor to update
        indices: List of indices to update
        value: Value to set at the specified indices
        safe_update_fn: Function to safely update tensor without breaking gradients
    """
    if safe_update_fn is not None:
        # Use provided safe update function
        for idx in indices:
            safe_update_fn(mask, value, index=idx)
    else:
        # Fallback - try to update safely without breaking gradients
        with torch.no_grad():
            for idx in indices:
                mask[idx] = value