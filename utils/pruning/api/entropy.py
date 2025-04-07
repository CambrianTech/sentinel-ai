"""
Entropy-based and Magnitude-based pruning methods.

This module provides implementations of attention head pruning strategies based on:
1. Entropy of attention distributions
2. Magnitude of attention weights
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm

def collect_attention_distributions(model, dataloader, num_batches=10, device="cuda"):
    """
    Collect attention output distributions from the model.
    
    Args:
        model: Model to collect attention distributions from
        dataloader: DataLoader with input data
        num_batches: Number of batches to process
        device: Device to run on
        
    Returns:
        Dictionary mapping (layer_idx, head_idx) to list of attention distributions
    """
    print(f"Collecting attention distributions across {num_batches} batches...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Dictionary to store attention distributions
    attention_data = {}
    
    # Process batches
    batch_iterator = iter(dataloader)
    with torch.no_grad():
        for _ in tqdm(range(min(num_batches, len(dataloader))), desc="Collecting attention"):
            try:
                # Get next batch
                batch = next(batch_iterator)
                
                # Handle different batch formats
                if isinstance(batch, tuple) and len(batch) >= 2:
                    input_ids, attention_mask = batch[0], batch[1]
                elif isinstance(batch, dict):
                    input_ids = batch["input_ids"]
                    attention_mask = batch.get("attention_mask", None)
                else:
                    raise ValueError(f"Unsupported batch format: {type(batch)}")
                
                # Move to device
                input_ids = input_ids.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                # Forward pass with hooks to capture attention
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, 
                               output_attentions=True)
                
                # Extract attention outputs
                if hasattr(outputs, "attentions") and outputs.attentions is not None:
                    attentions = outputs.attentions
                    
                    # Process attention outputs
                    for layer_idx, layer_attention in enumerate(attentions):
                        # Shape is typically [batch_size, num_heads, seq_len, seq_len]
                        num_heads = layer_attention.shape[1]
                        
                        for head_idx in range(num_heads):
                            # Get attention weights for this head
                            head_attention = layer_attention[:, head_idx, :, :]
                            
                            # Add to attention_data
                            key = (layer_idx, head_idx)
                            if key not in attention_data:
                                attention_data[key] = []
                            attention_data[key].append(head_attention.detach().cpu())
            except StopIteration:
                break
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
    
    # If we didn't collect any attention data using the normal method, try a different approach
    if not attention_data:
        print("No attention data collected. Trying alternative method...")
        attention_data = _collect_attention_fallback(model, dataloader, num_batches, device)
    
    print(f"Collected attention data for {len(attention_data)} heads")
    return attention_data

def _collect_attention_fallback(model, dataloader, num_batches=10, device="cuda"):
    """
    Fallback method to collect attention distributions when the standard method fails.
    This tries to access attention modules directly.
    
    Args:
        model: Model to collect attention distributions from
        dataloader: DataLoader with input data
        num_batches: Number of batches to process
        device: Device to run on
        
    Returns:
        Dictionary mapping (layer_idx, head_idx) to list of attention distributions
    """
    attention_data = {}
    
    # Try to get blocks/layers from model
    blocks = None
    
    # Check for common transformer architectures
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-2 style
        blocks = model.transformer.h
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        # BERT style
        blocks = model.encoder.layer
    elif hasattr(model, 'layers'):
        # Some other models
        blocks = model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Wrapped models
        blocks = model.model.layers
    
    if blocks is None:
        print("Could not determine model structure for attention collection")
        return attention_data
    
    print(f"Using direct attention collection from {len(blocks)} blocks")
    
    # Set up hooks
    attention_hooks = []
    hook_data = {}
    
    def get_attention_hook(layer_idx, head_idx):
        def hook(module, inputs, outputs):
            # Try to extract attention weights from outputs
            if isinstance(outputs, tuple) and len(outputs) > 0:
                # Some modules return attn_weights as second item in tuple
                attn_weights = outputs[1] if len(outputs) > 1 else outputs[0]
            else:
                attn_weights = outputs
            
            if isinstance(attn_weights, torch.Tensor):
                # Add batch dimension if needed
                if attn_weights.dim() == 3 and head_idx < attn_weights.size(0):
                    # [num_heads, seq_len, seq_len]
                    attn_weights = attn_weights[head_idx].unsqueeze(0)
                elif attn_weights.dim() == 4:
                    # [batch_size, num_heads, seq_len, seq_len]
                    attn_weights = attn_weights[:, head_idx]
                
                key = (layer_idx, head_idx)
                if key not in hook_data:
                    hook_data[key] = []
                hook_data[key].append(attn_weights.detach().cpu())
        return hook
    
    # Register hooks on attention modules
    for layer_idx, block in enumerate(blocks):
        # Try to find attention module
        attn_module = None
        
        # Check common patterns
        if hasattr(block, 'attention'):
            attn_module = block.attention
        elif hasattr(block, 'attn'):
            attn_module = block.attn
        elif hasattr(block, 'self_attention'):
            attn_module = block.self_attention
        elif hasattr(block, 'self_attn'):
            attn_module = block.self_attn
        
        if attn_module is not None:
            # Check if we can get number of heads
            num_heads = 12  # Default
            if hasattr(attn_module, 'num_heads'):
                num_heads = attn_module.num_heads
            elif hasattr(attn_module, 'n_head'):
                num_heads = attn_module.n_head
            elif hasattr(attn_module, 'num_attention_heads'):
                num_heads = attn_module.num_attention_heads
            
            # Find the output of the attention calculation
            if hasattr(attn_module, 'out'):
                hook_target = attn_module.out
            elif hasattr(attn_module, 'attn_dropout'):
                hook_target = attn_module.attn_dropout
            else:
                hook_target = attn_module
            
            # Register hooks for each head
            for head_idx in range(num_heads):
                hook = hook_target.register_forward_hook(
                    get_attention_hook(layer_idx, head_idx)
                )
                attention_hooks.append(hook)
    
    # Run forward passes with the hooks
    model.eval()
    with torch.no_grad():
        batch_iterator = iter(dataloader)
        for _ in tqdm(range(min(num_batches, len(dataloader))), desc="Collecting attention (fallback)"):
            try:
                # Get next batch
                batch = next(batch_iterator)
                
                # Handle different batch formats
                if isinstance(batch, tuple) and len(batch) >= 2:
                    input_ids, attention_mask = batch[0], batch[1]
                elif isinstance(batch, dict):
                    input_ids = batch["input_ids"]
                    attention_mask = batch.get("attention_mask", None)
                else:
                    continue
                
                # Move to device
                input_ids = input_ids.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                # Forward pass to trigger hooks
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
                
            except StopIteration:
                break
            except Exception as e:
                print(f"Error in fallback attention collection: {e}")
                continue
    
    # Remove hooks
    for hook in attention_hooks:
        hook.remove()
    
    return hook_data

def calculate_entropy(attention_maps):
    """
    Calculate entropy of attention maps.
    
    Args:
        attention_maps: List of attention maps
        
    Returns:
        Average entropy across all attention maps
    """
    # Concatenate all attention maps
    all_maps = torch.cat(attention_maps, dim=0)
    
    # Calculate entropy for each sample in the batch
    entropies = []
    for sample_idx in range(all_maps.shape[0]):
        # Get attention map for this sample
        attn_map = all_maps[sample_idx]
        
        # Apply small epsilon to avoid log(0)
        attn_map = attn_map + 1e-10
        attn_map = attn_map / attn_map.sum(-1, keepdim=True)
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -torch.sum(attn_map * torch.log(attn_map)) / attn_map.size(0)
        entropies.append(entropy.item())
    
    # Return average entropy
    return np.mean(entropies)

def entropy_based_pruning(model, attention_data, prune_ratio=0.1, safe_update_tensor_fn=None):
    """
    Prune heads based on their attention entropy.
    
    Args:
        model: Model to prune
        attention_data: Dictionary mapping (layer_idx, head_idx) to attention distributions
        prune_ratio: Fraction of heads to prune
        safe_update_tensor_fn: Function to safely update tensor values
        
    Returns:
        List of pruned head tuples (layer_idx, head_idx)
    """
    print("Computing entropy for each head...")
    
    # Calculate entropy for each head
    head_entropies = {}
    for (layer_idx, head_idx), attention_maps in attention_data.items():
        entropy = calculate_entropy(attention_maps)
        head_entropies[(layer_idx, head_idx)] = entropy
    
    # Sort heads by entropy (lower entropy means the head is more focused)
    # We prune heads with HIGHER entropy (less focused)
    sorted_heads = sorted(head_entropies.items(), key=lambda x: -x[1])
    
    # Determine number of heads to prune
    total_heads = len(head_entropies)
    num_to_prune = int(total_heads * prune_ratio)
    heads_to_prune = [head_info for head_info, _ in sorted_heads[:num_to_prune]]
    
    print(f"Pruning {len(heads_to_prune)} heads with highest entropy")
    
    # Apply pruning
    pruned_heads = []
    for layer_idx, head_idx in heads_to_prune:
        # Find the corresponding attention module
        if prune_head_in_model(model, layer_idx, head_idx, safe_update_tensor_fn):
            pruned_heads.append((layer_idx, head_idx))
    
    print(f"Successfully pruned {len(pruned_heads)} heads based on entropy")
    return pruned_heads

def calculate_weight_magnitude(model, layer_idx, head_idx):
    """
    Calculate the magnitude of weights for a specific attention head.
    
    Args:
        model: The model
        layer_idx: Layer index
        head_idx: Head index
        
    Returns:
        Magnitude of the head weights or 0 if not found
    """
    # Find the transformer blocks
    blocks = None
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-2 style
        blocks = model.transformer.h
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        # BERT style
        blocks = model.encoder.layer
    elif hasattr(model, 'layers'):
        # Some models
        blocks = model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Wrapped models
        blocks = model.model.layers
    
    if blocks is None or layer_idx >= len(blocks):
        return 0
    
    # Get the block
    block = blocks[layer_idx]
    
    # Find attention module
    attn_module = None
    if hasattr(block, 'attention'):
        attn_module = block.attention
    elif hasattr(block, 'attn'):
        attn_module = block.attn
    elif hasattr(block, 'self_attention'):
        attn_module = block.self_attention
    elif hasattr(block, 'self_attn'):
        attn_module = block.self_attn
    
    if attn_module is None:
        return 0
    
    # Try to find query, key, value weights
    magnitude = 0
    
    # Look for query projection
    if hasattr(attn_module, 'q_proj') and isinstance(attn_module.q_proj, nn.Linear):
        # Split weight into heads
        head_size = attn_module.q_proj.weight.size(0) // attn_module.num_heads
        q_weight = attn_module.q_proj.weight[head_idx * head_size:(head_idx+1) * head_size]
        magnitude += torch.norm(q_weight).item()
    
    # Look for key projection
    if hasattr(attn_module, 'k_proj') and isinstance(attn_module.k_proj, nn.Linear):
        # Split weight into heads
        head_size = attn_module.k_proj.weight.size(0) // attn_module.num_heads
        k_weight = attn_module.k_proj.weight[head_idx * head_size:(head_idx+1) * head_size]
        magnitude += torch.norm(k_weight).item()
    
    # Look for value projection
    if hasattr(attn_module, 'v_proj') and isinstance(attn_module.v_proj, nn.Linear):
        # Split weight into heads
        head_size = attn_module.v_proj.weight.size(0) // attn_module.num_heads
        v_weight = attn_module.v_proj.weight[head_idx * head_size:(head_idx+1) * head_size]
        magnitude += torch.norm(v_weight).item()
    
    # For multi-head attention with combined qkv
    if hasattr(attn_module, 'c_attn') and isinstance(attn_module.c_attn, nn.Linear):
        # Assuming GPT-2 style with combined QKV projection
        combined_weight = attn_module.c_attn.weight
        head_size = combined_weight.size(0) // (3 * attn_module.n_head)
        
        # Extract weights for this head (Q, K, V parts)
        q_start = head_idx * head_size
        q_end = (head_idx + 1) * head_size
        q_weight = combined_weight[q_start:q_end]
        magnitude += torch.norm(q_weight).item()
        
        k_start = attn_module.n_head * head_size + head_idx * head_size
        k_end = attn_module.n_head * head_size + (head_idx + 1) * head_size
        k_weight = combined_weight[k_start:k_end]
        magnitude += torch.norm(k_weight).item()
        
        v_start = 2 * attn_module.n_head * head_size + head_idx * head_size
        v_end = 2 * attn_module.n_head * head_size + (head_idx + 1) * head_size
        v_weight = combined_weight[v_start:v_end]
        magnitude += torch.norm(v_weight).item()
    
    # If we couldn't find specific structures, try a fallback
    if magnitude == 0:
        # Just compute magnitude of all parameters in the attention module
        for name, param in attn_module.named_parameters():
            if 'weight' in name:
                magnitude += torch.norm(param).item()
        
        # Assume equal contribution from all heads
        num_heads = 12  # Default
        if hasattr(attn_module, 'num_heads'):
            num_heads = attn_module.num_heads
        elif hasattr(attn_module, 'n_head'):
            num_heads = attn_module.n_head
        elif hasattr(attn_module, 'num_attention_heads'):
            num_heads = attn_module.num_attention_heads
        
        magnitude = magnitude / num_heads
    
    return magnitude

def magnitude_based_pruning(model, prune_ratio=0.1, safe_update_tensor_fn=None):
    """
    Prune heads based on weight magnitudes.
    
    Args:
        model: Model to prune
        prune_ratio: Fraction of heads to prune
        safe_update_tensor_fn: Function to safely update tensor values
        
    Returns:
        List of pruned head tuples (layer_idx, head_idx)
    """
    print("Computing weight magnitudes for each head...")
    
    # Find transformer blocks
    blocks = None
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-2 style
        blocks = model.transformer.h
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        # BERT style
        blocks = model.encoder.layer
    elif hasattr(model, 'layers'):
        # Some models
        blocks = model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Wrapped models
        blocks = model.model.layers
    
    if blocks is None:
        print("Could not find transformer blocks in model")
        return []
    
    # Count total number of heads and calculate magnitudes
    total_heads = 0
    head_magnitudes = {}
    
    # Process each layer and head
    for layer_idx, block in enumerate(blocks):
        # Find attention module
        attn_module = None
        if hasattr(block, 'attention'):
            attn_module = block.attention
        elif hasattr(block, 'attn'):
            attn_module = block.attn
        elif hasattr(block, 'self_attention'):
            attn_module = block.self_attention
        elif hasattr(block, 'self_attn'):
            attn_module = block.self_attn
        
        if attn_module is None:
            continue
        
        # Get number of heads
        num_heads = 12  # Default
        if hasattr(attn_module, 'num_heads'):
            num_heads = attn_module.num_heads
        elif hasattr(attn_module, 'n_head'):
            num_heads = attn_module.n_head
        elif hasattr(attn_module, 'num_attention_heads'):
            num_heads = attn_module.num_attention_heads
        
        # Calculate magnitude for each head
        for head_idx in range(num_heads):
            total_heads += 1
            magnitude = calculate_weight_magnitude(model, layer_idx, head_idx)
            head_magnitudes[(layer_idx, head_idx)] = magnitude
    
    # Sort heads by magnitude (lower magnitude first)
    sorted_heads = sorted(head_magnitudes.items(), key=lambda x: x[1])
    
    # Determine number of heads to prune
    num_to_prune = int(total_heads * prune_ratio)
    heads_to_prune = [head_info for head_info, _ in sorted_heads[:num_to_prune]]
    
    print(f"Pruning {len(heads_to_prune)} heads with lowest magnitude")
    
    # Apply pruning
    pruned_heads = []
    for layer_idx, head_idx in heads_to_prune:
        # Prune the head
        if prune_head_in_model(model, layer_idx, head_idx, safe_update_tensor_fn):
            pruned_heads.append((layer_idx, head_idx))
    
    print(f"Successfully pruned {len(pruned_heads)} heads based on magnitude")
    return pruned_heads

def prune_head_in_model(model, layer_idx, head_idx, safe_update_tensor_fn=None):
    """
    Prune a specific head in the model.
    
    Args:
        model: The model
        layer_idx: Layer index
        head_idx: Head index
        safe_update_tensor_fn: Function to safely update tensor values
        
    Returns:
        True if the head was pruned, False otherwise
    """
    # Find the transformer blocks
    blocks = None
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-2 style
        blocks = model.transformer.h
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        # BERT style
        blocks = model.encoder.layer
    elif hasattr(model, 'layers'):
        # Some models
        blocks = model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Wrapped models
        blocks = model.model.layers
    
    if blocks is None or layer_idx >= len(blocks):
        return False
    
    # Get the block
    block = blocks[layer_idx]
    
    # Find attention module
    attn_module = None
    if hasattr(block, 'attention'):
        attn_module = block.attention
    elif hasattr(block, 'attn'):
        attn_module = block.attn
    elif hasattr(block, 'self_attention'):
        attn_module = block.self_attention
    elif hasattr(block, 'self_attn'):
        attn_module = block.self_attn
    
    if attn_module is None:
        return False
    
    # Check for gate parameter
    if hasattr(attn_module, 'gate'):
        gate = attn_module.gate
        # Use safe update function if provided
        if safe_update_tensor_fn is not None:
            safe_update_tensor_fn(gate, 0.0, index=head_idx)
        else:
            with torch.no_grad():
                gate[head_idx] = 0.0
        return True
    
    # Check for head_gates parameter
    if hasattr(attn_module, 'head_gates'):
        head_gates = attn_module.head_gates
        # Use safe update function if provided
        if safe_update_tensor_fn is not None:
            safe_update_tensor_fn(head_gates, 0.0, index=head_idx)
        else:
            with torch.no_grad():
                head_gates[head_idx] = 0.0
        return True
    
    # Check for gating_weights parameter
    if hasattr(attn_module, 'gating_weights'):
        gating_weights = attn_module.gating_weights
        # Use safe update function if provided
        if safe_update_tensor_fn is not None:
            safe_update_tensor_fn(gating_weights, 0.0, index=head_idx)
        else:
            with torch.no_grad():
                gating_weights[head_idx] = 0.0
        return True
    
    # No gate parameter found
    return False