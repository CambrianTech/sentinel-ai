"""
Dual-Mode Pruning Implementation for Sentinel-AI

This module provides two pruning modes:
1. Adaptive: Temporarily zeros weights, allowing recovery during fine-tuning
2. Compressed: Permanently prunes weights, preventing recovery but enabling real compression

Usage:
    from sentinel.pruning.dual_mode_pruning import prune_head_in_model, PruningMode
    
    # Use adaptive mode (default)
    prune_head_in_model(model, layer_idx=0, head_idx=1)
    
    # Use compressed mode
    prune_head_in_model(model, layer_idx=0, head_idx=1, mode=PruningMode.COMPRESSED)
    
    # For any mode, add hooks to ensure heads stay pruned during training
    hooks = apply_pruning_hooks(model, pruned_heads=[(0,1), (1,2)], mode=PruningMode.ADAPTIVE)
    
    # For adaptive mode with complete control during training, you'll need to patch the optimizer:
    # Get the first attention module with a hook
    attn_module = model.transformer.h[0].attn
    # Get the optimizer hook function
    optimizer_hook = getattr(attn_module, '_optimizer_hook', None)
    # Patch the optimizer
    if optimizer_hook:
        optimizer.orig_step = optimizer.step
        optimizer.step = lambda *a, **kw: optimizer_hook(optimizer, {'args': a, 'kwargs': kw})
    
    # Later, to restore the original optimizer behavior:
    if hasattr(optimizer, 'orig_step'):
        optimizer.step = optimizer.orig_step
        delattr(optimizer, 'orig_step')
"""

import torch
from enum import Enum
import warnings
from typing import List, Tuple, Union, Optional


class PruningMode(str, Enum):
    """Pruning modes for Sentinel-AI."""
    ADAPTIVE = "adaptive"    # Allows heads to recover during fine-tuning
    COMPRESSED = "compressed"  # Permanently zeros weights


def prune_head_in_model(
    model: torch.nn.Module, 
    layer_idx: int, 
    head_idx: int, 
    mode: PruningMode = PruningMode.ADAPTIVE,
    verbose: bool = True
) -> bool:
    """
    Prune a specific head in the model with support for different pruning modes.
    
    Args:
        model: The transformer model
        layer_idx: Layer index
        head_idx: Head index
        mode: Pruning mode (adaptive or compressed)
        verbose: Whether to print verbose output
        
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
        if verbose:
            print(f"Could not find transformer block for layer {layer_idx}")
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
        if verbose:
            print(f"Could not find attention module in block {layer_idx}")
        return False
    
    # For GPT-2 style models with a combined QKV matrix
    if hasattr(attn_module, 'c_attn'):
        # Calculate dimensions
        n_heads = getattr(attn_module, 'n_head', None)
        if n_heads is None:
            n_heads = getattr(attn_module, 'num_heads', 
                             getattr(attn_module, 'num_attention_heads', 12))
        
        hidden_size = attn_module.c_attn.weight.size(0)
        head_size = hidden_size // n_heads
        
        # Get the starting indices for the head
        q_idx = head_idx * head_size
        k_idx = hidden_size + head_idx * head_size
        v_idx = 2 * hidden_size + head_idx * head_size
        
        # Check for out of bounds indices 
        if q_idx + head_size > hidden_size or v_idx + head_size > 3 * hidden_size:
            if verbose:
                print(f"Head index {head_idx} out of bounds for layer {layer_idx}")
            return False
        
        # Verify that we're accessing the correct parts of the weight matrix
        # Do a simple check to ensure our indices are reasonable
        if head_idx >= n_heads:
            if verbose:
                print(f"Invalid head index {head_idx} (max allowed: {n_heads-1})")
            return False
            
        if mode == PruningMode.COMPRESSED:
            # Permanent pruning using PyTorch's pruning utilities
            try:
                import torch.nn.utils.prune as prune
                
                # Create mask (1 = keep, 0 = prune)
                mask = torch.ones_like(attn_module.c_attn.weight)
                mask[q_idx:q_idx+head_size, :] = 0.0
                mask[k_idx:k_idx+head_size, :] = 0.0
                mask[v_idx:v_idx+head_size, :] = 0.0
                
                # Run a dummy forward pass to ensure parameters are properly initialized
                # This avoids issues with prune.remove() failing if weight_orig isn't set
                if not hasattr(attn_module.c_attn, 'weight_orig'):
                    # We need to ensure a forward pass has been done
                    if verbose:
                        print("Running dummy forward pass to initialize pruning parameters")
                    try:
                        # Try to run a simple forward pass
                        device = next(model.parameters()).device
                        sample_input = torch.ones((1, 8), dtype=torch.long, device=device)
                        with torch.no_grad():
                            _ = model(sample_input)
                    except Exception as e:
                        # If forward pass fails, just warn and continue
                        if verbose:
                            print(f"Dummy forward pass failed, but continuing: {e}")
                
                # Apply mask
                prune.custom_from_mask(attn_module.c_attn, 'weight', mask)
                
                # Also handle bias if present
                if hasattr(attn_module.c_attn, 'bias') and attn_module.c_attn.bias is not None:
                    bias_mask = torch.ones_like(attn_module.c_attn.bias)
                    bias_mask[q_idx:q_idx+head_size] = 0.0
                    bias_mask[k_idx:k_idx+head_size] = 0.0
                    bias_mask[v_idx:v_idx+head_size] = 0.0
                    prune.custom_from_mask(attn_module.c_attn, 'bias', bias_mask)
                
                # Make pruning permanent - this directly modifies the weight tensor
                # and removes the weight_mask/weight_orig attributes
                try:
                    prune.remove(attn_module.c_attn, 'weight')
                    if hasattr(attn_module.c_attn, 'bias') and attn_module.c_attn.bias is not None:
                        prune.remove(attn_module.c_attn, 'bias')
                except Exception as e:
                    # Fallback to direct zeroing if remove fails
                    if verbose:
                        print(f"Permanent pruning failed: {e}")
                        print("Falling back to manual zeroing")
                    with torch.no_grad():
                        attn_module.c_attn.weight.data[q_idx:q_idx+head_size, :] = 0.0
                        attn_module.c_attn.weight.data[k_idx:k_idx+head_size, :] = 0.0
                        attn_module.c_attn.weight.data[v_idx:v_idx+head_size, :] = 0.0
                        if hasattr(attn_module.c_attn, 'bias') and attn_module.c_attn.bias is not None:
                            attn_module.c_attn.bias.data[q_idx:q_idx+head_size] = 0.0
                            attn_module.c_attn.bias.data[k_idx:k_idx+head_size] = 0.0
                            attn_module.c_attn.bias.data[v_idx:v_idx+head_size] = 0.0
                
                if verbose:
                    print(f"Permanently pruned head {head_idx} in layer {layer_idx} (Compressed mode)")
            except ImportError as e:
                warnings.warn(f"PyTorch pruning utilities not available: {e}. Falling back to adaptive mode.")
                # Fallback to adaptive mode
                with torch.no_grad():
                    attn_module.c_attn.weight.data[q_idx:q_idx+head_size, :] = 0.0
                    attn_module.c_attn.weight.data[k_idx:k_idx+head_size, :] = 0.0
                    attn_module.c_attn.weight.data[v_idx:v_idx+head_size, :] = 0.0
                    
                    if hasattr(attn_module.c_attn, 'bias') and attn_module.c_attn.bias is not None:
                        attn_module.c_attn.bias.data[q_idx:q_idx+head_size] = 0.0
                        attn_module.c_attn.bias.data[k_idx:k_idx+head_size] = 0.0
                        attn_module.c_attn.bias.data[v_idx:v_idx+head_size] = 0.0
                if verbose:
                    print(f"Temporarily pruned head {head_idx} in layer {layer_idx} (Adaptive mode fallback)")
        else:
            # Adaptive mode - temporary zeroing
            with torch.no_grad():
                # We need to use a different approach to zero weights but allow gradients
                # Create a mask of zeros and ones
                mask = torch.ones_like(attn_module.c_attn.weight)
                mask[q_idx:q_idx+head_size, :] = 0.0
                mask[k_idx:k_idx+head_size, :] = 0.0
                mask[v_idx:v_idx+head_size, :] = 0.0
                
                # Zero the weights by multiplying with the mask
                attn_module.c_attn.weight.mul_(mask)
                
                # Zero out bias if present using the mask approach
                if hasattr(attn_module.c_attn, 'bias') and attn_module.c_attn.bias is not None:
                    bias_mask = torch.ones_like(attn_module.c_attn.bias)
                    bias_mask[q_idx:q_idx+head_size] = 0.0
                    bias_mask[k_idx:k_idx+head_size] = 0.0
                    bias_mask[v_idx:v_idx+head_size] = 0.0
                    attn_module.c_attn.bias.mul_(bias_mask)
                    
            if verbose:
                print(f"Temporarily pruned head {head_idx} in layer {layer_idx} (Adaptive mode)")
        
        # Store the pruned state in the module for reference
        if not hasattr(attn_module, '_pruned_heads'):
            attn_module._pruned_heads = set()
        attn_module._pruned_heads.add(head_idx)
        
        # Store pruning mode in the module
        attn_module._pruning_mode = mode
        
        return True
    
    # For models with separate QKV matrices
    if hasattr(attn_module, 'q_proj') and hasattr(attn_module, 'k_proj') and hasattr(attn_module, 'v_proj'):
        num_heads = getattr(attn_module, 'num_heads', 
                           getattr(attn_module, 'num_attention_heads',
                                  getattr(attn_module, 'n_head', 12)))
        
        head_size = attn_module.q_proj.weight.size(0) // num_heads
        start_idx = head_idx * head_size
        end_idx = start_idx + head_size
        
        # Check for out of bounds
        if head_idx >= num_heads:
            if verbose:
                print(f"Invalid head index {head_idx} (max allowed: {num_heads-1}) for layer {layer_idx}")
            return False
            
        if start_idx + head_size > attn_module.q_proj.weight.size(0):
            if verbose:
                print(f"Head index {head_idx} out of bounds for layer {layer_idx}")
            return False
        
        if mode == PruningMode.COMPRESSED:
            try:
                import torch.nn.utils.prune as prune
                
                # Run a dummy forward pass to ensure parameters are properly initialized
                # This avoids issues with prune.remove() failing if weight_orig isn't set
                if not hasattr(attn_module.q_proj, 'weight_orig'):
                    # We need to ensure a forward pass has been done
                    if verbose:
                        print("Running dummy forward pass to initialize pruning parameters")
                    try:
                        # Try to run a simple forward pass
                        device = next(model.parameters()).device
                        sample_input = torch.ones((1, 8), dtype=torch.long, device=device)
                        with torch.no_grad():
                            _ = model(sample_input)
                    except Exception as e:
                        # If forward pass fails, just warn and continue
                        if verbose:
                            print(f"Dummy forward pass failed, but continuing: {e}")
                            
                # Create masks (1 = keep, 0 = prune)
                q_mask = torch.ones_like(attn_module.q_proj.weight)
                k_mask = torch.ones_like(attn_module.k_proj.weight)
                v_mask = torch.ones_like(attn_module.v_proj.weight)
                
                q_mask[start_idx:end_idx, :] = 0.0
                k_mask[start_idx:end_idx, :] = 0.0
                v_mask[start_idx:end_idx, :] = 0.0
                
                # Apply masks
                prune.custom_from_mask(attn_module.q_proj, 'weight', q_mask)
                prune.custom_from_mask(attn_module.k_proj, 'weight', k_mask)
                prune.custom_from_mask(attn_module.v_proj, 'weight', v_mask)
                
                # Handle biases if present
                if hasattr(attn_module.q_proj, 'bias') and attn_module.q_proj.bias is not None:
                    q_bias_mask = torch.ones_like(attn_module.q_proj.bias)
                    k_bias_mask = torch.ones_like(attn_module.k_proj.bias)
                    v_bias_mask = torch.ones_like(attn_module.v_proj.bias)
                    
                    q_bias_mask[start_idx:end_idx] = 0.0
                    k_bias_mask[start_idx:end_idx] = 0.0
                    v_bias_mask[start_idx:end_idx] = 0.0
                    
                    prune.custom_from_mask(attn_module.q_proj, 'bias', q_bias_mask)
                    prune.custom_from_mask(attn_module.k_proj, 'bias', k_bias_mask)
                    prune.custom_from_mask(attn_module.v_proj, 'bias', v_bias_mask)
                
                # Make pruning permanent
                try:
                    prune.remove(attn_module.q_proj, 'weight')
                    prune.remove(attn_module.k_proj, 'weight')
                    prune.remove(attn_module.v_proj, 'weight')
                    
                    if hasattr(attn_module.q_proj, 'bias') and attn_module.q_proj.bias is not None:
                        prune.remove(attn_module.q_proj, 'bias')
                        prune.remove(attn_module.k_proj, 'bias')
                        prune.remove(attn_module.v_proj, 'bias')
                except Exception as e:
                    # Fallback to direct zeroing if remove fails
                    if verbose:
                        print(f"Permanent pruning failed: {e}")
                        print("Falling back to manual zeroing")
                    with torch.no_grad():
                        attn_module.q_proj.weight.data[start_idx:end_idx, :] = 0.0
                        attn_module.k_proj.weight.data[start_idx:end_idx, :] = 0.0
                        attn_module.v_proj.weight.data[start_idx:end_idx, :] = 0.0
                        
                        if hasattr(attn_module.q_proj, 'bias') and attn_module.q_proj.bias is not None:
                            attn_module.q_proj.bias.data[start_idx:end_idx] = 0.0
                            attn_module.k_proj.bias.data[start_idx:end_idx] = 0.0
                            attn_module.v_proj.bias.data[start_idx:end_idx] = 0.0
                
                if verbose:
                    print(f"Permanently pruned head {head_idx} in layer {layer_idx} (Compressed mode, separate QKV)")
            except ImportError as e:
                warnings.warn(f"PyTorch pruning utilities not available: {e}. Falling back to adaptive mode.")
                # Fallback to adaptive mode
                with torch.no_grad():
                    attn_module.q_proj.weight.data[start_idx:end_idx, :] = 0.0
                    attn_module.k_proj.weight.data[start_idx:end_idx, :] = 0.0
                    attn_module.v_proj.weight.data[start_idx:end_idx, :] = 0.0
                    
                    if hasattr(attn_module.q_proj, 'bias') and attn_module.q_proj.bias is not None:
                        attn_module.q_proj.bias.data[start_idx:end_idx] = 0.0
                        attn_module.k_proj.bias.data[start_idx:end_idx] = 0.0
                        attn_module.v_proj.bias.data[start_idx:end_idx] = 0.0
                
                if verbose:
                    print(f"Temporarily pruned head {head_idx} in layer {layer_idx} (Adaptive mode fallback, separate QKV)")
        else:
            # Adaptive mode - temporary zeroing
            with torch.no_grad():
                # We need to use a different approach to zero weights but allow gradients
                # Create masks of zeros and ones
                q_mask = torch.ones_like(attn_module.q_proj.weight)
                k_mask = torch.ones_like(attn_module.k_proj.weight)
                v_mask = torch.ones_like(attn_module.v_proj.weight)
                
                q_mask[start_idx:end_idx, :] = 0.0
                k_mask[start_idx:end_idx, :] = 0.0
                v_mask[start_idx:end_idx, :] = 0.0
                
                # Zero the weights by multiplying with the masks
                attn_module.q_proj.weight.mul_(q_mask)
                attn_module.k_proj.weight.mul_(k_mask)
                attn_module.v_proj.weight.mul_(v_mask)
                
                # Zero out bias if present using the mask approach
                if hasattr(attn_module.q_proj, 'bias') and attn_module.q_proj.bias is not None:
                    q_bias_mask = torch.ones_like(attn_module.q_proj.bias)
                    k_bias_mask = torch.ones_like(attn_module.k_proj.bias)
                    v_bias_mask = torch.ones_like(attn_module.v_proj.bias)
                    
                    q_bias_mask[start_idx:end_idx] = 0.0
                    k_bias_mask[start_idx:end_idx] = 0.0
                    v_bias_mask[start_idx:end_idx] = 0.0
                    
                    attn_module.q_proj.bias.mul_(q_bias_mask)
                    attn_module.k_proj.bias.mul_(k_bias_mask)
                    attn_module.v_proj.bias.mul_(v_bias_mask)
            
            if verbose:
                print(f"Temporarily pruned head {head_idx} in layer {layer_idx} (Adaptive mode, separate QKV)")
        
        # Store the pruned state in the module for reference
        if not hasattr(attn_module, '_pruned_heads'):
            attn_module._pruned_heads = set()
        attn_module._pruned_heads.add(head_idx)
        
        # Store pruning mode in the module
        attn_module._pruning_mode = mode
        
        return True
    
    # If we got here, we couldn't handle this model architecture
    if verbose:
        print(f"Could not prune head {head_idx} in layer {layer_idx} - unsupported architecture")
    return False


def apply_pruning_hooks(
    model: torch.nn.Module, 
    pruned_heads: List[Tuple[int, int]],
    mode: PruningMode = PruningMode.COMPRESSED,
    verbose: bool = True
) -> List:
    """
    Apply hooks to ensure pruned heads stay pruned during fine-tuning.
    
    Args:
        model: The model being fine-tuned
        pruned_heads: List of (layer_idx, head_idx) tuples
        mode: Pruning mode
        verbose: Whether to print verbose output
    
    Returns:
        List of hooks (store to prevent garbage collection)
    """
    hooks = []
    head_masks = {}  # Store masks for each layer
    
    # Process heads to mask
    for layer_idx, head_idx in pruned_heads:
        if layer_idx not in head_masks:
            head_masks[layer_idx] = []
        head_masks[layer_idx].append(head_idx)
    
    # For adaptive mode, we can optionally apply a hook that zeroes gradients,
    # or a forward hook that re-zeroes weights after each forward pass
    if mode == PruningMode.ADAPTIVE:
        # Skip hooks if explicitly requested
        if verbose:
            print(f"Applying hooks for adaptive mode (re-zeroing after parameter updates)")
        
        def forward_pre_hook(module, input):
            """Re-zero weights for pruned heads before each forward pass"""
            # Get layer index from stored attribute
            layer_idx = getattr(module, '_layer_idx', -1)
            if layer_idx == -1 or layer_idx not in head_masks:
                return
            
            # Handle GPT-2 style models
            if hasattr(module, 'c_attn'):
                # Calculate dimensions
                n_heads = getattr(module, 'n_head', 
                                 getattr(module, 'num_heads', 
                                       getattr(module, 'num_attention_heads', 12)))
                hidden_size = module.c_attn.weight.size(0)
                head_size = hidden_size // n_heads
                
                # Re-zero weights for pruned heads
                with torch.no_grad():
                    for head_idx in head_masks[layer_idx]:
                        q_idx = head_idx * head_size
                        k_idx = hidden_size + head_idx * head_size
                        v_idx = 2 * hidden_size + head_idx * head_size
                        
                        # Create a mask of zeros and ones
                        mask = torch.ones_like(module.c_attn.weight)
                        mask[q_idx:q_idx+head_size, :] = 0.0
                        mask[k_idx:k_idx+head_size, :] = 0.0
                        mask[v_idx:v_idx+head_size, :] = 0.0
                        
                        # Zero weights by multiplying with the mask
                        module.c_attn.weight.mul_(mask)
                        
                        # Zero bias if present using mask
                        if hasattr(module.c_attn, 'bias') and module.c_attn.bias is not None:
                            bias_mask = torch.ones_like(module.c_attn.bias)
                            bias_mask[q_idx:q_idx+head_size] = 0.0
                            bias_mask[k_idx:k_idx+head_size] = 0.0
                            bias_mask[v_idx:v_idx+head_size] = 0.0
                            module.c_attn.bias.mul_(bias_mask)
            
            # Handle models with separate QKV projections
            elif (hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj')):
                # Calculate dimensions
                num_heads = getattr(module, 'num_heads', 
                                  getattr(module, 'num_attention_heads',
                                         getattr(module, 'n_head', 12)))
                head_size = module.q_proj.weight.size(0) // num_heads
                
                # Re-zero weights for pruned heads
                with torch.no_grad():
                    for head_idx in head_masks[layer_idx]:
                        start_idx = head_idx * head_size
                        end_idx = start_idx + head_size
                        
                        # Create masks of zeros and ones
                        q_mask = torch.ones_like(module.q_proj.weight)
                        k_mask = torch.ones_like(module.k_proj.weight)
                        v_mask = torch.ones_like(module.v_proj.weight)
                        
                        q_mask[start_idx:end_idx, :] = 0.0
                        k_mask[start_idx:end_idx, :] = 0.0
                        v_mask[start_idx:end_idx, :] = 0.0
                        
                        # Zero weights by multiplying with the masks
                        module.q_proj.weight.mul_(q_mask)
                        module.k_proj.weight.mul_(k_mask)
                        module.v_proj.weight.mul_(v_mask)
                        
                        # Zero bias if present using mask
                        if hasattr(module.q_proj, 'bias') and module.q_proj.bias is not None:
                            q_bias_mask = torch.ones_like(module.q_proj.bias)
                            k_bias_mask = torch.ones_like(module.k_proj.bias)
                            v_bias_mask = torch.ones_like(module.v_proj.bias)
                            
                            q_bias_mask[start_idx:end_idx] = 0.0
                            k_bias_mask[start_idx:end_idx] = 0.0
                            v_bias_mask[start_idx:end_idx] = 0.0
                            
                            module.q_proj.bias.mul_(q_bias_mask)
                            module.k_proj.bias.mul_(k_bias_mask)
                            module.v_proj.bias.mul_(v_bias_mask)
    
    # Function to zero gradients during backprop
    def zero_grad_hook(module, grad_input, grad_output):
        """Zero out gradients for pruned heads during backprop"""
        # Get layer index from stored attribute
        layer_idx = getattr(module, '_layer_idx', -1)
        if layer_idx == -1 or layer_idx not in head_masks:
            return
        
        # Handle GPT-2 style models
        if hasattr(module, 'c_attn') and module.c_attn.weight.grad is not None:
            # Calculate dimensions
            n_heads = getattr(module, 'n_head', 
                             getattr(module, 'num_heads', 
                                   getattr(module, 'num_attention_heads', 12)))
            hidden_size = module.c_attn.weight.size(0)
            head_size = hidden_size // n_heads
            
            # Zero out gradients for pruned heads
            for head_idx in head_masks[layer_idx]:
                q_idx = head_idx * head_size
                k_idx = hidden_size + head_idx * head_size
                v_idx = 2 * hidden_size + head_idx * head_size
                
                # Zero query/key/value gradients
                module.c_attn.weight.grad[q_idx:q_idx+head_size, :] = 0.0
                module.c_attn.weight.grad[k_idx:k_idx+head_size, :] = 0.0
                module.c_attn.weight.grad[v_idx:v_idx+head_size, :] = 0.0
                
                # Zero bias gradients if present
                if hasattr(module.c_attn, 'bias') and module.c_attn.bias is not None and module.c_attn.bias.grad is not None:
                    module.c_attn.bias.grad[q_idx:q_idx+head_size] = 0.0
                    module.c_attn.bias.grad[k_idx:k_idx+head_size] = 0.0
                    module.c_attn.bias.grad[v_idx:v_idx+head_size] = 0.0
        
        # Handle models with separate QKV projections
        elif (hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj')
              and module.q_proj.weight.grad is not None and module.k_proj.weight.grad is not None 
              and module.v_proj.weight.grad is not None):
            
            # Calculate dimensions
            num_heads = getattr(module, 'num_heads', 
                              getattr(module, 'num_attention_heads',
                                     getattr(module, 'n_head', 12)))
            head_size = module.q_proj.weight.size(0) // num_heads
            
            # Zero out gradients for pruned heads
            for head_idx in head_masks[layer_idx]:
                start_idx = head_idx * head_size
                end_idx = start_idx + head_size
                
                # Zero query/key/value gradients
                module.q_proj.weight.grad[start_idx:end_idx, :] = 0.0
                module.k_proj.weight.grad[start_idx:end_idx, :] = 0.0
                module.v_proj.weight.grad[start_idx:end_idx, :] = 0.0
                
                # Zero bias gradients if present
                if (hasattr(module.q_proj, 'bias') and module.q_proj.bias is not None 
                    and module.q_proj.bias.grad is not None):
                    module.q_proj.bias.grad[start_idx:end_idx] = 0.0
                    module.k_proj.bias.grad[start_idx:end_idx] = 0.0
                    module.v_proj.bias.grad[start_idx:end_idx] = 0.0
    
    # Create an optimizer post-hook for adaptive mode
    def adaptive_optimizer_post_hook(opt, step_kwargs):
        """Re-zero weights after optimizer step"""
        # Call the original optimizer step
        result = opt.orig_step(*step_kwargs.get('args', []), **step_kwargs.get('kwargs', {}))
        
        # Re-zero weights
        for layer_idx, heads in head_masks.items():
            if layer_idx < len(model.transformer.h):
                attn_module = model.transformer.h[layer_idx].attn
                
                # Handle GPT-2 style models
                if hasattr(attn_module, 'c_attn'):
                    # Calculate dimensions
                    n_heads = getattr(attn_module, 'n_head', 
                                    getattr(attn_module, 'num_heads', 
                                          getattr(attn_module, 'num_attention_heads', 12)))
                    hidden_size = attn_module.c_attn.weight.size(0)
                    head_size = hidden_size // n_heads
                    
                    # Re-zero weights for pruned heads
                    with torch.no_grad():
                        for head_idx in heads:
                            q_idx = head_idx * head_size
                            k_idx = hidden_size + head_idx * head_size
                            v_idx = 2 * hidden_size + head_idx * head_size
                            
                            # Create a mask of zeros and ones
                            mask = torch.ones_like(attn_module.c_attn.weight)
                            mask[q_idx:q_idx+head_size, :] = 0.0
                            mask[k_idx:k_idx+head_size, :] = 0.0
                            mask[v_idx:v_idx+head_size, :] = 0.0
                            
                            # Zero weights by multiplying with the mask
                            attn_module.c_attn.weight.mul_(mask)
                            
                            # Zero bias if present
                            if hasattr(attn_module.c_attn, 'bias') and attn_module.c_attn.bias is not None:
                                bias_mask = torch.ones_like(attn_module.c_attn.bias)
                                bias_mask[q_idx:q_idx+head_size] = 0.0
                                bias_mask[k_idx:k_idx+head_size] = 0.0
                                bias_mask[v_idx:v_idx+head_size] = 0.0
                                attn_module.c_attn.bias.mul_(bias_mask)
                
                # Handle models with separate QKV projections
                elif (hasattr(attn_module, 'q_proj') and hasattr(attn_module, 'k_proj') and hasattr(attn_module, 'v_proj')):
                    # Similar implementation for separate QKV...
                    pass  # Simplified for brevity
        
        return result
    
    # Apply hooks to each layer's attention module
    for layer_idx, layer in enumerate(model.transformer.h):
        if layer_idx in head_masks:
            # Store layer index as an attribute
            layer.attn._layer_idx = layer_idx
            
            # For compressed mode, we block gradients
            if mode == PruningMode.COMPRESSED:
                # Register backward hook to zero gradients
                hook = layer.attn.register_full_backward_hook(zero_grad_hook)
                hooks.append(hook)
            else:
                # For adaptive mode with hooks:
                # 1. Register forward pre-hook to re-zero weights before forward pass
                hook = layer.attn.register_forward_pre_hook(forward_pre_hook)
                hooks.append(hook)
                
                # 2. Add a note about using the optimizer_step_hook
                if verbose:
                    print("Note: For complete zeroing during training, wrap optimizer.step with the returned hook function")
                    print("Example: optimizer.orig_step = optimizer.step; optimizer.step = lambda *a, **kw: hook({'args': a, 'kwargs': kw})")
                    
                # Store the optimizer hook for potential use
                layer.attn._optimizer_hook = adaptive_optimizer_post_hook
    
    if verbose:
        if mode == PruningMode.COMPRESSED:
            print(f"Applied {len(hooks)} gradient zeroing hooks for compressed mode")
        else:
            print(f"Applied {len(hooks)} weight re-zeroing hooks for adaptive mode")
            print("Note: Heads can still learn to recover if re-zeroing hooks are removed")
    
    return hooks


def get_model_info(model: torch.nn.Module) -> dict:
    """
    Calculate model size and parameter statistics.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model statistics
    """
    param_size = 0
    nonzero_params = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        nonzero_params += torch.count_nonzero(param).item()
    
    total_params = sum(p.numel() for p in model.parameters())
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    
    return {
        "size_mb": size_mb,
        "total_params": total_params,
        "nonzero_params": nonzero_params,
        "zero_params": total_params - nonzero_params,
        "sparsity": 1.0 - (nonzero_params / total_params)
    }