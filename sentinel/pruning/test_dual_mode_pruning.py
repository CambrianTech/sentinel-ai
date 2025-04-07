"""
Test script for dual-mode pruning implementation.

Usage:
    python -m sentinel.pruning.test_dual_mode_pruning
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from sentinel.pruning.dual_mode_pruning import prune_head_in_model, apply_pruning_hooks, PruningMode, get_model_info


def get_head_info(model, layer_idx=0, head_idx=0):
    """
    Get information about a specific attention head.
    
    Args:
        model: The transformer model
        layer_idx: Layer index
        head_idx: Head index
        
    Returns:
        Tuple of (attn_module, n_heads, hidden_size, head_size, q_idx, k_idx, v_idx)
    """
    # Get block and attention module
    block = model.transformer.h[layer_idx]
    attn_module = block.attn
    
    # Get number of heads (try different attributes)
    n_heads = None
    if hasattr(attn_module, 'n_head'):
        n_heads = attn_module.n_head
    elif hasattr(attn_module, 'num_heads'):
        n_heads = attn_module.num_heads
    elif hasattr(attn_module, 'num_attention_heads'):
        n_heads = attn_module.num_attention_heads
    elif hasattr(model.config, 'n_head'):
        n_heads = model.config.n_head
    elif hasattr(model.config, 'num_heads'):
        n_heads = model.config.num_heads
    elif hasattr(model.config, 'num_attention_heads'):
        n_heads = model.config.num_attention_heads
    else:
        # Default for distilgpt2
        n_heads = 12
    
    # Get dimensions
    hidden_size = attn_module.c_attn.weight.size(0)
    head_size = hidden_size // n_heads
    
    # Get indices for this head's weights
    q_idx = head_idx * head_size
    k_idx = hidden_size + head_idx * head_size
    v_idx = 2 * hidden_size + head_idx * head_size
    
    return attn_module, n_heads, hidden_size, head_size, q_idx, k_idx, v_idx


def test_adaptive_pruning():
    """Test adaptive pruning mode."""
    print("\n=== Testing ADAPTIVE pruning mode ===")
    
    # Load a small model
    model_name = "distilgpt2"
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Select heads to prune
    layer_idx, head_idx = 0, 0
    
    # Get head info
    attn_module, n_heads, hidden_size, head_size, q_idx, k_idx, v_idx = get_head_info(model, layer_idx, head_idx)
    
    # Check initial weights
    initial_q_weights = attn_module.c_attn.weight[q_idx:q_idx+head_size, 0:5].clone()
    print(f"Initial Q weights non-zero: {torch.any(initial_q_weights != 0).item()}")
    
    # Prune the head in adaptive mode
    print(f"Pruning head {head_idx} in layer {layer_idx} (adaptive mode)")
    prune_head_in_model(model, layer_idx, head_idx, mode=PruningMode.ADAPTIVE)
    
    # Check if weights are zeroed
    after_q_weights = attn_module.c_attn.weight[q_idx:q_idx+head_size, 0:5].clone()
    print(f"After pruning Q weights all zero: {torch.all(after_q_weights == 0).item()}")
    
    # Simulate a gradient update to test recovery
    print("Simulating gradient update to test recovery...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create dummy input and force backward pass
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer("Test input", return_tensors="pt")
    outputs = model(**inputs, labels=inputs.input_ids)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    
    # Check if weights have changed from zero
    final_q_weights = attn_module.c_attn.weight[q_idx:q_idx+head_size, 0:5].clone()
    weights_changed = torch.any(final_q_weights != 0).item()
    print(f"Weights changed after update: {weights_changed}")
    
    if weights_changed:
        print("✅ ADAPTIVE mode allows weights to recover during training as expected")
    else:
        print("❌ ADAPTIVE mode doesn't allow recovery - check implementation")
    
    return model, layer_idx, head_idx


def test_compressed_pruning():
    """Test compressed pruning mode."""
    print("\n=== Testing COMPRESSED pruning mode ===")
    
    # Load a small model
    model_name = "distilgpt2"
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Select heads to prune
    layer_idx, head_idx = 0, 0
    
    # Get head info
    attn_module, n_heads, hidden_size, head_size, q_idx, k_idx, v_idx = get_head_info(model, layer_idx, head_idx)
    
    # Check initial weights
    initial_q_weights = attn_module.c_attn.weight[q_idx:q_idx+head_size, 0:5].clone()
    print(f"Initial Q weights non-zero: {torch.any(initial_q_weights != 0).item()}")
    
    # Prune the head in compressed mode
    print(f"Pruning head {head_idx} in layer {layer_idx} (compressed mode)")
    prune_head_in_model(model, layer_idx, head_idx, mode=PruningMode.COMPRESSED)
    
    # Check if weights are zeroed
    after_q_weights = attn_module.c_attn.weight[q_idx:q_idx+head_size, 0:5].clone()
    print(f"After pruning Q weights all zero: {torch.all(after_q_weights == 0).item()}")
    
    # Add pruning hooks
    hooks = apply_pruning_hooks(model, [(layer_idx, head_idx)], mode=PruningMode.COMPRESSED)
    
    # Simulate a gradient update to test recovery
    print("Simulating gradient update to test recovery...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create dummy input and force backward pass
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer("Test input", return_tensors="pt")
    outputs = model(**inputs, labels=inputs.input_ids)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    
    # Check if weights have changed from zero
    final_q_weights = attn_module.c_attn.weight[q_idx:q_idx+head_size, 0:5].clone()
    weights_changed = torch.any(final_q_weights != 0).item()
    print(f"Weights changed after update: {weights_changed}")
    
    if not weights_changed:
        print("✅ COMPRESSED mode keeps weights at zero during training as expected")
    else:
        print("❌ COMPRESSED mode allows recovery - check implementation")
    
    # Remove hooks to clean up
    for hook in hooks:
        hook.remove()
    
    return model, layer_idx, head_idx


def test_model_info():
    """Test the model info utility."""
    print("\n=== Testing Model Info ===")
    
    # Load a small model
    model_name = "distilgpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Get initial model info
    initial_info = get_model_info(model)
    print("Initial model info:")
    print(f"  Size: {initial_info['size_mb']:.2f} MB")
    print(f"  Parameters: {initial_info['total_params']:,}")
    print(f"  Non-zero parameters: {initial_info['nonzero_params']:,}")
    print(f"  Sparsity: {initial_info['sparsity']:.2%}")
    
    # Prune several heads
    heads_to_prune = [(0, i) for i in range(3)] + [(1, i) for i in range(3)]
    for layer_idx, head_idx in heads_to_prune:
        prune_head_in_model(model, layer_idx, head_idx, mode=PruningMode.COMPRESSED, verbose=False)
    
    # Get model info after pruning
    after_info = get_model_info(model)
    print("\nAfter pruning info:")
    print(f"  Size: {after_info['size_mb']:.2f} MB")
    print(f"  Parameters: {after_info['total_params']:,}")
    print(f"  Non-zero parameters: {after_info['nonzero_params']:,}")
    print(f"  Zero parameters: {after_info['zero_params']:,}")
    print(f"  Sparsity: {after_info['sparsity']:.2%}")
    
    print(f"\nZeroed parameters: {after_info['zero_params'] - initial_info['zero_params']:,}")
    
    return initial_info, after_info


def test_adaptive_with_hooks():
    """Test adaptive pruning mode with re-zeroing hooks."""
    print("\n=== Testing ADAPTIVE mode with re-zeroing hooks ===")
    
    # Load a small model
    model_name = "distilgpt2"
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Select heads to prune
    layer_idx, head_idx = 0, 0
    
    # Get head info
    attn_module, n_heads, hidden_size, head_size, q_idx, k_idx, v_idx = get_head_info(model, layer_idx, head_idx)
    
    # Prune the head in adaptive mode
    print(f"Pruning head {head_idx} in layer {layer_idx} (adaptive mode with hooks)")
    prune_head_in_model(model, layer_idx, head_idx, mode=PruningMode.ADAPTIVE)
    
    # Add hooks to re-zero weights before each forward pass
    hooks = apply_pruning_hooks(model, [(layer_idx, head_idx)], mode=PruningMode.ADAPTIVE)
    
    # Simulate multiple gradient updates
    print("Simulating multiple gradient updates...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Higher learning rate
    
    # Create dummy input
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer("Test input for multiple steps", return_tensors="pt")
    
    # Perform multiple training steps
    n_steps = 5
    for step in range(n_steps):
        # Forward pass triggers hook to re-zero weights
        outputs = model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Check if weights stay zeroed
        mid_q_weights = attn_module.c_attn.weight[q_idx:q_idx+head_size, 0:5].clone()
        if torch.all(mid_q_weights == 0).item():
            print(f"  Step {step+1}/{n_steps}: Weights remain zeroed ✓")
        else:
            print(f"  Step {step+1}/{n_steps}: Weights changed! ✗")
    
    # Check if weights are still zeroed after training
    final_q_weights = attn_module.c_attn.weight[q_idx:q_idx+head_size, 0:5].clone()
    weights_still_zero = torch.all(final_q_weights == 0).item()
    
    if weights_still_zero:
        print("✅ ADAPTIVE mode with hooks keeps weights zeroed during training")
    else:
        print("❌ ADAPTIVE mode with hooks failed to maintain zeroed state")
    
    # Check if removing hooks allows recovery
    print("\nRemoving hooks to test recovery capability...")
    for hook in hooks:
        hook.remove()
    
    # Do one more training step
    outputs = model(**inputs, labels=inputs.input_ids)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Check if weights recovered
    post_hook_q_weights = attn_module.c_attn.weight[q_idx:q_idx+head_size, 0:5].clone()
    weights_changed = torch.any(post_hook_q_weights != 0).item()
    
    if weights_changed:
        print("✅ After removing hooks, weights can recover")
    else:
        print("❌ Weights still can't recover after removing hooks")
    
    return model, layer_idx, head_idx


def main():
    """Run all tests."""
    try:
        adaptive_model, _, _ = test_adaptive_pruning()
    except Exception as e:
        print(f"Error in adaptive pruning test: {e}")
    
    try:
        compressed_model, _, _ = test_compressed_pruning()
    except Exception as e:
        print(f"Error in compressed pruning test: {e}")
    
    try:
        adaptive_hook_model, _, _ = test_adaptive_with_hooks()
    except Exception as e:
        print(f"Error in adaptive with hooks test: {e}")
    
    try:
        initial_info, after_info = test_model_info()
        print("\n=== Test Summary ===")
        if initial_info['zero_params'] < after_info['zero_params']:
            print("✅ Pruning successfully increases model sparsity")
        else:
            print("❌ Pruning did not increase model sparsity")
    except Exception as e:
        print(f"Error in model info test: {e}")


if __name__ == "__main__":
    main()