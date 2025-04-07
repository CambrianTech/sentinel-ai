"""
Test script for dual-mode pruning implementation.

Usage:
    python -m sentinel.pruning.test_dual_mode_pruning
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from dual_mode_pruning import prune_head_in_model, apply_pruning_hooks, PruningMode, get_model_info


def test_adaptive_pruning():
    """Test adaptive pruning mode."""
    print("\n=== Testing ADAPTIVE pruning mode ===")
    
    # Load a small model
    model_name = "distilgpt2"
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Select heads to prune
    layer_idx, head_idx = 0, 0
    
    # Get initial weights for reference
    attn_module = model.transformer.h[layer_idx].attn
    n_heads = attn_module.n_head
    hidden_size = attn_module.c_attn.weight.size(0)
    head_size = hidden_size // n_heads
    
    q_idx = head_idx * head_size
    k_idx = hidden_size + head_idx * head_size
    v_idx = 2 * hidden_size + head_idx * head_size
    
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
    
    # Get initial weights for reference
    attn_module = model.transformer.h[layer_idx].attn
    n_heads = attn_module.n_head
    hidden_size = attn_module.c_attn.weight.size(0)
    head_size = hidden_size // n_heads
    
    q_idx = head_idx * head_size
    k_idx = hidden_size + head_idx * head_size
    v_idx = 2 * hidden_size + head_idx * head_size
    
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


def main():
    """Run all tests."""
    adaptive_model, _, _ = test_adaptive_pruning()
    compressed_model, _, _ = test_compressed_pruning()
    initial_info, after_info = test_model_info()
    
    print("\n=== Test Summary ===")
    if initial_info['zero_params'] < after_info['zero_params']:
        print("✅ Pruning successfully increases model sparsity")
    else:
        print("❌ Pruning did not increase model sparsity")


if __name__ == "__main__":
    main()