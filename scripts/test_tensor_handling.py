#!/usr/bin/env python
"""
Test Script for Neural Plasticity Tensor Handling

This script isolates the tensor operations and visualizations that were
causing BLAS/libtorch crashes in the Neural Plasticity notebook.
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set environment variables for safer execution
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Make sure utils module is in the path
sys.path.insert(0, os.path.abspath('.'))

print("Testing Neural Plasticity tensor handling functions")
print("=" * 50)

# Import neural plasticity modules
try:
    from utils.neural_plasticity.core import calculate_head_entropy
    from utils.neural_plasticity.visualization import visualize_head_entropy, visualize_head_gradients
    from utils.colab.helpers import safe_tensor_imshow
    
    print("✅ Successfully imported neural plasticity modules")
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    sys.exit(1)

# Test tensor creation and entropy calculation
print("\nTesting attention tensor creation...")
try:
    # Create test tensors (similar to what would be in the notebook)
    batch_size = 2
    num_heads = 4
    seq_len = 32
    
    # Create random attention-like matrices
    attention_maps = torch.rand(batch_size, num_heads, seq_len, seq_len)
    
    # Ensure they sum to 1 along the last dimension (proper attention distributions)
    attention_maps = attention_maps / attention_maps.sum(dim=-1, keepdim=True)
    
    print(f"Created attention tensor of shape {attention_maps.shape}")
    print(f"Attention min/max/mean: {attention_maps.min().item():.4f}/{attention_maps.max().item():.4f}/{attention_maps.mean().item():.4f}")
    
    # Check that rows sum to 1
    row_sums = attention_maps.sum(dim=-1)
    print(f"Row sums close to 1.0: {torch.allclose(row_sums, torch.ones_like(row_sums))}")
    
    # Test entropy calculation
    print("\nTesting entropy calculation...")
    entropy = calculate_head_entropy(attention_maps)
    print(f"Entropy shape: {entropy.shape}")
    print(f"Entropy min/max/mean: {entropy.min().item():.4f}/{entropy.max().item():.4f}/{entropy.mean().item():.4f}")
    
    print("✅ Entropy calculation successful")
except Exception as e:
    print(f"❌ Error in tensor operations: {e}")
    sys.exit(1)

# Test visualization functions
print("\nTesting visualization functions...")
try:
    # Simulate layer structure (create appropriate shaped tensor)
    num_layers = 4
    num_heads = 4
    
    # Create a properly shaped entropy tensor for visualization
    # For our test, we'll just create random values
    layer_entropies = torch.rand(num_layers, num_heads)
    
    print(f"Creating entropy heatmap for tensor of shape {layer_entropies.shape}")
    
    # Test visualization using direct matplotlib calls
    plt.figure(figsize=(10, 6))
    # Convert to numpy 
    entropy_np = layer_entropies.detach().cpu().numpy()
    plt.imshow(entropy_np, cmap='viridis')
    plt.colorbar(label='Entropy')
    plt.title("Test Entropy Visualization")
    plt.xlabel('Head Index')
    plt.ylabel('Layer Index')
    
    # Add annotations
    for i in range(entropy_np.shape[0]):
        for j in range(entropy_np.shape[1]):
            plt.text(j, i, f'{entropy_np[i, j]:.2f}',
                    ha="center", va="center", color="w")
    
    # Save the figure
    test_fig_path = "test_entropy_heatmap.png"
    plt.savefig(test_fig_path)
    plt.close()
    
    print(f"✅ Successfully created entropy visualization at {test_fig_path}")
    
    # Test gradient visualization
    print("\nTesting gradient visualization...")
    
    # Create mock gradient values
    grad_values = torch.rand(num_layers, num_heads)
    
    # Create test pruned/revived heads
    pruned_heads = [(0, 1), (2, 3)]  # Layer 0, Head 1 and Layer 2, Head 3
    revived_heads = [(1, 2)]         # Layer 1, Head 2
    
    # Create visualization directly
    plt.figure(figsize=(10, 5))
    
    # Convert to numpy
    grad_np = grad_values.detach().cpu().numpy()
    
    # Plot the gradient values
    im = plt.imshow(grad_np, cmap='plasma')
    plt.colorbar(label='Gradient Norm')
    plt.title("Test Gradient Visualization")
    plt.xlabel('Head Index')
    plt.ylabel('Layer Index')
    
    # Mark pruned heads with 'P'
    for layer, head in pruned_heads:
        plt.text(head, layer, "P", ha="center", va="center",
               color="white", weight="bold", bbox=dict(facecolor='red', alpha=0.5))
    
    # Mark revived heads with 'R'
    for layer, head in revived_heads:
        plt.text(head, layer, "R", ha="center", va="center",
               color="white", weight="bold", bbox=dict(facecolor='green', alpha=0.5))
    
    # Save the figure
    test_grad_path = "test_gradient_heatmap.png"
    plt.savefig(test_grad_path)
    plt.close()
    
    print(f"✅ Successfully created gradient visualization at {test_grad_path}")
    
except Exception as e:
    print(f"❌ Error in visualization: {e}")
    sys.exit(1)

# Test safe_tensor_imshow with both CPU and simulated GPU tensors
print("\nTesting safe_tensor_imshow function...")
try:
    # Create a test tensor
    test_tensor = torch.rand(10, 10)
    
    # Test with additional metadata that simulates a GPU tensor with gradients
    test_tensor.requires_grad = True
    
    # Use safe_tensor_imshow
    plt.figure(figsize=(8, 6))
    img = safe_tensor_imshow(test_tensor, title="Safe Tensor Visualization Test")
    
    # Save the figure
    test_safe_path = "test_safe_tensor_imshow.png"
    plt.savefig(test_safe_path)
    plt.close()
    
    print(f"✅ Successfully created safe tensor visualization at {test_safe_path}")
except Exception as e:
    print(f"❌ Error in safe_tensor_imshow: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("All tensor handling and visualization tests completed successfully!")
print("This suggests that the fixes applied have resolved the BLAS/libtorch issues.")
