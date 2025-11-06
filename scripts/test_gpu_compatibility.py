#!/usr/bin/env python
"""
Neural Plasticity End-to-End Test Script
This script tests the neural plasticity functionality on the current environment.
"""

import os
import sys
# Add the project root to path to ensure imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, root_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# Import the neural plasticity modules
from utils.neural_plasticity.core import (
    safe_matmul, 
    calculate_head_entropy,
    generate_pruning_mask,
    IS_APPLE_SILICON,
    HAS_GPU
)
from utils.neural_plasticity.visualization import (
    visualize_head_entropy,
    visualize_pruning_decisions
)
from utils.colab.helpers import (
    optimize_for_colab,
    safe_tensor_imshow
)

def run_end_to_end_test():
    """Run end-to-end test of neural plasticity functionality."""
    print(f"Starting neural plasticity end-to-end test")
    print(f"Environment: {'Apple Silicon' if IS_APPLE_SILICON else 'Standard'}, GPU: {'Yes' if HAS_GPU else 'No'}")
    
    # 1. Get optimized parameters for this environment
    print("\n1. Getting optimized parameters...")
    params = optimize_for_colab(model_size='small', verbose=True)
    
    # 2. Create test tensors
    print("\n2. Creating test tensors...")
    batch_size = 2
    num_heads = 8
    seq_len = 20
    
    # Create random attention maps (simulating transformer attention)
    attention_maps = torch.rand(batch_size, num_heads, seq_len, seq_len)
    print(f"Attention maps shape: {attention_maps.shape}, Device: {attention_maps.device}")
    
    # 3. Test safe_matmul function
    print("\n3. Testing safe_matmul...")
    a = torch.randn(5, 10)
    b = torch.randn(10, 5)
    
    start = time.time()
    result = safe_matmul(a, b)
    elapsed = time.time() - start
    
    print(f"Matrix multiplication result shape: {result.shape}")
    print(f"Time taken: {elapsed:.4f} seconds")
    
    # 4. Compute head entropy
    print("\n4. Computing head entropy...")
    start = time.time()
    entropy = calculate_head_entropy(attention_maps)
    elapsed = time.time() - start
    
    print(f"Entropy shape: {entropy.shape}")
    print(f"Time taken: {elapsed:.4f} seconds")
    print(f"Entropy values: Min={entropy.min().item():.4f}, Max={entropy.max().item():.4f}")
    
    # 5. Generate pruning mask
    print("\n5. Generating pruning mask...")
    # Create simulated gradient norms (shape should match entropy)
    grad_norms = torch.rand_like(entropy)
    
    start = time.time()
    mask = generate_pruning_mask(
        grad_norm_values=grad_norms,
        prune_percent=0.3,
        strategy="entropy",
        entropy_values=entropy
    )
    elapsed = time.time() - start
    
    print(f"Pruning mask shape: {mask.shape}")
    print(f"Number of heads to prune: {mask.sum().item()} out of {mask.numel()}")
    print(f"Time taken: {elapsed:.4f} seconds")
    
    # 6. Visualize entropy
    print("\n6. Visualizing entropy...")
    output_dir = os.path.join(root_dir, "output/neural_plasticity_test")
    os.makedirs(output_dir, exist_ok=True)
    
    start = time.time()
    fig1 = visualize_head_entropy(
        entropy_values=entropy,
        title="Attention Entropy Heatmap",
        save_path=os.path.join(output_dir, "entropy.png")
    )
    elapsed = time.time() - start
    
    print(f"Entropy visualization created")
    print(f"Time taken: {elapsed:.4f} seconds")
    
    # 7. Visualize pruning decisions
    print("\n7. Visualizing pruning decisions...")
    
    start = time.time()
    fig2 = visualize_pruning_decisions(
        grad_norm_values=grad_norms,
        pruning_mask=mask,
        title="Pruning Decisions Visualization",
        save_path=os.path.join(output_dir, "pruning.png")
    )
    elapsed = time.time() - start
    
    print(f"Pruning decisions visualization created")
    print(f"Time taken: {elapsed:.4f} seconds")
    
    print("\nNeural plasticity end-to-end test completed successfully!")
    print(f"Visualizations saved to {output_dir}")
    
    return True

if __name__ == "__main__":
    try:
        success = run_end_to_end_test()
        exit(0 if success else 1)
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()
        exit(1)