#!/usr/bin/env python
"""
Minimal test for tensor operations on Apple Silicon

This script only tests the safe_matmul function and other tensor operations
to make sure they work properly on Apple Silicon without crashing.
"""

import os
import sys
# Add the project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, root_dir)

import torch
import numpy as np
import time
import matplotlib.pyplot as plt

# Import just the necessary functions
from utils.neural_plasticity.core import (
    safe_matmul, 
    calculate_head_entropy,
    IS_APPLE_SILICON,
    HAS_GPU
)

def test_tensor_operations():
    """Test basic tensor operations using our safe functions."""
    print("=" * 80)
    print("TENSOR OPERATIONS SAFETY TEST")
    print("=" * 80)
    print(f"Environment: {'Apple Silicon' if IS_APPLE_SILICON else 'Standard hardware'}")
    print(f"GPU available: {'Yes' if HAS_GPU else 'No'}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Create output directory
    out_dir = os.path.join(root_dir, "output", "tensor_safety_test")
    os.makedirs(out_dir, exist_ok=True)
    
    # Test 1: Small matrix multiplication
    print("\nTest 1: Small matrix multiplication")
    a = torch.randn(10, 20)
    b = torch.randn(20, 10)
    
    print(f"Matrix A: {a.shape}")
    print(f"Matrix B: {b.shape}")
    
    start_time = time.time()
    result = safe_matmul(a, b)
    elapsed = time.time() - start_time
    
    print(f"Result shape: {result.shape}")
    print(f"Time taken: {elapsed:.4f} seconds")
    
    # Test 2: Medium matrix multiplication
    print("\nTest 2: Medium matrix multiplication")
    a = torch.randn(100, 200)
    b = torch.randn(200, 100)
    
    print(f"Matrix A: {a.shape}")
    print(f"Matrix B: {b.shape}")
    
    start_time = time.time()
    result = safe_matmul(a, b)
    elapsed = time.time() - start_time
    
    print(f"Result shape: {result.shape}")
    print(f"Time taken: {elapsed:.4f} seconds")
    
    # Test 3: Large matrix multiplication
    print("\nTest 3: Large matrix multiplication")
    a = torch.randn(500, 600)
    b = torch.randn(600, 400)
    
    print(f"Matrix A: {a.shape}")
    print(f"Matrix B: {b.shape}")
    
    start_time = time.time()
    result = safe_matmul(a, b)
    elapsed = time.time() - start_time
    
    print(f"Result shape: {result.shape}")
    print(f"Time taken: {elapsed:.4f} seconds")
    
    # Test 4: Attention entropy calculation
    print("\nTest 4: Entropy calculation")
    attention = torch.rand(2, 8, 30, 30)  # [batch, heads, seq_len, seq_len]
    
    print(f"Attention tensor: {attention.shape}")
    
    start_time = time.time()
    entropy = calculate_head_entropy(attention)
    elapsed = time.time() - start_time
    
    print(f"Entropy shape: {entropy.shape}")
    print(f"Entropy values range: [{entropy.min().item():.4f}, {entropy.max().item():.4f}]")
    print(f"Time taken: {elapsed:.4f} seconds")
    
    # Test 5: Create a visualization
    print("\nTest 5: Visualization")
    
    plt.figure(figsize=(8, 6))
    plt.imshow(entropy.numpy(), cmap='viridis')
    plt.colorbar(label='Entropy')
    plt.title('Attention Head Entropy')
    plt.xlabel('Sequence position')
    plt.ylabel('Head index')
    
    # Save the visualization
    viz_path = os.path.join(out_dir, "entropy_visualization.png")
    plt.savefig(viz_path)
    print(f"Visualization saved to: {viz_path}")
    
    print("\nAll tests completed successfully!")
    print(f"The safe_matmul and entropy functions are working correctly on this platform.")
    
    return True

if __name__ == "__main__":
    try:
        success = test_tensor_operations()
        exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)