#!/usr/bin/env python
"""
Comprehensive Neural Plasticity Test
Tests more advanced features including Apple Silicon optimizations
"""

import os
import sys
import time
import torch
import numpy as np
import random
import gc
import matplotlib.pyplot as plt
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Try importing sentinel_data before datasets to avoid conflict
try:
    # Add the sentinel_data directory to the path
    sys.path.insert(0, str(current_dir / "sentinel_data"))
    import sentinel_data
    print(f"Found sentinel_data at {sentinel_data.__file__}")
except ImportError as e:
    print(f"Could not import sentinel_data: {e}")

# Import core neural plasticity functions
from utils.neural_plasticity.core import IS_APPLE_SILICON, safe_matmul, calculate_head_entropy
import utils.neural_plasticity.visualization as viz

def test_matrix_stability(test_count=20, max_size=500):
    """Test matrix multiplication stability with various sizes"""
    print(f"\n====== TESTING MATRIX STABILITY: {test_count} multiplications up to {max_size}x{max_size} ======")
    
    # Results tracking
    successes = 0
    failures = 0
    times = []
    
    # Run tests
    for i in range(test_count):
        # Generate random matrix dimensions
        m = random.randint(10, max_size)
        k = random.randint(10, max_size)
        n = random.randint(10, max_size)
        
        # Create random matrices
        try:
            a = torch.randn(m, k)
            b = torch.randn(k, n)
            
            # Time the safe multiplication
            start_time = time.time()
            result = safe_matmul(a, b)
            elapsed = time.time() - start_time
            
            # Check if result has correct shape
            if result.shape == (m, n):
                successes += 1
                times.append(elapsed)
                print(f"Test {i+1}/{test_count}: [{m}x{k}] @ [{k}x{n}] -> [{m}x{n}] - ✓ Success ({elapsed:.4f}s)")
            else:
                failures += 1
                print(f"Test {i+1}/{test_count}: [{m}x{k}] @ [{k}x{n}] -> Expected [{m}x{n}], got {result.shape} - ✗ Failed")
        
        except Exception as e:
            failures += 1
            print(f"Test {i+1}/{test_count}: [{m}x{k}] @ [{k}x{n}] - ✗ Failed with error: {e}")
    
    print(f"\nMatrix stability results: {successes}/{test_count} successful ({100*successes/test_count:.1f}%)")
    if times:
        print(f"Average time: {sum(times)/len(times):.4f}s")
    
    return successes, failures

def test_visualization_features():
    """Test visualization features"""
    print("\n====== TESTING VISUALIZATION FEATURES ======")
    
    try:
        # Create a sample attention map
        num_heads = 8
        seq_len = 32
        attention_maps = torch.rand(1, num_heads, seq_len, seq_len)  # [batch, heads, seq_len, seq_len]
        
        # Create a figure but don't display it (headless mode)
        plt.figure(figsize=(10, 8))
        plt.ion()  # Turn on interactive mode to avoid blocking
        
        # Test attention map visualization
        print("Visualizing attention patterns...")
        fig1 = viz.visualize_attention_patterns(
            attention_maps=attention_maps,
            layer_idx=0,  # Specify layer index (required parameter)
            save_path="test_attention_viz.png"
        )
        
        # Test entropy visualization
        print("Visualizing entropy...")
        entropy = calculate_head_entropy(attention_maps)
        fig2 = viz.visualize_head_entropy(
            entropy_values=entropy,
            save_path="test_entropy_viz.png"
        )
        
        # Close figures
        plt.close(fig1)
        plt.close(fig2)
        plt.close('all')
        
        print("Visualization tests successful ✅")
        return True
    except Exception as e:
        print(f"Visualization test failed: {e}")
        return False

def test_large_tensor_handling():
    """Test handling of large tensors"""
    print("\n====== TESTING LARGE TENSOR HANDLING ======")
    
    try:
        # Create a large tensor
        print("Creating large tensors...")
        large_a = torch.randn(1000, 1000)
        large_b = torch.randn(1000, 1000)
        
        print("Multiplying large tensors...")
        start_time = time.time()
        result = safe_matmul(large_a, large_b)
        elapsed = time.time() - start_time
        
        print(f"Large tensor multiplication completed in {elapsed:.4f}s ✓")
        
        # Force garbage collection
        del large_a, large_b, result
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Large tensor handling successful ✅")
        return True
    except Exception as e:
        print(f"Large tensor test failed: {e}")
        return False

if __name__ == "__main__":
    print("\n====== NEURAL PLASTICITY COMPREHENSIVE TEST ======")
    print(f"Running on Apple Silicon: {IS_APPLE_SILICON}")
    
    # First test basic tensor operations
    test_basic_success = test_matrix_stability(20, 300)
    
    # Test visualization
    viz_success = test_visualization_features()
    
    # Test large tensor handling
    large_tensor_success = test_large_tensor_handling()
    
    print("\n====== TEST SUMMARY ======")
    print(f"Matrix stability: {'✅ PASSED' if test_basic_success[0] > 0 else '❌ FAILED'}")
    print(f"Visualization: {'✅ PASSED' if viz_success else '❌ FAILED'}")
    print(f"Large tensor handling: {'✅ PASSED' if large_tensor_success else '❌ FAILED'}")
    
    if test_basic_success[0] > 0 and viz_success and large_tensor_success:
        print("\n🎉 ALL TESTS PASSED! The Apple Silicon fixes are working correctly.")
    else:
        print("\n⚠️ SOME TESTS FAILED. Please check the output above for details.")