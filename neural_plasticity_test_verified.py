#!/usr/bin/env python
"""
Ultra-minimal Neural Plasticity test that focuses on just the tensor operations.
This script tests the core functions with minimal dependencies.
"""

import os
import sys
import time
import torch
import numpy as np
import random
import platform
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Set environment variables for improved stability
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['ACCELERATE_USE_SYSTEM_BLAS'] = '1'
os.environ['PYTORCH_JIT_USE_AUTOTUNER'] = '0'
    
# Import core functions
from utils.neural_plasticity.core import IS_APPLE_SILICON, safe_matmul, calculate_head_entropy

def test_matrix_multiplication():
    """Test matrix multiplication with various sizes"""
    print(f"\n[1/3] Testing safe matrix multiplication")
    
    sizes = [
        (10, 20, 10),     # Small
        (100, 200, 100),  # Medium
        (500, 500, 500)   # Large
    ]
    
    for i, (m, k, n) in enumerate(sizes):
        print(f"  Test {i+1}: Matrix multiplication [{m}x{k}] @ [{k}x{n}]")
        
        # Create random matrices
        a = torch.randn(m, k)
        b = torch.randn(k, n)
        
        # Perform safe matrix multiplication
        try:
            start_time = time.time()
            result = safe_matmul(a, b)
            elapsed = time.time() - start_time
            
            # Check if result has correct shape
            assert result.shape == (m, n), f"Expected shape {(m, n)}, got {result.shape}"
            print(f"  ✅ Shape correct: {result.shape}")
            print(f"  ✅ Completed in {elapsed:.4f}s")
            
            # Check if result has valid values (no NaN/Inf)
            has_nan = torch.isnan(result).any().item()
            has_inf = torch.isinf(result).any().item()
            
            assert not has_nan, "Result contains NaN values"
            assert not has_inf, "Result contains Inf values"
            print(f"  ✅ No NaN or Inf values")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            return False
    
    print(f"  ✅ All matrix multiplications successful")
    return True

def test_entropy_calculation():
    """Test entropy calculation on attention maps"""
    print(f"\n[2/3] Testing attention entropy calculation")
    
    try:
        # Create a sample attention map
        # [batch_size, num_heads, seq_len, seq_len]
        attention_maps = torch.rand(2, 8, 30, 30)
        
        print(f"  Attention maps shape: {attention_maps.shape}")
        
        # Calculate entropy
        start_time = time.time()
        entropy = calculate_head_entropy(attention_maps)
        elapsed = time.time() - start_time
        
        # Check resulting shape
        expected_shape = (8, 30)  # [heads, seq_len]
        assert entropy.shape == expected_shape, f"Expected shape {expected_shape}, got {entropy.shape}"
        
        print(f"  ✅ Entropy shape correct: {entropy.shape}")
        print(f"  ✅ Completed in {elapsed:.4f}s")
        
        # Check for valid values
        has_nan = torch.isnan(entropy).any().item()
        has_inf = torch.isinf(entropy).any().item()
        
        assert not has_nan, "Entropy contains NaN values"
        assert not has_inf, "Entropy contains Inf values"
        
        # Check entropy values are reasonable (entropy should be positive)
        min_val = entropy.min().item()
        max_val = entropy.max().item()
        
        assert min_val >= 0, f"Minimum entropy value {min_val} should be non-negative"
        assert max_val > 0, f"Maximum entropy value {max_val} should be positive"
        
        print(f"  ✅ No NaN or Inf values")
        print(f"  ✅ Entropy values in range [{min_val:.4f}, {max_val:.4f}]")
        
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

def test_large_tensor_operations():
    """Test operations on large tensors to ensure stability"""
    print(f"\n[3/3] Testing large tensor operations")
    
    try:
        # Create large tensors (1000x1000)
        print(f"  Creating large tensors (1000x1000)...")
        a = torch.randn(1000, 1000)
        b = torch.randn(1000, 1000)
        
        # Matrix multiplication
        print(f"  Performing large matrix multiplication...")
        start_time = time.time()
        result = safe_matmul(a, b)
        elapsed = time.time() - start_time
        
        assert result.shape == (1000, 1000), f"Expected shape (1000, 1000), got {result.shape}"
        print(f"  ✅ Large matrix multiplication successful in {elapsed:.4f}s")
        
        # Large attention map (simulate a large batch)
        print(f"  Creating large attention map (4, 16, 512, 512)...")
        large_attention = torch.rand(4, 16, 512, 512)
        
        # Calculate entropy on large attention map
        print(f"  Calculating entropy on large attention map...")
        start_time = time.time()
        entropy = calculate_head_entropy(large_attention)
        elapsed = time.time() - start_time
        
        assert entropy.shape == (16, 512), f"Expected shape (16, 512), got {entropy.shape}"
        print(f"  ✅ Large entropy calculation successful in {elapsed:.4f}s")
        
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

def main():
    """Run all tests"""
    # Print system information
    system = platform.system()
    processor = platform.processor()
    is_apple_silicon = system == "Darwin" and processor == "arm"
    
    print(f"=== NEURAL PLASTICITY VERIFICATION ===")
    print(f"System: {system} {processor}")
    print(f"Running on Apple Silicon: {is_apple_silicon}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    # Run tests
    matrix_test = test_matrix_multiplication()
    entropy_test = test_entropy_calculation()
    large_test = test_large_tensor_operations()
    
    # Print summary
    print("\n=== TEST SUMMARY ===")
    print(f"Matrix multiplication: {'✅ PASSED' if matrix_test else '❌ FAILED'}")
    print(f"Entropy calculation: {'✅ PASSED' if entropy_test else '❌ FAILED'}")
    print(f"Large tensor operations: {'✅ PASSED' if large_test else '❌ FAILED'}")
    
    # Final result
    if matrix_test and entropy_test and large_test:
        print("\n🎉 ALL TESTS PASSED! The tensor operations are working correctly.")
        return 0
    else:
        print("\n⚠️ SOME TESTS FAILED. Please check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())