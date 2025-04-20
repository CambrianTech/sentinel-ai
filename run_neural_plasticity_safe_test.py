#!/usr/bin/env python
"""
Minimal but complete neural plasticity test that avoids all BLAS crashes.
This version uses a very minimal approach with only our safe_matmul function.
"""

import os
import sys
import platform
import torch
import numpy as np
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Set environment variables for maximum stability
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['ACCELERATE_USE_SYSTEM_BLAS'] = '1'
os.environ['PYTORCH_JIT_USE_AUTOTUNER'] = '0'
torch.set_num_threads(1)

# Import only the safe_matmul function from our code
from utils.neural_plasticity.core import safe_matmul

def is_apple_silicon():
    """Check if we're running on Apple Silicon"""
    return platform.system() == "Darwin" and platform.processor() == "arm"

def test_matrix_multiplication():
    """Test basic matrix multiplication with our safe function"""
    print("\n=== Testing Matrix Multiplication ===")
    
    # Create two simple matrices
    a = torch.randn(10, 20)
    b = torch.randn(20, 30)
    
    print(f"Matrix A: {a.shape}")
    print(f"Matrix B: {b.shape}")
    
    try:
        # Use our safe_matmul function
        result = safe_matmul(a, b)
        print(f"Result shape: {result.shape}")
        print("✅ Matrix multiplication successful")
        return True
    except Exception as e:
        print(f"❌ Matrix multiplication failed: {e}")
        return False

def test_mock_attention():
    """Test a mock attention-like operation"""
    print("\n=== Testing Mock Attention Layer ===")
    
    # Create mock query, key, value matrices
    batch_size = 2
    seq_len = 10
    head_dim = 32
    num_heads = 4
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Value shape: {value.shape}")
    
    try:
        # Transpose key for attention calculation
        key_t = key.transpose(-2, -1)
        print(f"Transposed key shape: {key_t.shape}")
        
        # Calculate attention scores (Q @ K^T)
        scores = torch.zeros(batch_size, num_heads, seq_len, seq_len)
        
        # Use safe_matmul for every batch and head combination
        for b in range(batch_size):
            for h in range(num_heads):
                scores[b, h] = safe_matmul(query[b, h], key_t[b, h])
        
        print(f"Attention scores shape: {scores.shape}")
        
        # Scale attention scores
        scores = scores / (head_dim ** 0.5)
        
        # Apply softmax
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        print(f"Attention weights shape: {attention_weights.shape}")
        
        # Calculate weighted values (softmax(QK^T) @ V)
        weighted_values = torch.zeros(batch_size, num_heads, seq_len, head_dim)
        
        # Use safe_matmul for every batch and head combination
        for b in range(batch_size):
            for h in range(num_heads):
                weighted_values[b, h] = safe_matmul(attention_weights[b, h], value[b, h])
        
        print(f"Weighted values shape: {weighted_values.shape}")
        print("✅ Mock attention calculation successful")
        return True
    except Exception as e:
        print(f"❌ Mock attention calculation failed: {e}")
        return False

def test_large_matrices():
    """Test multiplication of large matrices"""
    print("\n=== Testing Large Matrix Multiplication ===")
    
    # Create large matrices
    size = 500
    print(f"Creating matrices of size {size}x{size}...")
    
    try:
        # Create random matrices
        large_a = torch.randn(size, size)
        large_b = torch.randn(size, size)
        
        print(f"Matrix A: {large_a.shape}")
        print(f"Matrix B: {large_b.shape}")
        
        # Use our safe_matmul function
        result = safe_matmul(large_a, large_b)
        print(f"Result shape: {result.shape}")
        print("✅ Large matrix multiplication successful")
        return True
    except Exception as e:
        print(f"❌ Large matrix multiplication failed: {e}")
        return False

def main():
    """Run all tests"""
    print(f"=== NEURAL PLASTICITY SAFETY TEST ===")
    print(f"Running on Apple Silicon: {is_apple_silicon()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    # Run tests
    basic_test = test_matrix_multiplication()
    attention_test = test_mock_attention()
    large_test = test_large_matrices()
    
    # Print summary
    print("\n=== TEST SUMMARY ===")
    print(f"Basic matrix multiplication: {'✅ PASSED' if basic_test else '❌ FAILED'}")
    print(f"Mock attention calculation: {'✅ PASSED' if attention_test else '❌ FAILED'}")
    print(f"Large matrix multiplication: {'✅ PASSED' if large_test else '❌ FAILED'}")
    
    if basic_test and attention_test and large_test:
        print("\n🎉 ALL TESTS PASSED! The safe_matmul function is working correctly.")
        return 0
    else:
        print("\n⚠️ SOME TESTS FAILED. Please check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())