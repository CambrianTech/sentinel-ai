#!/usr/bin/env python
"""
Minimal Neural Plasticity Test
"""

import os
import sys
import torch
import numpy as np
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

def test_basic_tensor_operations():
    print("\n====== TESTING BASIC TENSOR OPERATIONS ======")
    print(f"Is Apple Silicon: {IS_APPLE_SILICON}")

    # Test 1: Small matrix multiplication
    a = torch.randn(10, 20)
    b = torch.randn(20, 10)
    print("\nTest 1: Small matrix multiplication")
    print(f"Matrix A shape: {a.shape}")
    print(f"Matrix B shape: {b.shape}")
    result = safe_matmul(a, b)
    print(f"Result shape: {result.shape}")
    print("Test 1 successful ✅")

    # Test 2: Entropy calculation on attention maps
    print("\nTest 2: Entropy calculation")
    attention = torch.rand(2, 8, 30, 30)  # [batch, heads, seq_len, seq_len]
    print(f"Attention tensor shape: {attention.shape}")
    entropy = calculate_head_entropy(attention)
    print(f"Entropy shape: {entropy.shape}")
    print(f"Entropy values: {entropy}")
    print("Test 2 successful ✅")

    return True

if __name__ == "__main__":
    print("====== NEURAL PLASTICITY MINIMAL TEST ======")
    test_basic_tensor_operations()
    print("\nAll tests completed successfully! ✅")
