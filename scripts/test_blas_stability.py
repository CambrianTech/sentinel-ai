#!/usr/bin/env python
"""
BLAS Stability Test for Apple Silicon

This script intensively tests matrix operations to verify BLAS stability
across different matrix sizes and operations.
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
import random
from utils.neural_plasticity.core import (
    safe_matmul, 
    IS_APPLE_SILICON,
    HAS_GPU
)

def test_matrix_stability(test_count=50, max_size=1000):
    """Run extensive tests of matrix operations to verify stability."""
    print("=" * 80)
    print("BLAS STABILITY TEST FOR APPLE SILICON")
    print("=" * 80)
    print(f"Environment: {'Apple Silicon' if IS_APPLE_SILICON else 'Standard hardware'}")
    print(f"GPU available: {'Yes' if HAS_GPU else 'No'}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Running {test_count} matrix multiplications with random sizes up to {max_size}x{max_size}")
    
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
    
    # Report results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Total tests: {test_count}")
    print(f"Successes: {successes} ({100 * successes / test_count:.1f}%)")
    print(f"Failures: {failures} ({100 * failures / test_count:.1f}%)")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"Average execution time: {avg_time:.4f}s")
    
    return failures == 0

if __name__ == "__main__":
    try:
        success = test_matrix_stability(test_count=50)
        exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)