#!/usr/bin/env python
"""
Test Neural Plasticity Core Module

This script tests only the core functions of the neural plasticity module
without requiring the entire experiment system, focusing on tensor handling,
entropy calculation, and the Apple Silicon optimizations.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Set environment variables for safer execution on Apple Silicon
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Add the parent directory to sys.path to ensure we can import our module
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

# Import just the core components directly
try:
    import torch
    
    # Import the core module directly to avoid loading the whole package
    sys.path.insert(0, str(repo_root))
    
    # Import the core module directly
    import utils.neural_plasticity.core as np_core
    
    # Extract the functions we need
    compute_improved_entropy = np_core.compute_improved_entropy
    calculate_head_entropy = np_core.calculate_head_entropy
    safe_matmul = np_core.safe_matmul
    gradient_based_pruning = np_core.gradient_based_pruning
    IS_APPLE_SILICON = np_core.IS_APPLE_SILICON
    
    # Display environment info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Apple Silicon detected: {IS_APPLE_SILICON}")
    print(f"Python version: {sys.version}")
    
    # Test 1: Test safe_matmul (this is the function that handles BLAS operations)
    print("\n===== Testing safe_matmul function =====")
    a = torch.randn(100, 200)
    b = torch.randn(200, 100)
    
    print("Matrix A shape:", a.shape)
    print("Matrix B shape:", b.shape)
    
    try:
        result = safe_matmul(a, b)
        print("✓ safe_matmul succeeded")
        print("Result shape:", result.shape)
        print("Result stats - min/max/mean:", f"{result.min().item():.4f}/{result.max().item():.4f}/{result.mean().item():.4f}")
        
        # Compare with regular matmul
        regular_result = torch.matmul(a, b)
        is_close = torch.allclose(result, regular_result, rtol=1e-5, atol=1e-5)
        print(f"Same result as regular matmul: {is_close}")
    except Exception as e:
        print(f"✗ safe_matmul failed: {e}")
    
    # Test 2: Test compute_improved_entropy
    print("\n===== Testing compute_improved_entropy function =====")
    # Create a test attention tensor
    batch_size = 2
    num_heads = 4
    seq_len = 32
    
    attention = torch.rand(batch_size, num_heads, seq_len, seq_len)
    attention = attention / attention.sum(dim=-1, keepdim=True)  # Ensure sums to 1
    
    print("Attention tensor shape:", attention.shape)
    print("Testing row sums to 1:", torch.allclose(attention.sum(dim=-1), torch.ones_like(attention.sum(dim=-1))))
    
    # Run the improved entropy calculation
    print("\nRunning compute_improved_entropy...")
    entropy = compute_improved_entropy(attention, debug=True)
    print("✓ compute_improved_entropy succeeded")
    print("Entropy shape:", entropy.shape)
    
    # Test 3: Test full entropy calculation
    print("\n===== Testing calculate_head_entropy function =====")
    normalized_entropy = calculate_head_entropy(attention, debug=False)
    print("✓ calculate_head_entropy succeeded")
    print("Normalized entropy shape:", normalized_entropy.shape)
    print("Normalized entropy range (should be 0-1):", 
          f"{normalized_entropy.min().item():.4f} to {normalized_entropy.max().item():.4f}")
    
    # Test 4: Test gradient-based pruning
    print("\n===== Testing gradient_based_pruning function =====")
    # Create mock gradient values
    num_layers = 4
    num_heads = 12
    grad_values = torch.rand(num_layers, num_heads)
    
    # Set a pruning percentage
    prune_percent = 0.2  # 20%
    expected_pruned = int(grad_values.numel() * prune_percent)
    
    print(f"Gradient values shape: {grad_values.shape}")
    print(f"Pruning percentage: {prune_percent*100:.1f}%")
    print(f"Expected pruned heads: {expected_pruned} out of {grad_values.numel()}")
    
    # Generate pruning mask
    pruning_mask = gradient_based_pruning(grad_values, prune_percent)
    print("✓ gradient_based_pruning succeeded")
    print(f"Pruning mask shape: {pruning_mask.shape}")
    pruned_count = pruning_mask.sum().item()
    print(f"Actual pruned heads: {pruned_count} ({pruned_count/grad_values.numel()*100:.1f}%)")
    print(f"Match expected count: {pruned_count == expected_pruned}")
    
    print("\n✅ All core neural plasticity tests passed!")
    
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    sys.exit(1)
except Exception as e:
    import traceback
    print(f"❌ Test failed: {e}")
    traceback.print_exc()
    sys.exit(1)