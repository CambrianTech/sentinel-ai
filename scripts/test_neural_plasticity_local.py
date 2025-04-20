#!/usr/bin/env python
"""
Test Neural Plasticity Local Execution

This script provides a simplified test to verify that the tensor handling
and BLAS operations are working correctly in the local environment, without
requiring the entire notebook to run.
"""

import os
import sys
import importlib
import numpy as np
from pathlib import Path

# Set environment variables for safer execution
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Try to import torch and run a simple test
try:
    import torch
    
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    
    # Set up a simple test with attention-like matrices
    batch_size = 2
    num_heads = 4
    seq_len = 128
    
    # Create random attention-like matrices
    attn = torch.rand(batch_size, num_heads, seq_len, seq_len)
    
    # Ensure we have proper probability distributions (sum to 1 along last dim)
    attn = attn / attn.sum(dim=-1, keepdim=True)
    
    # Test entropy calculation (similar to what's in core.py)
    print("\nTesting entropy calculation...")
    eps = 1e-6
    
    # Handle potential NaN or Inf values first
    attn_probs = torch.where(
        torch.isfinite(attn),
        attn,
        torch.ones_like(attn) * eps
    )
    
    # Add small epsilon and renormalize
    attn_probs = attn_probs.clamp(min=eps)
    attn_probs = attn_probs / attn_probs.sum(dim=-1, keepdim=True)
    
    # Cast to float32 for better numerical stability
    if attn_probs.dtype != torch.float32:
        attn_probs = attn_probs.to(torch.float32)
    
    # Calculate entropy: -sum(p * log(p))
    entropy = -torch.sum(attn_probs * torch.log(attn_probs), dim=-1)
    
    # Get tensor stats
    print(f"Attention shape: {attn.shape}")
    print(f"Attention min/max/mean: {attn.min().item():.6f}/{attn.max().item():.6f}/{attn.mean().item():.6f}")
    print(f"Row sums close to 1.0: {torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)))}")
    
    print(f"Entropy shape: {entropy.shape}")
    print(f"Entropy min/max/mean: {entropy.min().item():.6f}/{entropy.max().item():.6f}/{entropy.mean().item():.6f}")
    
    # Test a basic matrix multiplication (this often triggers BLAS issues)
    print("\nTesting matrix multiplication...")
    a = torch.randn(300, 400)
    b = torch.randn(400, 500)
    try:
        c = torch.matmul(a, b)
        print(f"Matrix multiplication succeeded. Result shape: {c.shape}")
    except Exception as e:
        print(f"Matrix multiplication error: {e}")
    
    print("\nBasic PyTorch tests completed successfully!")

except ImportError:
    print("PyTorch not available. Skipping tensor tests.")

# Try matplotlib
try:
    import matplotlib.pyplot as plt
    print("\nMatplotlib available:", plt.__version__)
    
    # Test creating a basic plot
    plt.figure(figsize=(6, 4))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y)
    plt.title("Test Plot")
    
    # Save plot to verify it works
    test_plot_path = "test_plot.png"
    plt.savefig(test_plot_path)
    print(f"Test plot saved to {test_plot_path}")
    plt.close()
    
except ImportError:
    print("Matplotlib not available. Skipping visualization tests.")

print("\nAll tests completed.")
