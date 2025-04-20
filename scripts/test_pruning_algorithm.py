#!/usr/bin/env python
"""
Test Apple Silicon Pruning Algorithm Fix

This script tests the pruning algorithm for tensor shape compatibility
after our fix for the index out of bounds error.

Version: v0.0.55 (2025-04-19 22:30:00)
"""

import torch
import numpy as np
import sys
import os
import platform

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.neural_plasticity.core import (
    calculate_head_entropy,
    generate_pruning_mask,
    IS_APPLE_SILICON
)

def test_pruning_algorithm():
    """Test the fixed pruning algorithm with different tensor shapes."""
    print("Testing pruning algorithm...")
    
    # Create gradient norms tensor [layers, heads]
    grad_norms = torch.rand(6, 12)  # 6 layers, 12 heads
    
    # Create matching entropy values tensor
    entropy_values = torch.rand(6, 12)
    
    # Generate pruning mask with gradient strategy
    mask_gradient = generate_pruning_mask(
        grad_norm_values=grad_norms,
        prune_percent=0.2,
        strategy="gradient"
    )
    print(f"Gradient mask shape: {mask_gradient.shape}")
    print(f"Pruned heads (gradient): {mask_gradient.sum().item()}/{grad_norms.numel()}")
    
    # Generate pruning mask with entropy strategy
    mask_entropy = generate_pruning_mask(
        grad_norm_values=grad_norms,
        entropy_values=entropy_values,
        prune_percent=0.3,
        strategy="entropy"
    )
    print(f"Entropy mask shape: {mask_entropy.shape}")
    print(f"Pruned heads (entropy): {mask_entropy.sum().item()}/{grad_norms.numel()}")
    
    # Generate pruning mask with combined strategy
    mask_combined = generate_pruning_mask(
        grad_norm_values=grad_norms,
        entropy_values=entropy_values,
        prune_percent=0.15,
        strategy="combined"
    )
    print(f"Combined mask shape: {mask_combined.shape}")
    print(f"Pruned heads (combined): {mask_combined.sum().item()}/{grad_norms.numel()}")
    
    # Test when shapes don't match
    # Create intentionally mismatched shapes
    mismatched_entropy = torch.rand(4, 16)  # Different shape
    
    try:
        mask_mismatched = generate_pruning_mask(
            grad_norm_values=grad_norms,
            entropy_values=mismatched_entropy,
            prune_percent=0.2,
            strategy="entropy"
        )
        print("❌ Mismatched shape test failed! Should have raised an error.")
        return False
    except ValueError as e:
        print(f"✅ Correctly caught mismatched shapes: {e}")
    
    print("Testing extreme pruning percentages...")
    
    # Generate pruning mask with a very high pruning percentage
    mask_high = generate_pruning_mask(
        grad_norm_values=grad_norms,
        entropy_values=entropy_values,
        prune_percent=0.9,  # 90% pruning
        strategy="entropy"
    )
    print(f"High percentage mask shape: {mask_high.shape}")
    pruned_count = mask_high.sum().item()
    total_heads = grad_norms.numel()
    print(f"Pruned {pruned_count}/{total_heads} heads ({pruned_count/total_heads:.2%})")
    
    # Test with 100% pruning
    mask_full = generate_pruning_mask(
        grad_norm_values=grad_norms,
        entropy_values=entropy_values,
        prune_percent=1.0,  # 100% pruning
        strategy="entropy"
    )
    full_pruned = mask_full.sum().item()
    print(f"Full pruning: {full_pruned}/{total_heads} heads ({full_pruned/total_heads:.2%})")
    
    # Test with >100% pruning (should be capped at 100%)
    mask_over = generate_pruning_mask(
        grad_norm_values=grad_norms,
        entropy_values=entropy_values,
        prune_percent=1.5,  # 150% pruning (impossible, should be capped)
        strategy="entropy"
    )
    over_pruned = mask_over.sum().item()
    print(f"Over pruning: {over_pruned}/{total_heads} heads ({over_pruned/total_heads:.2%})")
    
    # Test the index out of bounds error case we fixed
    try:
        # This case previously caused index out of bounds (entropy tensor had more elements)
        # Now it should raise a ValueError about mismatched shapes
        oversized_entropy = torch.rand(12, 24)  # Twice the size of grad_norms
        mask_overflow = generate_pruning_mask(
            grad_norm_values=grad_norms,
            entropy_values=oversized_entropy,
            prune_percent=0.5,
            strategy="entropy"
        )
        print("❌ Oversized entropy test failed! Should have raised an error.")
        return False
    except ValueError as e:
        print(f"✅ Correctly caught oversized entropy: {e}")
    
    print("✅ All pruning algorithm tests passed!")
    return True

def main():
    """Run all tests to verify the pruning algorithm fix."""
    print(f"Running pruning algorithm tests on {platform.system()} {platform.processor()}")
    print(f"Apple Silicon detected: {IS_APPLE_SILICON}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print("=" * 50)
    
    success = test_pruning_algorithm()
    
    if success:
        print("\n🎉 All pruning algorithm tests passed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed. See errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())