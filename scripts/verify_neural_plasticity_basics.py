#!/usr/bin/env python
"""
Verify Neural Plasticity Basics

This script tests the most basic components of the neural plasticity module
to ensure they import correctly and can process tensors without errors.

Usage:
  python scripts/verify_neural_plasticity_basics.py
"""

import os
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt

# Add repository root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

def verify_neural_plasticity_basics():
    """Test basic functionality of neural plasticity modules."""
    
    # Create output directory
    output_dir = os.path.join(repo_root, "test_output", "minimal_test")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Verifying neural plasticity basics...")
    
    # Import the modules
    try:
        from utils.neural_plasticity.core import (
            calculate_head_entropy,
            generate_pruning_mask
        )
        print("✅ Core module imported successfully")
        
        # Create test data
        batch_size, num_heads, seq_len = 2, 4, 16
        
        # Create fake attention tensor
        attention = torch.rand(batch_size, num_heads, seq_len, seq_len)
        attention = attention / attention.sum(dim=-1, keepdim=True)
        
        # Test entropy calculation
        entropy = calculate_head_entropy(attention)
        print(f"✅ Entropy calculation successful with shape {entropy.shape}")
        
        # Create random gradient norms matrix (layers x heads)
        num_layers = 6  # e.g., for distilgpt2
        num_heads = 12
        grad_norms = torch.rand(num_layers, num_heads)
        
        # Test pruning mask generation
        pruning_mask = generate_pruning_mask(grad_norms, prune_percent=0.1)
        pruned_count = pruning_mask.sum().item()
        total_count = pruning_mask.numel()
        print(f"✅ Pruning mask generated with {pruned_count}/{total_count} heads pruned")
        
        # Test visualizing the pruning mask
        plt.figure(figsize=(10, 6))
        plt.imshow(pruning_mask.numpy(), cmap='Reds')
        plt.colorbar(label='Prune')
        plt.title('Pruning Mask')
        plt.xlabel('Head')
        plt.ylabel('Layer')
        plt.savefig(os.path.join(output_dir, "pruning_mask.png"))
        plt.close()
        print(f"✅ Pruning mask visualization saved to {output_dir}/pruning_mask.png")
        
        return True
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_neural_plasticity_basics()
    if success:
        print("\nBasic verification completed successfully")
    else:
        print("\nVerification failed")
    sys.exit(0 if success else 1)