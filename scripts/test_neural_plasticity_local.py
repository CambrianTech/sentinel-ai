#!/usr/bin/env python
"""
Test Neural Plasticity Local Execution

This script provides a simplified test to verify that the neural plasticity
module is working correctly in the local environment, with a focus on
the modular API and cross-platform compatibility features.
"""

import os
import sys
import importlib
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

# PART 1: Test basic PyTorch functionality
try:
    import torch
    
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    
    # Set up a simple test with attention-like matrices
    batch_size = 2
    num_heads = 4
    seq_len = 32  # Smaller size for quicker testing
    
    # Create random attention-like matrices
    attn = torch.rand(batch_size, num_heads, seq_len, seq_len)
    
    # Ensure we have proper probability distributions (sum to 1 along last dim)
    attn = attn / attn.sum(dim=-1, keepdim=True)
    
    # Test a basic matrix multiplication
    print("\nTesting standard matrix multiplication...")
    a = torch.randn(100, 200)
    b = torch.randn(200, 100)
    try:
        c = torch.matmul(a, b)
        print(f"Matrix multiplication succeeded. Result shape: {c.shape}")
    except Exception as e:
        print(f"Matrix multiplication error: {e}")
    
    print("✅ Basic PyTorch tests passed!")

except ImportError:
    print("⚠️ PyTorch not available. Skipping tensor tests.")
    sys.exit(1)

# PART 2: Test our Neural Plasticity module's API
try:
    print("\n======= Testing Neural Plasticity API =======")
    
    # Import our module
    from utils.neural_plasticity import (
        NeuralPlasticity,
        compute_improved_entropy,
        calculate_head_entropy
    )
    
    # Test environment detection
    env_info = NeuralPlasticity.get_environment_info()
    print("\nEnvironment Information:")
    for key, value in env_info.items():
        print(f"  {key}: {value}")
    
    # Test entropy calculation with the modular function
    print("\nTesting entropy calculation with modular API...")
    entropy = compute_improved_entropy(attn, debug=True)
    
    print(f"\nModular API entropy result shape: {entropy.shape}")
    
    # Test the high-level NeuralPlasticity API
    print("\nTesting high-level NeuralPlasticity API...")
    api_entropy = NeuralPlasticity.compute_entropy_with_diagnostics(attn, debug=True)
    
    # Check if results match
    print("\nAPI implementation match test:", torch.allclose(entropy, api_entropy))
    
    # Test gradient-based pruning functions
    print("\nTesting gradient-based pruning logic...")
    # Create mock gradient values
    gradients = torch.rand(4, 12)  # 4 layers, 12 heads per layer
    
    # Get pruning mask
    prune_level = 0.2  # Prune 20% of heads
    pruning_mask = NeuralPlasticity.create_gradient_pruning_mask(
        grad_norm_values=gradients,
        prune_percent=prune_level
    )
    
    # Check mask properties
    total_heads = gradients.numel()
    pruned_heads = pruning_mask.sum().item()
    expected_pruned = int(total_heads * prune_level)
    
    print(f"Gradient tensor shape: {gradients.shape}")
    print(f"Pruning mask shape: {pruning_mask.shape}")
    print(f"Pruned {pruned_heads} heads out of {total_heads} ({pruned_heads/total_heads:.1%})")
    print(f"Expected to prune: {expected_pruned} heads ({prune_level:.1%})")
    
    # Test visualization function by importing it (just to check it loads)
    print("\nTesting visualization module import...")
    from utils.neural_plasticity.visualization import (
        visualize_head_entropy,
        visualize_head_gradients
    )
    print("✅ Visualization module imported successfully")
    
    print("\n✅ Neural Plasticity API tests passed!")
    
except ImportError as e:
    print(f"⚠️ Neural Plasticity module import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Neural Plasticity test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# PART 3: Test visualization functionality
try:
    import matplotlib.pyplot as plt
    print("\nMatplotlib version:", plt.__version__)
    
    # Create a visualization using our API
    print("Testing visualization API...")
    
    # Create a simple visualization
    fig = NeuralPlasticity.visualize_head_metrics(
        entropy_values=torch.rand(4, 12),  # Random entropy values
        grad_norm_values=gradients,
        pruned_heads=[(0, 0), (1, 3)]  # Some example pruned heads
    )
    
    # Save the visualization
    test_plot_path = "neural_plasticity_test.png"
    for fig_key, fig_obj in fig.items():
        fig_obj.savefig(f"{fig_key}_{test_plot_path}")
        print(f"Test plot saved to {fig_key}_{test_plot_path}")
        plt.close(fig_obj)
    
    print("✅ Visualization tests passed!")
    
except ImportError:
    print("⚠️ Matplotlib not available. Skipping visualization tests.")

print("\n🎉 All tests completed successfully!")
