#!/usr/bin/env python
"""
Create Minimal Test for Neural Plasticity Tensor Handling

This script creates a minimal test script that isolates the tensor handling
and visualization code that was causing crashes in the notebook.
It avoids dataset loading and model training, focusing only on the
problematic parts.
"""

import os
import sys
import time
from pathlib import Path

def create_minimal_test_script():
    """Create a minimal test script for tensor handling."""
    
    script_path = Path("scripts/test_tensor_handling.py")
    
    script_content = """#!/usr/bin/env python
\"\"\"
Test Script for Neural Plasticity Tensor Handling

This script isolates the tensor operations and visualizations that were
causing BLAS/libtorch crashes in the Neural Plasticity notebook.
\"\"\"

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set environment variables for safer execution
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Make sure utils module is in the path
sys.path.insert(0, os.path.abspath('.'))

print("Testing Neural Plasticity tensor handling functions")
print("=" * 50)

# Import neural plasticity modules
try:
    from utils.neural_plasticity.core import calculate_head_entropy
    from utils.neural_plasticity.visualization import visualize_head_entropy, visualize_head_gradients
    from utils.colab.helpers import safe_tensor_imshow
    
    print("✅ Successfully imported neural plasticity modules")
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    sys.exit(1)

# Test tensor creation and entropy calculation
print("\\nTesting attention tensor creation...")
try:
    # Create test tensors (similar to what would be in the notebook)
    batch_size = 2
    num_heads = 4
    seq_len = 32
    
    # Create random attention-like matrices
    attention_maps = torch.rand(batch_size, num_heads, seq_len, seq_len)
    
    # Ensure they sum to 1 along the last dimension (proper attention distributions)
    attention_maps = attention_maps / attention_maps.sum(dim=-1, keepdim=True)
    
    print(f"Created attention tensor of shape {attention_maps.shape}")
    print(f"Attention min/max/mean: {attention_maps.min().item():.4f}/{attention_maps.max().item():.4f}/{attention_maps.mean().item():.4f}")
    
    # Check that rows sum to 1
    row_sums = attention_maps.sum(dim=-1)
    print(f"Row sums close to 1.0: {torch.allclose(row_sums, torch.ones_like(row_sums))}")
    
    # Test entropy calculation
    print("\\nTesting entropy calculation...")
    entropy = calculate_head_entropy(attention_maps)
    print(f"Entropy shape: {entropy.shape}")
    print(f"Entropy min/max/mean: {entropy.min().item():.4f}/{entropy.max().item():.4f}/{entropy.mean().item():.4f}")
    
    print("✅ Entropy calculation successful")
except Exception as e:
    print(f"❌ Error in tensor operations: {e}")
    sys.exit(1)

# Test visualization functions
print("\\nTesting visualization functions...")
try:
    # Simulate layer structure (just duplicate the same entropy values)
    num_layers = 4
    num_heads = 4
    layer_entropies = entropy.unsqueeze(0).expand(num_layers, num_heads)
    
    print(f"Creating entropy heatmap for tensor of shape {layer_entropies.shape}")
    
    # Test visualization
    fig = visualize_head_entropy(
        entropy_values=layer_entropies,
        title="Test Entropy Visualization",
        annotate=True,
        figsize=(10, 6)
    )
    
    # Save the figure
    test_fig_path = "test_entropy_heatmap.png"
    plt.savefig(test_fig_path)
    plt.close(fig)
    
    print(f"✅ Successfully created entropy visualization at {test_fig_path}")
    
    # Test gradient visualization
    print("\\nTesting gradient visualization...")
    
    # Create mock gradient values
    grad_values = torch.rand(num_layers, num_heads)
    
    # Create test pruned/revived heads
    pruned_heads = [(0, 1), (2, 3)]  # Layer 0, Head 1 and Layer 2, Head 3
    revived_heads = [(1, 2)]         # Layer 1, Head 2
    
    # Create visualization
    fig2 = visualize_head_gradients(
        grad_norm_values=grad_values,
        pruned_heads=pruned_heads,
        revived_heads=revived_heads,
        title="Test Gradient Visualization",
        figsize=(10, 5)
    )
    
    # Save the figure
    test_grad_path = "test_gradient_heatmap.png"
    plt.savefig(test_grad_path)
    plt.close(fig2)
    
    print(f"✅ Successfully created gradient visualization at {test_grad_path}")
    
except Exception as e:
    print(f"❌ Error in visualization: {e}")
    sys.exit(1)

# Test safe_tensor_imshow with both CPU and simulated GPU tensors
print("\\nTesting safe_tensor_imshow function...")
try:
    # Create a test tensor
    test_tensor = torch.rand(10, 10)
    
    # Test with additional metadata that simulates a GPU tensor with gradients
    test_tensor.requires_grad = True
    
    # Use safe_tensor_imshow
    plt.figure(figsize=(8, 6))
    img = safe_tensor_imshow(test_tensor, title="Safe Tensor Visualization Test")
    
    # Save the figure
    test_safe_path = "test_safe_tensor_imshow.png"
    plt.savefig(test_safe_path)
    plt.close()
    
    print(f"✅ Successfully created safe tensor visualization at {test_safe_path}")
except Exception as e:
    print(f"❌ Error in safe_tensor_imshow: {e}")
    sys.exit(1)

print("\\n" + "=" * 50)
print("All tensor handling and visualization tests completed successfully!")
print("This suggests that the fixes applied have resolved the BLAS/libtorch issues.")
"""
    
    # Write the script to disk
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    print(f"Created minimal test script at {script_path}")
    return script_path

def main():
    """Run the script creation."""
    script_path = create_minimal_test_script()
    
    print(f"\nTo test the tensor handling without running the full notebook, use:")
    print(f"  python {script_path}")
    
if __name__ == "__main__":
    main()