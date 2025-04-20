#!/usr/bin/env python
"""
Create Minimal Neural Plasticity Notebook

This script creates a minimal test notebook that focuses only on
the tensor operations and visualizations, avoiding the dataset
loading and full model training that was causing dependencies issues.
"""

import os
import sys
import time
import json
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def create_minimal_notebook():
    """Create a minimal test notebook."""
    
    # Create notebook structure
    notebook = new_notebook()
    
    # Add title cell
    title_cell = new_markdown_cell(
        "# Neural Plasticity Minimal Test (v0.0.53)\n\n"
        "This notebook tests the core tensor operations and visualizations "
        "of the neural plasticity module without loading datasets or full models."
    )
    notebook.cells.append(title_cell)
    
    # Add imports cell
    imports_cell = new_code_cell(
        "import os\n"
        "import sys\n"
        "import time\n"
        "import torch\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "%matplotlib inline\n\n"
        "# Set environment variables for safer execution\n"
        "os.environ['OMP_NUM_THREADS'] = '1'\n"
        "os.environ['OPENBLAS_NUM_THREADS'] = '1'\n"
        "os.environ['MKL_NUM_THREADS'] = '1'\n"
        "os.environ['NUMEXPR_NUM_THREADS'] = '1'\n\n"
        "# Add project root to path\n"
        "if not os.getcwd() in sys.path:\n"
        "    sys.path.append(os.getcwd())\n\n"
        "# Import neural plasticity modules\n"
        "from utils.neural_plasticity.core import calculate_head_entropy\n"
        "from utils.neural_plasticity.visualization import (\n"
        "    visualize_head_entropy,\n"
        "    visualize_head_gradients,\n"
        "    visualize_pruning_decisions,\n"
        "    visualize_attention_patterns\n"
        ")\n"
        "from utils.colab.helpers import safe_tensor_imshow\n\n"
        "print(\"Neural plasticity imports successful\")"
    )
    notebook.cells.append(imports_cell)
    
    # Add tensor creation cell
    tensor_cell = new_code_cell(
        "# Create test attention tensors\n"
        "batch_size = 2\n"
        "num_heads = 4\n"
        "num_layers = 6\n"
        "seq_len = 32\n\n"
        "# Create random attention-like matrices\n"
        "attention_maps = torch.rand(batch_size, num_heads, seq_len, seq_len)\n\n"
        "# Ensure they sum to 1 along the last dimension (proper attention distributions)\n"
        "attention_maps = attention_maps / attention_maps.sum(dim=-1, keepdim=True)\n\n"
        "print(f\"Created attention tensor of shape {attention_maps.shape}\")\n"
        "print(f\"Attention min/max/mean: {attention_maps.min().item():.4f}/{attention_maps.max().item():.4f}/{attention_maps.mean().item():.4f}\")\n\n"
        "# Check that rows sum to 1\n"
        "row_sums = attention_maps.sum(dim=-1)\n"
        "print(f\"Row sums close to 1.0: {torch.allclose(row_sums, torch.ones_like(row_sums))}\")"
    )
    notebook.cells.append(tensor_cell)
    
    # Add entropy calculation cell
    entropy_cell = new_code_cell(
        "# Test entropy calculation\n"
        "entropy = calculate_head_entropy(attention_maps)\n"
        "print(f\"Entropy shape: {entropy.shape}\")\n"
        "print(f\"Entropy min/max/mean: {entropy.min().item():.4f}/{entropy.max().item():.4f}/{entropy.mean().item():.4f}\")\n\n"
        "# Create example layer x head entropy tensor\n"
        "layer_entropies = torch.rand(num_layers, num_heads)\n"
        "print(f\"Layer entropies shape: {layer_entropies.shape}\")"
    )
    notebook.cells.append(entropy_cell)
    
    # Add entropy visualization cell
    viz_entropy_cell = new_code_cell(
        "# Test entropy visualization\n"
        "fig1 = visualize_head_entropy(\n"
        "    entropy_values=layer_entropies,\n"
        "    title=\"Test Entropy Visualization\",\n"
        "    annotate=True,\n"
        "    figsize=(10, 6)\n"
        ")\n\n"
        "# Show the figure\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
    notebook.cells.append(viz_entropy_cell)
    
    # Add gradient visualization cell
    viz_gradient_cell = new_code_cell(
        "# Create mock gradient values\n"
        "grad_values = torch.rand(num_layers, num_heads)\n\n"
        "# Create test pruned/revived heads\n"
        "pruned_heads = [(0, 1), (2, 3)]  # Layer 0, Head 1 and Layer 2, Head 3\n"
        "revived_heads = [(1, 2)]         # Layer 1, Head 2\n\n"
        "# Test gradient visualization\n"
        "fig2 = visualize_head_gradients(\n"
        "    grad_norm_values=grad_values,\n"
        "    pruned_heads=pruned_heads,\n"
        "    revived_heads=revived_heads,\n"
        "    title=\"Test Gradient Visualization\",\n"
        "    figsize=(10, 5)\n"
        ")\n\n"
        "# Show the figure\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
    notebook.cells.append(viz_gradient_cell)
    
    # Add pruning mask visualization cell
    viz_pruning_cell = new_code_cell(
        "# Create mock pruning mask\n"
        "pruning_mask = torch.zeros(num_layers, num_heads, dtype=torch.bool)\n"
        "for layer, head in pruned_heads:\n"
        "    pruning_mask[layer, head] = True\n\n"
        "# Test pruning decision visualization\n"
        "fig3 = visualize_pruning_decisions(\n"
        "    grad_norm_values=grad_values,\n"
        "    pruning_mask=pruning_mask,\n"
        "    title=\"Test Pruning Decisions\",\n"
        "    figsize=(10, 5)\n"
        ")\n\n"
        "# Show the figure\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
    notebook.cells.append(viz_pruning_cell)
    
    # Add attention pattern visualization cell
    viz_attention_cell = new_code_cell(
        "# Test attention pattern visualization\n"
        "# For single head\n"
        "fig4 = visualize_attention_patterns(\n"
        "    attention_maps=attention_maps,\n"
        "    layer_idx=0,\n"
        "    head_idx=0,\n"
        "    title=\"Single Head Attention Pattern\",\n"
        "    figsize=(8, 6)\n"
        ")\n\n"
        "# Show the figure\n"
        "plt.tight_layout()\n"
        "plt.show()\n\n"
        "# For multiple heads\n"
        "fig5 = visualize_attention_patterns(\n"
        "    attention_maps=attention_maps,\n"
        "    layer_idx=0,\n"
        "    head_idx=None,  # Show multiple heads\n"
        "    title=\"Multiple Heads Attention Patterns\",\n"
        "    figsize=(14, 6),\n"
        "    num_heads=4\n"
        ")\n\n"
        "# Show the figure\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
    notebook.cells.append(viz_attention_cell)
    
    # Add safe_tensor_imshow test cell
    tensor_imshow_cell = new_code_cell(
        "# Test safe_tensor_imshow with both CPU and simulated GPU tensors\n"
        "print(\"Testing safe_tensor_imshow function...\")\n\n"
        "# Create a test tensor\n"
        "test_tensor = torch.rand(10, 10)\n\n"
        "# Test with additional metadata that simulates a GPU tensor with gradients\n"
        "test_tensor.requires_grad = True\n\n"
        "# Use safe_tensor_imshow\n"
        "plt.figure(figsize=(8, 6))\n"
        "img = safe_tensor_imshow(\n"
        "    test_tensor, \n"
        "    title=\"Safe Tensor Visualization Test\",\n"
        "    cmap=\"inferno\"\n"
        ")\n\n"
        "plt.tight_layout()\n"
        "plt.show()\n\n"
        "print(\"✅ Safe tensor visualization completed\")"
    )
    notebook.cells.append(tensor_imshow_cell)
    
    # Add conclusion cell
    conclusion_cell = new_markdown_cell(
        "## Test Results\n\n"
        "If you've reached this point without errors, the neural plasticity tensor operations "
        "and visualizations are working correctly. This indicates that the fixes for BLAS/libtorch "
        "issues have been successful.\n\n"
        "### Next Steps\n\n"
        "1. Try running the full notebook in Colab with GPU acceleration\n"
        "2. If issues persist, examine specific components as needed"
    )
    notebook.cells.append(conclusion_cell)
    
    # Save notebook
    notebook_path = Path("neural_plasticity_minimal_test.ipynb")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(notebook, f)
    
    print(f"Created minimal test notebook at {notebook_path}")
    return notebook_path

def main():
    """Create the minimal notebook."""
    notebook_path = create_minimal_notebook()
    
    print(f"\nTo run the minimal test notebook:")
    print(f"  jupyter notebook {notebook_path}")
    
if __name__ == "__main__":
    main()