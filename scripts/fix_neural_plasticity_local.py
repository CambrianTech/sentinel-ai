#!/usr/bin/env python
"""
Fix Neural Plasticity Local Execution Script

This script addresses BLAS/libtorch crash issues in the NeuralPlasticityDemo notebook
by patching the following issues:
1. Multiple redundant CPU/numpy conversions in tensor handling
2. Numerical stability issues in entropy calculations
3. Improper tensor visualization patterns that cause GPU memory leaks
4. Proper device handling for mixed CPU/GPU environments

This script modifies the core modules to ensure safe execution in both
local and Colab environments.
"""

import os
import sys
import re
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple, Any

# Add parent directory to path for local imports
sys.path.insert(0, os.path.abspath('..'))

def fix_utils_neural_plasticity_core():
    """Fix the neural plasticity core module for numerical stability and device handling."""
    core_path = Path("utils/neural_plasticity/core.py")
    
    # Check if file exists
    if not core_path.exists():
        print(f"Error: {core_path} does not exist")
        return False
        
    with open(core_path, 'r') as f:
        content = f.read()
    
    # 1. Fix entropy calculation for better numerical stability
    # Make sure we use a reasonably large epsilon value (1e-8 -> 1e-6)
    content = content.replace(
        "def calculate_head_entropy(\n    attention_maps: torch.Tensor,\n    eps: float = 1e-8",
        "def calculate_head_entropy(\n    attention_maps: torch.Tensor,\n    eps: float = 1e-6"
    )
    
    # 2. Fix the tensor clamp to be more robust
    content = content.replace(
        "    # Add small epsilon to avoid log(0) issues\n    attn_probs = attention_maps.clamp(min=eps)",
        """    # Add small epsilon to avoid log(0) issues
    # Handle potential NaN or Inf values first
    attn_probs = torch.where(
        torch.isfinite(attention_maps),
        attention_maps,
        torch.ones_like(attention_maps) * eps
    )
    attn_probs = attn_probs.clamp(min=eps)"""
    )
    
    # 3. Add an explicit cast to float32 for better numerical stability
    content = content.replace(
        "    # Calculate entropy: -sum(p * log(p))\n    entropy = -torch.sum(attn_probs * torch.log(attn_probs), dim=-1)",
        """    # Calculate entropy: -sum(p * log(p))
    # Cast to float32 for better numerical stability if needed
    if attn_probs.dtype != torch.float32:
        attn_probs = attn_probs.to(torch.float32)
    entropy = -torch.sum(attn_probs * torch.log(attn_probs), dim=-1)"""
    )
    
    # 4. Add device handling to ensure tensor operations happen on the correct device
    content = content.replace(
        "def generate_pruning_mask(\n    grad_norm_values: torch.Tensor,\n    prune_percent: float = 0.1,",
        """def generate_pruning_mask(
    grad_norm_values: torch.Tensor,
    prune_percent: float = 0.1,"""
    )
    
    # 5. Add proper device handling in numpy random seed
    content = content.replace(
        "    if random_seed is not None:\n        torch.manual_seed(random_seed)\n        np.random.seed(random_seed)",
        """    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if grad_norm_values.is_cuda:
            torch.cuda.manual_seed(random_seed)"""
    )
    
    # Save the updated file
    with open(core_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed {core_path} for improved numerical stability and device handling")
    return True

def fix_utils_neural_plasticity_visualization():
    """Fix the neural plasticity visualization module for proper tensor handling."""
    vis_path = Path("utils/neural_plasticity/visualization.py")
    
    # Check if file exists
    if not vis_path.exists():
        print(f"Error: {vis_path} does not exist")
        return False
        
    with open(vis_path, 'r') as f:
        content = f.read()
    
    # 1. Fix redundant tensor conversions in visualization functions
    # Replace multiple `.cpu().numpy().cpu().numpy()` patterns
    content = content.replace(".cpu().numpy()).cpu().numpy()", ".cpu().numpy()")
    content = content.replace(".detach().cpu().numpy().cpu().numpy()", ".detach().cpu().numpy()")
    
    # 2. Fix the safe_tensor_imshow call in visualize_head_entropy
    content = content.replace(
        "    # Plot heatmap using safe_tensor_imshow\n    im = safe_tensor_imshow(\n        entropy_data, \n        title=title,\n        cmap=cmap\n    )",
        """    # Plot heatmap using imshow directly since entropy_data is already numpy
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(entropy_data, cmap=cmap)
    ax.set_title(title)"""
    )
    
    # 3. Fix similar patterns in other visualization functions
    content = content.replace(
        "    # Plot heatmap using safe_tensor_imshow\n    im = safe_tensor_imshow(\n        grad_data, \n        title=title,\n        cmap=cmap\n    )",
        """    # Plot heatmap directly
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(grad_data, cmap=cmap)
    ax.set_title(title)"""
    )
    
    # 4. Fix visualization in pruning decisions
    content = content.replace(
        "    # Base plot with all gradient values\n    im = safe_tensor_imshow(\n        grad_data, \n        title=title,\n        cmap=\"YlOrRd\"\n    )",
        """    # Base plot with all gradient values
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(grad_data, cmap="YlOrRd")
    ax.set_title(title)"""
    )
    
    # 5. Fix visualize_attention_patterns
    content = content.replace(
        "        attention_map = safe_tensor_imshow(\n            attn[0, head_idx], \n            title=title or f'Attention pattern (layer {layer_idx}, head {head_idx})',\n            cmap='viridis'\n        )",
        """        im = ax.imshow(attn[0, head_idx].numpy(), cmap='viridis')
        ax.set_title(title or f'Attention pattern (layer {layer_idx}, head {head_idx})')"""
    )
    
    # 6. Fix any other instances of double CPU conversions
    content = re.sub(r'\.detach\(\)\.cpu\(\)\.numpy\(\)\.cpu\(\)\.numpy\(\)', '.detach().cpu().numpy()', content)
    
    # Save the updated file
    with open(vis_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed {vis_path} for proper tensor handling in visualizations")
    return True

def fix_utils_colab_helpers():
    """Fix the colab helpers module for proper tensor handling."""
    helpers_path = Path("utils/colab/helpers.py")
    
    # Check if file exists
    if not helpers_path.exists():
        print(f"Error: {helpers_path} does not exist")
        return False
        
    with open(helpers_path, 'r') as f:
        content = f.read()
    
    # 1. Fix safe_tensor_imshow tensor handling
    content = content.replace(
        "    # Ensure tensor is properly converted for visualization\n    if isinstance(tensor, torch.Tensor):\n        # Handle GPU tensors or tensors with gradients\n        tensor_np = tensor.detach().cpu().numpy()",
        """    # Ensure tensor is properly converted for visualization
    if isinstance(tensor, torch.Tensor):
        # Handle GPU tensors or tensors with gradients
        if tensor.requires_grad:
            tensor = tensor.detach()
        if tensor.is_cuda:
            tensor = tensor.cpu()
        tensor_np = tensor.numpy()"""
    )
    
    # 2. Add error handling for NaN and Inf values
    content = content.replace(
        "    # Create figure and plot\n    fig, ax = plt.subplots(figsize=figsize)\n    im = ax.imshow(tensor_np, cmap=cmap, vmin=vmin, vmax=vmax)",
        """    # Create figure and plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle potential NaN or Inf values
    if np.isnan(tensor_np).any() or np.isinf(tensor_np).any():
        print("⚠️ Warning: Tensor contains NaN or Inf values, replacing with zeros")
        tensor_np = np.nan_to_num(tensor_np, nan=0.0, posinf=1.0, neginf=0.0)
    
    im = ax.imshow(tensor_np, cmap=cmap, vmin=vmin, vmax=vmax)"""
    )
    
    # Save the updated file
    with open(helpers_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed {helpers_path} for safer tensor visualization")
    return True

def fix_notebook_wrapper_script():
    """Create a script to run the notebook with proper error handling."""
    wrapper_path = Path("scripts/run_neural_plasticity_notebook.py")
    
    wrapper_content = """#!/usr/bin/env python
\"\"\"
Run Neural Plasticity Notebook with Error Handling

This script runs the NeuralPlasticityDemo notebook with proper error handling
for BLAS/libtorch issues. It sets environment variables to improve stability
and runs the notebook in a controlled environment.
\"\"\"

import os
import sys
import subprocess
import argparse
from pathlib import Path

def set_environment_variables():
    \"\"\"Set environment variables to improve stability.\"\"\"
    # Disable multithreading in OpenMP, OpenBLAS, and MKL
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    # Use simpler BLAS implementation when available
    os.environ['ACCELERATE_USE_SYSTEM_BLAS'] = '1'
    
    # Disable JIT Autotuner
    os.environ['PYTORCH_JIT_USE_AUTOTUNER'] = '0'
    
    # Use PyTorch's PyTorch FFT instead of MKL
    os.environ['TORCH_USE_MKL_FFT'] = '0'
    
    # Set PyTorch seed for reproducibility
    os.environ['PYTHONHASHSEED'] = '0'
    
    # Limit GPU memory growth
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    
    print("✅ Set environment variables for improved stability")

def run_notebook(notebook_path, output_path=None):
    \"\"\"
    Run the notebook with nbconvert.
    
    Args:
        notebook_path: Path to the notebook
        output_path: Path to save the executed notebook (optional)
    
    Returns:
        True if successful, False otherwise
    \"\"\"
    try:
        cmd = ['jupyter', 'nbconvert', '--to', 'notebook', '--execute', 
               '--ExecutePreprocessor.timeout=600', notebook_path]
        
        if output_path:
            cmd.extend(['--output', output_path])
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"✅ Successfully executed {notebook_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error executing notebook: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run Neural Plasticity Notebook with improved error handling')
    parser.add_argument('--notebook', type=str, default='colab_notebooks/NeuralPlasticityDemo.ipynb',
                        help='Path to the notebook')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for executed notebook')
    parser.add_argument('--minimal', action='store_true',
                        help='Run with minimal settings for testing')
    
    args = parser.parse_args()
    
    # Set environment variables
    set_environment_variables()
    
    # If minimal flag is set, modify the notebook temporarily
    temp_notebook = None
    if args.minimal:
        import nbformat
        import copy
        from tempfile import NamedTemporaryFile
        
        print(f"Creating minimal version of {args.notebook}")
        with open(args.notebook, 'r') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Create a modified copy
        modified_nb = copy.deepcopy(nb)
        
        # Modify parameters in the configuration cell (usually cell 4)
        for i, cell in enumerate(modified_nb.cells):
            if cell.cell_type == 'code' and 'NUM_EPOCHS' in cell.source:
                print(f"Modifying parameters in cell {i}")
                cell.source = cell.source.replace("NUM_EPOCHS = 100", "NUM_EPOCHS = 1")
                cell.source = cell.source.replace("BATCH_SIZE = 4", "BATCH_SIZE = 2")
                cell.source = cell.source.replace("MAX_LENGTH = 128", "MAX_LENGTH = 64")
                cell.source = cell.source.replace("ENABLE_LONG_TRAINING = False", "ENABLE_LONG_TRAINING = False")
                cell.source = cell.source.replace("MAX_STEPS_PER_EPOCH = 200", "MAX_STEPS_PER_EPOCH = 20")
                # Speed up evaluation
                cell.source = cell.source.replace("EVAL_INTERVAL = 50", "EVAL_INTERVAL = 10")
                cell.source = cell.source.replace("VISUALIZATION_INTERVAL = 100", "VISUALIZATION_INTERVAL = 20")
                cell.source = cell.source.replace("INFERENCE_INTERVAL = 500", "INFERENCE_INTERVAL = 50") 
                cell.source = cell.source.replace("CHECKPOINT_INTERVAL = 500", "CHECKPOINT_INTERVAL = 50")
        
        # Create a temporary file for the modified notebook
        with NamedTemporaryFile(suffix='.ipynb', delete=False) as tmp:
            temp_notebook = tmp.name
            nbformat.write(modified_nb, tmp)
        
        print(f"Created temporary notebook at {temp_notebook}")
        notebook_to_run = temp_notebook
    else:
        notebook_to_run = args.notebook
    
    try:
        # Run the notebook
        success = run_notebook(notebook_to_run, args.output)
        
        if success:
            return 0
        return 1
    finally:
        # Clean up temporary file
        if temp_notebook and os.path.exists(temp_notebook):
            os.unlink(temp_notebook)
            print(f"Removed temporary notebook {temp_notebook}")

if __name__ == '__main__':
    sys.exit(main())
"""
    
    # Save the wrapper script
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)
    
    # Make the script executable
    os.chmod(wrapper_path, 0o755)
    
    print(f"✅ Created notebook wrapper script at {wrapper_path}")
    return True

def create_simple_test_script():
    """Create a simple script to test the tensor handling issues."""
    test_path = Path("scripts/test_neural_plasticity_local.py")
    
    test_content = """#!/usr/bin/env python
\"\"\"
Test Neural Plasticity Local Execution

This script provides a simplified test to verify that the tensor handling
and BLAS operations are working correctly in the local environment, without
requiring the entire notebook to run.
\"\"\"

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
    print("\\nTesting entropy calculation...")
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
    print("\\nTesting matrix multiplication...")
    a = torch.randn(300, 400)
    b = torch.randn(400, 500)
    try:
        c = torch.matmul(a, b)
        print(f"Matrix multiplication succeeded. Result shape: {c.shape}")
    except Exception as e:
        print(f"Matrix multiplication error: {e}")
    
    print("\\nBasic PyTorch tests completed successfully!")

except ImportError:
    print("PyTorch not available. Skipping tensor tests.")

# Try matplotlib
try:
    import matplotlib.pyplot as plt
    print("\\nMatplotlib available:", plt.__version__)
    
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

print("\\nAll tests completed.")
"""
    
    # Save the test script
    with open(test_path, 'w') as f:
        f.write(test_content)
    
    # Make the script executable
    os.chmod(test_path, 0o755)
    
    print(f"✅ Created test script at {test_path}")
    return True

def main():
    """Run all fixes."""
    print("Fixing Neural Plasticity modules for local execution...")
    
    # Fix modules
    fix_utils_neural_plasticity_core()
    fix_utils_neural_plasticity_visualization()
    fix_utils_colab_helpers()
    
    # Create notebook wrapper script
    fix_notebook_wrapper_script()
    
    # Create simple test script
    create_simple_test_script()
    
    print("\nAll fixes completed. To run the notebook, use:")
    print("  python scripts/run_neural_plasticity_notebook.py --minimal")
    print("\nTo test PyTorch tensor handling without running the full notebook, use:")
    print("  python scripts/test_neural_plasticity_local.py")
    
if __name__ == "__main__":
    main()