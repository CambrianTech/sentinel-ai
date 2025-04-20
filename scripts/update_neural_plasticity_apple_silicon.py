#!/usr/bin/env python
"""
Update Neural Plasticity Notebook for Apple Silicon

This script updates the NeuralPlasticityDemo notebook to include Apple Silicon
optimizations for reliable execution on M1/M2/M3 platforms.

Version: v0.0.57 (2025-04-19 22:40:00)
"""

import os
import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell
from datetime import datetime

# Set version and timestamp
VERSION = "0.0.57"
TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def update_notebook():
    """Update Neural Plasticity notebook with Apple Silicon optimizations."""
    # Path to the notebook
    notebook_path = os.path.join("notebooks", "NeuralPlasticityDemo.ipynb")
    
    # Read the notebook
    try:
        nb = nbformat.read(notebook_path, as_version=4)
        print(f"Successfully read notebook: {notebook_path}")
    except Exception as e:
        print(f"Error reading notebook: {e}")
        return
    
    # Update the title cell to include version and timestamp
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown" and cell.source.startswith("# Neural Plasticity"):
            title = f"# Neural Plasticity in Transformer Models (v{VERSION})\n\n"
            description = "\n".join(cell.source.split('\n')[1:])
            nb.cells[i].source = title + description
            print("Updated title cell with new version")
            break
    
    # Add Apple Silicon support cell after imports
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and "import" in cell.source and "matplotlib" in cell.source:
            # Add Apple Silicon support cell after imports
            apple_silicon_cell = new_code_cell("""# Add Apple Silicon optimization support
import sys
import platform

# Detect Apple Silicon
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.processor() == "arm"

# Add Apple Silicon optimizations if needed
if IS_APPLE_SILICON:
    try:
        # Import the Apple Silicon optimization utilities
        from utils.apple_silicon import apply_tensor_patches, safe_context, ensure_cpu_model
        
        print("🍎 Apple Silicon detected - enabling optimizations")
        # Apply tensor safety patches for better stability
        apply_tensor_patches()
        
        # Force PyTorch to use CPU and single-threading for stability
        import torch
        torch.set_num_threads(1)
        
        # Also set matplotlib to use Agg backend for better stability
        import matplotlib
        matplotlib.use('Agg')
        print("🎨 Switched to Agg matplotlib backend for stability")
        
    except ImportError:
        print("⚠️ Apple Silicon detected but optimization utilities not available")
        print("   Some operations may be unstable. Consider installing utils.apple_silicon module.")
        
    print("ℹ️ When running on Apple Silicon, model operations will be forced to CPU")
    print("   This prevents BLAS/libtorch crashes that commonly occur on M1/M2/M3 chips")""")
            
            # Insert the Apple Silicon support cell after the imports
            nb.cells.insert(i + 1, apple_silicon_cell)
            print("Added Apple Silicon support cell")
            break
    
    # Add model CPU forcing for Apple Silicon after model loading
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and "MODEL_NAME" in cell.source and "distilgpt2" in cell.source:
            # Find the device setting line
            lines = cell.source.split('\n')
            for j, line in enumerate(lines):
                if "DEVICE" in line and "cuda" in line:
                    # Update the line to respect Apple Silicon
                    lines[j] = """# Set device, forcing CPU on Apple Silicon
DEVICE = "cpu" if IS_APPLE_SILICON else ("cuda" if torch.cuda.is_available() else "cpu")"""
                    nb.cells[i].source = '\n'.join(lines)
                    print("Updated device selection for Apple Silicon")
                    break
                    
    # Add a "Dealing with Apple Silicon" section after the overview
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown" and "## 1. Setup and Configuration" in cell.source:
            # Add explanation of Apple Silicon optimizations
            apple_silicon_info = new_markdown_cell("""## Note on Apple Silicon Support

This notebook has been updated with specific optimizations to run reliably on Apple Silicon (M1/M2/M3) hardware. 

These optimizations include:
1. Forcing CPU usage even if CUDA is ostensibly available
2. Single-threaded BLAS operations to prevent crashes
3. Safe matrix multiplication with proper tensor management
4. Memory layout optimization with contiguous tensors
5. NaN/Inf value detection and handling

If you encounter any issues on Apple Silicon, try:
1. Using smaller batch sizes
2. Running smaller models
3. Adding more tensor validation checks

For Colab users, these optimizations are automatically disabled to maintain performance.""")
            
            # Insert the Apple Silicon info cell before the setup section
            nb.cells.insert(i, apple_silicon_info)
            print("Added Apple Silicon info section")
            break
            
    # Save the updated notebook
    output_path = os.path.join("notebooks", f"NeuralPlasticityDemo_AppleSilicon.ipynb")
    nbformat.write(nb, output_path)
    print(f"Successfully saved updated notebook to: {output_path}")
    
    # Also overwrite the original if requested
    overwrite = True
    if overwrite:
        nbformat.write(nb, notebook_path)
        print(f"Overwrote original notebook: {notebook_path}")
    
if __name__ == "__main__":
    update_notebook()