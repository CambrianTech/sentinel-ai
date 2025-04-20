#!/usr/bin/env python
"""
Create Minimal Runnable Version of Neural Plasticity Notebook

This script creates a version of the Neural Plasticity notebook that:
1. Runs the full pipeline including model loading
2. Uses minimal settings (smaller model, fewer steps, smaller dataset)
3. Sets all necessary environment variables for stability
4. Keeps BLAS/libtorch operations safe with optimized configurations
"""

import os
import sys
import json
import time
import uuid
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

def create_runnable_notebook():
    """Create a runnable version of the Neural Plasticity notebook."""
    
    # Load the original notebook
    source_path = Path("colab_notebooks/NeuralPlasticityDemo.ipynb")
    if not source_path.exists():
        print(f"Error: Source notebook not found at {source_path}")
        return None
    
    try:
        with open(source_path, 'r', encoding='utf-8') as f:
            original_nb = nbformat.read(f, as_version=4)
    except Exception as e:
        print(f"Error reading notebook: {e}")
        return None
    
    # Create a new notebook
    new_nb = new_notebook()
    
    # Copy the metadata
    new_nb.metadata = original_nb.metadata
    
    # Update the title cell with minimal settings note and proper versioning
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    unique_id = str(uuid.uuid4())[:8]
    title_cell = new_markdown_cell(
        f"# Neural Plasticity Demo: Dynamic Pruning & Regrowth (v0.0.54 {current_time})\n\n"
        "This notebook demonstrates Sentinel AI's neural plasticity system, which allows transformer models to "
        "dynamically prune and regrow attention heads during training based on utility metrics. "
        f"[ID: {unique_id}]\n\n"
        "### Changes in v0.0.54:\n"
        "- Fixed GPU tensor handling for visualizations\n"
        "- Fixed redundant tensor conversion patterns\n"
        "- Improved numerical stability in entropy calculations\n"
        "- Modified settings for local execution\n"
        "- Added environment variables for BLAS stability\n\n"
        "## ⚠️ Minimal Settings Version ⚠️\n"
        "This is a minimal settings version for local testing with reduced:\n"
        "- Model size (using distilgpt2 instead of larger models)\n"
        "- Dataset size \n"
        "- Training iterations\n"
        "- Visualization frequency\n\n"
        "For full performance, use the standard version in Google Colab with GPU acceleration."
    )
    new_nb.cells.append(title_cell)
    
    # Process all cells
    for i, cell in enumerate(original_nb.cells):
        # Skip the title cell (already added a new one)
        if i == 0 and cell.cell_type == "markdown" and cell.source.startswith("# Neural Plasticity Demo"):
            continue
        
        # Add environment variables to the imports cell
        if i == 6 and cell.cell_type == "code" and "%matplotlib inline" in cell.source:
            # Add environment variables for stability
            env_vars_code = (
                "# Set environment variables for BLAS stability\n"
                "import os\n"
                "os.environ['OMP_NUM_THREADS'] = '1'\n"
                "os.environ['OPENBLAS_NUM_THREADS'] = '1'\n"
                "os.environ['MKL_NUM_THREADS'] = '1'\n"
                "os.environ['NUMEXPR_NUM_THREADS'] = '1'\n"
                "os.environ['PYTHONHASHSEED'] = '0'\n\n"
            )
            
            # Add tensor handling safety note
            safety_note = (
                "# Tensor handling and visualization fixes for BLAS/libtorch issues\n"
                "import gc\n"
                "def clear_memory():\n"
                "    \"\"\"Clear GPU memory cache and run garbage collection\"\"\"\n"
                "    gc.collect()\n"
                "    if torch.cuda.is_available():\n"
                "        torch.cuda.empty_cache()\n"
                "        torch.cuda.synchronize()\n\n"
            )
            
            modified_code = env_vars_code + safety_note + cell.source
            new_cell = new_code_cell(modified_code)
            new_nb.cells.append(new_cell)
            continue
        
        # Modify configuration cell (cell 4) for minimal settings
        if i == 4 and cell.cell_type == "code" and "MODEL_NAME = " in cell.source:
            # Keep the original code but with minimal settings
            modified_code = cell.source
            # Replace settings with smaller values
            replacements = {
                "NUM_EPOCHS = 100": "NUM_EPOCHS = 1",
                "BATCH_SIZE = 4": "BATCH_SIZE = 2",
                "MAX_LENGTH = 128": "MAX_LENGTH = 64",
                "EVAL_INTERVAL = 50": "EVAL_INTERVAL = 10",
                "VISUALIZATION_INTERVAL = 100": "VISUALIZATION_INTERVAL = 20",
                "INFERENCE_INTERVAL = 500": "INFERENCE_INTERVAL = 50",
                "CHECKPOINT_INTERVAL = 500": "CHECKPOINT_INTERVAL = 50",
                "MAX_STEPS_PER_EPOCH = 200": "MAX_STEPS_PER_EPOCH = 20"
            }
            
            for old, new in replacements.items():
                modified_code = modified_code.replace(old, new)
            
            new_cell = new_code_cell(modified_code)
            new_nb.cells.append(new_cell)
            continue
        
        # Modify warm-up cell for fewer steps
        if i == 10 and cell.cell_type == "code" and "Warm-up training loop" in cell.source:
            modified_code = cell.source.replace("patience = 15", "patience = 5")
            modified_code = modified_code.replace("min_warmup_steps = 50", "min_warmup_steps = 10")
            modified_code = modified_code.replace("max_warmup_steps = 150", "max_warmup_steps = 20")
            new_cell = new_code_cell(modified_code)
            new_nb.cells.append(new_cell)
            continue
        
        # Add clear_memory() calls after heavy operations
        if i == 31 and cell.cell_type == "code" and "Training loop" in cell.source:
            # Add clear_memory call at the end of each epoch
            if "epoch_steps = 0" in cell.source and "clear_memory()" not in cell.source:
                modified_code = cell.source.replace(
                    "print(f\"Completed Epoch {epoch+1} - Total steps: {global_step}\")",
                    "print(f\"Completed Epoch {epoch+1} - Total steps: {global_step}\")\n        # Clear memory at the end of each epoch\n        clear_memory()"
                )
                new_cell = new_code_cell(modified_code)
                new_nb.cells.append(new_cell)
                continue
        
        # Otherwise, add the cell as-is
        new_nb.cells.append(cell)
    
    # Save the modified notebook
    output_path = Path("neural_plasticity_runnable.ipynb")
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(new_nb, f)
    
    print(f"Created runnable notebook at {output_path}")
    return output_path

def main():
    """Create the runnable notebook."""
    notebook_path = create_runnable_notebook()
    
    if notebook_path:
        print(f"\nTo run the notebook:")
        print(f"  cd {os.getcwd()} && source .venv/bin/activate && jupyter notebook {notebook_path}")
        print(f"\nAlternatively, you can run it with nbconvert:")
        print(f"  cd {os.getcwd()} && source .venv/bin/activate && jupyter nbconvert --to notebook --execute {notebook_path} --output neural_plasticity_executed.ipynb")
    
if __name__ == "__main__":
    main()