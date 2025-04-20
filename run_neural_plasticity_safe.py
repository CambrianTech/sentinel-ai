#!/usr/bin/env python
"""
Run Neural Plasticity Demo Safely on Apple Silicon
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
import nbformat
import copy

def set_environment_variables():
    """Set environment variables to improve stability on Apple Silicon."""
    # Disable multithreading in OpenMP, OpenBLAS, and MKL
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    
    # Use simpler BLAS implementation when available
    os.environ['ACCELERATE_USE_SYSTEM_BLAS'] = '1'
    
    # Disable JIT Autotuner
    os.environ['PYTORCH_JIT_USE_AUTOTUNER'] = '0'
    
    # Use PyTorch's PyTorch FFT instead of MKL
    os.environ['TORCH_USE_MKL_FFT'] = '0'
    
    # Set PyTorch seed for reproducibility
    os.environ['PYTHONHASHSEED'] = '0'
    
    # Avoid potential conflicts with sentinel_data
    os.environ['PYTHONPATH'] = str(Path.cwd())
    
    print("✅ Set environment variables for improved stability")

def prepare_minimal_notebook(notebook_path):
    """
    Prepare a minimal version of the notebook for testing
    
    Args:
        notebook_path: Path to the original notebook
    
    Returns:
        Path to the minimal notebook
    """
    temp_notebook = f"temp_notebook_{int(time.time())}.ipynb"
    
    print(f"Creating minimal version of {notebook_path}")
    with open(notebook_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Create a modified copy
    modified_nb = copy.deepcopy(nb)
    
    # Add a new cell at the beginning to fix imports
    fix_cell = nbformat.v4.new_code_cell(
        source="""# Fix for path conflicts with sentinel_data
import sys
import os
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, os.getcwd())

# Prioritize the official datasets package
sys_paths = sys.path.copy()
project_path = None

for p in sys_paths:
    if p.endswith('sentinel-ai'):
        project_path = p
        # Move sentinel-ai to end of path
        sys.path.remove(p)
        sys.path.append(p)
        print(f"Moving sentinel-ai to end of import path")
        break

# Set environment variables for numerical stability on Apple Silicon
import platform
if platform.system() == "Darwin" and platform.processor() == "arm":
    # Force single-threaded BLAS operations
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["ACCELERATE_USE_SYSTEM_BLAS"] = "1"
    print("🍎 Environment optimized for Apple Silicon")\
"""
    )
    
    # Insert at beginning of notebook
    modified_nb.cells.insert(0, fix_cell)
    
    # Modify parameters in the configuration cells to speed up execution
    for i, cell in enumerate(modified_nb.cells):
        if cell.cell_type == 'code' and 'NUM_EPOCHS' in cell.source:
            print(f"Modifying parameters in cell {i}")
            cell.source = cell.source.replace("NUM_EPOCHS = 100", "NUM_EPOCHS = 1")
            cell.source = cell.source.replace("BATCH_SIZE = 4", "BATCH_SIZE = 2")
            cell.source = cell.source.replace("MAX_LENGTH = 128", "MAX_LENGTH = 64")
            cell.source = cell.source.replace("ENABLE_LONG_TRAINING = False", "ENABLE_LONG_TRAINING = False")
            cell.source = cell.source.replace("MAX_STEPS_PER_EPOCH = 200", "MAX_STEPS_PER_EPOCH = 10")
            # Speed up evaluation
            cell.source = cell.source.replace("EVAL_INTERVAL = 50", "EVAL_INTERVAL = 5")
            cell.source = cell.source.replace("VISUALIZATION_INTERVAL = 100", "VISUALIZATION_INTERVAL = 10")
            cell.source = cell.source.replace("INFERENCE_INTERVAL = 500", "INFERENCE_INTERVAL = 20") 
            cell.source = cell.source.replace("CHECKPOINT_INTERVAL = 500", "CHECKPOINT_INTERVAL = 20")
    
    # Save the modified notebook
    with open(temp_notebook, 'w', encoding='utf-8') as f:
        nbformat.write(modified_nb, f)
    
    print(f"Created temporary notebook at {temp_notebook}")
    return temp_notebook

def run_notebook(notebook_path, output_path=None):
    """
    Run the notebook with nbconvert.
    
    Args:
        notebook_path: Path to the notebook
        output_path: Path to save the executed notebook (optional)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = ['jupyter', 'nbconvert', '--to', 'notebook', '--execute', 
               '--ExecutePreprocessor.timeout=600', str(notebook_path)]
        
        if output_path:
            cmd.extend(['--output', str(output_path)])
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"✅ Successfully executed {notebook_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error executing notebook: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run Neural Plasticity Notebook safely on Apple Silicon')
    parser.add_argument('--notebook', type=str, default='notebooks/NeuralPlasticityDemo.ipynb',
                        help='Path to the notebook')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for executed notebook')
    
    args = parser.parse_args()
    
    # Set environment variables for stability
    set_environment_variables()
    
    # Create a minimal version of the notebook for testing
    temp_notebook = prepare_minimal_notebook(args.notebook)
    
    try:
        # Run the notebook
        success = run_notebook(temp_notebook, args.output)
        
        if success:
            print("\n🎉 Neural Plasticity Demo ran successfully on Apple Silicon!")
            return 0
        return 1
    finally:
        # Clean up temporary file
        if os.path.exists(temp_notebook):
            os.unlink(temp_notebook)
            print(f"Removed temporary notebook {temp_notebook}")

if __name__ == '__main__':
    sys.exit(main())