#!/usr/bin/env python
"""
Run Neural Plasticity Notebook with Error Handling

This script runs the NeuralPlasticityDemo notebook with proper error handling
for BLAS/libtorch issues. It sets environment variables to improve stability
and runs the notebook in a controlled environment.
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path

def set_environment_variables():
    """Set environment variables to improve stability."""
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
        temp_notebook = Path(f"temp_notebook_{int(time.time())}.ipynb")
        with open(temp_notebook, 'w', encoding='utf-8') as f:
            nbformat.write(modified_nb, f)
        
        print(f"Created temporary notebook at {temp_notebook}")
        notebook_to_run = str(temp_notebook)
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
        elif temp_notebook and Path(temp_notebook).exists():
            Path(temp_notebook).unlink()
            print(f"Removed temporary notebook {temp_notebook}")

if __name__ == '__main__':
    sys.exit(main())
