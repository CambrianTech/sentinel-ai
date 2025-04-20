#!/usr/bin/env python
"""
Run NeuralPlasticityDemo.ipynb end-to-end with reduced parameters

This script applies fixes to the notebook, validates it, and runs it
with reduced parameters for faster testing.
"""

import os
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import logging
import argparse
import json
import subprocess
import time
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_notebook(notebook_path, output_path):
    """Fix visualization issues in the notebook using the dedicated script"""
    logger.info(f"Fixing notebook visualization issues: {notebook_path} -> {output_path}")
    
    # Run fix script
    fix_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fix_neural_plasticity_demo.py")
    try:
        result = subprocess.run([sys.executable, fix_script, notebook_path, output_path], 
                              capture_output=True, text=True, check=True)
        logger.info(f"Fixed notebook: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error fixing notebook: {e.stderr}")
        return False

def validate_notebook(notebook_path):
    """Validate the notebook using the dedicated validation script"""
    logger.info(f"Validating notebook: {notebook_path}")
    
    # Run validation script
    validate_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                 "validate_neural_plasticity_notebook.py")
    try:
        result = subprocess.run([sys.executable, validate_script, notebook_path], 
                              capture_output=True, text=True, check=True)
        logger.info(f"Validation successful: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Validation failed: {e.stdout}")
        return False

def clean_notebook(notebook_path, output_path):
    """Clean notebook of incompatible fields (like 'id')"""
    # Read the notebook as JSON to preserve all fields
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb_json = json.load(f)
    
    # Remove 'id' field from all cells
    for cell in nb_json['cells']:
        if 'id' in cell:
            del cell['id']
    
    # Save cleaned notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb_json, f, indent=1)
    
    # Now read with nbformat for further processing
    with open(output_path, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def modify_notebook_for_testing(nb, output_path):
    """Modify notebook to use smaller parameters for testing"""
    
    # Find the cell with model configuration
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and 'MODEL_NAME = ' in cell.source and 'NUM_EPOCHS =' in cell.source:
            logger.info(f"Found configuration cell at index {i}")
            
            # Modify parameters for faster execution
            cell.source = cell.source.replace('NUM_EPOCHS = 100', 'NUM_EPOCHS = 1')
            cell.source = cell.source.replace('NUM_EPOCHS = 3', 'NUM_EPOCHS = 1')
            cell.source = cell.source.replace('MAX_STEPS_PER_EPOCH = 200', 'MAX_STEPS_PER_EPOCH = 10')
            cell.source = cell.source.replace('BATCH_SIZE = 4', 'BATCH_SIZE = 2')
            cell.source = cell.source.replace('MAX_LENGTH = 128', 'MAX_LENGTH = 64')
            cell.source = cell.source.replace('VISUALIZATION_INTERVAL = 100', 'VISUALIZATION_INTERVAL = 5')
            cell.source = cell.source.replace('EVAL_INTERVAL = 50', 'EVAL_INTERVAL = 5')
            cell.source = cell.source.replace('INFERENCE_INTERVAL = 500', 'INFERENCE_INTERVAL = 10')
            
            logger.info("Modified configuration for faster testing")
            break
    
    # Save the modified notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    logger.info(f"Saved test notebook to: {output_path}")
    return nb

def get_available_kernel():
    """Get an available Jupyter kernel"""
    try:
        # List available kernels
        output = subprocess.check_output(['jupyter', 'kernelspec', 'list'], 
                                        stderr=subprocess.STDOUT, 
                                        universal_newlines=True)
        
        # Parse output to find available kernels
        lines = output.splitlines()
        kernels = []
        for line in lines[1:]:  # Skip header line
            if line.strip():
                kernel_name = line.split()[0]
                kernels.append(kernel_name)
        
        if kernels:
            logger.info(f"Available kernels: {', '.join(kernels)}")
            return kernels[0]  # Return first available kernel
        else:
            logger.warning("No jupyter kernels found")
            return None
    except Exception as e:
        logger.warning(f"Error getting kernel list: {e}")
        
        # Try common kernel names
        for kernel in ['python3', 'python', 'pyenv', 'ipykernel']:
            try:
                output = subprocess.check_output(['jupyter', 'kernelspec', 'list', kernel], 
                                               stderr=subprocess.PIPE,
                                               universal_newlines=True)
                logger.info(f"Found kernel: {kernel}")
                return kernel
            except:
                pass
        
        return None

def run_manually(notebook_path):
    """Run notebook using subprocess instead of nbconvert"""
    logger.info(f"Running notebook manually: {notebook_path}")
    
    try:
        # Use jupyter nbconvert to execute the notebook
        command = [
            'jupyter', 'nbconvert', 
            '--to', 'notebook', 
            '--execute',
            '--output', os.path.splitext(os.path.basename(notebook_path))[0] + "_executed.ipynb",
            '--output-dir', os.path.dirname(notebook_path),
            notebook_path
        ]
        
        logger.info(f"Executing command: {' '.join(command)}")
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True)
        
        logger.info(output)
        return True, os.path.join(
            os.path.dirname(notebook_path),
            os.path.splitext(os.path.basename(notebook_path))[0] + "_executed.ipynb"
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with code {e.returncode}")
        logger.error(e.output)
        return False, None

def run_notebook(notebook_path, timeout=600, max_cells=None):
    """Run a notebook end-to-end"""
    start_time = time.time()
    logger.info(f"Running notebook: {notebook_path}")
    
    # Try to find an available kernel
    kernel_name = get_available_kernel()
    if not kernel_name:
        logger.warning("No available kernel found, will try manual execution")
        return run_manually(notebook_path)
    
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Limit the number of cells if specified
    if max_cells is not None:
        logger.info(f"Limiting execution to first {max_cells} code cells")
        code_cells = [i for i, cell in enumerate(nb.cells) if cell.cell_type == "code"]
        if len(code_cells) > max_cells:
            # Keep all cells up to the max_cells-th code cell
            max_cell_index = code_cells[max_cells-1]
            nb.cells = nb.cells[:max_cell_index+1]
    
    # Create output directory for notebook results
    output_dir = os.path.dirname(notebook_path)
    
    # Configure the execution environment
    ep = ExecutePreprocessor(timeout=timeout, kernel_name=kernel_name)
    
    try:
        # Execute the notebook
        logger.info(f"Starting execution with {kernel_name} kernel, timeout={timeout}s")
        ep.preprocess(nb, {'metadata': {'path': output_dir}})
        
        # Count cells with output
        cells_with_output = sum(1 for cell in nb.cells 
                               if cell.cell_type == 'code' and hasattr(cell, 'outputs') and len(cell.outputs) > 0)
        logger.info(f"Execution complete. {cells_with_output} cells have output.")
        
        # Save executed notebook
        executed_path = os.path.splitext(notebook_path)[0] + "_executed.ipynb"
        with open(executed_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        logger.info(f"Saved executed notebook to: {executed_path}")
        
        elapsed = time.time() - start_time
        logger.info(f"Notebook executed successfully in {timedelta(seconds=int(elapsed))}")
        
        return True, executed_path
        
    except Exception as e:
        # Save current state even if there was an error
        error_path = os.path.splitext(notebook_path)[0] + "_error.ipynb"
        with open(error_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        elapsed = time.time() - start_time
        logger.error(f"Error executing notebook after {timedelta(seconds=int(elapsed))}: {str(e)}")
        logger.info(f"Partial execution saved to: {error_path}")
        logger.info("Trying manual execution as fallback")
        
        # Try manual execution as fallback
        return run_manually(notebook_path)

def manage_disk_space():
    """Make sure we have enough disk space before running."""
    try:
        import shutil
        
        # Check for at least 1GB free space
        free_space = shutil.disk_usage("/").free
        free_space_gb = free_space / (1024 ** 3)
        
        if free_space_gb < 1.0:
            logger.warning(f"Low disk space: Only {free_space_gb:.1f} GB available")
            logger.info("Cleaning temporary directories...")
            
            # Clean temp directories to make space
            temp_dirs = ["/tmp"]
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    for item in os.listdir(temp_dir):
                        if item.startswith("tmp") or item.startswith("_MEI"):
                            try:
                                path = os.path.join(temp_dir, item)
                                if os.path.isfile(path):
                                    os.remove(path)
                                elif os.path.isdir(path):
                                    shutil.rmtree(path)
                            except Exception:
                                pass
            
            # Check again after cleaning
            free_space = shutil.disk_usage("/").free
            free_space_gb = free_space / (1024 ** 3)
            logger.info(f"After cleaning: {free_space_gb:.1f} GB available")
        
        logger.info(f"Disk space check passed: {free_space_gb:.1f} GB available")
        return True
    
    except Exception as e:
        logger.warning(f"Error checking disk space: {e}")
        logger.info("Continuing anyway...")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run NeuralPlasticityDemo.ipynb end-to-end')
    parser.add_argument('--notebook', type=str, default='colab_notebooks/NeuralPlasticityDemo.ipynb',
                       help='Path to the notebook')
    parser.add_argument('--timeout', type=int, default=1800,
                       help='Execution timeout in seconds')
    parser.add_argument('--test-mode', action='store_true',
                       help='Modify notebook for faster testing')
    parser.add_argument('--skip-fix', action='store_true',
                        help='Skip fixing visualization issues')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip notebook validation')
    parser.add_argument('--skip-run', action='store_true',
                        help='Skip notebook execution (only fix and validate)')
    parser.add_argument('--max-cells', type=int, default=None,
                        help='Maximum number of code cells to execute')
    
    args = parser.parse_args()
    
    # Check disk space
    manage_disk_space()
    
    # First clean the notebook of incompatible fields
    cleaned_path = os.path.splitext(args.notebook)[0] + "_cleaned.ipynb"
    logger.info(f"Cleaning notebook: {args.notebook} -> {cleaned_path}")
    nb = clean_notebook(args.notebook, cleaned_path)
    
    # Fix visualization issues if needed
    if not args.skip_fix:
        fixed_path = os.path.splitext(args.notebook)[0] + "_fixed.ipynb"
        if fix_notebook(cleaned_path, fixed_path):
            notebook_path = fixed_path
        else:
            logger.warning("Fixing failed, continuing with cleaned notebook")
            notebook_path = cleaned_path
    else:
        notebook_path = cleaned_path
    
    # Validate notebook if needed
    if not args.skip_validation and not args.skip_fix:
        if not validate_notebook(notebook_path):
            logger.warning("Validation failed, continuing anyway")
    
    # Modify for testing if requested
    if args.test_mode:
        # Modify notebook for testing
        test_path = os.path.splitext(args.notebook)[0] + "_test.ipynb"
        modify_notebook_for_testing(nb, test_path)
        notebook_path = test_path
    
    # Skip running if requested
    if args.skip_run:
        logger.info("Skipping notebook execution as requested")
        sys.exit(0)
    
    # Run the notebook
    success, output_path = run_notebook(notebook_path, args.timeout, args.max_cells)
    
    if success:
        logger.info("Notebook executed successfully!")
        sys.exit(0)
    else:
        logger.error("Notebook execution failed. See error details above.")
        sys.exit(1)

if __name__ == '__main__':
    main()