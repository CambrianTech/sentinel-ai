#!/usr/bin/env python
"""
Run Neural Plasticity Demo Notebook End-to-End

This script fixes the NeuralPlasticityDemo notebook and executes it end-to-end
to verify cross-platform compatibility.

Usage:
  python scripts/run_neural_plasticity_notebook_e2e.py [notebook_path] [output_path]
  
Default paths:
- Input: notebooks/NeuralPlasticityDemo.ipynb
- Output: test_output/NeuralPlasticityDemo_run.ipynb

Requirements:
  - nbformat: For fixing notebook (pip install nbformat)
  - jupyter: For executing notebook (pip install jupyter)
"""

import os
import sys
import platform
import subprocess
import datetime
import argparse

# Check for required dependencies
try:
    import nbformat
except ImportError:
    print("Error: This script requires nbformat.")
    print("Please install it with: pip install nbformat")
    sys.exit(1)

# Check if jupyter is available (needed for execution)
try:
    subprocess.run(["jupyter", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
except (subprocess.SubprocessError, FileNotFoundError):
    print("Warning: Jupyter not found in PATH.")
    print("You can install it with: pip install jupyter")
    print("Notebook fixing will still work, but execution will be skipped.")

def fix_and_run_notebook(input_path, output_path=None, run_notebook=True):
    """
    Fix and run the Neural Plasticity Demo notebook.
    
    Args:
        input_path: Path to the input notebook
        output_path: Path to save the executed notebook
        run_notebook: Whether to execute the notebook
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Determine the platform information
    is_apple_silicon = platform.system() == "Darwin" and platform.processor() == "arm"
    
    # Create output directory if it doesn't exist
    if output_path and os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Default output path if not provided
    if not output_path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("test_output", f"NeuralPlasticityDemo_run_{timestamp}.ipynb")
    
    # Step 1: Fix the notebook imports
    print(f"Step 1: Fixing dataset imports and cross-platform compatibility...")
    fixed_notebook_path = os.path.join(os.path.dirname(output_path), "NeuralPlasticityDemo_fixed.ipynb")
    
    try:
        # Import and run the fix script
        sys.path.insert(0, 'scripts')  # Ensure scripts directory is in path
        from fix_neural_plasticity_imports import fix_neural_plasticity_imports
        
        fix_success = fix_neural_plasticity_imports(input_path, fixed_notebook_path)
        if not fix_success:
            print("Failed to fix notebook imports")
            return False
        
        print(f"Successfully fixed notebook imports: {fixed_notebook_path}")
        
        # Step 2 (Optional): Run the notebook
        if run_notebook:
            print("\nStep 2: Running the notebook end-to-end...")
            print(f"Platform information: {'Apple Silicon' if is_apple_silicon else 'Standard Hardware'}")
            
            # Add environment variables for Apple Silicon
            env = os.environ.copy()
            if is_apple_silicon:
                env["OMP_NUM_THREADS"] = "1"
                env["OPENBLAS_NUM_THREADS"] = "1"
                env["MKL_NUM_THREADS"] = "1"
                env["VECLIB_MAXIMUM_THREADS"] = "1"
                env["NUMEXPR_NUM_THREADS"] = "1"
                env["ACCELERATE_USE_SYSTEM_BLAS"] = "1"
                env["PYTORCH_JIT_USE_AUTOTUNER"] = "0"
                print("Added environment variables for Apple Silicon stability")
            
            # Execute the notebook
            cmd = [
                "jupyter", "nbconvert", 
                "--to", "notebook", 
                "--execute",
                "--ExecutePreprocessor.timeout=600",  # 10 minutes timeout
                "--ExecutePreprocessor.kernel_name=python3",
                "--output", os.path.basename(output_path),
                "--output-dir", os.path.dirname(output_path),
                fixed_notebook_path
            ]
            
            print(f"Executing command: {' '.join(cmd)}")
            
            # Run the command
            try:
                process = subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    text=True
                )
                print(f"Notebook executed successfully and saved to {output_path}")
                print("\nOutput:")
                print(process.stdout)
                
                if process.stderr:
                    print("\nWarnings/Errors:")
                    print(process.stderr)
                
            except subprocess.CalledProcessError as e:
                print(f"Error executing notebook: {e}")
                print("\nOutput:")
                print(e.stdout)
                print("\nErrors:")
                print(e.stderr)
                return False
        
        print("\nProcess completed successfully!")
        return True
            
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix and run Neural Plasticity Demo notebook")
    parser.add_argument("--input", default="notebooks/NeuralPlasticityDemo.ipynb", help="Input notebook path")
    parser.add_argument("--output", default=None, help="Output notebook path")
    parser.add_argument("--fix-only", action="store_true", help="Only fix the notebook, don't run it")
    
    args = parser.parse_args()
    
    success = fix_and_run_notebook(
        input_path=args.input,
        output_path=args.output,
        run_notebook=not args.fix_only
    )
    
    sys.exit(0 if success else 1)