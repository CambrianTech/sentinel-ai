#!/usr/bin/env python
"""
Run the minimal Neural Plasticity test

This script creates and optionally runs a minimal test notebook for neural plasticity.
It serves as a quick validation test for the neural plasticity functionality.

Version: v0.0.60 (2025-04-20)
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def run_notebook(notebook_path, output_path=None):
    """
    Run a notebook via jupyter nbconvert.
    
    Args:
        notebook_path: Path to the notebook to run
        output_path: Path to save the executed notebook (default: add _executed suffix)
        
    Returns:
        Path to the executed notebook
    """
    if output_path is None:
        # Add _executed suffix to the notebook path
        notebook_name = Path(notebook_path).stem
        output_path = Path(notebook_path).parent / f"{notebook_name}_executed.ipynb"
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Run the notebook
    print(f"Running notebook: {notebook_path}")
    print(f"Output will be saved to: {output_path}")
    
    cmd = [
        "jupyter", "nbconvert", 
        "--to", "notebook", 
        "--execute",
        "--ExecutePreprocessor.timeout=600",  # 10 minute timeout
        "--output", str(output_path),
        str(notebook_path)
    ]
    
    try:
        # Run the notebook
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ Successfully ran notebook: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running notebook: {e}")
        print(f"Stdout: {e.stdout.decode('utf-8')}")
        print(f"Stderr: {e.stderr.decode('utf-8')}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Run the minimal neural plasticity test")
    parser.add_argument("--run", action="store_true", help="Run the notebook after creating it")
    parser.add_argument("--output", type=str, default=None, help="Output path for the notebook")
    
    args = parser.parse_args()
    
    # Create the minimal test notebook
    try:
        # Import the create_minimal_test function
        from scripts.create_minimal_test import create_minimal_test
        
        # Create the test notebook
        notebook_path = create_minimal_test(output_path=args.output)
        
        if args.run:
            # Run the notebook
            run_path = run_notebook(notebook_path)
            print(f"Neural plasticity minimal test complete: {run_path}")
            print("\nTo view the executed notebook:")
            print(f"jupyter notebook {run_path}")
        else:
            print("\nTest notebook created but not executed. To run the notebook:")
            print(f"jupyter notebook {notebook_path}")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())