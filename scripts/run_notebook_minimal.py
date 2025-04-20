#!/usr/bin/env python
"""
Run Neural Plasticity Notebook with Minimal Settings

This script runs the Neural Plasticity notebook with very minimal settings
to avoid resource issues while still verifying end-to-end functionality.

Usage:
  python scripts/run_notebook_minimal.py
"""

import os
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path
import traceback

# Add repository root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

def modify_notebook_for_minimal_run(notebook):
    """Modify notebook to use minimal settings for a quick test run."""
    print("Modifying notebook for minimal run...")
    
    # Settings to modify
    replacements = {
        "NUM_EPOCHS = 100": "NUM_EPOCHS = 1",
        "BATCH_SIZE = 4": "BATCH_SIZE = 2",
        "MAX_LENGTH = 128": "MAX_LENGTH = 64",
        "MAX_STEPS_PER_EPOCH = None": "MAX_STEPS_PER_EPOCH = 3",
        "ENABLE_LONG_TRAINING = False": "ENABLE_LONG_TRAINING = False",
        "WARMUP_MAX_EPOCHS = 1": "WARMUP_MAX_EPOCHS = 1",
        "max_warmup_steps = 150": "max_warmup_steps = 3",
        "VISUALIZATION_INTERVAL = 100": "VISUALIZATION_INTERVAL = 2",
        "INFERENCE_INTERVAL = 500": "INFERENCE_INTERVAL = 5",
        "CHECKPOINT_INTERVAL = 500": "CHECKPOINT_INTERVAL = 10",
        "EVAL_INTERVAL = 50": "EVAL_INTERVAL = 2",
        "train_dataset = load_dataset(DATASET, DATASET_CONFIG, split=\"train\")": 
            "train_dataset = load_dataset(DATASET, DATASET_CONFIG, split=\"train[:100]\")",
        "validation_dataset = load_dataset(DATASET, DATASET_CONFIG, split=\"validation\")": 
            "validation_dataset = load_dataset(DATASET, DATASET_CONFIG, split=\"validation[:20]\")"
    }
    
    # Apply replacements
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            for old, new in replacements.items():
                cell.source = cell.source.replace(old, new)
    
    # Add timestamp and execution notice
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == 'markdown' and cell.source.startswith('# Neural Plasticity Demo'):
            # Already has a notice from our previous work
            break
    
    return notebook

def run_notebook_minimal():
    """Run the Neural Plasticity notebook with minimal settings."""
    print("Starting minimal notebook execution...")
    
    # Path to notebook
    notebook_path = os.path.join(repo_root, "colab_notebooks", "NeuralPlasticityDemo.ipynb")
    output_path = os.path.join(repo_root, "test_output", "NeuralPlasticityDemo_run.ipynb")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Load notebook
        with open(notebook_path) as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Modify notebook for minimal run
        notebook = modify_notebook_for_minimal_run(notebook)
        
        # Configure execution
        ep = ExecutePreprocessor(
            timeout=600,  # 10 minutes max execution time
            kernel_name='python_sentinel',  # Use our installed kernel
            store_widget_state=True
        )
        
        # Execute notebook
        print("Executing notebook (this may take a few minutes)...")
        ep.preprocess(notebook, {'metadata': {'path': repo_root}})
        
        # Save executed notebook
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(notebook, f)
        
        print(f"Notebook execution completed successfully")
        print(f"Executed notebook saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error executing notebook: {e}")
        traceback.print_exc()
        
        # Try to save the partially executed notebook
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                nbformat.write(notebook, f)
            print(f"Partially executed notebook saved to: {output_path}")
        except:
            print("Could not save partially executed notebook")
            
        return False

if __name__ == "__main__":
    success = run_notebook_minimal()
    sys.exit(0 if success else 1)