#!/usr/bin/env python
"""
Test Neural Plasticity Notebook after modularization.

This script validates the updated NeuralPlasticityDemo.ipynb notebook to ensure:
1. The notebook loads correctly
2. Imports for modularized components work
3. Unique ID for cache busting is set properly

Usage:
  python scripts/test_neural_plasticity_notebook.py
"""

import os
import sys
import json
import nbformat
from pathlib import Path

# Add repository root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

def validate_notebook(notebook_path):
    """Validate the neural plasticity notebook."""
    print(f"Validating notebook: {notebook_path}")
    
    # Load the notebook
    try:
        with open(notebook_path, 'r') as f:
            notebook = nbformat.read(f, as_version=4)
        print("✅ Notebook loads correctly")
    except Exception as e:
        print(f"❌ Failed to load notebook: {e}")
        return False
    
    # Check for modularized imports
    module_imports_found = False
    neural_plasticity_imports_found = False
    
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            if 'from utils.neural_plasticity import NeuralPlasticity' in cell.source:
                neural_plasticity_imports_found = True
                module_imports_found = True
                print("✅ Neural plasticity API imports found")
            if 'from utils.neural_plasticity.core import' in cell.source:
                neural_plasticity_imports_found = True
                module_imports_found = True
                print("✅ Neural plasticity core module imports found")
            if 'from utils.neural_plasticity.visualization import' in cell.source:
                module_imports_found = True
                print("✅ Neural plasticity visualization module imports found")
            if 'from utils.neural_plasticity.training import' in cell.source:
                module_imports_found = True
                print("✅ Neural plasticity training module imports found")
    
    if not module_imports_found:
        print("❌ Modularized imports not found")
        return False
        
    if not neural_plasticity_imports_found:
        print("❌ Neural plasticity core imports not found")
        return False
    
    # Check for unique ID
    unique_id_found = False
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            if 'unique_id = ' in cell.source and 'print(f"Running modularized neural plasticity code [ID: {unique_id}]")' in cell.source:
                unique_id_found = True
                unique_id_value = None
                for line in cell.source.split('\n'):
                    if line.strip().startswith('unique_id = '):
                        try:
                            unique_id_value = line.split('=')[1].strip().strip('"\'')
                            print(f"✅ Unique ID found: {unique_id_value}")
                        except:
                            print("⚠️ Unique ID found but couldn't extract value")
                break
    
    if not unique_id_found:
        print("❌ Unique ID for cache busting not found")
        return False
    
    # Check for duplicate tensor conversion
    duplicate_conversions = False
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            if '.detach().cpu().numpy().cpu().numpy()' in cell.source:
                print(f"❌ Found duplicate tensor conversion in cell {i}")
                duplicate_conversions = True
            if '.cpu().numpy().cpu().numpy()' in cell.source:
                print(f"❌ Found duplicate tensor conversion in cell {i}")
                duplicate_conversions = True
    
    if duplicate_conversions:
        print("❌ Duplicate tensor conversions found in notebook")
        return False
    else:
        print("✅ No duplicate tensor conversions found")
        
    # Check for key modular API usage
    key_functions = [
        "NeuralPlasticity.run_warmup_training",
        "NeuralPlasticity.diagnose_attention_patterns", 
        "NeuralPlasticity.calculate_head_importance",
        "NeuralPlasticity.create_gradient_pruning_mask"
    ]
    
    found_functions = set()
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            for function in key_functions:
                if function in cell.source:
                    found_functions.add(function)
                    print(f"✅ Found modular API function: {function}")
    
    missing_functions = set(key_functions) - found_functions
    if missing_functions:
        print(f"❌ Missing key modular API functions: {missing_functions}")
        return False
    else:
        print("✅ All key modular API functions found")
    
    print("\n===== VALIDATION SUMMARY =====")
    print("✅ Notebook loads correctly")
    print("✅ Modularized imports found")
    print("✅ Unique ID for cache busting found")
    print("✅ No duplicate tensor conversions found")
    print("✅ NOTEBOOK IS READY FOR COLAB TESTING")
    
    return True

if __name__ == "__main__":
    # Path to the notebook
    notebook_path = os.path.join(repo_root, "colab_notebooks", "NeuralPlasticityDemo.ipynb")
    
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
        sys.exit(1)
    
    success = validate_notebook(notebook_path)
    sys.exit(0 if success else 1)