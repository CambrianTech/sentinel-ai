#!/usr/bin/env python
"""
Validate the NeuralPlasticityDemo.ipynb notebook after fixes

This script validates the notebook structure and checks for common issues.
"""

import nbformat
import sys
import re

def validate_notebook(notebook_path):
    """
    Validate a Jupyter notebook for common issues.
    
    Args:
        notebook_path: Path to the notebook
    
    Returns:
        Dictionary with validation results
    """
    print(f"Validating notebook: {notebook_path}")
    nb = nbformat.read(notebook_path, as_version=4)
    
    issues = {
        "detach_cpu_errors": [],
        "clim_errors": [],
        "undefined_monitor": [],
        "other_issues": []
    }
    
    # Check each cell for issues
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            # Check for detach/cpu issues
            if re.search(r"\.detach\(\.detach\(\)\.cpu\(\)\.numpy\(\)\)", cell.source):
                issues["detach_cpu_errors"].append(i)
                
            if re.search(r"\.cpu\(\)\.numpy\(\)\.cpu\(\)\.numpy\(\)", cell.source):
                issues["detach_cpu_errors"].append(i)
            
            # Check for plt.clim issues
            if "plt.clim(0, 1.0)  # Ensure proper scaling for attention values.numpy()" in cell.source:
                issues["clim_errors"].append(i)
                
            if "plt.clim(0, 1.0)  # Ensure proper scaling for attention  # Ensure proper scale" in cell.source:
                issues["clim_errors"].append(i)
            
            # Check for monitor usage without definition
            if "pruning_monitor.update" in cell.source and not any("pruning_monitor =" in c.source for c in nb.cells if c.cell_type == "code" and nb.cells.index(c) < i):
                issues["undefined_monitor"].append(i)
            
            # Check for (removed) text
            if "(removed)" in cell.source:
                issues["other_issues"].append(f"Cell {i} contains '(removed)' text")
    
    # Print validation report
    print("\nValidation Results:")
    all_clean = True
    
    for issue_type, instances in issues.items():
        if isinstance(instances, list) and instances:
            all_clean = False
            if isinstance(instances[0], int):
                print(f"- {issue_type}: Found in cells {', '.join(map(str, instances))}")
            else:
                print(f"- {issue_type}: {len(instances)} issues")
                for issue in instances:
                    print(f"  * {issue}")
    
    if all_clean:
        print("✅ No issues found! Notebook looks good.")
    else:
        print("❌ Issues found in the notebook. Please fix them.")
        
    return issues


if __name__ == "__main__":
    # Get notebook path from command line or use default
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else "colab_notebooks/NeuralPlasticityDemo.ipynb"
    
    # Validate the notebook
    validate_notebook(notebook_path)