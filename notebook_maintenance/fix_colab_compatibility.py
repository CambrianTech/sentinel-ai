#!/usr/bin/env python
"""
Fix Colab compatibility issues in NeuralPlasticityDemo.ipynb

This script applies fixes for Colab compatibility and best practices:
1. Adds %matplotlib inline to ensure proper plotting
2. Adds missing system dependency checks
3. Adds better error handling for model training
4. Deduplicates import statements
"""

import nbformat
import os
import sys
import re
from typing import Dict, List, Any


def fix_colab_compatibility(notebook_path: str, output_path: str = None) -> Dict[str, int]:
    """
    Fix Colab compatibility issues in the notebook
    
    Args:
        notebook_path: Path to the input notebook
        output_path: Path to save the fixed notebook (defaults to overwriting input)
        
    Returns:
        Dictionary with counts of fixes applied
    """
    print(f"Reading notebook: {notebook_path}")
    nb = nbformat.read(notebook_path, as_version=4)
    
    fixes_applied = {
        "added_matplotlib_inline": 0,
        "added_system_dependency_check": 0,
        "improved_error_handling": 0,
        "deduplicated_imports": 0,
        "added_cell_execution_counts": 0
    }
    
    # 1. Fix matplotlib display for Colab
    matplotlib_fixed = False
    for cell in nb.cells:
        if cell.cell_type == "code" and "%matplotlib inline" in cell.source:
            matplotlib_fixed = True
            break
    
    if not matplotlib_fixed:
        # Find the first import of matplotlib
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == "code" and "import matplotlib" in cell.source:
                # Add inline magic
                if not "%matplotlib inline" in cell.source:
                    cell.source = "%matplotlib inline\n" + cell.source
                    fixes_applied["added_matplotlib_inline"] += 1
                    matplotlib_fixed = True
                    break
        
        # If no matplotlib import found, add to the first code cell
        if not matplotlib_fixed:
            for i, cell in enumerate(nb.cells):
                if cell.cell_type == "code" and not cell.source.startswith("!"):
                    cell.source = "%matplotlib inline\n" + cell.source
                    fixes_applied["added_matplotlib_inline"] += 1
                    break
    
    # 2. Add system dependency check
    system_deps_added = False
    for cell in nb.cells:
        if cell.cell_type == "code" and "apt-get install" in cell.source:
            system_deps_added = True
            break
    
    if not system_deps_added:
        # Find the cell with pip install or repo cloning
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == "code" and ("pip install" in cell.source or "git clone" in cell.source):
                # Insert system dependency check before this cell
                system_deps_cell = nbformat.v4.new_code_cell(
                    "# Check and install system dependencies if needed\n"
                    "!apt-get update -qq > /dev/null\n"
                    "!apt-get install -qq libopenblas-dev > /dev/null  # For better performance"
                )
                nb.cells.insert(i, system_deps_cell)
                fixes_applied["added_system_dependency_check"] += 1
                break
    
    # 3. Improve error handling in the training loop
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and "for epoch in range(" in cell.source and "try:" not in cell.source:
            # Add error handling to the training loop
            cell_lines = cell.source.split('\n')
            new_lines = []
            indent = ""
            
            # Find the training loop
            for j, line in enumerate(cell_lines):
                if "for epoch in range(" in line:
                    new_lines.append("try:")
                    new_lines.append(f"    {line}")
                    indent = "    "
                    # Add all subsequent lines with increased indentation
                    for k in range(j+1, len(cell_lines)):
                        new_lines.append(f"    {cell_lines[k]}")
                    
                    # Add except block
                    new_lines.append("\nexcept Exception as e:")
                    new_lines.append("    print(f\"\\nError during training: {e}\")")
                    new_lines.append("    # Try to save checkpoint on error")
                    new_lines.append("    try:")
                    new_lines.append("        error_checkpoint_path = save_checkpoint(global_step, epoch + 1)")
                    new_lines.append("        print(f\"Checkpoint saved at {error_checkpoint_path}\")")
                    new_lines.append("    except Exception as save_error:")
                    new_lines.append("        print(f\"Could not save checkpoint: {save_error}\")")
                    
                    # Set new cell source and break out of loop
                    cell.source = '\n'.join(new_lines)
                    fixes_applied["improved_error_handling"] += 1
                    break
                else:
                    new_lines.append(line)
    
    # 4. Deduplicate import statements
    imports_seen = set()
    for cell in nb.cells:
        if cell.cell_type == "code":
            # Process each line
            new_lines = []
            removed_lines = 0
            
            for line in cell.source.split('\n'):
                line_stripped = line.strip()
                # Check if it's an import
                if line_stripped.startswith(('import ', 'from ')):
                    if line_stripped in imports_seen:
                        removed_lines += 1
                        continue
                    imports_seen.add(line_stripped)
                
                new_lines.append(line)
            
            if removed_lines > 0:
                cell.source = '\n'.join(new_lines)
                fixes_applied["deduplicated_imports"] += removed_lines
    
    # 5. Add execution counts for cells that are missing them
    next_execution_count = 1
    for cell in nb.cells:
        if cell.cell_type == "code":
            if not hasattr(cell, 'execution_count') or cell.execution_count is None:
                cell.execution_count = next_execution_count
                fixes_applied["added_cell_execution_counts"] += 1
                next_execution_count += 1
            else:
                next_execution_count = max(next_execution_count, cell.execution_count + 1)
    
    # Save the fixed notebook
    output_file = output_path or notebook_path
    print(f"Saving fixed notebook to: {output_file}")
    nbformat.write(nb, output_file)
    
    # Print summary of fixes
    print("\nFixes applied:")
    for fix, count in fixes_applied.items():
        print(f"- {fix}: {count} instances")
    
    return fixes_applied


if __name__ == "__main__":
    # Get notebook path from command line or use default
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else "colab_notebooks/NeuralPlasticityDemo.ipynb"
    
    # Option to specify output path as second argument
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Apply fixes
    fix_colab_compatibility(notebook_path, output_path)