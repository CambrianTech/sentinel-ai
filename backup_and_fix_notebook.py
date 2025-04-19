#!/usr/bin/env python
"""
Backup and fix NeuralPlasticityDemo.ipynb

This script creates a backup of the original notebook,
then applies the fixes using nbformat as per CLAUDE.md guidelines.
"""

import os
import sys
import shutil
import datetime
from fix_neural_plasticity_demo import fix_notebook

def backup_and_fix(notebook_path):
    """
    Create a backup of the notebook and then fix it
    
    Args:
        notebook_path: Path to the notebook
    """
    # Create backup filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    notebook_dir = os.path.dirname(notebook_path)
    notebook_name = os.path.basename(notebook_path)
    backup_name = f"{os.path.splitext(notebook_name)[0]}_backup_{timestamp}.ipynb"
    backup_path = os.path.join(notebook_dir, backup_name)
    
    # Create backup
    print(f"Creating backup: {backup_path}")
    shutil.copy2(notebook_path, backup_path)
    
    # Fix the notebook
    print(f"Fixing notebook: {notebook_path}")
    fixes_applied = fix_notebook(notebook_path)
    
    # Print summary
    print("\nBackup and fix completed:")
    print(f"- Original notebook backed up to: {backup_path}")
    print(f"- Fixed notebook saved to: {notebook_path}")
    print("\nFixes applied:")
    for fix, count in fixes_applied.items():
        print(f"- {fix}: {count} instances")


if __name__ == "__main__":
    # Get notebook path from command line or use default
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else "colab_notebooks/NeuralPlasticityDemo.ipynb"
    
    # Backup and fix
    backup_and_fix(notebook_path)