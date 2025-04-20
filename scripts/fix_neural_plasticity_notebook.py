#!/usr/bin/env python
"""
Fix Neural Plasticity Notebook Issues

This script fixes specific issues in the NeuralPlasticityDemo.ipynb notebook:
1. Cleans up the title cell to remove duplicate changelog entries
2. Fixes redundant .cpu().numpy() calls
3. Corrects any tensor visualization issues

Usage:
  python scripts/fix_neural_plasticity_notebook.py
"""

import os
import json
import nbformat
from datetime import datetime
import re
import uuid

def fix_notebook_issues(notebook_path, output_path=None):
    """
    Fix specific issues in the neural plasticity notebook.
    
    Args:
        notebook_path: Path to the notebook
        output_path: Path to save the fixed notebook (defaults to original path)
    """
    # Read notebook
    with open(notebook_path, 'r') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # If no output path provided, modify in-place
    if output_path is None:
        output_path = notebook_path
    
    # Create unique identifier to bypass cache
    unique_id = str(uuid.uuid4())[:8]
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Fix version in title cell
    try:
        title_cell = notebook.cells[0]
        if title_cell.cell_type == 'markdown' and title_cell.source.startswith('# Neural Plasticity Demo'):
            # Extract current version
            version_match = re.search(r'v(\d+\.\d+\.\d+)', title_cell.source)
            if version_match:
                current_version = version_match.group(1)
                # Increment patch version
                version_parts = current_version.split('.')
                new_version = f"{version_parts[0]}.{version_parts[1]}.{int(version_parts[2]) + 1}"
                
                # Create completely fresh title cell with only version, date and unique ID
                new_content = f"""# Neural Plasticity Demo: Dynamic Pruning & Regrowth (v{new_version} {current_time})

This notebook demonstrates Sentinel AI's neural plasticity system, which allows transformer models to dynamically prune and regrow attention heads during training based on utility metrics. [ID: {unique_id}]

### Changes in v{new_version}:
- Modularized notebook components for better reusability
- Integrated with utils.neural_plasticity module
- Improved tensor handling with safe_tensor_imshow
- Enhanced visualization utilities
- Added type hints and documentation
- Fixed tensor handling for GPU compatibility
- Fixed duplicate changelog entries

## What is Neural Plasticity?

Neural plasticity is the ability of neural networks to adapt their structure over time through pruning (removing unused connections) and regrowth (restoring useful connections). This mimics how biological brains form efficient neural pathways.

In this demo, we:
1. Track the entropy and gradient patterns of each attention head
2. Dynamically prune high-entropy, low-gradient heads (unfocused, less useful)
3. Selectively revive low-entropy, higher-gradient heads (potentially useful)
4. Visualize the "brain dynamics" over time

This allows models to form more efficient neural structures during training."""
                
                # Replace the entire cell content
                title_cell.source = new_content
    except Exception as e:
        print(f"Error updating title: {e}")
    
    # Fix tensor handling issues in all code cells
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            # Fix multiple .cpu().numpy() calls
            cell.source = cell.source.replace('.detach().cpu().numpy().cpu().numpy()', '.detach().cpu().numpy()')
            cell.source = cell.source.replace('.cpu().numpy().cpu().numpy()', '.cpu().numpy()')
            cell.source = cell.source.replace('.cpu().numpy())).cpu().numpy()', '.cpu().numpy())')
            cell.source = cell.source.replace('.cpu(.detach().cpu().numpy())', '.detach().cpu().numpy()')
            cell.source = cell.source.replace('.cpu().numpy()))))', '.cpu().numpy())')
            cell.source = cell.source.replace('im1 = plt.imshow(all_entropies.cpu(.detach().cpu().numpy()))', 'im1 = plt.imshow(all_entropies.detach().cpu().numpy())')
    
    # Update unique_id reference in import cell
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code' and 'import torch' in cell.source and 'print(f"Running modularized neural plasticity code [ID:' in cell.source:
            cell.source = cell.source.replace('print(f"Running modularized neural plasticity code [ID: {unique_id}]")', 
                                             f'# Define unique ID for cache busting\nunique_id = "{unique_id}"\nprint(f"Running modularized neural plasticity code [ID: {{unique_id}}]")')
            break
    
    # Write fixed notebook
    print(f"Writing fixed notebook to {output_path}")
    with open(output_path, 'w') as f:
        nbformat.write(notebook, f)
    
    return unique_id

if __name__ == "__main__":
    # Path to the notebook
    notebook_path = os.path.join("colab_notebooks", "NeuralPlasticityDemo.ipynb")
    
    # Make backup
    backup_path = os.path.splitext(notebook_path)[0] + f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"
    import shutil
    shutil.copy2(notebook_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Fix notebook issues
    unique_id = fix_notebook_issues(notebook_path)
    print(f"Notebook fixes applied successfully with unique ID: {unique_id}")
    print("You can verify you're running the updated version by checking for this ID in the notebook title and logs.")