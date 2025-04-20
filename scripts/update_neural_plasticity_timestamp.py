#!/usr/bin/env python
"""
Update Neural Plasticity Notebook Timestamp

This script updates the timestamp in the Neural Plasticity notebook title cell
to reflect the current time.

Usage:
  python scripts/update_neural_plasticity_timestamp.py
"""

import os
import re
import nbformat
from datetime import datetime
import uuid

def update_notebook_timestamp(notebook_path):
    """Update the timestamp in the notebook title cell."""
    print(f"Updating timestamp in: {notebook_path}")
    
    # Load the notebook
    with open(notebook_path, 'r') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Get current timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a new unique ID
    unique_id = str(uuid.uuid4())[:8]
    current_version = "0.0.53"  # Default increment
    
    # Update the title cell timestamp
    title_cell = notebook.cells[0]
    if title_cell.cell_type == 'markdown' and title_cell.source.startswith('# Neural Plasticity Demo'):
        # Extract current version
        version_match = re.search(r'v(\d+\.\d+\.\d+)', title_cell.source)
        if version_match:
            version_parts = version_match.group(1).split('.')
            # Increment patch version
            new_patch = int(version_parts[2]) + 1
            current_version = f"{version_parts[0]}.{version_parts[1]}.{new_patch}"
        
        # Update title line with new timestamp
        title_cell.source = re.sub(
            r'# Neural Plasticity Demo: Dynamic Pruning & Regrowth \(v.*?\)',
            f'# Neural Plasticity Demo: Dynamic Pruning & Regrowth (v{current_version} {current_time})',
            title_cell.source
        )
        
        print(f"Updated version to v{current_version} and timestamp to {current_time}")
    
    # Update unique ID in the appropriate cell
    for cell in notebook.cells:
        if cell.cell_type == 'code' and 'import torch' in cell.source:
            if 'unique_id =' in cell.source:
                # Replace all existing unique_id assignments with the new one
                cell.source = re.sub(
                    r'unique_id = ".*?"',
                    f'unique_id = "{unique_id}"',
                    cell.source
                )
                # Make sure we have only one unique_id line
                lines = cell.source.split('\n')
                unique_id_lines = [i for i, line in enumerate(lines) if 'unique_id =' in line]
                if len(unique_id_lines) > 1:
                    # Keep only the last one
                    for i in unique_id_lines[:-1]:
                        lines[i] = lines[i].replace('unique_id =', '# Old unique_id =')
                    cell.source = '\n'.join(lines)
                print(f"Updated unique ID to: {unique_id}")
                break
    
    # Write the updated notebook
    with open(notebook_path, 'w') as f:
        nbformat.write(notebook, f)
    
    return current_version, current_time, unique_id

if __name__ == "__main__":
    # Path to the notebook
    notebook_path = os.path.join("colab_notebooks", "NeuralPlasticityDemo.ipynb")
    
    version, timestamp, unique_id = update_notebook_timestamp(notebook_path)
    print(f"Notebook updated with version v{version}, timestamp {timestamp}, and unique ID {unique_id}")
    print("These changes should be committed to the repository.")
    print("After pushing, the notebook can be tested in Colab with a T4 GPU.")