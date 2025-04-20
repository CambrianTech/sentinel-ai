#!/usr/bin/env python
"""
Fix Neural Plasticity Notebook Title Cell

This script completely rewrites the title cell of the Neural Plasticity notebook
to remove duplicate changelog entries and create a clean, updated title.

Usage:
  python scripts/fix_neural_plasticity_title.py
"""

import os
import nbformat
from datetime import datetime
import uuid

def fix_notebook_title(notebook_path):
    """Fix the title cell in the neural plasticity notebook."""
    print(f"Fixing title cell in: {notebook_path}")
    
    # Load the notebook
    with open(notebook_path, 'r') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Get current timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a new unique ID
    unique_id = str(uuid.uuid4())[:8]
    
    # Define the new title cell content - completely fresh
    new_title_content = f"""# Neural Plasticity Demo: Dynamic Pruning & Regrowth (v0.0.53 {current_time})

This notebook demonstrates Sentinel AI's neural plasticity system, which allows transformer models to dynamically prune and regrow attention heads during training based on utility metrics. [ID: {unique_id}]

### Changes in v0.0.53:
- Modularized neural plasticity code into reusable components
- Fixed GPU tensor handling for visualizations
- Cleaned up redundant tensor conversion patterns
- Improved numerical stability in entropy calculations
- Added unique ID for Colab cache busting
- Removed duplicate changelog entries

## What is Neural Plasticity?

Neural plasticity is the ability of neural networks to adapt their structure over time through pruning (removing unused connections) and regrowth (restoring useful connections). This mimics how biological brains form efficient neural pathways.

In this demo, we:
1. Track the entropy and gradient patterns of each attention head
2. Dynamically prune high-entropy, low-gradient heads (unfocused, less useful)
3. Selectively revive low-entropy, higher-gradient heads (potentially useful)
4. Visualize the "brain dynamics" over time

This allows models to form more efficient neural structures during training."""
    
    # Replace the first cell with our clean title
    if notebook.cells and notebook.cells[0].cell_type == 'markdown':
        notebook.cells[0].source = new_title_content
        print("Title cell has been completely replaced with a clean version")
    else:
        print("Error: First cell is not a markdown cell as expected")
        return None
    
    # Update unique ID in the appropriate cell
    for cell in notebook.cells:
        if cell.cell_type == 'code' and 'import torch' in cell.source and 'unique_id =' in cell.source:
            # Replace all unique_id lines with our new one, but keep only one
            lines = cell.source.split('\n')
            cleaned_lines = []
            unique_id_added = False
            
            for line in lines:
                if 'unique_id =' in line:
                    if not unique_id_added:
                        cleaned_lines.append(f'unique_id = "{unique_id}"')
                        unique_id_added = True
                    else:
                        # Comment out additional unique_id lines
                        cleaned_lines.append(f'# {line}')
                else:
                    cleaned_lines.append(line)
            
            cell.source = '\n'.join(cleaned_lines)
            print(f"Updated unique ID to: {unique_id}")
            break
    
    # Write the updated notebook
    with open(notebook_path, 'w') as f:
        nbformat.write(notebook, f)
    
    return unique_id

if __name__ == "__main__":
    # Path to the notebook
    notebook_path = os.path.join("colab_notebooks", "NeuralPlasticityDemo.ipynb")
    
    unique_id = fix_notebook_title(notebook_path)
    if unique_id:
        print(f"Successfully fixed notebook title and updated unique ID to: {unique_id}")
        print("These changes should be committed to the repository.")
    else:
        print("Failed to fix the notebook title.")