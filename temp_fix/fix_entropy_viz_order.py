#!/usr/bin/env python
# Fix order of operations in entropy visualization

import json
import re
from pathlib import Path

def fix_entropy_viz_order(notebook_path):
    """Fix the order of imshow and clim calls in entropy visualization."""
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Find the cell with the entropy visualization
        entropy_viz_cell_idx = None
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                content = ''.join(cell['source'])
                if 'plt.title("Initial Head Entropy' in content and 'plt.clim' in content:
                    entropy_viz_cell_idx = i
                    print(f"Found entropy visualization cell at index {i}")
                    break
        
        if entropy_viz_cell_idx is None:
            print("ERROR: Could not find entropy visualization cell")
            return False
        
        # Get the cell content
        cell = notebook['cells'][entropy_viz_cell_idx]
        content = ''.join(cell['source'])
        
        # Extract and print the problematic section
        clim_line = re.search(r'plt\.clim\([^)]+\)', content).group(0)
        imshow_line = re.search(r'entropy_map = plt\.imshow\([^)]+\)', content).group(0)
        print(f"Current order: \n1. {clim_line}\n2. {imshow_line}")
        
        # Fix the order
        fixed_content = content.replace(
            "plt.title(\"Initial Head Entropy (higher = less focused attention)\")\n# Ensure entropy visualization has some range\nplt.clim(0, max(0.1, entropy_values.max().item()))\nentropy_map = plt.imshow",
            "plt.title(\"Initial Head Entropy (higher = less focused attention)\")\nentropy_map = plt.imshow"
        )
        
        # Add the clim call after imshow
        fixed_content = fixed_content.replace(
            "entropy_map = plt.imshow(entropy_values.detach().cpu().numpy(), cmap=\"viridis\", aspect=\"auto\")\nplt.colorbar",
            "entropy_map = plt.imshow(entropy_values.detach().cpu().numpy(), cmap=\"viridis\", aspect=\"auto\")\n# Ensure entropy visualization has some range\nplt.clim(0, max(0.1, entropy_values.max().item()))\nplt.colorbar"
        )
        
        # Update the cell
        cell['source'] = fixed_content.split('\n')
        
        # Save the notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print("Fixed order of imshow and clim calls")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_entropy_viz_order(notebook_path)