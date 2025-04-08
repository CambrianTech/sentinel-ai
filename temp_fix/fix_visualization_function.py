#!/usr/bin/env python
# Fix visualization function error in NeuralPlasticityDemo.ipynb

import json
import re
from pathlib import Path

def fix_visualization_function(notebook_path):
    """
    Fixes the missing visualization function definition that causes a runtime error.
    The error occurs in the training loop (cell 31) where the visualize_gradient_norms
    function is called but not defined there.
    """
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        print("Fixing visualization function definition...")
        
        # Find the training cell (where the error occurs)
        training_cell_idx = None
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
                if 'for epoch in range(NUM_EPOCHS):' in source and 'visualize_gradient_norms' in source:
                    training_cell_idx = i
                    print(f"Found training loop cell at index {i}")
                    break
        
        if training_cell_idx is None:
            print("Could not find training loop cell")
            return False
        
        # Get the function definition from the notebook (if it exists elsewhere)
        visualization_function = """
# Define visualization function before using it
def visualize_gradient_norms(grad_norm_values, pruned_heads=None, revived_heads=None, title="Gradient Norms", save_path=None):
    \"\"\"Create a visualization of gradient norms with markers for pruned/revived heads\"\"\"
    plt.figure(figsize=(10, 5))
    plt.imshow(grad_norm_values.detach().cpu().numpy(), cmap="plasma", aspect="auto")
    plt.colorbar(label="Gradient Norm")
        
    # Mark pruned heads with 'P'
    if pruned_heads:
        for layer, head in pruned_heads:
            plt.text(head, layer, "P", ha="center", va="center",
                      color="white", weight="bold", bbox=dict(facecolor='red', alpha=0.5))
        
    # Mark revived heads with 'R'
    if revived_heads:
        for layer, head in revived_heads:
            plt.text(head, layer, "R", ha="center", va="center",
                      color="white", weight="bold", bbox=dict(facecolor='green', alpha=0.5))
        
    plt.title(title)
    plt.xlabel("Head Index")
    plt.ylabel("Layer Index")
    plt.tight_layout()
        
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
    return plt.gcf()
"""
        
        # Get the training cell content
        cell_content = ''.join(notebook['cells'][training_cell_idx]['source']) if isinstance(notebook['cells'][training_cell_idx]['source'], list) else notebook['cells'][training_cell_idx]['source']
        
        # Check if the function is already defined
        if 'def visualize_gradient_norms' not in cell_content:
            # Add the function definition after the controller check part
            controller_check_part = "# Check if controller exists"
            parts = cell_content.split(controller_check_part)
            if len(parts) == 2:
                # Find where the controller check block ends
                end_of_check = parts[1].find('\n\ntry:')
                if end_of_check != -1:
                    # Insert the function definition between the controller check and the training loop
                    updated_content = parts[0] + controller_check_part + parts[1][:end_of_check] + "\n" + visualization_function + parts[1][end_of_check:]
                    
                    # Update the cell
                    if isinstance(notebook['cells'][training_cell_idx]['source'], list):
                        # Split into lines and add newlines
                        updated_lines = []
                        for line in updated_content.split('\n'):
                            updated_lines.append(line + '\n')
                        if updated_lines:
                            updated_lines[-1] = updated_lines[-1].rstrip('\n')
                        notebook['cells'][training_cell_idx]['source'] = updated_lines
                    else:
                        notebook['cells'][training_cell_idx]['source'] = updated_content
                    
                    print(f"Added visualization function definition to cell {training_cell_idx}")
                    
                    # Also update version number in first cell if not already done
                    if notebook['cells'][0]['cell_type'] == 'markdown':
                        # Get first cell content
                        first_cell = ''.join(notebook['cells'][0]['source']) if isinstance(notebook['cells'][0]['source'], list) else notebook['cells'][0]['source']
                        
                        # Check if version is already updated
                        if not "v0.0.35" in first_cell:
                            # Update version number and add changelog entry
                            first_cell = first_cell.replace("v0.0.34", "v0.0.35")
                            
                            # Add entry for v0.0.35 if not already there
                            if "### New in v0.0.35:" not in first_cell:
                                # Find position to insert (after "This allows models...")
                                split_text = "This allows models to form more efficient neural structures during training."
                                parts = first_cell.split(split_text)
                                if len(parts) > 1:
                                    new_entry = """

### New in v0.0.35:
- Fixed runtime error in visualization code
- Moved function definition before usage
- Ensured proper gradient visualization during training
"""
                                    first_cell = parts[0] + split_text + new_entry + parts[1].lstrip()
                                    
                                    # Update the cell
                                    if isinstance(notebook['cells'][0]['source'], list):
                                        # Split into lines and add newlines
                                        first_cell_lines = []
                                        for line in first_cell.split('\n'):
                                            first_cell_lines.append(line + '\n')
                                        if first_cell_lines:
                                            first_cell_lines[-1] = first_cell_lines[-1].rstrip('\n')
                                        notebook['cells'][0]['source'] = first_cell_lines
                                    else:
                                        notebook['cells'][0]['source'] = first_cell
                                    
                                    print("Updated version to v0.0.35 and added changelog entry")
                        else:
                            print("Version already updated to v0.0.35")
                else:
                    print("Could not find where to insert the function definition")
                    return False
            else:
                print("Could not find controller check part in the cell")
                return False
        else:
            print("Function is already defined in the cell")
        
        # Save the notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print("Successfully fixed visualization function definition error")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_visualization_function(notebook_path)