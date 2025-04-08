#!/usr/bin/env python
# Add safety checks for controller variable in NeuralPlasticityDemo.ipynb

import json
from pathlib import Path

def add_controller_safety_check(notebook_path):
    """Add explicit checks for controller variable to prevent NameError."""
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Find cells that use controller
        cells_using_controller = []
        
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                cell_content = ''.join(cell["source"])
                
                if "controller." in cell_content and "controller = " not in cell_content:
                    cells_using_controller.append(i)
                    print(f"Found cell using controller at index {i}")
        
        # Add safety check to each cell
        for i in cells_using_controller:
            cell = notebook["cells"][i]
            lines = cell["source"]
            
            # Add safety check at the beginning of the cell
            safety_check = """# Check if controller exists
try:
    controller
except NameError:
    print("ERROR: The controller variable is not defined. Please run the cell that creates the plasticity controller first.")
    # Create a simple dummy controller to avoid breaking the notebook flow
    from types import SimpleNamespace
    controller = SimpleNamespace()
    controller.collect_head_metrics = lambda *args, **kwargs: (None, None)
    controller.display_stats = lambda *args, **kwargs: None
    controller.stats = {}
    controller.total_layers = 0
    controller.heads_per_layer = 0

"""
            
            # Only add if not already present
            if "try:\n    controller\nexcept NameError:" not in ''.join(lines):
                # Insert at the beginning but after any comments or docstrings
                insert_pos = 0
                for j, line in enumerate(lines):
                    if not line.strip().startswith("#") and line.strip():
                        insert_pos = j
                        break
                
                # Split the safety check into lines
                safety_lines = safety_check.split('\n')
                safety_lines = [line + '\n' for line in safety_lines]
                
                # Insert the safety lines
                for j, line in enumerate(safety_lines):
                    lines.insert(insert_pos + j, line)
                
                # Update the cell source
                cell["source"] = lines
                print(f"Added safety check to cell {i}")
        
        # Save the modified notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print(f"Notebook saved with controller safety checks")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    add_controller_safety_check(notebook_path)