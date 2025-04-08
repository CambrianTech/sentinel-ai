#!/usr/bin/env python
# Fix controller execution order in NeuralPlasticityDemo.ipynb

import json
from pathlib import Path

def fix_controller_execution(notebook_path):
    """Ensure controller is defined before debugging cells try to use it."""
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Find the controller cell and debug cell
        controller_cell_idx = None
        debug_cell_idx = None
        
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                cell_content = ''.join(cell["source"])
                
                if "controller = create_plasticity_controller(" in cell_content:
                    controller_cell_idx = i
                    print(f"Found controller creation cell at index {i}")
                
                if "debug_entropy, debug_grads = controller.collect_head_metrics(" in cell_content:
                    debug_cell_idx = i
                    print(f"Found debug metrics cell at index {i}")
        
        if controller_cell_idx is None:
            print("ERROR: Couldn't find controller creation cell")
            return False
            
        if debug_cell_idx is None:
            print("ERROR: Couldn't find debug metrics cell")
            return False
        
        # If debug cell comes before controller cell (wrong order), fix it
        if debug_cell_idx < controller_cell_idx:
            print(f"Wrong execution order: Debug cell ({debug_cell_idx}) before controller cell ({controller_cell_idx})")
            print("Swapping cell positions...")
            
            # Swap the cells
            temp = notebook["cells"][debug_cell_idx]
            notebook["cells"][debug_cell_idx] = notebook["cells"][controller_cell_idx]
            notebook["cells"][controller_cell_idx] = temp
            
            print(f"Swapped cells: controller now at {debug_cell_idx}, debug at {controller_cell_idx}")
        
        # Add a dependency comment to the debug cell
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                cell_content = ''.join(cell["source"])
                
                if "debug_entropy, debug_grads = controller.collect_head_metrics(" in cell_content:
                    # Add dependency comment at the top
                    lines = cell["source"]
                    comment_line = "# NOTE: This cell requires the controller defined in the previous cell\n"
                    
                    if isinstance(lines, list) and lines and not any("requires the controller" in line for line in lines):
                        lines.insert(0, comment_line)
                        cell["source"] = lines
                        print(f"Added dependency comment to debug cell at index {i}")
        
        # Find all cells using controller and ensure they have dependency comments
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                cell_content = ''.join(cell["source"])
                
                if "controller." in cell_content and "debug_entropy" not in cell_content:
                    # Add dependency comment at the top if it doesn't exist
                    lines = cell["source"]
                    comment_line = "# NOTE: This cell requires the controller to be defined\n"
                    
                    if isinstance(lines, list) and lines and not any("requires the controller" in line for line in lines):
                        lines.insert(0, comment_line)
                        cell["source"] = lines
                        print(f"Added dependency comment to controller-using cell at index {i}")
        
        # Save the modified notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print(f"Notebook saved with controller execution fixes")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_controller_execution(notebook_path)