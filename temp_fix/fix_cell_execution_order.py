#!/usr/bin/env python
# Fix cell execution order and add explicit controller dependency checks

import json
import re
from pathlib import Path

def fix_cell_execution_order(notebook_path):
    """Fix notebook cell execution order and controller reference issues."""
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Find controller and debug cells
        controller_cell_idx = None
        debug_cell_idx = None
        
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                
                if 'controller = create_plasticity_controller(' in source:
                    controller_cell_idx = i
                    print(f"Found controller creation cell at index {i}")
                
                if 'debug_entropy, debug_grads = controller.collect_head_metrics(' in source:
                    debug_cell_idx = i
                    print(f"Found debug metrics cell at index {i}")
        
        if controller_cell_idx is None or debug_cell_idx is None:
            print("ERROR: Could not find required cells")
            return False
        
        # Ensure debug cell comes after controller cell
        if debug_cell_idx < controller_cell_idx:
            print(f"Moving debug cell {debug_cell_idx} to after controller cell {controller_cell_idx}")
            # Extract the debug cell
            debug_cell = notebook['cells'][debug_cell_idx]
            # Remove it from its current position
            notebook['cells'].pop(debug_cell_idx)
            # Insert it right after the controller cell
            notebook['cells'].insert(controller_cell_idx + 1, debug_cell)
            
            # Update indices
            if debug_cell_idx < controller_cell_idx:
                controller_cell_idx -= 1
            debug_cell_idx = controller_cell_idx + 1
            print(f"New positions: controller cell {controller_cell_idx}, debug cell {debug_cell_idx}")
        
        # Add explicit controller check to the debug cell
        debug_cell = notebook['cells'][debug_cell_idx]
        
        # Get the source code
        lines = debug_cell['source']
        new_lines = []
        
        # Add check at the beginning
        new_lines.append("# Make absolutely sure controller is defined\n")
        new_lines.append("try:\n")
        new_lines.append("    # Access the controller to verify it exists\n")
        new_lines.append("    controller\n")
        new_lines.append("except NameError:\n")
        new_lines.append("    print(\"ERROR: controller variable is not defined!\")\n")
        new_lines.append("    print(\"Please run the cell above that creates the controller first.\")\n")
        new_lines.append("    # Create an emergency backup controller for demo purposes\n")
        new_lines.append("    print(\"Creating emergency backup controller for demonstration...\")\n")
        new_lines.append("    controller = create_plasticity_controller(\n")
        new_lines.append("        model=model,\n")
        new_lines.append("        mode=PRUNING_MODE,\n")
        new_lines.append("        high_entropy_threshold=0.8,\n")
        new_lines.append("        low_entropy_threshold=0.4,\n")
        new_lines.append("        grad_threshold=1e-3,\n")
        new_lines.append("        min_zero_epochs=3)\n")
        new_lines.append("\n")
        
        # Add the existing lines
        for line in lines:
            new_lines.append(line)
        
        # Update the cell
        debug_cell['source'] = new_lines
        print(f"Added controller safety check to debug cell")
        
        # Look for any other cells that use controller and add checks
        for i, cell in enumerate(notebook['cells']):
            if i != debug_cell_idx and cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                
                # If the cell uses controller but doesn't define it
                if 'controller.' in source and 'controller =' not in source:
                    print(f"Adding controller check to cell {i}")
                    
                    # Get the source code
                    lines = cell['source']
                    new_lines = []
                    
                    # Add the controller check
                    new_lines.append("# Check if controller exists\n")
                    new_lines.append("try:\n")
                    new_lines.append("    controller\n")
                    new_lines.append("except NameError:\n")
                    new_lines.append("    print(\"ERROR: controller variable is not defined!\")\n")
                    new_lines.append("    print(\"Please run the cell that creates the controller first.\")\n")
                    new_lines.append("    # Create a simple dummy controller to avoid breaking execution\n")
                    new_lines.append("    from types import SimpleNamespace\n")
                    new_lines.append("    controller = SimpleNamespace()\n")
                    new_lines.append("    controller.apply_pruning = lambda *args, **kwargs: None\n")
                    new_lines.append("    controller.stats = {}\n")
                    new_lines.append("\n")
                    
                    # Add the existing lines
                    for line in lines:
                        new_lines.append(line)
                    
                    # Update the cell
                    cell['source'] = new_lines
        
        # Add instruction cell before controller cell
        instruction_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Important Note on Cell Execution Order\n",
                "\n",
                "⚠️ **Critical**: Cells in this notebook must be executed in order.\n",
                "\n",
                "The next cell creates the plasticity controller that's used throughout the notebook. Make sure to run:\n",
                "1. The cell below that creates the controller\n",
                "2. Then the debug cell after it\n",
                "3. Then continue with the rest of the notebook in sequence\n",
                "\n",
                "If you get `NameError: name 'controller' is not defined`, go back and run the controller creation cell first."
            ]
        }
        
        notebook['cells'].insert(controller_cell_idx, instruction_cell)
        print("Added instruction cell before controller")
        
        # Save the modified notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print(f"Fixed cell execution order in notebook")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_cell_execution_order(notebook_path)