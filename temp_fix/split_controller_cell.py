#!/usr/bin/env python
# Split the large controller cell into multiple smaller cells

import json
from pathlib import Path

def split_controller_cell(notebook_path):
    """Split the large controller cell into multiple focused cells."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find the controller cell (usually cell 13)
    controller_cell_idx = None
    for i, cell in enumerate(notebook['cells']):
        if (cell['cell_type'] == 'code' and 
            any(marker in ''.join(cell['source']) for marker in [
                '# Create a custom statistical pruning function',
                'def gradient_based_pruning',
                'controller = create_plasticity_controller'
            ])):
            controller_cell_idx = i
            print(f"Found controller cell at index {i}")
            break
    
    if controller_cell_idx is None:
        print("Could not find controller cell")
        return False
    
    # Split the cell into logical sections
    controller_content = ''.join(notebook['cells'][controller_cell_idx]['source'])
    
    # Define sections to split by
    sections = [
        # Function definition for gradient-based pruning
        "# Create a custom statistical pruning function based only on gradients",
        
        # Controller creation
        "# Create plasticity controller with default thresholds",
        
        # Debug section for entropy values
        "# Debug: Let's check the actual entropy values we're dealing with",
        
        # Test the pruning approach
        "# Test our gradient-only pruning approach",
        
        # Visualization of heads
        "# Visualize which heads would be pruned",
        
        # Visualize entropy and gradient
        "# Create a visual comparing entropy and gradient distributions",
        
        # Visualization with pruning decisions
        "# Create a visualization highlighting pruning decisions",
        
        # Debug attention distributions
        "# Debug attention distribution collection to see why entropy is zero"
    ]
    
    # Find where each section starts
    section_positions = []
    for section in sections:
        pos = controller_content.find(section)
        if pos >= 0:
            section_positions.append((pos, section))
    
    # Sort by position
    section_positions.sort()
    
    # Create cell contents
    cell_contents = []
    for i, (pos, section) in enumerate(section_positions):
        start = pos
        end = section_positions[i + 1][0] if i < len(section_positions) - 1 else len(controller_content)
        cell_contents.append(controller_content[start:end])
    
    # Replace the original cell with the first section
    if cell_contents:
        notebook['cells'][controller_cell_idx]['source'] = cell_contents[0].split('\n')
        
        # Insert the remaining sections as new cells
        for i, content in enumerate(cell_contents[1:], 1):
            new_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": content.split('\n')
            }
            notebook['cells'].insert(controller_cell_idx + i, new_cell)
        
        print(f"Split controller cell into {len(cell_contents)} cells")
    
    # Save the modified notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Notebook saved with split controller cell")
    return True

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    split_controller_cell(notebook_path)