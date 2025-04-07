#!/usr/bin/env python
# Split large cells in the notebook into multiple smaller cells

import json
from pathlib import Path
import re

def split_large_cells(notebook_path, version=20):
    """Split specific large cells in the notebook for better readability."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # First, let's identify large cells
    large_cells = []
    for i, cell in enumerate(notebook['cells']):
        if len(''.join(cell['source'])) > 5000:
            large_cells.append((i, cell['cell_type'], len(''.join(cell['source']))))
    
    print(f"Found {len(large_cells)} large cells to split")
    
    # Let's focus on cell 13 (controller cell) first
    if len(large_cells) >= 2 and large_cells[1][0] == 13:
        controller_idx = large_cells[1][0]
        print(f"Processing controller cell (index {controller_idx})")
        
        # Get the content
        controller_content = ''.join(notebook['cells'][controller_idx]['source'])
        
        # Split into logical sections
        sections = [
            # Section 1: Gradient-based pruning function
            "# Create a custom statistical pruning function based only on gradients",
            
            # Section 2: Controller creation and display initial stats
            "# Create plasticity controller with default thresholds",
            "# Display initial model stats",
            
            # Section 3: Debug entropy values
            "# Debug: Let's check the actual entropy values we're dealing with",
            "# Calculate statistics to help with threshold setting",
            
            # Section 4: Test pruning approach
            "# Test our gradient-only pruning approach",
            
            # Section 5: Visualizations
            "# Visualize which heads would be pruned",
            "# Create a visual comparing entropy and gradient distributions",
            "# Create a visualization highlighting pruning decisions based on gradient values",
            
            # Section 6: Debug attention
            "# Debug attention distribution collection to see why entropy is zero"
        ]
        
        # Find the position of each section
        section_starts = []
        for section in sections:
            pos = controller_content.find(section)
            if pos >= 0:
                section_starts.append((pos, section))
        
        # Sort by position
        section_starts.sort()
        
        if section_starts:
            # Extract content for each section
            cell_contents = []
            for i, (pos, section) in enumerate(section_starts):
                start = pos
                if i < len(section_starts) - 1:
                    end = section_starts[i+1][0]
                else:
                    end = len(controller_content)
                
                cell_contents.append(controller_content[start:end])
            
            # Replace original cell with first section
            notebook['cells'][controller_idx]['source'] = cell_contents[0].split('\n')
            
            # Insert new cells for remaining sections
            for i, content in enumerate(cell_contents[1:], 1):
                new_cell = {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": content.split('\n')
                }
                notebook['cells'].insert(controller_idx + i, new_cell)
            
            print(f"Split controller cell into {len(cell_contents)} cells")
    
    # Split cell 17 (training loop)
    if len(large_cells) >= 3 and large_cells[2][0] == 17:
        training_idx = large_cells[2][0]
        print(f"Processing training loop cell (index {training_idx})")
        
        # Get the content
        training_content = ''.join(notebook['cells'][training_idx]['source'])
        
        # Split into logical sections
        sections = [
            # Section 1: Initialize training components
            "# Initialize training components",
            
            # Section 2: Initialize metrics tracking
            "# Initialize metrics tracking",
            
            # Section 3: Custom pruning function
            "# Custom function to apply pruning based purely on gradients",
            
            # Section 4: Checkpoint and utility functions
            "# Convert stats dict to regular dict for serialization",
            "# Function to save checkpoint",
            "# Function to run inference",
            
            # Section 5: Training loop implementation
            "# Training loop",
            "try:"
        ]
        
        # Find the position of each section
        section_starts = []
        for section in sections:
            pos = training_content.find(section)
            if pos >= 0:
                section_starts.append((pos, section))
        
        # Sort by position
        section_starts.sort()
        
        if section_starts:
            # Extract content for each section
            cell_contents = []
            for i, (pos, section) in enumerate(section_starts):
                start = pos
                if i < len(section_starts) - 1:
                    end = section_starts[i+1][0]
                else:
                    end = len(training_content)
                
                cell_contents.append(training_content[start:end])
            
            # Replace original cell with first section
            notebook['cells'][training_idx]['source'] = cell_contents[0].split('\n')
            
            # Insert new cells for remaining sections
            for i, content in enumerate(cell_contents[1:], 1):
                new_cell = {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": content.split('\n')
                }
                notebook['cells'].insert(training_idx + i, new_cell)
            
            print(f"Split training loop cell into {len(cell_contents)} cells")
    
    # Fix markdown formatting in all cells
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'markdown':
            # Get the content
            md_content = ''.join(cell['source'])
            
            # Fix headings without newlines
            md_content = re.sub(r'(#+ [^\n]+)([^#\n])', r'\1\n\2', md_content)
            
            # Fix lists without spacing
            md_content = re.sub(r'(\n\d+\.)([^\n])', r'\n\1 \2', md_content)
            
            # Update the cell
            notebook['cells'][i]['source'] = md_content.split('\n')
    
    # Update version number in conclusion cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'markdown' and '## Conclusion' in ''.join(cell['source']):
            # Get content
            conclusion = ''.join(cell['source'])
            
            # Fix heading spacing
            conclusion = re.sub(r'(## Conclusion)([^\n])', r'\1\n\2', conclusion)
            conclusion = re.sub(r'(## Version History)([^\n])', r'\1\n\2', conclusion)
            
            # Update version history
            if "v0.0.20:" not in conclusion:
                version_entry = f"- v0.0.{version}: Split large cells into smaller, more manageable units, fixed markdown formatting throughout the notebook, improved code organization and readability\n\n"
                conclusion = re.sub(r'(## Version History\n\n)', r'\1' + version_entry, conclusion)
            
            # Update the cell
            notebook['cells'][i]['source'] = conclusion.split('\n')
            print("Updated conclusion cell")
    
    # Save the modified notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Notebook saved with split cells and improved markdown formatting")
    return True

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    split_large_cells(notebook_path)