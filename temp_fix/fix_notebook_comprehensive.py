#!/usr/bin/env python
# Comprehensive fix for notebook formatting and cell size issues

import json
import re
from pathlib import Path

def split_large_cell(cell_content, max_lines=100):
    """
    Split a large cell into multiple smaller cells.
    Returns a list of cell contents.
    """
    lines = cell_content.split('\n')
    
    # If the cell is not that large, return as is
    if len(lines) <= max_lines:
        return [cell_content]
    
    # Find logical splitting points (groups of related code)
    split_points = []
    
    # Look for logical sections to split
    for i, line in enumerate(lines):
        if i > 50 and i < len(lines) - 50:  # Don't split too early or too late
            # Look for natural boundaries like function declarations or major comment blocks
            if line.startswith('# ') and not line.startswith('# -') and len(line) > 3:
                split_points.append(i)
            elif line.startswith('def ') and i > 0 and lines[i-1] == '':
                split_points.append(i)
            elif line.startswith('controller') and lines[i-1] == '':
                split_points.append(i)
    
    # If no good splitting points found or too few, create artificial ones
    if len(split_points) < 1:
        chunk_size = max_lines
        split_points = list(range(chunk_size, len(lines), chunk_size))
    
    # Create cell contents from split points
    result = []
    start = 0
    
    for point in split_points:
        cell_text = '\n'.join(lines[start:point])
        result.append(cell_text)
        start = point
    
    # Add the last segment
    result.append('\n'.join(lines[start:]))
    
    return result

def fix_notebook_comprehensive(notebook_path, version_number=20):
    """
    Comprehensive fix for notebook:
    1. Fix markdown formatting
    2. Split large cells into smaller ones
    3. Update version number
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Fix all markdown cells
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'markdown':
            # Ensure proper line breaks after headings
            source = ''.join(cell['source'])
            source = re.sub(r'(#+ [^\n]+)([^#\n])', r'\1\n\2', source)
            
            # Fix lists that don't have proper spacing
            source = re.sub(r'(\n\d+\.)([^\n])', r'\n\1 \2', source)
            
            # Update the cell
            notebook['cells'][i]['source'] = source.split('\n')
    
    # Fix the header cell (cell 0)
    for i, cell in enumerate(notebook['cells']):
        if i == 0 and cell['cell_type'] == 'markdown':
            # Fix the header markdown cell formatting
            header_content = """# Neural Plasticity Demo: Dynamic Pruning & Regrowth (v0.0.{})

This notebook demonstrates Sentinel AI's neural plasticity system, which allows transformer models to dynamically prune and regrow attention heads during training based on utility metrics.

## What is Neural Plasticity?

Neural plasticity is the ability of neural networks to adapt their structure over time through pruning (removing unused connections) and regrowth (restoring useful connections). This mimics how biological brains form efficient neural pathways.

In this demo, we:
1. Track the entropy and gradient patterns of each attention head
2. Dynamically prune high-entropy, low-gradient heads (unfocused, less useful)
3. Selectively revive low-entropy, higher-gradient heads (potentially useful)
4. Visualize the "brain dynamics" over time

This allows models to form more efficient neural structures during training.

### New in v0.0.{}:
- Fixed cell size issues by splitting large cells
- Fixed markdown formatting throughout the notebook
- Improved cell organization and readability
- Fixed entropy calculation to prevent zero values
- Added numerical stability improvements 
- Properly normalized attention patterns

### New in v0.0.17:
- Fixed visualization scaling to prevent extremely large plots
- Added data downsampling for large training runs
- Set explicit DPI control to maintain reasonable image sizes
- Improved epoch boundary visualization

### Previous in v0.0.16:
- Fixed critical pruning logic to correctly target heads with LOWEST gradient norms
- Added comprehensive attention pattern visualization with log scaling
- Fixed serialization error when saving checkpoints
- Added detailed gradient statistics for pruned vs. kept heads
- Enhanced gradient visualization to better highlight pruning decisions

### Previous in v0.0.15:
- Improved warm-up phase to run until loss stabilizes with automatic detection
- Added maximum warm-up epoch limit with configurable parameter
- Added comprehensive warm-up monitoring with stabilization metrics
- Added progress tracking across epochs with early termination option""".format(version_number, version_number)
            
            # Update the cell content
            notebook['cells'][i]['source'] = header_content.split('\n')
            print(f"Fixed header cell formatting")
    
    # Now handle the large controller creation cell (typically cell 13)
    controller_cell_idx = None
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and ''.join(cell['source']).startswith('# Create a custom statistical pruning function'):
            controller_cell_idx = i
            print(f"Found large controller cell at index {i}")
            
            # Get the content
            cell_content = ''.join(cell['source'])
            
            # Split the cell
            new_cell_contents = split_large_cell(cell_content)
            
            if len(new_cell_contents) > 1:
                print(f"Split large cell into {len(new_cell_contents)} parts")
                
                # Update the first cell with the first part
                notebook['cells'][i]['source'] = new_cell_contents[0].split('\n')
                
                # Insert new cells for the remaining parts
                for j, content in enumerate(new_cell_contents[1:], 1):
                    new_cell = {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "source": content.split('\n')
                    }
                    notebook['cells'].insert(i + j, new_cell)
                
                print(f"Inserted {len(new_cell_contents) - 1} new cells")
            else:
                print("Cell not large enough to split")
    
    # Fix the conclusion cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'markdown' and '## Conclusion' in ''.join(cell['source']):
            conclusion_content = ''.join(cell['source'])
            
            # Ensure there are newlines after headings
            conclusion_content = re.sub(r'(## Conclusion)([^\n])', r'\1\n\2', conclusion_content)
            conclusion_content = re.sub(r'(## Version History)([^\n])', r'\1\n\2', conclusion_content)
            
            # Update version history
            version_pattern = r'- v0\.0\.(\d+):'
            matches = re.findall(version_pattern, conclusion_content)
            if matches:
                # Update the latest version entry or add a new one
                if not f'v0.0.{version_number}:' in conclusion_content:
                    new_entry = f"- v0.0.{version_number}: Fixed cell size issues and markdown formatting throughout the notebook, split large cells for better readability, fixed entropy calculation to prevent zero values"
                    
                    # Add the new entry at the top
                    conclusion_content = re.sub(
                        r'(## Version History\n\n)',
                        r'\1' + new_entry + '\n',
                        conclusion_content
                    )
            
            # Update the cell content
            notebook['cells'][i]['source'] = conclusion_content.split('\n')
            print(f"Fixed conclusion cell formatting")
    
    # Save the modified notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Notebook saved with fixed formatting, split cells, and updated to version v0.0.{version_number}")
    return True

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_notebook_comprehensive(notebook_path, version_number=20)