#!/usr/bin/env python
# Fix cells stored character-by-character

import json
import re
from pathlib import Path

def fix_character_cells(notebook_path, version=20):
    """Fix cells where content is stored character-by-character."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Check each cell
    fixed_cells = 0
    for i, cell in enumerate(notebook['cells']):
        source = cell['source']
        
        # Check if this cell is stored character-by-character
        if len(source) > 20 and all(len(s) <= 1 for s in source[:20]):
            print(f"Cell {i} is stored character-by-character, fixing...")
            
            # Reassemble the content
            full_content = ''.join(source)
            
            # Split by newlines to get proper line-by-line format
            lines = full_content.split('\n')
            
            # Add a newline to each line except the last one
            proper_lines = [line + '\n' for line in lines[:-1]]
            if lines:
                proper_lines.append(lines[-1])
            
            # Update the cell source
            notebook['cells'][i]['source'] = proper_lines
            fixed_cells += 1
    
    print(f"Fixed {fixed_cells} cells with character-by-character storage")
    
    # Save the fixed notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Saved notebook with fixed cells")
    return fixed_cells

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_character_cells(notebook_path)