#!/usr/bin/env python
# Check cell 13 content

import json
from pathlib import Path

def check_cell(notebook_path, cell_index=13):
    """Check content of a specific cell."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    if cell_index < len(notebook['cells']):
        cell = notebook['cells'][cell_index]
        content = ''.join(cell['source'])
        
        # Print first 100 chars
        print(f"First 100 chars: {content[:100]}")
        
        # Check the source structure
        print(f"Source type: {type(cell['source'])}")
        print(f"Source length: {len(cell['source'])}")
        
        # Print first few source elements
        if len(cell['source']) > 0:
            print(f"First source element: {cell['source'][0]}")
        
        # If the source seems to be character-by-character, fix it
        if len(cell['source']) > 20 and all(len(s) <= 1 for s in cell['source'][:20]):
            print("Found character-by-character source format, fixing...")
            
            # Convert to proper string list
            proper_content = []
            current_line = ""
            
            for char in cell['source']:
                if char == '\n':
                    proper_content.append(current_line)
                    current_line = ""
                else:
                    current_line += char
            
            if current_line:
                proper_content.append(current_line)
            
            print(f"Fixed source length: {len(proper_content)}")
            if proper_content:
                print(f"First fixed line: {proper_content[0]}")
            
            # Update the notebook
            notebook['cells'][cell_index]['source'] = proper_content
            
            # Save the fixed notebook
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1)
            
            print(f"Saved notebook with fixed cell {cell_index}")
    else:
        print(f"Cell index {cell_index} is out of range for notebook with {len(notebook['cells'])} cells")

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    check_cell(notebook_path, 13)