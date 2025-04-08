#!/usr/bin/env python
# Validate notebook structure and identify issues

import json
import sys
from pathlib import Path

def validate_notebook(notebook_path):
    """Validate the notebook structure and identify any issues."""
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        print(f"Successfully loaded notebook JSON structure")
        
        # Check basic structure
        if "cells" not in notebook:
            print("ERROR: Notebook does not have a 'cells' key")
            return False
        
        # Check each cell
        for i, cell in enumerate(notebook["cells"]):
            if "cell_type" not in cell:
                print(f"ERROR: Cell {i} does not have a 'cell_type' key")
                continue
                
            if "source" not in cell:
                print(f"ERROR: Cell {i} does not have a 'source' key")
                continue
                
            # Check source format for markdown cells
            if cell["cell_type"] == "markdown":
                source = cell["source"]
                if not isinstance(source, list):
                    print(f"ERROR: Cell {i} source is not a list (type: {type(source)})")
                    continue
                
                # Check for character-by-character split
                if len(source) > 20 and all(len(s) <= 1 for s in source[:20]):
                    print(f"ERROR: Cell {i} has character-by-character source format")
                    print(f"  First 10 source items: {source[:10]}")
                    continue
                
                # Check for malformed lines
                for j, line in enumerate(source[:5]):
                    if len(line) > 100:
                        print(f"Cell {i}, line {j} is very long ({len(line)} chars)")
                        print(f"  First 50 chars: {line[:50]}")
                
        # Check first cell content
        if notebook["cells"][0]["cell_type"] == "markdown":
            first_cell = notebook["cells"][0]
            content = ''.join(first_cell["source"])
            
            if "###" in content:
                print(f"WARNING: First cell contains '###' which might not render correctly:")
                print(f"  Example: {content.split('###')[0][:50]}###...")
            
            if "#" in content and "\n#" not in content:
                print(f"WARNING: First cell might have header formatting issues (missing newlines)")
        
        return True
    
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON structure: {str(e)}")
        return False
    except Exception as e:
        print(f"ERROR: Unknown validation error: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    if len(sys.argv) > 1:
        notebook_path = Path(sys.argv[1])
        
    validate_notebook(notebook_path)