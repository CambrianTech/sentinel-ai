#!/usr/bin/env python
# Completely rewrite intro cell with perfect formatting

import json
from pathlib import Path

def fix_intro_completely(notebook_path):
    """Completely rewrite the introduction cell with perfect formatting."""
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Get the title cell (first cell)
        if notebook['cells'][0]['cell_type'] != 'markdown':
            print("ERROR: First cell is not markdown")
            return False
        
        # Replace with a perfectly formatted introduction
        perfect_intro = """# Neural Plasticity Demo: Dynamic Pruning & Regrowth (v0.0.32)

This notebook demonstrates Sentinel AI's neural plasticity system, which allows transformer models to dynamically prune and regrow attention heads during training based on utility metrics.

## What is Neural Plasticity?

Neural plasticity is the ability of neural networks to adapt their structure over time through pruning (removing unused connections) and regrowth (restoring useful connections). This mimics how biological brains form efficient neural pathways.

In this demo, we:
1. Track the entropy and gradient patterns of each attention head
2. Dynamically prune high-entropy, low-gradient heads (unfocused, less useful)
3. Selectively revive low-entropy, higher-gradient heads (potentially useful)
4. Visualize the "brain dynamics" over time

This allows models to form more efficient neural structures during training.

### New in v0.0.32:
- Fixed order of visualization functions to prevent RuntimeError
- Corrected sequence of plt.imshow() and plt.clim() calls
- Further improved entropy visualization with color scaling

### New in v0.0.31:
- Fixed split headers where last letter appeared on new line
- Applied comprehensive fixes to header formatting
- Fixed markdown cell content formatting issues

### New in v0.0.30:
- Fixed markdown formatting throughout notebook
- Improved newline handling in all sections
- Fixed spacing between paragraphs and headers
- Enhanced list formatting and readability
- Fixed title formatting issues

### New in v0.0.29:
- Fixed visualization issues and rendering in matplotlib
- Improved training metrics visualization display
- Added better entropy visualization with non-zero ranges
- Fixed constrained_layout warnings

### New in v0.0.28:
- Fixed entropy calculation to prevent zero values
- Added improved visualization of attention patterns
- Fixed graph rendering issues in matplotlib
- Added direct calculation of entropy from attention values

### New in v0.0.27:
- Attempt at fixing an infinitely long graph

### New in v0.0.25:
- Fixed layout issues

### New in v0.0.23:
- Fixed visualization issues causing excessively large images
- Reduced figure sizes and DPI settings
- Fixed cell splitting in controller section"""
        
        # Update the cell with our perfectly formatted content
        # Split by lines and add newline character to each line
        perfect_lines = []
        for line in perfect_intro.split('\n'):
            perfect_lines.append(line + '\n')
        
        # Remove the newline from the last line
        if perfect_lines and perfect_lines[-1].endswith('\n'):
            perfect_lines[-1] = perfect_lines[-1].rstrip('\n')
        
        # Update the cell
        notebook['cells'][0]['source'] = perfect_lines
        
        # Save the notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print("Completely rewrote introduction cell with perfect formatting")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_intro_completely(notebook_path)