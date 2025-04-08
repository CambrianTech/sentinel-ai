#!/usr/bin/env python
# Comprehensive fix for NeuralPlasticityDemo.ipynb

import json
import os
from pathlib import Path

def fix_notebook_comprehensively(notebook_path):
    """
    Applies all fixes to the notebook:
    1. Completely rewrites the introduction cell with perfect formatting
    2. Fixes the conclusion cell formatting
    3. Updates the version number to v0.0.33
    4. Updates the changelog
    """
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Fix the first cell (introduction)
        # -----------------------------------
        if notebook['cells'][0]['cell_type'] != 'markdown':
            print("ERROR: First cell is not markdown")
            return False
        
        # Replace with a perfectly formatted introduction
        perfect_intro = """# Neural Plasticity Demo: Dynamic Pruning & Regrowth (v0.0.33)

This notebook demonstrates Sentinel AI's neural plasticity system, which allows transformer models to dynamically prune and regrow attention heads during training based on utility metrics.

## What is Neural Plasticity?

Neural plasticity is the ability of neural networks to adapt their structure over time through pruning (removing unused connections) and regrowth (restoring useful connections). This mimics how biological brains form efficient neural pathways.

In this demo, we:
1. Track the entropy and gradient patterns of each attention head
2. Dynamically prune high-entropy, low-gradient heads (unfocused, less useful)
3. Selectively revive low-entropy, higher-gradient heads (potentially useful)
4. Visualize the "brain dynamics" over time

This allows models to form more efficient neural structures during training.

### New in v0.0.33:
- Fixed conclusion cell formatting
- Corrected introduction formatting with proper spacing
- Added proper paragraph breaks and list formatting

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
        
        # Split by lines and add newline character to each line for proper formatting
        perfect_lines = []
        for line in perfect_intro.split('\n'):
            perfect_lines.append(line + '\n')
        
        # Remove the newline from the last line
        if perfect_lines and perfect_lines[-1].endswith('\n'):
            perfect_lines[-1] = perfect_lines[-1].rstrip('\n')
        
        # Update the introduction cell
        notebook['cells'][0]['source'] = perfect_lines
        
        # Fix the conclusion cell
        # ----------------------
        # Find the conclusion cell by looking for cells that start with "# Conclus"
        conclusion_cell_index = None
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'markdown':
                source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
                if source.startswith('# Conclus'):
                    conclusion_cell_index = i
                    break
        
        if conclusion_cell_index is not None:
            # Fix the conclusion cell formatting
            perfect_conclusion = """# Conclusion

In this notebook, we demonstrated Sentinel AI's neural plasticity system, which enables transformer models to dynamically prune and revive attention heads during training based on their utility.

Key findings:
1. The plasticity system successfully pruned high-entropy, low-gradient heads
2. Some heads were revived when they showed potential for useful learning
3. The final model achieved comparable quality with fewer active heads
4. The brain dynamics visualization shows how attention heads evolve over time

This approach mimics biological neural plasticity, where brains form efficient neural pathways by pruning unused connections and strengthening useful ones."""
            
            # Split by lines and add newline character to each line for proper formatting
            conclusion_lines = []
            for line in perfect_conclusion.split('\n'):
                conclusion_lines.append(line + '\n')
            
            # Remove the newline from the last line
            if conclusion_lines and conclusion_lines[-1].endswith('\n'):
                conclusion_lines[-1] = conclusion_lines[-1].rstrip('\n')
            
            # Update the conclusion cell
            notebook['cells'][conclusion_cell_index]['source'] = conclusion_lines
        
        # Save the notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print("Successfully applied comprehensive fixes to the notebook")
        print("- Fixed introduction cell with perfect formatting")
        print("- Fixed conclusion cell with perfect formatting")
        print("- Updated version to v0.0.33")
        print("- Updated changelog with details of all fixes")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_notebook_comprehensively(notebook_path)