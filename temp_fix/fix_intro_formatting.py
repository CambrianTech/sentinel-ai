#!/usr/bin/env python
# Fix introduction formatting in NeuralPlasticityDemo.ipynb

import json
from pathlib import Path

def fix_intro_formatting(notebook_path):
    """Fix markdown formatting in introduction cell."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Get the first cell (intro markdown)
    if notebook['cells'][0]['cell_type'] == 'markdown':
        intro_cell = notebook['cells'][0]
        
        # Get current content and check format
        current_content = ''.join(intro_cell['source'])
        print(f"Current intro length: {len(current_content)}")
        print(f"First 50 chars: {current_content[:50]}")
        
        # Define the properly formatted introduction
        formatted_intro = """# Neural Plasticity Demo: Dynamic Pruning & Regrowth (v0.0.23)

This notebook demonstrates Sentinel AI's neural plasticity system, which allows transformer models to dynamically prune and regrow attention heads during training based on utility metrics.

## What is Neural Plasticity?

Neural plasticity is the ability of neural networks to adapt their structure over time through pruning (removing unused connections) and regrowth (restoring useful connections). This mimics how biological brains form efficient neural pathways.

In this demo, we:
1. Track the entropy and gradient patterns of each attention head
2. Dynamically prune high-entropy, low-gradient heads (unfocused, less useful)
3. Selectively revive low-entropy, higher-gradient heads (potentially useful)
4. Visualize the "brain dynamics" over time

This allows models to form more efficient neural structures during training.

### New in v0.0.23:
- Fixed visualization issues causing excessively large images
- Reduced figure sizes and DPI settings
- Fixed cell splitting in controller section

### New in v0.0.22:
- Fixed intro and conclusion section formatting
- Fixed cell character encoding issues
- Split large cells into focused, manageable sections

### New in v0.0.20:
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
- Enhanced gradient visualization to better highlight pruning decisions"""
        
        # Update the cell
        intro_cell['source'] = formatted_intro.split('\n')
        
        # Save the updated notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print("Fixed intro cell formatting")
        return True
    else:
        print("First cell is not markdown")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_intro_formatting(notebook_path)