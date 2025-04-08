#!/usr/bin/env python
# Fix formatting issues in the notebook

import json
import re
from pathlib import Path

def fix_notebook_formatting(notebook_path, version_number=19):
    """Fix formatting issues in the notebook and update version number."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
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
- Fixed entropy calculation to prevent zero values
- Added numerical stability improvements 
- Properly normalized attention patterns
- Fixed markdown formatting issues

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
        
        # Fix the conclusion cell
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
                    new_entry = f"- v0.0.{version_number}: Fixed entropy calculation to prevent zero values, added numerical stability improvements, properly normalized attention patterns, fixed markdown formatting issues"
                    
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
    
    print(f"Notebook saved with fixed formatting and updated to version v0.0.{version_number}")
    return True

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_notebook_formatting(notebook_path, version_number=19)