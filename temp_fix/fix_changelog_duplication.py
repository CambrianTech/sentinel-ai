#!/usr/bin/env python
# Fix duplication in changelog

import json
import re
from pathlib import Path

def fix_changelog_duplication(notebook_path):
    """Fix duplicated changelog entries in the notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find the header cell with version history
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'markdown' and '# Neural Plasticity Demo' in ''.join(cell['source']):
            source = ''.join(cell['source'])
            
            # Match all version entries including duplicates
            matches = list(re.finditer(r'### New in v0\.0\.(\d+):', source))
            
            if matches and len(matches) > 1:
                print(f"Found {len(matches)} version history entries - checking for duplicates")
                
                # Get the current version
                current_version = int(matches[0].group(1))
                
                # Check if multiple entries for the same version
                if matches[0].group(1) == matches[1].group(1):
                    print(f"Found duplicate entries for v0.0.{current_version}")
                    
                    # Create a clean version entry
                    new_entry = (
                        f"### New in v0.0.{current_version}:\n"
                        "- Fixed entropy calculation to prevent zero values\n"
                        "- Added numerical stability improvements\n"
                        "- Properly normalized attention patterns\n"
                        "### New in v0.0.17:\n"
                        "- Fixed visualization scaling to prevent extremely large plots\n"
                        "- Added data downsampling for large training runs\n"
                        "- Set explicit DPI control to maintain reasonable image sizes\n"
                        "- Improved epoch boundary visualization\n"
                    )
                    
                    # Replace all version entries with the clean one
                    start_pos = matches[0].start()
                    end_pos = matches[2].start() if len(matches) > 2 else len(source)
                    fixed_source = source[:start_pos] + new_entry + source[end_pos:]
                    
                    # Update the cell
                    notebook['cells'][i]['source'] = fixed_source.split('\n')
                    print("Fixed duplicate changelog entries")
                    break
    
    # Save the fixed notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Saved fixed notebook to {notebook_path}")

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_changelog_duplication(notebook_path)