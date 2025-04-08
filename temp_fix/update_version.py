#!/usr/bin/env python
# Update version number in NeuralPlasticityDemo.ipynb

import json
import re
import os

def update_version(notebook_path, new_version=None):
    """Update version number in notebook and add changelog entry."""
    # Load the notebook
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Find the title cell and extract current version
    title_cell = notebook['cells'][0]
    title_content = ''.join(title_cell['source'])
    
    # Extract current version
    version_match = re.search(r'\(v0\.0\.(\d+)\)', title_content)
    if version_match:
        current_version = version_match.group(1)
        
        # Determine new version
        if new_version is None:
            new_version = str(int(current_version) + 1)
        
        print(f"Updating version from v0.0.{current_version} to v0.0.{new_version}")
        
        # Update version in title
        updated_title = title_content.replace(f'(v0.0.{current_version})', f'(v0.0.{new_version})')
        
        # Add new version information
        new_version_entry = (
            f"\n### New in v0.0.{new_version}:\n"
            "- Fixed entropy calculation to prevent zero values\n"
            "- Added improved visualization of attention patterns\n"
            "- Fixed graph rendering issues in matplotlib\n"
            "- Added direct calculation of entropy from attention values\n"
        )
        
        # Find where to insert new version info
        if "### New in v0.0." in updated_title:
            # Find the first existing version entry
            pattern = r"(# Neural Plasticity Demo.*?)(\n### New in v0\.0\.)"
            updated_title = re.sub(pattern, f"\\1{new_version_entry}\\2", updated_title)
        else:
            # Append at the end if no existing entries
            updated_title += new_version_entry
        
        # Update the title cell
        title_cell['source'] = updated_title.split('\n')
    else:
        print("Could not find version number in title")
        return False
    
    # Find and update the version history in conclusion cell
    for cell in notebook['cells']:
        if 'cell_type' in cell and cell['cell_type'] == 'markdown':
            content = ''.join(cell['source'])
            if '## Version History' in content:
                print("Found version history cell")
                
                # Create new version history entry
                new_entry = f"- v0.0.{new_version}: Fixed entropy calculation to prevent zero values, added improved visualization of attention patterns, fixed graph rendering issues"
                
                # Insert after the version history heading
                pattern = r"(## Version History\n+)"
                updated_content = re.sub(pattern, f"\\1{new_entry}\n\n", content)
                
                # Update the cell
                cell['source'] = updated_content.split('\n')
                break
    
    # Save the updated notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Updated version in {notebook_path} to v0.0.{new_version}")
    return True

if __name__ == "__main__":
    notebook_path = '/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb'
    update_version(notebook_path)