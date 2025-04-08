#!/usr/bin/env python
# Fix title formatting and increment version

import json
from pathlib import Path

def fix_title_version(notebook_path):
    """Fix title formatting and increment version."""
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Get the title cell
        title_cell = notebook['cells'][0]
        
        # Check if it's a markdown cell
        if title_cell['cell_type'] != 'markdown':
            print("ERROR: First cell is not markdown")
            return False
        
        # Get current content
        current_lines = title_cell['source']
        print(f"Current lines: {current_lines}")
        
        # Fix title and increment version
        fixed_title = "# Neural Plasticity Demo: Dynamic Pruning & Regrowth (v0.0.30)\n\n"
        
        # Build rest of content - skip the title line
        rest_of_content = []
        title_found = False
        
        for line in current_lines:
            if "# Neural Plasticity Demo" in line:
                title_found = True
                continue
            
            if title_found:
                rest_of_content.append(line)
        
        # Add changelog entry
        changelog_entry = (
            "### New in v0.0.30:\n"
            "- Fixed markdown formatting throughout notebook\n"
            "- Improved newline handling in all sections\n"
            "- Fixed spacing between paragraphs and headers\n"
            "- Enhanced list formatting and readability\n"
            "- Fixed title formatting issues\n\n"
        )
        
        # Create updated content
        updated_content = [fixed_title]
        
        # Look for existing changelog entries
        found_changelog = False
        for i, line in enumerate(rest_of_content):
            if "### New in v0.0." in line and not found_changelog:
                updated_content.append(changelog_entry)
                found_changelog = True
            updated_content.append(line)
        
        # Update the cell
        title_cell['source'] = updated_content
        
        # Find and update version history in conclusion
        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown':
                content = ''.join(cell['source'])
                if '## Version History' in content:
                    # Update version history
                    lines = []
                    version_found = False
                    
                    for line in cell['source']:
                        if '## Version History' in line and not version_found:
                            lines.append(line)
                            lines.append('\n')
                            lines.append("- v0.0.30: Fixed markdown formatting throughout notebook, improved newline handling, fixed spacing issues\n")
                            lines.append('\n')
                            version_found = True
                        else:
                            lines.append(line)
                    
                    cell['source'] = lines
                    print("Updated version history")
                    break
        
        # Save the notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print("Fixed title formatting and incremented version to v0.0.30")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_title_version(notebook_path)