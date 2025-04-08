#!/usr/bin/env python
# Fix markdown cell formatting - proper newlines

import json
import re
from pathlib import Path

def fix_markdown_newlines(notebook_path):
    """Fix newline formatting in markdown cells."""
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Rules for proper markdown formatting
        fixes_applied = 0
        
        # Process each markdown cell
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'markdown':
                content = ''.join(cell['source'])
                original_content = content
                
                # Fix 1: Ensure headers have newlines after them
                content = re.sub(r'(#+\s[^\n]+)([^#\n])', r'\1\n\n\2', content)
                
                # Fix 2: Ensure paragraphs have blank lines between them
                content = re.sub(r'([^\n])\n([^\n-#*])', r'\1\n\n\2', content)
                
                # Fix 3: Ensure list items have proper spacing
                content = re.sub(r'(\n\d+\.)([^\n])', r'\n\1 \2', content)
                
                # Fix 4: Special for version history - ensure entries have blank line between them
                content = re.sub(r'(- v0\.0\.\d+:.+)\n(- v0\.0\.)', r'\1\n\n\2', content)
                
                # Fix 5: Ensure proper spacing around header sections
                content = re.sub(r'([^\n])\n(#+\s)', r'\1\n\n\2', content)
                
                # Only update if changes were made
                if content != original_content:
                    # Process into proper lines with newlines
                    lines = []
                    for line in content.split('\n'):
                        lines.append(line + '\n')
                    
                    # Remove trailing newline from last line if present
                    if lines and lines[-1].endswith('\n'):
                        lines[-1] = lines[-1].rstrip('\n')
                    
                    # Update the cell
                    cell['source'] = lines
                    fixes_applied += 1
                    print(f"Fixed markdown formatting in cell {i}")
        
        # Save the notebook if changes were made
        if fixes_applied > 0:
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1)
            print(f"Applied {fixes_applied} markdown formatting fixes")
            return True
        else:
            print("No markdown formatting issues found")
            return False
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_markdown_newlines(notebook_path)