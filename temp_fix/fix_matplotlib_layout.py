#!/usr/bin/env python
# Fix matplotlib layout issues causing massive vertical images in Colab

import json
import re
from pathlib import Path

def fix_matplotlib_layout(notebook_path):
    """Fix matplotlib rendering issues in the notebook."""
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Find visualization cells using subplots
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                
                # Look for the training metrics visualization cell
                if 'Visualize training metrics with epochs' in source and 'plt.subplots(' in source:
                    print(f"Found visualization cell at index {i}")
                    
                    # Get all lines and replace the problematic subplot creation
                    lines = cell['source']
                    new_lines = []
                    
                    for line in lines:
                        # Replace subplot creation with constrained_layout version
                        if 'fig, (ax1, ax2, ax3) = plt.subplots(' in line:
                            new_lines.append("# Create visualization with constrained_layout for better Colab rendering\n")
                            new_lines.append("fig, (ax1, ax2, ax3) = plt.subplots(\n")
                            new_lines.append("    3, 1,\n")
                            new_lines.append("    figsize=(10, 6),\n")
                            new_lines.append("    dpi=100,\n")
                            new_lines.append("    sharex=True,\n")
                            new_lines.append("    constrained_layout=True  # Prevents massive vertical images in Colab\n")
                            new_lines.append(")\n")
                        # Skip problematic lines that cause layout issues
                        elif 'plt.tight_layout()' in line:
                            new_lines.append("# Removed plt.tight_layout() - not needed with constrained_layout=True\n")
                        elif 'plt.gcf().set_size_inches' in line:
                            new_lines.append("# Removed manual size adjustment - not needed with constrained_layout=True\n")
                        elif 'plt.gcf().set_dpi' in line:
                            new_lines.append("# Removed manual DPI adjustment - already set in subplots() call\n")
                        # Add diagnostic output at the end
                        elif 'plt.show()' in line:
                            new_lines.append("# Add diagnostic information about figure size\n")
                            new_lines.append("print(f\"Figure size: {fig.get_size_inches()} inches, DPI: {fig.get_dpi()}\")\n")
                            new_lines.append(line)
                        # Keep other lines unchanged
                        else:
                            new_lines.append(line)
                    
                    # Update the cell
                    cell['source'] = new_lines
                    print("Updated visualization cell with constrained_layout")
                
                # Also check for other problematic matplotlib calls in other cells
                if 'plt.figure(' in source:
                    print(f"Found figure() call in cell {i}")
                    
                    # Get the lines and fix any problematic figure creation
                    lines = cell['source']
                    new_lines = []
                    
                    for line in lines:
                        if 'plt.figure(' in line and 'figsize=' in line:
                            # Extract current figsize
                            match = re.search(r'figsize=\((\d+),\s*(\d+)\)', line)
                            if match:
                                width = int(match.group(1))
                                height = int(match.group(2))
                                
                                # If height is too large, reduce it
                                if height > 8:
                                    new_height = min(8, height)
                                    fixed_line = line.replace(f'figsize=({width}, {height})', f'figsize=({width}, {new_height})')
                                    new_lines.append(fixed_line)
                                    print(f"Fixed large figsize in cell {i}")
                                else:
                                    new_lines.append(line)
                            else:
                                new_lines.append(line)
                        # Remove problematic tight_layout calls
                        elif 'plt.tight_layout()' in line and not 'constrained_layout=True' in source:
                            new_lines.append("# Consider using constrained_layout=True instead of tight_layout()\n")
                            new_lines.append(line)
                        else:
                            new_lines.append(line)
                    
                    # Update the cell
                    cell['source'] = new_lines
        
        # Save the modified notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print(f"Fixed matplotlib layout issues in notebook")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_matplotlib_layout(notebook_path)