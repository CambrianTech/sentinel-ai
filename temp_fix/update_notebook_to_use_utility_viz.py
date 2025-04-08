"""
Updates the NeuralPlasticityDemo.ipynb to import and use the visualization_additions.py utilities.

This script modifies the notebook to:
1. Add an import for the visualization functions
2. Replace inline visualization functions with the imported versions
3. Update function calls to use the imported functions

Usage:
    python update_notebook_to_use_utility_viz.py
"""

import json
import os
import re
import sys
from pathlib import Path

# Path to the notebook and utility module
NOTEBOOK_PATH = "./colab_notebooks/NeuralPlasticityDemo.ipynb"
UTILITY_MODULE = "utils.pruning.visualization_additions"

def load_notebook(notebook_path):
    """Load notebook JSON file."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_notebook(notebook_data, notebook_path):
    """Save notebook JSON file."""
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_data, f, indent=1)
    print(f"Saved updated notebook to {notebook_path}")

def add_import_statement(notebook_data):
    """Add import statement for visualization utilities."""
    # Find first code cell after imports
    for i, cell in enumerate(notebook_data['cells']):
        if cell['cell_type'] == 'code' and 'import' in cell['source']:
            # Check if our import is already present
            if UTILITY_MODULE in cell['source']:
                print("Import already exists, skipping.")
                return notebook_data
            
            # Add our import
            import_statement = f"\n# Import visualization utilities\nfrom {UTILITY_MODULE} import (\n    visualize_gradient_norms,\n    visualize_attention_matrix,\n    visualize_entropy_heatmap,\n    visualize_normalized_entropy,\n    visualize_entropy_vs_gradient,\n    visualize_training_progress\n)"
            
            # Append to cell source
            if isinstance(cell['source'], list):
                cell['source'].append(import_statement)
            else:
                cell['source'] += import_statement
            
            print(f"Added import statement in cell {i}")
            return notebook_data
    
    print("Warning: Could not find an import cell to modify")
    return notebook_data

def remove_function_definitions(notebook_data):
    """
    Remove the inline function definitions that are now imported.
    
    The main functions to remove are:
    - visualize_gradient_norms
    """
    functions_to_remove = [
        "def visualize_gradient_norms",  # The main visualization function
    ]
    
    function_cells_modified = 0
    
    for i, cell in enumerate(notebook_data['cells']):
        if cell['cell_type'] == 'code':
            # Check if this cell contains a function definition to remove
            for func_name in functions_to_remove:
                if func_name in cell['source']:
                    # This cell contains a function to remove
                    
                    # Extract the whole function by finding start and end
                    if isinstance(cell['source'], list):
                        source = "".join(cell['source'])
                    else:
                        source = cell['source']
                    
                    # Use regex to find the entire function definition
                    pattern = rf"{func_name}.*?def \w+|$"
                    new_source = re.sub(pattern, "# Function removed - using imported version\n", 
                                       source, flags=re.DOTALL)
                    
                    # Update the cell
                    if isinstance(cell['source'], list):
                        cell['source'] = [new_source]
                    else:
                        cell['source'] = new_source
                    
                    function_cells_modified += 1
                    print(f"Removed function definition in cell {i}")
    
    print(f"Modified {function_cells_modified} cells with function definitions")
    return notebook_data

def update_function_references(notebook_data):
    """
    Update any references to the old functions to use the imported versions.
    
    This includes calls to the visualization functions.
    """
    function_calls_modified = 0
    
    for i, cell in enumerate(notebook_data['cells']):
        if cell['cell_type'] == 'code':
            # Check if this cell contains a function call we need to update
            if "visualize_gradient_norms" in cell['source']:
                function_calls_modified += 1
                print(f"Cell {i} contains function calls that might need updating")
                # The function signature is the same, so we don't need to modify the actual calls
    
    print(f"Found {function_calls_modified} cells with potential function calls")
    return notebook_data

def increment_version(notebook_data):
    """Increment the version number in the first cell."""
    if notebook_data['cells']:
        first_cell = notebook_data['cells'][0]
        if first_cell['cell_type'] == 'markdown':
            source = first_cell['source']
            if isinstance(source, list):
                source = "".join(source)
            
            # Find version pattern like v0.0.37
            version_match = re.search(r'v(\d+\.\d+\.\d+)', source)
            if version_match:
                version_str = version_match.group(0)
                version_num = version_match.group(1)
                major, minor, patch = map(int, version_num.split('.'))
                
                # Increment patch version
                new_patch = patch + 1
                new_version = f"v{major}.{minor}.{new_patch}"
                
                # Replace version
                new_source = source.replace(version_str, new_version)
                
                # Add changelog entry
                changelog_entry = f"\n### New in {new_version}:\n- Refactored visualization code to use utility module\n- Improved code organization and reusability\n- Added better typing and documentation to visualization functions\n"
                
                # Find position to insert changelog
                new_in_pos = new_source.find("### New in")
                if new_in_pos != -1:
                    new_source = new_source[:new_in_pos] + changelog_entry + new_source[new_in_pos:]
                
                # Update cell
                if isinstance(first_cell['source'], list):
                    first_cell['source'] = [new_source]
                else:
                    first_cell['source'] = new_source
                
                print(f"Incremented version from {version_str} to {new_version}")
    
    return notebook_data

def main():
    # Check for command-line arguments
    dry_run = False
    if len(sys.argv) > 1 and sys.argv[1] in ['-n', '--dry-run']:
        dry_run = True
        print("Running in dry-run mode (no changes will be saved)")
    
    print(f"Updating notebook: {NOTEBOOK_PATH}")
    
    # Check if notebook exists
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: Notebook not found at {NOTEBOOK_PATH}")
        return 1
    
    # Load notebook
    try:
        notebook_data = load_notebook(NOTEBOOK_PATH)
    except Exception as e:
        print(f"Error loading notebook: {e}")
        return 1
    
    # Make modifications
    notebook_data = add_import_statement(notebook_data)
    notebook_data = remove_function_definitions(notebook_data)
    notebook_data = update_function_references(notebook_data)
    notebook_data = increment_version(notebook_data)
    
    # Save updated notebook (unless in dry-run mode)
    if not dry_run:
        try:
            # Backup original first
            backup_path = NOTEBOOK_PATH + ".bak"
            if not os.path.exists(backup_path):
                print(f"Creating backup at {backup_path}")
                with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as src:
                    with open(backup_path, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
            
            save_notebook(notebook_data, NOTEBOOK_PATH)
            print("Notebook updated successfully")
        except Exception as e:
            print(f"Error saving notebook: {e}")
            return 1
    else:
        print("Dry run complete - no changes saved")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())