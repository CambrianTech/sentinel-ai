"""
Updates the NeuralPlasticityDemo.ipynb to import and use the visualization_additions.py utilities.

This script modifies the notebook to:
1. Add an import for the visualization functions
2. Replace inline visualization functions with the imported versions
3. Update function calls to use the imported functions
4. Add support for persistent display widgets
5. Fix attention visualization scaling

Usage:
    python update_notebook_to_use_utility_viz.py
"""

import json
import os
import re
import sys
from pathlib import Path

# Path to the notebook and utility module
NOTEBOOK_PATH = "/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb"
UTILITY_MODULE_PRUNING = "utils.pruning.visualization_additions"
UTILITY_MODULE_COLAB = "utils.colab.visualizations"

def load_notebook(notebook_path):
    """Load notebook JSON file."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_notebook(notebook_data, notebook_path):
    """Save notebook JSON file."""
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_data, f, indent=1)
    print(f"Saved updated notebook to {notebook_path}")

def add_import_statements(notebook_data):
    """Add import statements for visualization utilities."""
    # Find import cells
    import_added = False
    
    for i, cell in enumerate(notebook_data['cells']):
        if cell['cell_type'] == 'code' and 'import' in cell['source']:
            cell_source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            
            # Check if our import is already present
            if UTILITY_MODULE_PRUNING in cell_source:
                continue
                
            # Add imports for pruning visualizations
            if "matplotlib.pyplot as plt" in cell_source and UTILITY_MODULE_PRUNING not in cell_source:
                import_statement = f"\n# Import visualization utilities\nfrom {UTILITY_MODULE_PRUNING} import (\n    visualize_gradient_norms,\n    visualize_attention_matrix,\n    visualize_entropy_heatmap,\n    visualize_normalized_entropy,\n    visualize_entropy_vs_gradient,\n    visualize_training_progress,\n    calculate_proper_entropy\n)"
                
                # Append to cell source
                if isinstance(cell['source'], list):
                    cell['source'].append(import_statement)
                else:
                    cell['source'] += import_statement
                
                print(f"Added pruning visualization import in cell {i}")
                import_added = True
                
            # Add imports for persistent display
            if "from torch.utils.data import DataLoader" in cell_source and UTILITY_MODULE_COLAB not in cell_source:
                display_import = f"\n# Import visualization widgets\nfrom {UTILITY_MODULE_COLAB} import (\n    PersistentDisplay, \n    TrainingMonitor,\n    visualize_gradient_norms\n)"
                
                # Append to cell source
                if isinstance(cell['source'], list):
                    cell['source'].append(display_import)
                else:
                    cell['source'] += display_import
                
                print(f"Added persistent display import in cell {i}")
                import_added = True
                
    return import_added

def remove_function_definitions(notebook_data):
    """
    Remove the inline function definitions that are now imported.
    """
    functions_to_remove = [
        "def visualize_gradient_norms",  # The main visualization function to remove
    ]
    
    function_cells_modified = 0
    
    for i, cell in enumerate(notebook_data['cells']):
        if cell['cell_type'] == 'code':
            if isinstance(cell['source'], list):
                source = "".join(cell['source'])
            else:
                source = cell['source']
                
            # Check if this cell contains a function definition to remove
            for func_name in functions_to_remove:
                if func_name in source:
                    # This cell contains a function to remove
                    
                    # Extract the whole function definition
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
    return function_cells_modified > 0

def fix_attention_visualizations(notebook_data):
    """Fix attention visualization code in cells that generate attention heatmaps."""
    cells_fixed = 0
    
    # Look for cells that visualize attention matrices
    for i, cell in enumerate(notebook_data['cells']):
        if cell['cell_type'] == 'code':
            if isinstance(cell['source'], list):
                source = "".join(cell['source'])
            else:
                source = cell['source']
            
            # Check for attention visualizations
            attention_viz = (
                ('plt.imshow' in source and 'attn' in source) or
                ('attention_map = plt.imshow' in source) or
                ('im = ax1.imshow' in source and 'attn_pattern' in source)
            )
            
            if attention_viz:
                # Ensure proper colormap scaling
                if 'set_clim(0, 1.0)' not in source:
                    # Add colormap scaling if missing
                    new_source = re.sub(
                        r'(plt\.imshow\([^)]+\))', 
                        r'\1\nplt.clim(0, 1.0)  # Ensure proper scaling for attention values', 
                        source
                    )
                    
                    # Also ensure colorbar is present
                    if 'colorbar' not in new_source:
                        new_source = re.sub(
                            r'(plt\.imshow\([^)]+\)[^\n]*\n)', 
                            r'\1plt.colorbar(label="Attention probability")\n', 
                            new_source
                        )
                    
                    # Update the cell
                    if isinstance(cell['source'], list):
                        cell['source'] = [new_source]
                    else:
                        cell['source'] = new_source
                    
                    cells_fixed += 1
                    print(f"Fixed attention visualization in cell {i}")
                
                # Check for visualizations using imshow with axes
                elif 'ax' in source and 'imshow' in source and 'set_clim' not in source:
                    # Add colormap scaling for axes-based visualizations
                    new_source = re.sub(
                        r'(im = ax\d*\.imshow\([^)]+\))', 
                        r'\1\nim.set_clim(0, 1.0)  # Ensure proper scaling for attention', 
                        source
                    )
                    
                    # Update the cell
                    if isinstance(cell['source'], list):
                        cell['source'] = [new_source]
                    else:
                        cell['source'] = new_source
                    
                    cells_fixed += 1
                    print(f"Fixed axes-based attention visualization in cell {i}")
    
    print(f"Fixed {cells_fixed} cells with attention visualizations")
    return cells_fixed > 0

def fix_entropy_calculations(notebook_data):
    """Fix entropy calculation code in cells that compute entropy for attention matrices."""
    cells_fixed = 0
    
    # Look for cells that calculate entropy
    for i, cell in enumerate(notebook_data['cells']):
        if cell['cell_type'] == 'code':
            if isinstance(cell['source'], list):
                source = "".join(cell['source'])
            else:
                source = cell['source']
            
            # Check for entropy calculation code
            if 'calculate_proper_entropy(' in source or 'compute_improved_entropy(' in source:
                # Update to use imported function
                if 'calculate_proper_entropy(' in source and 'from utils.pruning.visualization_additions import calculate_proper_entropy' not in source:
                    # Replace with imported function
                    new_source = re.sub(
                        r'def calculate_proper_entropy\([^)]+\):.*?return entropy',
                        '# Function replaced with imported version from utils.pruning.visualization_additions',
                        source,
                        flags=re.DOTALL
                    )
                    
                    # Update the cell
                    if isinstance(cell['source'], list):
                        cell['source'] = [new_source]
                    else:
                        cell['source'] = new_source
                    
                    cells_fixed += 1
                    print(f"Fixed entropy calculation in cell {i}")
                
                # Fix entropy visualization scaling
                if 'plt.imshow' in source and 'entropy' in source.lower() and 'set_clim' not in source:
                    # Add proper scaling for entropy visualization
                    new_source = re.sub(
                        r'(plt\.imshow\([^)]+, cmap=[^)]+\))', 
                        r'\1\nplt.clim(0, max(0.1, entropy_data.max()))  # Ensure proper entropy range', 
                        source
                    )
                    
                    # Update the cell
                    if isinstance(cell['source'], list):
                        cell['source'] = [new_source]
                    else:
                        cell['source'] = new_source
                    
                    cells_fixed += 1
                    print(f"Fixed entropy visualization scaling in cell {i}")
    
    print(f"Fixed {cells_fixed} cells with entropy calculations")
    return cells_fixed > 0

def add_persistent_display(notebook_data):
    """Add persistent display widgets for training monitoring."""
    cells_modified = 0
    
    # Look for training loop cells
    for i, cell in enumerate(notebook_data['cells']):
        if cell['cell_type'] == 'code':
            if isinstance(cell['source'], list):
                source = "".join(cell['source'])
            else:
                source = cell['source']
            
            # Check for visualization code that could be improved with persistent display
            if 'pruning_monitor' in source and 'TrainingMonitor' in source:
                print(f"Persistent display already used in cell {i}")
                continue
                
            # Check for training loop
            if ('global_step' in source and 'metrics_history' in source and 
                'for epoch in range(NUM_EPOCHS)' in source):
                
                # Add pruning monitor before training loop
                monitor_code = """
# Create a persistent pruning monitor for visualizations
pruning_monitor = TrainingMonitor(
    title="Neural Plasticity Training Progress",
    metrics_to_track=["step", "epoch", "train_loss", "eval_loss", 
                     "pruned_heads", "revived_heads", "sparsity", "perplexity"]
)
"""
                # Insert before training loop
                loop_match = re.search(r'for epoch in range\(NUM_EPOCHS\):', source)
                if loop_match:
                    insert_pos = loop_match.start()
                    new_source = source[:insert_pos] + monitor_code + source[insert_pos:]
                    
                    # Update the cell
                    if isinstance(cell['source'], list):
                        cell['source'] = [new_source]
                    else:
                        cell['source'] = new_source
                    
                    cells_modified += 1
                    print(f"Added persistent display setup in cell {i}")
            
            # Find visualization code in training loops and update to use pruning_monitor
            if ('metrics_history' in source and 
                ('plt.figure' in source or 'plt.subplot' in source) and 
                'display_steps' in source):
                
                # Replace with monitor update - only if we can identify that this is plotting metrics
                if 'train_loss' in source and 'eval_loss' in source:
                    # Replace inline visualization with monitor update
                    new_source = re.sub(
                        r'# Visualize training metrics.*?plt\.show\(\)',
                        '# Update the pruning monitor with metrics\npruning_monitor.update_metrics(\n    current_metrics,\n    step=global_step,\n    epoch=epoch + 1,\n    plot=True\n)',
                        source,
                        flags=re.DOTALL
                    )
                    
                    # Update the cell
                    if isinstance(cell['source'], list):
                        cell['source'] = [new_source]
                    else:
                        cell['source'] = new_source
                    
                    cells_modified += 1
                    print(f"Updated visualization code to use persistent display in cell {i}")
    
    print(f"Modified {cells_modified} cells to use persistent display")
    return cells_modified > 0

def increment_version(notebook_data):
    """Increment the version number in the first cell and add changelog entry."""
    version_incremented = False
    
    # Get the first cell if it's markdown
    if notebook_data['cells'] and notebook_data['cells'][0]['cell_type'] == 'markdown':
        first_cell = notebook_data['cells'][0]
        
        # Get content as string
        if isinstance(first_cell['source'], list):
            source = "".join(first_cell['source'])
        else:
            source = first_cell['source']
        
        # Find version pattern like v0.0.43
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
            changelog_entry = f"\n### New in {new_version}:\n- Added persistent visualization widget to display training progress\n- Created dedicated visualization utilities in utils/colab/visualizations.py\n- Implemented TrainingMonitor class for in-place metric updates\n- Improved training loop to use single persistent display\n- Reduced output cell clutter by updating in-place instead of creating new cells\n"
            
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
            version_incremented = True
    
    return version_incremented

def main():
    """Main function to update the notebook."""
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
    imports_added = add_import_statements(notebook_data)
    functions_removed = remove_function_definitions(notebook_data)
    attention_viz_fixed = fix_attention_visualizations(notebook_data)
    entropy_fixed = fix_entropy_calculations(notebook_data)
    persistent_display_added = add_persistent_display(notebook_data)
    version_incremented = increment_version(notebook_data)
    
    # Summary of changes
    changes_made = imports_added or functions_removed or attention_viz_fixed or entropy_fixed or persistent_display_added or version_incremented
    
    if changes_made:
        print("\nChanges to be applied:")
        if imports_added:
            print("- Added visualization utility imports")
        if functions_removed:
            print("- Removed redundant function definitions")
        if attention_viz_fixed:
            print("- Fixed attention visualization scaling")
        if entropy_fixed:
            print("- Fixed entropy calculation and visualization")
        if persistent_display_added:
            print("- Added persistent display widgets")
        if version_incremented:
            print("- Incremented version and updated changelog")
    else:
        print("No changes needed")
    
    # Save updated notebook (unless in dry-run mode)
    if changes_made and not dry_run:
        try:
            # Backup original first
            backup_path = NOTEBOOK_PATH + ".bak"
            print(f"Creating backup at {backup_path}")
            with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as src:
                with open(backup_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            
            save_notebook(notebook_data, NOTEBOOK_PATH)
            print("Notebook updated successfully")
        except Exception as e:
            print(f"Error saving notebook: {e}")
            return 1
    elif dry_run:
        print("Dry run complete - no changes saved")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())