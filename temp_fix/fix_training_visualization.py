#!/usr/bin/env python
# Fix training visualization issues in NeuralPlasticityDemo.ipynb

import json
from pathlib import Path

def fix_training_visualization(notebook_path):
    """Fix visualization issues causing excessively large images."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Look for the metrics visualization cell (typically near the end)
    metrics_viz_cell_idx = None
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and 'Visualize training metrics with epochs' in ''.join(cell['source']):
            metrics_viz_cell_idx = i
            print(f"Found metrics visualization cell at index {i}")
            break
    
    if metrics_viz_cell_idx is None:
        print("Could not find metrics visualization cell")
        return False
    
    # Get cell content
    content = ''.join(notebook['cells'][metrics_viz_cell_idx]['source'])
    
    # Fix the figure size and improve downsampling
    modified_content = content.replace(
        "fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), dpi=100, sharex=True)",
        "# Create a more reasonably sized figure\nfig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), dpi=80, sharex=True)"
    )
    
    # Make sure we're using the downsampled variables correctly
    fixes = [
        ('ax1.plot(metrics_history["step"], metrics_history["train_loss"]', 'ax1.plot(display_steps, display_train_loss'),
        ('ax1.plot(metrics_history["step"], metrics_history["eval_loss"]', 'ax1.plot(display_steps, display_eval_loss'),
        ('ax2.bar(metrics_history["step"], metrics_history["pruned_heads"]', 'ax2.bar(display_steps, display_pruned_heads'),
        ('ax2.bar(metrics_history["step"], metrics_history["revived_heads"]', 'ax2.bar(display_steps, display_revived_heads'),
        ('ax3.plot(metrics_history["step"], metrics_history["sparsity"]', 'ax3.plot(display_steps, display_sparsity'),
        ('if "perplexity" in metrics_history:', 'if "perplexity" in metrics_history and len(display_perplexity) > 0:'),
        ('ax4.plot(metrics_history["step"], metrics_history["perplexity"]', 'ax4.plot(display_steps, display_perplexity')
    ]
    
    for old, new in fixes:
        modified_content = modified_content.replace(old, new)
    
    # Add explicit max figure size limit
    if 'plt.tight_layout()' in modified_content:
        modified_content = modified_content.replace(
            'plt.tight_layout()',
            '# Set explicit figure size limits before layout\nplt.gcf().set_size_inches(10, 10, forward=True)\nplt.tight_layout()'
        )
    
    # Update the cell source
    notebook['cells'][metrics_viz_cell_idx]['source'] = modified_content.split('\n')
    
    # Find other large visualizations
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            content = ''.join(cell['source'])
            
            # Fix entropy visualization with figsize=(15, 6)
            if 'plt.figure(figsize=(15, 6))' in content:
                notebook['cells'][i]['source'] = content.replace(
                    'plt.figure(figsize=(15, 6))',
                    'plt.figure(figsize=(10, 6))'
                ).split('\n')
                print(f"Fixed large entropy visualization in cell {i}")
            
            # Fix visualize_gradient_norms function with high DPI
            if 'def visualize_gradient_norms' in content and 'plt.savefig(save_path, dpi=300' in content:
                notebook['cells'][i]['source'] = content.replace(
                    'plt.savefig(save_path, dpi=300',
                    'plt.savefig(save_path, dpi=100'
                ).replace(
                    'plt.figure(figsize=(12, 6))',
                    'plt.figure(figsize=(10, 5))'
                ).split('\n')
                print(f"Fixed visualize_gradient_norms function in cell {i}")
    
    # Update version number in notebook title
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'markdown' and i == 0:
            content = ''.join(cell['source'])
            if 'Neural Plasticity Demo: Dynamic Pruning & Regrowth (v0.0.' in content:
                current_version = content.split('(v0.0.')[1].split(')')[0]
                new_version = str(int(current_version) + 1)
                updated_content = content.replace(f'(v0.0.{current_version})', f'(v0.0.{new_version})')
                notebook['cells'][i]['source'] = updated_content.split('\n')
                print(f"Updated version from v0.0.{current_version} to v0.0.{new_version}")
    
    # Update version history in conclusion cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'markdown' and '## Version History' in ''.join(cell['source']):
            content = ''.join(cell['source'])
            
            # Get the version number we determined earlier
            new_version = ""
            for j, c in enumerate(notebook['cells']):
                if c['cell_type'] == 'markdown' and j == 0:
                    title_content = ''.join(c['source'])
                    if 'Neural Plasticity Demo: Dynamic Pruning & Regrowth (v0.0.' in title_content:
                        new_version = title_content.split('(v0.0.')[1].split(')')[0]
                        break
            
            if not new_version:
                new_version = "23"  # Default if we couldn't find it
            
            # Add the new version entry
            version_entry = f"- v0.0.{new_version}: Fixed visualization issues causing excessively large images, reduced figure sizes and DPI settings\n"
            
            # Insert the entry after the Version History heading
            if '## Version History\n\n' in content:
                updated_content = content.replace(
                    '## Version History\n\n',
                    f'## Version History\n\n{version_entry}\n'
                )
                notebook['cells'][i]['source'] = updated_content.split('\n')
                print(f"Added version history entry for v0.0.{new_version}")
    
    # Save the modified notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Notebook saved with visualization fixes")
    return True

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_training_visualization(notebook_path)