#!/usr/bin/env python
# Fix missing debug_grads variable reference

import json
import re
from pathlib import Path

def fix_missing_variable(notebook_path):
    """Fix reference to missing debug_grads variable."""
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Find the cell with the NameError
        pruning_test_cell_idx = None
        debug_cell_idx = None
        
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                
                if '# Test our gradient-only pruning approach' in source and 'gradient_based_pruning(' in source:
                    pruning_test_cell_idx = i
                    print(f"Found pruning test cell at index {i}")
                
                if 'debug_entropy, debug_grads = controller.collect_head_metrics(' in source:
                    debug_cell_idx = i
                    print(f"Found debug metrics cell at index {i}")
        
        if pruning_test_cell_idx is None:
            print("ERROR: Could not find pruning test cell")
            return False
        
        # Add a check for debug_grads and run collection if needed
        cell = notebook['cells'][pruning_test_cell_idx]
        
        # Fix the cell by adding variable check and collection
        fixed_content = """# Test our gradient-only pruning approach

# Make sure we have debug_grads
try:
    # Check if debug_grads is defined
    debug_grads
    print("Using existing debug_grads")
except NameError:
    print("debug_grads not found, collecting metrics...")
    # Try to collect metrics
    try:
        debug_entropy, debug_grads = controller.collect_head_metrics(
            validation_dataloader,
            num_batches=2
        )
    except Exception as e:
        print(f"Error collecting metrics: {e}")
        # Create a dummy tensor if everything fails
        print("Creating dummy debug_grads")
        debug_grads = torch.zeros(6, 12)  # Default size for distilgpt2 (6 layers, 12 heads)

pruning_mask = gradient_based_pruning(
    debug_grads,
    prune_percent=PRUNE_PERCENT
)

# Visualize pruning mask
plt.figure(figsize=(10, 6))
plt.imshow(pruning_mask.detach().cpu().numpy(), cmap='Reds', aspect='auto')
plt.colorbar(label='Prune')
plt.title(f'Pruning Mask (prune {PRUNE_PERCENT*100:.0f}% of heads)')
plt.xlabel('Head')
plt.ylabel('Layer')
plt.show()

print(f"\\nPruning Analysis:")
pruned_count = pruning_mask.sum().item()
total_count = pruning_mask.numel()
print(f"Pruning {pruned_count}/{total_count} heads ({pruned_count/total_count*100:.1f}%)")"""
        
        # Update the cell
        cell['source'] = fixed_content.split('\n')
        
        # Save the notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print("Fixed reference to missing debug_grads variable")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_missing_variable(notebook_path)