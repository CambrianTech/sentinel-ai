import json
import sys
import os

# Load the notebook
notebook_path = '/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb'
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Find the visualization cell (it's cell 19 according to our previous analysis)
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source'])
        # Print the first 100 characters of each code cell to help identify it
        print(f"Cell {i}: {source_text[:200]}...")
        if '# Visualize training metrics with epochs' in source_text:
            print(f"Found visualization cell: {i}")
            cell_to_modify = cell
            break

# For testing, let's modify cell 19 (or whichever cell contains the visualization)
cell_to_modify = notebook['cells'][19]  # Assuming it's cell 19

# Get the current source
source = cell_to_modify['source']
print(f"Source starts with: {source[0]}")

# Fix the first part of the visualization code
new_source = []
for line in source:
    if '# Visualize training metrics with epochs' in line:
        new_source.append(line)
        new_source.append('fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), dpi=100, sharex=True)\n')
        new_source.append('\n')
        new_source.append('# Set maximum display limit to prevent excessively large plots\n')
        new_source.append('max_display_points = 100\n')
        new_source.append('display_steps = metrics_history["step"]\n')
        new_source.append('if len(display_steps) > max_display_points:\n')
        new_source.append('    # Downsample by selecting evenly spaced points\n')
        new_source.append('    indices = np.linspace(0, len(display_steps) - 1, max_display_points).astype(int)\n')
        new_source.append('    display_steps = [metrics_history["step"][i] for i in indices]\n')
        new_source.append('    display_train_loss = [metrics_history["train_loss"][i] for i in indices]\n')
        new_source.append('    display_eval_loss = [metrics_history["eval_loss"][i] for i in indices]\n')
        new_source.append('    display_pruned_heads = [metrics_history["pruned_heads"][i] for i in indices]\n')
        new_source.append('    display_revived_heads = [metrics_history["revived_heads"][i] for i in indices]\n')
        new_source.append('    display_sparsity = [metrics_history["sparsity"][i] for i in indices]\n')
        new_source.append('    display_epoch = [metrics_history["epoch"][i] for i in indices]\n')
        new_source.append('    display_perplexity = [metrics_history["perplexity"][i] for i in indices] if "perplexity" in metrics_history and metrics_history["perplexity"] else []\n')
        new_source.append('else:\n')
        new_source.append('    display_train_loss = metrics_history["train_loss"]\n')
        new_source.append('    display_eval_loss = metrics_history["eval_loss"]\n')
        new_source.append('    display_pruned_heads = metrics_history["pruned_heads"]\n')
        new_source.append('    display_revived_heads = metrics_history["revived_heads"]\n')
        new_source.append('    display_sparsity = metrics_history["sparsity"]\n')
        new_source.append('    display_epoch = metrics_history["epoch"]\n')
        new_source.append('    display_perplexity = metrics_history["perplexity"] if "perplexity" in metrics_history else []\n')
    elif 'fig, (ax1, ax2, ax3) = plt.subplots' in line:
        # Skip this line as we've already added it
        continue
    elif 'ax1.plot(metrics_history["step"], metrics_history["train_loss"]' in line:
        new_source.append('ax1.plot(display_steps, display_train_loss, label="Train Loss")\n')
    elif 'ax1.plot(metrics_history["step"], metrics_history["eval_loss"]' in line:
        new_source.append('ax1.plot(display_steps, display_eval_loss, label="Eval Loss")\n')
    elif 'ax2.bar(metrics_history["step"], metrics_history["pruned_heads"]' in line:
        new_source.append('ax2.bar(display_steps, display_pruned_heads, alpha=0.5, label="Pruned Heads", color="blue")\n')
    elif 'ax2.bar(metrics_history["step"], metrics_history["revived_heads"]' in line:
        new_source.append('ax2.bar(display_steps, display_revived_heads, alpha=0.5, label="Revived Heads", color="green")\n')
    elif 'ax3.plot(metrics_history["step"], metrics_history["sparsity"]' in line:
        new_source.append('ax3.plot(display_steps, display_sparsity, "r-", label="Sparsity")\n')
    elif 'if "epoch" in metrics_history and len(metrics_history["epoch"]) > 1:' in line:
        new_source.append('if "epoch" in metrics_history and len(display_epoch) > 1:\n')
    elif 'for i in range(1, len(metrics_history["epoch"])):' in line:
        new_source.append('    for i in range(1, len(display_epoch)):\n')
    elif '        if metrics_history["epoch"][i] != metrics_history["epoch"][i-1]:' in line:
        new_source.append('        if display_epoch[i] != display_epoch[i-1]:\n')
    elif '                ax.axvline(x=metrics_history["step"][i],' in line:
        new_source.append('                ax.axvline(x=display_steps[i], color="k", linestyle="--", alpha=0.3)\n')
    elif '                ax.text(metrics_history["step"][i],' in line:
        new_source.append('                ax.text(display_steps[i], ax.get_ylim()[1]*0.9, \n')
    elif 'ax4.plot(metrics_history["step"], metrics_history["perplexity"]' in line:
        new_source.append('    ax4.plot(display_steps, display_perplexity, "g-", label="Perplexity")\n')
    elif 'plt.tight_layout()' in line:
        # Add additional figure size controls before tight_layout
        new_source.append('# Ensure figure has reasonable dimensions\n')
        new_source.append('plt.gcf().set_dpi(100)\n')
        new_source.append('plt.tight_layout()\n')
    else:
        new_source.append(line)

# Update the cell source
cell_to_modify['source'] = new_source

# Save the modified notebook with a different name
output_path = notebook_path
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook updated successfully: {output_path}")