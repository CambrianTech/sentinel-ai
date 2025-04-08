#!/usr/bin/env python
# Fix visualization issues in NeuralPlasticityDemo.ipynb

import json
import re
from pathlib import Path

def fix_visualization_issues(notebook_path):
    """Fix multiple visualization issues in the notebook."""
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Fix 1: Initial Head Entropy visualization
        initial_entropy_viz_idx = None
        
        # Find the cell with Initial Head Entropy visualization
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if 'plt.title("Initial Head Entropy' in source:
                    initial_entropy_viz_idx = i
                    print(f"Found initial entropy visualization cell at index {i}")
                    break
        
        if initial_entropy_viz_idx is not None:
            # Fix the initial entropy visualization
            cell = notebook['cells'][initial_entropy_viz_idx]
            
            # Get the current content
            original_content = ''.join(cell['source'])
            
            # Fix by ensuring non-zero entropy values are displayed
            fixed_content = original_content.replace(
                'plt.title("Initial Head Entropy (higher = less focused attention)")',
                'plt.title("Initial Head Entropy (higher = less focused attention)")\n'
                '# Ensure entropy visualization has some range\n'
                'plt.clim(0, max(0.1, entropy_values.max().item()))'
            )
            
            # Update the cell
            cell['source'] = fixed_content.split('\n')
            print("Fixed initial entropy visualization")
        
        # Fix 2: Training metrics visualization with constrained_layout warning
        metrics_viz_idx = None
        
        # Find the training metrics visualization cell
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if 'Visualize training metrics with epochs' in source and 'plt.subplots(' in source:
                    metrics_viz_idx = i
                    print(f"Found training metrics visualization cell at index {i}")
                    break
        
        if metrics_viz_idx is not None:
            # Fix the metrics visualization to avoid constrained_layout warning
            fixed_content = """# Visualize training metrics with epochs using a compact layout
# Create figure without constrained_layout to avoid size warning
fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1,
    figsize=(10, 8),  # Increased height to prevent collapsed axes
    dpi=100,
    sharex=True,
    # constrained_layout removed to avoid warning
)

# Add extra space between subplots
plt.subplots_adjust(hspace=0.4)  # Add space between subplots

# Set maximum display limit to prevent excessively large plots
max_display_points = 100
display_steps = metrics_history["step"]
if len(display_steps) > max_display_points:
    # Downsample by selecting evenly spaced points
    indices = np.linspace(0, len(display_steps) - 1, max_display_points).astype(int)
    display_steps = [metrics_history["step"][i] for i in indices]
    display_train_loss = [metrics_history["train_loss"][i] for i in indices]
    display_eval_loss = [metrics_history["eval_loss"][i] for i in indices]
    display_pruned_heads = [metrics_history["pruned_heads"][i] for i in indices]
    display_revived_heads = [metrics_history["revived_heads"][i] for i in indices]
    display_sparsity = [metrics_history["sparsity"][i] for i in indices]
    display_epoch = [metrics_history["epoch"][i] for i in indices]
    display_perplexity = [metrics_history["perplexity"][i] for i in indices] if "perplexity" in metrics_history and metrics_history["perplexity"] else []
else:
    display_train_loss = metrics_history["train_loss"]
    display_eval_loss = metrics_history["eval_loss"]
    display_pruned_heads = metrics_history["pruned_heads"]
    display_revived_heads = metrics_history["revived_heads"]
    display_sparsity = metrics_history["sparsity"]
    display_epoch = metrics_history["epoch"]
    display_perplexity = metrics_history["perplexity"] if "perplexity" in metrics_history else []

# Plot losses
ax1.plot(display_steps, display_train_loss, label="Train Loss")
ax1.plot(display_steps, display_eval_loss, label="Eval Loss")
ax1.set_ylabel("Loss")
ax1.set_title("Training and Evaluation Loss")
ax1.legend()
ax1.grid(True)

# Mark epoch boundaries if available
if "epoch" in metrics_history and len(display_epoch) > 1:
    for i in range(1, len(display_epoch)):
        if display_epoch[i] != display_epoch[i-1]:
            # This is an epoch boundary
            for ax in [ax1, ax2, ax3]:
                ax.axvline(x=display_steps[i], color="k", linestyle="--", alpha=0.3)
                # Make sure ylim exists before using it
                try:
                    y_pos = ax.get_ylim()[1] * 0.9
                except:
                    y_pos = 1.0  # fallback position
                ax.text(display_steps[i], y_pos, f"Epoch {display_epoch[i]}", rotation=90, alpha=0.7)

# Plot pruning metrics
ax2.bar(display_steps, display_pruned_heads, alpha=0.5, label="Pruned Heads", color="blue")
ax2.bar(display_steps, display_revived_heads, alpha=0.5, label="Revived Heads", color="green")
ax2.set_ylabel("Count")
ax2.set_title("Head Pruning and Revival")
ax2.legend(loc="upper left")
ax2.grid(True)

# Plot sparsity and perplexity
ax3.plot(display_steps, display_sparsity, "r-", label="Sparsity")
ax3.set_xlabel("Step")
ax3.set_ylabel("Sparsity")
ax3.grid(True)

# Add perplexity line on secondary axis if available
if "perplexity" in metrics_history and len(metrics_history.get("perplexity", [])) > 0:
    ax4 = ax3.twinx()
    ax4.plot(display_steps, display_perplexity, "g-", label="Perplexity")
    ax4.set_ylabel("Perplexity")
    ax4.legend(loc="upper right")

# Add diagnostic information about figure size
print(f"Figure size: {fig.get_size_inches()} inches, DPI: {fig.get_dpi()}")

# Apply tight_layout AFTER all elements are added to prevent layout issues
plt.tight_layout()
plt.show()"""
            
            # Update the cell
            cell = notebook['cells'][metrics_viz_idx]
            cell['source'] = fixed_content.split('\n')
            print("Fixed training metrics visualization")
        
        # Save the notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print("Fixed visualization issues in notebook")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_visualization_issues(notebook_path)