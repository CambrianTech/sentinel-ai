#!/usr/bin/env python
# Fix duplication in visualization cell - hardcoded fix

import json
from pathlib import Path

def fix_visualization_duplication(notebook_path):
    """Fix the specific duplication issue in the visualization cell."""
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Find the visualization cell
        visualization_cell_idx = None
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if 'Visualize training metrics with epochs' in source and 'plt.subplots(' in source:
                    visualization_cell_idx = i
                    print(f"Found visualization cell at index {i}")
                    break
        
        if visualization_cell_idx is None:
            print("Visualization cell not found")
            return False
        
        # Get the cell content and check for the specific duplication pattern
        cell = notebook['cells'][visualization_cell_idx]
        
        # Create correct content - hardcoded known-good version
        correct_content = """# Visualize training metrics with epochs using a compact layout
fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1,
    figsize=(10, 6),
    dpi=100,
    sharex=True,
    constrained_layout=True  # Prevents massive vertical images in Colab
)

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
                ax.text(display_steps[i], ax.get_ylim()[1]*0.9, 
                        f"Epoch {display_epoch[i]}", rotation=90, alpha=0.7)

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
if "perplexity" in metrics_history and metrics_history["perplexity"]:
    ax4 = ax3.twinx()
    ax4.plot(display_steps, display_perplexity, "g-", label="Perplexity")
    ax4.set_ylabel("Perplexity")
    ax4.legend(loc="upper right")

# Add diagnostic information about figure size
print(f"Figure size: {fig.get_size_inches()} inches, DPI: {fig.get_dpi()}")
plt.show()"""
        
        # Replace the cell content with the correct version
        cell['source'] = correct_content.split('\n')
        
        # Save the notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print("Fixed visualization cell duplication")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_visualization_duplication(notebook_path)