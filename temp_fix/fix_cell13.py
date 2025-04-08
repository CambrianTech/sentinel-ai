#!/usr/bin/env python
# Fix cell 13 directly

import json
from pathlib import Path

def fix_cell13(notebook_path):
    """Fix cell 13 directly by replacing it with proper content."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    if len(notebook['cells']) > 13:
        cell13 = notebook['cells'][13]
        content = ''.join(cell13['source'])
        
        print(f"Cell 13 source type: {type(cell13['source'])}")
        print(f"Cell 13 source length: {len(cell13['source'])}")
        print(f"First 30 characters: {content[:30]}")
        
        # Directly replace cell 13 with a properly formatted version
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Create a custom statistical pruning function based only on gradients\n",
                "def gradient_based_pruning(grad_norm_values, prune_percent=0.1):\n",
                "    \"\"\"\n",
                "    Make pruning decisions based only on gradient norms.\n",
                "    We want to prune heads with LOWEST gradient norms, as they're\n",
                "    learning the least.\n",
                "    \n",
                "    Args:\n",
                "        grad_norm_values: Tensor of gradient norm values for all heads\n",
                "        prune_percent: Target percentage of heads to prune (0-1)\n",
                "        \n",
                "    Returns:\n",
                "        pruning_mask: Boolean tensor where True indicates a head should be pruned\n",
                "    \"\"\"\n",
                "    # Flatten tensor for calculating percentiles\n",
                "    flat_grad_norm = grad_norm_values.view(-1)\n",
                "    \n",
                "    # Calculate how many heads we want to prune\n",
                "    total_heads = grad_norm_values.numel()\n",
                "    target_prune_count = int(total_heads * prune_percent)\n",
                "    \n",
                "    # Get the indices of the heads with the LOWEST gradient norms\n",
                "    # Here's the fix: we use largest=False to get the lowest values\n",
                "    _, indices = torch.topk(flat_grad_norm, k=target_prune_count, largest=False)\n",
                "    \n",
                "    # Create pruning mask where True = head should be pruned (low gradient norm)\n",
                "    pruning_mask = torch.zeros_like(grad_norm_values, dtype=torch.bool)\n",
                "    pruning_mask.view(-1)[indices] = True\n",
                "    \n",
                "    print(f\"Gradient-based pruning - target: {target_prune_count} heads\")\n",
                "    print(f\"Final pruning decision: pruning {pruning_mask.sum().item()} heads\")\n",
                "    print(f\"Average grad norm of pruned heads: {grad_norm_values[pruning_mask].mean().item():.6f}\")\n",
                "    print(f\"Average grad norm of kept heads: {grad_norm_values[~pruning_mask].mean().item():.6f}\")\n",
                "    return pruning_mask\n",
                "\n",
                "# Create plasticity controller with default thresholds\n",
                "controller = create_plasticity_controller(\n",
                "    model=model,\n",
                "    mode=PRUNING_MODE,\n",
                "    high_entropy_threshold=0.8,  # These will be ignored by our custom approach\n",
                "    low_entropy_threshold=0.4,   # but we need to provide values\n",
                "    grad_threshold=1e-3,\n",
                "    min_zero_epochs=MIN_ZERO_EPOCHS\n",
                ")\n",
                "\n",
                "# Display initial model stats\n",
                "initial_stats = controller.get_summary()\n",
                "print(f\"Model has {initial_stats['total_heads']} attention heads across {controller.total_layers} layers\")\n"
            ]
        }
        
        # Insert a new cell for the debug section
        new_cell2 = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Debug: Let's check the actual entropy values we're dealing with\n",
                "print(\"\\nCollecting initial entropy and gradient metrics for debugging...\")\n",
                "debug_entropy, debug_grads = controller.collect_head_metrics(\n",
                "    validation_dataloader,\n",
                "    num_batches=2\n",
                ")\n",
                "\n",
                "# Calculate statistics to help with threshold setting\n",
                "print(\"\\nEntropy statistics:\")\n",
                "print(f\"Mean entropy: {debug_entropy.mean().item():.4f}\")\n",
                "print(f\"Min entropy: {debug_entropy.min().item():.4f}\")\n",
                "print(f\"Max entropy: {debug_entropy.max().item():.4f}\")\n",
                "print(f\"25th percentile: {torch.quantile(debug_entropy.flatten(), 0.25).item():.4f}\")\n",
                "print(f\"50th percentile: {torch.quantile(debug_entropy.flatten(), 0.5).item():.4f}\")\n",
                "print(f\"75th percentile: {torch.quantile(debug_entropy.flatten(), 0.75).item():.4f}\")\n",
                "print(f\"Are all entropy values the same? {torch.allclose(debug_entropy, debug_entropy[0,0])}\")\n",
                "print(f\"Non-zero values: {torch.count_nonzero(debug_entropy)}/{debug_entropy.numel()}\")\n",
                "\n",
                "print(\"\\nGradient norm statistics:\")\n",
                "print(f\"Mean grad norm: {debug_grads.mean().item():.6f}\")\n",
                "print(f\"Min grad norm: {debug_grads.min().item():.6f}\")\n",
                "print(f\"Max grad norm: {debug_grads.max().item():.6f}\")\n",
                "print(f\"25th percentile: {torch.quantile(debug_grads.flatten(), 0.25).item():.6f}\")\n",
                "print(f\"50th percentile: {torch.quantile(debug_grads.flatten(), 0.5).item():.6f}\")\n",
                "print(f\"75th percentile: {torch.quantile(debug_grads.flatten(), 0.75).item():.6f}\")\n",
                "print(f\"Are all gradient values the same? {torch.allclose(debug_grads, debug_grads[0,0])}\")\n"
            ]
        }
        
        # Insert a new cell for visualizations
        new_cell3 = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Test our gradient-only pruning approach\n",
                "pruning_mask = gradient_based_pruning(\n",
                "    debug_grads, \n",
                "    prune_percent=PRUNE_PERCENT\n",
                ")\n",
                "\n",
                "# Visualize which heads would be pruned\n",
                "plt.figure(figsize=(10, 6))\n",
                "plt.imshow(pruning_mask.detach().cpu().numpy(), cmap='Reds', aspect='auto')\n",
                "plt.colorbar(label='Prune')\n",
                "plt.title('Gradient-Based Pruning Decisions')\n",
                "plt.xlabel('Head Index')\n",
                "plt.ylabel('Layer Index')\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        }
        
        # Insert a new cell for more visualizations
        new_cell4 = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Create a visual comparing entropy and gradient distributions\n",
                "plt.figure(figsize=(15, 6))\n",
                "\n",
                "# Entropy subplot - enhance with scale adjustment\n",
                "plt.subplot(1, 2, 1)\n",
                "entropy_data = debug_entropy.detach().cpu().numpy()\n",
                "vmax = max(0.1, entropy_data.max())  # Increase minimum scale to make patterns visible\n",
                "im1 = plt.imshow(entropy_data, cmap='viridis', aspect='auto', vmin=0, vmax=vmax)\n",
                "plt.colorbar(im1, label='Entropy')\n",
                "plt.title(f'Attention Entropy Values (max={entropy_data.max():.4f})')\n",
                "plt.xlabel('Head Index')\n",
                "plt.ylabel('Layer Index')\n",
                "\n",
                "# Gradient subplot\n",
                "plt.subplot(1, 2, 2)\n",
                "im2 = plt.imshow(debug_grads.detach().cpu().numpy(), cmap='plasma', aspect='auto')\n",
                "plt.colorbar(im2, label='Gradient Norm')\n",
                "plt.title('Gradient Norms')\n",
                "plt.xlabel('Head Index')\n",
                "plt.ylabel('Layer Index')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        }
        
        # Update the notebook
        notebook['cells'][13] = new_cell
        notebook['cells'].insert(14, new_cell2)
        notebook['cells'].insert(15, new_cell3)
        notebook['cells'].insert(16, new_cell4)
        
        # Save the notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print("Fixed cell 13 and added 3 new cells to split the content")
    else:
        print(f"Notebook only has {len(notebook['cells'])} cells, cannot access cell 13")

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_cell13(notebook_path)