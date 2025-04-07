#!/usr/bin/env python
# Directly modify the large cells in the notebook

import json
from pathlib import Path

def find_controller_cell(notebook_path):
    """Find the controller cell in the notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    controller_cell = None
    controller_idx = None
    
    for i, cell in enumerate(notebook['cells']):
        if (cell['cell_type'] == 'code' and 
            'Create a custom statistical pruning function' in ''.join(cell['source'])):
            controller_cell = cell
            controller_idx = i
            break
    
    return notebook, controller_idx, controller_cell

def replace_controller_cell(notebook_path):
    """Replace the controller cell with multiple smaller cells."""
    notebook, controller_idx, controller_cell = find_controller_cell(notebook_path)
    
    if controller_idx is None:
        print("Could not find controller cell")
        return False
    
    print(f"Found controller cell at index {controller_idx}")
    
    # Create the new cells
    cell1 = {
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
            "    return pruning_mask"
        ]
    }
    
    cell2 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
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
            "print(f\"Model has {initial_stats['total_heads']} attention heads across {controller.total_layers} layers\")"
        ]
    }
    
    cell3 = {
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
            "print(f\"Are all gradient values the same? {torch.allclose(debug_grads, debug_grads[0,0])}\")"
        ]
    }
    
    cell4 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Test our gradient-only pruning approach\n",
            "pruning_mask = gradient_based_pruning(\n",
            "    debug_grads, \n",
            "    prune_percent=PRUNE_PERCENT\n",
            ")"
        ]
    }
    
    cell5 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Visualize which heads would be pruned\n",
            "plt.figure(figsize=(10, 6))\n",
            "plt.imshow(pruning_mask.detach().cpu().numpy(), cmap='Reds', aspect='auto')\n",
            "plt.colorbar(label='Prune')\n",
            "plt.title('Gradient-Based Pruning Decisions')\n",
            "plt.xlabel('Head Index')\n",
            "plt.ylabel('Layer Index')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
    
    cell6 = {
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
            "plt.show()"
        ]
    }
    
    cell7 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Create a visualization highlighting pruning decisions based on gradient values\n",
            "plt.figure(figsize=(12, 8))\n",
            "grad_data = debug_grads.detach().cpu().numpy()\n",
            "mask_data = pruning_mask.detach().cpu().numpy()\n",
            "\n",
            "# Create a masked array where pruned heads are highlighted\n",
            "masked_grads = np.ma.array(grad_data, mask=~mask_data)\n",
            "\n",
            "# Base plot with all gradient values\n",
            "plt.imshow(grad_data, cmap='Blues', aspect='auto')\n",
            "# Overlay plot with pruned heads highlighted\n",
            "plt.imshow(masked_grads, cmap='Reds', aspect='auto')\n",
            "plt.colorbar(label='Gradient Norm')\n",
            "plt.title('Gradient Norms with Pruning Decisions Highlighted')\n",
            "plt.xlabel('Head Index')\n",
            "plt.ylabel('Layer Index')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
    
    cell8 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Debug attention distribution collection to see why entropy is zero\n",
            "print(\"\\nDebugging attention distributions...\")\n",
            "try:\n",
            "    # Try to get attention directly to check if it's working\n",
            "    inputs = next(iter(validation_dataloader))\n",
            "    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}\n",
            "    \n",
            "    model.eval()\n",
            "    with torch.no_grad():\n",
            "        outputs = model(**inputs, output_attentions=True)\n",
            "    \n",
            "    if hasattr(outputs, 'attentions') and outputs.attentions is not None:\n",
            "        print(f\"Attention shape: {outputs.attentions[0].shape}\")\n",
            "        # Check if attention is uniform (which would give zero entropy)\n",
            "        attn = outputs.attentions[0]  # First layer's attention\n",
            "        first_head = attn[0, 0]  # First batch, first head\n",
            "        \n",
            "        print(f\"First head attention max: {first_head.max().item():.4f}, min: {first_head.min().item():.4f}\")\n",
            "        print(f\"First head attention std: {first_head.std().item():.4f}\")\n",
            "        \n",
            "        # Visualize actual attention patterns\n",
            "        plt.figure(figsize=(15, 12))\n",
            "        \n",
            "        # Display first head attention\n",
            "        plt.subplot(2, 2, 1)\n",
            "        plt.imshow(first_head.detach().cpu().numpy(), cmap='Blues')\n",
            "        plt.colorbar(label='Attention Weight')\n",
            "        plt.title('Attention Pattern for First Head')\n",
            "        plt.xlabel('Token (Key)')\n",
            "        plt.ylabel('Token (Query)')\n",
            "        \n",
            "        # Show a different head for comparison (if available)\n",
            "        if attn.size(1) > 1:  # If there's more than one head\n",
            "            other_head = attn[0, 1]  # Second head\n",
            "            plt.subplot(2, 2, 2)\n",
            "            plt.imshow(other_head.detach().cpu().numpy(), cmap='Blues')\n",
            "            plt.colorbar(label='Attention Weight')\n",
            "            plt.title('Attention Pattern for Second Head')\n",
            "            plt.xlabel('Token (Key)')\n",
            "            plt.ylabel('Token (Query)')\n",
            "        \n",
            "        # Apply log scaling to see subtle differences\n",
            "        plt.subplot(2, 2, 3)\n",
            "        log_attn = torch.log10(first_head.clamp(min=1e-10))\n",
            "        plt.imshow(log_attn.detach().cpu().numpy(), cmap='viridis')\n",
            "        plt.colorbar(label='Log10 Attention Weight')\n",
            "        plt.title('Log-Scaled Attention for First Head')\n",
            "        plt.xlabel('Token (Key)')\n",
            "        plt.ylabel('Token (Query)')\n",
            "        \n",
            "        # Show histogram of attention values\n",
            "        plt.subplot(2, 2, 4)\n",
            "        plt.hist(first_head.flatten().detach().cpu().numpy(), bins=50)\n",
            "        plt.title('Distribution of Attention Values (First Head)')\n",
            "        plt.xlabel('Attention Weight')\n",
            "        plt.ylabel('Frequency')\n",
            "        \n",
            "        plt.tight_layout()\n",
            "        plt.show()\n",
            "        \n",
            "        # Check if there's any NaN or inf\n",
            "        print(f\"Contains NaN: {torch.isnan(attn).any().item()}\")\n",
            "        print(f\"Contains Inf: {torch.isinf(attn).any().item()}\")\n",
            "    else:\n",
            "        print(\"Model did not return attention outputs\")\n",
            "except Exception as e:\n",
            "    print(f\"Error during attention debugging: {e}\")"
        ]
    }
    
    # Replace the original cell with our new cells
    notebook['cells'][controller_idx] = cell1
    notebook['cells'].insert(controller_idx + 1, cell2)
    notebook['cells'].insert(controller_idx + 2, cell3)
    notebook['cells'].insert(controller_idx + 3, cell4)
    notebook['cells'].insert(controller_idx + 4, cell5)
    notebook['cells'].insert(controller_idx + 5, cell6)
    notebook['cells'].insert(controller_idx + 6, cell7)
    notebook['cells'].insert(controller_idx + 7, cell8)
    
    print(f"Replaced controller cell with 8 smaller cells")
    
    # Save the modified notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Notebook saved with replaced controller cell")
    return True

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    replace_controller_cell(notebook_path)