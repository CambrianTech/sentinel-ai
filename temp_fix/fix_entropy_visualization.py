#!/usr/bin/env python
# Fix entropy calculation and visualization for better insights

import json
import re
from pathlib import Path

def fix_entropy_visualization(notebook_path):
    """Fix entropy calculation and add better visualizations for debugging."""
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Find our debug and visualization cells
        debug_cell_idx = None
        controller_cell_idx = None
        visualization_cell_idx = None
        
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                
                if 'debug_entropy, debug_grads = controller.collect_head_metrics(' in source:
                    debug_cell_idx = i
                    print(f"Found debug cell at index {i}")
                    
                if 'controller = create_plasticity_controller(' in source:
                    controller_cell_idx = i
                    print(f"Found controller cell at index {i}")
                
                # Look for visualization cell with entropy/gradient visualization
                if 'Create a visual comparing entropy and gradient distributions' in source:
                    visualization_cell_idx = i
                    print(f"Found visualization cell at index {i}")
        
        # Add an improved entropy visualizer cell right after the debug cell if found
        if debug_cell_idx is not None:
            # Create an additional entropy analysis cell
            entropy_analyzer_cell = {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Enhanced Entropy Analysis\n",
                    "# Run this after collecting the metrics to better understand the entropy issues\n",
                    "\n",
                    "# Function to compute improved entropy with diagnostics\n",
                    "def compute_improved_entropy(attn_probs, eps=1e-8, debug=True):\n",
                    "    \"\"\"Compute entropy with better numerical stability and detailed diagnostics.\"\"\"\n",
                    "    if debug:\n",
                    "        # Print raw attention stats\n",
                    "        print(f\"Raw attention shape: {attn_probs.shape}\")\n",
                    "        print(f\"Raw min/max/mean: {attn_probs.min().item():.6e}/{attn_probs.max().item():.6e}/{attn_probs.mean().item():.6e}\")\n",
                    "        \n",
                    "        # Check for numerical issues\n",
                    "        print(f\"Contains zeros: {(attn_probs == 0).any().item()}\")\n",
                    "        print(f\"Contains NaN: {torch.isnan(attn_probs).any().item()}\")\n",
                    "        print(f\"Contains Inf: {torch.isinf(attn_probs).any().item()}\")\n",
                    "        \n",
                    "        # Check distribution validity\n",
                    "        row_sums = attn_probs.sum(dim=-1)\n",
                    "        print(f\"Row sums min/max/mean: {row_sums.min().item():.6f}/{row_sums.max().item():.6f}/{row_sums.mean().item():.6f}\")\n",
                    "        print(f\"Rows sum to ~1: {torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-2)}\")\n",
                    "    \n",
                    "    # Apply numerical safeguards\n",
                    "    # 1. Ensure positive values\n",
                    "    attn_probs = attn_probs.clamp(min=eps)\n",
                    "    \n",
                    "    # 2. Normalize to ensure it sums to 1.0 along attention dimension\n",
                    "    attn_probs = attn_probs / attn_probs.sum(dim=-1, keepdim=True)\n",
                    "    \n",
                    "    if debug:\n",
                    "        print(\"\\nAfter preprocessing:\")\n",
                    "        print(f\"Min/max/mean: {attn_probs.min().item():.6e}/{attn_probs.max().item():.6e}/{attn_probs.mean().item():.6e}\")\n",
                    "        row_sums = attn_probs.sum(dim=-1)\n",
                    "        print(f\"Row sums min/max/mean: {row_sums.min().item():.6f}/{row_sums.max().item():.6f}/{row_sums.mean().item():.6f}\")\n",
                    "    \n",
                    "    # Compute entropy: -sum(p * log(p))\n",
                    "    log_probs = torch.log(attn_probs)\n",
                    "    entropy = -torch.sum(attn_probs * log_probs, dim=-1)\n",
                    "    \n",
                    "    if debug:\n",
                    "        print(\"\\nEntropy results:\")\n",
                    "        print(f\"Entropy shape: {entropy.shape}\")\n",
                    "        print(f\"Entropy min/max/mean: {entropy.min().item():.4f}/{entropy.max().item():.4f}/{entropy.mean().item():.4f}\")\n",
                    "        \n",
                    "        # Compute theoretical maximum entropy (uniform distribution)\n",
                    "        seq_len = attn_probs.size(-1)\n",
                    "        max_entropy = torch.log(torch.tensor(seq_len, dtype=torch.float))\n",
                    "        print(f\"Theoretical max entropy (log(seq_len)): {max_entropy.item():.4f}\")\n",
                    "        \n",
                    "        # Check if entropy is at maximum (uniform attention)\n",
                    "        print(f\"Percentage of maximum entropy: {entropy.mean().item()/max_entropy.item()*100:.2f}%\")\n",
                    "    \n",
                    "    return entropy\n",
                    "\n",
                    "# Get the raw attention patterns from the model for analysis\n",
                    "try:\n",
                    "    # Get a batch of data\n",
                    "    inputs = next(iter(validation_dataloader))\n",
                    "    if isinstance(inputs, dict):\n",
                    "        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}\n",
                    "    else:\n",
                    "        inputs = {\"input_ids\": inputs[0].to(device)}\n",
                    "    \n",
                    "    # Run model with attention outputs\n",
                    "    model.eval()\n",
                    "    with torch.no_grad():\n",
                    "        outputs = model(**inputs, output_attentions=True)\n",
                    "    \n",
                    "    # Extract attention patterns\n",
                    "    if hasattr(outputs, 'attentions') and outputs.attentions is not None:\n",
                    "        attn_list = outputs.attentions\n",
                    "        if len(attn_list) > 0:\n",
                    "            # Create a detailed visualization of attention patterns and entropy\n",
                    "            num_layers = len(attn_list)\n",
                    "            fig, axes = plt.subplots(num_layers, 2, figsize=(12, num_layers*3), constrained_layout=True)\n",
                    "            \n",
                    "            layer_entropies = []\n",
                    "            layer_entropies_norm = []\n",
                    "            \n",
                    "            for layer_idx in range(num_layers):\n",
                    "                attn = attn_list[layer_idx]\n",
                    "                \n",
                    "                # Compute entropy for this layer's attention\n",
                    "                print(f\"\\n=== Analyzing Layer {layer_idx} Attention ====\")\n",
                    "                layer_entropy = compute_improved_entropy(attn, debug=True)\n",
                    "                \n",
                    "                # Save mean entropy per head\n",
                    "                head_entropies = layer_entropy.mean(dim=(0, 1))  # Average over batch and sequence\n",
                    "                layer_entropies.append(head_entropies)\n",
                    "                \n",
                    "                # Normalize by max possible entropy\n",
                    "                seq_len = attn.size(-1)\n",
                    "                max_entropy = torch.log(torch.tensor(seq_len, dtype=torch.float, device=attn.device))\n",
                    "                norm_entropies = head_entropies / max_entropy.item()\n",
                    "                layer_entropies_norm.append(norm_entropies)\n",
                    "                \n",
                    "                # Plot attention pattern for first head\n",
                    "                if isinstance(axes, np.ndarray) and len(axes.shape) > 1:  # multiple rows and cols\n",
                    "                    ax1 = axes[layer_idx, 0]\n",
                    "                    ax2 = axes[layer_idx, 1]\n",
                    "                else:  # only 1 layer, so axes is 1D\n",
                    "                    ax1 = axes[0]\n",
                    "                    ax2 = axes[1]\n",
                    "                \n",
                    "                # Plot attention pattern\n",
                    "                attn_pattern = attn[0, 0].cpu().numpy()  # First batch, first head\n",
                    "                im = ax1.imshow(attn_pattern, cmap='viridis')\n",
                    "                ax1.set_title(f'Layer {layer_idx} - Head 0 Attention')\n",
                    "                ax1.set_xlabel('Position (To)')\n",
                    "                ax1.set_ylabel('Position (From)')\n",
                    "                plt.colorbar(im, ax=ax1)\n",
                    "                \n",
                    "                # Plot entropy values for all heads\n",
                    "                ax2.bar(range(len(head_entropies)), head_entropies.cpu().numpy())\n",
                    "                ax2.axhline(y=max_entropy.item(), color='r', linestyle='--', alpha=0.7, label='Max Entropy')\n",
                    "                ax2.set_title(f'Layer {layer_idx} - Head Entropies')\n",
                    "                ax2.set_xlabel('Head Index')\n",
                    "                ax2.set_ylabel('Entropy')\n",
                    "                ax2.legend()\n",
                    "                \n",
                    "                # Add entropy values as text on the bars\n",
                    "                for i, v in enumerate(head_entropies):\n",
                    "                    ax2.text(i, v.item() + 0.1, f'{v.item():.2f}', ha='center')\n",
                    "            \n",
                    "            plt.tight_layout()\n",
                    "            plt.show()\n",
                    "            \n",
                    "            # Create a heatmap of entropy across all layers and heads\n",
                    "            if num_layers > 1:\n",
                    "                all_entropies = torch.stack(layer_entropies).cpu().numpy()\n",
                    "                plt.figure(figsize=(10, 6))\n",
                    "                plt.imshow(all_entropies, cmap='viridis', aspect='auto')\n",
                    "                plt.colorbar(label='Entropy')\n",
                    "                plt.title('Entropy Heatmap Across All Layers and Heads')\n",
                    "                plt.xlabel('Head Index')\n",
                    "                plt.ylabel('Layer Index')\n",
                    "                \n",
                    "                # Add text annotations for each cell\n",
                    "                for i in range(all_entropies.shape[0]):\n",
                    "                    for j in range(all_entropies.shape[1]):\n",
                    "                        text = plt.text(j, i, f'{all_entropies[i, j]:.2f}',\n",
                    "                                      ha=\"center\", va=\"center\", color=\"w\")\n",
                    "                \n",
                    "                plt.tight_layout()\n",
                    "                plt.show()\n",
                    "                \n",
                    "                # Plot normalized entropy (as percentage of maximum)\n",
                    "                all_norm_entropies = torch.stack(layer_entropies_norm).cpu().numpy() * 100  # as percentage\n",
                    "                plt.figure(figsize=(10, 6))\n",
                    "                plt.imshow(all_norm_entropies, cmap='viridis', aspect='auto', vmin=0, vmax=100)\n",
                    "                plt.colorbar(label='% of Max Entropy')\n",
                    "                plt.title('Normalized Entropy (% of Maximum)')\n",
                    "                plt.xlabel('Head Index')\n",
                    "                plt.ylabel('Layer Index')\n",
                    "                \n",
                    "                # Add text annotations for each cell\n",
                    "                for i in range(all_norm_entropies.shape[0]):\n",
                    "                    for j in range(all_norm_entropies.shape[1]):\n",
                    "                        text = plt.text(j, i, f'{all_norm_entropies[i, j]:.1f}%',\n",
                    "                                      ha=\"center\", va=\"center\", color=\"w\")\n",
                    "                \n",
                    "                plt.tight_layout()\n",
                    "                plt.show()\n",
                    "        else:\n",
                    "            print(\"No attention tensors returned by the model\")\n",
                    "    else:\n",
                    "        print(\"Model outputs don't include attention weights\")\n",
                    "except Exception as e:\n",
                    "    print(f\"Error in entropy analysis: {e}\")"
                ]
            }
            
            # Insert right after the debug cell
            notebook['cells'].insert(debug_cell_idx + 1, entropy_analyzer_cell)
            print(f"Added enhanced entropy analyzer cell after debug cell")
        
        # Fix controller creation cell to use correct entropy calculation if found
        if controller_cell_idx is not None:
            controller_cell = notebook['cells'][controller_cell_idx]
            
            # Add proper entropy calculation patch
            patch_content = [
                "\n# Override the controller's entropy calculation to fix zero entropy issues\n",
                "# We're replacing the internal calculation method with a more numerically stable version\n",
                "def better_entropy_calculation(attn_probs, eps=1e-6):\n",
                "    \"\"\"Calculate entropy with better numerical stability.\"\"\"\n",
                "    # Add small epsilon to avoid log(0) issues\n",
                "    attn_probs = attn_probs.clamp(min=eps)\n",
                "    \n",
                "    # Normalize to ensure it's a proper probability distribution\n",
                "    attn_probs = attn_probs / attn_probs.sum(dim=-1, keepdim=True)\n",
                "    \n",
                "    # Standard entropy calculation\n",
                "    return -torch.sum(attn_probs * torch.log(attn_probs), dim=-1)\n",
                "\n",
                "# Monkey-patch the controller's entropy calculation method\n",
                "import types\n",
                "if hasattr(controller, 'calculate_attention_entropy'):\n",
                "    # For controllers with direct entropy calculation method\n",
                "    controller.calculate_attention_entropy = types.MethodType(\n",
                "        lambda self, attention_maps: better_entropy_calculation(attention_maps), controller)\n",
                "    print(\"Patched controller's entropy calculation\")\n",
                "elif hasattr(controller, '_compute_entropy'):\n",
                "    # For controllers with internal _compute_entropy method\n",
                "    old_compute_entropy = controller._compute_entropy\n",
                "    \n",
                "    def patched_compute_entropy(self, attention_maps, eps=1e-6):\n",
                "        \"\"\"Improved entropy calculation with better numerical stability.\"\"\"\n",
                "        if not attention_maps:\n",
                "            return 0.0\n",
                "        \n",
                "        # Concatenate all maps\n",
                "        maps = torch.cat(attention_maps, dim=0)\n",
                "        \n",
                "        # Apply the better entropy calculation to the raw attention maps\n",
                "        entropies = better_entropy_calculation(maps, eps=eps)\n",
                "        \n",
                "        # Average over batch and sequence length\n",
                "        avg_entropy = entropies.mean().item()\n",
                "        \n",
                "        # Normalize to [0,1] range\n",
                "        max_entropy = torch.log(torch.tensor(maps.size(-1), dtype=torch.float))\n",
                "        normalized_entropy = avg_entropy / max_entropy.item()\n",
                "        \n",
                "        return normalized_entropy\n",
                "    \n",
                "    controller._compute_entropy = types.MethodType(patched_compute_entropy, controller)\n",
                "    print(\"Patched controller's _compute_entropy method\")\n"
            ]
            
            # Check if the patch is already there
            cell_content = ''.join(controller_cell['source'])
            if 'better_entropy_calculation' not in cell_content:
                # Get all lines
                all_lines = controller_cell['source']
                
                # Find where to insert the patch
                for i, line in enumerate(all_lines):
                    if 'controller = create_plasticity_controller(' in line and ')' in line:
                        # Found the end of controller creation
                        insert_pos = i + 1
                        break
                else:
                    # Default to end of cell
                    insert_pos = len(all_lines)
                
                # Insert the patch
                for line in patch_content:
                    all_lines.insert(insert_pos, line)
                    insert_pos += 1
                
                controller_cell['source'] = all_lines
                print(f"Added entropy calculation fix to controller cell")
        
        # Save the modified notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print(f"Fixed entropy visualization issues in notebook")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    fix_entropy_visualization(notebook_path)