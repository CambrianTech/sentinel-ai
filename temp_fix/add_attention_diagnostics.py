#!/usr/bin/env python
# Add diagnostics to debug attention probability issues in NeuralPlasticityDemo

import json
import re
from pathlib import Path

def add_attention_diagnostics(notebook_path):
    """Add diagnostic code to visualize attention probability distributions."""
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Find cells where attention is collected and entropy is calculated
        debug_cell_idx = None
        controller_cell_idx = None
        
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                
                if 'debug_entropy, debug_grads = controller.collect_head_metrics(' in source:
                    debug_cell_idx = i
                    print(f"Found debug cell at index {i}")
                
                if 'controller = create_plasticity_controller(' in source:
                    controller_cell_idx = i
                    print(f"Found controller creation cell at index {i}")
        
        if debug_cell_idx is None:
            print("ERROR: Could not find debug cell to modify")
            return False
            
        # Add diagnostic code to the debug cell
        cell = notebook['cells'][debug_cell_idx]
        source = cell['source']
        
        # Insert diagnostic code after collecting entropy
        new_source = []
        insertion_point = None
        
        for i, line in enumerate(source):
            new_source.append(line)
            
            if 'debug_entropy, debug_grads = controller.collect_head_metrics(' in line:
                insertion_point = i + 1
        
        if insertion_point is not None:
            # Insert diagnostic code right after collect_head_metrics
            diagnostic_code = [
                "\n# Add diagnostic to debug attention probability tensor\n",
                "print(\"\\nDIAGNOSTIC: Checking raw attention probability distributions...\")\n",
                "inputs = next(iter(validation_dataloader))\n",
                "if isinstance(inputs, dict):\n",
                "    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}\n",
                "else:\n",
                "    inputs = {\"input_ids\": inputs[0].to(device)}\n",
                "\n",
                "model.eval()\n",
                "with torch.no_grad():\n",
                "    outputs = model(**inputs, output_attentions=True)\n",
                "\n",
                "# Get attention tensors\n",
                "if hasattr(outputs, 'attentions') and outputs.attentions is not None:\n",
                "    attn_tensors = outputs.attentions\n",
                "    layer_idx = 0  # Check first layer\n",
                "    \n",
                "    if len(attn_tensors) > 0:\n",
                "        attn = attn_tensors[layer_idx]  # First layer attention\n",
                "        \n",
                "        # Print attention tensor stats to verify it's a valid probability distribution\n",
                "        print(f\"Attention tensor shape: {attn.shape}\")\n",
                "        print(f\"Attention tensor dtype: {attn.dtype}\")\n",
                "        print(f\"Attention tensor stats: min={attn.min().item():.6e}, max={attn.max().item():.6e}, mean={attn.mean().item():.6e}\")\n",
                "        \n",
                "        # Check if values sum to 1 along attention dimension\n",
                "        attn_sum = attn.sum(dim=-1)\n",
                "        print(f\"Sum along attention dimension: min={attn_sum.min().item():.6f}, max={attn_sum.max().item():.6f}\")\n",
                "        print(f\"Close to 1.0? {torch.allclose(attn_sum, torch.ones_like(attn_sum), rtol=1e-3)}\")\n",
                "        \n",
                "        # Check for very small values that might cause log(0) issues\n",
                "        small_values = (attn < 1e-6).float().mean().item() * 100\n",
                "        print(f\"Percentage of very small values (<1e-6): {small_values:.2f}%\")\n",
                "        \n",
                "        # Check for NaN or infinity\n",
                "        print(f\"Contains NaN: {torch.isnan(attn).any().item()}\")\n",
                "        print(f\"Contains Inf: {torch.isinf(attn).any().item()}\")\n",
                "        \n",
                "        # Fix entropy calculation function with better defaults\n",
                "        def improved_entropy_calculation(attn_probs, eps=1e-8):\n",
                "            \"\"\"Compute entropy with better numerical stability.\"\"\"\n",
                "            # Ensure valid probability distribution\n",
                "            attn_probs = attn_probs.clamp(min=eps)\n",
                "            normalized_probs = attn_probs / attn_probs.sum(dim=-1, keepdim=True)\n",
                "            \n",
                "            # Compute entropy\n",
                "            log_probs = torch.log(normalized_probs)\n",
                "            entropy = -torch.sum(normalized_probs * log_probs, dim=-1)\n",
                "            return entropy\n",
                "        \n",
                "        # Calculate entropy using improved function\n",
                "        improved_entropy = improved_entropy_calculation(attn).mean(dim=(0, 1))\n",
                "        print(\"\\nImproved entropy calculation results:\")\n",
                "        print(f\"Mean entropy: {improved_entropy.mean().item():.4f}\")\n",
                "        print(f\"Min entropy: {improved_entropy.min().item():.4f}\")\n",
                "        print(f\"Max entropy: {improved_entropy.max().item():.4f}\")\n",
                "        \n",
                "        # Add visualization of attention patterns\n",
                "        print(\"\\nVisualizing attention pattern for one head...\")\n",
                "        head_idx = 0\n",
                "        plt.figure(figsize=(8, 6))\n",
                "        plt.imshow(attn[0, head_idx].cpu().numpy(), cmap='viridis')\n",
                "        plt.colorbar(label='Attention probability')\n",
                "        plt.title(f'Attention pattern (layer {layer_idx}, head {head_idx})')\n",
                "        plt.xlabel('Sequence position (to)')\n",
                "        plt.ylabel('Sequence position (from)')\n",
                "        plt.show()\n",
                "        \n",
                "        # Add histogram of attention values\n",
                "        plt.figure(figsize=(8, 4))\n",
                "        plt.hist(attn[0, head_idx].flatten().cpu().numpy(), bins=50, alpha=0.7)\n",
                "        plt.title('Histogram of attention probabilities')\n",
                "        plt.xlabel('Probability value')\n",
                "        plt.ylabel('Frequency')\n",
                "        plt.grid(alpha=0.3)\n",
                "        plt.show()\n",
                "else:\n",
                "    print(\"Could not get attention tensors from model output\")\n"
            ]
            
            # Insert diagnostic code
            for line in diagnostic_code:
                new_source.insert(insertion_point, line)
                insertion_point += 1
                
            # Update the cell source
            cell['source'] = new_source
        
        # Also add a patch to the controller cell if found
        if controller_cell_idx is not None:
            cell = notebook['cells'][controller_cell_idx]
            source = cell['source']
            
            # Add controller patch code after controller creation
            new_source = []
            insertion_point = None
            
            for i, line in enumerate(source):
                new_source.append(line)
                
                if 'controller = create_plasticity_controller(' in line and line.strip().endswith(')'):
                    insertion_point = i + 1
            
            if insertion_point is not None:
                # Add patch code
                patch_code = [
                    "\n# Monkey-patch the controller's entropy calculation to fix zero entropy issue\n",
                    "def patched_compute_entropy(self, attention_maps, eps=1e-6):\n",
                    "    \"\"\"Improved entropy calculation with better numerical stability.\"\"\"\n",
                    "    if not attention_maps:\n",
                    "        return 0.0\n",
                    "    \n",
                    "    # Concatenate all maps\n",
                    "    maps = torch.cat(attention_maps, dim=0)\n",
                    "    \n",
                    "    # Print diagnostics before processing\n",
                    "    print(f\"\\nDIAGNOSTIC - Attention maps inside controller:\")\n",
                    "    print(f\"Shape: {maps.shape}, min: {maps.min().item():.6e}, max: {maps.max().item():.6e}, mean: {maps.mean().item():.6e}\")\n",
                    "    \n",
                    "    # Better numerical stability approach\n",
                    "    maps = maps.clamp(min=eps)  # Ensure no zeros\n",
                    "    maps = maps / maps.sum(dim=-1, keepdim=True)  # Ensure proper normalization\n",
                    "    \n",
                    "    # Compute entropy: -sum(p * log(p))\n",
                    "    entropy = -torch.sum(maps * torch.log(maps), dim=-1)\n",
                    "    \n",
                    "    # Average over batch and sequence length\n",
                    "    avg_entropy = entropy.mean().item()\n",
                    "    \n",
                    "    # Normalize to [0,1] range\n",
                    "    max_entropy = torch.log(torch.tensor(maps.size(-1), dtype=torch.float))\n",
                    "    normalized_entropy = avg_entropy / max_entropy.item()\n",
                    "    \n",
                    "    print(f\"Computed entropy: {normalized_entropy:.6f}\")\n",
                    "    return normalized_entropy\n",
                    "\n",
                    "# Apply the patched function\n",
                    "import types\n",
                    "if hasattr(controller, '_compute_entropy'):\n",
                    "    controller._compute_entropy = types.MethodType(patched_compute_entropy, controller)\n",
                    "    print(\"Patched controller entropy calculation with improved version\")\n"
                ]
                
                # Insert patch code
                for line in patch_code:
                    new_source.insert(insertion_point, line)
                    insertion_point += 1
                
                # Update the cell
                cell['source'] = new_source
        
        # Save the modified notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print(f"Added attention diagnostics to notebook")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    add_attention_diagnostics(notebook_path)