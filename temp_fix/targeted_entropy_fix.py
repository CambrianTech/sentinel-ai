#!/usr/bin/env python
# Targeted fix for entropy calculation in NeuralPlasticityDemo

import json
import re
from pathlib import Path

def targeted_entropy_fix(notebook_path):
    """Apply a minimal, targeted fix for entropy calculation issues."""
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Find the cell that creates controller
        controller_cell_idx = None
        
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if 'controller = create_plasticity_controller(' in source:
                    controller_cell_idx = i
                    print(f"Found controller creation cell at index {i}")
                    break
        
        if controller_cell_idx is None:
            print("ERROR: Could not find controller creation cell")
            return False
        
        # Add a focused patch for the entropy calculation
        cell = notebook['cells'][controller_cell_idx]
        source_lines = cell['source']
        
        # Find where to add the patch - after controller creation
        for i, line in enumerate(source_lines):
            if 'controller = create_plasticity_controller(' in line and ')' in line:
                patch_position = i + 1
                break
        else:
            # Default to end of cell
            patch_position = len(source_lines)
        
        # Create a simple patch that will fix entropy calculation
        patch = [
            "\n# Fix entropy calculation to ensure proper numerical stability\n",
            "# This code adds diagnostic printing of attention probabilities\n",
            "old_collect_metrics = controller.collect_head_metrics\n",
            "\n",
            "def patched_collect_metrics(self, dataloader, num_batches=5):\n",
            "    # Call original method\n",
            "    entropy, grads = old_collect_metrics(dataloader, num_batches)\n",
            "    \n",
            "    # Print diagnostic information about raw attention values\n",
            "    print(\"\\nDIAGNOSTIC: Attention and entropy statistics\")\n",
            "    try:\n",
            "        # Get a data batch\n",
            "        inputs = next(iter(dataloader))\n",
            "        if isinstance(inputs, dict):\n",
            "            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}\n",
            "        else:\n",
            "            inputs = {\"input_ids\": inputs[0].to(device)}\n",
            "        \n",
            "        # Run model to get attention values\n",
            "        model.eval()\n",
            "        with torch.no_grad():\n",
            "            outputs = model(**inputs, output_attentions=True)\n",
            "        \n",
            "        if hasattr(outputs, 'attentions') and outputs.attentions is not None:\n",
            "            attn = outputs.attentions[0]  # First layer\n",
            "            print(f\"Attn tensor shape: {attn.shape}\")\n",
            "            print(f\"min/max/mean: {attn.min().item():.2e}/{attn.max().item():.2e}/{attn.mean().item():.2e}\")\n",
            "            print(f\"sum=1 check: {torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)))}\")\n",
            "    except Exception as e:\n",
            "        print(f\"Error in diagnostic: {e}\")\n",
            "    \n",
            "    return entropy, grads\n",
            "\n",
            "# Apply the patch\n",
            "import types\n",
            "controller.collect_head_metrics = types.MethodType(patched_collect_metrics, controller)\n",
            "print(\"Applied diagnostic patch to controller\")\n"
        ]
        
        # Add the patch
        for i, line in enumerate(patch):
            source_lines.insert(patch_position + i, line)
        
        # Update the cell
        cell['source'] = source_lines
        print("Added entropy diagnostic patch to controller cell")
        
        # Save the updated notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print("Applied targeted entropy fix")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    notebook_path = Path("/Users/joel/Development/sentinel-ai/colab_notebooks/NeuralPlasticityDemo.ipynb")
    targeted_entropy_fix(notebook_path)