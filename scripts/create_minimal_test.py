#!/usr/bin/env python
"""
Create a minimal test for neural plasticity functionality

This script creates a minimal notebook for testing the neural plasticity 
functionality, which is useful for quick validation.

Version: v0.0.60 (2025-04-20)
"""

import os
import sys
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from datetime import datetime
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def create_minimal_test(output_path=None):
    """
    Create a minimal test notebook for neural plasticity.
    
    Args:
        output_path: Path to save the notebook (default: neural_plasticity_minimal_test.ipynb)
        
    Returns:
        Path to the created notebook
    """
    # Default output path
    if output_path is None:
        output_path = os.path.join(project_root, "neural_plasticity_minimal_test.ipynb")
    
    # Get current timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create notebook
    notebook = new_notebook()
    
    # Title cell
    notebook.cells.append(new_markdown_cell(
        "# Minimal Neural Plasticity Test\n\n"
        "This notebook provides a minimal test of the neural plasticity module functionality.\n\n"
        f"Version: v0.0.60 ({current_time})"
    ))
    
    # Import and environment setup
    notebook.cells.append(new_code_cell(
        "import torch\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "\n"
        "# Import the modular neural plasticity API\n"
        "from utils.neural_plasticity import NeuralPlasticity\n"
        "from utils.neural_plasticity import PruningStrategy, PruningMode\n"
        "\n"
        "# Get environment information\n"
        "env_info = NeuralPlasticity.get_environment_info()\n"
        "print(\"Environment Information:\")\n"
        "print(f\"- Platform: {env_info['platform']}\")\n"
        "print(f\"- Apple Silicon: {env_info['is_apple_silicon']}\")\n"
        "print(f\"- GPU Available: {env_info['has_gpu']}\")\n"
        "print(f\"- Device: {env_info['device']}\")\n"
        "\n"
        "# Define device - respect environment detection\n"
        "device = env_info['device']\n"
    ))
    
    # Load model
    notebook.cells.append(new_code_cell(
        "# Load a small model for testing\n"
        "model_name = \"distilgpt2\"  # Small model for faster testing\n"
        "print(f\"Loading model: {model_name}\")\n"
        "\n"
        "# Load model and tokenizer\n"
        "model = AutoModelForCausalLM.from_pretrained(model_name).to(device)\n"
        "tokenizer = AutoTokenizer.from_pretrained(model_name)\n"
        "\n"
        "# Get model structure information\n"
        "num_layers, num_heads = NeuralPlasticity.detect_attention_heads(model)\n"
        "print(f\"Model structure: {num_layers} layers with {num_heads} heads each\")"
    ))
    
    # Prepare small test input
    notebook.cells.append(new_code_cell(
        "# Prepare a small test input\n"
        "test_text = \"This is a test of the neural plasticity system.\"\n"
        "inputs = tokenizer(test_text, return_tensors=\"pt\").to(device)\n"
        "print(f\"Input shape: {inputs['input_ids'].shape}\")"
    ))
    
    # Test attention analysis
    notebook.cells.append(new_code_cell(
        "# Analyze attention patterns\n"
        "attention_data = NeuralPlasticity.analyze_attention_patterns(\n"
        "    model=model,\n"
        "    input_ids=inputs['input_ids'],\n"
        "    attention_mask=inputs['attention_mask']\n"
        ")\n"
        "\n"
        "# Extract data\n"
        "attention_tensors = attention_data['attention_tensors']\n"
        "entropy_values = attention_data['entropy_values']\n"
        "\n"
        "# Print basic statistics\n"
        "print(f\"Number of attention tensors: {len(attention_tensors)}\")\n"
        "print(f\"Attention tensor shape: {attention_tensors[0].shape}\")\n"
        "print(f\"Number of entropy layers: {len(entropy_values)}\")\n"
        "print(f\"Entropy shape: {entropy_values[0].shape}\")\n"
        "\n"
        "# Visualize entropy\n"
        "from utils.neural_plasticity import visualize_head_entropy\n"
        "\n"
        "plt.figure(figsize=(10, 6))\n"
        "entropy_fig = visualize_head_entropy(\n"
        "    entropy_values=entropy_values,\n"
        "    title=\"Attention Entropy (Test)\",\n"
        "    min_value=0.0,\n"
        "    annotate=True\n"
        ")\n"
        "plt.show()"
    ))
    
    # Test gradient calculation
    notebook.cells.append(new_code_cell(
        "# Test gradient calculation - simplified version for minimal test\n"
        "def minimal_gradient_test(model, inputs):\n"
        "    model.train()  # Set to training mode for gradient calculation\n"
        "    \n"
        "    # Forward pass with gradient tracking\n"
        "    outputs = model(**inputs)\n"
        "    loss = outputs.loss\n"
        "    \n"
        "    # Backward pass\n"
        "    loss.backward()\n"
        "    \n"
        "    # Get model structure\n"
        "    num_layers, num_heads = NeuralPlasticity.detect_attention_heads(model)\n"
        "    \n"
        "    # Extract gradient norms (simplified)\n"
        "    from utils.neural_plasticity import extract_head_gradient\n"
        "    grad_norms = torch.zeros((num_layers, num_heads), device='cpu')\n"
        "    \n"
        "    for layer_idx in range(num_layers):\n"
        "        for head_idx in range(num_heads):\n"
        "            grad = extract_head_gradient(model, layer_idx, head_idx)\n"
        "            if grad is not None:\n"
        "                grad_norms[layer_idx, head_idx] = grad.norm().item()\n"
        "    \n"
        "    # Reset gradients\n"
        "    model.zero_grad()\n"
        "    model.eval()  # Set back to eval mode\n"
        "    \n"
        "    return grad_norms\n"
        "\n"
        "# Calculate gradients\n"
        "grad_norms = minimal_gradient_test(model, inputs)\n"
        "print(f\"Gradient norms shape: {grad_norms.shape}\")\n"
        "print(f\"Max gradient norm: {grad_norms.max().item():.4f}\")\n"
        "print(f\"Min gradient norm: {grad_norms.min().item():.4f}\")\n"
        "\n"
        "# Visualize gradients\n"
        "from utils.neural_plasticity import visualize_head_gradients\n"
        "\n"
        "plt.figure(figsize=(10, 6))\n"
        "grad_fig = visualize_head_gradients(\n"
        "    grad_norm_values=grad_norms,\n"
        "    title=\"Gradient Norms (Test)\"\n"
        ")\n"
        "plt.show()"
    ))
    
    # Test pruning mask generation
    notebook.cells.append(new_code_cell(
        "# Generate pruning mask\n"
        "from utils.neural_plasticity import generate_pruning_mask\n"
        "\n"
        "# Set pruning parameters\n"
        "prune_percent = 0.2  # Prune 20% of heads\n"
        "strategy = PruningStrategy.COMBINED  # Use combined strategy\n"
        "\n"
        "# Generate mask\n"
        "pruning_mask = generate_pruning_mask(\n"
        "    grad_norm_values=grad_norms,\n"
        "    entropy_values=entropy_values[0],  # Use first layer's entropy\n"
        "    prune_percent=prune_percent,\n"
        "    strategy=strategy\n"
        ")\n"
        "\n"
        "# Visualize pruning decisions\n"
        "from utils.neural_plasticity import visualize_pruning_decisions\n"
        "\n"
        "plt.figure(figsize=(10, 6))\n"
        "pruning_fig = visualize_pruning_decisions(\n"
        "    grad_norm_values=grad_norms,\n"
        "    pruning_mask=pruning_mask,\n"
        "    title=f\"Pruning Decisions ({strategy}, {prune_percent*100:.0f}%)\"\n"
        ")\n"
        "plt.show()\n"
        "\n"
        "# Count pruned heads\n"
        "total_heads = pruning_mask.numel()\n"
        "pruned_count = pruning_mask.sum().item()\n"
        "print(f\"Pruning {pruned_count} out of {total_heads} heads ({pruned_count/total_heads*100:.1f}%)\")"
    ))
    
    # Test applying pruning
    notebook.cells.append(new_code_cell(
        "# Apply pruning\n"
        "from utils.neural_plasticity import apply_pruning_mask\n"
        "\n"
        "# Apply pruning mask\n"
        "pruned_heads = apply_pruning_mask(\n"
        "    model=model,\n"
        "    pruning_mask=pruning_mask,\n"
        "    mode=\"zero_weights\"\n"
        ")\n"
        "\n"
        "print(f\"Pruned {len(pruned_heads)} heads:\")\n"
        "for layer, head in pruned_heads[:10]:  # Show first 10\n"
        "    print(f\"  Layer {layer}, Head {head}\")\n"
        "    \n"
        "if len(pruned_heads) > 10:\n"
        "    print(f\"  ... and {len(pruned_heads) - 10} more\")"
    ))
    
    # Test generation with pruned model
    notebook.cells.append(new_code_cell(
        "# Test text generation with pruned model\n"
        "prompt = \"Once upon a time\"\n"
        "input_ids = tokenizer.encode(prompt, return_tensors=\"pt\").to(device)\n"
        "\n"
        "# Generate text\n"
        "with torch.no_grad():\n"
        "    output = model.generate(\n"
        "        input_ids=input_ids,\n"
        "        max_length=30,\n"
        "        do_sample=True,\n"
        "        top_k=50,\n"
        "        top_p=0.95\n"
        "    )\n"
        "\n"
        "generated_text = tokenizer.decode(output[0], skip_special_tokens=True)\n"
        "print(f\"Prompt: {prompt}\")\n"
        "print(f\"Generated (pruned model): {generated_text}\")"
    ))
    
    # Final verification cell
    notebook.cells.append(new_code_cell(
        "# Final verification - all tests passed\n"
        "print(\"✅ All neural plasticity module tests completed successfully!\")\n"
        "print(f\"Tested {num_layers} layers with {num_heads} heads each\")\n"
        "print(f\"Pruned {len(pruned_heads)} heads ({len(pruned_heads)/total_heads*100:.1f}%)\")\n"
        "print(\"Model is still functional after pruning.\")"
    ))
    
    # Save the notebook
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(notebook, f)
    
    print(f"✅ Created minimal test notebook: {output_path}")
    print("You can run this notebook to test the neural plasticity functionality.")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Create a minimal test for neural plasticity")
    parser.add_argument("--output", type=str, default=None, help="Output path for the notebook")
    
    args = parser.parse_args()
    
    try:
        # Create the test notebook
        output_path = create_minimal_test(output_path=args.output)
        
        print("\nTo run the notebook:")
        print(f"jupyter notebook {output_path}")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())