#!/usr/bin/env python
"""
Create Minimal Neural Plasticity Notebook

This script creates a comprehensive minimal test notebook that:
1. Tests all core tensor operations and visualization functions
2. Loads a small model to demonstrate entropy and pruning
3. Works across all platforms (including Apple Silicon)
4. Avoids dataset dependencies that cause import issues

Version: v0.0.58 (2025-04-19 19:00:00)
"""

import os
import sys
import time
import json
import datetime
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def create_minimal_notebook():
    """Create a minimal but complete neural plasticity test notebook."""
    
    # Create notebook structure
    notebook = new_notebook()
    
    # Create timestamp for versioning
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add title cell
    title_cell = new_markdown_cell(
        f"# Neural Plasticity Demo - Minimal Edition (v0.0.58 {current_time})\n\n"
        "This notebook demonstrates all key components of the neural plasticity system with a minimal configuration:\n\n"
        "1. Environment detection (Apple Silicon, standard CPU, GPU environments)\n"
        "2. Model loading with proper cross-platform handling\n"
        "3. Entropy calculation with tensor safety measures\n"
        "4. Gradient-based pruning with visualization\n"
        "5. Text generation with pruned model\n\n"
        "This minimal version avoids dataset dependencies that can cause import issues while still exercising the "
        "complete neural plasticity pipeline."
    )
    notebook.cells.append(title_cell)
    
    # Add environment setup cell
    env_cell = new_code_cell(
        "# Environment setup and tensor operation safety\n"
        "import os\n"
        "import sys\n"
        "import torch\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import platform\n"
        "from datetime import datetime\n"
        "%matplotlib inline\n\n"
        "# Set environment variables for safer execution on all platforms\n"
        "os.environ['OMP_NUM_THREADS'] = '1'\n"
        "os.environ['OPENBLAS_NUM_THREADS'] = '1'\n"
        "os.environ['MKL_NUM_THREADS'] = '1'\n"
        "os.environ['VECLIB_MAXIMUM_THREADS'] = '1'\n"
        "os.environ['NUMEXPR_NUM_THREADS'] = '1'\n\n"
        "# Add project root to path\n"
        "if not os.getcwd() in sys.path:\n"
        "    sys.path.append(os.getcwd())\n\n"
        "# Create output directory for visualizations\n"
        "OUTPUT_DIR = \"minimal_output\"\n"
        "os.makedirs(OUTPUT_DIR, exist_ok=True)\n\n"
        "# Memory management helper\n"
        "def clear_memory():\n"
        "    \"\"\"Clear GPU memory cache and run garbage collection\"\"\"\n"
        "    import gc\n"
        "    gc.collect()\n"
        "    if torch.cuda.is_available():\n"
        "        torch.cuda.empty_cache()\n"
        "        torch.cuda.synchronize()\n\n"
        "# Detect environment (Apple Silicon, GPU, etc.)\n"
        "IS_APPLE_SILICON = platform.system() == \"Darwin\" and platform.processor() == \"arm\"\n"
        "HAS_GPU = torch.cuda.is_available()\n"
        "try:\n"
        "    import google.colab\n"
        "    IS_COLAB = True\n"
        "except ImportError:\n"
        "    IS_COLAB = False\n\n"
        "print(f\"Environment: {'Apple Silicon' if IS_APPLE_SILICON else 'Standard'}, {'Colab' if IS_COLAB else 'Local'}\")\n"
        "print(f\"PyTorch Version: {torch.__version__}\")\n"
        "print(f\"GPU Available: {HAS_GPU}\")\n"
        "if HAS_GPU:\n"
        "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n\n"
        "# Set device appropriately for environment\n"
        "if HAS_GPU and not IS_APPLE_SILICON:\n"
        "    device = torch.device('cuda')\n"
        "else:\n"
        "    device = torch.device('cpu')\n"
        "print(f\"Using device: {device}\")"
    )
    notebook.cells.append(env_cell)
    
    # Add neural plasticity imports cell
    imports_cell = new_code_cell(
        "# Import neural plasticity modules\n"
        "try:\n"
        "    from utils.neural_plasticity.core import (\n"
        "        calculate_head_entropy,\n"
        "        generate_pruning_mask,\n"
        "        apply_pruning_mask,\n"
        "        safe_matmul\n"
        "    )\n"
        "    from utils.neural_plasticity.visualization import (\n"
        "        visualize_head_entropy,\n"
        "        visualize_head_gradients,\n"
        "        visualize_pruning_decisions,\n"
        "        visualize_attention_patterns\n"
        "    )\n"
        "    from utils.colab.helpers import safe_tensor_imshow\n"
        "    \n"
        "    print(\"✅ Successfully imported neural plasticity modules\")\n"
        "except ImportError as e:\n"
        "    print(f\"❌ Error importing modules: {e}\")\n"
        "    raise"
    )
    notebook.cells.append(imports_cell)
    
    # Add model loading cell
    model_cell = new_code_cell(
        "# Load a small transformer model\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
        "MODEL_NAME = \"distilgpt2\"  # Small model for testing\n\n"
        "try:\n"
        "    # Load model and tokenizer\n"
        "    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)\n"
        "    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n"
        "    print(f\"✅ Loaded model {MODEL_NAME}\")\n"
        "    \n"
        "    # Set pad token if needed\n"
        "    if tokenizer.pad_token is None:\n"
        "        tokenizer.pad_token = tokenizer.eos_token\n"
        "        \n"
        "    # Move model to device\n"
        "    model = model.to(device)\n"
        "    print(f\"Model moved to {device}\")\n"
        "    \n"
        "    # Get model structure information\n"
        "    if hasattr(model, 'config'):\n"
        "        if hasattr(model.config, 'n_layer'):\n"
        "            num_layers = model.config.n_layer\n"
        "        elif hasattr(model.config, 'num_hidden_layers'):\n"
        "            num_layers = model.config.num_hidden_layers\n"
        "        else:\n"
        "            num_layers = 6  # Default for distilgpt2\n"
        "            \n"
        "        if hasattr(model.config, 'n_head'):\n"
        "            num_heads = model.config.n_head\n"
        "        elif hasattr(model.config, 'num_attention_heads'):\n"
        "            num_heads = model.config.num_attention_heads\n"
        "        else:\n"
        "            num_heads = 12  # Default for distilgpt2\n"
        "    else:\n"
        "        num_layers, num_heads = 6, 12  # Default for distilgpt2\n"
        "    \n"
        "    print(f\"Model has {num_layers} layers with {num_heads} attention heads each\")\n"
        "except Exception as e:\n"
        "    print(f\"❌ Error loading model: {e}\")\n"
        "    # Create dummy values for testing visualization if model loading fails\n"
        "    num_layers, num_heads = 6, 12\n"
        "    raise"
    )
    notebook.cells.append(model_cell)
    
    # Add sample input creation cell
    inputs_cell = new_code_cell(
        "# Create sample inputs for attention patterns\n"
        "sample_texts = [\n"
        "    \"The quick brown fox jumps over the lazy dog.\",\n"
        "    \"Neural networks have revolutionized artificial intelligence.\",\n"
        "    \"Transformers use attention mechanisms to process sequences.\",\n"
        "    \"The meaning of life is to give life meaning.\"\n"
        "]\n\n"
        "MAX_LENGTH = 64  # Shorter sequences for testing\n\n"
        "encoded_inputs = tokenizer(sample_texts, padding=True, truncation=True, \n"
        "                          max_length=MAX_LENGTH, return_tensors=\"pt\")\n"
        "encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}\n\n"
        "print(f\"Created inputs with {len(sample_texts)} samples\")\n"
        "print(f\"Input shape: {encoded_inputs['input_ids'].shape}\")"
    )
    notebook.cells.append(inputs_cell)
    
    # Add attention map collection cell
    attention_cell = new_code_cell(
        "# Collect attention maps from model\n"
        "try:\n"
        "    # Forward pass with attention outputs\n"
        "    model.eval()\n"
        "    with torch.no_grad():\n"
        "        outputs = model(**encoded_inputs, output_attentions=True)\n"
        "        \n"
        "    # Extract attention maps\n"
        "    if hasattr(outputs, 'attentions') and outputs.attentions is not None:\n"
        "        attention_maps = outputs.attentions\n"
        "        print(f\"✅ Collected {len(attention_maps)} attention layers\")\n"
        "        \n"
        "        # Verify that attention sums to 1 along correct dimension\n"
        "        attn_sum = attention_maps[0].sum(dim=-1)\n"
        "        print(f\"Attention sums to ~1: {torch.allclose(attn_sum, torch.ones_like(attn_sum), rtol=1e-3)}\")\n"
        "        \n"
        "        # Show shape and stats of first layer's attention\n"
        "        first_attn = attention_maps[0]\n"
        "        print(f\"First layer attention shape: {first_attn.shape}\")\n"
        "        print(f\"Min/max/mean: {first_attn.min().item():.4f}/{first_attn.max().item():.4f}/{first_attn.mean().item():.4f}\")\n"
        "    else:\n"
        "        raise ValueError(\"Model did not return attention tensors\")\n"
        "        \n"
        "except Exception as e:\n"
        "    print(f\"❌ Error collecting attention maps: {e}\")\n"
        "    # If we can't get real attention maps, create synthetic ones for testing\n"
        "    print(\"Creating synthetic attention maps for testing...\")\n"
        "    batch_size = 2\n"
        "    seq_len = 32\n"
        "    \n"
        "    # Create random attention-like matrices for num_layers layers\n"
        "    attention_maps = []\n"
        "    for _ in range(num_layers):\n"
        "        # Create random attention with proper distribution (sums to 1)\n"
        "        attn = torch.rand(batch_size, num_heads, seq_len, seq_len, device=device)\n"
        "        attn = attn / attn.sum(dim=-1, keepdim=True)\n"
        "        attention_maps.append(attn)\n"
        "    \n"
        "    print(f\"Created {len(attention_maps)} synthetic attention layers\")"
    )
    notebook.cells.append(attention_cell)
    
    # Add entropy calculation cell
    entropy_cell = new_code_cell(
        "# Calculate entropy for each attention layer\n"
        "try:\n"
        "    # Calculate entropy for each layer\n"
        "    all_entropies = []\n"
        "    \n"
        "    for i, attn in enumerate(attention_maps):\n"
        "        # Calculate entropy using our safe function\n"
        "        entropy = calculate_head_entropy(attn)\n"
        "        print(f\"Layer {i} entropy shape: {entropy.shape}\")\n"
        "        print(f\"Layer {i} entropy stats: min={entropy.min().item():.4f}, max={entropy.max().item():.4f}\")\n"
        "        all_entropies.append(entropy)\n"
        "    \n"
        "    # Stack all entropies into a single tensor for visualization\n"
        "    entropy_tensor = torch.stack(all_entropies)\n"
        "    print(f\"Combined entropy tensor shape: {entropy_tensor.shape}\")\n"
        "    \n"
        "    # Create visualization\n"
        "    plt.figure(figsize=(10, 6))\n"
        "    viz_entropy = visualize_head_entropy(\n"
        "        entropy_values=entropy_tensor,\n"
        "        title=\"Attention Head Entropy Across Layers\",\n"
        "        figsize=(10, 6),\n"
        "        annotate=True\n"
        "    )\n"
        "    \n"
        "    # Save the entropy visualization\n"
        "    entropy_path = os.path.join(OUTPUT_DIR, \"entropy_heatmap.png\")\n"
        "    plt.savefig(entropy_path)\n"
        "    plt.show()\n"
        "    print(f\"✅ Entropy calculation and visualization complete\")\n"
        "    \n"
        "except Exception as e:\n"
        "    print(f\"❌ Error calculating entropy: {e}\")\n"
        "    # Create random entropy values for visualization testing\n"
        "    entropy_tensor = torch.rand(num_layers, num_heads)\n"
        "    print(\"Created random entropy tensor for testing\")"
    )
    notebook.cells.append(entropy_cell)
    
    # Add gradient simulation cell
    gradient_cell = new_code_cell(
        "# Simulate gradient calculation for attention heads\n"
        "try:\n"
        "    # In a real scenario, we would calculate gradients from real data\n"
        "    # For this minimal test, we'll simulate gradients\n"
        "    gradient_norms = torch.rand(num_layers, num_heads, device=device) * 0.1\n"
        "    \n"
        "    # Make some heads clearly more important with higher gradients\n"
        "    important_heads = [(0, 3), (2, 0), (num_layers-1, num_heads-1)]  # Layer, head pairs\n"
        "    for layer, head in important_heads:\n"
        "        if layer < gradient_norms.shape[0] and head < gradient_norms.shape[1]:\n"
        "            gradient_norms[layer, head] = 0.9  # Higher gradient = more important\n"
        "    \n"
        "    print(f\"Generated gradient norms with shape {gradient_norms.shape}\")\n"
        "    print(f\"Gradient stats: min={gradient_norms.min().item():.4f}, max={gradient_norms.max().item():.4f}\")\n"
        "            \n"
        "    # Visualize gradients\n"
        "    plt.figure(figsize=(10, 5))\n"
        "    viz_grads = visualize_head_gradients(\n"
        "        grad_norm_values=gradient_norms,\n"
        "        title=\"Simulated Head Gradient Norms\",\n"
        "        figsize=(10, 5)\n"
        "    )\n"
        "    \n"
        "    # Save the gradient visualization\n"
        "    grads_path = os.path.join(OUTPUT_DIR, \"gradient_norms.png\")\n"
        "    plt.savefig(grads_path)\n"
        "    plt.show()\n"
        "    print(f\"✅ Gradient simulation and visualization complete\")\n"
        "    \n"
        "except Exception as e:\n"
        "    print(f\"❌ Error in gradient simulation: {e}\")\n"
        "    raise"
    )
    notebook.cells.append(gradient_cell)
    
    # Add pruning mask generation cell
    pruning_cell = new_code_cell(
        "# Generate pruning mask based on entropy and gradients\n"
        "PRUNING_PERCENT = 0.2  # Prune 20% of heads\n\n"
        "try:\n"
        "    # Generate pruning mask using both entropy and gradients\n"
        "    pruning_mask = generate_pruning_mask(\n"
        "        grad_norm_values=gradient_norms,\n"
        "        prune_percent=PRUNING_PERCENT,\n"
        "        strategy=\"combined\",  # Use both entropy and gradients\n"
        "        entropy_values=entropy_tensor\n"
        "    )\n"
        "    \n"
        "    # Count pruned heads\n"
        "    pruned_count = pruning_mask.sum().item()\n"
        "    total_count = pruning_mask.numel()\n"
        "    print(f\"Pruning {pruned_count}/{total_count} heads ({pruned_count/total_count*100:.1f}%)\")\n"
        "    \n"
        "    # Create list of (layer, head) tuples for pruned heads\n"
        "    pruned_heads = []\n"
        "    for layer in range(pruning_mask.size(0)):\n"
        "        for head in range(pruning_mask.size(1)):\n"
        "            if pruning_mask[layer, head]:\n"
        "                pruned_heads.append((layer, head))\n"
        "    \n"
        "    print(f\"Pruned heads: {pruned_heads}\")\n"
        "    \n"
        "    # Visualize pruning decisions\n"
        "    plt.figure(figsize=(12, 8))\n"
        "    viz_pruning = visualize_pruning_decisions(\n"
        "        grad_norm_values=gradient_norms,\n"
        "        pruning_mask=pruning_mask,\n"
        "        title=\"Combined Entropy-Gradient Pruning Decisions\",\n"
        "        figsize=(12, 8)\n"
        "    )\n"
        "    \n"
        "    # Save the pruning visualization\n"
        "    pruning_path = os.path.join(OUTPUT_DIR, \"pruning_decisions.png\")\n"
        "    plt.savefig(pruning_path)\n"
        "    plt.show()\n"
        "    print(f\"✅ Pruning mask generation and visualization complete\")\n"
        "    \n"
        "except Exception as e:\n"
        "    print(f\"❌ Error in pruning mask generation: {e}\")\n"
        "    raise"
    )
    notebook.cells.append(pruning_cell)
    
    # Add apply pruning mask cell
    apply_pruning_cell = new_code_cell(
        "# Apply pruning to model\n"
        "try:\n"
        "    # Apply pruning to model\n"
        "    if 'model' in globals():\n"
        "        pruned_heads = apply_pruning_mask(\n"
        "            model=model,\n"
        "            pruning_mask=pruning_mask,\n"
        "            mode=\"zero_weights\"  # Zero out attention head weights\n"
        "        )\n"
        "        \n"
        "        print(f\"Applied pruning to {len(pruned_heads)} heads\")\n"
        "        print(f\"The model now has {pruned_count/total_count*100:.1f}% of heads pruned\")\n"
        "    else:\n"
        "        print(\"Skipping actual pruning as model is not available\")\n"
        "    \n"
        "except Exception as e:\n"
        "    print(f\"❌ Error applying pruning mask: {e}\")"
    )
    notebook.cells.append(apply_pruning_cell)
    
    # Add visualization of attention patterns cell
    viz_attention_cell = new_code_cell(
        "# Visualize attention patterns from one layer\n"
        "try:\n"
        "    # Visualize attention from first layer\n"
        "    layer_idx = 0\n"
        "    \n"
        "    # Single head visualization\n"
        "    plt.figure(figsize=(10, 8))\n"
        "    visualize_attention_patterns(\n"
        "        attention_maps=attention_maps[layer_idx],\n"
        "        layer_idx=layer_idx,\n"
        "        head_idx=0,  # First head\n"
        "        title=f\"Layer {layer_idx}, Head 0 Attention Pattern\"\n"
        "    )\n"
        "    plt.tight_layout()\n"
        "    plt.savefig(os.path.join(OUTPUT_DIR, \"single_head_attention.png\"))\n"
        "    plt.show()\n"
        "    \n"
        "    # Multiple heads visualization\n"
        "    plt.figure(figsize=(14, 6))\n"
        "    visualize_attention_patterns(\n"
        "        attention_maps=attention_maps[layer_idx],\n"
        "        layer_idx=layer_idx,\n"
        "        head_idx=None,  # Show multiple heads\n"
        "        title=f\"Layer {layer_idx} - Multiple Attention Heads\",\n"
        "        num_heads=min(4, num_heads)  # Show up to 4 heads\n"
        "    )\n"
        "    plt.tight_layout()\n"
        "    plt.savefig(os.path.join(OUTPUT_DIR, \"multi_head_attention.png\"))\n"
        "    plt.show()\n"
        "    \n"
        "    print(\"✅ Attention pattern visualization complete\")\n"
        "except Exception as e:\n"
        "    print(f\"❌ Error visualizing attention patterns: {e}\")"
    )
    notebook.cells.append(viz_attention_cell)
    
    # Add text generation cell
    gen_text_cell = new_code_cell(
        "# Generate text with the pruned model\n"
        "try:\n"
        "    if 'model' in globals() and 'tokenizer' in globals():\n"
        "        # Clear memory before generation\n"
        "        clear_memory()\n"
        "        \n"
        "        # Set model to evaluation mode\n"
        "        model.eval()\n"
        "        \n"
        "        # Define generation function\n"
        "        def generate_text(prompt, max_length=50):\n"
        "            input_ids = tokenizer.encode(prompt, return_tensors=\"pt\").to(device)\n"
        "            \n"
        "            with torch.no_grad():\n"
        "                output = model.generate(\n"
        "                    input_ids=input_ids,\n"
        "                    max_length=max_length,\n"
        "                    temperature=0.8,\n"
        "                    do_sample=True,\n"
        "                    top_k=40,\n"
        "                    top_p=0.95,\n"
        "                    pad_token_id=tokenizer.eos_token_id\n"
        "                )\n"
        "            \n"
        "            return tokenizer.decode(output[0], skip_special_tokens=True)\n"
        "        \n"
        "        # Generate text with various prompts\n"
        "        prompts = [\n"
        "            \"The future of artificial intelligence\",\n"
        "            \"Once upon a time in a distant galaxy\",\n"
        "            \"The meaning of life is\"\n"
        "        ]\n"
        "        \n"
        "        # Generate and display text for each prompt\n"
        "        for prompt in prompts:\n"
        "            try:\n"
        "                generated_text = generate_text(prompt)\n"
        "                print(f\"\\nPrompt: {prompt}\")\n"
        "                print(f\"Generated: {generated_text}\")\n"
        "            except Exception as gen_error:\n"
        "                print(f\"Error generating for prompt '{prompt}': {gen_error}\")\n"
        "                \n"
        "        print(\"\\n✅ Text generation with pruned model complete\")\n"
        "    else:\n"
        "        print(\"Skipping text generation as model or tokenizer is not available\")\n"
        "except Exception as e:\n"
        "    print(f\"❌ Error in text generation: {e}\")"
    )
    notebook.cells.append(gen_text_cell)
    
    # Add conclusion cell
    conclusion_cell = new_markdown_cell(
        "## Summary\n\n"
        "This notebook has demonstrated all key components of the neural plasticity system across platforms:\n\n"
        "1. **Environment Detection**: Automatically detecting and optimizing for the current platform\n"
        "2. **Model Loading**: Loading a transformer model with proper device placement\n"
        "3. **Attention Analysis**: Extracting and analyzing attention patterns\n"
        "4. **Entropy Calculation**: Computing attention entropy with stable numerical operations\n"
        "5. **Gradient Simulation**: Representing the importance of different attention heads\n"
        "6. **Pruning**: Generating and applying pruning masks to zero out attention heads\n"
        "7. **Visualization**: Safely visualizing tensors across all environments\n"
        "8. **Text Generation**: Generating text with the pruned model\n\n"
        "All the visualizations were saved to the `minimal_output` directory.\n\n"
        "### Next Steps\n\n"
        "Now you can try running the full neural plasticity demo in a GPU-enabled environment like Google Colab."
    )
    notebook.cells.append(conclusion_cell)
    
    # Save notebook
    notebook_path = Path("neural_plasticity_cross_platform.ipynb")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(notebook, f)
    
    print(f"Created comprehensive cross-platform notebook at {notebook_path}")
    return notebook_path

def main():
    """Create the minimal notebook."""
    notebook_path = create_minimal_notebook()
    
    print(f"\nTo run the cross-platform notebook:")
    print(f"  jupyter notebook {notebook_path}")
    
if __name__ == "__main__":
    main()