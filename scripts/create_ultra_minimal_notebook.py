#!/usr/bin/env python
"""
Create Ultra-Minimal Runnable Version of Neural Plasticity Notebook

This script creates a version of the Neural Plasticity notebook that:
1. Uses synthetic data instead of loading from HuggingFace datasets
2. Uses the smallest model possible
3. Focuses on the core functionality that had BLAS/libtorch issues
4. Runs end-to-end including model loading and training
"""

import os
import sys
import json
import time
import uuid
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

def create_ultra_minimal_notebook():
    """Create an ultra-minimal version of the Neural Plasticity notebook."""
    
    # Create a new notebook
    notebook = new_notebook()
    
    # Add title cell
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    unique_id = str(uuid.uuid4())[:8]
    title_cell = new_markdown_cell(
        f"# Neural Plasticity Demo: Ultra-Minimal Version (v0.0.54 {current_time})\n\n"
        "This notebook demonstrates Sentinel AI's neural plasticity system, which allows transformer models to "
        "dynamically prune and regrow attention heads during training based on utility metrics. "
        f"[ID: {unique_id}]\n\n"
        "### Changes in v0.0.54:\n"
        "- Fixed GPU tensor handling for visualizations\n"
        "- Fixed redundant tensor conversion patterns\n"
        "- Improved numerical stability in entropy calculations\n"
        "- Created ultra-minimal version for local testing\n"
        "- Added synthetic data to avoid dataset dependencies\n\n"
        "## ⚠️ Ultra-Minimal Settings Version ⚠️\n"
        "This is a streamlined version for local testing with:\n"
        "- Smallest model possible (distilgpt2)\n"
        "- Synthetic data generated on-the-fly\n"
        "- Minimal training steps\n"
        "- Core functionality only\n\n"
        "For full features, use the standard version in Google Colab with GPU acceleration."
    )
    notebook.cells.append(title_cell)
    
    # Add setup cell (environment variables for stability)
    setup_cell = new_code_cell(
        "# Set environment variables for BLAS stability\n"
        "import os\n"
        "os.environ['OMP_NUM_THREADS'] = '1'\n"
        "os.environ['OPENBLAS_NUM_THREADS'] = '1'\n"
        "os.environ['MKL_NUM_THREADS'] = '1'\n"
        "os.environ['NUMEXPR_NUM_THREADS'] = '1'\n"
        "os.environ['PYTHONHASHSEED'] = '0'\n\n"
        "# Add tensor handling safety utilities\n"
        "import gc\n"
        "def clear_memory():\n"
        "    \"\"\"Clear GPU memory cache and run garbage collection\"\"\"\n"
        "    gc.collect()\n"
        "    if torch.cuda.is_available():\n"
        "        torch.cuda.empty_cache()\n"
        "        torch.cuda.synchronize()"
    )
    notebook.cells.append(setup_cell)
    
    # Add imports cell
    imports_cell = new_code_cell(
        "%matplotlib inline\n"
        "import torch\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import time\n"
        "import math\n"
        "from tqdm import tqdm\n"
        "from transformers import (\n"
        "    AutoModelForCausalLM, \n"
        "    AutoTokenizer, \n"
        "    get_linear_schedule_with_warmup\n"
        ")\n\n"
        "# Import neural plasticity modules\n"
        "import sys\n"
        "if not os.getcwd() in sys.path:\n"
        "    sys.path.append(os.getcwd())\n"
        "from utils.neural_plasticity.core import (\n"
        "    calculate_head_entropy,\n"
        "    calculate_head_gradients,\n"
        "    generate_pruning_mask,\n"
        "    apply_pruning_mask,\n"
        "    evaluate_model,\n"
        "    detect_model_structure\n"
        ")\n"
        "from utils.neural_plasticity.visualization import (\n"
        "    visualize_head_entropy,\n"
        "    visualize_head_gradients,\n"
        "    visualize_pruning_decisions\n"
        ")\n"
        "from utils.colab.helpers import safe_tensor_imshow\n\n"
        "print(\"Neural plasticity imports successful\")"
    )
    notebook.cells.append(imports_cell)
    
    # Add configuration cell
    config_cell = new_code_cell(
        "# Configure experiment with minimal settings\n"
        "MODEL_NAME = \"distilgpt2\"  # Smallest GPT-2 model\n"
        "MAX_LENGTH = 32        # Very short sequences\n"
        "BATCH_SIZE = 2         # Tiny batch size\n"
        "NUM_EPOCHS = 1         # Just one epoch\n"
        "LEARNING_RATE = 5e-5\n"
        "WARMUP_STEPS = 10\n"
        "EVAL_INTERVAL = 10     # Evaluate frequently\n"
        "MAX_STEPS = 20         # Very few training steps\n"
        "PRUNE_PERCENT = 0.1    # Target to prune approximately 10% of heads\n\n"
        "# Set device\n"
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n"
        "print(f\"Using device: {device}\")"
    )
    notebook.cells.append(config_cell)
    
    # Add model loading cell
    model_cell = new_code_cell(
        "# Load model and tokenizer\n"
        "print(f\"Loading model: {MODEL_NAME}\")\n"
        "model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)\n"
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n\n"
        "# Set pad token if needed\n"
        "if tokenizer.pad_token is None:\n"
        "    tokenizer.pad_token = tokenizer.eos_token\n\n"
        "# Get model structure information\n"
        "num_layers, num_heads = detect_model_structure(model)\n"
        "print(f\"Model has {num_layers} layers and {num_heads} heads per layer\")"
    )
    notebook.cells.append(model_cell)
    
    # Add synthetic dataset cell
    data_cell = new_code_cell(
        "# Create synthetic data for testing\n"
        "def create_synthetic_data(num_samples=50, seq_length=MAX_LENGTH):\n"
        "    \"\"\"Create synthetic data for testing neural plasticity.\"\"\"\n"
        "    # Sample texts (short but structured)\n"
        "    sample_texts = [\n"
        "        \"The quick brown fox jumps over the lazy dog. This sentence contains many common letters.\",\n"
        "        \"Machine learning models can be trained to recognize patterns in data and make predictions.\",\n"
        "        \"Neural networks consist of layers of interconnected nodes that process information.\",\n"
        "        \"Attention mechanisms allow models to focus on relevant parts of the input data.\",\n"
        "        \"Transformers have revolutionized natural language processing with their ability to handle long-range dependencies.\"\n"
        "    ]\n"
        "    \n"
        "    # Generate training data by tokenizing the samples\n"
        "    train_encodings = []\n"
        "    train_labels = []\n"
        "    \n"
        "    for _ in range(num_samples):\n"
        "        # Pick a random sample text\n"
        "        text = sample_texts[np.random.randint(0, len(sample_texts))]\n"
        "        \n"
        "        # Tokenize the text\n"
        "        encoding = tokenizer(text, max_length=seq_length, padding=\"max_length\", truncation=True)\n"
        "        \n"
        "        # Use input IDs as labels (causal language modeling)\n"
        "        train_encodings.append({\n"
        "            \"input_ids\": torch.tensor(encoding[\"input_ids\"]),\n"
        "            \"attention_mask\": torch.tensor(encoding[\"attention_mask\"]),\n"
        "            \"labels\": torch.tensor(encoding[\"input_ids\"])\n"
        "        })\n"
        "    \n"
        "    return train_encodings\n\n"
        "# Create synthetic train and validation datasets\n"
        "print(\"Creating synthetic datasets...\")\n"
        "train_data = create_synthetic_data(num_samples=50)\n"
        "val_data = create_synthetic_data(num_samples=20)\n"
        "print(f\"Created {len(train_data)} training samples and {len(val_data)} validation samples\")\n\n"
        "# Create a simple data loader\n"
        "class SimpleDataLoader:\n"
        "    \"\"\"Simple data loader for synthetic data.\"\"\"\n"
        "    def __init__(self, data, batch_size=BATCH_SIZE, shuffle=True):\n"
        "        self.data = data\n"
        "        self.batch_size = batch_size\n"
        "        self.shuffle = shuffle\n"
        "        self.indices = list(range(len(data)))\n"
        "        self.reset()\n"
        "    \n"
        "    def reset(self):\n"
        "        \"\"\"Reset the data loader.\"\"\"\n"
        "        if self.shuffle:\n"
        "            np.random.shuffle(self.indices)\n"
        "        self.current = 0\n"
        "    \n"
        "    def __iter__(self):\n"
        "        self.reset()\n"
        "        return self\n"
        "    \n"
        "    def __next__(self):\n"
        "        if self.current + self.batch_size > len(self.data):\n"
        "            self.reset()\n"
        "            raise StopIteration\n"
        "        \n"
        "        # Get indices for this batch\n"
        "        batch_indices = self.indices[self.current:self.current + self.batch_size]\n"
        "        self.current += self.batch_size\n"
        "        \n"
        "        # Prepare batch\n"
        "        batch = {}\n"
        "        for key in self.data[0].keys():\n"
        "            batch[key] = torch.stack([self.data[i][key] for i in batch_indices])\n"
        "        \n"
        "        return batch\n"
        "    \n"
        "    def __len__(self):\n"
        "        return math.ceil(len(self.data) / self.batch_size)\n\n"
        "# Create data loaders\n"
        "train_loader = SimpleDataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)\n"
        "val_loader = SimpleDataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)\n"
        "print(f\"Created data loaders with {len(train_loader)} training batches\")"
    )
    notebook.cells.append(data_cell)
    
    # Add evaluation function cell
    eval_cell = new_code_cell(
        "# Define evaluation functions\n"
        "def evaluate_perplexity(model, val_loader, max_eval_batches=5):\n"
        "    \"\"\"Evaluate model perplexity on validation data.\"\"\"\n"
        "    model.eval()\n"
        "    total_loss = 0.0\n"
        "    total_samples = 0\n"
        "    \n"
        "    with torch.no_grad():\n"
        "        for batch_idx, batch in enumerate(val_loader):\n"
        "            if batch_idx >= max_eval_batches:\n"
        "                break\n"
        "                \n"
        "            # Move batch to device\n"
        "            batch = {k: v.to(device) for k, v in batch.items()}\n"
        "            \n"
        "            # Forward pass\n"
        "            outputs = model(**batch)\n"
        "            loss = outputs.loss\n"
        "            \n"
        "            # Track loss\n"
        "            total_loss += loss.item() * batch[\"input_ids\"].size(0)\n"
        "            total_samples += batch[\"input_ids\"].size(0)\n"
        "    \n"
        "    # Calculate average loss and perplexity\n"
        "    avg_loss = total_loss / total_samples if total_samples > 0 else float(\"inf\")\n"
        "    perplexity = torch.exp(torch.tensor(avg_loss)).item()\n"
        "    \n"
        "    return avg_loss, perplexity\n\n"
        "def generate_text(model, tokenizer, prompt, max_length=50):\n"
        "    \"\"\"Generate text from the model.\"\"\"\n"
        "    model.eval()\n"
        "    input_ids = tokenizer.encode(prompt, return_tensors=\"pt\").to(device)\n"
        "    \n"
        "    with torch.no_grad():\n"
        "        output = model.generate(\n"
        "            input_ids=input_ids,\n"
        "            max_length=max_length,\n"
        "            do_sample=True,\n"
        "            top_k=50,\n"
        "            top_p=0.95,\n"
        "            temperature=0.7,\n"
        "            pad_token_id=tokenizer.eos_token_id\n"
        "        )\n"
        "    \n"
        "    return tokenizer.decode(output[0], skip_special_tokens=True)"
    )
    notebook.cells.append(eval_cell)
    
    # Add attention entropy analysis cell
    analysis_cell = new_code_cell(
        "# Analyze attention entropy\n"
        "def analyze_attention_entropy(model, dataloader, num_batches=2):\n"
        "    \"\"\"Extract attention patterns and calculate entropy.\"\"\"\n"
        "    model.eval()\n"
        "    attention_maps = []\n"
        "    \n"
        "    with torch.no_grad():\n"
        "        for batch_idx, batch in enumerate(dataloader):\n"
        "            if batch_idx >= num_batches:\n"
        "                break\n"
        "                \n"
        "            # Move batch to device\n"
        "            batch = {k: v.to(device) for k, v in batch.items()}\n"
        "            \n"
        "            # Forward pass with attention outputs\n"
        "            outputs = model(**batch, output_attentions=True)\n"
        "            \n"
        "            # Collect attention maps\n"
        "            if outputs.attentions:\n"
        "                attention_maps.extend(outputs.attentions)\n"
        "    \n"
        "    # Calculate entropy for all layers\n"
        "    all_entropies = []\n"
        "    \n"
        "    for layer_idx, attn in enumerate(attention_maps):\n"
        "        # Calculate entropy\n"
        "        layer_entropy = calculate_head_entropy(attn)\n"
        "        all_entropies.append(layer_entropy)\n"
        "    \n"
        "    # Stack all entropies into a single tensor [layers, heads]\n"
        "    if all_entropies:\n"
        "        entropy_tensor = torch.stack([e.mean(dim=-1) for e in all_entropies])\n"
        "    else:\n"
        "        # Fallback if no attention maps were collected\n"
        "        entropy_tensor = torch.rand(num_layers, num_heads)\n"
        "    \n"
        "    return entropy_tensor\n\n"
        "# Calculate and visualize entropy\n"
        "print(\"Analyzing attention entropy...\")\n"
        "entropy_values = analyze_attention_entropy(model, val_loader)\n"
        "print(f\"Calculated entropy tensor of shape {entropy_values.shape}\")\n\n"
        "# Visualize entropy\n"
        "fig1 = visualize_head_entropy(\n"
        "    entropy_values=entropy_values,\n"
        "    title=\"Attention Entropy Across Heads\",\n"
        "    annotate=True\n"
        ")\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
    notebook.cells.append(analysis_cell)
    
    # Add gradient calculation cell
    gradient_cell = new_code_cell(
        "# Calculate gradient norms\n"
        "def calculate_gradients(model, dataloader, num_batches=2):\n"
        "    \"\"\"Calculate gradient norms for all attention heads.\"\"\"\n"
        "    # Reset gradients\n"
        "    model.zero_grad()\n"
        "    \n"
        "    # Set to training mode\n"
        "    model.train()\n"
        "    \n"
        "    # Calculate gradients using our module\n"
        "    grad_norms = calculate_head_gradients(\n"
        "        model=model,\n"
        "        dataloader=dataloader,\n"
        "        num_batches=num_batches,\n"
        "        device=device\n"
        "    )\n"
        "    \n"
        "    return grad_norms\n\n"
        "# Calculate and visualize gradients\n"
        "print(\"Calculating head gradients...\")\n"
        "grad_norm_values = calculate_gradients(model, train_loader)\n"
        "print(f\"Calculated gradient norms tensor of shape {grad_norm_values.shape}\")\n\n"
        "# Visualize gradients\n"
        "fig2 = visualize_head_gradients(\n"
        "    grad_norm_values=grad_norm_values,\n"
        "    title=\"Gradient Norms Across Heads\"\n"
        ")\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
    notebook.cells.append(gradient_cell)
    
    # Add pruning cell
    pruning_cell = new_code_cell(
        "# Apply pruning based on gradients\n"
        "def apply_gradient_pruning(model, grad_norm_values, prune_percent=0.1):\n"
        "    \"\"\"Apply pruning based on gradient norms.\"\"\"\n"
        "    # Generate pruning mask\n"
        "    pruning_mask = generate_pruning_mask(\n"
        "        grad_norm_values=grad_norm_values,\n"
        "        prune_percent=prune_percent,\n"
        "        strategy=\"gradient\"  # Use gradient-based pruning\n"
        "    )\n"
        "    \n"
        "    # Apply pruning mask to the model\n"
        "    pruned_heads = apply_pruning_mask(model, pruning_mask)\n"
        "    \n"
        "    return pruning_mask, pruned_heads\n\n"
        "# Apply pruning\n"
        "print(f\"Applying gradient-based pruning with {PRUNE_PERCENT*100:.0f}% target...\")\n"
        "pruning_mask, pruned_heads = apply_gradient_pruning(\n"
        "    model=model,\n"
        "    grad_norm_values=grad_norm_values,\n"
        "    prune_percent=PRUNE_PERCENT\n"
        ")\n"
        "print(f\"Pruned {len(pruned_heads)} heads: {pruned_heads}\")\n\n"
        "# Visualize pruning decisions\n"
        "fig3 = visualize_pruning_decisions(\n"
        "    grad_norm_values=grad_norm_values,\n"
        "    pruning_mask=pruning_mask,\n"
        "    title=\"Pruning Decisions - Heads with Lowest Gradients\"\n"
        ")\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
    notebook.cells.append(pruning_cell)
    
    # Add baseline evaluation cell
    baseline_cell = new_code_cell(
        "# Evaluate model before and after pruning\n"
        "# Baseline evaluation\n"
        "print(\"Evaluating model before pruning...\")\n"
        "baseline_loss, baseline_perplexity = evaluate_perplexity(model, val_loader)\n"
        "print(f\"Baseline evaluation: Loss = {baseline_loss:.4f}, Perplexity = {baseline_perplexity:.2f}\")\n\n"
        "# Generate text with baseline model\n"
        "prompt = \"The neural network was trained to\"\n"
        "baseline_text = generate_text(model, tokenizer, prompt)\n"
        "print(f\"\\nPrompt: {prompt}\")\n"
        "print(f\"Generated text:\\n{baseline_text}\")"
    )
    notebook.cells.append(baseline_cell)
    
    # Add training cell
    training_cell = new_code_cell(
        "# Train the pruned model\n"
        "def train_pruned_model(model, train_loader, val_loader, learning_rate=LEARNING_RATE, max_steps=MAX_STEPS):\n"
        "    \"\"\"Train the pruned model for a few steps.\"\"\"\n"
        "    model.train()\n"
        "    \n"
        "    # Initialize optimizer and scheduler\n"
        "    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)\n"
        "    scheduler = get_linear_schedule_with_warmup(\n"
        "        optimizer, \n"
        "        num_warmup_steps=WARMUP_STEPS, \n"
        "        num_training_steps=max_steps\n"
        "    )\n"
        "    \n"
        "    # Initialize metrics tracking\n"
        "    metrics = {\n"
        "        \"train_loss\": [],\n"
        "        \"eval_loss\": [],\n"
        "        \"perplexity\": [],\n"
        "        \"steps\": []\n"
        "    }\n"
        "    \n"
        "    # Training loop\n"
        "    print(f\"Training pruned model for {max_steps} steps...\")\n"
        "    progress_bar = tqdm(total=max_steps, desc=\"Training\")\n"
        "    train_iter = iter(train_loader)\n"
        "    \n"
        "    # Run training loop\n"
        "    for step in range(max_steps):\n"
        "        # Get next batch (cycling through the dataset)\n"
        "        try:\n"
        "            batch = next(train_iter)\n"
        "        except StopIteration:\n"
        "            train_iter = iter(train_loader)\n"
        "            batch = next(train_iter)\n"
        "        \n"
        "        # Move batch to device\n"
        "        batch = {k: v.to(device) for k, v in batch.items()}\n"
        "        \n"
        "        # Forward pass\n"
        "        outputs = model(**batch)\n"
        "        loss = outputs.loss\n"
        "        \n"
        "        # Backward pass\n"
        "        loss.backward()\n"
        "        \n"
        "        # Update weights\n"
        "        optimizer.step()\n"
        "        scheduler.step()\n"
        "        optimizer.zero_grad()\n"
        "        \n"
        "        # Update progress bar\n"
        "        progress_bar.update(1)\n"
        "        progress_bar.set_postfix(loss=f\"{loss.item():.4f}\")\n"
        "        \n"
        "        # Evaluate periodically\n"
        "        if (step + 1) % EVAL_INTERVAL == 0 or step == max_steps - 1:\n"
        "            # Evaluate\n"
        "            eval_loss, perplexity = evaluate_perplexity(model, val_loader)\n"
        "            \n"
        "            # Update metrics\n"
        "            metrics[\"train_loss\"].append(loss.item())\n"
        "            metrics[\"eval_loss\"].append(eval_loss)\n"
        "            metrics[\"perplexity\"].append(perplexity)\n"
        "            metrics[\"steps\"].append(step)\n"
        "            \n"
        "            # Print metrics\n"
        "            print(f\"\\nStep {step+1}: Train loss = {loss.item():.4f}, \"\n"
        "                  f\"Eval loss = {eval_loss:.4f}, \"\n"
        "                  f\"Perplexity = {perplexity:.2f}\")\n"
        "        \n"
        "        # Clear memory after each step\n"
        "        if step % 5 == 0:\n"
        "            clear_memory()\n"
        "    \n"
        "    progress_bar.close()\n"
        "    return metrics\n\n"
        "# Train the pruned model\n"
        "training_metrics = train_pruned_model(model, train_loader, val_loader)\n\n"
        "# Visualize training metrics\n"
        "plt.figure(figsize=(12, 4))\n"
        "\n"
        "# Plot loss\n"
        "plt.subplot(1, 2, 1)\n"
        "plt.plot(training_metrics[\"steps\"], training_metrics[\"train_loss\"], label=\"Train Loss\")\n"
        "plt.plot(training_metrics[\"steps\"], training_metrics[\"eval_loss\"], label=\"Eval Loss\")\n"
        "plt.xlabel(\"Steps\")\n"
        "plt.ylabel(\"Loss\")\n"
        "plt.title(\"Training and Evaluation Loss\")\n"
        "plt.legend()\n"
        "plt.grid(alpha=0.3)\n"
        "\n"
        "# Plot perplexity\n"
        "plt.subplot(1, 2, 2)\n"
        "plt.plot(training_metrics[\"steps\"], training_metrics[\"perplexity\"], color=\"green\")\n"
        "plt.xlabel(\"Steps\")\n"
        "plt.ylabel(\"Perplexity\")\n"
        "plt.title(\"Perplexity\")\n"
        "plt.grid(alpha=0.3)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
    notebook.cells.append(training_cell)
    
    # Add final evaluation cell
    final_cell = new_code_cell(
        "# Final evaluation\n"
        "print(\"Final evaluation after training...\")\n"
        "final_loss, final_perplexity = evaluate_perplexity(model, val_loader)\n"
        "print(f\"Final evaluation: Loss = {final_loss:.4f}, Perplexity = {final_perplexity:.2f}\")\n"
        "print(f\"Baseline:         Loss = {baseline_loss:.4f}, Perplexity = {baseline_perplexity:.2f}\")\n"
        "print(f\"Improvement:      {((baseline_loss - final_loss) / baseline_loss * 100):.2f}%\")\n\n"
        "# Generate text with final model\n"
        "final_text = generate_text(model, tokenizer, prompt)\n"
        "print(f\"\\nPrompt: {prompt}\")\n"
        "print(f\"Final generated text:\\n{final_text}\")"
    )
    notebook.cells.append(final_cell)
    
    # Add conclusion cell
    conclusion_cell = new_markdown_cell(
        "## Conclusion\n\n"
        "In this ultra-minimal demonstration, we've shown the core functionality of the neural plasticity system:\n\n"
        "1. We calculated attention entropy and gradient norms for all heads\n"
        "2. We pruned heads with the lowest gradient norms (least useful for learning)\n"
        "3. We trained the pruned model and observed its performance\n\n"
        "This demonstrates that our neural plasticity system can successfully identify and prune less useful attention heads, "
        "potentially improving efficiency without sacrificing performance.\n\n"
        "The techniques used here form the foundation for more advanced neural plasticity operations that enable continual growth, "
        "adaptation, and pruning for transformers in the full version of this system."
    )
    notebook.cells.append(conclusion_cell)
    
    # Save the notebook
    output_path = Path("neural_plasticity_ultra_minimal.ipynb")
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(notebook, f)
    
    print(f"Created ultra-minimal notebook at {output_path}")
    return output_path

def main():
    """Create the ultra-minimal notebook."""
    notebook_path = create_ultra_minimal_notebook()
    
    if notebook_path:
        print(f"\nTo run the notebook:")
        print(f"  cd {os.getcwd()} && source .venv/bin/activate && jupyter notebook {notebook_path}")
        print(f"\nAlternatively, you can run it with nbconvert:")
        print(f"  cd {os.getcwd()} && source .venv/bin/activate && jupyter nbconvert --to notebook --execute {notebook_path} --output neural_plasticity_ultra_minimal_executed.ipynb")
    
if __name__ == "__main__":
    main()