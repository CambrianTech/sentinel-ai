#!/usr/bin/env python
"""
Update Neural Plasticity Notebook

This script updates the NeuralPlasticityDemo.ipynb notebook to use the
modularized neural plasticity components.

It performs the following changes:
1. Updates import statements to use the new modules
2. Replaces inline function definitions with module imports
3. Updates visualization code to use the new visualization utilities
4. Ensures proper tensor handling with safe_tensor_imshow

Usage:
  python scripts/update_neural_plasticity_notebook.py
"""

import os
import json
import nbformat
from datetime import datetime
from pathlib import Path
import uuid


def update_notebook_to_use_modules(notebook_path, output_path=None):
    """
    Update notebook to use modularized components.
    
    Args:
        notebook_path: Path to the notebook
        output_path: Path to save the updated notebook (defaults to original path)
    """
    # Read notebook
    with open(notebook_path, 'r') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # If no output path provided, modify in-place
    if output_path is None:
        output_path = notebook_path
    
    # Create unique identifier to bypass cache
    unique_id = str(uuid.uuid4())[:8]
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Update version and timestamp in title cell
    try:
        title_cell = notebook.cells[0]
        if title_cell.cell_type == 'markdown' and title_cell.source.startswith('# Neural Plasticity Demo'):
            # Extract current version
            import re
            version_match = re.search(r'v(\d+\.\d+\.\d+)', title_cell.source)
            if version_match:
                current_version = version_match.group(1)
                # Increment patch version
                version_parts = current_version.split('.')
                new_version = f"{version_parts[0]}.{version_parts[1]}.{int(version_parts[2]) + 1}"
                
                # No need to do this replacement since we're rebuilding the entire content
                # Just kept for reference of the original approach
                
                # Create new changelog entry (now directly used in the title cell)
                
                # Create completely fresh title cell with only the current version's changes
                new_content = f"""# Neural Plasticity Demo: Dynamic Pruning & Regrowth (v{new_version} {current_time})

### Changes in v{new_version}:
- Modularized notebook components for better reusability
- Integrated with utils.neural_plasticity module
- Improved tensor handling with safe_tensor_imshow
- Enhanced visualization utilities
- Added type hints and documentation
- Removed duplicated code
- Added unique ID ({unique_id}) to bypass Colab caching

This notebook demonstrates Sentinel AI's neural plasticity system, which allows transformer models to dynamically prune and regrow attention heads during training based on utility metrics.

## What is Neural Plasticity?

Neural plasticity is the ability of neural networks to adapt their structure over time through pruning (removing unused connections) and regrowth (restoring useful connections). This mimics how biological brains form efficient neural pathways.

In this demo, we:
1. Track the entropy and gradient patterns of each attention head
2. Dynamically prune high-entropy, low-gradient heads (unfocused, less useful)
3. Selectively revive low-entropy, higher-gradient heads (potentially useful)
4. Visualize the "brain dynamics" over time

This allows models to form more efficient neural structures during training."""
                
                # Replace the entire cell content
                title_cell.source = new_content
    except Exception as e:
        print(f"Error updating title: {e}")
    
    # Update import cell
    found_import_cell = False
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code' and '%matplotlib inline' in cell.source and 'import torch' in cell.source:
            found_import_cell = True
            
            # Create updated imports with neural plasticity modules
            updated_imports = """
%matplotlib inline
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    default_data_collator,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader
from datasets import load_dataset

# Import neural plasticity modules
from utils.neural_plasticity.core import (
    calculate_head_entropy,
    calculate_head_gradients,
    generate_pruning_mask,
    apply_pruning_mask,
    evaluate_model
)

from utils.neural_plasticity.visualization import (
    visualize_head_entropy,
    visualize_head_gradients,
    visualize_pruning_decisions,
    visualize_training_metrics,
    visualize_attention_patterns
)

from utils.neural_plasticity.training import (
    create_plasticity_trainer,
    run_plasticity_loop,
    train_with_plasticity
)

# Import visualization utilities
from utils.colab.visualizations import TrainingMonitor
from utils.colab.helpers import safe_tensor_imshow

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
print(f"Loading model: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Set pad token if needed
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load datasets
print(f"Loading dataset: {DATASET}/{DATASET_CONFIG}")
train_dataset = load_dataset(DATASET, DATASET_CONFIG, split="train")
validation_dataset = load_dataset(DATASET, DATASET_CONFIG, split="validation")

# Define tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_LENGTH
    )

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
validation_dataset = validation_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Add labels for language modeling
def add_labels(examples):
    examples["labels"] = examples["input_ids"].copy()
    return examples

train_dataset = train_dataset.map(add_labels)
validation_dataset = validation_dataset.map(add_labels)

# Set format
train_dataset = train_dataset.with_format("torch")
validation_dataset = validation_dataset.with_format("torch")

# Create dataloaders
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=default_data_collator
)

validation_dataloader = DataLoader(
    validation_dataset, 
    batch_size=BATCH_SIZE, 
    collate_fn=default_data_collator
)

print(f"Train dataset size: {len(train_dataset)} examples")
print(f"Validation dataset size: {len(validation_dataset)} examples")

# Print unique ID to verify cache bypass
print(f"Running modularized neural plasticity code [ID: {unique_id}]")
"""
            # Update the import cell
            cell.source = updated_imports
            break
    
    if not found_import_cell:
        print("Warning: Could not find import cell to update")
    
    # Update custom pruning function cell
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code' and 'def gradient_based_pruning(' in cell.source:
            # Replace with import from module
            cell.source = """# Custom function to apply pruning based purely on gradients
def gradient_based_pruning(grad_norm_values, prune_percent=0.1):
    \"\"\"
    Make pruning decisions based only on gradient norms.
    We want to prune heads with LOWEST gradient norms, as they're
    learning the least.
    
    Args:
        grad_norm_values: Tensor of gradient norm values for all heads
        prune_percent: Target percentage of heads to prune (0-1)
        
    Returns:
        pruning_mask: Boolean tensor where True indicates a head should be pruned
    \"\"\"
    # Use the module function
    return generate_pruning_mask(
        grad_norm_values=grad_norm_values,
        prune_percent=prune_percent,
        strategy="gradient"
    )
"""
    
    # Update visualization cells to use the module functions and fix CPU conversion
    for i, cell in enumerate(notebook.cells):
        # First fix redundant .cpu().numpy() calls
        if cell.cell_type == 'code' and 'detach().cpu().numpy().cpu().numpy()' in cell.source:
            cell.source = cell.source.replace('.detach().cpu().numpy().cpu().numpy()', '.detach().cpu().numpy()')
            cell.source = cell.source.replace('.cpu().numpy().cpu().numpy()', '.cpu().numpy()')
            cell.source = cell.source.replace('.cpu().numpy())).cpu().numpy()', '.cpu().numpy())')
            cell.source = cell.source.replace('.cpu().numpy()))))', '.cpu().numpy())')

        # Then update to use the module visualization functions
        if cell.cell_type == 'code' and 'safe_tensor_imshow(' in cell.source:
            if 'entropy_values' in cell.source and 'visualize_entropy' in cell.source:
                cell.source = cell.source.replace(
                    'safe_tensor_imshow(entropy_values',
                    'visualize_head_entropy(entropy_values'
                )
            elif 'grad_norm_values' in cell.source and 'visualize_gradient' in cell.source:
                cell.source = cell.source.replace(
                    'safe_tensor_imshow(grad_norm_values',
                    'visualize_head_gradients(grad_norm_values'
                )
            elif 'attention_map' in cell.source and 'Attention pattern' in cell.source:
                # This is an attention visualization
                cell.source = cell.source.replace(
                    'attention_map = safe_tensor_imshow(attn[0, head_idx]',
                    'visualize_attention_patterns(attn, layer_idx=layer_idx, head_idx=head_idx'
                )
    
    # Write updated notebook
    print(f"Writing updated notebook to {output_path}")
    with open(output_path, 'w') as f:
        nbformat.write(notebook, f)

    return unique_id


if __name__ == "__main__":
    # Path to the notebook
    notebook_path = Path("colab_notebooks/NeuralPlasticityDemo.ipynb")
    
    # Make backup
    backup_path = notebook_path.with_stem(f"{notebook_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    import shutil
    shutil.copy2(notebook_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Update notebook
    unique_id = update_notebook_to_use_modules(notebook_path)
    print(f"Notebook updated successfully with unique ID: {unique_id}")
    print("You can verify you're running the updated version by checking for this ID in the notebook title and logs.")