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
                
                # Get current timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Update title with new version and timestamp
                title_cell.source = title_cell.source.replace(
                    f"v{current_version}", f"v{new_version} {timestamp}"
                )
                
                # Add new changelog entry at the top of the "New in v..." section
                changelog_entry = f"\n### New in v{new_version}:\n- Modularized notebook components for better reusability\n- Integrated with utils.neural_plasticity module\n- Improved tensor handling with safe_tensor_imshow\n- Enhanced visualization utilities\n- Added type hints and documentation\n- Removed duplicated code\n"
                
                new_in_match = re.search(r'### New in v', title_cell.source)
                if new_in_match:
                    # Insert at the beginning of the changelog section
                    insert_pos = new_in_match.start()
                    title_cell.source = title_cell.source[:insert_pos] + changelog_entry + title_cell.source[insert_pos:]
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
    """
    Make pruning decisions based only on gradient norms.
    We want to prune heads with LOWEST gradient norms, as they're
    learning the least.
    
    Args:
        grad_norm_values: Tensor of gradient norm values for all heads
        prune_percent: Target percentage of heads to prune (0-1)
        
    Returns:
        pruning_mask: Boolean tensor where True indicates a head should be pruned
    """
    # Use the module function
    return generate_pruning_mask(
        grad_norm_values=grad_norm_values,
        prune_percent=prune_percent,
        strategy="gradient"
    )
"""
    
    # Update visualization cells to use the module functions
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code' and 'safe_tensor_imshow(' in cell.source:
            # Update to use the module visualization functions
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


if __name__ == "__main__":
    # Path to the notebook
    notebook_path = Path("colab_notebooks/NeuralPlasticityDemo.ipynb")
    
    # Make backup
    backup_path = notebook_path.with_stem(f"{notebook_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    import shutil
    shutil.copy2(notebook_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Update notebook
    update_notebook_to_use_modules(notebook_path)
    print("Notebook updated successfully")