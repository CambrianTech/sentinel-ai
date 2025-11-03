#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create a properly structured Jupyter notebook for pruning and fine-tuning experiments.
This script generates a notebook with multiple cells properly organized with markdown
and code sections.
"""

import json
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# Create properly structured notebook
notebook = new_notebook()

# Add header markdown cell
notebook.cells.append(new_markdown_cell('''# Make a GPT-2 Model Smaller and More Powerful (v0.0.34)

This notebook demonstrates how to make a GPT-2 model both smaller and more powerful by:
1. Applying pruning to remove less important attention heads
2. Fine-tuning the pruned model to recover and improve performance
3. Showing clear metrics of improvement throughout the process

We use real data (Wikitext) rather than synthetic data for realistic evaluation.

Version History:
- v0.0.34 (April 2025): Fixed undefined variable error, visualization issues and enhanced CUDA error handling
- v0.0.33 (April 2025): Fixed visualization issues, improved model compatibility and enhanced error handling
- v0.0.32 (April 2025): Added CUDA error handling for Colab compatibility and memory management
- v0.0.31 (April 2025): Fixed get_strategy parameters issue and improved Colab compatibility
- v0.0.30 (April 2025): Added OPT model support and chart improvements

---
**Note**: This notebook is part of the SentinelAI project. For detailed documentation, see `PruningAndFineTuningColab.md`.'''))

# Add setup markdown cell
notebook.cells.append(new_markdown_cell('''## Setup

Let's start by installing the required dependencies and configuring our environment.'''))

# Add setup code cell
notebook.cells.append(new_code_cell('''# Memory management utility for Colab
def display_available_memory():
    """Display available memory in Colab."""
    if IS_COLAB:
        # Get GPU memory info
        try:
            !nvidia-smi --query-gpu=memory.total,memory.used --format=csv
        except:
            pass
        
        # Get system memory info
        !free -h

# Install required packages
!pip install -q transformers==4.38.0 datasets==2.17.0 torch matplotlib tqdm

# Check if we're running in Colab
try:
    import google.colab
    IS_COLAB = True
    print("Running in Google Colab!")
    
    # Add file download helper for Colab
    from google.colab import files
    
    def download_files(file_paths):
        """Helper function to download files from Colab."""
        for file_path in file_paths:
            if os.path.exists(file_path):
                files.download(file_path)
                print(f"Downloaded: {file_path}")
            else:
                print(f"File not found: {file_path}")
    
    # Free up memory for Colab
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    
    # Display memory status
    display_available_memory()
    
except:
    IS_COLAB = False
    print("Not running in Google Colab")
    
    # Dummy function for non-Colab environments
    def download_files(file_paths):
        print("File download only works in Google Colab")
        print(f"Files would be downloaded: {file_paths}")
        
    def display_available_memory():
        print("Memory display not available outside Colab")'''))

# Add imports markdown cell
notebook.cells.append(new_markdown_cell('''## Imports and Configuration

Import required libraries and set up the configuration for the experiment.'''))

# Add imports code cell
notebook.cells.append(new_code_cell('''import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from tqdm.notebook import tqdm
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    get_linear_schedule_with_warmup, 
    GPT2LMHeadModel
)

# Initialize plotting style
plt.style.use('seaborn-v0_8-pastel')

# Configure device and optimize for Colab environment
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Half-precision for GPU to reduce memory usage
USE_FP16 = DEVICE == "cuda"

# Handle TPU if available (Colab-specific optimization)
if 'COLAB_TPU_ADDR' in os.environ:
    try:
        import torch_xla.core.xla_model as xm
        DEVICE = xm.xla_device()
        print(f"TPU detected and configured!")
        USE_FP16 = False  # TPUs have their own optimization
    except ImportError:
        print("TPU environment detected but torch_xla not installed.")

# Set up directories
OUTPUT_DIR = "pruning_results"
MODEL_CACHE_DIR = "model_cache"
DATA_DIR = "data"

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Using FP16: {USE_FP16}")
print(f"PyTorch version: {torch.__version__}")

# CUDA memory management helper
def clear_gpu_memory():
    """Clear GPU memory to avoid CUDA out of memory errors."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("GPU memory cleared")

# Import garbage collector for memory management
import gc

# For better GPU memory management, we'll use a context manager
try:
    import contextlib
    @contextlib.contextmanager
    def autocast_if_available():
        """Use autocast if available for better memory efficiency."""
        if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast') and USE_FP16:
            with torch.cuda.amp.autocast():
                yield
        else:
            yield
except:
    # Fallback if the import fails
    @contextlib.contextmanager
    def autocast_if_available():
        yield'''))

# Add progress tracking markdown cell
notebook.cells.append(new_markdown_cell('''## Progress Tracking

We'll create a class to track metrics and visualize progress throughout the pruning and fine-tuning process.'''))

# Add progress metrics class
notebook.cells.append(new_code_cell('''class ProgressMetrics:
    """Track metrics throughout the pruning and fine-tuning process."""
    
    def __init__(self):
        self.metrics = {
            "loss": [],
            "perplexity": [],
            "steps": [],
            "pruning_level": None,
            "strategy": None,
            "pruned_heads": [],
            "gate_values": [],
            "head_importance": [],
            "generation_samples": []
        }
        
        # Create visualizations
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 5))
        self.loss_line = None
        self.ppl_line = None
        
    def update(self, step, loss, perplexity, head_info=None, gate_values=None, 
               generation_sample=None):
        """Update metrics with new values."""
        self.metrics["steps"].append(step)
        self.metrics["loss"].append(loss)
        self.metrics["perplexity"].append(perplexity)
        
        if head_info is not None:
            self.metrics["head_importance"] = head_info
            
        if gate_values is not None:
            self.metrics["gate_values"] = gate_values
            
        if generation_sample is not None:
            self.metrics["generation_samples"].append({
                "step": step,
                "text": generation_sample
            })
        
        # Update visualization
        self._update_plots()
        
    def set_pruning_info(self, strategy, level, pruned_heads):
        """Set pruning information."""
        self.metrics["strategy"] = strategy
        self.metrics["pruning_level"] = level
        self.metrics["pruned_heads"] = pruned_heads
        
    def _update_plots(self):
        """Update visualization plots."""
        steps = self.metrics["steps"]
        loss = self.metrics["loss"]
        ppl = self.metrics["perplexity"]
        
        if not steps:
            return
            
        # Clear previous plots
        self.axes[0].clear()
        self.axes[1].clear()
        
        # Plot loss
        self.axes[0].plot(steps, loss, 'b-')
        self.axes[0].set_title('Training Loss')
        self.axes[0].set_xlabel('Step')
        self.axes[0].set_ylabel('Loss')
        self.axes[0].grid(True)
        
        # Plot perplexity
        self.axes[1].plot(steps, ppl, 'r-')
        self.axes[1].set_title('Perplexity (lower is better)')
        self.axes[1].set_xlabel('Step')
        self.axes[1].set_ylabel('Perplexity')
        self.axes[1].grid(True)
        
        self.fig.tight_layout()
        plt.draw()
        plt.pause(0.001)
        
    def save_plots(self, path):
        """Save plots to file."""
        plt.savefig(path)
        
    def save_metrics(self, path):
        """Save metrics to file."""
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def get_summary(self):
        """Return a summary of key metrics."""
        if not self.metrics["perplexity"] or len(self.metrics["perplexity"]) <= 1:
            return {"error": "Not enough data points for summary"}
            
        return {
            "strategy": self.metrics["strategy"],
            "pruning_level": self.metrics["pruning_level"],
            "pruned_heads_count": len(self.metrics["pruned_heads"]),
            "initial_loss": self.metrics["loss"][0],
            "final_loss": self.metrics["loss"][-1],
            "initial_perplexity": self.metrics["perplexity"][0],
            "final_perplexity": self.metrics["perplexity"][-1],
            "improvement_percent": ((self.metrics["perplexity"][0] - self.metrics["perplexity"][-1]) / 
                                   self.metrics["perplexity"][0] * 100)
        }'''))

# Add data loading markdown cell
notebook.cells.append(new_markdown_cell('''## Data Loading

We'll use the Wikitext-2 dataset for fine-tuning and evaluation, which provides real-world text content.'''))

# Add data loading code cell
notebook.cells.append(new_code_cell('''def setup_directories():
    """Create necessary directories for outputs and data."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    return OUTPUT_DIR, MODEL_CACHE_DIR, DATA_DIR

def download_wikitext():
    """Download Wikitext dataset if not already present."""
    wikitext_file = os.path.join(DATA_DIR, "wikitext-2-raw-v1-validation.txt")
    
    if not os.path.exists(wikitext_file):
        print("Downloading Wikitext-2 dataset...")
        try:
            # Using HF datasets library
            from datasets import load_dataset
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
            
            # Save validation text
            with open(wikitext_file, "w", encoding="utf-8") as f:
                for item in tqdm(dataset["validation"], desc="Saving dataset"):
                    if item["text"].strip():
                        f.write(item["text"] + "\\n")
                        
            print(f"Dataset saved to {wikitext_file}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            
            # Fallback: download using requests
            try:
                import requests
                url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
                r = requests.get(url)
                
                # Save zip file
                zip_path = os.path.join(DATA_DIR, "wikitext-2-raw-v1.zip")
                with open(zip_path, "wb") as f:
                    f.write(r.content)
                
                # Extract
                import zipfile
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(DATA_DIR)
                
                print(f"Dataset downloaded and extracted to {DATA_DIR}")
            except Exception as e2:
                print(f"Fallback download also failed: {e2}")
                return False
    
    return True

def prepare_dataset(paragraphs, tokenizer, max_length, batch_size):
    """Tokenize and prepare paragraphs into a PyTorch dataset."""
    # Tokenize
    tokenized = tokenizer(
        paragraphs,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    
    # Create dataset
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def load_wikitext_data(tokenizer, max_length=512, batch_size=4):
    """Load and prepare Wikitext data for fine-tuning and evaluation."""
    wikitext_file = os.path.join(DATA_DIR, "wikitext-2-raw-v1-validation.txt")
    
    if not os.path.exists(wikitext_file):
        success = download_wikitext()
        if not success:
            print("Failed to download dataset")
            return None, None
    
    # Read the data
    print("Loading Wikitext-2 data...")
    with open(wikitext_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Split into train and validation (80/20)
    paragraphs = [p for p in text.split("\\n\\n") if p.strip()]
    
    # Ensure we have at least 100 paragraphs of reasonable length
    paragraphs = [p for p in paragraphs if len(p) > 100]
    
    if len(paragraphs) < 100:
        # Fall back to splitting by newline if needed
        paragraphs = [p for p in text.split("\\n") if len(p.strip()) > 100]
    
    # Shuffle and split
    np.random.seed(42)
    np.random.shuffle(paragraphs)
    
    split_idx = int(len(paragraphs) * 0.8)
    train_paragraphs = paragraphs[:split_idx]
    val_paragraphs = paragraphs[split_idx:]
    
    print(f"Tokenizing {len(train_paragraphs)} training and {len(val_paragraphs)} validation paragraphs...")
    
    # Tokenize and prepare datasets
    train_data = prepare_dataset(train_paragraphs, tokenizer, max_length, batch_size)
    val_data = prepare_dataset(val_paragraphs, tokenizer, max_length, batch_size)
    
    return train_data, val_data'''))

# Add model loading markdown cell
notebook.cells.append(new_markdown_cell('''## Model Loading

Load the pre-trained model and prepare it for pruning.'''))

# Add model loading code cell
notebook.cells.append(new_code_cell('''def load_model_and_tokenizer(model_name, cache_dir=None):
    """Load pre-trained model and tokenizer."""
    print(f"Loading model: {model_name}")
    
    # Determine model type from name
    if "gpt2" in model_name.lower():
        model_type = "gpt2"
    elif "opt" in model_name.lower() or "facebook" in model_name.lower():
        model_type = "opt"
    elif "pythia" in model_name.lower() or "eleutherai" in model_name.lower():
        model_type = "pythia"
    else:
        model_type = "gpt2"  # Default to gpt2
        
    print(f"Detected model type: {model_type}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with potential FP16 optimization
    try:
        if USE_FP16:
            print("Using FP16 for model loading")
            # For FP16, we need to set torch_dtype
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                torch_dtype=torch.float16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    except Exception as e:
        print(f"Error loading model with AutoModelForCausalLM: {e}")
        print("Falling back to GPT2LMHeadModel")
        model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)
    
    model.to(DEVICE)
    
    # Store model type for later use
    model.model_type = model_type
    
    # Print model size information
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {param_count/1e6:.2f}M parameters")
    
    # Add head_count attribute if we can determine it
    try:
        # Count number of attention heads
        if hasattr(model.config, "n_head"):
            model.head_count = model.config.n_head
        elif hasattr(model.config, "num_attention_heads"):
            model.head_count = model.config.num_attention_heads
        elif hasattr(model.config, "num_heads"):
            model.head_count = model.config.num_heads
        else:
            model.head_count = 12  # Reasonable default
        print(f"Model has {model.head_count} attention heads per layer")
    except Exception as e:
        print(f"Could not determine head count: {e}")
        model.head_count = 12  # Reasonable default
    
    return model, tokenizer'''))

# Add attention module extraction markdown cell
notebook.cells.append(new_markdown_cell('''## Attention Module Extraction

Identify and extract attention modules from the model architecture.'''))

# Add attention module extraction code cell
notebook.cells.append(new_code_cell('''def get_attention_modules(model):
    """Extract attention modules from model."""
    # Set default model type if not already set
    if not hasattr(model, "model_type"):
        model.model_type = "gpt2"
    
    attention_modules = []
    
    # Function to safely get attribute path
    def get_nested_attr(obj, attr_path):
        """Safely get attribute path without raising AttributeError."""
        attrs = attr_path.split(".")
        current = obj
        for attr in attrs:
            if hasattr(current, attr):
                current = getattr(current, attr)
            else:
                return None
        return current
    
    # Try different model architectures
    if model.model_type == "gpt2":
        # GPT-2 style models
        transformer = get_nested_attr(model, "transformer")
        if transformer:
            blocks = get_nested_attr(transformer, "h")
            if blocks:
                for i, block in enumerate(blocks):
                    attn = get_nested_attr(block, "attn")
                    if attn:
                        attention_modules.append((i, attn))
    
    elif model.model_type == "opt":
        # OPT models
        model_module = get_nested_attr(model, "model")
        if model_module:
            decoder = get_nested_attr(model_module, "decoder")
            if decoder:
                layers = get_nested_attr(decoder, "layers")
                if layers:
                    for i, layer in enumerate(layers):
                        self_attn = get_nested_attr(layer, "self_attn")
                        if self_attn:
                            attention_modules.append((i, self_attn))
    
    elif model.model_type == "pythia":
        # Pythia models (similar to GPT-2)
        transformer = get_nested_attr(model, "transformer") or get_nested_attr(model, "gpt_neox")
        if transformer:
            blocks = get_nested_attr(transformer, "h") or get_nested_attr(transformer, "layers")
            if blocks:
                for i, block in enumerate(blocks):
                    attn = get_nested_attr(block, "attn") or get_nested_attr(block, "attention")
                    if attn:
                        attention_modules.append((i, attn))
    
    # Generic fallback if nothing matched
    if not attention_modules:
        # Try common patterns across different architectures
        candidate_paths = [
            "transformer.h",              # GPT-2 style
            "model.decoder.layers",       # OPT style
            "encoder.layers",             # Encoder style models
            "decoder.layers",             # Decoder style models
            "layers",                     # Direct layers attribute
            "transformer.layers",         # Some transformers
            "gpt_neox.layers"             # Pythia/GPT-NeoX
        ]
        
        for path in candidate_paths:
            try:
                blocks = get_nested_attr(model, path)
                if blocks and isinstance(blocks, (list, tuple)) or hasattr(blocks, "__getitem__"):
                    for i, block in enumerate(blocks):
                        # Try common attention module names
                        for attn_name in ["attn", "attention", "self_attn", "self_attention", "mha"]:
                            attn = get_nested_attr(block, attn_name)
                            if attn:
                                attention_modules.append((i, attn))
                                break
                    
                    if attention_modules:
                        # Found some attention modules, can stop looking
                        break
            except Exception as e:
                continue
    
    if attention_modules:
        print(f"Found {len(attention_modules)} attention modules")
        
        # Try to add head_size attribute if not present
        for _, attn in attention_modules:
            if not hasattr(attn, "head_size") and hasattr(model, "head_count"):
                # Try to determine head size from attention module
                if hasattr(attn, "head_dim"):
                    attn.head_size = attn.head_dim
                elif hasattr(model.config, "hidden_size"):
                    attn.head_size = model.config.hidden_size // model.head_count
                elif hasattr(attn, "q_proj") and hasattr(attn.q_proj, "weight"):
                    # Common in models like OPT
                    attn.head_size = attn.q_proj.weight.shape[0] // model.head_count
                elif hasattr(attn, "c_attn") and hasattr(attn.c_attn, "weight"):
                    # Common in GPT-2 models
                    q_weight = attn.c_attn.weight
                    attn.head_size = q_weight.shape[1] // (3 * model.head_count)
            
            # Add num_heads attribute if not present
            if not hasattr(attn, "num_heads") and hasattr(model, "head_count"):
                attn.num_heads = model.head_count
    else:
        print("Warning: Could not find attention modules. Unsupported model architecture.")
    
    return attention_modules'''))

# Add model evaluation markdown cell
notebook.cells.append(new_markdown_cell('''## Model Evaluation

Define functions to evaluate model performance and generate text.'''))

# Add model evaluation code cell
notebook.cells.append(new_code_cell('''def evaluate_model(model, dataloader):
    """Evaluate model loss and perplexity on the provided dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    # Don't track gradients for evaluation
    with torch.no_grad():
        with autocast_if_available():
            for batch_idx, (input_ids, attention_mask) in enumerate(tqdm(dataloader, desc="Evaluating")):
                try:
                    # Move tensors to the correct device
                    input_ids = input_ids.to(DEVICE)
                    attention_mask = attention_mask.to(DEVICE)
                    
                    # Create labels (shift input_ids right)
                    labels = input_ids.clone()
                    
                    # Forward pass
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    
                    # Accumulate loss
                    batch_tokens = torch.sum(attention_mask).item()
                    total_loss += loss.item() * batch_tokens
                    total_tokens += batch_tokens
                    
                except Exception as e:
                    if DEVICE == "cuda" and "CUDA" in str(e):
                        print(f"CUDA error during evaluation at batch {batch_idx}: {e}")
                        print("Attempting to continue evaluation on CPU...")
                        
                        # Transfer model to CPU for this batch
                        model = model.cpu()
                        DEVICE_BACKUP = "cpu"
                        
                        # Try again on CPU
                        input_ids = input_ids.to(DEVICE_BACKUP)
                        attention_mask = attention_mask.to(DEVICE_BACKUP)
                        labels = labels.to(DEVICE_BACKUP)
                        
                        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                        
                        batch_tokens = torch.sum(attention_mask).item()
                        total_loss += loss.item() * batch_tokens
                        total_tokens += batch_tokens
                        
                        # Move model back to GPU for next batches
                        model = model.to(DEVICE)
                    else:
                        print(f"Error during evaluation at batch {batch_idx}: {e}")
                        # Skip this batch
                        continue
    
    # Calculate average loss and perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7):
    """Generate text from the model with the given prompt."""
    print(f"Generating text with prompt: '{prompt}'")
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Tokenize the prompt
    encoded_prompt = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded_prompt["input_ids"].to(DEVICE)
    attention_mask = encoded_prompt["attention_mask"].to(DEVICE)
    
    try:
        with torch.no_grad():
            with autocast_if_available():
                # Generate text
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
        
        # Decode the generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text
    
    except Exception as e:
        if DEVICE == "cuda" and "CUDA" in str(e):
            print(f"CUDA error during text generation: {e}")
            print("Falling back to CPU for generation...")
            
            # Free up memory
            clear_gpu_memory()
            
            # Move to CPU and try again
            cpu_model = model.cpu()
            input_ids = input_ids.cpu()
            attention_mask = attention_mask.cpu()
            
            try:
                with torch.no_grad():
                    output = cpu_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=max_length,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Move model back to GPU
                model.to(DEVICE)
                
                # Decode the generated text
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                return generated_text
            except Exception as e2:
                print(f"Generation also failed on CPU: {e2}")
                return f"[Generation failed: {str(e2)}]"
        else:
            print(f"Error during text generation: {e}")
            return f"[Generation failed: {str(e)}]"'''))

# Add head importance markdown cell
notebook.cells.append(new_markdown_cell('''## Head Importance Calculation

Calculate the importance of each attention head using different strategies.'''))

# Add head importance code cell
notebook.cells.append(new_code_cell('''def register_attention_hooks(model, attention_modules):
    """Register hooks to collect attention patterns."""
    attention_patterns = []
    
    def hook_fn(module, input, output):
        # For GPT-2, output[0] contains attention weights of shape [batch_size, num_heads, seq_len, seq_len]
        # Store it for later analysis
        attention_patterns.append(output[0].detach())
    
    hooks = []
    for layer_idx, attn_module in attention_modules:
        # Check if module has the expected attributes for attention weights
        if hasattr(attn_module, "_attn") and callable(attn_module._attn):
            # GPT-2 style hook
            hook = attn_module._attn.__name__ if hasattr(attn_module._attn, "__name__") else "hook"
            hooks.append(attn_module.register_forward_hook(hook_fn))
        elif hasattr(attn_module, "forward") and callable(attn_module.forward):
            # Generic attention hook
            hook = attn_module.forward.__name__ if hasattr(attn_module.forward, "__name__") else "hook"
            hooks.append(attn_module.register_forward_hook(hook_fn))
    
    return hooks, attention_patterns

def compute_entropy(attention_weights):
    """Compute entropy of attention patterns."""
    # attention_weights shape: [batch_size, num_heads, seq_len, seq_len]
    # Clamp values to avoid log(0)
    eps = 1e-8
    attention_weights = torch.clamp(attention_weights, eps, 1.0)
    
    # Compute entropy per head: -sum(p * log(p))
    entropy = -torch.sum(attention_weights * torch.log(attention_weights), dim=-1)  # [batch, num_heads, seq_len]
    
    # Average over sequence length and batch
    entropy = entropy.mean(dim=-1).mean(dim=0)  # [num_heads]
    
    return entropy

def gather_head_importance(model, dataloader, attention_modules, strategy="entropy", num_batches=10):
    """Calculate importance of each attention head using the specified strategy."""
    # Prepare to gather head importance
    model.eval()
    num_layers = len(attention_modules)
    num_heads = model.head_count if hasattr(model, "head_count") else 12
    
    # Initialize importance scores
    if strategy == "random":
        # Random importance
        importance = torch.rand(num_layers, num_heads)
        print("Using random importance scores")
        return importance
        
    # For magnitude or entropy, we need to compute them
    importance = torch.zeros(num_layers, num_heads)
    
    if strategy == "magnitude":
        # Compute L2 norm of weight matrices
        print("Computing magnitude-based importance...")
        for layer_idx, attn_module in attention_modules:
            if hasattr(attn_module, "c_attn") and hasattr(attn_module.c_attn, "weight"):
                # GPT-2 style
                weight = attn_module.c_attn.weight
                head_size = weight.shape[1] // (3 * num_heads)
                
                # Compute importance for each head
                for head_idx in range(num_heads):
                    # Get query weight matrix for this head
                    start_idx = head_idx * head_size
                    end_idx = (head_idx + 1) * head_size
                    head_weight = weight[:, start_idx:end_idx]
                    
                    # Compute L2 norm
                    importance[layer_idx, head_idx] = torch.norm(head_weight)
            elif hasattr(attn_module, "q_proj") and hasattr(attn_module.q_proj, "weight"):
                # OPT style
                weight = attn_module.q_proj.weight
                head_size = weight.shape[0] // num_heads
                
                # Compute importance for each head
                for head_idx in range(num_heads):
                    # Get query weight matrix for this head
                    start_idx = head_idx * head_size
                    end_idx = (head_idx + 1) * head_size
                    head_weight = weight[start_idx:end_idx, :]
                    
                    # Compute L2 norm
                    importance[layer_idx, head_idx] = torch.norm(head_weight)
            else:
                # Fallback: random importance for this layer
                importance[layer_idx, :] = torch.rand(num_heads)
    
    elif strategy == "entropy":
        # Register hooks to capture attention patterns
        hooks, attention_patterns = register_attention_hooks(model, attention_modules)
        
        try:
            # Evaluate model on some batches to get attention patterns
            print("Capturing attention patterns for entropy calculation...")
            with torch.no_grad():
                for batch_idx, (input_ids, attention_mask) in enumerate(tqdm(dataloader, desc="Computing entropy")):
                    if batch_idx >= num_batches:
                        break
                        
                    try:
                        # Move to device
                        input_ids = input_ids.to(DEVICE)
                        attention_mask = attention_mask.to(DEVICE)
                        
                        # Forward pass
                        with autocast_if_available():
                            _ = model(input_ids, attention_mask=attention_mask)
                    except Exception as e:
                        print(f"Error computing entropy for batch {batch_idx}: {e}")
                        continue
            
            # Process collected attention patterns
            if attention_patterns:
                for layer_idx, attn_pattern in enumerate(attention_patterns):
                    if layer_idx < num_layers:
                        # Compute entropy for each head
                        entropy = compute_entropy(attn_pattern)
                        importance[layer_idx, :num_heads] = entropy
            else:
                print("No attention patterns collected. Using random importance.")
                importance = torch.rand(num_layers, num_heads)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return importance

def get_strategy(model_type, strategy_name):
    """Get the appropriate head importance strategy."""
    # For Pythia models, entropy computation can be unstable, use magnitude by default
    if model_type == "pythia" and strategy_name == "entropy":
        print("Warning: Entropy strategy may be unstable for Pythia models. Using magnitude instead.")
        return "magnitude"
    return strategy_name'''))

# Add attention pruning markdown cell
notebook.cells.append(new_markdown_cell('''## Attention Pruning

Implement attention gating for pruning less important heads.'''))

# Add attention pruning code cell
notebook.cells.append(new_code_cell('''def add_attention_gating(model, attention_modules):
    """Add attention gates to model by modifying the attention computation."""
    num_layers = len(attention_modules)
    num_heads = model.head_count if hasattr(model, "head_count") else 12
    
    # Create gate parameters
    gates = torch.ones(num_layers, num_heads, requires_grad=True)
    model.attention_gates = torch.nn.Parameter(gates)
    
    # Function to apply gating to attention
    def apply_gating_to_attention(attn_module, layer_idx):
        """Apply gating to an attention module."""
        original_attn = None
        
        # GPT-2 style gating
        if hasattr(attn_module, "_attn") and callable(attn_module._attn):
            original_attn = attn_module._attn
            
            def gated_attention(self, query, key, value, attention_mask=None, head_mask=None):
                """Gated version of the attention function."""
                # Call original attention
                attn_output = original_attn(query, key, value, attention_mask, head_mask)
                
                # Apply gating (attn_output has shape [batch, num_heads, seq_len, head_dim])
                gates = model.attention_gates[layer_idx].view(1, -1, 1, 1)
                gated_output = attn_output * gates
                
                return gated_output
            
            # Replace attention function
            attn_module._attn = types.MethodType(gated_attention, attn_module)
        
        # OPT style gating (apply to bmm operation)
        elif hasattr(attn_module, "forward") and callable(attn_module.forward):
            original_forward = attn_module.forward
            
            def gated_forward(self, *args, **kwargs):
                """Gated version of the forward function."""
                # Call original forward
                output = original_forward(*args, **kwargs)
                
                # Check if we have multiple outputs
                if isinstance(output, tuple):
                    attn_output = output[0]
                    
                    # Reshape gates to match attention output
                    gates = model.attention_gates[layer_idx].view(1, -1, 1, 1)
                    
                    # Apply gating
                    try:
                        # Handle different output formats
                        if attn_output.dim() == 4 and attn_output.shape[1] == num_heads:
                            # Standard shape [batch, num_heads, seq_len, head_dim]
                            gated_output = attn_output * gates
                            return (gated_output,) + output[1:]
                        else:
                            # Unknown format, try not to break things
                            print(f"Warning: Couldn't apply gating to output with shape {attn_output.shape}")
                            return output
                    except Exception as e:
                        print(f"Error applying gates: {e}")
                        return output
                else:
                    # Single output
                    attn_output = output
                    
                    # Reshape gates to match attention output
                    gates = model.attention_gates[layer_idx].view(1, -1, 1, 1)
                    
                    # Apply gating
                    try:
                        if attn_output.dim() == 4 and attn_output.shape[1] == num_heads:
                            gated_output = attn_output * gates
                            return gated_output
                        else:
                            print(f"Warning: Couldn't apply gating to output with shape {attn_output.shape}")
                            return output
                    except Exception as e:
                        print(f"Error applying gates: {e}")
                        return output
            
            # Replace forward function
            attn_module.forward = types.MethodType(gated_forward, attn_module)
        
        return original_attn is not None
    
    # Import types module for MethodType
    import types
    
    # Apply gating to all attention modules
    modified_count = 0
    for layer_idx, attn_module in attention_modules:
        if apply_gating_to_attention(attn_module, layer_idx):
            modified_count += 1
    
    print(f"Added attention gating to {modified_count}/{len(attention_modules)} modules")
    return modified_count > 0

def apply_head_pruning(model, importance, pruning_level, max_display_items=40):
    """Apply pruning to less important heads."""
    # Flatten importance to get global ranking
    flat_importance = importance.view(-1)
    num_heads_total = flat_importance.shape[0]
    
    # Determine heads to prune
    k = int(num_heads_total * pruning_level)
    if k <= 0:
        print("Pruning level too low, no heads will be pruned")
        return []
    
    # Get heads with lowest importance values
    _, indices = torch.topk(flat_importance, k, largest=False)
    heads_to_prune = [(idx // importance.shape[1], idx % importance.shape[1]) for idx in indices]
    
    # Sort by layer then head for better visualization
    heads_to_prune.sort()
    
    # Apply pruning by setting gates to zero
    for layer_idx, head_idx in heads_to_prune:
        model.attention_gates[layer_idx, head_idx] = 0.0
    
    # Display pruned heads
    print(f"Pruned {len(heads_to_prune)} attention heads ({pruning_level*100:.1f}% of {num_heads_total} total heads)")
    
    # Visually show which heads were pruned with limited display
    if len(heads_to_prune) > max_display_items:
        print(f"Showing a subset of pruned heads (displaying {max_display_items} out of {len(heads_to_prune)} heads)")
        # Always show some heads from the beginning, middle, and end
        display_heads = heads_to_prune[:max_display_items//3] + heads_to_prune[len(heads_to_prune)//2-max_display_items//6:len(heads_to_prune)//2+max_display_items//6] + heads_to_prune[-max_display_items//3:]
    else:
        display_heads = heads_to_prune
    
    # Show pruned heads in a grid
    num_layers = importance.shape[0]
    num_heads = importance.shape[1]
    grid = []
    
    for layer_idx in range(num_layers):
        row = []
        for head_idx in range(num_heads):
            if (layer_idx, head_idx) in heads_to_prune:
                if (layer_idx, head_idx) in display_heads:
                    row.append("🔴")  # Red circle for pruned heads in display set
                else:
                    row.append("•")   # Small dot for pruned heads not in display set
            else:
                row.append("⚪")      # White circle for kept heads
        grid.append("".join(row))
    
    # Print the grid with layer numbers
    for layer_idx, row in enumerate(grid):
        print(f"Layer {layer_idx:2d}: {row}")
    
    return heads_to_prune

def visualize_head_importance(importance, pruned_heads=None, max_display_items=40):
    """Visualize the importance of attention heads."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get dimensions
    num_layers, num_heads = importance.shape
    
    # Convert to numpy
    importance_np = importance.cpu().numpy()
    
    # Create a heatmap
    im = ax.imshow(importance_np, cmap="viridis")
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label="Importance")
    
    # Add labels
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("Attention Head Importance")
    
    # Set ticks
    if num_heads <= 20:
        ax.set_xticks(np.arange(num_heads))
        ax.set_xticklabels([str(i) for i in range(num_heads)])
    else:
        # Show fewer ticks for readability
        ax.set_xticks(np.arange(0, num_heads, 2))
        ax.set_xticklabels([str(i) for i in range(0, num_heads, 2)])
    
    if num_layers <= 12:
        ax.set_yticks(np.arange(num_layers))
        ax.set_yticklabels([str(i) for i in range(num_layers)])
    else:
        # Show fewer ticks for readability
        ax.set_yticks(np.arange(0, num_layers, 2))
        ax.set_yticklabels([str(i) for i in range(0, num_layers, 2)])
    
    # Rotate x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Mark pruned heads if provided
    if pruned_heads:
        # If we have a lot of pruned heads, only plot a subset
        if len(pruned_heads) > max_display_items:
            # Prioritize variety - sample across layers
            subset_indices = np.linspace(0, len(pruned_heads)-1, max_display_items).astype(int)
            display_heads = [pruned_heads[i] for i in subset_indices]
            print(f"Showing {max_display_items} out of {len(pruned_heads)} pruned heads in the visualization")
        else:
            display_heads = pruned_heads
        
        # Plot pruned heads as red squares
        for layer_idx, head_idx in display_heads:
            rect = plt.Rectangle((head_idx - 0.5, layer_idx - 0.5), 1, 1, fill=False, 
                                 edgecolor='red', linewidth=2)
            ax.add_patch(rect)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    return fig'''))

# Add fine-tuning markdown cell
notebook.cells.append(new_markdown_cell('''## Fine-tuning Implementation

Fine-tune the pruned model to recover performance.'''))

# Add fine-tuning code cell
notebook.cells.append(new_code_cell('''def fine_tune_model(model, train_dataloader, val_dataloader, optimizer, scheduler, metrics, num_epochs=3):
    """Fine-tune the model and track metrics."""
    print(f"Starting fine-tuning for {num_epochs} epochs")
    
    step = 0
    total_steps = len(train_dataloader) * num_epochs
    evaluation_freq = max(1, len(train_dataloader) // 5)  # Evaluate 5 times per epoch
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        
        for batch_idx, (input_ids, attention_mask) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            try:
                # Move data to device
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                
                # Create labels by shifting input_ids right
                labels = input_ids.clone()
                
                # Clear previous gradients
                optimizer.zero_grad()
                
                # Forward pass
                with autocast_if_available():
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                optimizer.step()
                
                # Update learning rate
                scheduler.step()
                
                # Evaluate periodically
                if batch_idx % evaluation_freq == 0 or batch_idx == len(train_dataloader) - 1:
                    # Generate text sample periodically
                    if batch_idx % (evaluation_freq * 2) == 0:
                        prompt = "The quick brown fox"
                        sample = generate_text(model, tokenizer, prompt)
                    else:
                        sample = None
                    
                    # Evaluate model
                    val_loss, val_ppl = evaluate_model(model, val_dataloader)
                    print(f"Step {step+1}/{total_steps} | Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
                    
                    # Update metrics
                    metrics.update(step, val_loss, val_ppl, generation_sample=sample)
                    
                    # Save checkpoint
                    if (epoch == num_epochs - 1) and (batch_idx == len(train_dataloader) - 1):
                        checkpoint_path = os.path.join(OUTPUT_DIR, "pruned_finetuned_model.pt")
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'step': step,
                            'loss': loss.item(),
                            'val_loss': val_loss,
                            'val_ppl': val_ppl
                        }, checkpoint_path)
                        print(f"Saved checkpoint to {checkpoint_path}")
                
                # Increment step
                step += 1
                
            except Exception as e:
                if DEVICE == "cuda" and "CUDA" in str(e):
                    print(f"CUDA error during training at batch {batch_idx}, epoch {epoch+1}: {e}")
                    print("Attempting to continue training on CPU...")
                    
                    # Clear GPU memory
                    clear_gpu_memory()
                    
                    # Try again on CPU
                    try:
                        # Move to CPU
                        model = model.cpu()
                        input_ids = input_ids.cpu()
                        attention_mask = attention_mask.cpu()
                        labels = labels.cpu()
                        
                        # Forward pass
                        optimizer.zero_grad()
                        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                        cpu_loss = outputs.loss
                        
                        # Backward pass
                        cpu_loss.backward()
                        optimizer.step()
                        scheduler.step()
                        
                        # Evaluate
                        val_loss, val_ppl = evaluate_model(model, val_dataloader)
                        print(f"CPU Step {step+1}/{total_steps} | Loss: {cpu_loss.item():.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
                        
                        # Update metrics
                        metrics.update(step, val_loss, val_ppl)
                        
                        # Move back to GPU if possible
                        if torch.cuda.is_available():
                            model = model.to(DEVICE)
                        
                        step += 1
                    except Exception as e2:
                        print(f"Training also failed on CPU: {e2}")
                else:
                    print(f"Error during training at batch {batch_idx}, epoch {epoch+1}: {e}")
                
                # Skip to next batch
                continue
    
    # Final evaluation
    final_loss, final_ppl = evaluate_model(model, val_dataloader)
    print(f"Final evaluation - Loss: {final_loss:.4f}, Perplexity: {final_ppl:.2f}")
    
    return final_loss, final_ppl'''))

# Add experiment runner markdown cell
notebook.cells.append(new_markdown_cell('''## Run the Experiment

Execute the full pruning and fine-tuning pipeline.'''))

# Add experiment runner code cell
notebook.cells.append(new_code_cell('''def run_experiment(model_name="gpt2", 
                   pruning_strategy="entropy", 
                   pruning_level=0.3, 
                   fine_tuning_epochs=3, 
                   learning_rate=5e-5,
                   batch_size=4,
                   prompt="The quick brown fox jumps over the lazy dog. In recent years,"):
    """Run the complete pruning and fine-tuning experiment."""
    # Step 1: Initialize and setup
    print(f"=== Running Pruning and Fine-tuning Experiment ===")
    print(f"Model: {model_name}")
    print(f"Pruning strategy: {pruning_strategy}")
    print(f"Pruning level: {pruning_level}")
    print(f"Fine-tuning epochs: {fine_tuning_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {DEVICE}")
    
    # Initialize metrics tracker
    metrics = ProgressMetrics()
    
    # Step 2: Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, cache_dir=MODEL_CACHE_DIR)
    
    # Step 3: Load data
    train_dataloader, val_dataloader = load_wikitext_data(tokenizer, batch_size=batch_size)
    if train_dataloader is None or val_dataloader is None:
        print("Failed to load data. Aborting experiment.")
        return None
    
    # Step 4: Evaluate initial performance
    print("\nEvaluating initial model performance...")
    initial_loss, initial_ppl = evaluate_model(model, val_dataloader)
    print(f"Initial performance - Loss: {initial_loss:.4f}, Perplexity: {initial_ppl:.2f}")
    
    # Track initial metrics
    metrics.update(0, initial_loss, initial_ppl)
    
    # Step 5: Generate initial text sample
    print("\nGenerating initial text sample...")
    initial_generation = generate_text(model, tokenizer, prompt)
    print(f"Initial generation:\\n{initial_generation}")
    
    # Step 6: Extract attention modules
    attention_modules = get_attention_modules(model)
    if not attention_modules:
        print("Failed to extract attention modules. Aborting experiment.")
        return None
    
    # Step 7: Add attention gating
    success = add_attention_gating(model, attention_modules)
    if not success:
        print("Failed to add attention gating. Aborting experiment.")
        return None
    
    # Step 8: Calculate head importance
    print("\nCalculating head importance...")
    strategy = get_strategy(model.model_type, pruning_strategy)
    importance = gather_head_importance(model, val_dataloader, attention_modules, strategy=strategy)
    
    # Step 9: Apply pruning
    print("\nApplying pruning...")
    pruned_heads = apply_head_pruning(model, importance, pruning_level)
    
    # Update metrics with pruning info
    metrics.set_pruning_info(strategy, pruning_level, pruned_heads)
    
    # Visualize head importance
    print("\nVisualizing head importance...")
    fig = visualize_head_importance(importance, pruned_heads)
    
    # Step 10: Evaluate pruned model
    print("\nEvaluating pruned model performance...")
    pruned_loss, pruned_ppl = evaluate_model(model, val_dataloader)
    print(f"After pruning: loss: {pruned_loss:.4f}, perplexity: {pruned_ppl:.2f}")
    
    # Step 11: Generate example text with pruned model
    pruned_generation = generate_text(model, tokenizer, prompt)
    print(f"Generation after pruning:\\n{pruned_generation}")
    
    # Update metrics with pruned model performance
    metrics.update(1, pruned_loss, pruned_ppl, 
                  head_info=importance.cpu().numpy().tolist(), 
                  generation_sample=pruned_generation)
    
    # Step 12: Set up optimizer and scheduler for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Create scheduler with warmup
    num_training_steps = len(train_dataloader) * fine_tuning_epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )
    
    # Step 13: Fine-tune the pruned model
    print("\nFine-tuning pruned model...")
    final_loss, final_ppl = fine_tune_model(
        model, 
        train_dataloader, 
        val_dataloader, 
        optimizer, 
        scheduler, 
        metrics, 
        num_epochs=fine_tuning_epochs
    )
    
    # Step 14: Generate final text sample
    final_generation = generate_text(model, tokenizer, prompt)
    print(f"Final generation after fine-tuning:\\n{final_generation}")
    
    # Step 15: Save final metrics and plots
    metrics_path = os.path.join(OUTPUT_DIR, "pruning_finetuning_metrics.json")
    metrics.save_metrics(metrics_path)
    
    plots_path = os.path.join(OUTPUT_DIR, "pruning_finetuning_plots.png")
    metrics.save_plots(plots_path)
    
    # Step 16: Print summary
    summary = metrics.get_summary()
    print("\n=== Experiment Summary ===")
    print(f"Model: {model_name}")
    print(f"Pruning strategy: {summary.get('strategy', strategy)}")
    print(f"Pruning level: {summary.get('pruning_level', pruning_level)}")
    print(f"Pruned heads: {summary.get('pruned_heads_count', len(pruned_heads))}")
    print(f"Initial perplexity: {summary.get('initial_perplexity', initial_ppl):.2f}")
    print(f"After pruning perplexity: {pruned_ppl:.2f}")
    print(f"Final perplexity: {summary.get('final_perplexity', final_ppl):.2f}")
    print(f"Improvement: {summary.get('improvement_percent', ((initial_ppl - final_ppl) / initial_ppl) * 100):.2f}%")
    
    # If in Colab, offer to download results
    if IS_COLAB:
        print("\nDownloading result files...")
        try:
            download_files([metrics_path, plots_path])
        except Exception as e:
            print(f"Error downloading files: {e}")
    
    return metrics'''))

# Add user interface markdown cell
notebook.cells.append(new_markdown_cell('''## User Interface

Run the experiment with customizable parameters.'''))

# Add user interface code cell
notebook.cells.append(new_code_cell('''# Run the experiment with the specified parameters
# You can customize these parameters
MODEL_NAME = "distilgpt2"  # Smaller GPT-2 model for faster demonstration
PRUNING_STRATEGY = "entropy"  # Options: "random", "magnitude", "entropy"
PRUNING_LEVEL = 0.3  # Percentage of heads to prune (0.0 to 1.0)
FINE_TUNING_EPOCHS = 3  # Number of epochs for fine-tuning
LEARNING_RATE = 5e-5  # Learning rate for fine-tuning
BATCH_SIZE = 4  # Batch size for training and evaluation
PROMPT = "The quick brown fox jumps over the lazy dog. In recent years,"  # Prompt for text generation

# Run the experiment
experiment_metrics = run_experiment(
    model_name=MODEL_NAME,
    pruning_strategy=PRUNING_STRATEGY,
    pruning_level=PRUNING_LEVEL,
    fine_tuning_epochs=FINE_TUNING_EPOCHS,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    prompt=PROMPT
)'''))

# Set notebook metadata
notebook.metadata = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'name': 'python',
        'codemirror_mode': {
            'name': 'ipython',
            'version': 3
        }
    },
    'colab': {
        'name': 'PruningAndFineTuningColab',
        'provenance': [],
        'collapsed_sections': []
    }
}

# Output filename
output_file = '/Users/joel/Development/sentinel-ai/colab_notebooks/PruningAndFineTuningColab.ipynb'

if __name__ == "__main__":
    # Save the notebook
    with open(output_file, 'w') as f:
        nbformat.write(notebook, f)
    print(f'Created properly structured notebook at {output_file}')