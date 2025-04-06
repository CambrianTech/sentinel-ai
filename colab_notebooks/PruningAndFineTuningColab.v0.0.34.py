#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pruning and Fine-Tuning Colab (v0.0.34)

This script demonstrates making a GPT-2 model smaller and more powerful by:
1. Applying pruning to remove less important attention heads
2. Fine-tuning the pruned model to recover performance
3. Showing clear metrics of improvement

It's designed to be run in Google Colab using real-world data (not tiny Shakespeare).

Version History:
- v0.0.34 (April 2025): Fixed missing variable error, visualization issues and enhanced CUDA error handling
- v0.0.33 (April 2025): Fixed visualization issues, improved model compatibility and enhanced error handling
- v0.0.32 (April 2025): Added CUDA error handling for Colab compatibility and memory management
- v0.0.31 (April 2025): Fixed get_strategy parameters issue and improved Colab compatibility 
- v0.0.30 (April 2025): Added OPT model support and chart improvements
"""

import os
import sys
import argparse
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

# Global variables
OUTPUT_DIR = "pruning_results"
MODEL_CACHE_DIR = "model_cache"
DATA_DIR = "data"

class ProgressMetrics:
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
        }

def setup_directories():
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
                        f.write(item["text"] + "\n")
                        
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
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    
    # Ensure we have at least 100 paragraphs of reasonable length
    paragraphs = [p for p in paragraphs if len(p) > 100]
    
    if len(paragraphs) < 100:
        # Fall back to splitting by newline if needed
        paragraphs = [p for p in text.split("\n") if len(p.strip()) > 100]
    
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
    
    return train_data, val_data

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

def load_model_and_tokenizer(model_name, cache_dir=None):
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
    
    return model, tokenizer

def get_attention_modules(model):
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
    
    return attention_modules

def get_head_importances(model, val_dataloader, strategy="entropy"):
    """
    Calculate importance scores for each attention head.
    
    Args:
        model: The model to analyze
        val_dataloader: Validation data for computing metrics
        strategy: Pruning strategy ('entropy', 'magnitude', 'random')
        
    Returns:
        List of (layer_idx, head_idx, importance) tuples
    """
    print(f"Calculating head importances using {strategy} strategy...")
    attention_modules = get_attention_modules(model)
    head_importances = []
    
    # Set default model type if not already set
    if not hasattr(model, "model_type"):
        model.model_type = "gpt2"
    
    if strategy == "random":
        # For random strategy, just assign random importances
        for layer_idx, attn in attention_modules:
            # Get number of heads based on model type
            if hasattr(attn, "num_heads"):
                num_heads = attn.num_heads
            elif hasattr(attn, "num_attention_heads"):
                num_heads = attn.num_attention_heads
            else:
                # Try to infer from model name
                if model.model_type == "gpt2":
                    num_heads = 12  # Default for GPT-2
                elif model.model_type == "opt":
                    num_heads = 12  # Default for smaller OPT
                elif model.model_type == "pythia":
                    num_heads = 12  # Default for smaller Pythia
                else:
                    num_heads = 12  # fallback
                print(f"Warning: Could not determine num_heads, using default: {num_heads}")
                    
            for head_idx in range(num_heads):
                importance = np.random.random()
                head_importances.append((layer_idx, head_idx, importance))
    
    elif strategy == "magnitude":
        # For magnitude strategy, use the L2 norm of the head weights
        for layer_idx, attn in attention_modules:
            # Determine number of heads
            if hasattr(attn, "num_heads"):
                num_heads = attn.num_heads
            elif hasattr(attn, "num_attention_heads"):
                num_heads = attn.num_attention_heads
            else:
                # Model-specific defaults
                if model.model_type == "gpt2":
                    num_heads = 12
                elif model.model_type == "opt":
                    num_heads = 12
                elif model.model_type == "pythia":
                    num_heads = 12
                else:
                    num_heads = 12
                print(f"Warning: Could not determine num_heads, using default: {num_heads}")
            
            # Get the appropriate projection weights based on model type
            if model.model_type == "gpt2":
                if hasattr(attn, "c_attn") and hasattr(attn, "head_size"):
                    q_weight = attn.c_attn.weight
                    head_size = attn.head_size
                else:
                    print(f"Warning: Layer {layer_idx} doesn't have expected attributes")
                    continue
            elif model.model_type == "opt":
                if hasattr(attn, "out_proj") and hasattr(attn, "out_proj"):
                    q_weight = attn.q_proj.weight
                    head_size = q_weight.shape[0] // num_heads
                else:
                    print(f"Warning: Layer {layer_idx} doesn't have expected attributes")
                    continue
            elif model.model_type == "pythia":
                if hasattr(attn, "c_attn") and hasattr(attn, "head_size"):
                    q_weight = attn.c_attn.weight
                    head_size = attn.head_size  
                else:
                    print(f"Warning: Layer {layer_idx} doesn't have expected attributes")
                    continue
            else:
                # Default to GPT-2 pattern
                if hasattr(attn, "c_attn") and hasattr(attn, "head_size"):
                    q_weight = attn.c_attn.weight
                    head_size = attn.head_size
                else:
                    print(f"Warning: Layer {layer_idx} doesn't have expected attributes")
                    continue
                
            # Compute importance for each head
            for head_idx in range(num_heads):
                try:
                    # Get weights for this head
                    start_idx = head_idx * head_size
                    end_idx = (head_idx + 1) * head_size
                    
                    # Extract weights for Q, K, V for this head - GPT2-specific
                    if model.model_type == "gpt2" or model.model_type == "pythia":
                        q_head = q_weight[:, start_idx:end_idx]
                        k_head = q_weight[:, num_heads*head_size + start_idx:num_heads*head_size + end_idx]
                        v_head = q_weight[:, 2*num_heads*head_size + start_idx:2*num_heads*head_size + end_idx]
                    elif model.model_type == "opt":
                        # For OPT, we need to get separate Q, K, V projections
                        q_head = attn.q_proj.weight[start_idx:end_idx, :]
                        k_head = attn.k_proj.weight[start_idx:end_idx, :]
                        v_head = attn.v_proj.weight[start_idx:end_idx, :]
                    else:
                        # Fallback to GPT2 pattern
                        q_head = q_weight[:, start_idx:end_idx]
                        k_head = q_weight[:, num_heads*head_size + start_idx:num_heads*head_size + end_idx]
                        v_head = q_weight[:, 2*num_heads*head_size + start_idx:2*num_heads*head_size + end_idx]
                    
                    # Compute L2 norm (magnitude)
                    q_norm = torch.norm(q_head).item()
                    k_norm = torch.norm(k_head).item()
                    v_norm = torch.norm(v_head).item()
                    
                    # Use average of Q, K, V norms as importance
                    importance = (q_norm + k_norm + v_norm) / 3
                    head_importances.append((layer_idx, head_idx, importance))
                except Exception as e:
                    print(f"Error processing head {head_idx} in layer {layer_idx}: {e}")
                    # Assign random importance as fallback
                    importance = np.random.random()
                    head_importances.append((layer_idx, head_idx, importance))
    
    elif strategy == "entropy":
        # For entropy strategy, measure attention entropy on validation data
        model.eval()
        
        # Store attention outputs
        attention_outputs = {}
        
        # Register hooks to capture attention
        handles = []
        
        def get_attention_hook(layer_idx):
            def hook(module, input, output):
                # Shape is usually [batch, num_heads, seq_len, seq_len]
                # But format can differ by model type
                if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], torch.Tensor):
                    attention_outputs[layer_idx] = output[1].detach()
                elif isinstance(output, torch.Tensor):
                    # Some models directly return attention weights
                    attention_outputs[layer_idx] = output.detach()
            return hook
        
        # Register hooks for each attention module
        for layer_idx, attn in attention_modules:
            handles.append(attn.register_forward_hook(get_attention_hook(layer_idx)))
        
        # Run a few batches to collect attention patterns
        with torch.no_grad():
            for batch_idx, (input_ids, attention_mask) in enumerate(val_dataloader):
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                
                # Forward pass to trigger hooks
                try:
                    model(input_ids=input_ids, attention_mask=attention_mask)
                except Exception as e:
                    print(f"Error during forward pass: {e}")
                    continue
                
                if batch_idx >= 5:  # Collect data from 5 batches
                    break
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        if not attention_outputs:
            print("Warning: No attention outputs captured. Falling back to magnitude strategy.")
            return get_head_importances(model, val_dataloader, strategy="magnitude")
        
        # Calculate entropy for each head
        for layer_idx, attn in attention_modules:
            if layer_idx not in attention_outputs:
                continue
                
            attn_outputs = attention_outputs[layer_idx]
            
            # Determine number of heads
            if hasattr(attn, "num_heads"):
                num_heads = attn.num_heads
            elif hasattr(attn, "num_attention_heads"):
                num_heads = attn.num_attention_heads
            else:
                # Try to infer from the output shape
                if len(attn_outputs.shape) >= 2:
                    num_heads = attn_outputs.shape[1]
                else:
                    # Model-specific defaults
                    if model.model_type == "gpt2":
                        num_heads = 12
                    elif model.model_type == "opt":
                        num_heads = 12
                    elif model.model_type == "pythia":
                        num_heads = 12
                    else:
                        num_heads = 12
                    print(f"Warning: Could not determine num_heads, using default: {num_heads}")
                
            for head_idx in range(num_heads):
                try:
                    # Extract attention weights for this head
                    if head_idx < attn_outputs.shape[1]:  # Check if index is valid
                        head_attn = attn_outputs[:, head_idx, :, :]
                    else:
                        print(f"Warning: Head index {head_idx} out of bounds. Skipping.")
                        continue
                    
                    # Calculate entropy (we want low entropy = focused attention)
                    entropy = 0
                    
                    # Process each item in the batch
                    for item_idx in range(head_attn.size(0)):
                        item_attn = head_attn[item_idx]
                        
                        # Calculate entropy along the attention dimension
                        # Add small epsilon to avoid log(0)
                        eps = 1e-10
                        item_entropy = -torch.sum(item_attn * torch.log(item_attn + eps), dim=-1)
                        entropy += torch.mean(item_entropy).item()
                    
                    # Average entropy across batch
                    entropy /= head_attn.size(0)
                    
                    # Negated entropy, so that higher values = more important (focused attention)
                    importance = -entropy
                    head_importances.append((layer_idx, head_idx, importance))
                    
                except Exception as e:
                    print(f"Error calculating entropy for head {head_idx} in layer {layer_idx}: {e}")
                    # Fall back to random importance
                    importance = np.random.random()
                    head_importances.append((layer_idx, head_idx, importance))
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # If no head importances were calculated, fall back to random
    if not head_importances:
        print("Warning: No head importances calculated. Falling back to random strategy.")
        return get_head_importances(model, val_dataloader, strategy="random")
    
    # Sort by importance (ascending order, so lowest importance first)
    head_importances.sort(key=lambda x: x[2])
    
    return head_importances

def prune_heads(model, head_importances, pruning_level=0.3):
    """
    Prune specified fraction of attention heads.
    
    Args:
        model: The model to prune
        head_importances: List of (layer_idx, head_idx, importance) tuples
        pruning_level: Fraction of heads to prune (0.0 to 1.0)
        
    Returns:
        List of pruned heads as (layer_idx, head_idx) tuples
    """
    attention_modules = get_attention_modules(model)
    
    # Count total heads
    total_heads = sum(attn.num_heads for _, attn in attention_modules)
    
    # Calculate how many heads to prune
    num_to_prune = int(total_heads * pruning_level)
    
    # Get heads to prune (lowest importance first)
    heads_to_prune = [(layer_idx, head_idx) for layer_idx, head_idx, _ in head_importances[:num_to_prune]]
    
    print(f"Pruning {len(heads_to_prune)}/{total_heads} attention heads ({pruning_level:.1%})")
    
    # Create/initialize gates if they don't exist
    for layer_idx, attn in attention_modules:
        num_heads = attn.num_heads
        
        # Check if gate exists
        if not hasattr(attn, "head_gates"):
            # Create gates with default value 1.0
            attn.head_gates = torch.ones(num_heads, device=DEVICE)
    
    # Apply pruning by setting gates to 0
    for layer_idx, head_idx in heads_to_prune:
        for i, (module_layer_idx, attn) in enumerate(attention_modules):
            if module_layer_idx == layer_idx:
                attn.head_gates[head_idx] = 0.0
                break
    
    # Modify forward pass to use gates
    for layer_idx, attn in attention_modules:
        # Save original method if not already saved
        if not hasattr(attn, "original_forward"):
            attn.original_forward = attn.forward
            
            # Create gated forward method
            def make_gated_forward(original_forward, head_gates):
                def gated_forward(self, *args, **kwargs):
                    # Call original forward
                    outputs = original_forward(*args, **kwargs)
                    
                    # Apply gates to attention outputs
                    # outputs[0] = result, outputs[1] = attention weights
                    if len(outputs) > 1 and isinstance(outputs[1], torch.Tensor):
                        # outputs[1] shape: [batch_size, num_heads, seq_len, seq_len]
                        gates = head_gates.view(1, -1, 1, 1)
                        gated_attention = outputs[1] * gates
                        
                        return (outputs[0], gated_attention) + outputs[2:] if len(outputs) > 2 else (outputs[0], gated_attention)
                    
                    return outputs
                
                return gated_forward
            
            # Set new forward method
            attn.forward = make_gated_forward(attn.original_forward, attn.head_gates).__get__(attn, type(attn))
    
    return heads_to_prune

def fine_tune_model(model, train_dataloader, val_dataloader, tokenizer, 
                   learning_rate=5e-5, num_epochs=3, progress_tracker=None):
    """
    Fine-tune model after pruning.
    
    Args:
        model: The model to fine-tune
        train_dataloader: Training data
        val_dataloader: Validation data
        tokenizer: Tokenizer
        learning_rate: Learning rate for optimization
        num_epochs: Number of training epochs
        progress_tracker: ProgressMetrics object for tracking progress
        
    Returns:
        Dictionary with training results
    """
    model.train()
    
    # Prepare optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Calculate total steps and prepare scheduler
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Import memory monitoring if available
    try:
        import psutil
        has_psutil = True
    except ImportError:
        has_psutil = False
    
    # Train the model
    step = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training loop
        model.train()
        epoch_losses = []
        
        # Check available memory before starting epoch
        if has_psutil and DEVICE == "cuda" and torch.cuda.is_available():
            try:
                # Get available system memory
                available_memory_gb = psutil.virtual_memory().available / (1024**3)
                # Get GPU memory info
                gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                
                print(f"Memory before epoch - System: {available_memory_gb:.2f}GB free, "
                      f"GPU: {gpu_memory_allocated:.2f}GB allocated, {gpu_memory_reserved:.2f}GB reserved")
                
                # If memory is low, try to clear it
                if available_memory_gb < 2.0 or gpu_memory_allocated > 3.0:
                    print("Low memory detected - clearing caches")
                    gc.collect()
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Memory check error: {e}")
        
        # Create progress bar
        progress_bar = tqdm(train_dataloader, desc=f"Training epoch {epoch+1}")
        
        for batch_idx, (input_ids, attention_mask) in enumerate(progress_bar):
            try:
                # Prepare data
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                # Track metrics
                loss_val = loss.item()
                epoch_losses.append(loss_val)
                perplexity = torch.exp(torch.tensor(loss_val)).item()
                
                progress_bar.set_postfix(loss=f"{loss_val:.4f}", ppl=f"{perplexity:.2f}")
                
                # Generate sample text every 50 steps with error handling
                if step % 50 == 0:
                    try:
                        sample_text = generate_text(model, tokenizer, prompt="A large language model is")
                        
                        if progress_tracker:
                            # Get gate values for visualization
                            gate_values = get_gate_values(model)
                            
                            progress_tracker.update(
                                step=step,
                                loss=loss_val,
                                perplexity=perplexity,
                                gate_values=gate_values,
                                generation_sample=sample_text
                            )
                    except Exception as gen_error:
                        print(f"Error during sample generation: {gen_error}")
                        # Still update metrics without the generation sample
                        if progress_tracker:
                            progress_tracker.update(
                                step=step,
                                loss=loss_val,
                                perplexity=perplexity
                            )
                
                step += 1
                
                # Clear memory periodically
                if step % 100 == 0 and DEVICE == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    print(f"\nCUDA error during training: {e}")
                    
                    # Try to recover by clearing memory
                    if DEVICE == "cuda":
                        print("Attempting to clear GPU memory...")
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    # Skip this batch and continue
                    print("Skipping batch and continuing...")
                    continue
                else:
                    # For non-CUDA errors, print and continue
                    print(f"Error during training: {e}")
                    continue
            except Exception as gen_error:
                print(f"Unexpected error: {gen_error}")
                continue
        
        # Evaluate after each epoch with error handling
        try:
            eval_loss, eval_ppl = evaluate_model(model, val_dataloader)
            
            # Print epoch summary
            if epoch_losses:  # Make sure we have some losses to average
                epoch_loss = sum(epoch_losses) / len(epoch_losses)
                epoch_ppl = torch.exp(torch.tensor(epoch_loss)).item()
                
                print(f"Epoch {epoch+1} summary:")
                print(f"  Train loss: {epoch_loss:.4f}, perplexity: {epoch_ppl:.2f}")
                print(f"  Val loss: {eval_loss:.4f}, perplexity: {eval_ppl:.2f}")
            else:
                print(f"Epoch {epoch+1} summary:")
                print(f"  No valid training steps completed")
                print(f"  Val loss: {eval_loss:.4f}, perplexity: {eval_ppl:.2f}")
            
            # Track validation metrics
            if progress_tracker:
                try:
                    sample_text = generate_text(
                        model, tokenizer, 
                        prompt="In recent years, artificial intelligence has"
                    )
                    progress_tracker.update(
                        step=step,
                        loss=eval_loss,
                        perplexity=eval_ppl,
                        generation_sample=sample_text
                    )
                except Exception as e:
                    print(f"Error during evaluation text generation: {e}")
                    # Still update metrics
                    progress_tracker.update(
                        step=step,
                        loss=eval_loss,
                        perplexity=eval_ppl
                    )
        except Exception as eval_error:
            print(f"Error during evaluation: {eval_error}")
            # Try to continue to the next epoch
    
    # Final evaluation with error handling
    try:
        final_loss, final_ppl = evaluate_model(model, val_dataloader)
        print(f"Final evaluation - Loss: {final_loss:.4f}, Perplexity: {final_ppl:.2f}")
    except Exception as e:
        print(f"Error during final evaluation: {e}")
        # Use the last known evaluation metrics
        if 'eval_loss' in locals() and 'eval_ppl' in locals():
            final_loss, final_ppl = eval_loss, eval_ppl
        else:
            # If we have no metrics at all, use placeholder values
            final_loss, final_ppl = 999.0, 999.0
        print(f"Using fallback metrics - Loss: {final_loss:.4f}, Perplexity: {final_ppl:.2f}")
    
    return {
        "final_loss": final_loss,
        "final_perplexity": final_ppl,
        "steps": step
    }

def evaluate_model(model, dataloader):
    """Evaluate model on dataloader and return loss and perplexity."""
    model.eval()
    total_loss = 0
    total_batches = 0
    
    with torch.no_grad():
        for input_ids, attention_mask in dataloader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                total_batches += 1
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"CUDA error during evaluation, attempting fallback: {e}")
                    try:
                        # Try with CPU
                        cpu_model = model.cpu()
                        cpu_input_ids = input_ids.cpu()
                        cpu_attention_mask = attention_mask.cpu()
                        
                        outputs = cpu_model(
                            input_ids=cpu_input_ids,
                            attention_mask=cpu_attention_mask,
                            labels=cpu_input_ids
                        )
                        
                        loss = outputs.loss
                        total_loss += loss.item()
                        total_batches += 1
                        
                        # Move model back
                        model.to(DEVICE)
                    except Exception as cpu_error:
                        print(f"CPU fallback failed: {cpu_error}")
                        # Skip this batch
                        continue
                else:
                    print(f"Error during evaluation: {e}")
                    # Skip this batch
                    continue
    
    # Calculate average loss and perplexity
    if total_batches == 0:
        return 999.0, 999.0  # Return high values if all batches failed
        
    avg_loss = total_loss / total_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity

def get_gate_values(model):
    """Extract gate values from model for visualization."""
    attention_modules = get_attention_modules(model)
    
    gate_values = {}
    for layer_idx, attn in attention_modules:
        if hasattr(attn, "head_gates"):
            gate_values[f"layer_{layer_idx}"] = attn.head_gates.detach().cpu().numpy()
    
    return gate_values

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.7):
    """Generate text from a prompt using the model."""
    model.eval()
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    
    try:
        with torch.no_grad():
            # First try with all options enabled
            output = model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode the output
        return tokenizer.decode(output[0], skip_special_tokens=True)
    
    except RuntimeError as e:
        if "CUDA" in str(e):
            print("CUDA error during generation, attempting fallback to CPU...")
            # Try moving to CPU for generation
            try:
                cpu_model = model.cpu()
                cpu_input_ids = input_ids.cpu()
                
                with torch.no_grad():
                    output = cpu_model.generate(
                        cpu_input_ids,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        num_return_sequences=1
                    )
                
                # Move model back to original device
                model.to(DEVICE)
                
                # Decode the output
                return tokenizer.decode(output[0], skip_special_tokens=True)
            
            except Exception as cpu_error:
                print(f"CPU fallback also failed: {cpu_error}")
                # Try with safer parameters
                model.to(DEVICE)  # Ensure model is back on the original device
                try:
                    with torch.no_grad():
                        # Try with simpler generation parameters
                        output = model.generate(
                            input_ids,
                            max_length=max_length,
                            do_sample=False,  # Use greedy decoding
                            pad_token_id=tokenizer.eos_token_id,
                            num_return_sequences=1
                        )
                    
                    return tokenizer.decode(output[0], skip_special_tokens=True)
                except Exception as safe_error:
                    print(f"Safe generation also failed: {safe_error}")
                    return f"{prompt} [Error: Text generation failed]"
        
        # For other types of errors, try with safer parameters
        try:
            print(f"Error during generation: {e}, trying with safer parameters...")
            with torch.no_grad():
                # Try with simpler generation parameters
                output = model.generate(
                    input_ids,
                    max_length=max_length,
                    do_sample=False,  # Use greedy decoding
                    pad_token_id=tokenizer.eos_token_id
                )
            
            return tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as safe_error:
            print(f"Safe generation also failed: {safe_error}")
            return f"{prompt} [Error: Text generation failed]"

def create_head_importance_visualization(head_importances, pruned_heads, output_path=None):
    """Create visualization of head importances and pruned heads."""
    # Organize by layer
    layers = {}
    for layer_idx, head_idx, importance in head_importances:
        if layer_idx not in layers:
            layers[layer_idx] = []
        layers[layer_idx].append((head_idx, importance))
    
    # Convert pruned_heads to a set for faster lookup
    pruned_set = set((layer, head) for layer, head in pruned_heads)
    
    # Create figure - adjust size based on layer count
    num_layers = len(layers)
    max_heads_per_layer = max([len(heads) for heads in layers.values()]) if layers else 0
    
    # Calculate appropriate figure height (minimum 6, scales with layer count)
    fig_height = max(6, num_layers * 0.4)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    # Prepare data for plotting - LIMIT displayed items if too many
    layer_labels = []
    head_importance_data = []
    colors = []
    
    # Determine if we need to limit labels (if too many layers/heads)
    too_many_layers = num_layers > 30
    
    # Track which layers we're showing
    included_layers = set()
    
    # Process data for visualization, prioritizing pruned heads and limiting total items
    max_labels = 40  # Maximum number of labels to show
    
    # First add all pruned heads
    for layer_idx in sorted(layers.keys()):
        heads = layers[layer_idx]
        for head_idx, importance in sorted(heads, key=lambda x: x[0]):
            if (layer_idx, head_idx) in pruned_set:
                layer_labels.append(f"L{layer_idx}-H{head_idx}")
                head_importance_data.append(importance)
                colors.append('red')  # Pruned heads are red
                included_layers.add(layer_idx)
    
    # Then add remaining heads until we hit the limit, sampling across layers
    remaining_slots = max_labels - len(layer_labels)
    if remaining_slots > 0:
        # Get layers that have at least one non-pruned head
        remaining_layers = sorted(set(layer_idx for layer_idx, head_idx, _ in head_importances 
                               if (layer_idx, head_idx) not in pruned_set))
        
        # If we have too many layers, sample evenly
        if len(remaining_layers) > remaining_slots:
            # Sample layers evenly
            step = len(remaining_layers) / remaining_slots
            sampled_indices = [int(i * step) for i in range(remaining_slots)]
            remaining_layers = [remaining_layers[i] for i in sampled_indices]
        
        # Add one head from each remaining layer
        for layer_idx in remaining_layers:
            if len(layer_labels) >= max_labels:
                break
                
            heads = layers[layer_idx]
            # Find first non-pruned head
            for head_idx, importance in sorted(heads, key=lambda x: x[0]):
                if (layer_idx, head_idx) not in pruned_set:
                    layer_labels.append(f"L{layer_idx}-H{head_idx}")
                    head_importance_data.append(importance)
                    colors.append('blue')  # Kept heads are blue
                    included_layers.add(layer_idx)
                    break
    
    # Create horizontal bar chart
    y_pos = np.arange(len(layer_labels))
    ax.barh(y_pos, head_importance_data, color=colors)
    
    # Add labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(layer_labels)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Importance Score')
    
    # Create title with info on displayed vs total heads
    total_heads = sum(len(heads) for heads in layers.values())
    displayed_heads = len(layer_labels)
    title = f'Attention Head Importance (red = pruned, showing {displayed_heads}/{total_heads} heads)'
    ax.set_title(title)
    
    # Add a note if not all heads are shown
    if displayed_heads < total_heads:
        total_layers = len(layers)
        shown_layers = len(included_layers)
        plt.figtext(0.5, 0.01, 
                   f"Note: Only a subset of heads are shown for clarity. {shown_layers}/{total_layers} layers represented.", 
                   ha="center", fontsize=9, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Save figure if path provided
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    
    return fig

def visualize_gate_values(gate_values, output_path=None):
    """Create visualization of gate values across layers."""
    if not gate_values:
        return None
    
    # Prepare data
    layers = sorted(gate_values.keys())
    data = [gate_values[layer] for layer in layers]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot heatmap-like visualization
    for i, (layer, values) in enumerate(zip(layers, data)):
        # Create scatter plot for each layer
        x = np.arange(len(values))
        y = np.ones_like(x) * i
        
        # Use values to determine color and size
        colors = ['red' if v < 0.01 else 'blue' for v in values]
        sizes = [10 + 40 * v for v in values]
        
        ax.scatter(x, y, c=colors, s=sizes, alpha=0.7)
    
    # Customize plot
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels([layer.replace('layer_', 'Layer ') for layer in layers])
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Layer')
    ax.set_title('Attention Head Gate Values (Red = Pruned)')
    
    # Add colorbar legend
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', label='Pruned (gate ≈ 0)')
    blue_patch = mpatches.Patch(color='blue', label='Active (gate = 1)')
    ax.legend(handles=[red_patch, blue_patch], loc='upper right')
    
    # Save figure if path provided
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    
    return fig

def main(args):
    """Main function."""
    print("Starting pruning and fine-tuning experiment...")
    print(f"Running on device: {DEVICE}")
    
    # Clear memory if possible (helpful for large models)
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared CUDA cache")
    except Exception as e:
        print(f"Memory management error: {e}")
    
    # Set up directories
    output_dir, model_cache_dir, data_dir = setup_directories()
    
    # Create timestamp for output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"{args.model_name.replace('/', '_')}_{args.strategy}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Initialize progress tracker
    progress = ProgressMetrics()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name, cache_dir=model_cache_dir)
    
    # Load data
    train_dataloader, val_dataloader = load_wikitext_data(
        tokenizer, 
        max_length=args.max_length, 
        batch_size=args.batch_size
    )
    
    if train_dataloader is None or val_dataloader is None:
        print("Failed to load data. Exiting.")
        return 1
    
    # Initial evaluation
    print("Evaluating initial model performance...")
    initial_loss, initial_ppl = evaluate_model(model, val_dataloader)
    print(f"Initial loss: {initial_loss:.4f}, perplexity: {initial_ppl:.2f}")
    
    # Generate example text
    initial_generation = generate_text(
        model, tokenizer, 
        prompt="Artificial intelligence is becoming increasingly important because"
    )
    print("\nInitial generation example:")
    print(initial_generation)
    
    # Record initial metrics
    progress.update(step=0, loss=initial_loss, perplexity=initial_ppl, 
                   generation_sample=initial_generation)
    
    # Calculate head importances
    head_importances = get_head_importances(model, val_dataloader, strategy=args.strategy)
    
    # Prune heads
    pruned_heads = prune_heads(model, head_importances, pruning_level=args.pruning_level)
    
    # Record pruning information
    progress.set_pruning_info(args.strategy, args.pruning_level, pruned_heads)
    
    # Save head importance visualization
    importance_viz_path = os.path.join(run_dir, "head_importances.png")
    create_head_importance_visualization(
        head_importances, pruned_heads, importance_viz_path
    )
    
    # Evaluate after pruning
    print("\nEvaluating pruned model performance...")
    pruned_loss, pruned_ppl = evaluate_model(model, val_dataloader)
    print(f"After pruning: loss: {pruned_loss:.4f}, perplexity: {pruned_ppl:.2f}")
    
    # Generate example text with pruned model
    pruned_generation = generate_text(
        model, tokenizer, 
        prompt="Artificial intelligence is becoming increasingly important because"
    )
    print("\nAfter pruning generation example:")
    print(pruned_generation)
    
    # Record metrics after pruning
    progress.update(step=1, loss=pruned_loss, perplexity=pruned_ppl, 
                   gate_values=get_gate_values(model),
                   generation_sample=pruned_generation)
    
    # Save gate visualization
    gate_viz_path = os.path.join(run_dir, "gate_values.png")
    visualize_gate_values(get_gate_values(model), gate_viz_path)
    
    # Fine-tune pruned model
    print("\nFine-tuning pruned model...")
    try:
        # Clear GPU memory before fine-tuning
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Check if we should use CPU directly
        try:
            import psutil
            ram_gb = psutil.virtual_memory().available / (1024 ** 3)
            if DEVICE == "cpu" or (DEVICE == "cuda" and ram_gb < 2.0):
                print("Using CPU for fine-tuning due to memory constraints")
                model = model.cpu()
                DEVICE = "cpu"
                
                # Reduce batch size and epochs on CPU
                train_dataloader = torch.utils.data.DataLoader(
                    train_dataloader.dataset, 
                    batch_size=max(1, args.batch_size // 2),
                    shuffle=True
                )
                val_dataloader = torch.utils.data.DataLoader(
                    val_dataloader.dataset, 
                    batch_size=max(1, args.batch_size // 2),
                    shuffle=False
                )
                
                args.epochs = min(args.epochs, 1)
        except ImportError:
            pass
        
        # Use autocast for better memory efficiency
        import contextlib
        
        @contextlib.contextmanager
        def autocast_if_available():
            if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast') and USE_FP16:
                with torch.cuda.amp.autocast():
                    yield
            else:
                yield
                
        with autocast_if_available():
            fine_tune_results = fine_tune_model(
                model, 
                train_dataloader, 
                val_dataloader, 
                tokenizer,
                learning_rate=args.learning_rate,
                num_epochs=args.epochs,
                progress_tracker=progress
            )
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"\nCUDA error during fine-tuning: {e}")
            print("\nAttempting to continue with CPU...\n")
            
            # Try CPU fallback
            try:
                # Reset device to CPU
                DEVICE = "cpu"
                model = model.cpu()
                
                # Try with reduced parameters
                fine_tune_results = fine_tune_model(
                    model, 
                    train_dataloader, 
                    val_dataloader, 
                    tokenizer,
                    learning_rate=args.learning_rate / 2,  # Lower learning rate
                    num_epochs=1,  # Single epoch
                    progress_tracker=progress
                )
            except Exception as cpu_error:
                print(f"CPU fallback also failed: {cpu_error}")
                # Use pruned model metrics since fine-tuning failed
                fine_tune_results = {
                    "final_loss": pruned_loss, 
                    "final_perplexity": pruned_ppl,
                    "steps": 0
                }
        else:
            print(f"Error during fine-tuning: {e}")
            # Use pruned model metrics since fine-tuning failed
            fine_tune_results = {
                "final_loss": pruned_loss,
                "final_perplexity": pruned_ppl,
                "steps": 0
            }
    
    # Generate final examples
    final_generation = generate_text(
        model, tokenizer, 
        prompt="Artificial intelligence is becoming increasingly important because",
        temperature=0.7
    )
    print("\nFinal generation example:")
    print(final_generation)
    
    # Compare results
    summary = progress.get_summary()
    
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Model: {args.model_name}")
    print(f"Pruning strategy: {args.strategy}")
    print(f"Pruning level: {args.pruning_level:.1%}")
    print(f"Pruned heads: {len(pruned_heads)}")
    print("\nPerformance:")
    print(f"  Initial perplexity: {initial_ppl:.2f}")
    print(f"  After pruning: {pruned_ppl:.2f} ({(pruned_ppl-initial_ppl)/initial_ppl*100:+.2f}%)")
    print(f"  After fine-tuning: {fine_tune_results['final_perplexity']:.2f} ({(fine_tune_results['final_perplexity']-initial_ppl)/initial_ppl*100:+.2f}%)")
    
    # Save final plots
    progress.save_plots(os.path.join(run_dir, "training_progress.png"))
    progress.save_metrics(os.path.join(run_dir, "metrics.json"))
    
    # Save text samples
    with open(os.path.join(run_dir, "text_samples.txt"), "w") as f:
        f.write("INITIAL MODEL\n")
        f.write("============\n")
        f.write(initial_generation)
        f.write("\n\nAFTER PRUNING\n")
        f.write("============\n")
        f.write(pruned_generation)
        f.write("\n\nAFTER FINE-TUNING\n")
        f.write("===============\n")
        f.write(final_generation)
    
    print(f"\nAll results and visualizations saved to: {run_dir}")
    
    return 0

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prune and fine-tune GPT-2")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                        help="Name of the model to use (e.g., distilgpt2, facebook/opt-125m, EleutherAI/pythia-70m)")
    parser.add_argument("--max_length", type=int, default=256,
                       help="Maximum sequence length")
    
    # Pruning configuration
    parser.add_argument("--strategy", type=str, default="entropy",
                        choices=["random", "magnitude", "entropy"],
                        help="Pruning strategy to use")
    parser.add_argument("--pruning_level", type=float, default=0.3,
                       help="Fraction of heads to prune (0.0 to 1.0)")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate for fine-tuning")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--fp16", action="store_true",
                       help="Use FP16/mixed precision (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Adjust batch size and sequence length based on device
    if DEVICE == "cpu" and args.batch_size > 2:
        print("Running on CPU - reducing batch size to 2")
        args.batch_size = 2
        
    if DEVICE == "cpu" and args.max_length > 128:
        print("Running on CPU - reducing max_length to 128")
        args.max_length = 128
        
    # Override FP16 setting if specified
    if args.fp16:
        global USE_FP16
        USE_FP16 = True
        print("FP16 manually enabled")
    
    return args

if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))