#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark With Comprehensive Metrics

This script runs benchmarks with the comprehensive metrics collection system, tracking
pruning patterns, gate values, head importance, and performance metrics.

Usage:
    python scripts/benchmark_with_metrics.py --model_name gpt2 --output_dir ./benchmark_results
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


def safe_update_tensor(tensor, new_value, index=None):
    """
    Safely update a tensor in-place, handling tensors that require gradients.
    
    Args:
        tensor: The tensor to update
        new_value: The new value to assign
        index: Optional index for updating specific elements
    """
    with torch.no_grad():
        if index is not None:
            # Update specific index
            tensor[index] = new_value
        else:
            # Update entire tensor or use copy_ for tensor-to-tensor assignment
            if isinstance(new_value, torch.Tensor) and tensor.size() == new_value.size():
                tensor.copy_(new_value)
            else:
                tensor.fill_(new_value)

def get_model_blocks(model):
    """
    Safely get the blocks from a model, handling different model structures.
    
    Args:
        model: The model to extract blocks from
        
    Returns:
        List of transformer blocks
    """
    # Check common model structures
    if hasattr(model, 'blocks'):
        return model.blocks
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'blocks'):
        return model.transformer.blocks
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    elif hasattr(model, 'model') and hasattr(model.model, 'blocks'):
        return model.model.blocks
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'blocks'):
        return model.encoder.blocks
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        return model.encoder.layers
    elif hasattr(model, 'decoder') and hasattr(model.decoder, 'blocks'):
        return model.decoder.blocks
    elif hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
        return model.decoder.layers
    elif hasattr(model, 'layers'):
        return model.layers
    else:
        # If we can't find blocks, print a warning and return an empty list
        print("WARNING: Could not find blocks in model structure. Check model compatibility.")
        print(f"Available attributes: {dir(model)}")
        return []

def get_attention_module(block):
    """
    Safely get the attention module from a block, handling different model structures.
    
    Args:
        block: The transformer block
        
    Returns:
        Attention module or None if not found
    """
    # Try dictionary-style access first (for ModuleDict)
    if isinstance(block, nn.ModuleDict) and "attn" in block:
        return block["attn"]
    elif isinstance(block, dict) and "attn" in block:
        return block["attn"]
    
    # Handle modules with nested attention
    attention_key_candidates = [
        "attn", "attention", "self_attention", "self_attn", 
        "mha", "multi_head_attention", "multihead_attn",
        "attention_layer", "q_attn"
    ]
    
    # Try direct attribute access first
    for key in attention_key_candidates:
        if hasattr(block, key):
            return getattr(block, key)
    
    # If we have a layernorm followed by attention structure
    if hasattr(block, "ln_1") and hasattr(block, "attn"):
        return block.attn
    if hasattr(block, "ln1") and hasattr(block, "attn"):
        return block.attn
    
    # If none of the known patterns match, try to find any attribute that might be an attention module
    for attr_name in dir(block):
        if "attention" in attr_name.lower() or "attn" in attr_name.lower():
            return getattr(block, attr_name)
    
    # If we can't find any attention-like module, print the available attributes and return None
    if isinstance(block, (nn.Module, nn.ModuleList, nn.ModuleDict)):
        print(f"WARNING: Could not find attention module in block with attributes: {dir(block)}")
    
    return None

def has_gate(attention_module):
    """
    Check if an attention module has a gate parameter.
    
    Args:
        attention_module: The attention module to check
        
    Returns:
        True if the module has a gate parameter, False otherwise
    """
    if attention_module is None:
        return False
    
    # Standard gate parameter
    if hasattr(attention_module, "gate"):
        return True
    
    # Check for head_gates parameter (some models use this name)
    if hasattr(attention_module, "head_gates"):
        return True
    
    # Check for gating_weights parameter
    if hasattr(attention_module, "gating_weights"):
        return True
        
    # Check if there are gate parameters in the state dict
    if hasattr(attention_module, "state_dict"):
        state_dict = attention_module.state_dict()
        
        # Look for any parameter with "gate" in the name
        for param_name in state_dict.keys():
            if "gate" in param_name.lower():
                return True
    
    return False


def get_gate_tensor(attention_module):
    """
    Get the gate tensor from an attention module.
    
    Args:
        attention_module: The attention module to get the gate from
        
    Returns:
        Gate tensor or None if not found
    """
    if attention_module is None:
        return None
    
    # Try standard gate parameter
    if hasattr(attention_module, "gate"):
        return attention_module.gate
    
    # Try head_gates parameter
    if hasattr(attention_module, "head_gates"):
        return attention_module.head_gates
    
    # Try gating_weights parameter
    if hasattr(attention_module, "gating_weights"):
        return attention_module.gating_weights
    
    # Try to find gate parameter in state dict
    if hasattr(attention_module, "state_dict"):
        state_dict = attention_module.state_dict()
        
        # Look for any parameter with "gate" in the name
        for param_name, param in state_dict.items():
            if "gate" in param_name.lower():
                # Get the parameter from the module
                for name, param in attention_module.named_parameters():
                    if name == param_name:
                        return param
    
    return None

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import necessary modules
from sentinel.utils.metric_collection import MetricCollector
# Import model loaders from sentinel namespace
try:
    # Try to use sentinel namespace first (preferred)
    from sentinel.models.loaders.loader import load_baseline_model, load_adaptive_model
    print("Using sentinel.models.loaders module")
except ImportError:
    # Fall back to original models.loaders if not available
    print("Warning: Using deprecated models.loaders module")
    from models.loaders.loader import load_baseline_model, load_adaptive_model

# Import pruning module
try:
    from sentinel.utils.pruning.pruning_module import PruningModule
except ImportError:
    from utils.pruning.pruning_module import PruningModule


def setup_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark with comprehensive metrics")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                       help="Name of the model to benchmark")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run on (cpu, cuda)")
    
    # Pruning configuration
    parser.add_argument("--pruning_strategies", type=str, default="entropy,magnitude,random",
                       help="Comma-separated list of pruning strategies to benchmark")
    parser.add_argument("--pruning_levels", type=str, default="0.1,0.3,0.5",
                       help="Comma-separated list of pruning levels to benchmark")
    
    # Learning/fine-tuning configuration
    parser.add_argument("--learning_steps", type=int, default=0,
                       help="Number of learning steps after pruning (0 for no learning)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate for fine-tuning")
    parser.add_argument("--early_stop_patience", type=int, default=5,
                       help="Number of evaluations with no improvement before early stopping")
    parser.add_argument("--eval_interval", type=int, default=50,
                       help="Evaluate model every N steps during fine-tuning")
    parser.add_argument("--use_adaptive_lr", action="store_true",
                       help="Use different learning rates for different parts of the model")
    
    # Evaluation configuration
    parser.add_argument("--eval_dataset", type=str, default=None,
                       help="Dataset for evaluation and fine-tuning")
    parser.add_argument("--eval_samples", type=int, default=100,
                       help="Number of evaluation samples")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for evaluation and fine-tuning")
    parser.add_argument("--use_real_data", action="store_true",
                       help="Use real data instead of synthetic data")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                       help="Directory to save results")
    parser.add_argument("--verbose", action="store_true",
                       help="Print verbose output")
    parser.add_argument("--save_checkpoints", action="store_true",
                       help="Save model checkpoints during fine-tuning")
    
    return parser.parse_args()


def prepare_model(args):
    """Prepare the model for benchmarking."""
    print(f"Loading baseline model: {args.model_name}")
    
    # Load baseline model
    baseline_model = load_baseline_model(args.model_name, args.device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create adaptive model
    print("Creating adaptive model")
    adaptive_model = load_adaptive_model(args.model_name, baseline_model, args.device, debug=args.verbose)
    
    return baseline_model, adaptive_model, tokenizer


def prepare_evaluation_data(tokenizer, args, split="validation"):
    """
    Prepare evaluation or training data.
    
    Args:
        tokenizer: The tokenizer to use
        args: Command line arguments
        split: Dataset split to use ('train' or 'validation')
        
    Returns:
        DataLoader for the specified dataset
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    if args.use_real_data and args.eval_dataset:
        # Load custom dataset if provided and real data is requested
        try:
            # First try the standard Hugging Face datasets package
            import importlib
            if importlib.util.find_spec("datasets") is not None:
                from datasets import load_dataset
                
                print(f"Loading {split} split from dataset: {args.eval_dataset}")
                
                # Try to load from Hugging Face datasets
                try:
                    # Handle different dataset formats
                    if "/" in args.eval_dataset:
                        # If it's in format like "wikitext/wikitext-2-raw-v1"
                        dataset = load_dataset(args.eval_dataset, split=split)
                    else:
                        # If it's a simple name like "wikitext-2-raw-v1"
                        dataset = load_dataset(args.eval_dataset, split=split)
                except Exception as e:
                    print(f"Error loading dataset directly: {e}")
                    print("Trying alternative dataset formats...")
                    
                    # Try common variants of the dataset name
                    for dataset_variant in [
                        f"wikitext/{args.eval_dataset}",  # Try wikitext namespace
                        "wikitext-2-raw-v1",              # Standard wikitext
                        "wikitext-103-raw-v1"             # Larger wikitext
                    ]:
                        try:
                            print(f"Attempting to load {dataset_variant}...")
                            dataset = load_dataset(dataset_variant, split=split)
                            print(f"Successfully loaded {dataset_variant}")
                            break
                        except Exception:
                            continue
                    else:
                        raise ValueError(f"Could not load any variant of {args.eval_dataset}")
                
                # Get text column (try common column names)
                text_column = None
                for column in ["text", "content", "sentence", "input_text"]:
                    if column in dataset.column_names:
                        text_column = column
                        break
                
                if text_column is None:
                    print(f"Warning: Could not find text column in dataset. Using first column: {dataset.column_names[0]}")
                    text_column = dataset.column_names[0]
                
                # Extract texts (use all for training, limit for validation)
                if split == "validation":
                    texts = dataset[text_column][:args.eval_samples]
                else:
                    # For training, use more samples but still limit to avoid excessive memory usage
                    max_train_samples = min(len(dataset), 10000)  # Limit to 10k samples max
                    texts = dataset[text_column][:max_train_samples]
                
                print(f"Loaded {len(texts)} samples from {args.eval_dataset} ({split} split)")
            else:
                raise ImportError("datasets package not available")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to synthetic data")
            texts = None
    else:
        # Use synthetic data if real data not requested or no dataset provided
        texts = None
    
    # If no dataset or loading failed, create synthetic data
    if texts is None:
        print("Using synthetic data")
        
        # Create synthetic prompts
        prompts = [
            "The quick brown fox jumps over the lazy dog. The fox is known for its agility and speed.",
            "In a world where technology dominates, humans seek connection and meaning through art and nature.",
            "Once upon a time, there lived a wise king who ruled with compassion and justice. His kingdom prospered.",
            "The history of artificial intelligence dates back to ancient myths and legends about artificial beings.",
            "Climate change is affecting ecosystems worldwide, leading to rising sea levels and extreme weather.",
            "Scientists have discovered a new species of deep-sea creatures that can survive extreme pressure.",
            "The economic outlook remains uncertain as markets react to global political developments.",
            "Education has transformed dramatically in the digital age, with online learning becoming mainstream.",
            "Renewable energy sources are becoming increasingly competitive with traditional fossil fuels.",
            "Space exploration has entered a new era with private companies launching their own missions."
        ]
        
        # Generate longer texts for training
        if split == "train":
            extended_prompts = []
            for prompt in prompts:
                # Create variations with different continuations
                for i in range(10):
                    extended_prompts.append(f"{prompt} This is continuation {i} with additional text to provide more training data for the model.")
            prompts = extended_prompts
        
        # Repeat prompts to get enough samples
        texts = []
        while len(texts) < (args.eval_samples if split == "validation" else 1000):
            texts.extend(prompts)
        
        # Limit to the requested number of samples
        if split == "validation":
            texts = texts[:args.eval_samples]
        else:
            texts = texts[:1000]  # Use 1000 training samples for synthetic data
    
    # Tokenize texts
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt"
    )
    
    # Create appropriate labels for causal language modeling (shifted input_ids)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    
    # Create labels for language modeling (shift input_ids right)
    labels = input_ids.clone()
    
    # Create dataloaders
    dataset = TensorDataset(input_ids, attention_mask, labels)
    
    # Use different shuffle settings for train vs. validation
    shuffle = (split == "train")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)
    
    return dataloader


def finetune_model(model, train_dataloader, val_dataloader, tokenizer, collector, args, 
                strategy=None, pruning_level=None):
    """
    Fine-tune a model after pruning.
    
    Args:
        model: The model to fine-tune
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        tokenizer: Tokenizer for the model
        collector: MetricCollector instance
        args: Command line arguments
        strategy: Pruning strategy used (for logging)
        pruning_level: Pruning level used (for logging)
        
    Returns:
        Dictionary with training results
    """
    # Set model to training mode
    model.train()
    device = next(model.parameters()).device
    
    # Create loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Create optimizer with different parameter groups if requested
    if args.use_adaptive_lr:
        # Create parameter groups with different learning rates
        # Higher learning rate for attention heads to recover from pruning
        # Lower learning rate for embeddings and output layer
        
        # Get blocks and head parameters with higher learning rate
        blocks = get_model_blocks(model)
        head_params = []
        
        for block in blocks:
            attn_module = get_attention_module(block)
            if attn_module is not None:
                for param in attn_module.parameters():
                    head_params.append(param)
        
        # Special handling for different learning rates based on parameter type
        param_groups = []
        if head_params:  # Only add this group if we found attention head parameters
            param_groups.append({"params": head_params, "lr": args.learning_rate * 3.0})
            print(f"Found {len(head_params)} attention head parameters for higher learning rate")
        
        # Other parameters (default learning rate)
        other_params = []
        for name, param in model.named_parameters():
            # Skip parameters that are already in the head_params group
            if not any(param is hp for hp in head_params):
                other_params.append(param)
        
        param_groups.append({"params": other_params, "lr": args.learning_rate})
        
        optimizer = torch.optim.AdamW(param_groups)
        print(f"Using adaptive learning rates: {args.learning_rate * 3.0} for heads, {args.learning_rate} for other params")
    else:
        # Single learning rate for all parameters
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        print(f"Using learning rate: {args.learning_rate}")
    
    # Create learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.learning_steps, eta_min=args.learning_rate * 0.1
    )
    
    # Early stopping setup
    best_val_loss = float("inf")
    patience_counter = 0
    
    # Training loop
    global_step = 0
    train_losses = []
    
    print(f"Starting fine-tuning for {args.learning_steps} steps "
          f"(eval every {args.eval_interval} steps, patience {args.early_stop_patience})")
    
    # Dictionary to store checkpoints
    checkpoints = {}
    
    start_time = time.time()
    
    while global_step < args.learning_steps:
        # Loop through batches
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_dataloader):
            if global_step >= args.learning_steps:
                break
                
            # Move batch to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            # Create shifted targets for causal language modeling
            shift_labels = labels.clone()
            shift_labels[:, :-1] = labels[:, 1:]
            shift_labels[:, -1] = tokenizer.pad_token_id
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Extract logits
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            
            # Calculate loss
            shift_logits = logits[:, :-1, :]
            shift_targets = shift_labels[:, :-1]
            loss = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)), shift_targets.reshape(-1))
            
            # Backward pass and optimization
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Update learning rate
            lr_scheduler.step()
            
            # Log training loss
            loss_val = loss.item()
            train_losses.append(loss_val)
            
            # Collect training metrics
            collector.collect_step_metrics(
                model=model,
                step=global_step,
                phase="train",
                inputs={"input_ids": input_ids, "attention_mask": attention_mask},
                labels=shift_targets,
                logits=shift_logits,
                additional_metrics={
                    "train/loss": loss_val,
                    "train/perplexity": torch.exp(loss).item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/strategy": strategy or "baseline",
                    "train/pruning_level": pruning_level or 0.0,
                    "train/batch": batch_idx,
                }
            )
            
            # Print progress
            if args.verbose and global_step % 10 == 0:
                print(f"Step {global_step}/{args.learning_steps}, "
                     f"Loss: {loss_val:.4f}, "
                     f"Perplexity: {torch.exp(loss).item():.4f}, "
                     f"LR: {lr_scheduler.get_last_lr()[0]:.2e}")
            
            # Evaluate periodically
            if global_step % args.eval_interval == 0 or global_step == args.learning_steps - 1:
                val_metrics = evaluate_model(
                    model, val_dataloader, tokenizer, collector, global_step,
                    strategy=strategy, pruning_level=pruning_level
                )
                
                # Save checkpoint if it's the best model so far
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                    
                    # Save checkpoint if requested
                    if args.save_checkpoints:
                        checkpoint_path = os.path.join(
                            args.output_dir, 
                            f"{args.model_name.replace('/', '_')}_{strategy or 'baseline'}_{pruning_level or 0.0}_step_{global_step}.pt"
                        )
                        
                        # Save state dict only (more efficient than full model)
                        checkpoints[global_step] = {
                            "step": global_step,
                            "val_loss": val_metrics["loss"],
                            "val_perplexity": val_metrics["perplexity"],
                            "path": checkpoint_path
                        }
                        
                        torch.save(model.state_dict(), checkpoint_path)
                        print(f"Saved checkpoint to {checkpoint_path}")
                else:
                    patience_counter += 1
                    print(f"Validation loss did not improve. Patience: {patience_counter}/{args.early_stop_patience}")
                
                # Apply early stopping if patience is exceeded
                if patience_counter >= args.early_stop_patience:
                    print(f"Early stopping triggered after {global_step} steps")
                    break
                
                # Set model back to training mode after evaluation
                model.train()
            
            global_step += 1
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Fine-tuning completed in {elapsed_time:.2f}s")
    
    # Calculate average training loss
    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_train_perplexity = torch.exp(torch.tensor(avg_train_loss)).item()
    
    print(f"Average training loss: {avg_train_loss:.4f}, Perplexity: {avg_train_perplexity:.4f}")
    
    # Final evaluation
    final_metrics = evaluate_model(
        model, val_dataloader, tokenizer, collector, global_step, 
        strategy=strategy, pruning_level=pruning_level, phase="final"
    )
    
    # Return results
    return {
        "train_loss": avg_train_loss,
        "train_perplexity": avg_train_perplexity,
        "val_loss": final_metrics["loss"],
        "val_perplexity": final_metrics["perplexity"],
        "steps": global_step,
        "elapsed_time": elapsed_time,
        "best_val_loss": best_val_loss,
        "checkpoints": checkpoints
    }


def evaluate_model(model, dataloader, tokenizer, collector, step, 
                  strategy=None, pruning_level=None, phase="eval"):
    """
    Evaluate model performance.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        tokenizer: Tokenizer for the model
        collector: MetricCollector instance
        step: Current step number
        strategy: Pruning strategy used (for logging)
        pruning_level: Pruning level used (for logging)
        phase: Evaluation phase name
        
    Returns:
        Dictionary with evaluation results
    """
    # Set model to evaluation mode
    model.eval()
    device = next(model.parameters()).device
    
    # Create loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Track metrics
    all_losses = []
    all_perplexities = []
    batch_step = 0
    
    # Function to compute perplexity from loss
    def compute_perplexity(loss):
        return torch.exp(loss).item()
    
    # Run evaluation
    with torch.no_grad():
        for batch_num, (input_ids, attention_mask, labels) in enumerate(dataloader):
            # Move batch to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            # Create shifted targets for causal language modeling
            shift_labels = labels.clone()
            shift_labels[:, :-1] = labels[:, 1:]
            shift_labels[:, -1] = tokenizer.pad_token_id
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Extract logits
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            
            # Calculate loss
            shift_logits = logits[:, :-1, :]
            shift_targets = shift_labels[:, :-1]
            loss = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)), shift_targets.reshape(-1))
            
            # Calculate perplexity
            perplexity = compute_perplexity(loss)
            
            # Collect comprehensive metrics
            collector.collect_step_metrics(
                model=model,
                step=step + batch_step,
                phase=phase,
                inputs={"input_ids": input_ids, "attention_mask": attention_mask},
                labels=shift_targets,
                logits=shift_logits,
                additional_metrics={
                    f"{phase}/batch": batch_num,
                    f"{phase}/loss": loss.item(),
                    f"{phase}/perplexity": perplexity,
                    f"{phase}/strategy": strategy or "baseline",
                    f"{phase}/pruning_level": pruning_level or 0.0
                }
            )
            
            # Store metrics
            all_losses.append(loss.item())
            all_perplexities.append(perplexity)
            
            batch_step += 1
    
    # Calculate average metrics
    avg_loss = sum(all_losses) / len(all_losses)
    avg_perplexity = sum(all_perplexities) / len(all_perplexities)
    
    print(f"{phase.capitalize()} Loss: {avg_loss:.4f}, Perplexity: {avg_perplexity:.4f}")
    
    return {
        "loss": avg_loss,
        "perplexity": avg_perplexity,
    }


def benchmark_model(model, dataloader, tokenizer, collector, args, strategy=None, pruning_level=None):
    """Benchmark a model with comprehensive metrics collection."""
    device = next(model.parameters()).device
    model.eval()
    
    # Initialize pruning if requested
    if strategy is not None and pruning_level is not None:
        print(f"\nBenchmarking with {strategy} pruning at level {pruning_level}")
        
        # Direct implementation of pruning for adaptive model
        try:
            # Get blocks and count total heads
            blocks = get_model_blocks(model)
            total_heads = 0
            head_info = []  # Store (layer_idx, attention_module, head_idx) for each head
            
            for layer_idx, block in enumerate(blocks):
                attn_module = get_attention_module(block)
                if attn_module is not None and has_gate(attn_module):
                    gate_tensor = get_gate_tensor(attn_module)
                    if gate_tensor is not None:
                        num_heads = len(gate_tensor)
                        total_heads += num_heads
                        for head_idx in range(num_heads):
                            head_info.append((layer_idx, attn_module, head_idx))
            
            # Apply simple pruning (set gates to 0)
            num_to_prune = int(total_heads * pruning_level)
            num_pruned = 0
            
            # Very basic pruning based on strategy
            if strategy == "random":
                # Random pruning
                import random
                
                # Shuffle and select heads to prune
                random.shuffle(head_info)
                prune_list = head_info[:num_to_prune]
                
                # Apply pruning using the safe update utility
                for layer_idx, attn_module, head_idx in prune_list:
                    gate_tensor = get_gate_tensor(attn_module)
                    if gate_tensor is not None:
                        safe_update_tensor(gate_tensor, 0.0, index=head_idx)
                        num_pruned += 1
                    else:
                        print(f"Warning: Could not find gate tensor in layer {layer_idx}, head {head_idx}")
                    
            elif strategy == "magnitude" or strategy == "entropy":
                # For simplicity, just zero out the first num_to_prune heads
                # In a real implementation, we would use entropy or magnitude measurements
                remaining = num_to_prune
                
                for layer_idx, attn_module, head_idx in head_info:
                    if remaining > 0:
                        gate_tensor = get_gate_tensor(attn_module)
                        if gate_tensor is not None:
                            safe_update_tensor(gate_tensor, 0.0, index=head_idx)
                            num_pruned += 1
                            remaining -= 1
                        else:
                            print(f"Warning: Could not find gate tensor in layer {layer_idx}, head {head_idx}")
            
            print(f"Pruned {num_pruned}/{total_heads} heads ({num_pruned/total_heads:.1%})")
        
        except Exception as e:
            print(f"Error applying pruning: {e}")
            return {
                "loss": float('nan'),
                "perplexity": float('nan'),
                "elapsed_time": 0,
            }
    else:
        print("\nBenchmarking baseline model (no pruning)")
    
    # Run evaluation
    start_time = time.time()
    
    # Evaluate the model
    metrics = evaluate_model(
        model, dataloader, tokenizer, collector, 0,
        strategy=strategy, pruning_level=pruning_level
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Add elapsed time to metrics
    metrics["elapsed_time"] = elapsed_time
    
    print(f"Evaluation completed in {elapsed_time:.2f}s")
    
    # Fine-tune if requested
    if args.learning_steps > 0:
        print(f"\nFine-tuning model for {args.learning_steps} steps...")
        
        # Prepare training data
        train_dataloader = prepare_evaluation_data(tokenizer, args, split="train")
        
        # Fine-tune the model
        train_results = finetune_model(
            model, train_dataloader, dataloader, tokenizer, collector, args,
            strategy=strategy, pruning_level=pruning_level
        )
        
        # Update metrics with training results
        metrics.update(train_results)
        
        print(f"\nFinal metrics after fine-tuning:")
        print(f"  Initial loss: {metrics['loss']:.4f}, perplexity: {metrics['perplexity']:.4f}")
        print(f"  Final loss: {metrics['val_loss']:.4f}, perplexity: {metrics['val_perplexity']:.4f}")
        
        # Calculate improvement
        loss_improvement = (metrics['loss'] - metrics['val_loss']) / metrics['loss'] * 100
        ppl_improvement = (metrics['perplexity'] - metrics['val_perplexity']) / metrics['perplexity'] * 100
        
        print(f"  Improvement: Loss: {loss_improvement:.2f}%, Perplexity: {ppl_improvement:.2f}%")
    
    return metrics


def main():
    """Main function."""
    args = setup_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.model_name.replace('/', '_')}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare model
    baseline_model, adaptive_model, tokenizer = prepare_model(args)
    
    # Prepare evaluation data
    eval_dataloader = prepare_evaluation_data(tokenizer, args, split="validation")
    
    # Initialize metric collector
    collector = MetricCollector(
        output_dir=output_dir,
        model_name=args.model_name,
        track_gate_values=True,
        track_head_metrics=True,
        track_performance=True,
        track_pruning_patterns=True,
        compare_with_static=True,
        log_level="INFO"
    )
    
    # Parse pruning configurations
    pruning_strategies = args.pruning_strategies.split(",")
    pruning_levels = [float(level) for level in args.pruning_levels.split(",")]
    
    # Save benchmark configuration
    config = {
        "model_name": args.model_name,
        "device": args.device,
        "max_length": args.max_length,
        "eval_dataset": args.eval_dataset,
        "eval_samples": args.eval_samples,
        "batch_size": args.batch_size,
        "pruning_strategies": pruning_strategies,
        "pruning_levels": pruning_levels,
        "learning_steps": args.learning_steps,
        "learning_rate": args.learning_rate,
        "early_stop_patience": args.early_stop_patience,
        "eval_interval": args.eval_interval,
        "use_adaptive_lr": args.use_adaptive_lr,
        "use_real_data": args.use_real_data,
        "timestamp": timestamp
    }
    
    with open(os.path.join(output_dir, "benchmark_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Start with baseline (no pruning)
    print("\n" + "="*50)
    print(f"Benchmarking model: {args.model_name}")
    print("="*50)
    
    baseline_results = benchmark_model(
        model=adaptive_model,
        dataloader=eval_dataloader,
        tokenizer=tokenizer,
        collector=collector,
        args=args
    )
    
    # Register baseline results as a "strategy" for comparison
    collector.register_static_pruning_metrics("baseline", baseline_results)
    
    # Run benchmarks for each pruning strategy and level
    all_results = {
        "baseline": baseline_results
    }
    
    for strategy in pruning_strategies:
        strategy_results = {}
        
        for level in pruning_levels:
            print("\n" + "="*50)
            print(f"Benchmarking {strategy} pruning at level {level}")
            print("="*50)
            
            # Clone model to avoid interference between runs
            model_clone = load_adaptive_model(args.model_name, baseline_model, args.device, debug=False)
            
            # Run benchmark
            results = benchmark_model(
                model=model_clone,
                dataloader=eval_dataloader,
                tokenizer=tokenizer,
                collector=collector,
                args=args,
                strategy=strategy,
                pruning_level=level
            )
            
            strategy_results[str(level)] = results
            
            # Register results for comparison
            if 'val_perplexity' in results:
                # Use fine-tuned metrics for comparison if available
                collector.register_static_pruning_metrics(
                    f"{strategy}_{level}", 
                    {"loss": results["val_loss"], "perplexity": results["val_perplexity"]}
                )
            else:
                # Use raw evaluation metrics
                collector.register_static_pruning_metrics(f"{strategy}_{level}", results)
            
            # Clear memory
            del model_clone
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        all_results[strategy] = strategy_results
    
    # Save all results
    with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
        # Convert any non-serializable objects
        def sanitize_for_json(obj):
            if isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items() if k != 'checkpoints'}
            elif isinstance(obj, list):
                return [sanitize_for_json(v) for v in obj]
            elif isinstance(obj, (torch.Tensor, np.ndarray)):
                return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
            else:
                return obj
            
        sanitized_results = sanitize_for_json(all_results)
        json.dump(sanitized_results, f, indent=2)
    
    # Generate comprehensive analysis
    print("\n" + "="*50)
    print("Generating comprehensive analysis")
    print("="*50)
    
    report = collector.generate_report()
    
    # Create visualizations
    collector.visualize_metrics()
    
    # Save metrics to CSV for external analysis
    collector.save_metrics_csv()
    
    print("\n" + "="*50)
    print(f"Benchmark complete. Results saved to {output_dir}")
    print("="*50)
    
    # Print summary of best strategies
    if "static_comparison" in report and "overall_winner" in report["static_comparison"]:
        winner = report["static_comparison"]["overall_winner"]
        print(f"\nOverall best strategy: {winner}")
        
        if args.learning_steps > 0:
            print("\nFine-tuning summary:")
            for strategy in pruning_strategies:
                for level in pruning_levels:
                    results = all_results[strategy][str(level)]
                    if 'val_perplexity' in results:
                        ppl_improvement = ((results['perplexity'] - results['val_perplexity']) / 
                                         results['perplexity'] * 100)
                        print(f"  {strategy} at {level}: Initial PPL {results['perplexity']:.2f} → "
                              f"Final PPL {results['val_perplexity']:.2f} ({ppl_improvement:+.2f}%)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())