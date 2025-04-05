#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for entropy and magnitude-based pruning implementations.

This script demonstrates how to use the entropy_magnitude module for pruning
transformer models. It performs a simplified benchmark using different pruning
strategies and displays the results.

Usage:
    python scripts/test_entropy_magnitude_pruning.py --model_name distilgpt2 [--device cuda]
"""

import os
import sys
import argparse
import torch
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)


def setup_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test entropy and magnitude pruning")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                       help="Name of the model to benchmark")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run on (cpu, cuda)")
    
    # Pruning configuration
    parser.add_argument("--pruning_strategies", type=str, default="random,entropy,magnitude",
                       help="Comma-separated list of pruning strategies to benchmark")
    parser.add_argument("--pruning_levels", type=str, default="0.1,0.3,0.5",
                       help="Comma-separated list of pruning levels to benchmark")
    
    # Evaluation configuration
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for evaluation")
    parser.add_argument("--num_batches", type=int, default=5,
                       help="Number of batches to collect attention distributions")
    parser.add_argument("--verbose", action="store_true",
                       help="Print verbose output")
                       
    return parser.parse_args()


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


def prepare_model(args):
    """Prepare the model for testing."""
    print(f"Loading model: {args.model_name}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to device
    model = model.to(args.device)
    
    # Add gate attributes to attention modules if they don't exist
    for layer in model.transformer.h:
        if not hasattr(layer.attn, "gate"):
            # Different models use different attribute names for number of heads
            num_heads = getattr(layer.attn, "num_heads", 
                              getattr(layer.attn, "n_head", 
                                    getattr(layer.attn, "n_heads", 12)))
            layer.attn.register_buffer("gate", torch.ones(num_heads, device=args.device))
            print(f"Added gate attribute to layer with {num_heads} heads")
    
    return model, tokenizer


def prepare_sample_data(tokenizer, args):
    """Create a small sample dataset for testing."""
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create sample prompts
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a world where technology dominates, humans seek connection.",
        "Once upon a time, there lived a wise king who ruled with compassion.",
        "The history of artificial intelligence dates back to ancient myths.",
        "Climate change is affecting ecosystems worldwide, leading to rising sea levels.",
        "Scientists have discovered a new species of deep-sea creatures.",
        "The economic outlook remains uncertain as markets react to global developments.",
        "Education has transformed dramatically in the digital age.",
        "Renewable energy sources are becoming increasingly competitive.",
        "Space exploration has entered a new era with private companies."
    ]
    
    # Tokenize prompts
    tokenized = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt"
    )
    
    # Create dataset and dataloader
    input_ids = tokenized["input_ids"].to(args.device)
    attention_mask = tokenized["attention_mask"].to(args.device)
    labels = input_ids.clone()
    
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    return dataloader


def evaluate_perplexity(model, dataloader):
    """Evaluate model perplexity on a dataloader."""
    model.eval()
    device = next(model.parameters()).device
    
    # Create loss function
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    
    all_losses = []
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            # Move data to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Extract logits
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            
            # Create shifted targets for causal language modeling
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            
            # Calculate loss
            loss = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)), 
                          shift_labels.reshape(-1))
            
            all_losses.append(loss.item())
    
    # Calculate average loss and perplexity
    avg_loss = sum(all_losses) / len(all_losses)
    perplexity = np.exp(avg_loss)
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity
    }


def generate_text_sample(model, tokenizer, prompt="Once upon a time", max_length=50):
    """Generate a sample text using the model."""
    model.eval()
    device = next(model.parameters()).device
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    
    # Generate text
    with torch.no_grad():
        # Create attention mask (all 1s for the input length)
        attention_mask = torch.ones_like(input_ids)
        
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


def apply_pruning(model, dataloader, strategy, pruning_level, args):
    """Apply a pruning strategy to the model."""
    from sentinel.pruning.entropy_magnitude import (
        collect_attention_distributions,
        entropy_based_pruning,
        magnitude_based_pruning
    )
    
    print(f"\nApplying {strategy} pruning at level {pruning_level}")
    
    start_time = time.time()
    
    # Count total heads
    total_heads = sum(layer.attn.gate.numel() for layer in model.transformer.h)
    num_to_prune = int(total_heads * pruning_level)
    
    print(f"Model has {total_heads} attention heads, pruning {num_to_prune} heads ({pruning_level:.1%})")
    
    if strategy == "random":
        # Random pruning - just set random gates to 0
        import random
        
        # Get all heads
        all_heads = []
        for layer_idx, layer in enumerate(model.transformer.h):
            for head_idx in range(layer.attn.gate.numel()):
                all_heads.append((layer_idx, head_idx))
        
        # Shuffle and select heads to prune
        random.shuffle(all_heads)
        heads_to_prune = all_heads[:num_to_prune]
        
        # Apply pruning
        for layer_idx, head_idx in tqdm(heads_to_prune, desc="Pruning heads"):
            safe_update_tensor(model.transformer.h[layer_idx].attn.gate, 0.0, index=head_idx)
        
        pruned_heads = [(layer_idx, head_idx, 0.0) for layer_idx, head_idx in heads_to_prune]
        
    elif strategy == "entropy":
        # Entropy-based pruning
        print("Collecting attention distributions...")
        distributions = collect_attention_distributions(
            model,
            dataloader,
            num_batches=args.num_batches
        )
        
        # Apply entropy-based pruning
        pruned_heads = entropy_based_pruning(
            model,
            distributions,
            prune_ratio=pruning_level,
            safe_update_tensor_fn=safe_update_tensor
        )
        
    elif strategy == "magnitude":
        # Magnitude-based pruning
        pruned_heads = magnitude_based_pruning(
            model,
            prune_ratio=pruning_level,
            safe_update_tensor_fn=safe_update_tensor
        )
    
    else:
        raise ValueError(f"Unknown pruning strategy: {strategy}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Pruned {len(pruned_heads)} heads in {elapsed_time:.2f}s")
    
    return pruned_heads


def main():
    """Main function."""
    args = setup_args()
    
    # Prepare model
    model, tokenizer = prepare_model(args)
    
    # Create sample data
    dataloader = prepare_sample_data(tokenizer, args)
    
    # Parse pruning configurations
    pruning_strategies = args.pruning_strategies.split(",")
    pruning_levels = [float(level) for level in args.pruning_levels.split(",")]
    
    # Track results
    results = {}
    
    # First, evaluate the baseline model
    print("\n===== Evaluating baseline model (no pruning) =====")
    baseline_metrics = evaluate_perplexity(model, dataloader)
    print(f"Baseline perplexity: {baseline_metrics['perplexity']:.4f}")
    
    # Generate a sample
    baseline_sample = generate_text_sample(model, tokenizer)
    print(f"\nBaseline generation sample:\n{baseline_sample}\n")
    
    results["baseline"] = {
        "perplexity": baseline_metrics["perplexity"],
        "sample": baseline_sample
    }
    
    # Evaluate each pruning strategy and level
    for strategy in pruning_strategies:
        results[strategy] = {}
        
        for level in pruning_levels:
            # Create a copy of the model to avoid interference between runs
            model_copy = AutoModelForCausalLM.from_pretrained(args.model_name).to(args.device)
            
            # Add gate attributes if they don't exist
            for layer in model_copy.transformer.h:
                if not hasattr(layer.attn, "gate"):
                    layer.attn.register_buffer("gate", torch.ones(layer.attn.n_heads, device=args.device))
            
            # Apply pruning
            pruned_heads = apply_pruning(model_copy, dataloader, strategy, level, args)
            
            # Evaluate the pruned model
            print(f"\n===== Evaluating {strategy} pruning at level {level} =====")
            metrics = evaluate_perplexity(model_copy, dataloader)
            print(f"Perplexity after pruning: {metrics['perplexity']:.4f}")
            
            # Calculate degradation
            degradation = (metrics['perplexity'] - baseline_metrics['perplexity']) / baseline_metrics['perplexity'] * 100
            print(f"Performance degradation: {degradation:.2f}%")
            
            # Generate a sample
            pruned_sample = generate_text_sample(model_copy, tokenizer)
            print(f"\nGeneration sample after pruning:\n{pruned_sample}\n")
            
            # Store results
            results[strategy][str(level)] = {
                "perplexity": metrics["perplexity"],
                "degradation": degradation,
                "sample": pruned_sample
            }
            
            # Clean up
            del model_copy
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Print summary
    print("\n===== SUMMARY =====")
    print(f"Model: {args.model_name}")
    print(f"Baseline perplexity: {baseline_metrics['perplexity']:.4f}")
    
    for strategy in pruning_strategies:
        print(f"\nStrategy: {strategy}")
        for level in pruning_levels:
            level_str = str(level)
            if level_str in results[strategy]:
                degradation = results[strategy][level_str]["degradation"]
                print(f"  Level {level}: Perplexity {results[strategy][level_str]['perplexity']:.4f} (degradation: {degradation:+.2f}%)")
    
    # Identify best strategy
    best_strategy = None
    best_level = None
    best_degradation = float('inf')
    
    for strategy in pruning_strategies:
        for level in pruning_levels:
            level_str = str(level)
            if level_str in results[strategy]:
                degradation = results[strategy][level_str]["degradation"]
                if degradation < best_degradation:
                    best_degradation = degradation
                    best_strategy = strategy
                    best_level = level
    
    if best_strategy:
        print(f"\nBest pruning configuration: {best_strategy} at level {best_level} (degradation: {best_degradation:.2f}%)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())