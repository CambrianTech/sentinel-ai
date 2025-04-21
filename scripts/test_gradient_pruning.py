#!/usr/bin/env python
"""
Test script for the gradient-based pruning controller.

This script tests that the gradient-based pruning controller successfully prunes
heads based on gradient norms alone, which is useful when entropy calculations fail.
"""

import torch
import argparse
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from torch.utils.data import DataLoader
from datasets import load_dataset

from sentinel.pruning.gradient_based_pruning_controller import create_gradient_pruning_controller
from sentinel.pruning.dual_mode_pruning import PruningMode


def parse_args():
    parser = argparse.ArgumentParser(description="Test gradient-based pruning controller")
    parser.add_argument("--model", type=str, default="distilgpt2", help="Model name")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset config name")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--prune_percent", type=float, default=0.1, help="Percentage of heads to prune in each step")
    parser.add_argument("--steps", type=int, default=3, help="Number of pruning steps to simulate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--output", type=str, default="gradient_pruning_test.png", help="Output image path")
    return parser.parse_args()


def main(args):
    print(f"Testing gradient-based pruning on {args.model}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_dataset(args.dataset, args.dataset_config, split="validation")
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=args.max_length
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Add labels for language modeling
    tokenized_dataset = tokenized_dataset.map(
        lambda examples: {"labels": examples["input_ids"].copy()}
    )
    
    # Create dataloader
    tokenized_dataset = tokenized_dataset.with_format("torch")
    dataloader = DataLoader(
        tokenized_dataset, 
        batch_size=args.batch_size, 
        collate_fn=default_data_collator
    )
    
    # Create pruning controller
    controller = create_gradient_pruning_controller(
        model=model,
        mode=PruningMode.ADAPTIVE,
        prune_percent=args.prune_percent,
        gradient_percentile=30,
        min_zero_epochs=1,
        max_prune_percent=0.5
    )
    
    # Get initial summary
    summary = controller.get_summary()
    print(f"Model has {summary['total_heads']} attention heads")
    
    # Track pruning history
    pruning_history = []
    
    # Run multiple pruning steps
    for step in range(args.steps):
        print(f"\nStep {step+1}/{args.steps}")
        
        # Apply pruning
        pruned_heads, metrics = controller.step(dataloader, num_batches=2, verbose=True)
        
        # Update history
        pruning_history.append({
            "step": step,
            "pruned_heads": len(pruned_heads),
            "total_pruned": metrics["total_pruned"],
            "sparsity": metrics["sparsity"]
        })
        
        # Print status
        print(f"Pruned {len(pruned_heads)} heads in this step")
        print(f"Total pruned: {metrics['total_pruned']} out of {summary['total_heads']} heads")
        print(f"Sparsity: {metrics['sparsity']:.4f}")
        
        # Visualize current gradient patterns
        controller.visualize_gradient_patterns(figsize=(10, 6))
        plt.title(f"Gradient Patterns After Step {step+1}")
        plt.tight_layout()
        plt.savefig(f"gradient_patterns_step_{step+1}.png")
        plt.close()
    
    # Visualize pruning history
    plt.figure(figsize=(10, 6))
    plt.plot(
        [h["step"] for h in pruning_history],
        [h["total_pruned"] for h in pruning_history],
        "o-", label="Total Pruned Heads"
    )
    plt.xlabel("Step")
    plt.ylabel("Number of Pruned Heads")
    plt.title("Gradient-Based Pruning Progress")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output)
    
    print(f"\nTest completed. {pruning_history[-1]['total_pruned']} heads pruned.")
    print(f"Final model sparsity: {pruning_history[-1]['sparsity']:.4f}")
    print(f"Output saved to {args.output}")


if __name__ == "__main__":
    args = parse_args()
    main(args)