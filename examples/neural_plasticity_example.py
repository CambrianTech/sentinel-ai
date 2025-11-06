#!/usr/bin/env python
"""
Neural Plasticity Example

This script demonstrates how to use the neural plasticity components to:
1. Load a transformer model
2. Measure head entropy and gradients
3. Prune low-utility heads 
4. Train with differential learning rates
5. Evaluate and visualize the results

Usage:
  python examples/neural_plasticity_example.py
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from transformers import default_data_collator
from torch.utils.data import DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt
import argparse

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
    run_plasticity_loop
)


def parse_args():
    parser = argparse.ArgumentParser(description="Neural Plasticity Example")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="Model name")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset config")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--pruning_level", type=float, default=0.2, help="Pruning level (0-1)")
    parser.add_argument("--pruning_strategy", type=str, default="gradient", 
                        choices=["gradient", "entropy", "random", "combined"], 
                        help="Pruning strategy")
    parser.add_argument("--training_steps", type=int, default=500, help="Training steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Base learning rate")
    parser.add_argument("--output_dir", type=str, default="./neural_plasticity_output", 
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to run on (defaults to cuda if available)")
    parser.add_argument("--no_differential_lr", action="store_true", 
                        help="Disable differential learning rates")
    
    return parser.parse_args()


def prepare_dataset(args):
    """Prepare dataset for training and evaluation"""
    print(f"Loading dataset: {args.dataset}/{args.dataset_config}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    train_dataset = load_dataset(args.dataset, args.dataset_config, split="train")
    validation_dataset = load_dataset(args.dataset, args.dataset_config, split="validation")
    
    # Define tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=args.max_length
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
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=default_data_collator
    )
    
    validation_dataloader = DataLoader(
        validation_dataset, 
        batch_size=args.batch_size, 
        collate_fn=default_data_collator
    )
    
    print(f"Train dataset size: {len(train_dataset)} examples")
    print(f"Validation dataset size: {len(validation_dataset)} examples")
    
    return tokenizer, train_dataloader, validation_dataloader


def save_visualizations(results, output_dir):
    """Save visualizations to output directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize head gradients
    grad_fig = visualize_head_gradients(
        results["grad_norm_values"],
        pruned_heads=results["pruned_heads"],
        title="Head Gradient Norms with Pruned Heads"
    )
    grad_fig.savefig(os.path.join(output_dir, "head_gradients.png"))
    plt.close(grad_fig)
    
    # Visualize entropy if available
    if results["entropy_values"] is not None:
        entropy_fig = visualize_head_entropy(
            results["entropy_values"],
            title="Head Entropy Values"
        )
        entropy_fig.savefig(os.path.join(output_dir, "head_entropy.png"))
        plt.close(entropy_fig)
    
    # Visualize pruning decisions
    pruning_fig = visualize_pruning_decisions(
        results["grad_norm_values"],
        results["pruning_mask"],
        title=f"Pruning Decisions ({results['strategy']} strategy, {results['pruning_level']:.0%} level)"
    )
    pruning_fig.savefig(os.path.join(output_dir, "pruning_decisions.png"))
    plt.close(pruning_fig)
    
    # Visualize training metrics
    metrics_fig = visualize_training_metrics(
        results["training_metrics"],
        title="Training Progress"
    )
    metrics_fig.savefig(os.path.join(output_dir, "training_metrics.png"))
    plt.close(metrics_fig)
    
    # Save results summary
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write(f"Neural Plasticity Results\n")
        f.write(f"=======================\n\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Pruning Strategy: {results['strategy']}\n")
        f.write(f"Pruning Level: {results['pruning_level']:.0%}\n")
        f.write(f"Pruned Heads: {len(results['pruned_heads'])}\n\n")
        
        f.write(f"Baseline Perplexity: {results['baseline_metrics']['perplexity']:.2f}\n")
        f.write(f"Pruned Perplexity: {results['pruned_metrics']['perplexity']:.2f}\n")
        f.write(f"Final Perplexity: {results['final_metrics']['perplexity']:.2f}\n\n")
        
        f.write(f"Perplexity Improvement: {results['perplexity_improvement']*100:.2f}%\n")
        f.write(f"Recovery Rate: {results['recovery_rate']*100:.2f}%\n")
    
    print(f"Saved visualizations to {output_dir}")


def generate_sample_text(model, tokenizer, prompt="Once upon a time", max_length=100, device="cuda"):
    """Generate sample text from the model"""
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and return text
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main(args):
    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    
    # Prepare dataset
    tokenizer, train_dataloader, validation_dataloader = prepare_dataset(args)
    
    # Define progress tracking callback
    def progress_callback(phase, step, metrics):
        if phase == "baseline":
            print(f"Baseline evaluation: Perplexity = {metrics['perplexity']:.2f}")
        elif phase == "post_pruning":
            print(f"Post-pruning evaluation: Perplexity = {metrics['perplexity']:.2f}")
        elif phase == "training" and step % 100 == 0:
            print(f"Training step {step}: Loss = {metrics['train_loss']:.4f}, Perplexity = {metrics['perplexity']:.2f}")
        elif phase == "final":
            print(f"Final evaluation: Perplexity = {metrics['perplexity']:.2f}")
    
    # Run neural plasticity loop
    results = run_plasticity_loop(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=validation_dataloader,
        pruning_level=args.pruning_level,
        strategy=args.pruning_strategy,
        learning_rate=args.learning_rate,
        training_steps=args.training_steps,
        use_differential_lr=not args.no_differential_lr,
        callback=progress_callback
    )
    
    # Generate sample text before and after
    baseline_text = results.get("baseline_generation")
    if baseline_text is None:
        baseline_text = generate_sample_text(model, tokenizer, device=device)
    
    final_text = generate_sample_text(model, tokenizer, device=device)
    
    # Print sample text
    print("\nGenerated Text Comparison:")
    print(f"Baseline: {baseline_text[:100]}...")
    print(f"Final: {final_text[:100]}...")
    
    # Save visualizations
    save_visualizations(results, args.output_dir)
    
    # Print summary
    print("\nNeural Plasticity Summary:")
    print(f"Pruned Heads: {len(results['pruned_heads'])}")
    print(f"Baseline Perplexity: {results['baseline_metrics']['perplexity']:.2f}")
    print(f"Pruned Perplexity: {results['pruned_metrics']['perplexity']:.2f}")
    print(f"Final Perplexity: {results['final_metrics']['perplexity']:.2f}")
    print(f"Perplexity Improvement: {results['perplexity_improvement']*100:.2f}%")
    print(f"Recovery Rate: {results['recovery_rate']*100:.2f}%")


if __name__ == "__main__":
    args = parse_args()
    main(args)