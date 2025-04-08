#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural Plasticity Demonstration Script

This script demonstrates using the PlasticityController to dynamically prune and regrow
attention heads during training, showing "live" neural plasticity.
"""

import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    default_data_collator, 
    get_linear_schedule_with_warmup,
    Trainer, 
    TrainingArguments
)
from torch.utils.data import DataLoader
from datasets import load_dataset
from sentinel.pruning.dual_mode_pruning import PruningMode
from sentinel.pruning.plasticity_controller import create_plasticity_controller

# Constants
DATE_FORMAT = "%Y%m%d-%H%M%S"
DEFAULT_OUTPUT_DIR = os.path.join("output", "plasticity", f"run_{datetime.now().strftime(DATE_FORMAT)}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run neural plasticity experiment")
    
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                       help="Model name or path (default: distilgpt2)")
    
    parser.add_argument("--dataset", type=str, default="wikitext",
                       help="Dataset name (default: wikitext)")
    
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1",
                       help="Dataset configuration (default: wikitext-2-raw-v1)")
    
    parser.add_argument("--max_length", type=int, default=128,
                       help="Max sequence length (default: 128)")
    
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size (default: 4)")
    
    parser.add_argument("--num_epochs", type=int, default=5,
                       help="Number of epochs (default: 5)")
    
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate (default: 5e-5)")
    
    parser.add_argument("--warmup_steps", type=int, default=200,
                       help="Warmup steps (default: 200)")
    
    parser.add_argument("--eval_interval", type=int, default=100,
                       help="Evaluation interval in steps (default: 100)")
    
    parser.add_argument("--plasticity_mode", type=str, choices=["adaptive", "compressed"], default="adaptive",
                       help="Plasticity mode (adaptive or compressed, default: adaptive)")
    
    parser.add_argument("--high_entropy_threshold", type=float, default=0.8,
                       help="High entropy threshold for pruning (default: 0.8)")
    
    parser.add_argument("--low_entropy_threshold", type=float, default=0.4,
                       help="Low entropy threshold for revival (default: 0.4)")
    
    parser.add_argument("--grad_threshold", type=float, default=1e-4,
                       help="Gradient norm threshold (default: 1e-4)")
    
    parser.add_argument("--min_zero_epochs", type=int, default=3,
                       help="Minimum epochs to keep head zeroed (default: 3)")
    
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    
    parser.add_argument("--device", type=str, default=None,
                       help="Device (default: auto)")
    
    return parser.parse_args()


def load_and_prepare_data(args):
    """Load and prepare datasets."""
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
    
    return tokenizer, train_dataloader, validation_dataloader


def run_plasticity_experiment(args):
    """Run the neural plasticity experiment."""
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    
    # Load and prepare data
    tokenizer, train_dataloader, validation_dataloader = load_and_prepare_data(args)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment parameters
    with open(output_dir / "parameters.txt", "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Set pruning mode
    pruning_mode = PruningMode.ADAPTIVE if args.plasticity_mode == "adaptive" else PruningMode.COMPRESSED
    
    # Create plasticity controller
    controller = create_plasticity_controller(
        model=model,
        mode=pruning_mode,
        high_entropy_threshold=args.high_entropy_threshold,
        low_entropy_threshold=args.low_entropy_threshold,
        grad_threshold=args.grad_threshold,
        min_zero_epochs=args.min_zero_epochs
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Initialize learning rate scheduler
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=total_steps
    )
    
    # Initialize metrics tracking
    metrics_history = {
        "train_loss": [],
        "eval_loss": [],
        "pruned_heads": [],
        "revived_heads": [],
        "sparsity": [],
        "step": []
    }
    
    # Baseline evaluation
    baseline_loss = evaluate_model(model, validation_dataloader, device)
    print(f"Baseline validation loss: {baseline_loss:.4f}")
    
    # Training loop
    global_step = 0
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        model.train()
        
        epoch_loss = 0.0
        epoch_steps = 0
        
        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Track loss
            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1
            
            # Periodically evaluate and apply plasticity
            if global_step % args.eval_interval == 0:
                # Evaluate
                model.eval()
                eval_loss = evaluate_model(model, validation_dataloader, device)
                
                # Apply plasticity
                pruned, revived, plasticity_metrics = controller.step(
                    validation_dataloader, 
                    num_batches=min(5, len(validation_dataloader)),
                    verbose=True
                )
                
                # Update metrics
                metrics_history["train_loss"].append(epoch_loss / epoch_steps)
                metrics_history["eval_loss"].append(eval_loss)
                metrics_history["pruned_heads"].append(len(pruned))
                metrics_history["revived_heads"].append(len(revived))
                metrics_history["sparsity"].append(plasticity_metrics["sparsity"])
                metrics_history["step"].append(global_step)
                
                # Print status
                print(f"  Step {global_step} - Train loss: {epoch_loss / epoch_steps:.4f}, Eval loss: {eval_loss:.4f}")
                print(f"  Pruned: {len(pruned)} heads, Revived: {len(revived)} heads, Total pruned: {plasticity_metrics['total_pruned']}")
                print(f"  Sparsity: {plasticity_metrics['sparsity']:.4f}")
                
                # Reset for next interval
                epoch_loss = 0.0
                epoch_steps = 0
                
                # Back to training mode
                model.train()
        
        # End of epoch - visualize current state
        visualize_and_save_metrics(metrics_history, output_dir / f"metrics_epoch_{epoch+1}.png")
        
        # Visualize head dynamics
        controller.visualize_head_dynamics(
            metric='entropy', 
            save_path=output_dir / f"entropy_dynamics_epoch_{epoch+1}.png"
        )
        controller.visualize_head_dynamics(
            metric='decision', 
            save_path=output_dir / f"decision_dynamics_epoch_{epoch+1}.png"
        )
        
        # Save model checkpoint
        model.save_pretrained(output_dir / f"checkpoint_epoch_{epoch+1}")
        tokenizer.save_pretrained(output_dir / f"checkpoint_epoch_{epoch+1}")
    
    # Final evaluation
    model.eval()
    final_loss = evaluate_model(model, validation_dataloader, device)
    print(f"\nFinal validation loss: {final_loss:.4f}")
    print(f"Improvement over baseline: {(baseline_loss - final_loss) / baseline_loss * 100:.2f}%")
    
    # Get final summary
    summary = controller.get_summary()
    print("\nFinal Controller Summary:")
    print(f"  Total heads: {summary['total_heads']}")
    print(f"  Pruned heads: {summary['pruned_heads']} ({summary['pruning_rate']:.2%})")
    print(f"  Model sparsity: {summary['sparsity']:.4f}")
    print(f"  Model size: {summary['model_size_mb']:.2f} MB")
    
    # Final visualizations
    visualize_and_save_metrics(metrics_history, output_dir / "metrics_final.png")
    controller.visualize_head_dynamics(metric='entropy', save_path=output_dir / "entropy_dynamics_final.png")
    controller.visualize_head_dynamics(metric='decision', save_path=output_dir / "decision_dynamics_final.png")
    
    # Save model and tokenizer
    model.save_pretrained(output_dir / "final_model")
    tokenizer.save_pretrained(output_dir / "final_model")
    
    # Save final metrics
    torch.save(metrics_history, output_dir / "metrics_history.pt")
    
    print(f"\nExperiment completed. Results saved to: {output_dir}")


def evaluate_model(model, dataloader, device):
    """Evaluate model on the provided dataloader."""
    model.eval()
    total_loss = 0.0
    total_steps = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            total_loss += loss.item()
            total_steps += 1
    
    return total_loss / total_steps if total_steps > 0 else float("inf")


def visualize_and_save_metrics(metrics_history, save_path):
    """Visualize and save training metrics."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot losses
    ax1.plot(metrics_history["step"], metrics_history["train_loss"], label="Train Loss")
    ax1.plot(metrics_history["step"], metrics_history["eval_loss"], label="Eval Loss")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Evaluation Loss")
    ax1.legend()
    ax1.grid(True)
    
    # Plot pruning metrics
    ax2.bar(metrics_history["step"], metrics_history["pruned_heads"], alpha=0.5, label="Pruned Heads", color="blue")
    ax2.bar(metrics_history["step"], metrics_history["revived_heads"], alpha=0.5, label="Revived Heads", color="green")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Count")
    ax2.set_title("Head Pruning and Revival")
    ax2.legend(loc="upper left")
    ax2.grid(True)
    
    # Add sparsity line on secondary axis
    ax3 = ax2.twinx()
    ax3.plot(metrics_history["step"], metrics_history["sparsity"], "r-", label="Sparsity")
    ax3.set_ylabel("Sparsity")
    ax3.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main function."""
    args = parse_args()
    run_plasticity_experiment(args)


if __name__ == "__main__":
    main()