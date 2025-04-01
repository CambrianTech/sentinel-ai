#!/usr/bin/env python

"""
Fine-Tuning Script for Pruned Models

This script implements a targeted fine-tuning strategy for pruned transformer models
to recover accuracy lost during pruning. It focuses on efficient fine-tuning techniques
that preserve the performance benefits of pruning while improving output quality.
"""

import os
import sys
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# Import model handling utilities
from models.loaders.loader import load_baseline_model, load_adaptive_model
from models.loaders.loader_optimized import load_optimized_adaptive_model
from scripts.pruning_comparison.pruning_agency_comparison import apply_pruning


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune pruned models to recover accuracy")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="gpt2", 
                       help="Name of the model to fine-tune")
    parser.add_argument("--optimization_level", type=int, default=3,
                      help="Optimization level to use (0-3)")
    parser.add_argument("--pruning_level", type=float, default=0.5,
                       help="Pruning level to apply before fine-tuning (0.0-1.0)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for training")
    
    # Dataset and training options
    parser.add_argument("--dataset", type=str, default="wikitext",
                       help="Dataset to use for fine-tuning (default: wikitext)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate for fine-tuning")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--max_seq_length", type=int, default=128,
                       help="Maximum sequence length for training")
    
    # Specialized fine-tuning options
    parser.add_argument("--active_heads_only", action="store_true",
                       help="Only fine-tune active attention heads after pruning")
    parser.add_argument("--per_head_lr", action="store_true",
                       help="Use per-head learning rates based on importance")
    parser.add_argument("--adaptive_masking", action="store_true",
                       help="Use adaptive masking during fine-tuning")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="finetuned_models",
                      help="Directory to save fine-tuned models")
    parser.add_argument("--save_steps", type=int, default=1000,
                      help="Save model every X steps")
    parser.add_argument("--evaluation_steps", type=int, default=500,
                      help="Evaluate model every X steps")
    parser.add_argument("--visualize", action="store_true",
                      help="Generate training visualizations")
    
    return parser.parse_args()


class TextDataset(Dataset):
    """Dataset for language model fine-tuning."""
    def __init__(self, tokenizer, file_path, block_size=128):
        """
        Initialize dataset from text file.
        
        Args:
            tokenizer: Tokenizer to use
            file_path: Path to text file
            block_size: Maximum sequence length
        """
        if os.path.isdir(file_path):
            self.examples = []
            for file in os.listdir(file_path):
                if file.endswith(".txt"):
                    with open(os.path.join(file_path, file), encoding="utf-8") as f:
                        text = f.read()
                        self.examples.extend(self._tokenize_and_block(tokenizer, text, block_size))
        else:
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
                self.examples = self._tokenize_and_block(tokenizer, text, block_size)
    
    def _tokenize_and_block(self, tokenizer, text, block_size):
        """Tokenize text and split into blocks of specified length."""
        tokenized = tokenizer.encode(text)
        examples = []
        
        for i in range(0, len(tokenized) - block_size, block_size):
            examples.append(
                tokenizer.build_inputs_with_special_tokens(tokenized[i:i + block_size])
            )
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


def load_dataset(dataset_name, tokenizer, max_seq_length):
    """Load dataset for fine-tuning."""
    if dataset_name == "wikitext":
        try:
            from datasets import load_dataset
            dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
            texts = dataset["text"]
            
            # Filter out empty texts and join
            texts = [text for text in texts if text.strip()]
            full_text = "\n\n".join(texts)
            
            # Create custom dataset
            return TextDataset(tokenizer, full_text, block_size=max_seq_length)
        except Exception as e:
            print(f"Error loading wikitext: {e}")
            print("Using dummy dataset for demonstration...")
            
            # Create a simple dummy dataset
            dummy_text = "This is a dummy dataset for fine-tuning a language model. " * 1000
            tmp_file = "dummy_dataset.txt"
            with open(tmp_file, "w") as f:
                f.write(dummy_text)
            
            dataset = TextDataset(tokenizer, tmp_file, block_size=max_seq_length)
            os.remove(tmp_file)
            
            return dataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def prepare_model(args):
    """Load and prepare model for fine-tuning."""
    print(f"Loading model {args.model_name} with optimization level {args.optimization_level}...")
    
    # Load baseline model
    baseline_model = load_baseline_model(args.model_name, args.device)
    
    # Load optimized model if specified
    if args.optimization_level > 0:
        model = load_optimized_adaptive_model(
            args.model_name,
            baseline_model,
            args.device,
            optimization_level=args.optimization_level
        )
    else:
        model = load_adaptive_model(args.model_name, baseline_model, args.device)
    
    # Apply pruning if specified
    if args.pruning_level > 0:
        print(f"Applying {args.pruning_level*100}% pruning...")
        model, pruned_count, pruned_heads = apply_pruning(
            model, 
            args.pruning_level,
            verbose=True,
            quiet=False
        )
        print(f"Pruned {pruned_count} heads")
    
    # Return model in training mode
    model.train()
    return model


def create_optimizer_and_scheduler(model, args, total_steps):
    """Create optimizer and learning rate scheduler with specialized options."""
    # Standard parameter groups
    param_groups = [{"params": [p for p in model.parameters() if p.requires_grad]}]
    
    # Use per-head learning rates if specified
    if args.per_head_lr and hasattr(model, "blocks"):
        # Reset parameter groups to be more granular
        param_groups = []
        
        # Add parameters that are not part of attention heads
        non_head_params = []
        for name, param in model.named_parameters():
            if "attn" not in name and "attention" not in name and param.requires_grad:
                non_head_params.append(param)
        
        if non_head_params:
            param_groups.append({"params": non_head_params, "lr": args.learning_rate})
        
        # Add attention head parameters with specialized learning rates
        for i, block in enumerate(model.blocks):
            attn_module = block.attn if hasattr(block, "attn") else block.attention
            
            # Only fine-tune active heads if specified
            if args.active_heads_only and hasattr(attn_module, "gate"):
                active_heads = []
                for head_idx, gate_val in enumerate(attn_module.gate):
                    if float(gate_val) > 0.2:  # Head is active
                        active_heads.append(head_idx)
                
                # Adjust learning rate based on head activity
                for head_idx in range(attn_module.num_heads):
                    head_params = []
                    head_lr = args.learning_rate
                    
                    # Find parameters for this specific head
                    head_active = head_idx in active_heads
                    
                    # Only include active heads if active_heads_only is set
                    if not args.active_heads_only or head_active:
                        # Find parameters for this specific head and add them
                        for name, param in attn_module.named_parameters():
                            if param.requires_grad and (
                                # Match head-specific parameters
                                f"heads.{head_idx}" in name or
                                f"head_{head_idx}" in name or
                                # Handle special cases for different architectures
                                (name == "gate" and head_idx < len(param))
                            ):
                                head_params.append(param)
                        
                        # Adjust learning rate for inactive heads
                        if not head_active:
                            head_lr *= 0.1  # Lower learning rate for inactive heads
                        
                        # Add parameter group for this head
                        if head_params:
                            param_groups.append({
                                "params": head_params,
                                "lr": head_lr,
                                "layer": i,
                                "head": head_idx,
                                "active": head_active
                            })
            else:
                # Standard approach - add all attention parameters
                attn_params = [p for n, p in attn_module.named_parameters() if p.requires_grad]
                if attn_params:
                    param_groups.append({
                        "params": attn_params,
                        "lr": args.learning_rate,
                        "layer": i
                    })
    
    # Create AdamW optimizer
    optimizer = AdamW(param_groups, lr=args.learning_rate)
    
    # Create learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=total_steps
    )
    
    return optimizer, scheduler


def train_model(model, tokenizer, dataset, args):
    """Fine-tune the model on the specified dataset."""
    # Create data loader
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
    )
    
    # Calculate total training steps
    total_steps = len(data_loader) * args.epochs
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, args, total_steps)
    
    # Initialize tracking variables
    losses = []
    perplexities = []
    global_step = 0
    
    # Training loop
    print(f"Starting fine-tuning for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Iterate through data loader with progress bar
        for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}"):
            # Move batch to device
            batch = batch.to(args.device)
            
            # Forward pass
            outputs = model(batch, labels=batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update tracking variables
            losses.append(loss.item())
            perplexity = torch.exp(loss).item()
            perplexities.append(perplexity)
            
            # Print progress
            if global_step % 100 == 0:
                print(f"Step {global_step}: Loss = {loss.item():.4f}, Perplexity = {perplexity:.2f}")
            
            # Save model checkpoint
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                print(f"Saved model checkpoint to {output_dir}")
            
            # Evaluate model
            if args.evaluation_steps > 0 and global_step % args.evaluation_steps == 0:
                # Run evaluation (can be expanded)
                print(f"Evaluating model at step {global_step}...")
                eval_loss = evaluate_model(model, tokenizer, dataset, args)
                print(f"Evaluation loss: {eval_loss:.4f}, Perplexity: {torch.exp(torch.tensor(eval_loss)).item():.2f}")
            
            global_step += 1
    
    # Save final model
    final_output_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_output_dir, exist_ok=True)
    try:
        model.save_pretrained(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
    except Exception as e:
        print(f"Error saving model: {e}")
        # Fallback method: save state dict
        torch.save(model.state_dict(), os.path.join(final_output_dir, "pytorch_model.bin"))
    
    print(f"Saved final model to {final_output_dir}")
    
    # Create training visualizations
    if args.visualize:
        create_training_visualizations(losses, perplexities, args)
    
    return model, losses, perplexities


def evaluate_model(model, tokenizer, dataset, args):
    """Evaluate model on validation set."""
    # Simple evaluation on a subset of the training data
    # In a real scenario, you would use a separate validation dataset
    eval_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    # Use a small subset for quick evaluation
    eval_steps = min(100, len(eval_loader))
    
    # Switch to evaluation mode
    model.eval()
    
    total_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= eval_steps:
                break
                
            # Move batch to device
            batch = batch.to(args.device)
            
            # Forward pass
            outputs = model(batch, labels=batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            
            # Accumulate loss
            total_loss += loss.item()
    
    # Switch back to training mode
    model.train()
    
    # Return average loss
    return total_loss / eval_steps


def create_training_visualizations(losses, perplexities, args):
    """Create visualizations of training progress."""
    print("Creating training visualizations...")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.title(f"Training Loss (Pruning: {args.pruning_level*100}%, Opt Level: {args.optimization_level})")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot perplexity
    plt.subplot(2, 1, 2)
    plt.plot(perplexities)
    plt.title("Training Perplexity")
    plt.xlabel("Training Steps")
    plt.ylabel("Perplexity")
    plt.yscale("log")  # Log scale for perplexity
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_progress.png"))
    print(f"Saved training visualizations to {os.path.join(args.output_dir, 'training_progress.png')}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_dataset(args.dataset, tokenizer, args.max_seq_length)
    
    # Prepare model
    model = prepare_model(args)
    
    # Train model
    trained_model, losses, perplexities = train_model(model, tokenizer, dataset, args)
    
    # Evaluate final model
    print("Evaluating final model...")
    eval_loss = evaluate_model(trained_model, tokenizer, dataset, args)
    print(f"Final evaluation loss: {eval_loss:.4f}, Perplexity: {torch.exp(torch.tensor(eval_loss)).item():.2f}")
    
    print("Fine-tuning complete!")


if __name__ == "__main__":
    main()