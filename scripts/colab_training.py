#!/usr/bin/env python
"""
Enhanced training script for Google Colab with Google Drive integration.

This script provides a comprehensive training setup for Sentinel-AI in Colab:
1. Google Drive integration for saving/loading models
2. Support for pruning and progressive learning
3. Visualization of training progress and model behavior
4. Simplified interface for beginners

Usage in Colab:
```python
!git clone https://github.com/CambrianTech/sentinel-ai.git
%cd sentinel-ai
!pip install -r requirements.txt
from google.colab import drive
drive.mount('/content/drive')
!python scripts/colab_training.py --drive_path /content/drive/MyDrive/sentinel-ai
```
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.loaders.loader import load_baseline_model, load_adaptive_model
from data_modules.dataset_loader import load_dataset
from utils.model_wrapper import wrap_model_for_generation
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.generation_wrapper import generate_text
from controller.controller_manager import ControllerManager
from controller.metrics.head_metrics import collect_head_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Sentinel-AI Colab Training Script")
    
    # Model and dataset configuration
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                        help="Model name or path (e.g., distilgpt2, gpt2)")
    parser.add_argument("--dataset", type=str, default="tiny_shakespeare",
                        help="Dataset to use (tiny_shakespeare, wikitext, tiny_stories)")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps for LR scheduler")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for optimizer")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients")
    
    # Controller parameters
    parser.add_argument("--enable_controller", action="store_true",
                        help="Enable the controller for dynamic pruning")
    parser.add_argument("--controller_interval", type=int, default=100,
                        help="Number of steps between controller updates")
    parser.add_argument("--controller_lr", type=float, default=0.01,
                        help="Learning rate for the controller")
    parser.add_argument("--target_pruning", type=float, default=0.3,
                        help="Target pruning level for the controller")
    
    # Pruning parameters
    parser.add_argument("--initial_pruning", type=float, default=0.0,
                        help="Initial pruning level (0.0-1.0)")
    parser.add_argument("--pruning_strategy", type=str, default="none",
                        choices=["none", "random", "entropy", "gradient"],
                        help="Pruning strategy to use for initial pruning")
    
    # Checkpoint configuration
    parser.add_argument("--save_every", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--drive_path", type=str, default="",
                        help="Google Drive path for saving/loading checkpoints")
    parser.add_argument("--resume_checkpoint", type=str, default="",
                        help="Path to checkpoint for resuming training")
    
    # Evaluation parameters
    parser.add_argument("--eval_every", type=int, default=100,
                        help="Evaluate every N steps")
    parser.add_argument("--eval_samples", type=int, default=3,
                        help="Number of samples to generate during evaluation")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--save_results", action="store_true",
                        help="Save metrics and images to disk")
    
    return parser.parse_args()

def apply_initial_pruning(model, strategy, pruning_level, device):
    """Apply initial pruning to the model before training."""
    if pruning_level <= 0 or strategy == "none":
        print("No initial pruning applied")
        return model
        
    print(f"Applying initial {strategy} pruning at {pruning_level:.1%} level")
    
    # Get model dimensions
    num_layers = len(model.blocks)
    num_heads = model.blocks[0]["attn"].num_heads
    total_heads = num_layers * num_heads
    heads_to_prune = int(total_heads * pruning_level)
    
    # Create dummy input for collecting metrics
    batch_size = 2
    seq_len = 32
    dummy_input = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    dummy_batch = {"input_ids": dummy_input, 
                   "attention_mask": torch.ones_like(dummy_input)}
    
    # Apply pruning based on strategy
    if strategy == "random":
        # Get a flattened list of (layer, head) tuples
        all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]
        
        # Randomly select heads to prune
        pruned_head_indices = np.random.choice(len(all_heads), heads_to_prune, replace=False)
        
        # Set gates to near-zero for pruned heads
        with torch.no_grad():
            for idx in pruned_head_indices:
                layer_idx, head_idx = all_heads[idx]
                model.blocks[layer_idx]["attn"].gate[head_idx] = torch.tensor(0.001, device=device)
    
    elif strategy in ["entropy", "gradient"]:
        # Collect metrics
        metrics = collect_head_metrics(model, batch=dummy_batch)
        
        if strategy == "entropy" and "entropy" in metrics:
            head_scores = metrics["entropy"]
            # Higher entropy = less focused attention = more likely to be pruned
            descending = True
        elif strategy == "gradient" and "grad_norm" in metrics:
            head_scores = metrics["grad_norm"]
            # Lower gradient norm = less important head = more likely to be pruned
            descending = False
        else:
            print(f"Warning: {strategy} metrics not available, using random pruning")
            return apply_initial_pruning(model, "random", pruning_level, device)
        
        # Reshape and flatten scores
        if not isinstance(head_scores, torch.Tensor):
            head_scores = torch.tensor(head_scores, device=device)
            
        if len(head_scores.shape) < 2:
            head_scores = head_scores.reshape(num_layers, num_heads)
            
        flat_scores = head_scores.view(-1)
        
        # Sort scores
        _, indices = torch.sort(flat_scores, descending=descending)
        indices_to_prune = indices[:heads_to_prune]
        
        # Apply pruning
        with torch.no_grad():
            for idx in indices_to_prune:
                layer_idx = idx.item() // num_heads
                head_idx = idx.item() % num_heads
                model.blocks[layer_idx]["attn"].gate[head_idx] = torch.tensor(0.001, device=device)
    
    # Count pruned heads for verification
    pruned_count = 0
    with torch.no_grad():
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                if model.blocks[layer_idx]["attn"].gate[head_idx].item() < 0.01:
                    pruned_count += 1
    
    print(f"Pruned {pruned_count} of {total_heads} heads ({pruned_count/total_heads:.1%})")
    return model

def get_save_path(args, step=None):
    """Get the path for saving checkpoints, supporting Google Drive."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Base directory for saving
    if args.drive_path:
        base_dir = os.path.join(args.drive_path, "checkpoints")
    else:
        base_dir = os.path.join("checkpoints")
    
    # Create model-specific subdirectory
    model_dir = os.path.join(base_dir, f"{args.model_name}_{args.dataset}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Checkpoint path
    if step is not None:
        checkpoint_path = os.path.join(model_dir, f"checkpoint_step{step}_{timestamp}.pt")
    else:
        checkpoint_path = os.path.join(model_dir, f"checkpoint_final_{timestamp}.pt")
    
    return checkpoint_path

def setup_visualization(args):
    """Set up visualization directories and return paths."""
    if args.drive_path and args.save_results:
        viz_dir = os.path.join(args.drive_path, "visualizations")
    elif args.save_results:
        viz_dir = "visualizations"
    else:
        viz_dir = None
        
    if viz_dir:
        os.makedirs(viz_dir, exist_ok=True)
        metrics_path = os.path.join(viz_dir, "training_metrics.png")
        gates_path = os.path.join(viz_dir, "gate_activity.png")
        return viz_dir, metrics_path, gates_path
    
    return None, None, None

def visualize_gates(model, path=None):
    """Visualize gate activity across layers and heads."""
    num_layers = len(model.blocks)
    num_heads = model.blocks[0]["attn"].num_heads
    
    # Create matrix of gate values
    gate_values = torch.zeros(num_layers, num_heads)
    for l in range(num_layers):
        for h in range(num_heads):
            gate_values[l, h] = model.blocks[l]["attn"].gate[h].item()
    
    # Create heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(gate_values.numpy(), cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(label="Gate Value")
    plt.xlabel("Attention Head")
    plt.ylabel("Transformer Layer")
    plt.title("Attention Head Gate Activity")
    
    # Add grid lines
    plt.grid(False)
    plt.xticks(range(num_heads))
    plt.yticks(range(num_layers))
    
    if path:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    
    return gate_values.numpy()

def visualize_metrics(metrics, path=None):
    """Plot training metrics."""
    steps = list(range(len(metrics["loss"])))
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot loss
    axs[0].plot(steps, metrics["loss"], label="Training")
    if "eval_loss" in metrics and metrics["eval_loss"]:
        eval_steps = [s for i, s in enumerate(steps) if i % (len(steps) // len(metrics["eval_loss"])) == 0][:len(metrics["eval_loss"])]
        axs[0].plot(eval_steps, metrics["eval_loss"], label="Evaluation", linestyle="--")
    axs[0].set_xlabel("Steps")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Training and Evaluation Loss")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Plot active heads
    if "active_heads" in metrics and metrics["active_heads"]:
        axs[1].plot(steps, metrics["active_heads"], label="Active Heads")
        axs[1].set_xlabel("Steps")
        axs[1].set_ylabel("Number of Active Heads")
        axs[1].set_title("Active Attention Heads During Training")
        axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if path:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def train_adaptive_model(args):
    """Main training function for the adaptive transformer."""
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    baseline_model = load_baseline_model(args.model_name, device)
    model = load_adaptive_model(args.model_name, baseline_model, device)
    
    # Apply initial pruning if specified
    if args.initial_pruning > 0 and args.pruning_strategy != "none":
        model = apply_initial_pruning(
            model, args.pruning_strategy, args.initial_pruning, device
        )
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    train_dataset, eval_dataset = load_dataset(
        args.dataset, tokenizer, args.max_length
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    # Setup controller if enabled
    if args.enable_controller:
        print("Initializing controller...")
        controller_manager = ControllerManager(
            model=model,
            target_sparsity=args.target_pruning,
            learning_rate=args.controller_lr
        )
    else:
        controller_manager = None
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Calculate total training steps
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    
    # Create learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    
    if args.resume_checkpoint:
        checkpoint_path = args.resume_checkpoint
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = load_checkpoint(checkpoint_path, model, optimizer)
            start_epoch = checkpoint.get("epoch", 0)
            global_step = checkpoint.get("global_step", 0)
            print(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    # Setup visualization
    viz_dir, metrics_path, gates_path = setup_visualization(args)
    
    # Initialize metrics storage
    metrics = {
        "loss": [],
        "eval_loss": [],
        "active_heads": [],
        "perplexity": []
    }
    
    # Main training loop
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        
        # Training loop
        epoch_loss = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            # Skip steps for resuming
            if global_step < step:
                continue
                
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            # Calculate loss
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Record metrics
            epoch_loss += loss.item() * args.gradient_accumulation_steps
            metrics["loss"].append(loss.item() * args.gradient_accumulation_steps)
            
            # Count active heads
            num_layers = len(model.blocks)
            num_heads = model.blocks[0]["attn"].num_heads
            active_heads = sum(
                1 for l in range(num_layers) for h in range(num_heads) 
                if model.blocks[l]["attn"].gate[h].item() > 0.01
            )
            metrics["active_heads"].append(active_heads)
            
            # Optimizer step with gradient accumulation
            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1 == len(train_loader)):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # Update model parameters
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update controller if enabled
                if controller_manager and global_step % args.controller_interval == 0:
                    print(f"\nStep {global_step}: Updating controller...")
                    
                    # Get batch for metrics calculation
                    eval_batch = next(iter(eval_loader))
                    eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                    
                    # Calculate metrics for controller
                    with torch.no_grad():
                        eval_outputs = model(
                            input_ids=eval_batch["input_ids"],
                            attention_mask=eval_batch["attention_mask"],
                            labels=eval_batch["labels"]
                        )
                        eval_loss = eval_outputs.loss.item()
                    
                    # Update controller based on metrics
                    controller_metrics = {
                        "loss": eval_loss,
                        "perplexity": torch.exp(torch.tensor(eval_loss)).item()
                    }
                    
                    controller_manager.step(
                        metrics_dict=controller_metrics,
                        dataloader=eval_loader
                    )
                    
                    # Visualize gate activity
                    if viz_dir:
                        gate_viz_path = os.path.join(viz_dir, f"gates_step{global_step}.png")
                        visualize_gates(model, gate_viz_path)
                
                # Evaluation
                if global_step % args.eval_every == 0:
                    print(f"\nStep {global_step}: Running evaluation...")
                    model.eval()
                    
                    # Calculate evaluation loss
                    eval_loss = 0
                    eval_steps = 0
                    
                    with torch.no_grad():
                        for eval_batch in eval_loader:
                            eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                            eval_outputs = model(
                                input_ids=eval_batch["input_ids"],
                                attention_mask=eval_batch["attention_mask"],
                                labels=eval_batch["labels"]
                            )
                            eval_loss += eval_outputs.loss.item()
                            eval_steps += 1
                            
                            # Limit evaluation steps for speed
                            if eval_steps >= 10:
                                break
                    
                    avg_eval_loss = eval_loss / eval_steps
                    metrics["eval_loss"].append(avg_eval_loss)
                    perplexity = torch.exp(torch.tensor(avg_eval_loss)).item()
                    metrics["perplexity"].append(perplexity)
                    
                    print(f"Evaluation loss: {avg_eval_loss:.4f}, Perplexity: {perplexity:.2f}")
                    
                    # Generate sample text
                    model.eval()
                    for i in range(min(args.eval_samples, 2)):
                        sample_text = train_dataset[np.random.randint(0, len(train_dataset))]["input_ids"]
                        sample_text = tokenizer.decode(sample_text[:20])
                        
                        generated_text = generate_text(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=sample_text,
                            max_length=50,
                            temperature=0.7,
                            device=device
                        )
                        
                        print(f"\nSample {i+1}:")
                        print(f"Prompt: {sample_text}")
                        print(f"Generated: {generated_text}")
                    
                    # Switch back to train mode
                    model.train()
                
                # Save checkpoint
                if args.save_every > 0 and global_step % args.save_every == 0:
                    checkpoint_path = get_save_path(args, global_step)
                    print(f"\nSaving checkpoint to {checkpoint_path}")
                    
                    save_checkpoint(
                        checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        head_lr_multipliers=None,  # Not used in this setup
                        epoch=epoch,
                        step=global_step
                    )
                    
                    # Visualize metrics
                    if metrics_path:
                        visualize_metrics(metrics, metrics_path)
            
            global_step += 1
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} completed, avg loss: {avg_epoch_loss:.4f}")
    
    # End of training
    print("Training completed!")
    
    # Final save
    final_checkpoint_path = get_save_path(args)
    print(f"Saving final model to {final_checkpoint_path}")
    save_checkpoint(
        final_checkpoint_path,
        model=model,
        optimizer=optimizer,
        head_lr_multipliers=None,
        epoch=args.epochs,
        step=global_step
    )
    
    # Final visualizations
    if viz_dir:
        # Save gate activity visualization
        visualize_gates(model, gates_path)
        
        # Save metrics visualization
        visualize_metrics(metrics, metrics_path)
    
    return model, tokenizer, metrics

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Print configuration
    print("\n=== Sentinel-AI Colab Training ===")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Controller enabled: {args.enable_controller}")
    print(f"Initial pruning: {args.initial_pruning:.1%} ({args.pruning_strategy})")
    print(f"Drive path: {args.drive_path if args.drive_path else 'Not using Google Drive'}")
    
    # Check if Drive path exists when specified
    if args.drive_path and not os.path.exists(args.drive_path):
        print(f"WARNING: Drive path {args.drive_path} does not exist.")
        print("Make sure you've mounted Google Drive with drive.mount('/content/drive')")
        return
    
    # Run training
    try:
        model, tokenizer, metrics = train_adaptive_model(args)
        
        # Save final metrics
        if args.save_results:
            save_dir = args.drive_path if args.drive_path else "."
            metrics_file = os.path.join(save_dir, "final_metrics.json")
            import json
            
            # Convert metrics to serializable format
            serializable_metrics = {
                "loss": [float(x) for x in metrics["loss"]],
                "eval_loss": [float(x) for x in metrics["eval_loss"]],
                "active_heads": [int(x) for x in metrics["active_heads"]],
                "perplexity": [float(x) for x in metrics["perplexity"]]
            }
            
            with open(metrics_file, "w") as f:
                json.dump(serializable_metrics, f, indent=2)
        
        # Return model for interactive use in notebooks
        return model, tokenizer
    
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()