#!/usr/bin/env python
"""
Training script for the Adaptive Transformer with Sentinel Gates.

This script implements the training procedure described in the paper, including:
- Loading a pretrained model and adapting it with our architecture
- Dynamic controller for adaptive head pruning
- U-Net style skip connections
- Metrics tracking and visualization
"""

import os
import argparse
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler
from models.loaders.loader import load_baseline_model, load_adaptive_model
from datasets.dataset_loader import load_dataset
from utils.metrics_logger import MetricsLogger
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.progress_tracker import ProgressTracker
from utils.training import compute_loss
from controller.controller_manager import ControllerManager

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(args):
    """Main training function"""
    # Set device
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"🚀 Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"🎲 Random seed set to: {args.seed}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load baseline model
    print(f"⚙️ Loading baseline model: {args.model_name}")
    baseline_model = load_baseline_model(args.model_name, device)
    
    # Create adaptive model
    print("⚙️ Creating adaptive transformer model")
    debug_mode = args.debug
    model = load_adaptive_model(args.model_name, baseline_model, device, debug=debug_mode)
    
    # Load dataset
    print(f"📚 Loading dataset: {args.dataset}")
    train_dataset, eval_dataset = load_dataset(args.dataset, tokenizer, args.max_length)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size
    )
    
    # Setup optimizer - different learning rates for different components
    # We use lower learning rates for pretrained components and higher
    # for new components like gates
    params = []
    
    # Gate parameters (controller parameters) get higher learning rate
    gate_params = []
    for name, param in model.named_parameters():
        if 'gate' in name:
            gate_params.append(param)
    
    # Track different parameter groups for the optimizer
    # Gate parameters get higher learning rate
    head_lr_multipliers = {
        'gate': args.gate_lr_multiplier,
        'skip_fuse': args.skip_lr_multiplier,
    }
    
    # Group parameters by their learning rate multiplier
    param_groups = {}
    for name, param in model.named_parameters():
        multiplier = 1.0  # Default multiplier
        for key, value in head_lr_multipliers.items():
            if key in name:
                multiplier = value
                break
        
        if multiplier not in param_groups:
            param_groups[multiplier] = []
        
        param_groups[multiplier].append(param)
    
    # Create optimizer parameter groups
    optimizer_param_groups = []
    for multiplier, params in param_groups.items():
        optimizer_param_groups.append({
            'params': params,
            'lr': args.learning_rate * multiplier
        })
    
    # Create optimizer
    optimizer = torch.optim.AdamW(optimizer_param_groups, lr=args.learning_rate)
    
    # Create scheduler
    num_training_steps = args.epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Initialize controller manager
    controller_config = {
        "controller_type": args.controller_type,
        "update_frequency": args.controller_update_freq,
        "warmup_steps": args.controller_warmup,
        "max_pruned_heads": args.max_pruned_heads,
        "controller_config": {
            "init_value": args.gate_init_value,
            "reg_weight": args.gate_reg_weight
        }
    }
    controller = ControllerManager(model, controller_config)
    
    # Configure when to enable U-Net connections
    unet_start_epoch = args.unet_start_epoch 
    
    # Initialize metrics logger
    metrics_logger = MetricsLogger(
        log_dir=args.output_dir,
        model_name=f"{args.model_name.split('/')[-1]}_adaptive"
    )
    
    # Initialize progress tracker
    progress = ProgressTracker(
        total_epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        log_interval=args.log_interval,
        eval_interval=args.eval_interval
    )
    
    # Load checkpoint if resuming training
    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Resuming from checkpoint: {args.resume}")
        model, optimizer, lr_scheduler, start_epoch, global_step = load_checkpoint(
            model, optimizer, head_lr_multipliers, args.resume, device
        )
        
        # Also load controller state if available
        if os.path.exists(os.path.join(os.path.dirname(args.resume), "controller.pt")):
            controller_path = os.path.join(os.path.dirname(args.resume), "controller.pt")
            controller.load_state_dict(torch.load(controller_path, map_location=device))
            print(f"📂 Loaded controller state from {controller_path}")
    
    # Training loop
    print(f"🏋️ Starting training for {args.epochs} epochs")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        
        # Enable U-Net connections if we've reached the configured epoch
        if epoch == unet_start_epoch:
            print(f"🔄 Enabling U-Net skip connections at epoch {epoch}")
            controller.enable_unet_connections(True)
        
        for step, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            
            # Compute loss
            loss = compute_loss(outputs.logits, batch["labels"])
            
            # Add regularization loss from controller
            reg_loss = controller.get_regularization_loss()
            total_loss = loss + reg_loss
            
            # Backward pass
            total_loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # Update parameters
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Update global step counter
            global_step += 1
            
            # Update controller periodically
            controller_update = controller.step()
            
            # Log metrics
            if global_step % args.log_interval == 0:
                # Get current learning rate
                current_lr = lr_scheduler.get_last_lr()[0]
                
                # Log metrics
                metrics = {
                    "loss": loss.item(),
                    "reg_loss": reg_loss.item(),
                    "total_loss": total_loss.item(),
                    "learning_rate": current_lr,
                    "epoch": epoch,
                    "pruned_percent": controller._get_pruned_percent()
                }
                
                metrics_logger.log_metrics(metrics, global_step)
                
                # Print progress
                progress.log_train_step(epoch, step, metrics)
            
            # Evaluate periodically
            if global_step % args.eval_interval == 0:
                eval_metrics = evaluate(model, eval_loader, device, controller)
                metrics_logger.log_metrics(eval_metrics, global_step, prefix="eval_")
                progress.log_eval_step(epoch, step, eval_metrics)
                
                # Save checkpoint
                if args.save_interval > 0 and global_step % args.save_interval == 0:
                    checkpoint_path = os.path.join(
                        args.output_dir, 
                        f"checkpoint_epoch{epoch}_step{global_step}.pt"
                    )
                    save_checkpoint(
                        model, optimizer, lr_scheduler, 
                        epoch, global_step, 
                        checkpoint_path
                    )
                    
                    # Also save controller state
                    controller_path = os.path.join(
                        args.output_dir, 
                        "controller.pt"
                    )
                    torch.save(controller.save_state_dict(), controller_path)
                    print(f"💾 Saved checkpoint to {checkpoint_path}")
        
        # End of epoch - save checkpoint
        if args.save_every_epoch:
            checkpoint_path = os.path.join(
                args.output_dir, 
                f"checkpoint_epoch{epoch}.pt"
            )
            save_checkpoint(
                model, optimizer, lr_scheduler, 
                epoch, global_step, 
                checkpoint_path
            )
            
            # Also save controller state
            controller_path = os.path.join(
                args.output_dir, 
                f"controller_epoch{epoch}.pt"
            )
            torch.save(controller.save_state_dict(), controller_path)
            print(f"💾 Saved epoch checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, "final_model.pt")
    save_checkpoint(
        model, optimizer, lr_scheduler, 
        args.epochs, global_step, 
        final_path
    )
    
    # Save final controller state
    final_controller_path = os.path.join(args.output_dir, "final_controller.pt")
    torch.save(controller.save_state_dict(), final_controller_path)
    
    print(f"🎉 Training completed! Final model saved to {final_path}")
    return model, tokenizer

def evaluate(model, eval_loader, device, controller=None):
    """Evaluate the model on the evaluation dataset"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            
            # Compute loss
            loss = compute_loss(outputs.logits, batch["labels"])
            
            # Accumulate loss
            batch_size = batch["input_ids"].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    # Calculate average loss
    avg_loss = total_loss / total_samples
    
    # Get controller metrics
    metrics = {"loss": avg_loss}
    if controller:
        metrics["pruned_percent"] = controller._get_pruned_percent()
        
        active_gates = controller._get_active_gates()
        # Count active heads by layer
        for layer_idx, heads in active_gates.items():
            metrics[f"active_heads_layer{layer_idx}"] = len(heads)
    
    model.train()
    return metrics

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Adaptive Transformer with Sentinel Gates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="HuggingFace model name to use as base")
    parser.add_argument("--dataset", type=str, default="wikitext",
                        help="Dataset to use for training (wikitext|c4|custom)")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--gate_lr_multiplier", type=float, default=10.0,
                        help="Multiplier for gate parameter learning rate")
    parser.add_argument("--skip_lr_multiplier", type=float, default=5.0,
                        help="Multiplier for skip connection parameter learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--lr_scheduler", type=str, default="linear",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant"],
                        help="Learning rate scheduler")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of warmup steps for the scheduler")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    
    # Controller parameters
    parser.add_argument("--controller_type", type=str, default="ann",
                        help="Type of controller to use (ann)")
    parser.add_argument("--controller_update_freq", type=int, default=100,
                        help="How often to update the controller (steps)")
    parser.add_argument("--controller_warmup", type=int, default=1000,
                        help="Warmup steps before controller starts pruning")
    parser.add_argument("--gate_init_value", type=float, default=3.0,
                        help="Initial value for gate logits")
    parser.add_argument("--gate_reg_weight", type=float, default=1e-4,
                        help="Weight for L1 regularization of gates")
    parser.add_argument("--max_pruned_heads", type=float, default=0.3,
                        help="Maximum fraction of heads to prune")
    
    # U-Net configuration
    parser.add_argument("--unet_start_epoch", type=int, default=1,
                        help="Epoch to start using U-Net skip connections")
    
    # Logging and saving
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log metrics every N steps")
    parser.add_argument("--eval_interval", type=int, default=500,
                        help="Evaluate every N steps")
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="Save checkpoint every N steps (0 to disable)")
    parser.add_argument("--save_every_epoch", action="store_true",
                        help="Save checkpoint after each epoch")
    
    # Resuming training
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None,
                        help="Device to use (default: auto-detect)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    train(args)