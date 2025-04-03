#!/usr/bin/env python
"""
Training script for the Adaptive Transformer with Sentinel Gates.

This script implements the training procedure described in the paper, including:
- Loading a pretrained model and adapting it with our architecture
- Dynamic controller for adaptive head pruning
- U-Net style skip connections
- Metrics tracking and visualization
- Controller learning rate scheduling
- Early stopping based on gate activity plateaus

Key features:
1. Dynamic learning rate for controller: The controller's learning rate is automatically
   scheduled to decrease over time, stabilizing gate values in later stages of training.

2. Early stopping mechanism: Detects when gate activity plateaus across multiple updates,
   preventing oscillations and stabilizing the pruning process.

3. Head regrowth capability: The controller can reactivate previously pruned heads
   based on importance metrics, allowing for dynamic architecture adaptation.

4. U-Net skip connections: Hierarchical connections between encoder and decoder layers,
   enabling richer representations and improved adaptation.

5. Per-head learning rate adjustment: When heads are pruned or regrown, their learning
   rates are temporarily boosted to allow faster adaptation to their new role, then
   gradually decayed back to the base learning rate. This improves stability during
   architectural changes.
"""

import os
import argparse
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler
from models.loaders.loader import load_baseline_model, load_adaptive_model
from data_modules.dataset_loader import load_dataset
from utils.metrics_logger import MetricsLogger
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.progress_tracker import ProgressTracker
from utils.training import compute_loss
from utils.head_lr_manager import HeadLRManager
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
    # If verbose flag is set, it overrides quiet mode
    if args.verbose:
        quiet_mode = False
    else:
        quiet_mode = args.quiet or os.environ.get("QUIET", "0") == "1"
    model = load_adaptive_model(args.model_name, baseline_model, device, debug=debug_mode, quiet=quiet_mode)
    
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
    # Account for gradient accumulation in num_training_steps calculation
    num_training_steps = args.epochs * len(train_loader) // args.gradient_accumulation_steps
    if args.gradient_accumulation_steps > 1:
        print(f"🔄 Using gradient accumulation. Steps per update: {args.gradient_accumulation_steps}")
        print(f"🔄 Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Initialize controller manager with enhanced configuration
    controller_config = {
        "controller_type": args.controller_type,
        "update_frequency": args.controller_update_freq,
        "warmup_steps": args.controller_warmup,
        "max_pruned_heads": args.max_pruned_heads,
        "controller_config": {
            "init_value": args.gate_init_value,
            "reg_weight": args.gate_reg_weight
        },
        # Learning rate scheduling for controller
        "controller_lr": args.controller_lr,
        "controller_lr_decay": args.controller_lr_decay,
        "controller_lr_decay_steps": args.controller_lr_decay_steps,
        "min_controller_lr": args.min_controller_lr,
        # Early stopping configuration
        "enable_early_stopping": args.enable_early_stopping,
        "early_stopping_patience": args.early_stopping_patience,
        "min_gate_change": args.min_gate_change,
        # Verbosity control - if verbose flag is set, override quiet mode
        "quiet": quiet_mode
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
        eval_interval=args.eval_interval,
        quiet=quiet_mode
    )
    
    # Initialize the head learning rate manager if enabled
    head_lr_manager = None
    if args.enable_head_lr:
        if not quiet_mode:
            print(f"🔄 Initializing per-head learning rate manager")
            print(f"   - Boost factor: {args.head_lr_boost}")
            print(f"   - Warmup steps: {args.head_lr_warmup}")
            print(f"   - Cooldown steps: {args.head_lr_cooldown}")
        
        head_lr_manager = HeadLRManager(
            model=model,
            optimizer=optimizer,
            base_lr=args.learning_rate,
            boost_factor=args.head_lr_boost,
            decay_factor=args.head_lr_decay,
            warmup_steps=args.head_lr_warmup,
            cooldown_steps=args.head_lr_cooldown
        )
    
    # Load checkpoint if resuming training
    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        # Always show checkpoint loading, even in quiet mode
        print(f"📂 Resuming from checkpoint: {args.resume}")
        model, optimizer, lr_scheduler, start_epoch, global_step = load_checkpoint(
            model, optimizer, head_lr_multipliers, args.resume, device
        )
        
        # Also load controller state if available
        if os.path.exists(os.path.join(os.path.dirname(args.resume), "controller.pt")):
            controller_path = os.path.join(os.path.dirname(args.resume), "controller.pt")
            controller.load_state_dict(torch.load(controller_path, map_location=device))
            if not quiet_mode:
                print(f"📂 Loaded controller state from {controller_path}")
            
        # Load head learning rate manager state if available
        if args.enable_head_lr and os.path.exists(os.path.join(os.path.dirname(args.resume), "head_lr.pt")):
            head_lr_path = os.path.join(os.path.dirname(args.resume), "head_lr.pt")
            head_lr_manager.load_state_dict(torch.load(head_lr_path, map_location=device))
            if not quiet_mode:
                print(f"📂 Loaded head learning rate state from {head_lr_path}")
    
    # Training loop
    print(f"🏋️ Starting training for {args.epochs} epochs")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        
        # Enable U-Net connections if we've reached the configured epoch
        if epoch == unet_start_epoch:
            if not args.quiet:
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
            
            # Scale loss by gradient accumulation steps for consistent gradients
            scaled_loss = total_loss / args.gradient_accumulation_steps
            
            # Backward pass
            scaled_loss.backward()
            
            # Only update parameters after accumulating gradients for specified steps
            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1 == len(train_loader)):
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # Update parameters
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Log accumulated step
                if args.gradient_accumulation_steps > 1 and args.debug and not quiet_mode:
                    print(f"  Accumulated gradients for {args.gradient_accumulation_steps} steps, optimizer update performed")
            
            # Update global step counter - only count actual optimizer updates
            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1 == len(train_loader)):
                global_step += 1
            
                # Update controller periodically - only on actual update steps
                controller_update = controller.step(head_lr_manager=head_lr_manager)
            
            # Log metrics - only on actual update steps
            if (step + 1) % args.gradient_accumulation_steps == 0 and global_step % args.log_interval == 0:
                # Get current learning rate
                current_lr = lr_scheduler.get_last_lr()[0]
                
                # Log metrics including controller-specific information
                metrics = {
                    "loss": loss.item(),
                    "reg_loss": reg_loss.item(),
                    "total_loss": total_loss.item(),
                    "learning_rate": current_lr,
                    "epoch": epoch,
                    "pruned_percent": controller._get_pruned_percent()
                }
                
                # Add controller metrics if available from update
                if isinstance(controller_update, dict):
                    if "controller_lr" in controller_update:
                        metrics["controller_lr"] = controller_update["controller_lr"]
                    if "plateau_counter" in controller_update:
                        metrics["plateau_counter"] = controller_update["plateau_counter"]
                    if "early_stopping_triggered" in controller_update and controller_update["early_stopping_triggered"]:
                        metrics["early_stopping"] = 1.0
                    if "gate_changes" in controller_update:
                        metrics["gate_changes"] = controller_update["gate_changes"]
                    
                    # Add head learning rate info if available
                    if "head_lr_info" in controller_update and controller_update["head_lr_info"]:
                        head_lr_info = controller_update["head_lr_info"]
                        
                        if "head_status" in head_lr_info:
                            status_info = head_lr_info["head_status"]
                            metrics["newly_activated_heads"] = len(status_info.get("newly_activated", []))
                            metrics["newly_pruned_heads"] = len(status_info.get("newly_deactivated", []))
                            metrics["cooling_down_heads"] = status_info.get("cooling_down", 0)
                        
                        if "lr_updates" in head_lr_info:
                            lr_info = head_lr_info["lr_updates"]
                            metrics["max_head_lr_multiplier"] = lr_info.get("max_multiplier", 1.0)
                            metrics["avg_head_lr_multiplier"] = lr_info.get("avg_multiplier", 1.0)
                            
                            # Print verbose info if changes were made and debug mode is on
                            if args.debug and lr_info.get("changes_made", False) and not quiet_mode:
                                print(f"  📊 Head LR adjustments: avg={metrics['avg_head_lr_multiplier']:.2f}x, max={metrics['max_head_lr_multiplier']:.2f}x")
                
                metrics_logger.log_metrics(metrics, global_step)
                
                # Print progress
                progress.log_train_step(epoch, step, metrics)
            
            # Evaluate periodically - only on actual update steps
            if (step + 1) % args.gradient_accumulation_steps == 0 and global_step % args.eval_interval == 0:
                eval_metrics = evaluate(model, eval_loader, device, controller)
                metrics_logger.log_metrics(eval_metrics, global_step, prefix="eval_")
                progress.log_eval_step(epoch, step, eval_metrics)
                
                # Save checkpoint - only on actual update steps
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
                    
                    # Save head learning rate manager state if enabled
                    if args.enable_head_lr and head_lr_manager is not None:
                        head_lr_path = os.path.join(
                            args.output_dir,
                            "head_lr.pt"
                        )
                        torch.save(head_lr_manager.save_state_dict(), head_lr_path)
                    
                    # Include info about gradient accumulation in checkpoint message
                    if not quiet_mode:
                        if args.gradient_accumulation_steps > 1:
                            print(f"💾 Saved checkpoint to {checkpoint_path} (effective batch size: {args.batch_size * args.gradient_accumulation_steps})")
                        else:
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
            
            # Save head learning rate manager state if enabled
            if args.enable_head_lr and head_lr_manager is not None:
                head_lr_path = os.path.join(
                    args.output_dir,
                    f"head_lr_epoch{epoch}.pt"
                )
                torch.save(head_lr_manager.save_state_dict(), head_lr_path)
                
            if not quiet_mode:
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
    
    # Save final head learning rate manager state if enabled
    if args.enable_head_lr and head_lr_manager is not None:
        final_head_lr_path = os.path.join(args.output_dir, "final_head_lr.pt")
        torch.save(head_lr_manager.save_state_dict(), final_head_lr_path)
    
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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients before performing update")
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
    
    # Controller learning rate scheduling
    parser.add_argument("--controller_lr", type=float, default=0.01,
                        help="Initial learning rate for controller updates")
    parser.add_argument("--controller_lr_decay", type=float, default=0.95,
                        help="Decay factor for controller learning rate")
    parser.add_argument("--controller_lr_decay_steps", type=int, default=1000,
                        help="Number of steps between learning rate decays")
    parser.add_argument("--min_controller_lr", type=float, default=0.001,
                        help="Minimum controller learning rate")
    
    # Early stopping for controller
    parser.add_argument("--enable_early_stopping", action="store_true",
                        help="Enable early stopping based on gate activity plateau")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Number of updates with minimal change before stopping")
    parser.add_argument("--min_gate_change", type=float, default=0.01,
                        help="Minimum gate change to continue updates")
    
    # Per-head learning rate configuration
    parser.add_argument("--enable_head_lr", action="store_true",
                        help="Enable per-head learning rate adjustments during pruning/regrowth")
    parser.add_argument("--head_lr_boost", type=float, default=5.0,
                        help="Factor to boost learning rates for newly activated heads")
    parser.add_argument("--head_lr_decay", type=float, default=0.9,
                        help="Decay factor for head learning rate boosts")
    parser.add_argument("--head_lr_warmup", type=int, default=200,
                        help="Warmup steps for gradually increasing head learning rates")
    parser.add_argument("--head_lr_cooldown", type=int, default=1000,
                        help="Cooldown steps before returning to base learning rate")
    
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
    parser.add_argument("--quiet", action="store_true", default=True,
                        help="Reduce verbose loading and training output (enabled by default)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed loading and training output (disables --quiet)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    train(args)