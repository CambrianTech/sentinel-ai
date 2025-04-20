"""
Neural Plasticity Training

This module provides training utilities for neural plasticity experiments,
including differential learning rates for pruned vs. unpruned heads,
specialized optimizers, and training loops with plasticity.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from tqdm import tqdm
import time
import gc

import os
import traceback
from .core import (
    calculate_head_entropy,
    calculate_head_gradients,
    generate_pruning_mask,
    apply_pruning_mask,
    evaluate_model,
    IS_APPLE_SILICON
)


def create_warmup_visualization(
    warmup_losses: List[float],
    smoothed_losses: List[float],
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create visualization for warmup training.
    
    Args:
        warmup_losses: List of raw loss values
        smoothed_losses: List of smoothed (rolling average) loss values
        figsize: Figure size as (width, height) in inches
        
    Returns:
        Matplotlib figure with loss plots
    """
    # Create loss visualization
    fig, axs = plt.subplots(2, 1, figsize=figsize)
    
    # Raw loss
    axs[0].plot(warmup_losses)
    axs[0].set_title("Warm-up Loss (Raw)")
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Loss")
    axs[0].grid(True)
    
    # Smoothed loss if we have enough data
    if len(smoothed_losses) > 1:
        x = range(0, len(smoothed_losses)*5, 5)
        axs[1].plot(x, smoothed_losses)
        axs[1].set_title("Warm-up Loss (5-step Rolling Average)")
        axs[1].set_xlabel("Step")
        axs[1].set_ylabel("Loss")
        axs[1].grid(True)
        
        # Add trend line to smoothed plot
        try:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(x, smoothed_losses)
            axs[1].plot(x, [slope*xi + intercept for xi in x], 'r--', 
                     label=f'Trend: slope={slope:.6f}, R²={r_value**2:.2f}')
            axs[1].legend()
        except (ImportError, ValueError) as e:
            # Skip the trend line if an error occurs
            print(f"Could not add trend line: {e}")
    
    plt.tight_layout()
    return fig


def analyze_warmup_segments(
    warmup_losses: List[float],
    verbose: bool = True,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze training segments to assess improvement during warmup.
    
    Args:
        warmup_losses: List of loss values during warmup
        verbose: Whether to print analysis results
        save_path: Optional path to save the analysis results
        
    Returns:
        Dictionary with segment analysis metrics
    """
    # Initialize default values
    first_avg = 0
    last_avg = 0
    improvement = 0
    still_improving = False
    segment_size = 0
    
    # Only perform analysis if we have enough data points
    if len(warmup_losses) > 6:
        segment_size = len(warmup_losses) // 3
        first_segment = warmup_losses[:segment_size]
        last_segment = warmup_losses[-segment_size:]
        first_avg = sum(first_segment) / len(first_segment)
        last_avg = sum(last_segment) / len(last_segment)
        improvement = (1 - last_avg/first_avg) * 100
        
        # Calculate if still improving significantly
        still_improving = (first_avg - last_avg) / first_avg > 0.01  # More than 1% improvement
        
        # Generate and print analysis
        if verbose:
            report = [
                "\nWarm-up Segment Analysis:",
                f"First {segment_size} steps average loss: {first_avg:.4f}",
                f"Last {segment_size} steps average loss: {last_avg:.4f}",
                f"Improvement during warm-up: {improvement:.1f}%",
                f"Is model still significantly improving? {'Yes' if still_improving else 'No'}"
            ]
            
            # Print to console
            for line in report:
                print(line)
                
            # Save to file if path provided
            if save_path:
                try:
                    # Create parent directory if needed
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    
                    # Write analysis to file
                    with open(save_path, 'w') as f:
                        f.write('\n'.join(report))
                        
                    if verbose:
                        print(f"✅ Saved segment analysis to {save_path}")
                except Exception as e:
                    if verbose:
                        print(f"❌ Error saving segment analysis: {e}")
    
    # Return analysis dictionary
    return {
        "first_segment_avg": first_avg,
        "last_segment_avg": last_avg,
        "improvement": improvement,
        "still_improving": still_improving,
        "segment_size": segment_size
    }


class PlasticityTrainer:
    """
    Trainer for neural plasticity experiments with differential learning rates
    and specialized training dynamics for pruned models.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 5e-5,
        use_differential_lr: bool = True,
        pruned_head_lr_multiplier: float = 3.0
    ):
        """
        Initialize the plasticity trainer.
        
        Args:
            model: The model to train
            learning_rate: Base learning rate
            use_differential_lr: Whether to use different learning rates for pruned layers
            pruned_head_lr_multiplier: Learning rate multiplier for pruned heads
        """
        self.model = model
        self.base_lr = learning_rate
        self.use_differential_lr = use_differential_lr
        self.pruned_head_lr_multiplier = pruned_head_lr_multiplier
        self.device = next(model.parameters()).device
        self.pruned_heads = []
        self.optimizer = None
        self.scheduler = None
        
    def prepare_optimizer(
        self,
        pruned_heads: List[Tuple[int, int]],
        warmup_steps: int = 500,
        total_steps: int = 1000
    ):
        """
        Create optimizer with layer-specific learning rates.
        
        Args:
            pruned_heads: List of (layer_idx, head_idx) tuples of pruned heads
            warmup_steps: Number of warmup steps for learning rate scheduler
            total_steps: Total number of training steps
        """
        self.pruned_heads = pruned_heads
        
        if not self.use_differential_lr:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=self.base_lr * 0.1
            )
            return
        
        # Extract pruned layers
        pruned_layers = set(layer_idx for layer_idx, _ in pruned_heads)
        
        # Access model blocks based on architecture
        blocks = None
        if hasattr(self.model, 'blocks'):
            blocks = self.model.blocks
        else:
            # Try to access through transformer attribute
            transformer = None
            if hasattr(self.model, 'transformer'):
                transformer = self.model.transformer
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'transformer'):
                transformer = self.model.model.transformer
            elif hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'transformer'):
                transformer = self.model.base_model.transformer
                
            if transformer and hasattr(transformer, 'h'):
                blocks = transformer.h
            elif transformer and hasattr(transformer, 'layer'):
                blocks = transformer.layer
        
        if blocks is None:
            print("Could not identify model blocks, using default learning rate")
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.base_lr)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=self.base_lr * 0.1
            )
            return
        
        # Group parameters with different learning rates
        pruned_layer_params = []
        other_params = []
        
        for layer_idx, layer in enumerate(blocks):
            # Check if this layer had pruned heads
            layer_pruned = layer_idx in pruned_layers
            
            for name, param in layer.named_parameters():
                if layer_pruned and 'attn' in name:
                    # Higher learning rate for attention in pruned layers
                    pruned_layer_params.append(param)
                else:
                    other_params.append(param)
        
        # Add any parameters not in the blocks
        for name, param in self.model.named_parameters():
            if not any(param is p for p in pruned_layer_params + other_params):
                other_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': pruned_layer_params, 'lr': self.base_lr * self.pruned_head_lr_multiplier},
            {'params': other_params, 'lr': self.base_lr}
        ]
        
        print(f"Using adaptive learning rates: {self.base_lr * self.pruned_head_lr_multiplier} for pruned layers, {self.base_lr} for others")
        
        self.optimizer = torch.optim.AdamW(param_groups)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=self.base_lr * 0.1
        )
    
    def train_step(self, batch):
        """
        Perform a single training step.
        
        Args:
            batch: Training batch data
            
        Returns:
            Loss value
        """
        self.model.train()
        
        # Move batch to device
        if isinstance(batch, dict):
            inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        else:
            inputs = {"input_ids": batch[0].to(self.device)}
            if len(batch) > 1:
                inputs["attention_mask"] = batch[1].to(self.device)
            if len(batch) > 2:
                inputs["labels"] = batch[2].to(self.device)
        
        # Forward pass
        outputs = self.model(**inputs)
        
        # Get loss
        if hasattr(outputs, "loss"):
            loss = outputs.loss
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
            
            if "labels" in inputs:
                labels = inputs["labels"]
            else:
                # For causal language modeling, shift labels
                labels = inputs["input_ids"].clone()
                labels = labels[:, 1:].contiguous()
                logits = outputs.logits[:, :-1, :].contiguous()
            
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def train(
        self,
        train_dataloader,
        eval_dataloader,
        steps: int,
        eval_interval: int = 50,
        callback: Optional[Callable[[int, Dict[str, float]], None]] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_dataloader: DataLoader for training data
            eval_dataloader: DataLoader for evaluation data
            steps: Number of training steps
            eval_interval: Interval for evaluation and callback
            callback: Optional callback function called after each evaluation
            
        Returns:
            Dictionary with training metrics history
        """
        if self.optimizer is None:
            self.prepare_optimizer(
                pruned_heads=self.pruned_heads,
                warmup_steps=steps // 10,
                total_steps=steps
            )
        
        # Initialize metrics tracking
        metrics_history = {
            "train_loss": [],
            "eval_loss": [],
            "perplexity": [],
            "step": []
        }
        
        # Training loop
        self.model.train()
        global_step = 0
        train_iterator = iter(train_dataloader)
        progress_bar = tqdm(total=steps, desc="Training")
        
        while global_step < steps:
            # Get next batch (with cycling)
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                batch = next(train_iterator)
            
            # Train step
            loss = self.train_step(batch)
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{loss:.4f}", lr=f"{self.scheduler.get_last_lr()[0]:.2e}")
            
            # Evaluate and log metrics
            if global_step % eval_interval == 0 or global_step == steps - 1:
                # Switch to eval mode temporarily
                self.model.eval()
                
                # Evaluate
                eval_metrics = evaluate_model(self.model, eval_dataloader)
                
                # Update metrics history
                metrics_history["train_loss"].append(loss)
                metrics_history["eval_loss"].append(eval_metrics["loss"])
                metrics_history["perplexity"].append(eval_metrics["perplexity"])
                metrics_history["step"].append(global_step)
                
                # Call callback if provided
                if callback:
                    callback(global_step, {
                        "train_loss": loss,
                        "eval_loss": eval_metrics["loss"],
                        "perplexity": eval_metrics["perplexity"]
                    })
                
                # Switch back to train mode
                self.model.train()
            
            global_step += 1
        
        progress_bar.close()
        return metrics_history


def create_plasticity_trainer(
    model: torch.nn.Module,
    learning_rate: float = 5e-5,
    use_differential_lr: bool = True,
    pruned_head_lr_multiplier: float = 3.0
) -> PlasticityTrainer:
    """
    Create a trainer for neural plasticity experiments.
    
    Args:
        model: The model to train
        learning_rate: Base learning rate
        use_differential_lr: Whether to use different learning rates for pruned layers
        pruned_head_lr_multiplier: Learning rate multiplier for pruned heads
        
    Returns:
        Configured PlasticityTrainer
    """
    return PlasticityTrainer(
        model=model,
        learning_rate=learning_rate,
        use_differential_lr=use_differential_lr,
        pruned_head_lr_multiplier=pruned_head_lr_multiplier
    )


def get_plasticity_optimizer(
    model: torch.nn.Module,
    pruned_heads: List[Tuple[int, int]],
    learning_rate: float = 5e-5,
    use_differential_lr: bool = True,
    pruned_head_lr_multiplier: float = 3.0
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """
    Create optimizer and scheduler with differential learning rates.
    
    Args:
        model: The model to optimize
        pruned_heads: List of (layer_idx, head_idx) tuples of pruned heads
        learning_rate: Base learning rate
        use_differential_lr: Whether to use different learning rates for pruned layers
        pruned_head_lr_multiplier: Learning rate multiplier for pruned heads
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    device = next(model.parameters()).device
    
    if not use_differential_lr:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1000, eta_min=learning_rate * 0.1
        )
        return optimizer, scheduler
    
    # Extract pruned layers
    pruned_layers = set(layer_idx for layer_idx, _ in pruned_heads)
    
    # Access model blocks based on architecture
    blocks = None
    if hasattr(model, 'blocks'):
        blocks = model.blocks
    else:
        # Try to access through transformer attribute
        transformer = None
        if hasattr(model, 'transformer'):
            transformer = model.transformer
        elif hasattr(model, 'model') and hasattr(model.model, 'transformer'):
            transformer = model.model.transformer
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'transformer'):
            transformer = model.base_model.transformer
            
        if transformer and hasattr(transformer, 'h'):
            blocks = transformer.h
        elif transformer and hasattr(transformer, 'layer'):
            blocks = transformer.layer
    
    if blocks is None:
        print("Could not identify model blocks, using default learning rate")
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1000, eta_min=learning_rate * 0.1
        )
        return optimizer, scheduler
    
    # Group parameters with different learning rates
    pruned_layer_params = []
    other_params = []
    
    for layer_idx, layer in enumerate(blocks):
        # Check if this layer had pruned heads
        layer_pruned = layer_idx in pruned_layers
        
        for name, param in layer.named_parameters():
            if layer_pruned and 'attn' in name:
                # Higher learning rate for attention in pruned layers
                pruned_layer_params.append(param)
            else:
                other_params.append(param)
    
    # Add any parameters not in the blocks
    for name, param in model.named_parameters():
        if not any(param is p for p in pruned_layer_params + other_params):
            other_params.append(param)
    
    # Create parameter groups with different learning rates
    param_groups = [
        {'params': pruned_layer_params, 'lr': learning_rate * pruned_head_lr_multiplier},
        {'params': other_params, 'lr': learning_rate}
    ]
    
    print(f"Using adaptive learning rates: {learning_rate * pruned_head_lr_multiplier} for pruned layers, {learning_rate} for others")
    
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1000, eta_min=learning_rate * 0.1
    )
    
    return optimizer, scheduler


def run_plasticity_loop(
    model: torch.nn.Module,
    train_dataloader,
    eval_dataloader,
    pruning_level: float = 0.2,
    strategy: str = "gradient",
    learning_rate: float = 5e-5,
    training_steps: int = 500,
    use_differential_lr: bool = True,
    callback: Optional[Callable[[str, int, Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Run complete neural plasticity loop: Prune → Train → Evaluate.
    
    Args:
        model: The model to train
        train_dataloader: DataLoader for training data
        eval_dataloader: DataLoader for evaluation data
        pruning_level: Fraction of heads to prune
        strategy: Pruning strategy - "gradient", "entropy", "random", or "combined"
        learning_rate: Base learning rate for training
        training_steps: Number of training steps
        use_differential_lr: Whether to use different learning rates for pruned layers
        callback: Optional callback function for monitoring progress
        
    Returns:
        Dictionary with experiment results
    """
    device = next(model.parameters()).device
    
    # 1. Initial evaluation
    print("Evaluating baseline model...")
    baseline_metrics = evaluate_model(model, eval_dataloader)
    print(f"Baseline evaluation: Loss = {baseline_metrics['loss']:.4f}, Perplexity = {baseline_metrics['perplexity']:.2f}")
    
    if callback:
        callback("baseline", 0, baseline_metrics)
    
    # 2. Calculate metrics for pruning
    print(f"Calculating metrics for {strategy} pruning...")
    
    # Calculate gradient norms
    grad_norm_values = calculate_head_gradients(model, eval_dataloader)
    
    # Calculate entropy if needed
    entropy_values = None
    if strategy in ["entropy", "combined"]:
        # Get attention distributions from model
        with torch.no_grad():
            batch = next(iter(eval_dataloader))
            
            # Move batch to device
            if isinstance(batch, dict):
                inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            else:
                inputs = {"input_ids": batch[0].to(device)}
                if len(batch) > 1:
                    inputs["attention_mask"] = batch[1].to(device)
            
            # Forward pass with attention outputs
            outputs = model(**inputs, output_attentions=True)
            
            # Extract attention maps
            if hasattr(outputs, 'attentions') and outputs.attentions:
                # Calculate entropy for each layer
                entropy_values = torch.stack([
                    calculate_head_entropy(layer_attn) 
                    for layer_attn in outputs.attentions
                ])
    
    # 3. Generate pruning mask
    print(f"Generating pruning mask with {strategy} strategy at level {pruning_level}...")
    pruning_mask = generate_pruning_mask(
        grad_norm_values=grad_norm_values,
        prune_percent=pruning_level,
        strategy=strategy,
        entropy_values=entropy_values
    )
    
    # 4. Apply pruning
    print("Applying pruning...")
    pruned_heads = apply_pruning_mask(model, pruning_mask)
    print(f"Pruned {len(pruned_heads)} heads")
    
    if callback:
        callback("pruning", 0, {
            "grad_norm_values": grad_norm_values,
            "entropy_values": entropy_values,
            "pruning_mask": pruning_mask,
            "pruned_heads": pruned_heads
        })
    
    # 5. Post-pruning evaluation
    print("Evaluating pruned model...")
    pruned_metrics = evaluate_model(model, eval_dataloader)
    print(f"Pruned model evaluation: Loss = {pruned_metrics['loss']:.4f}, Perplexity = {pruned_metrics['perplexity']:.2f}")
    
    if callback:
        callback("post_pruning", 0, pruned_metrics)
    
    # 6. Train pruned model
    print(f"Training pruned model for {training_steps} steps...")
    trainer = PlasticityTrainer(
        model=model,
        learning_rate=learning_rate,
        use_differential_lr=use_differential_lr
    )
    
    trainer.prepare_optimizer(
        pruned_heads=pruned_heads,
        warmup_steps=training_steps // 10,
        total_steps=training_steps
    )
    
    # Define callback that forwards to the main callback
    def train_callback(step, metrics):
        if callback:
            callback("training", step, metrics)
    
    training_metrics = trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        steps=training_steps,
        eval_interval=max(1, training_steps // 10),
        callback=train_callback
    )
    
    # 7. Final evaluation
    print("Performing final evaluation...")
    final_metrics = evaluate_model(model, eval_dataloader)
    print(f"Final evaluation: Loss = {final_metrics['loss']:.4f}, Perplexity = {final_metrics['perplexity']:.2f}")
    
    if callback:
        callback("final", training_steps, final_metrics)
    
    # 8. Calculate improvement metrics
    perplexity_improvement = (baseline_metrics["perplexity"] - final_metrics["perplexity"]) / baseline_metrics["perplexity"]
    recovery_rate = 0.0
    if pruned_metrics["perplexity"] > baseline_metrics["perplexity"]:
        # Model got worse after pruning, then recovered
        pruning_degradation = pruned_metrics["perplexity"] - baseline_metrics["perplexity"]
        recovery_amount = pruned_metrics["perplexity"] - final_metrics["perplexity"]
        recovery_rate = recovery_amount / pruning_degradation if pruning_degradation > 0 else 0.0
    
    print(f"Perplexity improvement: {perplexity_improvement*100:.2f}%")
    print(f"Recovery rate: {recovery_rate*100:.2f}%")
    
    # 9. Return results
    return {
        "baseline_metrics": baseline_metrics,
        "pruned_metrics": pruned_metrics,
        "final_metrics": final_metrics,
        "training_metrics": training_metrics,
        "pruned_heads": pruned_heads,
        "perplexity_improvement": perplexity_improvement,
        "recovery_rate": recovery_rate,
        "grad_norm_values": grad_norm_values.detach().cpu(),
        "entropy_values": entropy_values.detach().cpu() if entropy_values is not None else None,
        "pruning_mask": pruning_mask.detach().cpu(),
        "strategy": strategy,
        "pruning_level": pruning_level
    }


def train_with_plasticity(
    model: torch.nn.Module,
    train_dataloader,
    eval_dataloader,
    pruned_heads: List[Tuple[int, int]],
    learning_rate: float = 5e-5,
    training_steps: int = 500,
    use_differential_lr: bool = True,
    eval_interval: int = 50,
    callback: Optional[Callable[[int, Dict[str, float]], None]] = None
) -> Dict[str, List[float]]:
    """
    Train a model with neural plasticity.
    
    Args:
        model: The model to train
        train_dataloader: DataLoader for training data
        eval_dataloader: DataLoader for evaluation data
        pruned_heads: List of (layer_idx, head_idx) tuples of pruned heads
        learning_rate: Base learning rate for training
        training_steps: Number of training steps
        use_differential_lr: Whether to use different learning rates for pruned layers
        eval_interval: Interval for evaluation and callback
        callback: Optional callback function for monitoring progress
        
    Returns:
        Dictionary with training metrics history
    """
    trainer = PlasticityTrainer(
        model=model,
        learning_rate=learning_rate,
        use_differential_lr=use_differential_lr
    )
    
    trainer.prepare_optimizer(
        pruned_heads=pruned_heads,
        warmup_steps=training_steps // 10,
        total_steps=training_steps
    )
    
    return trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        steps=training_steps,
        eval_interval=eval_interval,
        callback=callback
    )


def run_warmup_phase(
    model: torch.nn.Module,
    train_dataloader,
    max_epochs: int = 1,
    learning_rate: float = 5e-5,
    warmup_steps: int = 100,
    patience: int = 15,
    min_warmup_steps: int = 50,
    max_warmup_steps: int = 150,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    visualize: bool = True,
    save_visualizations: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a warm-up phase until loss stabilizes.
    
    This function runs a short training loop to stabilize the model's
    parameters and metrics before applying neural plasticity.
    
    Args:
        model: The model to warm up
        train_dataloader: DataLoader for training data
        max_epochs: Maximum number of epochs to run
        learning_rate: Learning rate for optimizer
        warmup_steps: Number of warmup steps for scheduler
        patience: Number of steps with no decrease to consider loss stabilized
        min_warmup_steps: Minimum number of warm-up steps before checking stabilization
        max_warmup_steps: Maximum number of warm-up steps per epoch
        device: Device to run on (defaults to model's device)
        verbose: Whether to print progress information
        visualize: Whether to generate and return visualization
        save_visualizations: Whether to save visualizations to disk
        output_dir: Directory to save visualizations (created if it doesn't exist)
        
    Returns:
        Dictionary with warmup metrics and information
    """
    # Automatically handle device placement (with Apple Silicon safety)
    if device is None:
        if IS_APPLE_SILICON:
            device = torch.device('cpu')
            if verbose:
                print("🍎 Using CPU device for warmup on Apple Silicon")
        else:
            device = next(model.parameters()).device
            if verbose and torch.cuda.is_available():
                print(f"Using GPU device for warmup: {torch.cuda.get_device_name(0)}")
    
    # Move model to device
    model = model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * max_epochs
    scheduler = torch.optim.lr_scheduler.get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    if verbose:
        print(f"Running warm-up until loss stabilizes (max {max_epochs} epochs)...")
    
    # Warm-up training loop
    model.train()
    warmup_losses = []
    warmup_step_losses = []  # For smoothed tracking
    last_loss_decrease = 0
    
    # Helper function to determine if loss has stabilized
    def is_loss_stabilized(losses, min_steps, patience_steps, window_size=5):
        """
        Determine if the loss has stabilized.
        
        Args:
            losses: List of loss values
            min_steps: Minimum number of steps before stabilization can be determined
            patience_steps: Number of steps with no decrease to consider stabilized
            window_size: Window size for rolling average comparison
            
        Returns:
            Tuple of (is_stable, steps_since_decrease)
        """
        # Not enough steps yet
        if len(losses) < min_steps:
            return False, 0

        # Not enough steps since last decrease
        steps_since_decrease = len(losses) - last_loss_decrease
        if steps_since_decrease < patience_steps:
            return False, steps_since_decrease
        
        # Check if recent trend is flat or increasing using rolling average
        if len(losses) >= window_size * 2:
            recent_window = sum(losses[-window_size:]) / window_size
            previous_window = sum(losses[-(window_size*2):-window_size]) / window_size
            # If recent average is lower than previous, we're still decreasing
            if recent_window < previous_window * 0.99:  # Allow 1% variation
                return False, steps_since_decrease
                
        return True, steps_since_decrease
    
    # Initialize metrics collection
    epoch_metrics = []
    total_steps_completed = 0
    warmup_start_time = time.time()
    
    try:
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            
            for step, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
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
                loss_val = loss.item()
                epoch_loss += loss_val
                epoch_steps += 1
                warmup_losses.append(loss_val)
                total_steps_completed += 1
                
                # Update loss tracking
                if len(warmup_losses) > 1:
                    # Track non-increasing steps
                    if loss_val <= warmup_losses[-2]:
                        last_loss_decrease = len(warmup_losses)
                    
                    # For visualization, track a smoothed version (rolling average of 5)
                    if len(warmup_losses) % 5 == 0:
                        avg_loss = sum(warmup_losses[-5:]) / 5
                        warmup_step_losses.append(avg_loss)
                
                # Print progress every 5 steps or as requested
                if verbose and step % 5 == 0:
                    print(f"Warm-up Epoch {epoch+1}, Step {step}: Loss = {loss_val:.4f}", end='\r')
                
                # Check if loss has stabilized
                is_stable, steps_without_decrease = is_loss_stabilized(
                    warmup_losses, min_warmup_steps, patience
                )
                
                if is_stable:
                    if verbose:
                        print(f"\nWarm-up loss stabilized after {len(warmup_losses)} steps")
                        print(f"Loss has been non-decreasing for {steps_without_decrease} steps")
                    break
                    
                # Stop after max_warmup_steps for faster execution in demo
                if step >= max_warmup_steps:
                    if verbose:
                        print(f"\nReached maximum warm-up steps per epoch ({max_warmup_steps})")
                    break
            
            # Store epoch metrics
            if epoch_steps > 0:
                epoch_metrics.append({
                    "epoch": epoch + 1,
                    "loss": epoch_loss / epoch_steps,
                    "steps": epoch_steps
                })
                
                if verbose:
                    print(f"\nWarm-up Epoch {epoch+1} completed: Average Loss = {epoch_loss / epoch_steps:.4f}")
            
            # Check if loss has stabilized across epochs
            is_stable, steps_without_decrease = is_loss_stabilized(
                warmup_losses, min_warmup_steps, patience
            )
            
            if is_stable:
                if verbose:
                    print(f"Loss has stabilized with {steps_without_decrease} steps without significant decrease.")
                    print(f"Ending warm-up early after {epoch+1} epochs.")
                break
        
        # Calculate warmup statistics
        warmup_duration = time.time() - warmup_start_time
        
        # Create visualization if requested
        fig = None
        if visualize and len(warmup_losses) > 5:
            fig = create_warmup_visualization(
                warmup_losses=warmup_losses,
                smoothed_losses=warmup_step_losses
            )
        
        # Perform segment analysis
        segment_analysis = analyze_warmup_segments(
            warmup_losses=warmup_losses,
            verbose=verbose,
            save_path=os.path.join(output_dir, "segment_analysis.txt") if save_visualizations and output_dir else None
        )
        
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Handle visualization saving
        viz_paths = {}
        if save_visualizations:
            # Create output directory if it doesn't exist
            if output_dir is None:
                import datetime
                
                # Use a timestamp-based directory in the current working directory
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.join(os.getcwd(), f"neural_plasticity_warmup_{timestamp}")
            
            # Create the directory if it doesn't exist
            try:
                os.makedirs(output_dir, exist_ok=True)
                
                # Save the warmup visualization
                if fig is not None:
                    warmup_viz_path = os.path.join(output_dir, "warmup_loss_visualization.png")
                    try:
                        fig.savefig(warmup_viz_path, dpi=100, bbox_inches='tight')
                        viz_paths["warmup_loss"] = warmup_viz_path
                        if verbose:
                            print(f"✅ Saved warmup visualization to {warmup_viz_path}")
                    except Exception as e:
                        if verbose:
                            print(f"❌ Error saving visualization: {e}")
                
                # Save raw loss data for external analysis
                try:
                    loss_data_path = os.path.join(output_dir, "warmup_loss_data.txt")
                    with open(loss_data_path, 'w') as f:
                        f.write("step,loss\n")
                        for i, loss in enumerate(warmup_losses):
                            f.write(f"{i},{loss}\n")
                    viz_paths["loss_data"] = loss_data_path
                    if verbose:
                        print(f"✅ Saved loss data to {loss_data_path}")
                except Exception as e:
                    if verbose:
                        print(f"❌ Error saving loss data: {e}")
            except Exception as e:
                if verbose:
                    print(f"❌ Error creating output directory: {e}")
        
        # Return comprehensive warmup information
        return {
            "losses": warmup_losses,
            "smoothed_losses": warmup_step_losses,
            "epochs": epoch_metrics,
            "total_steps": total_steps_completed,
            "duration": warmup_duration,
            "initial_loss": warmup_losses[0] if warmup_losses else 0,
            "final_loss": warmup_losses[-1] if warmup_losses else 0,
            "improvement_percent": (1 - warmup_losses[-1]/warmup_losses[0])*100 if len(warmup_losses) > 1 else 0,
            "is_stable": is_stable if 'is_stable' in locals() else False,
            "steps_without_decrease": steps_without_decrease if 'steps_without_decrease' in locals() else 0,
            "segment_analysis": segment_analysis,
            "visualization": fig,
            "visualization_paths": viz_paths
        }
                
    except Exception as e:
        if verbose:
            print(f"\n❌ Error during warm-up: {e}")
        
        # Try to save error information if output_dir is available
        if save_visualizations and output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                error_log_path = os.path.join(output_dir, "warmup_error.log")
                
                with open(error_log_path, 'w') as f:
                    import traceback
                    f.write(f"Error during warm-up: {e}\n\n")
                    f.write(traceback.format_exc())
                    f.write("\n\nPartial data:\n")
                    f.write(f"Total steps completed: {total_steps_completed}\n")
                    f.write(f"Available loss values: {len(warmup_losses)}\n")
                
                if verbose:
                    print(f"✅ Error details saved to {error_log_path}")
            except Exception as log_error:
                if verbose:
                    print(f"Could not save error log: {log_error}")
        
        # Return partial data on error
        return {
            "losses": warmup_losses,
            "error": str(e),
            "error_traceback": traceback.format_exc() if 'traceback' in locals() else None,
            "total_steps": total_steps_completed,
            "duration": time.time() - warmup_start_time
        }