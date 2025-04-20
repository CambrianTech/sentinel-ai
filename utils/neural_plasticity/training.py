"""
Neural Plasticity Training

This module provides training utilities for neural plasticity experiments,
including differential learning rates for pruned vs. unpruned heads,
specialized optimizers, and training loops with plasticity.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from tqdm import tqdm
import time

from .core import (
    calculate_head_entropy,
    calculate_head_gradients,
    generate_pruning_mask,
    apply_pruning_mask,
    evaluate_model
)


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