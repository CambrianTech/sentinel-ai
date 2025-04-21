"""
Neural Plasticity Training v0.0.65 (2025-04-20 14:45:00)

This module provides training utilities for neural plasticity experiments,
including differential learning rates for pruned vs. unpruned heads,
specialized optimizers, and training loops with plasticity.

Added features:
- Sample text generation during training for better monitoring
- Per-token perplexity calculation and display 
- Polynomial curve fitting for improved loss stabilization detection
- Integration with dashboard visualization
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

# Try to import dashboard utilities
try:
    from .dashboard import DashboardReporter
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

# Fix for scheduler import
try:
    from transformers import get_linear_schedule_with_warmup
except ImportError:
    # Provide a simple implementation if transformers version is missing it
    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        """
        Create a schedule with a learning rate that decreases linearly from the initial lr.
        This is a simple fallback implementation for when transformers is not available.
        """
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
            
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

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
    stabilization_point: Optional[int] = None,
    is_stable: bool = False,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create visualization for warmup training.
    
    Args:
        warmup_losses: List of raw loss values
        smoothed_losses: List of smoothed (rolling average) loss values
        stabilization_point: Step at which stability was detected (if any)
        is_stable: Whether stabilization was detected
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
    
    # Mark stabilization point on raw loss curve if provided
    if stabilization_point is not None and stabilization_point < len(warmup_losses):
        # Add vertical line at stabilization point
        axs[0].axvline(x=stabilization_point, color='green', linestyle='--', alpha=0.7, 
                     label=f'Stabilization detected')
        
        # Mark the point
        axs[0].plot(stabilization_point, warmup_losses[stabilization_point], 
                  'go', markersize=8, label=f'Loss: {warmup_losses[stabilization_point]:.4f}')
        
        # Add note about stabilization
        stabilization_text = f"Stabilization at step {stabilization_point}"
        axs[0].text(stabilization_point + len(warmup_losses)*0.01, 
                  warmup_losses[stabilization_point]*0.95, 
                  stabilization_text, fontsize=9, color='green')
        
        axs[0].legend()
    
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
            
            # Add polynomial fit to smoothed plot for stabilization detection
            try:
                from scipy.optimize import curve_fit
                
                # Define polynomial function (degree 2 = quadratic)
                def poly_func(x, a, b, c):
                    return a * x**2 + b * x + c
                
                # Fit polynomial to the recent loss values
                params, _ = curve_fit(poly_func, x, smoothed_losses)
                a, b, c = params
                
                # Plot polynomial fit
                x_dense = np.linspace(min(x), max(x), 100)
                y_fit = poly_func(x_dense, a, b, c)
                axs[1].plot(x_dense, y_fit, 'g-', 
                          label=f'Poly fit: {a:.5f}x² + {b:.5f}x + {c:.5f}')
                
                # Calculate derivatives at the end point
                x_end = max(x)
                deriv_1 = 2 * a * x_end + b  # First derivative
                deriv_2 = 2 * a              # Second derivative
                
                # Show stabilization info in title
                if is_stable:
                    status = "Loss stabilized"
                else:
                    if a > 0:
                        status = "Upward curve"
                    elif abs(deriv_1) < 0.01:
                        status = "Flat curve"
                    else:
                        status = "Still decreasing"
                
                axs[1].set_title(f"Warm-up Loss (5-step Rolling Average) - {status}")
            except Exception as e:
                print(f"Could not add polynomial fit: {e}")
                
            axs[1].legend()
        except (ImportError, ValueError) as e:
            # Skip the trend line if an error occurs
            print(f"Could not add trend line: {e}")
            
        # Mark stabilization point on smoothed curve if it falls within the data
        if stabilization_point is not None:
            # Calculate which index in smoothed_losses corresponds to stabilization_point
            smooth_idx = stabilization_point // 5
            if smooth_idx < len(smoothed_losses):
                smooth_x = smooth_idx * 5  # Convert back to original x scale
                axs[1].axvline(x=smooth_x, color='green', linestyle='--', alpha=0.7)
                axs[1].plot(smooth_x, smoothed_losses[smooth_idx], 'go', markersize=8)
    
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
    
    def train_step(self, batch, show_samples=False, tokenizer=None, sample_step=10):
        """
        Perform a single training step.
        
        Args:
            batch: Training batch data
            show_samples: Whether to display sample texts and predictions
            tokenizer: Tokenizer for decoding tokens (required if show_samples=True)
            sample_step: Only show samples every N steps
            
        Returns:
            Dictionary with loss value and optionally sample information
        """
        self.model.train()
        result = {"loss": None, "sample": None}
        
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
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            
            if "labels" in inputs:
                labels = inputs["labels"]
                logits = outputs.logits
            else:
                # For causal language modeling, shift labels
                labels = inputs["input_ids"].clone()
                labels = labels[:, 1:].contiguous()
                logits = outputs.logits[:, :-1, :].contiguous()
            
            # Get per-token loss for perplexity calculation if needed
            per_token_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = per_token_loss.mean()
        
        # Process sample text and predictions if requested
        if show_samples and tokenizer is not None and hasattr(self, "_step_counter"):
            self._step_counter += 1
            if self._step_counter % sample_step == 0:
                result["sample"] = self._process_sample(inputs, outputs, tokenizer)
        elif show_samples and tokenizer is not None:
            self._step_counter = 1
            result["sample"] = self._process_sample(inputs, outputs, tokenizer)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        result["loss"] = loss.item()
        return result
        
    def _process_sample(self, inputs, outputs, tokenizer):
        """
        Process a sample from the batch to show model predictions.
        
        Args:
            inputs: Input tensors
            outputs: Model outputs
            tokenizer: Tokenizer for decoding
            
        Returns:
            Dictionary with sample information
        """
        try:
            with torch.no_grad():
                # Get input text from a random position in the batch
                batch_size = inputs["input_ids"].shape[0]
                sample_idx = torch.randint(0, batch_size, (1,)).item()
                
                # Get the input sequence
                input_ids = inputs["input_ids"][sample_idx]
                input_text = tokenizer.decode(input_ids)
                
                # Get predictions
                logits = outputs.logits[sample_idx]
                
                # For each position, get the predicted next token
                predictions = []
                for i in range(min(len(input_ids) - 1, 10)):  # Limit to 10 positions for clarity
                    pos_logits = logits[i]
                    next_token_id = torch.argmax(pos_logits).item()
                    next_token = tokenizer.decode([next_token_id])
                    
                    # Get probability and actual next token
                    probs = torch.softmax(pos_logits, dim=0)
                    next_token_prob = probs[next_token_id].item()
                    
                    actual_token_id = input_ids[i + 1].item()
                    actual_token = tokenizer.decode([actual_token_id])
                    actual_token_prob = probs[actual_token_id].item()
                    
                    # Calculate per-token perplexity (exp of negative log probability)
                    token_perplexity = torch.exp(-torch.log(probs[actual_token_id])).item()
                    
                    predictions.append({
                        "position": i,
                        "context": tokenizer.decode(input_ids[:i+1]),
                        "predicted_token": next_token,
                        "predicted_prob": next_token_prob,
                        "actual_token": actual_token,
                        "actual_prob": actual_token_prob,
                        "perplexity": token_perplexity
                    })
                
                return {
                    "input_text": input_text,
                    "predictions": predictions
                }
        except Exception as e:
            return {
                "error": str(e),
                "input_text": "Error processing sample"
            }
    
    def train(
        self,
        train_dataloader,
        eval_dataloader,
        steps: int,
        eval_interval: int = 50,
        callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
        show_samples: bool = False,
        tokenizer=None,
        sample_interval: int = 10,
        sample_callback: Optional[Callable[[Dict], None]] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_dataloader: DataLoader for training data
            eval_dataloader: DataLoader for evaluation data
            steps: Number of training steps
            eval_interval: Interval for evaluation and callback
            callback: Optional callback function called after each evaluation
            show_samples: Whether to display sample predictions during training
            tokenizer: Tokenizer for decoding tokens (required if show_samples=True)
            sample_interval: Interval for showing sample predictions
            sample_callback: Optional callback for handling sample data
            
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
        
        # Add sample tracking if enabled
        if show_samples:
            metrics_history["samples"] = []
            if tokenizer is None:
                print("Warning: show_samples=True but no tokenizer provided. Samples will not be shown.")
                show_samples = False
        
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
            
            # Train step with sample processing if enabled
            step_result = self.train_step(
                batch,
                show_samples=show_samples,
                tokenizer=tokenizer,
                sample_step=sample_interval
            )
            
            loss = step_result["loss"]
            sample_data = step_result.get("sample")
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{loss:.4f}", lr=f"{self.scheduler.get_last_lr()[0]:.2e}")
            
            # Handle sample data if available
            if sample_data is not None:
                # Store sample in history
                if "samples" in metrics_history:
                    metrics_history["samples"].append({
                        "step": global_step,
                        "data": sample_data
                    })
                
                # Call sample callback if provided
                if sample_callback:
                    sample_callback(global_step, sample_data)
                elif show_samples:
                    # Print sample data directly if no callback is provided
                    self._display_sample(global_step, sample_data)
            
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
        
    def _display_sample(self, step, sample_data):
        """
        Display sample prediction data in a formatted way.
        
        Args:
            step: Current training step
            sample_data: Sample prediction data
        """
        if "error" in sample_data:
            print(f"\nSample processing error at step {step}: {sample_data['error']}")
            return
            
        # Print header
        print(f"\n{'='*80}")
        print(f"Sample predictions at step {step}:")
        print(f"{'='*80}")
        
        # Print input text (truncated if too long)
        input_text = sample_data["input_text"]
        if len(input_text) > 100:
            input_text = input_text[:97] + "..."
        print(f"Input text: {input_text}")
        print(f"{'-'*80}")
        
        # Print predictions
        if "predictions" in sample_data and sample_data["predictions"]:
            print(f"{'Position':<10} {'Context':<30} {'Predicted':<15} {'Actual':<15} {'Perplexity':<10}")
            print(f"{'-'*80}")
            
            for pred in sample_data["predictions"]:
                # Truncate context if too long
                context = pred["context"]
                if len(context) > 28:
                    context = "..." + context[-25:]
                
                # Format prediction and actual token
                predicted = f"{pred['predicted_token']} ({pred['predicted_prob']:.2f})"
                actual = f"{pred['actual_token']} ({pred['actual_prob']:.2f})"
                
                print(f"{pred['position']:<10} {context:<30} {predicted:<15} {actual:<15} {pred['perplexity']:<10.2f}")
        else:
            print("No predictions available")
        
        print(f"{'='*80}\n")


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
    callback: Optional[Callable[[str, int, Dict[str, Any]], None]] = None,
    show_samples: bool = False,
    tokenizer=None,
    sample_interval: int = 20,
    use_dashboard: bool = False,
    dashboard_dir: str = "dashboard",
    dashboard_name: str = "neural_plasticity_dashboard.html"
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
        show_samples: Whether to display sample predictions during training
        tokenizer: Tokenizer for decoding tokens (required if show_samples=True)
        sample_interval: Interval for showing sample predictions
        use_dashboard: Whether to generate an interactive HTML dashboard
        dashboard_dir: Directory to save dashboard files
        dashboard_name: Name of the main dashboard HTML file
        
    Returns:
        Dictionary with experiment results
    """
    device = next(model.parameters()).device
    
    # Verify tokenizer if samples are requested
    if show_samples and tokenizer is None:
        print("Warning: show_samples=True but no tokenizer provided. Samples will not be shown.")
        show_samples = False
    
    # Set up dashboard reporter if requested
    dashboard_reporter = None
    if use_dashboard and DASHBOARD_AVAILABLE:
        dashboard_reporter = DashboardReporter(
            output_dir=dashboard_dir,
            dashboard_name=dashboard_name,
            auto_update=True,
            update_interval=min(training_steps // 10, 10)  # Update every ~10% of training or 10 steps, whichever is smaller
        )
        print(f"🔍 Interactive dashboard will be available at: {os.path.join(dashboard_dir, dashboard_name)}")
    elif use_dashboard and not DASHBOARD_AVAILABLE:
        print("⚠️ Dashboard visualization requested but not available. Make sure dashboard.py is accessible.")
    
    # 1. Initial evaluation
    print("Evaluating baseline model...")
    baseline_metrics = evaluate_model(model, eval_dataloader)
    print(f"Baseline evaluation: Loss = {baseline_metrics['loss']:.4f}, Perplexity = {baseline_metrics['perplexity']:.2f}")
    
    # Update dashboard with baseline metrics
    if dashboard_reporter:
        dashboard_reporter.add_metrics({"eval_loss": baseline_metrics['loss'], 
                                       "perplexity": baseline_metrics['perplexity']}, 0)
    
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
    
    # Update dashboard with pruning information
    if dashboard_reporter:
        dashboard_reporter.update_pruning_info(
            entropy_values=entropy_values,
            grad_norm_values=grad_norm_values,
            pruning_mask=pruning_mask,
            pruned_heads=pruned_heads
        )
        
        # Add model information
        model_info = {}
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        model_size_mb = (param_size + buffer_size) / 1024**2
        total_params = sum(p.numel() for p in model.parameters())
        
        model_info = {
            "model_size_mb": model_size_mb,
            "total_params": total_params,
            "sparsity": len(pruned_heads) / (grad_norm_values.numel()) if grad_norm_values is not None else 0.0,
            "pruned_heads_count": len(pruned_heads)
        }
        dashboard_reporter.set_model_info(model_info)
        
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
    
    # Update dashboard with post-pruning metrics
    if dashboard_reporter:
        dashboard_reporter.add_metrics({
            "eval_loss": pruned_metrics['loss'], 
            "perplexity": pruned_metrics['perplexity'],
            "sparsity": len(pruned_heads) / (grad_norm_values.numel()) if grad_norm_values is not None else 0.0
        }, 0)
    
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
    
    # Define callbacks that forward to the main callback
    # Define callbacks
    def train_callback(step, metrics):
        if callback:
            callback("training", step, metrics)
        
        # Update dashboard with training metrics
        if dashboard_reporter:
            dashboard_reporter.add_metrics(metrics, step)
    
    def sample_callback(step, sample_data):
        if callback:
            callback("sample", step, sample_data)
        
        # Update dashboard with sample predictions
        if dashboard_reporter and sample_data:
            # Process sample data for dashboard
            try:
                # Transform prediction format for dashboard
                input_text = sample_data.get("input_text", "")
                predictions = sample_data.get("predictions", [])
                
                if predictions:
                    predicted_tokens = [p["predicted_token"] for p in predictions]
                    predicted_probs = [p["predicted_prob"] for p in predictions]
                    actual_tokens = [p["actual_token"] for p in predictions]
                    actual_probs = [p["actual_prob"] for p in predictions]
                    perplexities = [p["perplexity"] for p in predictions]
                    
                    dashboard_reporter.add_sample(
                        step=step,
                        input_text=input_text,
                        predicted_tokens=predicted_tokens,
                        predicted_probs=predicted_probs,
                        actual_tokens=actual_tokens,
                        actual_probs=actual_probs,
                        perplexities=perplexities
                    )
            except Exception as e:
                print(f"⚠️ Error adding sample to dashboard: {e}")
    
    # Set up dashboard sample callback
    dashboard_sample_callback = None
    if dashboard_reporter and show_samples:
        dashboard_sample_callback = dashboard_reporter.get_sample_callback()
    
    # Train with sample display if requested
    training_metrics = trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        steps=training_steps,
        eval_interval=max(1, training_steps // 10),
        callback=train_callback,
        show_samples=show_samples,
        tokenizer=tokenizer,
        sample_interval=sample_interval,
        sample_callback=sample_callback if callback else dashboard_sample_callback
    )
    
    # 7. Final evaluation
    print("Performing final evaluation...")
    final_metrics = evaluate_model(model, eval_dataloader)
    print(f"Final evaluation: Loss = {final_metrics['loss']:.4f}, Perplexity = {final_metrics['perplexity']:.2f}")
    
    # Update dashboard with final metrics
    if dashboard_reporter:
        # Update metrics
        dashboard_reporter.add_metrics({
            "eval_loss": final_metrics['loss'],
            "perplexity": final_metrics['perplexity'],
            "sparsity": len(pruned_heads) / (grad_norm_values.numel()) if grad_norm_values is not None else 0.0,
            "train_loss": training_metrics.get("train_loss", [-1])[-1] if "train_loss" in training_metrics else -1
        }, training_steps)
        
        # Add attention visualization if possible
        try:
            # Get attention maps for visualization
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
                
                # Add attention maps to dashboard
                if hasattr(outputs, 'attentions') and outputs.attentions:
                    for layer_idx, layer_attn in enumerate(outputs.attentions):
                        dashboard_reporter.add_attention_map(layer_attn, layer_idx)
        except Exception as e:
            print(f"⚠️ Error adding attention visualization to dashboard: {e}")
        
        # Generate final dashboard
        dashboard_path = dashboard_reporter.update_dashboard()
        print(f"🔍 Final dashboard generated at: {dashboard_path}")
    
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
    results = {
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
    
    # Store samples if available
    if show_samples and "samples" in training_metrics:
        results["samples"] = training_metrics["samples"]
    
    # Add dashboard path to results if available
    if dashboard_reporter:
        results["dashboard_path"] = dashboard_reporter.update_dashboard()
    
    return results


def train_with_plasticity(
    model: torch.nn.Module,
    train_dataloader,
    eval_dataloader,
    pruned_heads: List[Tuple[int, int]],
    learning_rate: float = 5e-5,
    training_steps: int = 500,
    use_differential_lr: bool = True,
    eval_interval: int = 50,
    callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    show_samples: bool = False,
    tokenizer=None,
    sample_interval: int = 20,
    sample_callback: Optional[Callable[[int, Dict], None]] = None
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
        show_samples: Whether to display sample predictions during training
        tokenizer: Tokenizer for decoding tokens (required if show_samples=True)
        sample_interval: Interval for showing sample predictions
        sample_callback: Optional callback for handling sample data
        
    Returns:
        Dictionary with training metrics history
    """
    # Verify tokenizer if samples are requested
    if show_samples and tokenizer is None:
        print("Warning: show_samples=True but no tokenizer provided. Samples will not be shown.")
        show_samples = False
    
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
        callback=callback,
        show_samples=show_samples,
        tokenizer=tokenizer,
        sample_interval=sample_interval,
        sample_callback=sample_callback
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
    scheduler = get_linear_schedule_with_warmup(
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
        Determine if the loss has stabilized using polynomial fitting and multiple indicators.
        
        Args:
            losses: List of loss values
            min_steps: Minimum number of steps before stabilization can be determined
            patience_steps: Number of steps with no decrease to consider stabilized
            window_size: Window size for trend analysis
            
        Returns:
            Tuple of (is_stable, steps_since_decrease)
        """
        # Not enough steps yet
        if len(losses) < min_steps:
            return False, 0

        # Track steps since last decrease for traditional methods
        steps_since_decrease = len(losses) - last_loss_decrease
        
        # Early return if we haven't reached patience threshold
        if steps_since_decrease < patience_steps:
            return False, steps_since_decrease
        
        # Get more robust indicators by analyzing in multiple ways
        try:
            import numpy as np
            from scipy.optimize import curve_fit
            
            # 1. POLYNOMIAL FITTING - Analyze recent trend with curve fitting
            # Use last 30 steps or all available steps if fewer for better curve fitting
            analysis_window = min(30, len(losses))
            recent_losses = losses[-analysis_window:]
            x_data = np.array(range(len(recent_losses)))
            y_data = np.array(recent_losses)
            
            # Define polynomial function (degree 2 = quadratic)
            def poly_func(x, a, b, c):
                return a * x**2 + b * x + c
                
            # Fit polynomial to the recent loss values
            params, _ = curve_fit(poly_func, x_data, y_data)
            a, b, c = params
            
            # Calculate derivatives at the end point (most recent)
            x_end = analysis_window - 1
            deriv_1 = 2 * a * x_end + b  # First derivative
            deriv_2 = 2 * a              # Second derivative (constant for quadratic)
            
            # 2. WINDOW COMPARISON - Compare sequence of adjacent windows
            # Use 5 sequential windows of size window_size to analyze trend
            if len(losses) >= window_size * 5:
                window_avgs = []
                for i in range(5):
                    start_idx = len(losses) - (5-i) * window_size
                    end_idx = start_idx + window_size
                    window_avgs.append(sum(losses[start_idx:end_idx]) / window_size)
                
                # Calculate window-to-window improvements
                window_improvements = [
                    (window_avgs[i] - window_avgs[i+1]) / window_avgs[i] 
                    for i in range(4)
                ]
                
                # Calculate rate of diminishing returns
                diminishing_returns = [
                    window_improvements[i] - window_improvements[i+1]
                    for i in range(3)
                ]
                
                # Recent windows show stabilization if improvements are diminishing
                windows_stabilized = (
                    all(imp >= 0 for imp in diminishing_returns) and  # Consistently diminishing
                    window_improvements[-1] < 0.01  # Last improvement is very small
                )
            else:
                windows_stabilized = False
                
            # 3. RELATIVE IMPROVEMENT - Analyze total improvement so far vs. recent improvement
            if len(losses) >= min_steps * 2:
                # Compare first vs current loss
                total_improvement = (losses[0] - losses[-1]) / losses[0]
                
                # Compare halfway point vs current
                midpoint = len(losses) // 2
                recent_improvement = (losses[midpoint] - losses[-1]) / losses[midpoint]
                
                # Stable if recent improvement is much smaller than total improvement
                # (indicates most gains happened earlier)
                relative_improvement_stable = (
                    total_improvement > 0.05 and  # Had meaningful improvement overall
                    recent_improvement < total_improvement * 0.2  # Recent improvement is small fraction of total
                )
            else:
                relative_improvement_stable = False
            
            # Print diagnostics
            if verbose and len(losses) % 5 == 0:
                print(f"  Polynomial fit: y = {a:.6f}x² + {b:.6f}x + {c:.6f}")
                print(f"  First derivative: {deriv_1:.6f}, Second derivative: {deriv_2:.6f}")
                if 'windows_stabilized' in locals():
                    print(f"  Window analysis: {'Stable' if windows_stabilized else 'Not stable'}")
                if 'relative_improvement_stable' in locals():
                    print(f"  Improvement analysis: {'Stable' if relative_improvement_stable else 'Not stable'}")
                
            # Combine all indicators:
            poly_stable = (
                (a > 0.0005) or                  # Upward curve
                (abs(deriv_1) < 0.01 and abs(deriv_2) < 0.005) or  # Flat curve
                (deriv_1 > 0.01)                 # Increasing slope
            )
            
            # Multiple indicators suggest stability
            is_stable = (
                (poly_stable and windows_stabilized) or  # Both main indicators agree
                (poly_stable and relative_improvement_stable) or  # Polynomial and improvement indicators agree
                (windows_stabilized and relative_improvement_stable) or  # Window and improvement indicators agree
                (steps_since_decrease > patience_steps * 2)  # Been a long time with no improvement
            )
            
            if is_stable and verbose:
                print(f"💡 Loss has stabilized based on multiple indicators")
                print(f"  - Polynomial analysis: {'Stable' if poly_stable else 'Not stable'}")
                print(f"  - Window analysis: {'Stable' if 'windows_stabilized' in locals() and windows_stabilized else 'Not stable/Not enough data'}")
                print(f"  - Improvement analysis: {'Stable' if 'relative_improvement_stable' in locals() and relative_improvement_stable else 'Not stable/Not enough data'}")
                print(f"  - Steps since decrease: {steps_since_decrease} (threshold: {patience_steps})")
                
            return is_stable, steps_since_decrease
                
        except (ImportError, ValueError, RuntimeError) as e:
            # Fallback to traditional method if curve fitting fails
            if verbose:
                print(f"Polynomial fitting failed: {e}. Using traditional stabilization detection.")
            pass
        
        # Fallback: Check if recent trend is flat or increasing using rolling average
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
            # Determine stabilization point (if it occurred)
            stabilization_point = None
            if 'is_stable' in locals() and is_stable:
                # Stabilization occurred at the step when we decided to stop
                stabilization_point = len(warmup_losses) - 1
            
            fig = create_warmup_visualization(
                warmup_losses=warmup_losses,
                smoothed_losses=warmup_step_losses,
                stabilization_point=stabilization_point,
                is_stable=is_stable if 'is_stable' in locals() else False
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