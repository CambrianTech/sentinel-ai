"""
Adaptive Fine-tuner for Neural Plasticity

This module provides a fine-tuning framework for pruned transformer models
with adaptive learning rate capabilities for different attention heads.

Version: v0.0.34 (2025-04-20 17:00:00)
"""

import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from .plasticity_tracker import PlasticityTracker

logger = logging.getLogger(__name__)

class AdaptiveFinetuner:
    """
    Fine-tunes pruned transformer models with adaptive learning rates.
    
    This class is responsible for:
    1. Fine-tuning models after pruning
    2. Applying differential learning rates to pruned heads
    3. Tracking plasticity metrics during fine-tuning
    4. Optimizing recovery of pruned model performance
    """
    
    def __init__(
        self,
        model,
        learning_rate: float = 5e-5,
        use_differential_lr: bool = True,
        pruned_head_lr_multiplier: float = 3.0
    ):
        """
        Initialize the adaptive fine-tuner.
        
        Args:
            model: Pre-trained model to fine-tune
            learning_rate: Base learning rate for fine-tuning
            use_differential_lr: Whether to use different learning rates for pruned heads
            pruned_head_lr_multiplier: Learning rate multiplier for pruned heads
        """
        self.model = model
        self.learning_rate = learning_rate
        self.use_differential_lr = use_differential_lr
        self.pruned_head_lr_multiplier = pruned_head_lr_multiplier
        
        # Initialize plasticity tracker
        self.plasticity_tracker = PlasticityTracker()
        
    def _create_optimizer(self, pruned_heads: List[Tuple[int, int, float]]) -> torch.optim.Optimizer:
        """
        Create an optimizer with potentially differential learning rates.
        
        Args:
            pruned_heads: List of (layer, head, score) tuples for pruned heads
            
        Returns:
            PyTorch optimizer
        """
        if not self.use_differential_lr:
            # Use standard Adam optimizer with uniform learning rate
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
        # Create parameter groups with different learning rates
        pruned_params = []
        other_params = []
        pruned_head_indices = set([(l, h) for l, h, _ in pruned_heads])
        
        # Collect parameters from attention heads
        for name, param in self.model.named_parameters():
            if 'query' in name or 'key' in name or 'value' in name or 'attention' in name:
                # Check if this parameter belongs to a pruned head
                # This is a heuristic and might need adaptation for different model architectures
                is_pruned = False
                for layer_idx, head_idx in pruned_head_indices:
                    layer_str = f"layer.{layer_idx}"
                    head_str = f"head.{head_idx}"
                    
                    if layer_str in name and head_str in name:
                        is_pruned = True
                        break
                    # Also check for other naming patterns common in transformer models
                    elif layer_str in name and f"attention" in name:
                        # For models where heads are not explicitly named
                        is_pruned = True
                        break
                        
                if is_pruned:
                    pruned_params.append(param)
                else:
                    other_params.append(param)
            else:
                other_params.append(param)
                
        logger.info(f"Applying {self.pruned_head_lr_multiplier}x higher learning rate to {len(pruned_params)} pruned head parameters")
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': pruned_params, 'lr': self.learning_rate * self.pruned_head_lr_multiplier},
            {'params': other_params, 'lr': self.learning_rate}
        ]
        
        return torch.optim.Adam(param_groups)
        
    def fine_tune(
        self,
        train_dataloader,
        eval_dataloader,
        pruned_heads: List[Tuple[int, int, float]],
        steps: int = 500,
        evaluator = None,
        eval_steps: int = 50,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Fine-tune the pruned model to recover performance.
        
        Args:
            train_dataloader: DataLoader for training data
            eval_dataloader: DataLoader for evaluation data
            pruned_heads: List of (layer, head, score) tuples for pruned heads
            steps: Number of fine-tuning steps
            evaluator: Evaluator object for model evaluation
            eval_steps: Frequency of evaluation during training
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary of training results
        """
        # Create optimizer with potentially differential learning rates
        optimizer = self._create_optimizer(pruned_heads)
        
        # Training loop
        self.model.train()
        step = 0
        train_iter = iter(train_dataloader)
        
        while step < steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                # Restart iterator if we run out of data
                train_iter = iter(train_dataloader)
                batch = next(train_iter)
                
            # Move batch to device
            batch = {k: v.to(self.model.device) if hasattr(v, 'to') else v 
                     for k, v in batch.items()}
                     
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record metrics
            loss_value = loss.item()
            perplexity = float(np.exp(loss_value))
            self.plasticity_tracker.record_performance(step, loss_value, perplexity)
            
            # Periodic evaluation
            if evaluator is not None and step % eval_steps == 0:
                eval_metrics = self._evaluate_model(evaluator, eval_dataloader)
                
                # Log evaluation metrics
                logger.info(f"Step {step}: Loss = {loss_value:.4f}, " +
                           f"Perplexity = {perplexity:.2f}, " +
                           f"Eval Loss = {eval_metrics['loss']:.4f}, " +
                           f"Eval Perplexity = {eval_metrics['perplexity']:.2f}")
            else:
                # Log training metrics only
                logger.info(f"Step {step}: Loss = {loss_value:.4f}, Perplexity = {perplexity:.2f}")
                
            # Update progress if callback provided
            if progress_callback is not None:
                progress_callback(step, steps)
                
            step += 1
            
        # Final evaluation
        final_metrics = None
        if evaluator is not None:
            final_metrics = self._evaluate_model(evaluator, eval_dataloader)
            logger.info(f"Final: Eval Loss = {final_metrics['loss']:.4f}, " +
                       f"Eval Perplexity = {final_metrics['perplexity']:.2f}")
            
        # Prepare training results
        results = {
            "steps": steps,
            "performance_history": self.plasticity_tracker.performance_history,
            "final_metrics": final_metrics
        }
        
        return results
        
    def _evaluate_model(self, evaluator, eval_dataloader) -> Dict[str, float]:
        """
        Evaluate the model using the provided evaluator.
        
        Args:
            evaluator: Evaluator object for model evaluation
            eval_dataloader: DataLoader for evaluation data
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Temporarily set model to eval mode
        was_training = self.model.training
        self.model.eval()
        
        # Evaluate
        metrics = evaluator.evaluate(self.model, eval_dataloader)
        
        # Restore model's previous training state
        if was_training:
            self.model.train()
            
        return metrics