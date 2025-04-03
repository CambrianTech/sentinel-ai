"""
Training utilities for neural plasticity experiments.

This module provides a simplified fine-tuning implementation for training
transformer models during the neural plasticity cycle.
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
import copy
from tqdm import tqdm

class FineTuner:
    """
    Simple fine-tuner for transformer models using JAX/Flax.
    
    This class provides basic training functionality for fine-tuning models
    during the neural plasticity cycle. It supports differential learning 
    rates for newly added heads through the HeadLRManager.
    """
    
    def __init__(self, pruning_module, dataset, learning_rate=5e-5, head_lr_manager=None):
        """
        Initialize the fine-tuner.
        
        Args:
            pruning_module: PruningModule instance with the model to train
            dataset: Dataset loader for training data
            learning_rate: Base learning rate
            head_lr_manager: Optional HeadLRManager for differential learning rates
        """
        self.pruning_module = pruning_module
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.head_lr_manager = head_lr_manager
        
        # Get model
        self.model = pruning_module.model
        
        # Initialize parameters (will be overridden by set_params if called)
        self.params = self.model.params
        
        # Initialize optimizer
        self.optimizer = optax.adam(learning_rate=learning_rate)
        self.opt_state = self.optimizer.init(self.params)
        
        # Setup training state
        self.step = 0
        
        # Initialize loss tracking
        self.losses = []
    
    def set_params(self, params):
        """
        Set the model parameters for training.
        
        Args:
            params: Model parameters to use
        """
        self.params = params
        # Reinitialize optimizer state
        self.opt_state = self.optimizer.init(self.params)
    
    def get_params(self):
        """
        Get the current model parameters.
        
        Returns:
            Current model parameters
        """
        return copy.deepcopy(self.params)
    
    def compute_loss(self, params, batch):
        """
        Compute the loss for a batch.
        
        Args:
            params: Model parameters
            batch: Batch of data
            
        Returns:
            Loss value
        """
        # Forward pass
        outputs = self.model(**batch, params=params)
        
        # Get logits and labels
        logits = outputs.logits
        input_ids = batch["input_ids"]
        
        # Shift for next token prediction
        shift_logits = logits[:, :-1]
        shift_labels = input_ids[:, 1:]
        
        # Calculate loss
        loss = optax.softmax_cross_entropy_with_integer_labels(
            shift_logits, shift_labels
        ).mean()
        
        return loss
    
    @staticmethod
    def clip_grads(grads, max_norm=1.0):
        """
        Clip gradients to prevent NaN issues.
        
        Args:
            grads: Gradients to clip
            max_norm: Maximum gradient norm
            
        Returns:
            Clipped gradients
        """
        # Calculate gradient norm
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_leaves(grads)))
        
        # Clip if necessary
        clip_factor = jnp.minimum(1.0, max_norm / (grad_norm + 1e-6))
        
        # Apply clipping
        clipped_grads = jax.tree_map(lambda g: g * clip_factor, grads)
        
        return clipped_grads
    
    def apply_differential_learning_rates(self, grads):
        """
        Apply differential learning rates to gradients if head_lr_manager is provided.
        
        Args:
            grads: Gradients to modify
            
        Returns:
            Gradients with modified learning rates
        """
        if self.head_lr_manager is None:
            return grads
        
        # Apply head-specific learning rates if head_lr_manager is provided
        # This is a simplified implementation - in a real system, this would map
        # gradients to specific heads based on parameter paths
        
        # In a real implementation, we would identify which parameters belong to which heads
        # and scale their gradients by the corresponding learning rate factors
        
        # For this demo, we'll apply a simple approximation, scaling query/key/value parameters
        # which would be associated with specific heads
        
        # Placeholder implementation
        return grads
    
    def train_step(self):
        """
        Perform a single training step.
        
        Returns:
            Loss value for the step
        """
        # Get a batch of data
        batch = next(self.dataset.train_dataloader)
        
        # Define gradient function
        def compute_grads(params):
            loss = self.compute_loss(params, batch)
            return loss, loss
        
        # Compute gradients
        (loss_value, _), grads = jax.value_and_grad(compute_grads, has_aux=True)(self.params)
        
        # Clip gradients to prevent instability
        grads = self.clip_grads(grads)
        
        # Apply differential learning rates if needed
        if self.head_lr_manager is not None:
            grads = self.apply_differential_learning_rates(grads)
        
        # Update parameters
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        self.params = optax.apply_updates(self.params, updates)
        
        # Update step counter
        self.step += 1
        
        # Track loss - convert JAX array to float
        loss_value_float = float(loss_value)
        self.losses.append(loss_value_float)
        
        return loss_value_float
    
    def train(self, num_steps=100, eval_every=50):
        """
        Train the model for a specified number of steps.
        
        Args:
            num_steps: Number of training steps
            eval_every: Evaluate every N steps
            
        Returns:
            Dictionary with training results
        """
        # Training loop
        for step in tqdm(range(num_steps)):
            # Train for one step
            loss = self.train_step()
            
            # Evaluate periodically
            if step % eval_every == 0 or step == num_steps - 1:
                # Print progress
                print(f"Step {step}: Loss = {loss:.4f}")
        
        # Return training results
        return {
            "final_loss": loss,
            "losses": self.losses,
            "steps": num_steps
        }
    
    def save_checkpoint(self, path):
        """
        Save a checkpoint of the current training state.
        
        Args:
            path: Path to save the checkpoint
        """
        import pickle
        
        checkpoint = {
            "params": self.params,
            "optimizer_state": self.opt_state,
            "step": self.step,
            "losses": self.losses,
            "learning_rate": self.learning_rate
        }
        
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def load_checkpoint(self, path):
        """
        Load a checkpoint of a previous training state.
        
        Args:
            path: Path to the checkpoint
        """
        import pickle
        
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.params = checkpoint["params"]
        self.opt_state = checkpoint["optimizer_state"]
        self.step = checkpoint["step"]
        self.losses = checkpoint["losses"]
        self.learning_rate = checkpoint["learning_rate"]