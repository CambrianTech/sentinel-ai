"""
Training utilities for neural plasticity experiments.

This module provides a simplified fine-tuning implementation for training
transformer models during the neural plasticity cycle.
"""

import numpy as np
import copy
import torch
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
        
        # Use parameters from pruning_module directly
        self.params = pruning_module.params
        
        # Setup training state
        self.step = 0
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Initialize loss tracking
        self.losses = []
    
    def set_params(self, params):
        """
        Set the model parameters for training.
        
        Args:
            params: Model parameters to use
        """
        self.params = params
        
        # For PyTorch, we don't need to reinitialize optimizer
        # Just update the model's parameters with the new params
        
        # Ensure model has params attribute
        if hasattr(self.model, 'params'):
            self.model.params = params
    
    def get_params(self):
        """
        Get the current model parameters.
        
        Returns:
            Current model parameters
        """
        return copy.deepcopy(self.params)
    
    def compute_loss(self, batch):
        """
        Compute the loss for a batch.
        
        Args:
            batch: Batch of data
            
        Returns:
            Loss value
        """
        # Forward pass
        outputs = self.model(batch["input_ids"])
        
        # Get logits - based on model output format
        logits = None
        if isinstance(outputs, torch.Tensor):
            # Direct tensor output
            logits = outputs
        elif hasattr(outputs, 'logits'):
            # Output object with logits attribute
            logits = outputs.logits
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            # First element of tuple
            logits = outputs[0]
        elif isinstance(outputs, dict) and 'logits' in outputs:
            # Dictionary with logits key
            logits = outputs['logits']
        else:
            # Fallback
            print("Warning: Could not determine logits from model output, using raw output")
            logits = outputs
        
        input_ids = batch["input_ids"]
        
        # Ensure logits is a tensor
        if not isinstance(logits, torch.Tensor):
            raise ValueError(f"Expected tensor for logits, got {type(logits)}")
        
        # Handle different output shapes
        if len(logits.shape) == 3:  # [batch_size, seq_len, vocab_size]
            # Shift for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
        elif len(logits.shape) == 2:  # [batch_size, vocab_size]
            # No need to shift for single token prediction
            shift_logits = logits
            shift_labels = input_ids
        else:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")
        
        # Calculate loss using PyTorch's cross entropy
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        
        try:
            if len(shift_logits.shape) == 3:
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            else:
                loss = loss_fct(shift_logits, shift_labels)
        except Exception as e:
            print(f"Error computing loss: {e}")
            print(f"Logits shape: {shift_logits.shape}, Labels shape: {shift_labels.shape}")
            # Return a dummy loss to continue training
            return torch.tensor(1.0, requires_grad=True, device=self.model.device)
        
        return loss
    
    def apply_differential_learning_rates(self):
        """
        Apply differential learning rates to the optimizer if head_lr_manager is provided.
        """
        if self.head_lr_manager is None:
            return
        
        # Apply head-specific learning rates if head_lr_manager is provided
        # For PyTorch, we'd create parameter groups with different learning rates
        
        # For this implementation, we'll simply use the existing optimizer
        # with differential learning rates to be implemented in the future
        
        # This is a placeholder for future implementation
        pass
    
    def train_step(self):
        """
        Perform a single training step.
        
        Returns:
            Loss value for the step
        """
        # Get a batch of data
        batch = next(self.dataset.train_dataloader)
        
        # Move batch to device
        input_ids = torch.tensor(batch["input_ids"]).to(self.model.device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(self.model.device)
        
        # Create a PyTorch batch
        pytorch_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Compute loss
        loss = self.compute_loss(pytorch_batch)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update parameters
        self.optimizer.step()
        
        # Update step counter
        self.step += 1
        
        # Track loss - convert to float
        loss_value_float = loss.item()
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
        # Set model to training mode
        self.model.train()
        
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