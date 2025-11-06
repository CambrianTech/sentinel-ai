"""Training utilities - DEPRECATED MODULE

This module is a backwards compatibility layer for the old utils.train_utils module.
The functionality has been moved to sentinel.utils.train_utils.

Please update your imports to use the new module path.
"""

import warnings
import torch
import copy
from tqdm import tqdm

# Emit deprecation warning
warnings.warn(
    "The module utils.train_utils is deprecated. "
    "Please use sentinel.utils.train_utils instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location
from sentinel.utils.train_utils import FineTuner as _NewFineTuner

# Create a subclass that maintains backward compatibility
class FineTuner(_NewFineTuner):
    """Legacy FineTuner with backward compatibility"""
    
    def __init__(self, pruning_module, dataset, learning_rate=5e-5, head_lr_manager=None):
        super().__init__(pruning_module, dataset, learning_rate, head_lr_manager)
        
        # For backward compatibility with original implementation
        self.model = pruning_module.model
        self.losses = []
        
        # Initialize optimizer (if needed)
        if hasattr(self.model, 'parameters'):
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
    
    # Legacy methods with backward compatibility
    def compute_loss(self, batch):
        """Backward compatible loss computation"""
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
    
    def train(self, num_steps=100, eval_every=50):
        """Backward compatible training method"""
        # Set model to training mode
        if hasattr(self.model, 'train'):
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
        """Save a checkpoint of the current training state."""
        import pickle
        
        checkpoint = {
            "params": self.params,
            "optimizer_state": self.opt_state if hasattr(self, 'opt_state') else None,
            "step": self.step,
            "losses": self.losses,
            "learning_rate": self.learning_rate
        }
        
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def load_checkpoint(self, path):
        """Load a checkpoint of a previous training state."""
        import pickle
        
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.params = checkpoint["params"]
        if hasattr(self, 'opt_state') and "optimizer_state" in checkpoint:
            self.opt_state = checkpoint["optimizer_state"]
        self.step = checkpoint["step"]
        self.losses = checkpoint["losses"]
        self.learning_rate = checkpoint["learning_rate"]


# Add all imported symbols to __all__
__all__ = ["FineTuner"]