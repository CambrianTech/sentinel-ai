"""
Simplified Head Learning Rate Manager for the neural plasticity experiments.

This module provides a basic implementation of differential learning rates
for newly added heads during the growth phase of neural plasticity.
"""

import jax
import jax.numpy as jnp
import numpy as np

class HeadLRManager:
    """
    Manages differential learning rates for attention heads during fine-tuning
    after head growth. Newly added heads get higher learning rates to accelerate
    their integration.
    """
    
    def __init__(self, base_lr=5e-5, new_head_multiplier=5.0, new_heads=None):
        """
        Initialize the head learning rate manager.
        
        Args:
            base_lr: Base learning rate for all parameters
            new_head_multiplier: Factor to increase learning rate for newly added heads
            new_heads: List of (layer_idx, head_idx) tuples for newly added heads
        """
        self.base_lr = base_lr
        self.new_head_multiplier = new_head_multiplier
        self.new_heads = set(new_heads) if new_heads else set()
        
    def get_head_lr(self, layer_idx, head_idx):
        """
        Get the learning rate for a specific head.
        
        Args:
            layer_idx: Layer index
            head_idx: Head index within the layer
            
        Returns:
            Learning rate for the specified head
        """
        if (layer_idx, head_idx) in self.new_heads:
            return self.base_lr * self.new_head_multiplier
        return self.base_lr
    
    def get_param_lr_scale(self, param_info):
        """
        Get learning rate scale for a parameter based on its association with heads.
        
        Args:
            param_info: Dictionary with layer_idx, head_idx (if applicable)
            
        Returns:
            Learning rate scale factor for the parameter
        """
        if 'layer_idx' in param_info and 'head_idx' in param_info:
            layer_idx = param_info['layer_idx']
            head_idx = param_info['head_idx']
            
            if (layer_idx, head_idx) in self.new_heads:
                return self.new_head_multiplier
        
        return 1.0
    
    def create_lr_scales(self, num_layers, num_heads):
        """
        Create a matrix of learning rate scales for all heads.
        
        Args:
            num_layers: Number of layers in the model
            num_heads: Number of heads per layer
            
        Returns:
            Array of shape [num_layers, num_heads] with learning rate scales
        """
        scales = np.ones((num_layers, num_heads), dtype=np.float32)
        
        for layer_idx, head_idx in self.new_heads:
            if 0 <= layer_idx < num_layers and 0 <= head_idx < num_heads:
                scales[layer_idx, head_idx] = self.new_head_multiplier
        
        return scales
    
    def save_state_dict(self):
        """
        Prepare a state dictionary for checkpointing.
        
        Returns:
            Dictionary with all necessary state
        """
        return {
            "base_lr": self.base_lr,
            "new_head_multiplier": self.new_head_multiplier,
            "new_heads": list(self.new_heads)
        }
    
    def load_state_dict(self, state_dict):
        """
        Load state from a checkpoint.
        
        Args:
            state_dict: Dictionary containing saved state
        """
        self.base_lr = state_dict.get("base_lr", self.base_lr)
        self.new_head_multiplier = state_dict.get("new_head_multiplier", self.new_head_multiplier)
        self.new_heads = set(state_dict.get("new_heads", []))