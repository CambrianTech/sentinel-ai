"""
Learning rate management for attention heads.

This module provides functionality to apply different learning rates to different
attention heads, particularly useful for newly grown heads.
"""

from typing import List, Tuple, Dict, Optional


class HeadLRManager:
    """
    Manager for head-specific learning rates.
    Allows differential learning rates for different heads, especially useful
    for newly added heads that need higher learning rates.
    """
    
    def __init__(
        self, 
        base_lr: float = 1e-4,
        new_head_multiplier: float = 5.0,
        new_heads: Optional[List[Tuple[int, int]]] = None,
        head_multipliers: Optional[Dict[Tuple[int, int], float]] = None
    ):
        """
        Initialize the manager.
        
        Args:
            base_lr: Base learning rate for all heads
            new_head_multiplier: Multiplier for newly added heads
            new_heads: List of (layer_idx, head_idx) tuples for new heads
            head_multipliers: Dict mapping (layer_idx, head_idx) to LR multipliers
        """
        self.base_lr = base_lr
        self.new_head_multiplier = new_head_multiplier
        self.new_heads = new_heads or []
        self.head_multipliers = head_multipliers or {}
        
        # Apply default multiplier to new heads if not specified
        for head in self.new_heads:
            if head not in self.head_multipliers:
                self.head_multipliers[head] = self.new_head_multiplier
    
    def get_lr(self, layer_idx: int, head_idx: int) -> float:
        """
        Get the learning rate for a specific head.
        
        Args:
            layer_idx: Layer index
            head_idx: Head index
            
        Returns:
            Learning rate for the head
        """
        head = (layer_idx, head_idx)
        multiplier = self.head_multipliers.get(head, 1.0)
        return self.base_lr * multiplier
    
    def update_multipliers(self, step: int, warmup_steps: int, cooldown_steps: int = 1000):
        """
        Update multipliers over time (e.g., gradually reduce high LRs for new heads).
        
        Args:
            step: Current training step
            warmup_steps: Steps for warmup
            cooldown_steps: Steps to gradually return to base LR
        """
        # Simple linear cooldown for new heads
        if step > warmup_steps:
            cooldown_factor = max(0.0, min(1.0, (step - warmup_steps) / cooldown_steps))
            remaining_ratio = 1.0 - cooldown_factor
            
            for head in self.new_heads:
                original_multiplier = self.new_head_multiplier
                current_multiplier = 1.0 + (original_multiplier - 1.0) * remaining_ratio
                self.head_multipliers[head] = current_multiplier
    
    def get_param_groups(self, model) -> List[Dict]:
        """
        Create parameter groups for an optimizer.
        
        Args:
            model: The model with head parameters
            
        Returns:
            List of parameter groups for optimizer
        """
        # This is a placeholder implementation
        # In a real implementation, would separate out head parameters
        # and assign them to parameter groups with appropriate LRs
        
        param_groups = [
            {
                "params": model.parameters(),
                "lr": self.base_lr
            }
        ]
        
        return param_groups