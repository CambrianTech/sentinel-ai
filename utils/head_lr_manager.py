"""
Head Learning Rate Manager for dynamic transformer architecture.

This module handles per-head learning rate adjustments during pruning and regrowth
operations, allowing newly activated or reactivated heads to learn more quickly.
"""

import torch
import numpy as np
from torch.optim import Optimizer


class HeadLRManager:
    """
    Manages per-head learning rates during training to enable faster adaptation
    of pruned or regrown attention heads.
    
    When architectural changes occur (pruning or regrowth), this class identifies
    the affected heads and adjusts their learning rates according to a predefined
    schedule, allowing them to adapt more quickly to their new role.
    
    Enhanced with agency awareness to respect head states and modify learning
    rates based on agency signals.
    """
    
    def __init__(
        self,
        model,
        optimizer: Optimizer,
        base_lr: float,
        boost_factor: float = 5.0,
        decay_factor: float = 0.9,
        warmup_steps: int = 200,
        cooldown_steps: int = 1000
    ):
        """
        Initialize the head learning rate manager.
        
        Args:
            model: The adaptive transformer model
            optimizer: PyTorch optimizer
            base_lr: Base learning rate for all parameters
            boost_factor: Factor to increase learning rate for newly activated heads
            decay_factor: Factor to decay the boosted learning rate each step
            warmup_steps: Number of steps for warming up newly activated heads
            cooldown_steps: Number of steps before returning to base learning rate
        """
        self.model = model
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.boost_factor = boost_factor
        self.decay_factor = decay_factor
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        
        # Track head status (prune/grow history)
        self.num_layers = len(model.blocks)
        self.num_heads = model.blocks[0]["attn"].num_heads
        
        # Initialize head status tracking
        # -1: pruned, 0: stable, >0: steps since activation
        self.head_status = np.zeros((self.num_layers, self.num_heads), dtype=np.int32)
        
        # Maps head indices to parameter group indices in optimizer
        self.head_to_param_mapping = self._build_head_parameter_mapping()
        
        # Current multipliers for each head
        self.lr_multipliers = np.ones((self.num_layers, self.num_heads), dtype=np.float32)
        
        # Agency LR modifiers - lookup table for different states
        self.agency_lr_modifiers = {
            "active": 1.0,     # Normal learning rate
            "overloaded": 0.5, # Reduced learning rate for overloaded heads
            "misaligned": 0.7, # Moderately reduced for misaligned heads
            "withdrawn": 0.0   # No learning for withdrawn heads (respects consent)
        }
        
        # Track agency-based modifications
        self.agency_modified_heads = set()
    
    def _build_head_parameter_mapping(self):
        """
        Build a mapping from (layer, head) to optimizer parameter groups.
        
        Returns:
            Dictionary mapping (layer_idx, head_idx) to list of parameter indices
        """
        mapping = {}
        
        # Build naming patterns for head parameters
        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                key = (layer_idx, head_idx)
                mapping[key] = []
                
                # Find parameters associated with this head
                # Each head has parameters in Q, K, V, and output projections
                head_patterns = [
                    f"blocks.{layer_idx}.attn.W_q.{head_idx}",
                    f"blocks.{layer_idx}.attn.W_k.{head_idx}",
                    f"blocks.{layer_idx}.attn.W_v.{head_idx}",
                    f"blocks.{layer_idx}.attn.W_o.{head_idx}"
                ]
                
                # Find matching parameter groups in optimizer
                for group_idx, group in enumerate(self.optimizer.param_groups):
                    for pattern in head_patterns:
                        if any(pattern in name for name, _ in group.get('named_params', [])):
                            mapping[key].append(group_idx)
                            break
        
        return mapping
    
    def update_head_status(self, gate_values, prev_gate_values=None):
        """
        Update the head status based on current and previous gate values.
        
        Args:
            gate_values: Current gate values tensor [num_layers, num_heads]
            prev_gate_values: Previous gate values tensor (optional)
        
        Returns:
            Dictionary with information about head changes
        """
        # Convert gate values to numpy for easier processing
        current_gates = gate_values.detach().cpu().numpy()
        
        # Default previous gates to current if not provided
        if prev_gate_values is None:
            prev_gates = current_gates
        else:
            prev_gates = prev_gate_values.detach().cpu().numpy()
        
        # Define thresholds for activation/deactivation
        active_threshold = 0.2
        inactive_threshold = 0.1
        
        # Track changes for reporting
        newly_activated = []
        newly_deactivated = []
        
        # Update head status for each layer and head
        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                current_gate = current_gates[layer_idx, head_idx]
                prev_gate = prev_gates[layer_idx, head_idx]
                
                # Check for newly activated heads (pruned -> active)
                if current_gate > active_threshold and prev_gate <= inactive_threshold:
                    self.head_status[layer_idx, head_idx] = 1  # Just activated
                    newly_activated.append((layer_idx, head_idx))
                
                # Check for newly deactivated heads (active -> pruned)
                elif current_gate <= inactive_threshold and prev_gate > active_threshold:
                    self.head_status[layer_idx, head_idx] = -1  # Pruned
                    newly_deactivated.append((layer_idx, head_idx))
                
                # Update counter for recently activated heads
                elif self.head_status[layer_idx, head_idx] > 0:
                    self.head_status[layer_idx, head_idx] += 1
                    
                    # Reset to stable after cooldown period
                    if self.head_status[layer_idx, head_idx] > self.cooldown_steps:
                        self.head_status[layer_idx, head_idx] = 0
        
        return {
            "newly_activated": newly_activated,
            "newly_deactivated": newly_deactivated,
            "cooling_down": np.sum(self.head_status > 0)
        }
    
    def apply_agency_modifiers(self, agency_state):
        """
        Apply learning rate modifiers based on agency state.
        
        This adjusts learning rates according to head states:
        - Overloaded heads get reduced learning rate (0.5x)
        - Misaligned heads get moderately reduced learning rate (0.7x)
        - Withdrawn heads get zero learning rate (respecting consent)
        
        Args:
            agency_state: Dictionary mapping (layer_idx, head_idx) to agency state info
            
        Returns:
            Dictionary with information about modifications
        """
        self.agency_modified_heads.clear()
        
        # Process each head with agency state
        for (layer_idx, head_idx), head_state in agency_state.items():
            if not (0 <= layer_idx < self.num_layers and 0 <= head_idx < self.num_heads):
                continue
                
            # Get head state and modifier
            state = head_state.get("state", "active")
            consent = head_state.get("consent", True)
            
            # If head has withdrawn consent, force zero learning rate
            if not consent or state == "withdrawn":
                modifier = 0.0
            else:
                # Otherwise use the state-based modifier
                modifier = self.agency_lr_modifiers.get(state, 1.0)
            
            # Store for future reference
            if modifier != 1.0:
                self.agency_modified_heads.add((layer_idx, head_idx))
        
        # Return info about modifications (actual application happens in update_learning_rates)
        return {
            "modified_heads": len(self.agency_modified_heads),
            "states": {state: count for state, count in 
                      [(state, sum(1 for (l, h) in self.agency_modified_heads 
                                 if (l, h) in agency_state and agency_state[(l, h)].get("state") == state))
                       for state in self.agency_lr_modifiers.keys()]}
        }
    
    def update_learning_rates(self):
        """
        Update learning rates for all heads based on their current status.
        
        This implements the logic of:
        - Boosting LR for newly activated heads (with warmup)
        - Gradually reducing the boosted LR over time
        - Returning to base LR after cooldown period
        - Applying agency-based modifiers (if available)
        
        Returns:
            Dictionary with info about learning rate changes
        """
        changes_made = False
        
        # Reset multipliers to base
        self.lr_multipliers.fill(1.0)
        
        # Update learning rates based on head status
        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                status = self.head_status[layer_idx, head_idx]
                
                # Skip stable or pruned heads unless agency-modified
                if status <= 0 and (layer_idx, head_idx) not in self.agency_modified_heads:
                    continue
                
                # Calculate multiplier based on time since activation
                if status <= self.warmup_steps and status > 0:
                    # Warmup phase: linearly increase from 1.0 to boost_factor
                    progress = status / self.warmup_steps
                    multiplier = 1.0 + (self.boost_factor - 1.0) * progress
                elif status > self.warmup_steps:
                    # Cooldown phase: exponentially decay back to 1.0
                    steps_after_warmup = status - self.warmup_steps
                    decay_progress = steps_after_warmup / (self.cooldown_steps - self.warmup_steps)
                    multiplier = max(
                        1.0,
                        self.boost_factor * (self.decay_factor ** steps_after_warmup)
                    )
                else:
                    # Default for stable or pruned heads
                    multiplier = 1.0
                
                # Apply agency modifier if this head is modified
                head_key = (layer_idx, head_idx)
                if head_key in self.agency_modified_heads:
                    # Get head state from the model if possible
                    if hasattr(self.model.blocks[layer_idx]["attn"], "agency_signals"):
                        agency_signals = self.model.blocks[layer_idx]["attn"].agency_signals
                        if head_idx in agency_signals:
                            state = agency_signals[head_idx].get("state", "active")
                            consent = agency_signals[head_idx].get("consent", True)
                            
                            # Apply corresponding modifier
                            if not consent or state == "withdrawn":
                                agency_modifier = 0.0
                            else:
                                agency_modifier = self.agency_lr_modifiers.get(state, 1.0)
                                
                            # Apply agency modifier (overrides other modifiers for withdrawn)
                            if agency_modifier == 0.0:
                                multiplier = 0.0
                            else:
                                multiplier *= agency_modifier
                
                # Apply the multiplier
                self.lr_multipliers[layer_idx, head_idx] = multiplier
                
                # Apply to optimizer parameter groups
                if head_key in self.head_to_param_mapping:
                    for group_idx in self.head_to_param_mapping[head_key]:
                        if group_idx < len(self.optimizer.param_groups):
                            self.optimizer.param_groups[group_idx]['lr'] = self.base_lr * multiplier
                            changes_made = True
        
        return {
            "changes_made": changes_made,
            "max_multiplier": np.max(self.lr_multipliers),
            "min_multiplier": np.min(self.lr_multipliers),
            "avg_multiplier": np.mean(self.lr_multipliers),
            "agency_modified_count": len(self.agency_modified_heads)
        }
    
    def get_lr_multipliers(self):
        """
        Get current learning rate multipliers for visualization.
        
        Returns:
            Numpy array of current multipliers [num_layers, num_heads]
        """
        return self.lr_multipliers.copy()
    
    def save_state_dict(self):
        """
        Prepare a state dictionary for checkpointing.
        
        Returns:
            Dictionary with all necessary state
        """
        return {
            "head_status": self.head_status.copy(),
            "lr_multipliers": self.lr_multipliers.copy(),
            "base_lr": self.base_lr,
            "boost_factor": self.boost_factor,
            "decay_factor": self.decay_factor,
            "warmup_steps": self.warmup_steps,
            "cooldown_steps": self.cooldown_steps,
            "agency_lr_modifiers": self.agency_lr_modifiers.copy(),
            "agency_modified_heads": list(self.agency_modified_heads)
        }
    
    def load_state_dict(self, state_dict):
        """
        Load state from a checkpoint.
        
        Args:
            state_dict: Dictionary containing saved state
        """
        self.head_status = state_dict["head_status"].copy()
        self.lr_multipliers = state_dict["lr_multipliers"].copy()
        self.base_lr = state_dict.get("base_lr", self.base_lr)
        self.boost_factor = state_dict.get("boost_factor", self.boost_factor)
        self.decay_factor = state_dict.get("decay_factor", self.decay_factor)
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        self.cooldown_steps = state_dict.get("cooldown_steps", self.cooldown_steps)
        
        # Load agency-related state
        if "agency_lr_modifiers" in state_dict:
            self.agency_lr_modifiers = state_dict["agency_lr_modifiers"].copy()
            
        if "agency_modified_heads" in state_dict:
            self.agency_modified_heads = set(state_dict["agency_modified_heads"])
        
        # Update optimizer with loaded multipliers
        self.update_learning_rates()