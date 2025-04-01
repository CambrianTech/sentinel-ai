import torch
import numpy as np
from torch import nn
from .controller_ann import ANNController
from .metrics.head_metrics import collect_head_metrics

class ControllerManager:
    """
    Manager class that coordinates the dynamic controller, metrics collection,
    and gate updates during the training process.
    
    This class handles:
    1. Initializing the controller
    2. Scheduling periodic controller updates 
    3. Collecting head metrics
    4. Applying the controller decisions to model gates
    5. Managing the overall pruning strategy
    """
    
    def __init__(self, model, config=None):
        """
        Initialize the controller manager with a model and configuration.
        
        Args:
            model: The adaptive transformer model
            config: Configuration dictionary with controller settings
        """
        self.model = model
        self.config = config or {}
        
        # Extract model dimensions
        self.num_layers = len(model.blocks)
        self.num_heads = model.blocks[0]["attn"].num_heads
        
        # Initialize the controller
        controller_type = self.config.get("controller_type", "ann")
        if controller_type == "ann":
            self.controller = ANNController(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                config=self.config.get("controller_config", {})
            )
        else:
            raise ValueError(f"Unsupported controller type: {controller_type}")
        
        # Move controller to the same device as the model
        self.controller.to(next(model.parameters()).device)
        
        # Initialize update settings
        self.update_frequency = self.config.get("update_frequency", 100)  # Update every N steps
        self.current_step = 0
        self.warmup_steps = self.config.get("warmup_steps", 500)  # Wait before pruning
        self.max_pruned_heads = self.config.get("max_pruned_heads", 0.3)  # Max fraction to prune
        
        # Track gate activity for monitoring
        self.gate_history = []
    
    def step(self, metrics_dict=None, dataloader=None, loss_fn=None):
        """
        Perform one step of the controller manager, potentially updating gates.
        
        Args:
            metrics_dict: Optional pre-computed metrics dictionary
            dataloader: Optional dataloader for computing metrics
            loss_fn: Optional loss function for computing importance metrics
            
        Returns:
            Dictionary with update information
        """
        self.current_step += 1
        device = next(self.model.parameters()).device
        
        # Skip updates during warmup phase
        if self.current_step < self.warmup_steps:
            return {"status": "warmup", "step": self.current_step}
        
        # Check if it's time for an update
        if self.current_step % self.update_frequency != 0:
            return {"status": "skipped", "step": self.current_step}
        
        # Collect metrics if not provided
        if metrics_dict is None:
            metrics_dict = collect_head_metrics(
                self.model, 
                dataloader=dataloader,
                loss_fn=loss_fn,
                device=device
            )
        
        # Update controller gates based on metrics
        self.controller.update_gates(metrics_dict)
        
        # Get updated gate values from controller
        gate_values = self.controller.forward()
        
        # Apply gates to model
        self._apply_gates_to_model(gate_values)
        
        # Track gate activity
        self.gate_history.append(self._get_active_gates())
        
        return {
            "status": "updated",
            "step": self.current_step,
            "active_gates": self._get_active_gates(),
            "pruned_percent": self._get_pruned_percent()
        }
    
    def _apply_gates_to_model(self, gate_values):
        """
        Apply the controller's gate values to the model's attention gates.
        
        Args:
            gate_values: Tensor of gate values from controller [num_layers, num_heads]
        """
        with torch.no_grad():
            for layer_idx, block in enumerate(self.model.blocks):
                # Copy gate values to model's attention module
                block["attn"].gate.copy_(gate_values[layer_idx])
    
    def _get_active_gates(self):
        """
        Get the count of active gates (gate value > threshold) per layer.
        
        Returns:
            Dictionary mapping layer indices to lists of active head indices
        """
        active_gates = {}
        threshold = self.config.get("active_threshold", 0.1)
        
        for layer_idx, block in enumerate(self.model.blocks):
            active_heads = []
            for head_idx in range(block["attn"].num_heads):
                if block["attn"].gate[head_idx].item() > threshold:
                    active_heads.append(head_idx)
            active_gates[layer_idx] = active_heads
        
        return active_gates
    
    def _get_pruned_percent(self):
        """
        Calculate the percentage of pruned heads.
        
        Returns:
            Float representing the percentage of pruned heads
        """
        total_heads = self.num_layers * self.num_heads
        active_heads = sum(len(heads) for heads in self._get_active_gates().values())
        return 100.0 * (total_heads - active_heads) / total_heads
    
    def get_regularization_loss(self):
        """
        Get the current regularization loss from the controller.
        
        Returns:
            Tensor representing the regularization loss term
        """
        return self.controller.regularization_loss() * self.controller.reg_weight
    
    def enable_unet_connections(self, enable=True, connection_scale=None):
        """
        Enable or disable U-Net style skip connections in the model.
        
        Args:
            enable: Whether to enable the connections
            connection_scale: Optional scaling factor for skip connections
                             If None, use default progressive scaling
        """
        if not hasattr(self.model, 'blocks'):
            return
        
        # Default progressive scaling: deeper decoder layers get weaker connections
        midpoint = self.num_layers // 2
        
        with torch.no_grad():
            for i in range(midpoint, self.num_layers):
                # Skip connection from encoder to decoder
                decoder_index = i
                encoder_index = self.num_layers - i - 1
                
                if connection_scale is None:
                    # Calculate how deep we are in the decoder (0.0 to 1.0)
                    decoder_progress = (i - midpoint) / (self.num_layers - midpoint)
                    
                    # Use a diminishing scale for deeper layers
                    scale = 0.1 * (1.0 - decoder_progress * 0.5)
                else:
                    scale = connection_scale
                
                # Enable or disable in model config
                if enable:
                    self.model.blocks[decoder_index].use_skip_connection = True
                    self.model.blocks[decoder_index].skip_source = encoder_index
                    self.model.blocks[decoder_index].skip_scale = scale
                else:
                    self.model.blocks[decoder_index].use_skip_connection = False
    
    def save_state_dict(self):
        """
        Get a state dictionary for saving the controller state.
        
        Returns:
            Dictionary containing all necessary state
        """
        return {
            "controller": self.controller.state_dict(),
            "config": self.config,
            "current_step": self.current_step,
            "gate_history": self.gate_history,
        }
    
    def load_state_dict(self, state_dict):
        """
        Load controller state from a state dictionary.
        
        Args:
            state_dict: Dictionary containing saved state
        """
        self.controller.load_state_dict(state_dict["controller"])
        self.config = state_dict.get("config", self.config)
        self.current_step = state_dict.get("current_step", 0)
        self.gate_history = state_dict.get("gate_history", [])
        
        # Apply the loaded gates to the model
        self._apply_gates_to_model(self.controller())