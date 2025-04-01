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
    6. Scheduling learning rate adjustments for controller
    7. Implementing early stopping based on gate activity
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
        
        # Initialize learning rate scheduling for controller
        self.controller_lr = self.config.get("controller_lr", 0.01)
        self.controller_lr_decay = self.config.get("controller_lr_decay", 0.95)
        self.controller_lr_decay_steps = self.config.get("controller_lr_decay_steps", 1000)
        self.min_controller_lr = self.config.get("min_controller_lr", 0.001)
        
        # Initialize early stopping parameters
        self.patience = self.config.get("early_stopping_patience", 5)  # Number of checks before stopping
        self.min_gate_change = self.config.get("min_gate_change", 0.01)  # Minimum change to continue
        self.plateau_counter = 0
        self.last_avg_gate_value = None
        self.early_stopping_active = self.config.get("enable_early_stopping", True)
        
        # Track gate activity for monitoring
        self.gate_history = []
    
    def step(self, metrics_dict=None, dataloader=None, loss_fn=None, head_lr_manager=None):
        """
        Perform one step of the controller manager, potentially updating gates.
        
        Args:
            metrics_dict: Optional pre-computed metrics dictionary
            dataloader: Optional dataloader for computing metrics
            loss_fn: Optional loss function for computing importance metrics
            head_lr_manager: Optional manager for per-head learning rates
            
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
        
        # Update controller learning rate (decay over time)
        if self.current_step % self.controller_lr_decay_steps == 0:
            self._update_controller_learning_rate()
        
        # Collect metrics if not provided
        if metrics_dict is None:
            metrics_dict = collect_head_metrics(
                self.model, 
                dataloader=dataloader,
                loss_fn=loss_fn,
                device=device
            )
        
        # Add learning rate to metrics before updating controller
        metrics_dict["controller_lr"] = torch.tensor(self.controller_lr, device=device)
        
        # Get current gate values before update (for tracking changes)
        prev_gate_values = self.controller.forward()
        
        # Update controller gates based on metrics
        self.controller.update_gates(metrics_dict)
        
        # Get updated gate values from controller
        gate_values = self.controller.forward()
        
        # Apply gates to model
        self._apply_gates_to_model(gate_values)
        
        # Update per-head learning rates if manager is provided
        head_lr_info = {}
        if head_lr_manager is not None:
            # Update head status based on gate changes
            head_status_info = head_lr_manager.update_head_status(gate_values, prev_gate_values)
            
            # Update learning rates based on status
            lr_update_info = head_lr_manager.update_learning_rates()
            
            # Combine all info for reporting
            head_lr_info = {
                "head_status": head_status_info,
                "lr_updates": lr_update_info
            }
        
        # Track gate activity
        self.gate_history.append(self._get_active_gates())
        
        # Check for early stopping based on gate activity plateau
        early_stopping_triggered = False
        if self.early_stopping_active:
            early_stopping_triggered = self._check_gate_activity_plateau(gate_values)
        
        return {
            "status": "updated",
            "step": self.current_step,
            "active_gates": self._get_active_gates(),
            "pruned_percent": self._get_pruned_percent(),
            "controller_lr": self.controller_lr,
            "early_stopping_triggered": early_stopping_triggered,
            "plateau_counter": self.plateau_counter,
            "head_lr_info": head_lr_info,
            "gate_changes": torch.sum(torch.abs(gate_values - prev_gate_values)).item()
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
    
    def _update_controller_learning_rate(self):
        """
        Update the learning rate for controller updates according to the decay schedule.
        
        This reduces the magnitude of gate updates over time as the model converges,
        providing more stability in the latter stages of training.
        """
        # Apply decay to current learning rate
        self.controller_lr = max(
            self.controller_lr * self.controller_lr_decay,
            self.min_controller_lr
        )
        
        # Log the learning rate update
        if self.controller_lr <= self.min_controller_lr:
            print(f"🔄 Controller learning rate reached minimum value: {self.controller_lr:.6f}")
        else:
            print(f"🔄 Controller learning rate updated: {self.controller_lr:.6f}")
            
    def _check_gate_activity_plateau(self, gate_values):
        """
        Check if gate activity has plateaued over multiple updates.
        
        This implements early stopping based on gate activity to prevent
        oscillations and stabilize the pruning process.
        
        Args:
            gate_values: Current gate values from controller
            
        Returns:
            Boolean indicating whether early stopping should be triggered
        """
        # Calculate average gate value across all active gates
        avg_gate = torch.mean(gate_values).item()
        
        # Skip if this is the first check
        if self.last_avg_gate_value is None:
            self.last_avg_gate_value = avg_gate
            return False
        
        # Calculate absolute change in average gate value
        gate_change = abs(avg_gate - self.last_avg_gate_value)
        
        # Check if change is below threshold
        if gate_change < self.min_gate_change:
            self.plateau_counter += 1
            if self.plateau_counter >= self.patience:
                print(f"⚠️ Gate activity plateau detected! Early stopping controller updates.")
                print(f"   Average gate value: {avg_gate:.4f}, Change: {gate_change:.4f}")
                return True
        else:
            # Reset counter if we see significant changes
            self.plateau_counter = 0
        
        # Update reference value for next check
        self.last_avg_gate_value = avg_gate
        return False
    
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
            "controller_lr": self.controller_lr,
            "plateau_counter": self.plateau_counter,
            "last_avg_gate_value": self.last_avg_gate_value
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
        self.controller_lr = state_dict.get("controller_lr", self.controller_lr)
        self.plateau_counter = state_dict.get("plateau_counter", 0)
        self.last_avg_gate_value = state_dict.get("last_avg_gate_value", None)
        
        # Apply the loaded gates to the model
        self._apply_gates_to_model(self.controller())