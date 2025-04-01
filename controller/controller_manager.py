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
        
        # Handle different model structures (original dict-style vs optimized object-style)
        if hasattr(model.blocks[0], 'attn'):
            # New optimized model structure
            self.num_heads = model.blocks[0].attn.num_heads
        else:
            # Original structure
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
        
        # Verbosity control
        self.quiet = self.config.get("quiet", False)
    
    def step(self, metrics_dict=None, dataloader=None, loss_fn=None, head_lr_manager=None, agency_state=None):
        """
        Perform one step of the controller manager, potentially updating gates.
        
        Args:
            metrics_dict: Optional pre-computed metrics dictionary
            dataloader: Optional dataloader for computing metrics
            loss_fn: Optional loss function for computing importance metrics
            head_lr_manager: Optional manager for per-head learning rates
            agency_state: Optional dictionary mapping (layer_idx, head_idx) to agency state
                         information, used to adjust gate values based on head states
            
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
        
        # If agency state is provided, add it to metrics for controller awareness
        if agency_state is not None:
            # Convert agency state to tensors for controller processing
            agency_metrics = self._process_agency_state(agency_state, device)
            # Add agency metrics to the metrics dictionary
            metrics_dict.update(agency_metrics)
        
        # Get current gate values before update (for tracking changes)
        prev_gate_values = self.controller.forward()
        
        # Update controller gates based on metrics (now includes agency awareness)
        self.controller.update_gates(metrics_dict)
        
        # Get updated gate values from controller
        gate_values = self.controller.forward()
        
        # Apply gates to model (with agency awareness)
        self._apply_gates_to_model(gate_values, agency_state)
        
        # Update per-head learning rates if manager is provided
        head_lr_info = {}
        if head_lr_manager is not None:
            # Update head status based on gate changes
            head_status_info = head_lr_manager.update_head_status(gate_values, prev_gate_values)
            
            # Apply agency-aware learning rate adjustments if agency state is provided
            if agency_state is not None and hasattr(head_lr_manager, 'apply_agency_modifiers'):
                head_lr_manager.apply_agency_modifiers(agency_state)
            
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
        
        # If agency state is provided, emit state signals for bidirectional awareness
        agency_signals = {}
        if agency_state is not None and self.current_step > self.warmup_steps + 100:
            agency_signals = self._emit_state_signals(gate_values, agency_state)
        
        return {
            "status": "updated",
            "step": self.current_step,
            "active_gates": self._get_active_gates(),
            "pruned_percent": self._get_pruned_percent(),
            "controller_lr": self.controller_lr,
            "early_stopping_triggered": early_stopping_triggered,
            "plateau_counter": self.plateau_counter,
            "head_lr_info": head_lr_info,
            "gate_changes": torch.sum(torch.abs(gate_values - prev_gate_values)).item(),
            "agency_signals": agency_signals
        }
    
    def _process_agency_state(self, agency_state, device):
        """
        Process agency state information into a format usable by the controller.
        
        Args:
            agency_state: Dictionary mapping (layer_idx, head_idx) to agency state information
            device: PyTorch device for tensor creation
            
        Returns:
            Dictionary with agency metrics in tensor form for controller consumption
        """
        # Initialize tensors to track different agency states
        agency_tensor = torch.zeros((self.num_layers, self.num_heads), device=device)
        consent_tensor = torch.ones((self.num_layers, self.num_heads), device=device)
        utilization_tensor = torch.zeros((self.num_layers, self.num_heads), device=device)
        
        # State mapping for numeric representation
        state_values = {
            "active": 3.0,
            "misaligned": 2.0,
            "overloaded": 1.0,
            "withdrawn": 0.0
        }
        
        # Process each head's agency information
        for (layer_idx, head_idx), head_state in agency_state.items():
            # Extract state information
            state = head_state.get("state", "active")
            consent = head_state.get("consent", True)
            utilization = head_state.get("utilization", 0.0)
            
            # Convert to numeric representations
            state_value = state_values.get(state, 3.0)  # Default to active if unknown
            consent_value = 1.0 if consent else 0.0
            
            # Update tensors
            if 0 <= layer_idx < self.num_layers and 0 <= head_idx < self.num_heads:
                agency_tensor[layer_idx, head_idx] = state_value
                consent_tensor[layer_idx, head_idx] = consent_value
                utilization_tensor[layer_idx, head_idx] = utilization
        
        # Return as metrics dictionary for controller use
        return {
            "agency_state": agency_tensor,
            "consent": consent_tensor,
            "utilization": utilization_tensor
        }
        
    def _emit_state_signals(self, gate_values, agency_state):
        """
        Emit state change signals to attention heads based on gate values.
        
        This implements bidirectional awareness where the controller can suggest
        state changes to heads based on sustained gate patterns.
        
        Args:
            gate_values: Tensor of gate values from controller [num_layers, num_heads]
            agency_state: Dictionary mapping (layer_idx, head_idx) to agency state information
            
        Returns:
            Dictionary with information about emitted signals
        """
        signals_emitted = []
        gate_values_np = gate_values.detach().cpu().numpy()
        
        # Look for patterns that suggest state changes
        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                gate_value = gate_values_np[layer_idx, head_idx]
                head_key = (layer_idx, head_idx)
                
                # Skip if head not in agency state dict
                if head_key not in agency_state:
                    continue
                    
                current_state = agency_state[head_key].get("state", "active")
                
                # Propose state changes based on gate patterns
                proposed_state = None
                
                # Very low gate value suggests withdrawal
                if gate_value < 0.05 and current_state != "withdrawn":
                    proposed_state = "withdrawn"
                    
                # Oscillating gate suggests misalignment
                elif hasattr(self, '_prev_gate_values') and 0.1 < gate_value < 0.4:
                    prev_gate = self._prev_gate_values[layer_idx, head_idx] \
                        if hasattr(self, '_prev_gate_values') else 0.0
                    
                    gate_change = abs(gate_value - prev_gate)
                    if gate_change > 0.1 and current_state != "misaligned":
                        proposed_state = "misaligned"
                        
                # High gate value but previously withdrawn suggests reactivation
                elif gate_value > 0.7 and current_state == "withdrawn":
                    proposed_state = "active"
                
                # If we have a proposed state change, signal the head
                if proposed_state:
                    # Get the attention module based on model structure
                    block = self.model.blocks[layer_idx]
                    if hasattr(block, 'attn'):
                        # New optimized model structure
                        attn_module = block.attn
                    else:
                        # Original structure
                        attn_module = block["attn"]
                    
                    # Direct signal to the head if possible
                    if hasattr(attn_module, "set_head_state"):
                        attn_module.set_head_state(
                            head_idx, proposed_state, None  # Don't change consent
                        )
                        signals_emitted.append({
                            "layer": layer_idx,
                            "head": head_idx,
                            "from_state": current_state,
                            "to_state": proposed_state,
                            "gate_value": float(gate_value)
                        })
        
        # Save current gate values for change detection next time
        self._prev_gate_values = gate_values.detach().clone()
        
        return {
            "signals_emitted": signals_emitted,
            "count": len(signals_emitted)
        }
    
    def _apply_gates_to_model(self, gate_values, agency_state=None):
        """
        Apply the controller's gate values to the model's attention gates,
        with respect to agency state if provided.
        
        Args:
            gate_values: Tensor of gate values from controller [num_layers, num_heads]
            agency_state: Optional dictionary mapping (layer_idx, head_idx) to agency state
        """
        with torch.no_grad():
            # If no agency state, use simple assignment
            if agency_state is None:
                for layer_idx, block in enumerate(self.model.blocks):
                    # Handle different model structures
                    if hasattr(block, 'attn'):
                        # New optimized model structure
                        block.attn.gate.copy_(gate_values[layer_idx])
                    else:
                        # Original structure
                        block["attn"].gate.copy_(gate_values[layer_idx])
                return
                
            # With agency state, respect head agency and consent
            for layer_idx, block in enumerate(self.model.blocks):
                # Handle different model structures
                if hasattr(block, 'attn'):
                    # New optimized model structure
                    attn_module = block.attn
                else:
                    # Original structure
                    attn_module = block["attn"]
                    
                for head_idx in range(self.num_heads):
                    head_key = (layer_idx, head_idx)
                    gate_value = gate_values[layer_idx, head_idx]
                    
                    # Apply normal gate value if head not in agency state
                    if head_key not in agency_state:
                        attn_module.gate[head_idx].copy_(gate_value)
                        continue
                        
                    # Get head agency state
                    head_state = agency_state[head_key]
                    state = head_state.get("state", "active")
                    consent = head_state.get("consent", True)
                    
                    # Respect withdrawn consent - set gate to zero regardless of controller
                    if not consent or state == "withdrawn":
                        attn_module.gate[head_idx].zero_()
                        # Log if this was a significant change (potential consent violation)
                        if gate_value > 0.5 and hasattr(attn_module, "_log_consent_violation"):
                            attn_module._log_consent_violation(
                                head_idx, "controller gate override prevented", self.current_step
                            )
                    # Adjust gate value based on state for non-withdrawn heads
                    elif state == "overloaded":
                        # Reduce gate value for overloaded heads
                        adjusted_gate = gate_value * 0.5
                        attn_module.gate[head_idx].copy_(adjusted_gate)
                    elif state == "misaligned":
                        # Reduce gate value for misaligned heads
                        adjusted_gate = gate_value * 0.7
                        attn_module.gate[head_idx].copy_(adjusted_gate)
                    else:
                        # Normal assignment for active heads
                        attn_module.gate[head_idx].copy_(gate_value)
    
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
            # Handle different model structures
            if hasattr(block, 'attn'):
                # New optimized model structure
                attn_module = block.attn
            else:
                # Original structure
                attn_module = block["attn"]
                
            for head_idx in range(attn_module.num_heads):
                if attn_module.gate[head_idx].item() > threshold:
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
                
                # Get the block
                block = self.model.blocks[decoder_index]
                
                # Enable or disable in model config based on model structure
                if enable:
                    if hasattr(block, 'use_skip_connection'):
                        # Direct attribute access (new optimized structure)
                        block.use_skip_connection = True
                        block.skip_source = encoder_index
                        block.skip_scale = scale
                    else:
                        # Dictionary access (original structure)
                        block.use_skip_connection = True
                        block.skip_source = encoder_index
                        block.skip_scale = scale
                else:
                    if hasattr(block, 'use_skip_connection'):
                        # Direct attribute access
                        block.use_skip_connection = False
                    else:
                        # Dictionary access
                        block.use_skip_connection = False
    
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
        
        # Log the learning rate update if not in quiet mode
        if not self.quiet:
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
                # Always print early stopping messages, even in quiet mode
                # as this is a significant event in the training process
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