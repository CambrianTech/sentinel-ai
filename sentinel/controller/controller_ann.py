# sentinel/controller/controller_ann.py
"""
Implementation of the ANN-based Dynamic Controller as described in Section 3 of the paper.

The controller maintains learnable gate logits for each attention head and provides
mechanisms for dynamically updating these gates based on feedback metrics during training.
"""

import torch
from torch import nn

class ANNController(nn.Module):
    """
    ANN-based Dynamic Controller for the Adaptive Transformer.
    
    As described in the paper (Section 3):
    - Maintains learnable gate logits for each attention head
    - Provides mechanisms for adjusting gates based on feedback metrics
    - Implements L1 regularization to encourage sparsity
    
    This controller allows for both gradient-based learning and direct adjustment
    of gates based on runtime metrics like attention entropy and gradient norms.
    """
    def __init__(self, num_layers: int, num_heads: int, config: dict = None):
        """
        Initialize the ANN controller with learnable parameters for each attention head.
        
        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads per layer
            config: Optional configuration dict with parameters like 'init_value', 'reg_weight'
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        cfg = config or {}
        
        # Initialize gate logits with positive bias for active heads at the start
        # As mentioned in paper: "Initially, gates are biased towards 1 (active heads), 
        # allowing the model to gradually identify and prune less useful heads."
        init_val = cfg.get("init_value", 3.0)  # 3.0 -> sigmoid ~ 0.95
        
        # L1 regularization weight to encourage sparsity
        # Paper Section 4.1: "We incorporate an L1 regularization penalty on gate values."
        self.reg_weight = cfg.get("reg_weight", 1e-4)
        
        # Learnable parameters for gate logits (one per attention head per layer)
        # Paper Section 3.1: "The controller maintains learnable gate logits per head"
        self.gate_logits = nn.Parameter(init_val * torch.ones(num_layers, num_heads))
        
        # Note: More complex controllers could include neural network components
        # that take metrics as input to predict optimal gates, but this simpler version
        # relies on learned parameters and heuristic updates.

    def forward(self, metrics=None):
        """
        Compute the attention head gate values from learnable parameters.
        
        This implements the core functionality described in Section 3.1:
        "G = σ(GateLogits), G ∈ ℝ^(L×H)"
        
        Args:
            metrics: Optional dictionary of feedback metrics for dynamic adjustments
            
        Returns:
            Tensor of gate values for each attention head (shape: [num_layers, num_heads])
        """
        # Apply sigmoid to convert logits to gate values in range (0,1)
        gate_values = torch.sigmoid(self.gate_logits)  # Shape: [num_layers, num_heads]
        return gate_values

    def regularization_loss(self):
        """
        Calculate L1 regularization loss to encourage head pruning.
        
        As per paper Section 4.1: "We incorporate an L1 regularization penalty on 
        gate values to encourage sparsity and efficient pruning: 
        L_total = L_LM + λ_gate * Σ g_l,h"
        
        Returns:
            L1 norm of gate values (to be weighted by reg_weight)
        """
        return torch.sum(torch.sigmoid(self.gate_logits))

    def update_gates(self, metrics_dict):
        """
        Dynamically adjust gate logits based on runtime feedback metrics.
        
        This implements the controller feedback loop described in Section 3.2:
        "The controller updates gate logits periodically, applying heuristics:
        - Decrease gate logits where entropy is consistently high
        - Slightly reduce gate logits where gradients are consistently small"
        
        Enhanced with agency awareness to respect head states and consent.
        
        Args:
            metrics_dict: Dictionary containing metrics like 'entropy', 'grad_norm',
                         and agency-related metrics if available
        """
        with torch.no_grad():
            # Get learning rate for controller updates
            controller_lr = metrics_dict.get("controller_lr", torch.tensor(0.01)).item()
            
            # Check for agency state information
            has_agency_info = all(k in metrics_dict for k in ["agency_state", "consent", "utilization"])
            
            # Create masks for different agency states if available
            if has_agency_info:
                agency_state = metrics_dict["agency_state"]
                consent = metrics_dict["consent"]
                utilization = metrics_dict["utilization"]
                
                # Create masks for different states (active=3, misaligned=2, overloaded=1, withdrawn=0)
                withdrawn_mask = (agency_state == 0) | (consent == 0)
                overloaded_mask = (agency_state == 1) & (consent == 1)
                misaligned_mask = (agency_state == 2) & (consent == 1)
                active_mask = (agency_state == 3) & (consent == 1)
            else:
                # Default masks when no agency info is available
                withdrawn_mask = torch.zeros_like(self.gate_logits.data, dtype=torch.bool)
                overloaded_mask = torch.zeros_like(self.gate_logits.data, dtype=torch.bool)
                misaligned_mask = torch.zeros_like(self.gate_logits.data, dtype=torch.bool)
                active_mask = torch.ones_like(self.gate_logits.data, dtype=torch.bool)
            
            # AGENCY-AWARE GATE UPDATES
                        
            # 1. Respect withdrawn consent - don't update gates for withdrawn heads
            # This ensures controller doesn't try to reactivate withdrawn heads without consent
            update_mask = ~withdrawn_mask
            
            # 2. Update based on attention entropy - with agency awareness
            if 'entropy' in metrics_dict:
                entropy = metrics_dict['entropy']  # Shape: [num_layers, num_heads]
                entropy_threshold = metrics_dict.get('entropy_threshold', 1.5)
                
                # "Decrease gate logits where entropy is consistently high"
                high_entropy = entropy > entropy_threshold
                gate_active = torch.sigmoid(self.gate_logits.data) > 0.2
                
                # Reduce gate values for high-entropy, still-active heads
                # But respect agency: more aggressive for overloaded, less for misaligned
                base_adjustment = 0.1 * controller_lr
                
                # Standard adjustment for active heads
                standard_reduction = torch.where(
                    high_entropy & gate_active & active_mask & update_mask,
                    -base_adjustment,
                    torch.zeros_like(self.gate_logits.data)
                )
                
                # More aggressive adjustment for overloaded heads (showing entropy issues)
                overloaded_reduction = torch.where(
                    high_entropy & gate_active & overloaded_mask & update_mask,
                    -1.5 * base_adjustment,  # 50% more reduction
                    torch.zeros_like(self.gate_logits.data)
                )
                
                # Less aggressive for misaligned heads (might recover)
                misaligned_reduction = torch.where(
                    high_entropy & gate_active & misaligned_mask & update_mask,
                    -0.8 * base_adjustment,  # 20% less reduction
                    torch.zeros_like(self.gate_logits.data)
                )
                
                # Combine all adjustments
                entropy_adjustment = standard_reduction + overloaded_reduction + misaligned_reduction
                self.gate_logits.data = self.gate_logits.data + entropy_adjustment
            
            # 3. Update based on gradient norm - with agency awareness
            if 'grad_norm' in metrics_dict:
                grad_norm = metrics_dict['grad_norm']
                grad_threshold = metrics_dict.get('grad_threshold', 1e-3)
                
                # "Slightly reduce gate logits where gradients are consistently small"
                low_grad = grad_norm < grad_threshold
                gate_active = torch.sigmoid(self.gate_logits.data) > 0.2
                
                # Base adjustment with agency awareness
                base_adjustment = 0.05 * controller_lr
                
                # Standard adjustment for active heads
                standard_reduction = torch.where(
                    low_grad & gate_active & active_mask & update_mask,
                    -base_adjustment,
                    torch.zeros_like(self.gate_logits.data)
                )
                
                # More aggressive for overloaded (may be saturated)
                overloaded_reduction = torch.where(
                    low_grad & gate_active & overloaded_mask & update_mask,
                    -1.5 * base_adjustment,
                    torch.zeros_like(self.gate_logits.data)
                )
                
                # Combine adjustments
                grad_adjustment = standard_reduction + overloaded_reduction
                self.gate_logits.data = self.gate_logits.data + grad_adjustment
            
            # 4. Consider head importance for potential regrowth - with agency awareness
            if 'head_importance' in metrics_dict:
                importance = metrics_dict['head_importance']
                importance_threshold = metrics_dict.get('importance_threshold', 0.7)
                
                # Increase gate values for inactive but potentially important heads
                # But respect consent - never try to reactivate withdrawn heads
                gate_inactive = torch.sigmoid(self.gate_logits.data) < 0.1
                high_importance = importance > importance_threshold
                
                # Allow regrowth only for non-withdrawn heads
                regrowth_candidates = gate_inactive & high_importance & ~withdrawn_mask
                
                # Adjust boost based on current state
                base_adjustment = 0.2 * controller_lr
                
                # Boost inactive heads with consent (respecting withdrawal)
                self.gate_logits.data = torch.where(
                    regrowth_candidates,
                    self.gate_logits.data + base_adjustment,
                    self.gate_logits.data
                )
            
            # 5. If we have utilization data, use it for adaptive adjustment
            if has_agency_info and 'utilization' in metrics_dict:
                # High utilization might indicate overload risk
                high_utilization = utilization > 0.85
                gate_value = torch.sigmoid(self.gate_logits.data)
                
                # For highly utilized active heads, slightly reduce gates
                # to prevent future overload (preemptive protection)
                utilization_adjustment = 0.03 * controller_lr
                self.gate_logits.data = torch.where(
                    high_utilization & (gate_value > 0.7) & active_mask & update_mask,
                    self.gate_logits.data - utilization_adjustment,
                    self.gate_logits.data
                )
            
            # Stability measure: clamp gate logits to prevent numerical issues
            self.gate_logits.data.clamp_(-10, 10)