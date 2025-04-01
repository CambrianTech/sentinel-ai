# controller/controller_ann.py
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
        
        Args:
            metrics_dict: Dictionary containing metrics like 'entropy' and 'grad_norm'
                         with shapes matching [num_layers, num_heads]
        """
        with torch.no_grad():
            # Update based on attention entropy
            # High entropy indicates less specialized/useful heads
            if 'entropy' in metrics_dict:
                entropy = metrics_dict['entropy']  # Shape: [num_layers, num_heads]
                entropy_threshold = metrics_dict.get('entropy_threshold', 1.5)
                
                # Paper: "Decrease gate logits where entropy is consistently high"
                high_entropy = entropy > entropy_threshold
                gate_active = torch.sigmoid(self.gate_logits.data) > 0.2
                
                # Reduce gate values for high-entropy, still-active heads
                self.gate_logits.data = torch.where(
                    high_entropy & gate_active,
                    self.gate_logits.data - 0.1,  # Reduce logit (reduces gate value)
                    self.gate_logits.data
                )
            
            # Update based on gradient norm
            # Low gradient norms suggest saturated learning (as per Section 3.2)
            if 'grad_norm' in metrics_dict:
                grad_norm = metrics_dict['grad_norm']
                grad_threshold = metrics_dict.get('grad_threshold', 1e-3)
                
                # Paper: "Slightly reduce gate logits where gradients are consistently small"
                low_grad = grad_norm < grad_threshold
                gate_active = torch.sigmoid(self.gate_logits.data) > 0.2
                
                # Reduce gate values for low-gradient, still-active heads
                self.gate_logits.data = torch.where(
                    low_grad & gate_active,
                    self.gate_logits.data - 0.05,  # Smaller reduction for gradient-based updates
                    self.gate_logits.data
                )
            
            # Stability measure: clamp gate logits to prevent numerical issues
            self.gate_logits.data.clamp_(-10, 10)
