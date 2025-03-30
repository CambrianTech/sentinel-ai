# controller/controller_ann.py
import torch
from torch import nn

class ANNController(nn.Module):
    def __init__(self, num_layers: int, num_heads: int, config: dict = None):
        """
        ANN Controller for adaptive transformer.
        Keeps per-head gate parameters and updates them based on metrics.
        config can include 'init_value', 'reg_weight', etc.
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        cfg = config or {}
        init_val = cfg.get("init_value", 3.0)  # initial gate logits (3.0 -> sigmoid ~ 0.95)
        self.reg_weight = cfg.get("reg_weight", 1e-4)  # L1 regularization weight for gates
        # Learnable logits for gates
        self.gate_logits = nn.Parameter(init_val * torch.ones(num_layers, num_heads))
        # Optionally, one could define additional neural network components here 
        # that take metrics as input to adjust gates, e.g. an MLP, but not required for now.

    def forward(self, metrics=None):
        """
        Compute the head mask from gate parameters.
        Optionally use input metrics to adjust gates (if a dynamic mechanism is used).
        """
        gate_values = torch.sigmoid(self.gate_logits)  # shape (num_layers, num_heads)
        # If metrics are provided, one could adjust gate_values here or compute a different output.
        # For simplicity, this controller relies mainly on learned parameters + optional external updates.
        return gate_values

    def regularization_loss(self):
        """
        L1 regularization encouraging smaller gate values (head pruning).
        Multiply this by reg_weight when adding to main loss.
        """
        # We use mean or sum of gate values as a sparsity penalty.
        return torch.sum(torch.sigmoid(self.gate_logits))

    def update_gates(self, metrics_dict):
        """
        Adjust gate logits based on external metrics (non-gradient updates).
        metrics_dict: expected keys might be 'entropy' or 'grad_norm' with values as 
        tensors of shape (num_layers, num_heads) or (num_layers, num_heads).
        This method is called periodically to nudge gates.
        """
        with torch.no_grad():
            if 'entropy' in metrics_dict:
                entropy = metrics_dict['entropy']  # assumed shape [num_layers, num_heads]
                # Example heuristic: if entropy is high and gate is high, reduce the gate a bit
                high_entropy = entropy > metrics_dict.get('entropy_threshold', 1.5)
                # We subtract a small amount from gate_logits where high entropy and gate is not already near zero
                self.gate_logits.data = torch.where(
                    high_entropy & (torch.sigmoid(self.gate_logits.data) > 0.2),
                    self.gate_logits.data - 0.1,  # reduce logit (this will reduce gate value)
                    self.gate_logits.data
                )
            if 'grad_norm' in metrics_dict:
                # (Placeholder for potential use: e.g., if grad_norm is very low for a head, reduce its gate)
                grad_norm = metrics_dict['grad_norm']
                low_grad = grad_norm < metrics_dict.get('grad_threshold', 1e-3)
                self.gate_logits.data = torch.where(
                    low_grad & (torch.sigmoid(self.gate_logits.data) > 0.2),
                    self.gate_logits.data - 0.05,
                    self.gate_logits.data
                )
            # Ensure gate_logits don't go to -inf or inf (clamp for stability)
            self.gate_logits.data.clamp_(-10, 10)
