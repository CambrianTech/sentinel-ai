"""
Optimized Attention Implementation for Adaptive Transformer

This module provides an optimized implementation of the gated multi-head
self-attention mechanism with agency features. The key optimizations include:
- Batched processing of attention heads
- Fused tensor operations
- Vectorized agency checks
- Memory-efficient implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Dict, List, Optional, Tuple, Union


class OptimizedGatedMultiHeadAttention(nn.Module):
    """
    Optimized implementation of Gated Multi-Head Self-Attention with Agency features.
    
    This implementation processes all heads in parallel using batched operations,
    significantly improving performance over the original sequential implementation.
    It maintains all the agency features while being much more efficient.
    
    Key optimizations:
    1. Parallel head processing
    2. Fused QKV projection
    3. Vectorized agency operations
    4. Improved memory usage
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_prob: float = 0.1,
        qkv_bias: bool = False,
        debug: bool = False
    ):
        super().__init__()
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.debug = debug
        
        # Fused query, key, value projections
        # This is more efficient than separate projections
        self.fused_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=qkv_bias)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        
        # Dropout for attention
        self.attn_dropout = nn.Dropout(dropout_prob)
        
        # Initialize sentinel gates (one per head)
        self.gate = nn.Parameter(torch.ones(num_heads))
        
        # Agency tracking
        self.agency_signals = {}
        self.consent_violations = {}
        
        # Flag to store attention weights
        self.store_attention_weights = False
        self.attention_weights = {}
    
    def set_head_state(self, head_idx: int, state: str, consent: Optional[bool] = None):
        """
        Set agency state for a specific attention head.
        
        Args:
            head_idx: Index of the attention head
            state: State to set ("active", "overloaded", "misaligned", "withdrawn")
            consent: Whether the head consents to updates (if None, unchanged)
        """
        if head_idx >= self.num_heads:
            raise ValueError(f"Invalid head index: {head_idx}, max: {self.num_heads-1}")
            
        # Initialize if this is the first signal for this head
        if head_idx not in self.agency_signals:
            self.agency_signals[head_idx] = {
                "state": "active",
                "consent": True,
                "utilization": 0.0,
                "timestamp": time.time()
            }
        
        # Update state
        self.agency_signals[head_idx]["state"] = state
        self.agency_signals[head_idx]["timestamp"] = time.time()
        
        # Update consent if provided
        if consent is not None:
            self.agency_signals[head_idx]["consent"] = consent
            
        # For withdrawn state, automatically update gate value
        if state == "withdrawn" or (consent is not None and not consent):
            with torch.no_grad():
                self.gate[head_idx] = 0.0
    
    def _log_consent_violation(self, head_idx: int, violation_type: str, step: Optional[int] = None):
        """Log potential consent violations for analysis."""
        if head_idx not in self.consent_violations:
            self.consent_violations[head_idx] = []
            
        self.consent_violations[head_idx].append({
            "type": violation_type,
            "timestamp": time.time(),
            "step": step
        })
    
    def get_effective_gates(self):
        """
        Get effective gate values accounting for agency states.
        
        Returns:
            Tensor of effective gate values
        """
        # Start with raw gate values
        effective_gates = self.gate.clone()
        
        # Apply agency state modifications
        for head_idx, signals in self.agency_signals.items():
            # Withdrawn state or withdrawn consent means zero gate
            if signals["state"] == "withdrawn" or not signals["consent"]:
                effective_gates[head_idx] = 0.0
            # Modify gate based on state
            elif signals["state"] == "overloaded":
                effective_gates[head_idx] *= 0.5
            elif signals["state"] == "misaligned":
                effective_gates[head_idx] *= 0.8
        
        return effective_gates
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for optimized gated multi-head attention.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, 1, 1, seq_len]
            head_mask: Optional mask for specific heads [batch_size, num_heads, seq_len, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Get effective gate values accounting for agency states
        effective_gates = self.get_effective_gates()
        
        # Check if all gates are closed (very rare but possible)
        if torch.all(effective_gates < 1e-10):
            # Return zeros if all gates are closed
            return torch.zeros_like(hidden_states)
        
        # Compute fused QKV projections
        qkv = self.fused_qkv(hidden_states)
        
        # Reshape: [batch, seq, 3*hidden] -> [batch, seq, 3, heads, head_dim]
        new_shape = qkv.size()[:-1] + (3, self.num_heads, self.head_dim)
        qkv = qkv.view(*new_shape)
        
        # Transpose: [batch, seq, 3, heads, head_dim] -> [3, batch, heads, seq, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        # Separate Q, K, V
        q, k, v = qkv
        
        # Compute scaled dot-product attention
        # [batch, heads, seq, head_dim] x [batch, heads, head_dim, seq] -> [batch, heads, seq, seq]
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # Apply head mask if provided
        if head_mask is not None:
            attention_scores = attention_scores + head_mask
            
        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        
        # Store attention weights if requested
        if self.store_attention_weights:
            self.attention_weights = {h: attention_probs[:, h:h+1] for h in range(self.num_heads)}
        
        # Update utilization metrics for each head
        for head_idx in range(self.num_heads):
            if head_idx in self.agency_signals:
                # Calculate average activation as a proxy for utilization
                activation_level = attention_probs[:, head_idx].abs().mean().item()
                
                # Update exponential moving average of utilization
                prev_util = self.agency_signals[head_idx].get("utilization", 0.0)
                alpha = 0.9  # Smoothing factor
                new_util = alpha * prev_util + (1-alpha) * activation_level
                self.agency_signals[head_idx]["utilization"] = new_util
        
        # Apply attention to values
        # [batch, heads, seq, seq] x [batch, heads, seq, head_dim] -> [batch, heads, seq, head_dim]
        context = torch.matmul(attention_probs, v)
        
        # Transpose: [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim]
        context = context.transpose(1, 2).contiguous()
        
        # Reshape: [batch, seq, heads, head_dim] -> [batch, seq, hidden]
        context = context.view(batch_size, seq_len, self.hidden_size)
        
        # Apply gates - more efficient vectorized version
        # Unsqueeze gates to allow broadcasting
        # [num_heads] -> [1, 1, num_heads, 1] -> [batch, seq, hidden]
        gate_weights = effective_gates.view(1, 1, self.num_heads, 1).expand(batch_size, seq_len, self.num_heads, self.head_dim)
        gate_weights = gate_weights.reshape(batch_size, seq_len, self.hidden_size)
        
        # Apply gates element-wise
        context = context * gate_weights
        
        # Output projection and dropout
        output = self.out_proj(context)
        
        # Optional normalization for stable outputs with many pruned heads
        active_gates = torch.sum(effective_gates > 1e-4).item()
        if active_gates < self.num_heads:
            # Normalize by the ratio of active heads to total heads
            output = output * (active_gates / self.num_heads)
        
        return output

# Alias for backward compatibility
OptimizedAttention = OptimizedGatedMultiHeadAttention