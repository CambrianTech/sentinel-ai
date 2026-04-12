"""cross_attention.py — Cross-attention injection for Many-Worlds substrate.

Instead of blindly adding a delta to the residual stream (where it's
30-300× smaller than the hidden state norm and gets ignored), we inject
the substrate output via cross-attention. The target model QUERIES the
substrate field — "what does the source model know about this token?"

This is how every working multi-modal architecture does it:
  - Flamingo: cross-attention to vision features
  - LLaVA: cross-attention to CLIP embeddings
  - Q-Former: learned queries attend to frozen encoder

The substrate field IS another modality. The target model attends to it
the way it would attend to image tokens or audio features.

Architecture:
    Source model layer L_s → Project → substrate μ → field (seq, substrate_dim)
                                                        ↓
    Target model layer L_t → [self-attn] → [CROSS-ATTN] → [FFN]
                                              ↑
                              Q from target hidden (target_dim → head_dim)
                              K,V from substrate field (substrate_dim → head_dim)

The cross-attention block is small (~1M params) and learned during
substrate training. It replaces the additive residual injection.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubstrateCrossAttention(nn.Module):
    """Cross-attention from target model's residual stream to substrate field.

    The target model queries the substrate: "what relevant information
    does the source model's representation contain for this token?"

    Unlike additive injection, cross-attention:
    1. Doesn't compete with residual magnitude (separate pathway)
    2. Lets the target model SELECT what to pull (attention weights)
    3. Naturally handles sequence length mismatches (different tokenizers)
    4. Has a learned gate that starts at zero (no disruption at init)
    """

    def __init__(
        self,
        target_hidden_dim: int,
        substrate_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.target_hidden_dim = target_hidden_dim
        self.substrate_dim = substrate_dim
        self.num_heads = num_heads
        self.head_dim = substrate_dim // num_heads
        assert substrate_dim % num_heads == 0, f"substrate_dim {substrate_dim} not divisible by num_heads {num_heads}"

        # Query: target hidden → head space
        self.q_proj = nn.Linear(target_hidden_dim, substrate_dim, bias=False)
        # Key/Value: substrate field → head space
        self.k_proj = nn.Linear(substrate_dim, substrate_dim, bias=False)
        self.v_proj = nn.Linear(substrate_dim, substrate_dim, bias=False)
        # Output: head space → target hidden
        self.out_proj = nn.Linear(substrate_dim, target_hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Init: standard Xavier on Q/K/V so attention patterns form immediately.
        # Small Xavier on out_proj — the out_proj magnitude IS the gate.
        # No separate gate parameter. The network learns how much to inject
        # through the out_proj weights directly. This avoids the degenerate
        # optimization where a separate gate stays closed while the network
        # learns to align a zero-magnitude vector (cheap trick, no learning).
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)  # starts small, grows

    def forward(
        self,
        target_hidden: torch.Tensor,    # (batch, tgt_seq, target_hidden_dim)
        substrate_field: torch.Tensor,   # (batch, src_seq, substrate_dim)
    ) -> torch.Tensor:
        """Cross-attend from target to substrate field.

        Returns:
            delta: (batch, tgt_seq, target_hidden_dim) — to be ADDED
                to the target residual stream. Gated by self.gate
                so it starts at zero and grows during training.
        """
        B, T, _ = target_hidden.shape
        S = substrate_field.shape[1]

        # Project queries from target, keys/values from substrate
        Q = self.q_proj(target_hidden)   # (B, T, substrate_dim)
        K = self.k_proj(substrate_field)  # (B, S, substrate_dim)
        V = self.v_proj(substrate_field)  # (B, S, substrate_dim)

        # Reshape for multi-head attention
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, D)
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, D)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, T, S)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Attend to values
        out = torch.matmul(attn, V)  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, self.substrate_dim)  # (B, T, substrate_dim)

        # Project back to target hidden dim
        out = self.out_proj(out)  # (B, T, target_hidden_dim)

        # No separate gate — the out_proj magnitude is the learned gate.
        return out


class SubstrateCrossAttentionHook:
    """Manages the forward hook that injects cross-attention at a target layer.

    Usage during training:
        hook = SubstrateCrossAttentionHook(cross_attn, target_model, layer_idx)
        hook.set_substrate_field(field)  # from source model's projection
        # ... run target model forward/backward ...
        hook.remove()

    Usage during eval:
        hook = SubstrateCrossAttentionHook(cross_attn, target_model, layer_idx)
        hook.set_substrate_field(field)
        output = target_model.generate(...)
        hook.remove()
    """

    def __init__(
        self,
        cross_attn: SubstrateCrossAttention,
        target_model,
        layer_idx: int,
    ):
        self.cross_attn = cross_attn
        self.layer_idx = layer_idx
        self._substrate_field: Optional[torch.Tensor] = None

        # Find layers — unwrap PEFT if needed
        m = target_model
        if hasattr(m, 'base_model'):  # PEFT wrapper
            m = m.base_model
        if hasattr(m, 'model') and hasattr(m.model, 'model') and hasattr(m.model.model, 'layers'):
            layers = m.model.model.layers  # PEFT → base → model → layers
        elif hasattr(m, 'model') and hasattr(m.model, 'layers'):
            layers = m.model.layers
        elif hasattr(m, 'transformer') and hasattr(m.transformer, 'h'):
            layers = m.transformer.h
        elif hasattr(m, 'model') and hasattr(m.model, 'transformer') and hasattr(m.model.transformer, 'h'):
            layers = m.model.transformer.h  # PEFT → base → transformer → h
        else:
            raise RuntimeError(f"Can't find layers in {type(target_model)}")

        self._handle = layers[layer_idx].register_forward_hook(self._hook_fn)

    def set_substrate_field(self, field: torch.Tensor):
        """Set the substrate field for the next forward pass."""
        self._substrate_field = field

    def _hook_fn(self, module, input, output):
        if self._substrate_field is None:
            return output

        hidden = output[0] if isinstance(output, tuple) else output
        B, T, D = hidden.shape

        # Expand substrate field to match batch size if needed
        field = self._substrate_field
        if field.shape[0] == 1 and B > 1:
            field = field.expand(B, -1, -1)

        # Cross-attend
        delta = self.cross_attn(hidden.float(), field.float())
        hidden = hidden + delta.to(hidden.dtype)

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    def remove(self):
        self._handle.remove()
