"""qformer.py — Q-Former bridge from substrate field to soft prompt tokens.

The Q-Former pattern (from BLIP-2) solves exactly our problem: bridge a
frozen encoder's output to a frozen LLM's input. Learned query tokens
attend to the full substrate field via cross-attention, each query
extracting a DIFFERENT aspect of the source model's knowledge.

This replaces the dumb linear projection (SubstrateToSoftPrompt) which
mapped one pooled vector into 16 identical-information tokens.

Architecture:
    substrate field (seq, substrate_dim) — per-token, NOT pooled
         ↓ K, V
    learned queries (num_queries, query_dim) → cross-attention → self-attention
         ↓
    output projection → (num_queries, target_embed_dim)
         ↓
    soft prompt tokens for the target model

Each query learns to extract a different semantic aspect:
    query 0: "what data structures are involved"
    query 1: "what algorithm pattern is this"
    query 2: "what edge cases matter"
    ...etc (learned, not hand-designed)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubstrateQFormer(nn.Module):
    """Q-Former that bridges substrate field to target model embedding space.

    Learned queries cross-attend to the substrate field, then self-attend
    to share information between queries, then project to target embed dim.
    """

    def __init__(
        self,
        substrate_dim: int,
        target_embed_dim: int,
        num_queries: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.substrate_dim = substrate_dim
        self.target_embed_dim = target_embed_dim

        # Learned query tokens — these are the "questions" the Q-Former
        # asks of the substrate field. Initialized as learnable parameters.
        self.queries = nn.Parameter(torch.randn(1, num_queries, substrate_dim) * 0.02)

        # Q-Former layers: each layer has cross-attention + self-attention + FFN
        self.layers = nn.ModuleList([
            QFormerLayer(substrate_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Vocabulary-grounded output: instead of projecting to an arbitrary
        # vector in embedding space, project to LOGITS over the target model's
        # vocabulary, then softmax → weighted sum of REAL token embeddings.
        # Every soft token is a "mixture word" that the model already knows
        # how to process, because it's made of words the model was trained on.
        #
        # This is the adapter pattern from Continuum: the output must be in a
        # format the CONSUMER understands. The consumer (Phi-2) understands
        # token embeddings. We translate substrate concepts into Phi-2's
        # native vocabulary.
        self.norm = nn.LayerNorm(substrate_dim)
        self.vocab_proj = nn.Linear(substrate_dim, target_embed_dim)
        # vocab_proj outputs are used to compute attention over the embedding table
        # (not a direct vocab logit — we attend in embedding space)

        nn.init.xavier_uniform_(self.vocab_proj.weight, gain=0.1)
        nn.init.zeros_(self.vocab_proj.bias)

        # The target model's embedding table — set via set_embedding_table()
        # after loading the target model. NOT a parameter (frozen).
        self.register_buffer("embed_table", torch.zeros(1, 1))  # placeholder
        self._embed_table_set = False

        # Confidence gate — learns WHEN to contribute.
        # Takes the Q-Former's processed queries and produces a per-query
        # scalar confidence in [0, 1]. When the substrate has useful info,
        # confidence is high. When the target already knows the answer,
        # confidence is low and the soft tokens fade toward the mean
        # embedding (effectively becoming padding that the model ignores).
        self.confidence_head = nn.Sequential(
            nn.Linear(substrate_dim, substrate_dim // 4),
            nn.GELU(),
            nn.Linear(substrate_dim // 4, 1),
        )
        # Initialize slightly negative so confidence starts LOW (~0.3)
        # The gate must EARN the right to contribute by proving it helps
        nn.init.zeros_(self.confidence_head[2].weight)
        nn.init.constant_(self.confidence_head[2].bias, -1.0)  # sigmoid(-1) ≈ 0.27

    def set_target_model(self, target_model, target_tokenizer, embed_layer):
        """Set the target model for input-conditioned confidence gating.

        The Q-Former measures the target model's uncertainty on each input
        and gates its contribution accordingly. High uncertainty = gate open.
        Low uncertainty = gate closed, output fades to neutral.
        """
        self._target_model = target_model
        self._target_tok = target_tokenizer
        self._embed_layer = embed_layer

    def set_embedding_table(self, embed_weight: torch.Tensor):
        """Set the target model's embedding table (frozen, not trained).

        Call this after loading the target model:
            qformer.set_embedding_table(target_model.embed_tokens.weight)
        """
        self.embed_table = embed_weight.detach()
        self._embed_table_set = True

    def forward(self, substrate_fields, target_input_ids=None) -> torch.Tensor:
        """
        Args:
            substrate_fields: either a single tensor (batch, src_seq, substrate_dim)
                or a LIST of tensors from N source models. When multiple fields
                are provided, they're concatenated along the sequence dimension
                so the queries can attend to ALL source models' knowledge
                simultaneously. The attention weights naturally learn which
                source tokens from which model are most relevant per query.

        Returns:
            soft_tokens: (batch, num_queries, target_embed_dim)
                Ready to prepend to the target model's input embeddings.
        """
        # Handle single tensor or list of tensors
        if isinstance(substrate_fields, (list, tuple)):
            # Concatenate all source fields: [(B, seq_1, D), (B, seq_2, D), ...] → (B, seq_total, D)
            substrate_field = torch.cat(substrate_fields, dim=1)
        else:
            substrate_field = substrate_fields

        B = substrate_field.shape[0]

        # Expand learned queries for the batch
        queries = self.queries.expand(B, -1, -1)  # (B, num_queries, substrate_dim)

        # Pass through Q-Former layers — queries attend to ALL source models
        for layer in self.layers:
            queries = layer(queries, substrate_field)

        # Vocabulary-grounded output: each query → attention weights over
        # the target model's real token embeddings → weighted sum.
        # The result is guaranteed to be in the target model's embedding
        # space because it IS a combination of real embeddings.
        queries = self.norm(queries)
        query_proj = self.vocab_proj(queries)  # (B, num_queries, target_embed_dim)

        # Compute attention weights over the vocabulary
        # query_proj: (B, Q, D) @ embed_table.T: (D, V) → (B, Q, V)
        vocab = self.embed_table.float()  # (V, D) — frozen target embeddings
        attn_logits = torch.matmul(query_proj, vocab.t())  # (B, Q, V)

        # Temperature-scaled softmax — sharp attention picks specific tokens,
        # smooth attention blends many tokens. Learned temperature.
        attn_logits = attn_logits / (self.target_embed_dim ** 0.5)
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, Q, V)

        # Weighted sum of real embeddings — result IS in embedding space
        soft_tokens = torch.matmul(attn_weights, vocab)  # (B, Q, D)

        # INPUT-CONDITIONED confidence gate.
        # The target model's own uncertainty determines how much the
        # substrate contributes. Computed from the target's hidden states
        # on this specific input — not a learned global scalar.
        #
        # High target uncertainty (doesn't know the answer) → gate opens
        # Low target uncertainty (already knows) → gate stays shut
        if target_input_ids is not None and hasattr(self, '_target_model') and self._target_model is not None:
            with torch.no_grad():
                tgt_out = self._target_model(target_input_ids)
                # Entropy of last token's prediction = uncertainty
                logits = tgt_out.logits[0, -1].float()
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum()
                max_entropy = torch.log(torch.tensor(float(logits.shape[-1])))
                # uncertainty in [0, 1]: 0 = certain, 1 = maximally uncertain
                uncertainty = (entropy / max_entropy).clamp(0, 1)

            # Scale soft tokens by uncertainty — confident inputs get near-zero
            # contribution, uncertain inputs get full substrate signal
            soft_tokens = soft_tokens * uncertainty
        else:
            # Fallback: use learned per-query confidence (training mode)
            confidence = torch.sigmoid(self.confidence_head(queries))
            vocab_mean = vocab.mean(dim=0, keepdim=True)
            soft_tokens = confidence * soft_tokens + (1 - confidence) * vocab_mean

        return soft_tokens


class QFormerLayer(nn.Module):
    """One Q-Former layer: cross-attention → self-attention → FFN."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        assert dim % num_heads == 0

        # Cross-attention: queries attend to substrate field
        self.cross_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.cross_norm = nn.LayerNorm(dim)

        # Self-attention: queries attend to each other (share info)
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.self_norm = nn.LayerNorm(dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim * 4, dim),
        )
        self.ffn_norm = nn.LayerNorm(dim)

        # Init FFN output near zero for residual stability
        nn.init.xavier_uniform_(self.ffn[0].weight)
        nn.init.xavier_uniform_(self.ffn[-1].weight, gain=0.1)

    def forward(self, queries: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: (B, num_queries, dim)
            kv: (B, src_seq, dim) — substrate field

        Returns:
            updated queries: (B, num_queries, dim)
        """
        # Cross-attention to substrate field (pre-norm)
        q_normed = self.cross_norm(queries)
        queries = queries + self.cross_attn(q_normed, kv, kv)

        # Self-attention between queries (pre-norm)
        q_normed = self.self_norm(queries)
        queries = queries + self.self_attn(q_normed, q_normed, q_normed)

        # FFN (pre-norm)
        q_normed = self.ffn_norm(queries)
        queries = queries + self.ffn(q_normed)

        return queries


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, T, _ = q.shape
        S = k.shape[1]

        Q = self.q_proj(q).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(k).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(v).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)
