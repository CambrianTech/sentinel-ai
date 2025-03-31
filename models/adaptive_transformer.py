import torch
from torch import nn
import math
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

class GatedMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Important: Match GPT2's dimension ordering for weight matrices
        # GPT2 has weights as (input_dim, output_dim)
        self.W_q = nn.ModuleList([nn.Linear(embed_dim, self.head_dim, bias=True) for _ in range(num_heads)])
        self.W_k = nn.ModuleList([nn.Linear(embed_dim, self.head_dim, bias=True) for _ in range(num_heads)])
        self.W_v = nn.ModuleList([nn.Linear(embed_dim, self.head_dim, bias=True) for _ in range(num_heads)])
        self.W_o = nn.ModuleList([nn.Linear(self.head_dim, embed_dim, bias=True) for _ in range(num_heads)])

        # Sentinel gate for each attention head
        self.gate = nn.Parameter(torch.ones(num_heads))

        # Print dimensions for debugging
        print(f"Attention: q_weight={self.W_q[0].weight.shape}, o_weight={self.W_o[0].weight.shape}")

    def forward(self, hidden_states, attn_mask=None):
        # Get batch size and sequence length
        shape = hidden_states.size()
        B, T = shape[0], shape[1]  # Safe way to extract first two dimensions
        outputs = []

        for i in range(self.num_heads):
            if float(self.gate[i]) < 1e-6:
                outputs.append(torch.zeros(B, T, self.embed_dim, device=hidden_states.device))
                continue

            Q = self.W_q[i](hidden_states)
            K = self.W_k[i](hidden_states)
            V = self.W_v[i](hidden_states)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attn_mask is not None:
                scores = scores + attn_mask  # Use per-head mask

            weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(weights, V)

            projected = self.W_o[i](output) * self.gate[i]
            outputs.append(projected)

        return sum(outputs)  # More efficient than torch.stack(outputs).sum(0)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_dim=None):
        super().__init__()
        # Determine feed-forward dimension
        if ffn_dim is None:
            # Default to 4x embed_dim, which is common in transformers
            ffn_dim = 4 * embed_dim
            
        # Important: Match GPT2's dimension ordering
        # GPT2 uses (input_dim, output_dim) for weight matrices
        self.dense_in = nn.Linear(embed_dim, ffn_dim, bias=True)
        self.act = nn.GELU()
        self.dense_out = nn.Linear(ffn_dim, embed_dim, bias=True)

        # Print dimensions for debugging
        print(f"FeedForward: in_weight={self.dense_in.weight.shape}, out_weight={self.dense_out.weight.shape}")

    def forward(self, x):
        return self.dense_out(self.act(self.dense_in(x)))


class AdaptiveTransformerModel(nn.Module):
    def __init__(self, config, token_embeddings, position_embeddings):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size if hasattr(config, 'hidden_size') else config.n_embd
        self.num_heads = config.num_attention_heads if hasattr(config, 'num_attention_heads') else config.n_head
        self.num_layers = config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else config.n_layer

        # Determine FFN dimension from config
        ffn_dim = None
        if hasattr(config, 'n_inner') and config.n_inner is not None:
            ffn_dim = config.n_inner
        elif hasattr(config, 'intermediate_size'):
            ffn_dim = config.intermediate_size
        else:
            # Default to 4x hidden size if not specified
            ffn_dim = 4 * self.embed_dim
            
        # Print for debugging
        print(f"Model initialization: embed_dim={self.embed_dim}, num_heads={self.num_heads}, ffn_dim={ffn_dim}")
        
        # Use token and position embeddings passed in
        self.wte = token_embeddings
        self.wpe = position_embeddings

        # Build transformer blocks
        self.blocks = nn.ModuleList()
        midpoint = self.num_layers // 2

        for layer_idx in range(self.num_layers):
            attn = GatedMultiHeadSelfAttention(self.embed_dim, self.num_heads)
            ffn = FeedForward(self.embed_dim, ffn_dim)
            block = nn.ModuleDict({
                "attn": attn,
                "ffn": ffn,
                "ln1": nn.LayerNorm(self.embed_dim),
                "ln2": nn.LayerNorm(self.embed_dim),
                "skip_fuse": nn.Linear(2 * self.embed_dim, self.embed_dim)
            })
            self.blocks.append(block)

        self.ln_f = nn.LayerNorm(self.embed_dim)
        self.lm_head = nn.Linear(self.embed_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # Weight tying

        # Register causal attention mask
        self.register_buffer("bias", torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024))

    def forward(self, input_ids, attention_mask=None, **kwargs):
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get position ids
        position_ids = torch.arange(seq_len, device=device)
        position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
        
        # Apply token and position embeddings
        inputs_embeds = self.wte(input_ids) + self.wpe(position_ids)
        hidden_states = inputs_embeds

        # Store encoder outputs for U-Net style skip connections
        encoder_outputs = {}
        midpoint = self.num_layers // 2

        # Process through transformer blocks
        for i, block in enumerate(self.blocks):
            # First layernorm before attention (pre-norm architecture)
            h = block["ln1"](hidden_states)
            
            # Create causal attention mask
            attn_mask = None
            if seq_len <= 1024:  # Use pre-computed mask if possible
                causal_mask = self.bias[:, :, :seq_len, :seq_len]
                # Convert to -inf for masked positions
                attn_mask = (1.0 - causal_mask) * -10000.0
                # Reshape to match expected format for attention
                attn_mask = attn_mask.squeeze(0).squeeze(0)  # shape: (seq_len, seq_len)
            else:  # Create mask on the fly for longer sequences
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=device) * -10000.0, 
                    diagonal=1
                )
                attn_mask = causal_mask

            # Self-attention with gating
            attn_out = block["attn"](h, attn_mask=attn_mask)
            hidden_states = hidden_states + attn_out

            # Store first half outputs for skip connections
            if i < midpoint:
                encoder_outputs[i] = hidden_states.clone()

            # Second layernorm before FFN
            h2 = block["ln2"](hidden_states)
            ffn_out = block["ffn"](h2)
            hidden_states = hidden_states + ffn_out

            # U-Net style skip connections for second half of layers
            if i >= midpoint:
                encoder_layer = self.num_layers - i - 1
                if encoder_layer in encoder_outputs:
                    enc_out = encoder_outputs[encoder_layer]
                    # Concatenate current and skip features then project back to hidden size
                    fused = torch.cat([hidden_states, enc_out], dim=-1)
                    hidden_states = block["skip_fuse"](fused)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        return logits


class AdaptiveCausalLmWrapper(AdaptiveTransformerModel, GenerationMixin):
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False

    def __init__(self, config, token_embeddings, position_embeddings):
        super().__init__(config, token_embeddings, position_embeddings)
        self.config = config
        from transformers import GenerationConfig
        try:
            self.generation_config = GenerationConfig.from_model_config(config)
        except Exception as e:
            print(f"Warning: Could not create generation config, falling back to defaults: {e}")
            self.generation_config = GenerationConfig()

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        logits = super().forward(input_ids, attention_mask=attention_mask)
        return CausalLMOutput(logits=logits) if return_dict else logits

    def can_generate(self):
        return True

    @property
    def _supports_cache_class(self):
        return False

    @property
    def device(self):
        return next(self.parameters()).device