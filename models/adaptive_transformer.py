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

        self.W_q = nn.ModuleList([nn.Linear(embed_dim, self.head_dim, bias=True) for _ in range(num_heads)])
        self.W_k = nn.ModuleList([nn.Linear(embed_dim, self.head_dim, bias=True) for _ in range(num_heads)])
        self.W_v = nn.ModuleList([nn.Linear(embed_dim, self.head_dim, bias=True) for _ in range(num_heads)])
        self.W_o = nn.ModuleList([nn.Linear(self.head_dim, embed_dim, bias=True) for _ in range(num_heads)])

        self.gate = nn.Parameter(torch.ones(num_heads))

        print(f"[Attention] W_q[0]={self.W_q[0].weight.shape}, W_o[0]={self.W_o[0].weight.shape}")

    def forward(self, hidden_states, attn_mask=None):
        B, T = hidden_states.shape[:2]
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
                scores = scores + attn_mask

            weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(weights, V)

            projected = self.W_o[i](output) * self.gate[i]
            outputs.append(projected)

        return sum(outputs)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_dim=None):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = 4 * embed_dim

        self.dense_in = nn.Linear(embed_dim, ffn_dim, bias=True)
        self.act = nn.GELU()
        self.dense_out = nn.Linear(ffn_dim, embed_dim, bias=True)

        print(f"[FFN] dense_in={self.dense_in.weight.shape}, dense_out={self.dense_out.weight.shape}")

    def forward(self, x):
        return self.dense_out(self.act(self.dense_in(x)))


class AdaptiveTransformerModel(nn.Module):
    def __init__(self, config, token_embeddings, position_embeddings):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size if hasattr(config, 'hidden_size') else config.n_embd
        self.num_heads = config.num_attention_heads if hasattr(config, 'num_attention_heads') else config.n_head
        self.num_layers = config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else config.n_layer

        ffn_dim = getattr(config, 'n_inner', None) or getattr(config, 'intermediate_size', None) or 4 * self.embed_dim

        print(f"[Model Init] embed_dim={self.embed_dim}, num_heads={self.num_heads}, ffn_dim={ffn_dim}")

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.blocks = nn.ModuleList()
        midpoint = self.num_layers // 2

        for i in range(self.num_layers):
            print(f"[Block {i}] Initializing...")
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
        self.lm_head.weight = self.wte.weight

        self.register_buffer("bias", torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024))

    def forward(self, input_ids, attention_mask=None, **kwargs):
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)

        encoder_outputs = {}
        midpoint = self.num_layers // 2

        for i, block in enumerate(self.blocks):
            h = block["ln1"](hidden_states)

            if seq_len <= 1024:
                causal_mask = self.bias[:, :, :seq_len, :seq_len]
                attn_mask = (1.0 - causal_mask) * -10000.0
                attn_mask = attn_mask.squeeze(0).squeeze(0)
            else:
                attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * -10000.0, diagonal=1)

            attn_out = block["attn"](h, attn_mask=attn_mask)
            hidden_states = hidden_states + attn_out

            if i < midpoint:
                encoder_outputs[i] = hidden_states.clone()

            h2 = block["ln2"](hidden_states)
            ffn_out = block["ffn"](h2)
            hidden_states = hidden_states + ffn_out

            if i >= midpoint:
                encoder_layer = self.num_layers - i - 1
                if encoder_layer in encoder_outputs:
                    enc_out = encoder_outputs[encoder_layer]
                    fused = torch.cat([hidden_states, enc_out], dim=-1)
                    hidden_states = hidden_states + block["skip_fuse"](fused)

        hidden_states = self.ln_f(hidden_states)
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
