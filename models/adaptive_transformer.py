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

        self.W_q = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_heads)])
        self.W_k = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_heads)])
        self.W_v = nn.ModuleList([nn.Linear(embed_dim, self.head_dim) for _ in range(num_heads)])
        self.W_o = nn.ModuleList([nn.Linear(self.head_dim, embed_dim) for _ in range(num_heads)])

        self.gate = nn.Parameter(torch.ones(num_heads))

    def forward(self, hidden_states, attn_mask=None):
        B, T, _ = hidden_states.size()
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
                scores += attn_mask[:, i]  # Use per-head mask

            weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(weights, V)

            projected = self.W_o[i](output) * self.gate[i]
            outputs.append(projected)

        return torch.stack(outputs).sum(0)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_dim):
        super().__init__()
        self.dense_in = nn.Linear(embed_dim, ffn_dim)
        self.act = nn.GELU()
        self.dense_out = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        return self.dense_out(self.act(self.dense_in(x)))


class AdaptiveTransformerModel(nn.Module):
    def __init__(self, config, token_embeddings, position_embeddings):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head
        self.num_layers = config.n_layer

        self.wte = token_embeddings
        self.wpe = position_embeddings

        self.blocks = nn.ModuleList()
        midpoint = self.num_layers // 2

        for _ in range(self.num_layers):
            attn = GatedMultiHeadSelfAttention(self.embed_dim, self.num_heads)
            ffn = FeedForward(self.embed_dim, config.n_inner if config.n_inner else 4 * self.embed_dim)
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

        self.register_buffer("bias", torch.tril(torch.ones(1024, 1024)).view(1,1,1024,1024))

    def forward(self, input_ids, attention_mask=None, **kwargs):
        bsz, seq_len = input_ids.shape
        inputs_embeds = self.wte(input_ids) + self.wpe(torch.arange(seq_len, device=input_ids.device))
        hidden_states = inputs_embeds

        encoder_outputs = {}
        midpoint = self.num_layers // 2

        for i, block in enumerate(self.blocks):
            h = block["ln1"](hidden_states)
            attn_mask = self.bias[:, :, :seq_len, :seq_len]  # [1, 1, T, T]
            attn_mask = (attn_mask == 0).to(hidden_states.dtype) * -1e9  # [1, 1, T, T]
            attn_mask = attn_mask.expand(hidden_states.size(0), self.num_heads, seq_len, seq_len)  # [B, H, T, T]
            attn_out = block["attn"](h, attn_mask=attn_mask)

            hidden_states = hidden_states + attn_out

            if i < midpoint:
                encoder_outputs[i] = hidden_states

            h2 = block["ln2"](hidden_states)
            ffn_out = block["ffn"](h2)
            hidden_states = hidden_states + ffn_out

            if i >= midpoint:
                encoder_layer = self.num_layers - i - 1
                if encoder_layer in encoder_outputs:
                    enc_out = encoder_outputs[encoder_layer]
                    fused = torch.cat([hidden_states, enc_out], dim=-1)
                    hidden_states = block["skip_fuse"](fused)

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
        self.generation_config = GenerationConfig.from_model_config(config)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        logits = super().forward(input_ids, attention_mask=attention_mask)
        return CausalLMOutput(logits=logits)

    def can_generate(self):
        return True

    @property
    def _supports_cache_class(self):
        return False

    @property
    def device(self):
        return next(self.parameters()).device
