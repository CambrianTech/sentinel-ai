"""
Clean GPT-2 Adaptive Transformer Loader

Philosophy:
1. Load pre-trained GPT-2 (baseline)
2. Create AdaptiveTransformer structure (empty)
3. Transfer weights from baseline to adaptive (split QKV matrices)
4. Wrap with AdaptiveCausalLmWrapper (for generation)

This preserves ALL pre-trained knowledge while enabling per-head pruning.
"""

import torch
import torch.nn as nn
from transformers import AutoConfig

from sentinel.models.adaptive_transformer import AdaptiveTransformer, AdaptiveCausalLmWrapper


def transfer_attention_weights(baseline_layer, adaptive_layer, config):
    """
    Transfer attention weights from fused QKV to per-head Q, K, V.

    GPT-2 format: c_attn.weight is [hidden_size, 3*hidden_size]
                  Organized as [Q_all | K_all | V_all]

    Adaptive format: Per head Q, K, V projections
    """
    num_heads = config.n_head
    hidden_size = config.n_embd
    head_dim = hidden_size // num_heads

    # Get baseline QKV weights (might need transpose)
    qkv_weight = baseline_layer.attn.c_attn.weight.data  # [3*hidden, hidden] or [hidden, 3*hidden]
    qkv_bias = baseline_layer.attn.c_attn.bias.data      # [3*hidden]

    # Handle transpose if needed
    if qkv_weight.shape[0] == hidden_size:
        # Shape is [hidden, 3*hidden] - need to transpose
        qkv_weight = qkv_weight.t()  # Now [3*hidden, hidden]

    # Get output projection (c_proj maps concatenated heads back to hidden)
    out_weight = baseline_layer.attn.c_proj.weight.data
    out_bias = baseline_layer.attn.c_proj.bias.data

    # Transpose if needed to get [hidden, hidden] = [OUT, IN] format
    if out_weight.shape[0] != hidden_size:
        out_weight = out_weight.t()  # Now [hidden, hidden]

    # Split QKV into Q, K, V chunks (qkv_weight is [3*hidden, hidden] = [OUT, IN] after transpose)
    # We need [IN, OUT] for matmul, so transpose each chunk
    q_all = qkv_weight[:hidden_size, :].t()        # [hidden, hidden]
    k_all = qkv_weight[hidden_size:2*hidden_size, :].t()
    v_all = qkv_weight[2*hidden_size:, :].t()

    q_bias_all = qkv_bias[:hidden_size]
    k_bias_all = qkv_bias[hidden_size:2*hidden_size]
    v_bias_all = qkv_bias[2*hidden_size:]

    # Transfer per-head weights
    for h in range(num_heads):
        h_start = h * head_dim
        h_end = (h + 1) * head_dim

        # Extract this head's weights (slice along output dimension since q_all is [IN, OUT])
        q_h = q_all[:, h_start:h_end]  # [hidden, head_dim]
        k_h = k_all[:, h_start:h_end]
        v_h = v_all[:, h_start:h_end]

        q_bias_h = q_bias_all[h_start:h_end]
        k_bias_h = k_bias_all[h_start:h_end]
        v_bias_h = v_bias_all[h_start:h_end]

        # Output projection per head
        # CRITICAL: Conv1D stores weights as [IN, OUT], NOT [OUT, IN]!
        # out_weight is [IN, OUT] = [768, 768] (no transpose happens since shape[0]=hidden_size)
        # Concatenated heads form the input: [h0 | h1 | ... | h11]
        # To get head i's contribution: slice input dimensions [i*64:(i+1)*64]
        # Result: [head_dim, hidden] = [64, 768] = [IN_i, OUT]
        # This is exactly what we need for: head_output = context @ W_o[h]
        o_h = out_weight[h_start:h_end, :]  # [head_dim, hidden] - NO transpose!
        o_bias_h = out_bias / num_heads  # Shared bias, divide equally

        # Copy to adaptive model (W_q, W_k, W_v, W_o are Parameters, not Linear layers)
        adaptive_layer.attn.W_q[h].data.copy_(q_h)
        adaptive_layer.attn.b_q[h].data.copy_(q_bias_h)

        adaptive_layer.attn.W_k[h].data.copy_(k_h)
        adaptive_layer.attn.b_k[h].data.copy_(k_bias_h)

        adaptive_layer.attn.W_v[h].data.copy_(v_h)
        adaptive_layer.attn.b_v[h].data.copy_(v_bias_h)

        adaptive_layer.attn.W_o[h].data.copy_(o_h)
        adaptive_layer.attn.b_o[h].data.copy_(o_bias_h)

    # Initialize gate to 1.0 (all heads active)
    adaptive_layer.attn.gate.data.fill_(1.0)


def transfer_ffn_weights(baseline_layer, adaptive_layer):
    """Transfer feedforward network weights."""
    # MLP: c_fc (expansion) and c_proj (projection)
    # Adaptive FFN is a Sequential: [Linear, GELU, Dropout, Linear, Dropout]

    fc_weight = baseline_layer.mlp.c_fc.weight.data
    fc_bias = baseline_layer.mlp.c_fc.bias.data

    proj_weight = baseline_layer.mlp.c_proj.weight.data
    proj_bias = baseline_layer.mlp.c_proj.bias.data

    # Handle transpose if needed
    if fc_weight.shape[0] != adaptive_layer.ffn[0].weight.shape[0]:
        fc_weight = fc_weight.t()
    if proj_weight.shape[0] != adaptive_layer.ffn[3].weight.shape[0]:
        proj_weight = proj_weight.t()

    # Copy
    adaptive_layer.ffn[0].weight.data.copy_(fc_weight)
    adaptive_layer.ffn[0].bias.data.copy_(fc_bias)

    adaptive_layer.ffn[3].weight.data.copy_(proj_weight)
    adaptive_layer.ffn[3].bias.data.copy_(proj_bias)


def transfer_layer_norm_weights(baseline_layer, adaptive_layer):
    """Transfer layer normalization weights."""
    # ln_1 (pre-attention) → norm1
    adaptive_layer.norm1.weight.data.copy_(baseline_layer.ln_1.weight.data)
    adaptive_layer.norm1.bias.data.copy_(baseline_layer.ln_1.bias.data)

    # ln_2 (pre-FFN) → norm2
    adaptive_layer.norm2.weight.data.copy_(baseline_layer.ln_2.weight.data)
    adaptive_layer.norm2.bias.data.copy_(baseline_layer.ln_2.bias.data)


def transfer_all_weights(baseline_model, adaptive_transformer, config):
    """
    Transfer all weights from baseline to adaptive transformer.

    This is the critical function that preserves pre-trained knowledge.
    """
    print("Transferring weights from baseline to adaptive...")

    baseline_layers = baseline_model.transformer.h
    adaptive_layers = adaptive_transformer.blocks

    for i, (base_layer, adapt_layer) in enumerate(zip(baseline_layers, adaptive_layers)):
        print(f"  Layer {i}...")

        # Transfer attention weights (the complex part)
        transfer_attention_weights(base_layer, adapt_layer, config)

        # Transfer FFN weights (straightforward)
        transfer_ffn_weights(base_layer, adapt_layer)

        # Transfer layer norms (straightforward)
        transfer_layer_norm_weights(base_layer, adapt_layer)

    # Transfer final layer norm
    adaptive_transformer.norm.weight.data.copy_(baseline_model.transformer.ln_f.weight.data)
    adaptive_transformer.norm.bias.data.copy_(baseline_model.transformer.ln_f.bias.data)

    print("✅ Weight transfer complete")


def load_adaptive_model_gpt_clean(model_name, baseline_model, config, device, quiet=False):
    """
    Load an adaptive GPT-2 model with proper weight transfer.

    Args:
        model_name: Name of the model (e.g., "gpt2", "distilgpt2")
        baseline_model: Pre-trained GPT-2 model from HuggingFace
        config: Model configuration
        device: Device to load on
        quiet: Suppress progress messages

    Returns:
        AdaptiveCausalLmWrapper ready for training/inference
    """
    if not quiet:
        print("\n" + "="*80)
        print("LOADING ADAPTIVE GPT-2 MODEL (CLEAN)")
        print("="*80)
        print(f"Model: {model_name}")
        print(f"Layers: {config.n_layer}")
        print(f"Heads: {config.n_head}")
        print(f"Hidden size: {config.n_embd}")
        print(f"Head dim: {config.n_embd // config.n_head}")

    # Step 1: Extract embeddings (we'll reference them, not copy)
    token_embeddings = baseline_model.get_input_embeddings()
    position_embeddings = baseline_model.transformer.wpe

    if not quiet:
        print(f"\n✅ Extracted embeddings")
        print(f"   Token vocab: {token_embeddings.num_embeddings}")
        print(f"   Max position: {position_embeddings.num_embeddings}")

    # Step 2: Create empty adaptive transformer
    if not quiet:
        print(f"\n📦 Creating adaptive transformer structure...")

    adaptive_transformer = AdaptiveTransformer(config=config, debug=(not quiet))

    # Step 3: Transfer weights from baseline to adaptive
    if not quiet:
        print(f"\n🔄 Transferring pre-trained weights...")

    transfer_all_weights(baseline_model, adaptive_transformer, config)

    # Step 4: Wrap with LM head
    if not quiet:
        print(f"\n📦 Wrapping with language model head...")

    wrapper = AdaptiveCausalLmWrapper(
        base_model=baseline_model,
        transformer=adaptive_transformer,
        config=config
    )

    # Move to device
    wrapper = wrapper.to(device)

    if not quiet:
        print(f"\n✅ Adaptive model loaded successfully!")
        print(f"   Total parameters: {sum(p.numel() for p in wrapper.parameters()):,}")
        print(f"   Device: {device}")
        print("="*80 + "\n")

    return wrapper
