import torch
import torch.nn as nn
from models.adaptive_transformer import AdaptiveCausalLmWrapper

def load_adaptive_model_gpt(model_name, baseline_model, config, device):
    """
    Load an adaptive transformer model initialized from a baseline GPT model.
    """
    print("\n==== DEBUG INFO ====")
    print(f"Model name: {model_name}")
    print(f"Config: {config.__class__.__name__}")
    print(f"Hidden size: {config.hidden_size if hasattr(config, 'hidden_size') else config.n_embd}")
    print(f"Number of heads: {config.num_attention_heads if hasattr(config, 'num_attention_heads') else config.n_head}")
    if hasattr(config, 'n_inner'):
        print(f"FFN inner dim: {config.n_inner}")
    elif hasattr(config, 'intermediate_size'):
        print(f"FFN inner dim (intermediate_size): {config.intermediate_size}")
    else:
        print("FFN inner dim not found in config")
    print("=====================\n")

    # Get the token embeddings and position embeddings from the baseline model
    token_embeddings = baseline_model.get_input_embeddings()
    
    # Get position embeddings correctly
    # For GPT2, the position embeddings are stored in transformer.wpe
    if hasattr(baseline_model, 'transformer') and hasattr(baseline_model.transformer, 'wpe'):
        position_embeddings = baseline_model.transformer.wpe
    else:
        # Fallback - create new position embeddings
        position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        print("Warning: Could not get position embeddings from baseline model, created new ones")
    
    model = AdaptiveCausalLmWrapper(config, token_embeddings, position_embeddings).to(device)
    model.eval()

    baseline_state = baseline_model.state_dict()
    adaptive_state = model.state_dict()

    loaded, skipped = [], []

    num_heads = config.n_head if hasattr(config, "n_head") else config.num_attention_heads
    hidden_size = config.n_embd if hasattr(config, "n_embd") else config.hidden_size
    head_dim = hidden_size // num_heads

    # Copy final layer norm
    try:
        adaptive_state["ln_f.weight"].copy_(baseline_state["transformer.ln_f.weight"])
        adaptive_state["ln_f.bias"].copy_(baseline_state["transformer.ln_f.bias"])
        loaded.extend(["ln_f.weight", "ln_f.bias"])
    except Exception:
        skipped.extend(["ln_f.weight", "ln_f.bias"])

    # Process each transformer block
    for layer_idx in range(config.n_layer):
        try:
            adaptive_state[f"blocks.{layer_idx}.ln1.weight"].copy_(
                baseline_state[f"transformer.h.{layer_idx}.ln_1.weight"])
            adaptive_state[f"blocks.{layer_idx}.ln1.bias"].copy_(
                baseline_state[f"transformer.h.{layer_idx}.ln_1.bias"])
            adaptive_state[f"blocks.{layer_idx}.ln2.weight"].copy_(
                baseline_state[f"transformer.h.{layer_idx}.ln_2.weight"])
            adaptive_state[f"blocks.{layer_idx}.ln2.bias"].copy_(
                baseline_state[f"transformer.h.{layer_idx}.ln_2.bias"])
            loaded.extend([
                f"blocks.{layer_idx}.ln1.weight", f"blocks.{layer_idx}.ln1.bias",
                f"blocks.{layer_idx}.ln2.weight", f"blocks.{layer_idx}.ln2.bias"
            ])
        except Exception:
            skipped.extend([
                f"blocks.{layer_idx}.ln1.weight", f"blocks.{layer_idx}.ln1.bias",
                f"blocks.{layer_idx}.ln2.weight", f"blocks.{layer_idx}.ln2.bias"
            ])

        # Feedforward layers
        try:
            in_w = f"blocks.{layer_idx}.ffn.dense_in.weight"
            in_b = f"blocks.{layer_idx}.ffn.dense_in.bias"
            out_w = f"blocks.{layer_idx}.ffn.dense_out.weight"
            out_b = f"blocks.{layer_idx}.ffn.dense_out.bias"

            base_in_w = f"transformer.h.{layer_idx}.mlp.c_fc.weight"
            base_in_b = f"transformer.h.{layer_idx}.mlp.c_fc.bias"
            base_out_w = f"transformer.h.{layer_idx}.mlp.c_proj.weight"
            base_out_b = f"transformer.h.{layer_idx}.mlp.c_proj.bias"

            if adaptive_state[in_w].shape == baseline_state[base_in_w].t().shape:
                adaptive_state[in_w].copy_(baseline_state[base_in_w].t())
            else:
                adaptive_state[in_w].copy_(baseline_state[base_in_w])
            adaptive_state[in_b].copy_(baseline_state[base_in_b])

            if adaptive_state[out_w].shape == baseline_state[base_out_w].t().shape:
                adaptive_state[out_w].copy_(baseline_state[base_out_w].t())
            else:
                adaptive_state[out_w].copy_(baseline_state[base_out_w])
            adaptive_state[out_b].copy_(baseline_state[base_out_b])

            loaded.extend([in_w, in_b, out_w, out_b])
        except Exception:
            skipped.extend([in_w, in_b, out_w, out_b])

        # Gate - initialize with slightly different values across heads
        # This breaks initial symmetry which helps during training
        try:
            gate_key = f"blocks.{layer_idx}.attn.gate"
            # Instead of all gates being 1.0, use a graduated approach
            # Core heads (middle ones) get higher values than peripheral heads
            num_heads = config.n_head if hasattr(config, "n_head") else config.num_attention_heads
            gate_values = torch.ones_like(adaptive_state[gate_key])
            
            # Apply slight variations to initial gates to break symmetry
            # Middle heads stay closer to 1.0, outer heads get slightly lower values
            for head_idx in range(num_heads):
                # Calculate distance from middle (0.0 to 1.0)
                distance = abs(head_idx - (num_heads-1)/2) / ((num_heads-1)/2)
                # Reduce gate value slightly based on distance (1.0 → 0.9)
                gate_values[head_idx] = 1.0 - 0.1 * distance
                
            adaptive_state[gate_key].copy_(gate_values)
            loaded.append(gate_key)
        except Exception:
            skipped.append(gate_key)

        # Attention QKV + O
        try:
            qkv_w = baseline_state[f"transformer.h.{layer_idx}.attn.c_attn.weight"]
            qkv_b = baseline_state[f"transformer.h.{layer_idx}.attn.c_attn.bias"]
            c_proj_w = baseline_state[f"transformer.h.{layer_idx}.attn.c_proj.weight"]
            c_proj_b = baseline_state[f"transformer.h.{layer_idx}.attn.c_proj.bias"]

            for head_idx in range(num_heads):
                hs, he = head_idx * head_dim, (head_idx + 1) * head_dim
                qw, qb = qkv_w[:, hs:he].t(), qkv_b[hs:he]
                kw, kb = qkv_w[:, hidden_size + hs:hidden_size + he].t(), qkv_b[hidden_size + hs:hidden_size + he]
                vw, vb = qkv_w[:, 2 * hidden_size + hs:2 * hidden_size + he].t(), qkv_b[2 * hidden_size + hs:2 * hidden_size + he]
                ow = c_proj_w[:, hs:he].t() if c_proj_w[:, hs:he].t().shape == adaptive_state[f"blocks.{layer_idx}.attn.W_o.{head_idx}.weight"].shape else c_proj_w[:, hs:he]
                ob = c_proj_b / num_heads

                for name, val in [
                    (f"blocks.{layer_idx}.attn.W_q.{head_idx}.weight", qw),
                    (f"blocks.{layer_idx}.attn.W_q.{head_idx}.bias", qb),
                    (f"blocks.{layer_idx}.attn.W_k.{head_idx}.weight", kw),
                    (f"blocks.{layer_idx}.attn.W_k.{head_idx}.bias", kb),
                    (f"blocks.{layer_idx}.attn.W_v.{head_idx}.weight", vw),
                    (f"blocks.{layer_idx}.attn.W_v.{head_idx}.bias", vb),
                    (f"blocks.{layer_idx}.attn.W_o.{head_idx}.weight", ow),
                    (f"blocks.{layer_idx}.attn.W_o.{head_idx}.bias", ob)
                ]:
                    adaptive_state[name].copy_(val)
                    loaded.append(name)
        except Exception:
            for head_idx in range(num_heads):
                skipped.extend([
                    f"blocks.{layer_idx}.attn.W_q.{head_idx}.weight",
                    f"blocks.{layer_idx}.attn.W_q.{head_idx}.bias",
                    f"blocks.{layer_idx}.attn.W_k.{head_idx}.weight",
                    f"blocks.{layer_idx}.attn.W_k.{head_idx}.bias",
                    f"blocks.{layer_idx}.attn.W_v.{head_idx}.weight",
                    f"blocks.{layer_idx}.attn.W_v.{head_idx}.bias",
                    f"blocks.{layer_idx}.attn.W_o.{head_idx}.weight",
                    f"blocks.{layer_idx}.attn.W_o.{head_idx}.bias"
                ])

    # Final patch for remaining weights
    adaptive_state["wte.weight"].copy_(baseline_state["transformer.wte.weight"])
    adaptive_state["wpe.weight"][:baseline_state["transformer.wpe.weight"].shape[0]].copy_(
        baseline_state["transformer.wpe.weight"]
    )
    adaptive_state["lm_head.weight"].copy_(baseline_state["transformer.wte.weight"])
    loaded.extend(["wte.weight", "wpe.weight", "lm_head.weight"])

    if "bias" in adaptive_state:
        with torch.no_grad():
            adaptive_state["bias"].zero_()
        loaded.append("bias")

    for i in range(config.n_layer):
        fuse_w = f"blocks.{i}.skip_fuse.weight"
        fuse_b = f"blocks.{i}.skip_fuse.bias"
        if fuse_w in adaptive_state:
            with torch.no_grad():
                # Initialize skip connection weights with very small values to start
                # Standard practice for residual/skip connections - use small standard deviation
                # Too large initialization causes skip connections to dominate
                adaptive_state[fuse_w].normal_(mean=0.0, std=0.01)
                adaptive_state[fuse_b].zero_()
                
                # Apply additional scaling to deep layers (closer to output) to prevent instability
                if i > config.n_layer // 2:
                    layer_scale = 1.0 - (i - config.n_layer // 2) / (config.n_layer // 2) * 0.5
                    adaptive_state[fuse_w] *= layer_scale
                    
            loaded.extend([fuse_w, fuse_b])

    print(f"✅ Adaptive model initialized from {model_name} weights ({len(loaded)}/{len(adaptive_state)} parameters loaded)")
    return model
