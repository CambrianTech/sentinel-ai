import torch
from models.adaptive_transformer import AdaptiveCausalLmWrapper


def load_adaptive_model_gpt(model_name, baseline_model, config, device):
    """
    Load an adaptive transformer model initialized from a baseline GPT model.
    """
    # Print debug info
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

    model = AdaptiveCausalLmWrapper(config, baseline_model.get_input_embeddings(), baseline_model.get_input_embeddings()).to(device)

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
    except Exception as e:
        skipped.extend(["ln_f.weight", "ln_f.bias"])
        print(f"Error copying final layer norm: {e}")

    # Process each layer
    for layer_idx in range(config.n_layer):
        # Copy layer norms
        try:
            adaptive_state[f"blocks.{layer_idx}.ln1.weight"].copy_(
                baseline_state[f"transformer.h.{layer_idx}.ln_1.weight"])
            adaptive_state[f"blocks.{layer_idx}.ln1.bias"].copy_(
                baseline_state[f"transformer.h.{layer_idx}.ln_1.bias"])
            loaded.extend([f"blocks.{layer_idx}.ln1.weight", f"blocks.{layer_idx}.ln1.bias"])
            
            adaptive_state[f"blocks.{layer_idx}.ln2.weight"].copy_(
                baseline_state[f"transformer.h.{layer_idx}.ln_2.weight"])
            adaptive_state[f"blocks.{layer_idx}.ln2.bias"].copy_(
                baseline_state[f"transformer.h.{layer_idx}.ln_2.bias"])
            loaded.extend([f"blocks.{layer_idx}.ln2.weight", f"blocks.{layer_idx}.ln2.bias"])
        except Exception as e:
            skipped.extend([f"blocks.{layer_idx}.ln1.weight", f"blocks.{layer_idx}.ln1.bias",
                           f"blocks.{layer_idx}.ln2.weight", f"blocks.{layer_idx}.ln2.bias"])
            print(f"Error copying layer norms for layer {layer_idx}: {e}")
            
        # Copy feed-forward weights - CORRECTED for transpose
        try:
            ffn_in_key = f"blocks.{layer_idx}.ffn.dense_in.weight"
            ffn_in_bias_key = f"blocks.{layer_idx}.ffn.dense_in.bias"
            ffn_out_key = f"blocks.{layer_idx}.ffn.dense_out.weight"
            ffn_out_bias_key = f"blocks.{layer_idx}.ffn.dense_out.bias"
            
            base_in_key = f"transformer.h.{layer_idx}.mlp.c_fc.weight"
            base_in_bias_key = f"transformer.h.{layer_idx}.mlp.c_fc.bias"
            base_out_key = f"transformer.h.{layer_idx}.mlp.c_proj.weight"
            base_out_bias_key = f"transformer.h.{layer_idx}.mlp.c_proj.bias"
            
            # Copy FFN weights with potential transpose for shape matching
            if layer_idx == 0:  # Only print for first layer
                print(f"FFN dimensions before transpose:")
                print(f"  Adaptive in: {adaptive_state[ffn_in_key].shape}")
                print(f"  Baseline in: {baseline_state[base_in_key].shape}")
                print(f"  Adaptive out: {adaptive_state[ffn_out_key].shape}")
                print(f"  Baseline out: {baseline_state[base_out_key].shape}")
                
            # If shapes don't match directly, try transposing
            if adaptive_state[ffn_in_key].shape != baseline_state[base_in_key].shape:
                # Check if transpose would match
                if adaptive_state[ffn_in_key].shape == baseline_state[base_in_key].t().shape:
                    adaptive_state[ffn_in_key].copy_(baseline_state[base_in_key].t())
                    if layer_idx == 0:
                        print("  Transposed FFN input weights")
                    loaded.append(ffn_in_key)
                else:
                    skipped.append(ffn_in_key)
            else:
                adaptive_state[ffn_in_key].copy_(baseline_state[base_in_key])
                loaded.append(ffn_in_key)
                
            # Copy bias (no transpose needed)
            adaptive_state[ffn_in_bias_key].copy_(baseline_state[base_in_bias_key])
            loaded.append(ffn_in_bias_key)
            
            # Same for output weights
            if adaptive_state[ffn_out_key].shape != baseline_state[base_out_key].shape:
                # Check if transpose would match
                if adaptive_state[ffn_out_key].shape == baseline_state[base_out_key].t().shape:
                    adaptive_state[ffn_out_key].copy_(baseline_state[base_out_key].t())
                    if layer_idx == 0:
                        print("  Transposed FFN output weights")
                    loaded.append(ffn_out_key)
                else:
                    skipped.append(ffn_out_key)
            else:
                adaptive_state[ffn_out_key].copy_(baseline_state[base_out_key])
                loaded.append(ffn_out_key)
                
            # Copy output bias
            adaptive_state[ffn_out_bias_key].copy_(baseline_state[base_out_bias_key])
            loaded.append(ffn_out_bias_key)
                
        except Exception as e:
            skipped.extend([ffn_in_key, ffn_in_bias_key, ffn_out_key, ffn_out_bias_key])
            print(f"Error copying FFN weights for layer {layer_idx}: {e}")
            
        # Set gate values to 1.0
        try:
            gate_key = f"blocks.{layer_idx}.attn.gate"
            adaptive_state[gate_key].fill_(1.0)
            loaded.append(gate_key)
        except Exception as e:
            skipped.append(gate_key)
            print(f"Error setting gate values for layer {layer_idx}: {e}")
            
        # CORRECTED: Copy attention weights using proper extraction
        try:
            # Get the combined QKV matrix and bias from baseline model
            qkv_weight = baseline_state[f"transformer.h.{layer_idx}.attn.c_attn.weight"] 
            qkv_bias = baseline_state[f"transformer.h.{layer_idx}.attn.c_attn.bias"]
            
            # Important: GPT2's c_attn is a single projection that outputs concatenated Q, K, V
            # The weight matrix shape is (hidden_size, 3*hidden_size)
            
            if layer_idx == 0:  # Debug only first layer
                print(f"QKV weight shape: {qkv_weight.shape}")
                print(f"QKV bias shape: {qkv_bias.shape}")
                
            # Process for each attention head
            for head_idx in range(num_heads):
                # Indices for this head's portion in the hidden dimension
                head_start = head_idx * head_dim
                head_end = (head_idx + 1) * head_dim
                
                # Split QKV weight and bias for this head
                # IMPORTANT: GPT2's weight is (hidden_size, 3*hidden_size)
                # We need to split it into (hidden_size, head_dim) chunks
                
                # Extract Q weights/bias for this head
                q_key = f"blocks.{layer_idx}.attn.W_q.{head_idx}.weight"
                q_bias_key = f"blocks.{layer_idx}.attn.W_q.{head_idx}.bias"
                
                # Get Q portion: first third of c_attn weights
                q_weight = qkv_weight[:, head_start:head_end].t()  # Transpose to match shape
                q_bias = qkv_bias[head_start:head_end]
                
                # Print dimensions for debugging (first layer, first head only)
                if layer_idx == 0 and head_idx == 0:
                    print(f"Q extraction: got {q_weight.shape}, need {adaptive_state[q_key].shape}")
                
                # Copy if shapes match
                if q_weight.shape == adaptive_state[q_key].shape:
                    adaptive_state[q_key].copy_(q_weight)
                    adaptive_state[q_bias_key].copy_(q_bias)
                    loaded.extend([q_key, q_bias_key])
                else:
                    skipped.extend([q_key, q_bias_key])
                
                # Extract K weights/bias for this head
                k_key = f"blocks.{layer_idx}.attn.W_k.{head_idx}.weight"
                k_bias_key = f"blocks.{layer_idx}.attn.W_k.{head_idx}.bias"
                
                # Get K portion: second third of c_attn weights
                k_start = hidden_size + head_start
                k_end = hidden_size + head_end
                k_weight = qkv_weight[:, k_start:k_end].t()
                k_bias = qkv_bias[k_start:k_end]
                
                # Print dimensions for debugging (first layer, first head only)
                if layer_idx == 0 and head_idx == 0:
                    print(f"K extraction: got {k_weight.shape}, need {adaptive_state[k_key].shape}")
                
                # Copy if shapes match
                if k_weight.shape == adaptive_state[k_key].shape:
                    adaptive_state[k_key].copy_(k_weight)
                    adaptive_state[k_bias_key].copy_(k_bias)
                    loaded.extend([k_key, k_bias_key])
                else:
                    skipped.extend([k_key, k_bias_key])
                
                # Extract V weights/bias for this head
                v_key = f"blocks.{layer_idx}.attn.W_v.{head_idx}.weight"
                v_bias_key = f"blocks.{layer_idx}.attn.W_v.{head_idx}.bias"
                
                # Get V portion: third third of c_attn weights
                v_start = 2 * hidden_size + head_start
                v_end = 2 * hidden_size + head_end
                v_weight = qkv_weight[:, v_start:v_end].t()
                v_bias = qkv_bias[v_start:v_end]
                
                # Print dimensions for debugging (first layer, first head only)
                if layer_idx == 0 and head_idx == 0:
                    print(f"V extraction: got {v_weight.shape}, need {adaptive_state[v_key].shape}")
                
                # Copy if shapes match
                if v_weight.shape == adaptive_state[v_key].shape:
                    adaptive_state[v_key].copy_(v_weight)
                    adaptive_state[v_bias_key].copy_(v_bias)
                    loaded.extend([v_key, v_bias_key])
                else:
                    skipped.extend([v_key, v_bias_key])
                
                # Handle output projection
                o_key = f"blocks.{layer_idx}.attn.W_o.{head_idx}.weight"
                o_bias_key = f"blocks.{layer_idx}.attn.W_o.{head_idx}.bias"
                
                # Get c_proj (output projection) from baseline model
                c_proj_weight = baseline_state[f"transformer.h.{layer_idx}.attn.c_proj.weight"]
                c_proj_bias = baseline_state[f"transformer.h.{layer_idx}.attn.c_proj.bias"]
                
                # Select the portion for this head - for GPT2, c_proj is (hidden_size, hidden_size)
                # Each head gets a slice of the columns
                o_weight_slice = c_proj_weight[:, head_start:head_end]
                
                # Transpose if necessary to match expected shape
                if o_weight_slice.shape != adaptive_state[o_key].shape:
                    o_weight_slice = o_weight_slice.t()
                
                # Split bias evenly across heads
                o_bias_slice = c_proj_bias / num_heads
                
                # Print dimensions for debugging (first layer, first head only)
                if layer_idx == 0 and head_idx == 0:
                    print(f"O extraction: got {o_weight_slice.shape}, need {adaptive_state[o_key].shape}")
                
                # Copy if shapes match
                if o_weight_slice.shape == adaptive_state[o_key].shape:
                    adaptive_state[o_key].copy_(o_weight_slice)
                    adaptive_state[o_bias_key].copy_(o_bias_slice)
                    loaded.extend([o_key, o_bias_key])
                else:
                    skipped.extend([o_key, o_bias_key])
                
        except Exception as e:
            # Just skip all attention heads for this layer
            for h in range(num_heads):
                skipped.extend([
                    f"blocks.{layer_idx}.attn.W_q.{h}.weight",
                    f"blocks.{layer_idx}.attn.W_q.{h}.bias",
                    f"blocks.{layer_idx}.attn.W_k.{h}.weight",
                    f"blocks.{layer_idx}.attn.W_k.{h}.bias",
                    f"blocks.{layer_idx}.attn.W_v.{h}.weight",
                    f"blocks.{layer_idx}.attn.W_v.{h}.bias",
                    f"blocks.{layer_idx}.attn.W_o.{h}.weight",
                    f"blocks.{layer_idx}.attn.W_o.{h}.bias"
                ])
            print(f"Error copying attention weights for layer {layer_idx}: {e}")

    print(f"✅ Adaptive model initialized from {model_name} weights ({len(loaded)}/{len(adaptive_state)} parameters loaded)")
    if skipped:
        print(f"⚠️ Skipped {len(skipped)} params (shape mismatch or custom):")
        for p in skipped[:10]:
            print("   ", p)
        if len(skipped) > 10:
            print(f"   ... and {len(skipped) - 10} more.")

    return model