import torch
import torch.nn as nn
from models.adaptive_transformer import AdaptiveCausalLmWrapper

def load_adaptive_model_gpt(model_name, baseline_model, config, device, debug=False):
    """
    Improved loading of adaptive transformer model from a baseline GPT model.
    
    This loader properly handles the weight transfer to ensure maximum accuracy
    and coherence in the adaptive model's outputs.
    
    Args:
        model_name: Name of the model to load
        baseline_model: The pretrained model to adapt
        config: Configuration object for the model
        device: Device to load the model on
        debug: Whether to print debug information
    
    Returns:
        The adaptive transformer model with loaded weights
    """
    if debug:
        print(f"✅ Using fixed GPT2 loader")
    
    # Get the token embeddings and position embeddings from the baseline model
    token_embeddings = baseline_model.get_input_embeddings()
    
    # Get position embeddings correctly
    if hasattr(baseline_model, 'transformer') and hasattr(baseline_model.transformer, 'wpe'):
        position_embeddings = baseline_model.transformer.wpe
    else:
        # Fallback - create new position embeddings
        position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        if debug:
            print("Warning: Could not get position embeddings from baseline model, created new ones")
    
    model = AdaptiveCausalLmWrapper(config, token_embeddings, position_embeddings, debug=debug).to(device)
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
        print(f"\n[Processing layer {layer_idx}]")
        
        # Load layer norms
        try:
            adaptive_state[f"blocks.{layer_idx}.ln1.weight"].copy_(
                baseline_state[f"transformer.h.{layer_idx}.ln_1.weight"])
            adaptive_state[f"blocks.{layer_idx}.ln1.bias"].copy_(
                baseline_state[f"transformer.h.{layer_idx}.ln_1.bias"])
            adaptive_state[f"blocks.{layer_idx}.ln2.weight"].copy_(
                baseline_state[f"transformer.h.{layer_idx}.ln_2.weight"])
            adaptive_state[f"blocks.{layer_idx}.ln2.bias"].copy_(
                baseline_state[f"transformer.h.{layer_idx}.ln_2.bias"])
            print("  ✓ Loaded layer norms")
            loaded.extend([
                f"blocks.{layer_idx}.ln1.weight", f"blocks.{layer_idx}.ln1.bias",
                f"blocks.{layer_idx}.ln2.weight", f"blocks.{layer_idx}.ln2.bias"
            ])
        except Exception as e:
            print(f"  ✗ Failed to load layer norms: {e}")
            skipped.extend([
                f"blocks.{layer_idx}.ln1.weight", f"blocks.{layer_idx}.ln1.bias",
                f"blocks.{layer_idx}.ln2.weight", f"blocks.{layer_idx}.ln2.bias"
            ])

        # Load feedforward layers
        try:
            in_w = f"blocks.{layer_idx}.ffn.dense_in.weight"
            in_b = f"blocks.{layer_idx}.ffn.dense_in.bias"
            out_w = f"blocks.{layer_idx}.ffn.dense_out.weight"
            out_b = f"blocks.{layer_idx}.ffn.dense_out.bias"

            base_in_w = f"transformer.h.{layer_idx}.mlp.c_fc.weight"
            base_in_b = f"transformer.h.{layer_idx}.mlp.c_fc.bias"
            base_out_w = f"transformer.h.{layer_idx}.mlp.c_proj.weight"
            base_out_b = f"transformer.h.{layer_idx}.mlp.c_proj.bias"

            # Match dimensions properly
            if baseline_state[base_in_w].shape[0] == adaptive_state[in_w].shape[1]:
                # Need to transpose weights
                adaptive_state[in_w].copy_(baseline_state[base_in_w].t())
            else:
                adaptive_state[in_w].copy_(baseline_state[base_in_w])
            
            # Copy biases directly
            adaptive_state[in_b].copy_(baseline_state[base_in_b])

            # Also match dimensions for output projection
            if baseline_state[base_out_w].shape[0] == adaptive_state[out_w].shape[1]:
                adaptive_state[out_w].copy_(baseline_state[base_out_w].t())
            else:
                adaptive_state[out_w].copy_(baseline_state[base_out_w])
                
            adaptive_state[out_b].copy_(baseline_state[base_out_b])
            
            print("  ✓ Loaded feedforward layers")
            loaded.extend([in_w, in_b, out_w, out_b])
        except Exception as e:
            print(f"  ✗ Failed to load feedforward layers: {e}")
            skipped.extend([in_w, in_b, out_w, out_b])

        # Initialize gate values with slight asymmetry
        try:
            gate_key = f"blocks.{layer_idx}.attn.gate"
            # Create different initial gate values for each head
            gate_values = torch.ones_like(adaptive_state[gate_key])
            
            # Apply a slight asymmetry to break symmetry during training
            for head_idx in range(num_heads):
                # Calculate distance from middle (0 to 1 scale)
                distance = abs(head_idx - (num_heads-1)/2) / ((num_heads-1)/2)
                # Apply small variation (max 5%)
                gate_values[head_idx] = 1.0 - 0.05 * distance
                
            adaptive_state[gate_key].copy_(gate_values)
            print("  ✓ Initialized gate values with slight asymmetry")
            loaded.append(gate_key)
        except Exception as e:
            print(f"  ✗ Failed to initialize gate values: {e}")
            skipped.append(gate_key)

        # Load attention weights with careful handling of shapes
        try:
            # Get attention weights from baseline
            qkv_w = baseline_state[f"transformer.h.{layer_idx}.attn.c_attn.weight"] 
            qkv_b = baseline_state[f"transformer.h.{layer_idx}.attn.c_attn.bias"]
            proj_w = baseline_state[f"transformer.h.{layer_idx}.attn.c_proj.weight"]
            proj_b = baseline_state[f"transformer.h.{layer_idx}.attn.c_proj.bias"]
            
            # Debug
            print(f"  • QKV weight shape: {qkv_w.shape}, Proj weight shape: {proj_w.shape}")
            
            # Check shapes to determine layout format
            if qkv_w.shape[0] == hidden_size and qkv_w.shape[1] == 3 * hidden_size:
                # Standard GPT-2 shape: [hidden_size, 3*hidden_size]
                print("  • Using standard GPT-2 convention (transpose needed)")
                qkv_w = qkv_w.t()  # Transpose to [3*hidden_size, hidden_size]
                if proj_w.shape[0] == hidden_size:
                    proj_w = proj_w.t()
            elif qkv_w.shape[0] == 3 * hidden_size and qkv_w.shape[1] == hidden_size:
                # Already in the format we need [3*hidden_size, hidden_size]
                print("  • Using pre-transposed weights format")
            else:
                print(f"  • Unexpected weight shape: {qkv_w.shape}, attempting to adjust...")
                # Try to adjust to the expected shape
                if qkv_w.numel() == 3 * hidden_size * hidden_size:
                    qkv_w = qkv_w.reshape(3 * hidden_size, hidden_size)
            
            # Process each head separately
            for head_idx in range(num_heads):
                # Calculate slices for this head in the combined weights
                h_start = head_idx * head_dim
                h_end = (head_idx + 1) * head_dim
                
                # Extract Q, K, V slices for this head
                q_slice = slice(h_start, h_end)
                k_slice = slice(hidden_size + h_start, hidden_size + h_end)
                v_slice = slice(2 * hidden_size + h_start, 2 * hidden_size + h_end)
                
                # Get weights and biases
                if qkv_w.shape[1] == hidden_size:
                    # This is the case where the QKV weight matrix is [3*hidden_size, hidden_size]
                    print(f"  • Using altered slicing for shape {qkv_w.shape}")
                    q_start, q_end = 0, hidden_size // num_heads
                    k_start, k_end = hidden_size, hidden_size + hidden_size // num_heads
                    v_start, v_end = 2 * hidden_size, 2 * hidden_size + hidden_size // num_heads
                    
                    # Extract slices for this head
                    qw = qkv_w[h_start:h_end, :].clone()  # [head_dim, hidden_size]
                    qb = qkv_b[h_start:h_end].clone()     # [head_dim]
                    
                    kw = qkv_w[hidden_size + h_start:hidden_size + h_end, :].clone()
                    kb = qkv_b[hidden_size + h_start:hidden_size + h_end].clone()
                    
                    vw = qkv_w[2 * hidden_size + h_start:2 * hidden_size + h_end, :].clone()
                    vb = qkv_b[2 * hidden_size + h_start:2 * hidden_size + h_end].clone()
                else:
                    # Standard case where QKV weight matrix is [hidden_size, 3*hidden_size]
                    qw = qkv_w[:, q_slice].clone()
                    qb = qkv_b[q_slice].clone()
                    
                    kw = qkv_w[:, k_slice].clone()
                    kb = qkv_b[k_slice].clone()
                    
                    vw = qkv_w[:, v_slice].clone()
                    vb = qkv_b[v_slice].clone()
                
                # Get projection weights for this head
                # For projection, we need to distribute the projection evenly
                # Original is [hidden_size, hidden_size], we want [head_dim, hidden_size]
                ow_full = proj_w.clone()
                ow = ow_full[h_start:h_end, :].clone()
                
                # Output bias is shared across heads, so divide by num_heads
                ob = proj_b.clone() / num_heads
                
                # Check if we need to transpose individual weights
                if qw.shape[0] != head_dim and qw.shape[1] == head_dim:
                    qw = qw.t()
                    kw = kw.t()
                    vw = vw.t()
                
                if ow.shape[0] != hidden_size and ow.shape[1] == hidden_size:
                    ow = ow.t()
                
                # Copy to our model's separated head weights
                adaptive_state[f"blocks.{layer_idx}.attn.W_q.{head_idx}.weight"].copy_(qw)
                adaptive_state[f"blocks.{layer_idx}.attn.W_q.{head_idx}.bias"].copy_(qb)
                
                adaptive_state[f"blocks.{layer_idx}.attn.W_k.{head_idx}.weight"].copy_(kw)
                adaptive_state[f"blocks.{layer_idx}.attn.W_k.{head_idx}.bias"].copy_(kb)
                
                adaptive_state[f"blocks.{layer_idx}.attn.W_v.{head_idx}.weight"].copy_(vw)
                adaptive_state[f"blocks.{layer_idx}.attn.W_v.{head_idx}.bias"].copy_(vb)
                
                adaptive_state[f"blocks.{layer_idx}.attn.W_o.{head_idx}.weight"].copy_(ow)
                adaptive_state[f"blocks.{layer_idx}.attn.W_o.{head_idx}.bias"].copy_(ob)
                
                # Add to loaded list
                loaded.extend([
                    f"blocks.{layer_idx}.attn.W_q.{head_idx}.weight",
                    f"blocks.{layer_idx}.attn.W_q.{head_idx}.bias",
                    f"blocks.{layer_idx}.attn.W_k.{head_idx}.weight",
                    f"blocks.{layer_idx}.attn.W_k.{head_idx}.bias",
                    f"blocks.{layer_idx}.attn.W_v.{head_idx}.weight",
                    f"blocks.{layer_idx}.attn.W_v.{head_idx}.bias",
                    f"blocks.{layer_idx}.attn.W_o.{head_idx}.weight",
                    f"blocks.{layer_idx}.attn.W_o.{head_idx}.bias"
                ])
                
            print(f"  ✓ Loaded attention weights for layer {layer_idx}")
                
        except Exception as e:
            print(f"  ✗ Failed to load attention weights: {e}")
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
        
        # Initialize skip connection fusion with small values
        try:
            fuse_w = f"blocks.{layer_idx}.skip_fuse.weight"
            fuse_b = f"blocks.{layer_idx}.skip_fuse.bias"
            
            with torch.no_grad():
                # Initialize with small values
                adaptive_state[fuse_w].normal_(mean=0.0, std=0.01)
                adaptive_state[fuse_b].zero_()
                
                # Apply scaling for deeper layers
                scale = 1.0
                if layer_idx > config.n_layer // 2:
                    # Gradually reduce scale for deeper layers
                    progress = (layer_idx - config.n_layer // 2) / (config.n_layer - config.n_layer // 2)
                    scale = 1.0 - progress * 0.5  # Scale from 1.0 down to 0.5
                    
                adaptive_state[fuse_w] *= scale
                
            print(f"  ✓ Initialized skip connection with small values (scale={scale:.2f})")
            loaded.extend([fuse_w, fuse_b])
        except Exception as e:
            print(f"  ✗ Failed to initialize skip connection: {e}")
            skipped.extend([fuse_w, fuse_b])

    # Copy embeddings and output weights
    try:
        adaptive_state["wte.weight"].copy_(baseline_state["transformer.wte.weight"])
        adaptive_state["wpe.weight"][:baseline_state["transformer.wpe.weight"].shape[0]].copy_(
            baseline_state["transformer.wpe.weight"]
        )
        adaptive_state["lm_head.weight"].copy_(baseline_state["transformer.wte.weight"])
        loaded.extend(["wte.weight", "wpe.weight", "lm_head.weight"])
        print(f"✓ Loaded embeddings and output weights")
    except Exception as e:
        print(f"✗ Failed to load embeddings and output weights: {e}")
        skipped.extend(["wte.weight", "wpe.weight", "lm_head.weight"])

    if "bias" in adaptive_state:
        with torch.no_grad():
            adaptive_state["bias"].zero_()
        loaded.append("bias")

    print(f"\n✅ Adaptive model initialized from {model_name} weights ({len(loaded)}/{len(loaded) + len(skipped)} parameters loaded)")
    if skipped:
        print(f"   Skipped {len(skipped)} parameters")
        
    return model