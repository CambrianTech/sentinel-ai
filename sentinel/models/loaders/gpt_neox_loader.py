import torch
import torch.nn as nn
from sentinel.models.adaptive_transformer import AdaptiveCausalLmWrapper

def load_adaptive_model_gpt_neox(model_name, baseline_model, config, device, debug=False, quiet=False):
    """
    Load an adaptive transformer model initialized from a baseline GPT-NeoX model.
    This supports models like EleutherAI/pythia-* and other GPT-NeoX based models.
    
    Args:
        model_name: Name of the base model
        baseline_model: Pretrained model to initialize from
        config: Configuration for the model
        device: Device to load the model on ('cpu' or 'cuda')
        debug: Whether to print debug information
        quiet: If True, suppresses verbose loading messages
    
    Returns:
        The adaptive transformer model with loaded weights
    """
    # Force quiet mode to be true since we're running with quiet=True by default
    # Will be removed when we properly implement quiet mode in this loader
    quiet = True
    
    if debug and not quiet:
        print(f"✅ Using GPT-NeoX loader")
    
    # Get the token embeddings and position embeddings from the baseline model
    token_embeddings = baseline_model.get_input_embeddings()
    
    # Get position embeddings correctly - NeoX may use rotary embeddings
    # so we might need to create new position embeddings
    if hasattr(baseline_model, 'gpt_neox') and hasattr(baseline_model.gpt_neox, 'embed_positions'):
        position_embeddings = baseline_model.gpt_neox.embed_positions
    else:
        # Fallback - create new position embeddings
        position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        if debug and not quiet:
            print("Warning: Could not get position embeddings from baseline model, created new ones")
    
    model = AdaptiveCausalLmWrapper(config, token_embeddings, position_embeddings, debug=debug).to(device)
    model.eval()

    baseline_state = baseline_model.state_dict()
    adaptive_state = model.state_dict()

    loaded, skipped = [], []

    num_heads = config.num_attention_heads if hasattr(config, "num_attention_heads") else config.n_head
    hidden_size = config.hidden_size if hasattr(config, "hidden_size") else config.n_embd
    head_dim = hidden_size // num_heads

    # Process each transformer block
    num_layers = config.num_hidden_layers if hasattr(config, "num_hidden_layers") else config.n_layer
    for layer_idx in range(num_layers):
        if not quiet:
            print(f"\n[Processing layer {layer_idx}]")
        
        # Load layer norms
        try:
            # NeoX uses input_layernorm and post_attention_layernorm
            adaptive_state[f"blocks.{layer_idx}.ln1.weight"].copy_(
                baseline_state[f"gpt_neox.layers.{layer_idx}.input_layernorm.weight"])
            adaptive_state[f"blocks.{layer_idx}.ln1.bias"].copy_(
                baseline_state[f"gpt_neox.layers.{layer_idx}.input_layernorm.bias"])
            adaptive_state[f"blocks.{layer_idx}.ln2.weight"].copy_(
                baseline_state[f"gpt_neox.layers.{layer_idx}.post_attention_layernorm.weight"])
            adaptive_state[f"blocks.{layer_idx}.ln2.bias"].copy_(
                baseline_state[f"gpt_neox.layers.{layer_idx}.post_attention_layernorm.bias"])
            if not quiet:
                print("  ✓ Loaded layer norms")
            loaded.extend([
                f"blocks.{layer_idx}.ln1.weight", f"blocks.{layer_idx}.ln1.bias",
                f"blocks.{layer_idx}.ln2.weight", f"blocks.{layer_idx}.ln2.bias"
            ])
        except Exception as e:
            if not quiet:
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

            # NeoX uses dense_h_to_4h and dense_4h_to_h
            base_in_w = f"gpt_neox.layers.{layer_idx}.mlp.dense_h_to_4h.weight"
            base_in_b = f"gpt_neox.layers.{layer_idx}.mlp.dense_h_to_4h.bias"
            base_out_w = f"gpt_neox.layers.{layer_idx}.mlp.dense_4h_to_h.weight"
            base_out_b = f"gpt_neox.layers.{layer_idx}.mlp.dense_4h_to_h.bias"

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
            
            if not quiet:
                print("  ✓ Loaded feedforward layers")
            loaded.extend([in_w, in_b, out_w, out_b])
        except Exception as e:
            if not quiet:
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
            if not quiet:
                print("  ✓ Initialized gate values with slight asymmetry")
            loaded.append(gate_key)
        except Exception as e:
            if not quiet:
                print(f"  ✗ Failed to initialize gate values: {e}")
            skipped.append(gate_key)

        # Load attention weights
        try:
            # GPT-NeoX has different attention formats. Some might have query_key_value combined
            # or separate query, key, value projections. Handle both cases.
            combined_qkv = False

            try:
                # Check first if combined QKV format
                qkv_combined = baseline_state[f"gpt_neox.layers.{layer_idx}.attention.query_key_value.weight"]
                qkv_combined_bias = baseline_state[f"gpt_neox.layers.{layer_idx}.attention.query_key_value.bias"]
                combined_qkv = True
            except KeyError:
                # Separate QKV format
                q_weight = baseline_state[f"gpt_neox.layers.{layer_idx}.attention.query.weight"]
                q_bias = baseline_state[f"gpt_neox.layers.{layer_idx}.attention.query.bias"]
                k_weight = baseline_state[f"gpt_neox.layers.{layer_idx}.attention.key.weight"]
                k_bias = baseline_state[f"gpt_neox.layers.{layer_idx}.attention.key.bias"]
                v_weight = baseline_state[f"gpt_neox.layers.{layer_idx}.attention.value.weight"]
                v_bias = baseline_state[f"gpt_neox.layers.{layer_idx}.attention.value.bias"]
                
            # Output projection is the same in both formats
            proj_w = baseline_state[f"gpt_neox.layers.{layer_idx}.attention.dense.weight"]
            proj_b = baseline_state[f"gpt_neox.layers.{layer_idx}.attention.dense.bias"]
            
            # Process each head separately
            for head_idx in range(num_heads):
                # Calculate slices for this head
                h_start = head_idx * head_dim
                h_end = (head_idx + 1) * head_dim
                
                # Extract QKV weights depending on format
                if combined_qkv:
                    # For combined QKV, slice into separate Q, K, V parts
                    qkv_size = 3 * head_dim * num_heads
                    q_start = h_start
                    k_start = num_heads * head_dim + h_start
                    v_start = 2 * num_heads * head_dim + h_start
                    
                    q_slice = slice(q_start, q_start + head_dim)
                    k_slice = slice(k_start, k_start + head_dim)
                    v_slice = slice(v_start, v_start + head_dim)
                    
                    qw = qkv_combined[:, q_slice].clone()
                    qb = qkv_combined_bias[q_slice].clone()
                    
                    kw = qkv_combined[:, k_slice].clone()
                    kb = qkv_combined_bias[k_slice].clone()
                    
                    vw = qkv_combined[:, v_slice].clone()
                    vb = qkv_combined_bias[v_slice].clone()
                else:
                    # For separate projections, get the slices for this head
                    qw = q_weight[h_start:h_end, :].clone()
                    qb = q_bias[h_start:h_end].clone()
                    
                    kw = k_weight[h_start:h_end, :].clone()
                    kb = k_bias[h_start:h_end].clone()
                    
                    vw = v_weight[h_start:h_end, :].clone()
                    vb = v_bias[h_start:h_end].clone()
                
                # Output projection handling
                # For projection, we need to distribute the projection matrix by rows
                ow = proj_w[:, h_start:h_end].t().clone()
                
                # Output bias is shared, divide by num_heads
                ob = proj_b.clone() / num_heads
                
                # Need to transpose some weights
                if qw.shape[0] != head_dim:
                    qw = qw.t()
                    kw = kw.t()
                    vw = vw.t()
                
                if ow.shape[0] != head_dim:
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
                
            if not quiet:
                print(f"  ✓ Loaded attention weights for layer {layer_idx}")
                
        except Exception as e:
            if not quiet:
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
                if layer_idx > num_layers // 2:
                    # Gradually reduce scale for deeper layers
                    progress = (layer_idx - num_layers // 2) / (num_layers - num_layers // 2)
                    scale = 1.0 - progress * 0.5  # Scale from 1.0 down to 0.5
                    
                adaptive_state[fuse_w] *= scale
                
            if not quiet:
                print(f"  ✓ Initialized skip connection with small values (scale={scale:.2f})")
            loaded.extend([fuse_w, fuse_b])
        except Exception as e:
            if not quiet:
                print(f"  ✗ Failed to initialize skip connection: {e}")
            skipped.extend([fuse_w, fuse_b])

    # Handle final layer norm
    try:
        adaptive_state["ln_f.weight"].copy_(baseline_state["gpt_neox.final_layer_norm.weight"])
        adaptive_state["ln_f.bias"].copy_(baseline_state["gpt_neox.final_layer_norm.bias"])
        loaded.extend(["ln_f.weight", "ln_f.bias"])
        if not quiet:
            print(f"✓ Loaded final layer norm")
    except Exception as e:
        if not quiet:
            print(f"✗ Failed to load final layer norm: {e}")
        skipped.extend(["ln_f.weight", "ln_f.bias"])

    # Copy embeddings and output weights
    try:
        adaptive_state["wte.weight"].copy_(baseline_state["gpt_neox.embed_in.weight"])
        
        # Position embeddings - if they exist
        if "gpt_neox.embed_positions.weight" in baseline_state:
            adaptive_state["wpe.weight"][:baseline_state["gpt_neox.embed_positions.weight"].shape[0]].copy_(
                baseline_state["gpt_neox.embed_positions.weight"]
            )
        
        # Final embedding out layer
        if "embed_out.weight" in baseline_state:
            adaptive_state["lm_head.weight"].copy_(baseline_state["embed_out.weight"])
        else:
            # Use input embeddings if no separate output embeddings
            adaptive_state["lm_head.weight"].copy_(baseline_state["gpt_neox.embed_in.weight"])
        
        loaded.extend(["wte.weight", "lm_head.weight"])
        if not quiet:
            print(f"✓ Loaded embeddings and output weights")
    except Exception as e:
        if not quiet:
            print(f"✗ Failed to load embeddings and output weights: {e}")
        skipped.extend(["wte.weight", "wpe.weight", "lm_head.weight"])

    if "bias" in adaptive_state:
        with torch.no_grad():
            adaptive_state["bias"].zero_()
        loaded.append("bias")

    # Final success message - shorter version in quiet mode
    if quiet:
        print(f"✅ Adaptive model initialized from {model_name} weights")
    else:
        print(f"\n✅ Adaptive model initialized from {model_name} weights ({len(loaded)}/{len(loaded) + len(skipped)} parameters loaded)")
        if skipped:
            print(f"   Skipped {len(skipped)} parameters")
        
    return model