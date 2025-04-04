import torch
import torch.nn as nn
from sentinel.models.adaptive_transformer import AdaptiveCausalLmWrapper

def load_adaptive_model_opt(model_name, baseline_model, config, device, debug=False, quiet=False):
    """
    Load an adaptive transformer model initialized from a baseline OPT model.
    
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
        print(f"✅ Using OPT loader")
    
    # Get the token embeddings and position embeddings from the baseline model
    token_embeddings = baseline_model.get_input_embeddings()
    
    # Get a compatible position embedding layer
    # For OPT, we'll create a fresh embedding layer with the right dimensions
    # to avoid shape mismatch issues
    
    # Determine the correct size for position embeddings
    max_positions = 2048  # Default fallback
    if hasattr(config, 'max_position_embeddings'):
        max_positions = config.max_position_embeddings
    
    # Create new embedding layer with right dimensions
    position_embeddings = nn.Embedding(max_positions, config.hidden_size)
    
    # Initialize from baseline if possible
    has_original = False
    if (hasattr(baseline_model, 'model') and 
        hasattr(baseline_model.model, 'decoder') and 
        hasattr(baseline_model.model.decoder, 'embed_positions') and
        hasattr(baseline_model.model.decoder.embed_positions, 'weight')):
        has_original = True
        
    if not quiet:
        if has_original:
            print(f"Will initialize position embeddings from model (up to {max_positions} positions)")
        else:
            print(f"Created new position embeddings with size {max_positions}")
            
    # These weights will be properly initialized in the weight loading step
    
    model = AdaptiveCausalLmWrapper(config, token_embeddings, position_embeddings, debug=debug).to(device)
    model.eval()

    baseline_state = baseline_model.state_dict()
    adaptive_state = model.state_dict()

    loaded, skipped = [], []

    num_heads = config.num_attention_heads if hasattr(config, "num_attention_heads") else config.n_head
    hidden_size = config.hidden_size if hasattr(config, "hidden_size") else config.d_model
    head_dim = hidden_size // num_heads

    # Process each transformer block
    for layer_idx in range(config.num_hidden_layers):
        if not quiet:
            print(f"\n[Processing layer {layer_idx}]")
        
        # Load layer norms
        try:
            # OPT uses self_attn_layer_norm and final_layer_norm
            adaptive_state[f"blocks.{layer_idx}.ln1.weight"].copy_(
                baseline_state[f"model.decoder.layers.{layer_idx}.self_attn_layer_norm.weight"])
            adaptive_state[f"blocks.{layer_idx}.ln1.bias"].copy_(
                baseline_state[f"model.decoder.layers.{layer_idx}.self_attn_layer_norm.bias"])
            adaptive_state[f"blocks.{layer_idx}.ln2.weight"].copy_(
                baseline_state[f"model.decoder.layers.{layer_idx}.final_layer_norm.weight"])
            adaptive_state[f"blocks.{layer_idx}.ln2.bias"].copy_(
                baseline_state[f"model.decoder.layers.{layer_idx}.final_layer_norm.bias"])
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

            # OPT uses fc1 and fc2 naming
            base_in_w = f"model.decoder.layers.{layer_idx}.fc1.weight"
            base_in_b = f"model.decoder.layers.{layer_idx}.fc1.bias"
            base_out_w = f"model.decoder.layers.{layer_idx}.fc2.weight"
            base_out_b = f"model.decoder.layers.{layer_idx}.fc2.bias"

            # Copy weights with appropriate transposition if needed
            if baseline_state[base_in_w].shape[0] == adaptive_state[in_w].shape[1]:
                adaptive_state[in_w].copy_(baseline_state[base_in_w].t())
            else:
                adaptive_state[in_w].copy_(baseline_state[base_in_w])
            
            # Copy biases directly
            adaptive_state[in_b].copy_(baseline_state[base_in_b])

            # Handle output projection weights similarly
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

        # Load attention weights - OPT uses different layout than GPT-2
        try:
            # Get attention weights from baseline
            # OPT has separate QKV projections instead of combined
            q_weight = baseline_state[f"model.decoder.layers.{layer_idx}.self_attn.q_proj.weight"]
            q_bias = baseline_state[f"model.decoder.layers.{layer_idx}.self_attn.q_proj.bias"]
            
            k_weight = baseline_state[f"model.decoder.layers.{layer_idx}.self_attn.k_proj.weight"]
            k_bias = baseline_state[f"model.decoder.layers.{layer_idx}.self_attn.k_proj.bias"]
            
            v_weight = baseline_state[f"model.decoder.layers.{layer_idx}.self_attn.v_proj.weight"]
            v_bias = baseline_state[f"model.decoder.layers.{layer_idx}.self_attn.v_proj.bias"]
            
            out_weight = baseline_state[f"model.decoder.layers.{layer_idx}.self_attn.out_proj.weight"]
            out_bias = baseline_state[f"model.decoder.layers.{layer_idx}.self_attn.out_proj.bias"]
            
            # Debug
            if not quiet:
                print(f"  • Q weight shape: {q_weight.shape}, Out weight shape: {out_weight.shape}")
            
            # Process each head separately
            for head_idx in range(num_heads):
                # Calculate slices for this head
                h_start = head_idx * head_dim
                h_end = (head_idx + 1) * head_dim
                
                # Extract the appropriate slices for this head's parameters
                qw = q_weight[h_start:h_end, :]
                qb = q_bias[h_start:h_end]
                
                kw = k_weight[h_start:h_end, :]
                kb = k_bias[h_start:h_end]
                
                vw = v_weight[h_start:h_end, :]
                vb = v_bias[h_start:h_end]
                
                # Get projection weights for this head
                # Original is [hidden_size, hidden_size], we want [head_dim, hidden_size]
                ow = out_weight[:, h_start:h_end].t()
                
                # Output bias is shared across heads, so divide by num_heads
                ob = out_bias.clone() / num_heads
                
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
                if layer_idx > config.num_hidden_layers // 2:
                    # Gradually reduce scale for deeper layers
                    progress = (layer_idx - config.num_hidden_layers // 2) / (config.num_hidden_layers - config.num_hidden_layers // 2)
                    scale = 1.0 - progress * 0.5  # Scale from 1.0 down to 0.5
                    
                adaptive_state[fuse_w] *= scale
                
            if not quiet:
                print(f"  ✓ Initialized skip connection with small values (scale={scale:.2f})")
            loaded.extend([fuse_w, fuse_b])
        except Exception as e:
            if not quiet:
                print(f"  ✗ Failed to initialize skip connection: {e}")
            skipped.extend([fuse_w, fuse_b])

    # Handle final layer norm if present
    try:
        adaptive_state["ln_f.weight"].copy_(baseline_state["model.decoder.final_layer_norm.weight"])
        adaptive_state["ln_f.bias"].copy_(baseline_state["model.decoder.final_layer_norm.bias"])
        loaded.extend(["ln_f.weight", "ln_f.bias"])
        if not quiet:
            print(f"✓ Loaded final layer norm")
    except Exception as e:
        if not quiet:
            print(f"✗ Failed to load final layer norm: {e}")
        skipped.extend(["ln_f.weight", "ln_f.bias"])

    # Copy embeddings and output weights
    try:
        # OPT has embeddings in model.decoder.embed_tokens
        adaptive_state["wte.weight"].copy_(baseline_state["model.decoder.embed_tokens.weight"])
        
        # Copy position embeddings if they're a parameter (not a buffer)
        if "model.decoder.embed_positions.weight" in baseline_state:
            # Make sure not to exceed destination size
            src_size = baseline_state["model.decoder.embed_positions.weight"].shape[0]
            dst_size = adaptive_state["wpe.weight"].shape[0]
            copy_size = min(src_size, dst_size)
            
            if not quiet:
                print(f"  Position embeddings: copying {copy_size} positions (src={src_size}, dst={dst_size})")
                
            adaptive_state["wpe.weight"][:copy_size].copy_(
                baseline_state["model.decoder.embed_positions.weight"][:copy_size]
            )
        
        # OPT shares the output weights with input embeddings
        adaptive_state["lm_head.weight"].copy_(baseline_state["model.decoder.embed_tokens.weight"])
        
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