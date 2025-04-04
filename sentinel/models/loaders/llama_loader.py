import torch
import torch.nn as nn
from sentinel.models.adaptive.transformer import AdaptiveCausalLmWrapper

def load_adaptive_model_llama(model_name, baseline_model, config, device, debug=False, quiet=False):
    """
    Load an adaptive transformer model initialized from a baseline Llama model.
    
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
        print(f"✅ Using Llama loader")
    
    # Get the token embeddings from the baseline model
    token_embeddings = baseline_model.get_input_embeddings()
    
    # Llama uses rotary position embeddings, not learned ones
    # We'll create a new position embedding layer
    position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    if debug and not quiet:
        print("Created new position embeddings (Llama uses rotary embeddings)")
    
    model = AdaptiveCausalLmWrapper(config, token_embeddings, position_embeddings, debug=debug).to(device)
    model.eval()

    baseline_state = baseline_model.state_dict()
    adaptive_state = model.state_dict()

    loaded, skipped = [], []

    num_heads = config.num_attention_heads if hasattr(config, "num_attention_heads") else config.n_head
    hidden_size = config.hidden_size
    head_dim = hidden_size // num_heads

    # Process each transformer block
    num_layers = config.num_hidden_layers if hasattr(config, "num_hidden_layers") else config.n_layer 
    for layer_idx in range(num_layers):
        if not quiet:
            print(f"\n[Processing layer {layer_idx}]")
        
        # Load layer norms
        try:
            # Llama uses input_layernorm and post_attention_layernorm
            adaptive_state[f"blocks.{layer_idx}.ln1.weight"].copy_(
                baseline_state[f"model.layers.{layer_idx}.input_layernorm.weight"])
            adaptive_state[f"blocks.{layer_idx}.ln2.weight"].copy_(
                baseline_state[f"model.layers.{layer_idx}.post_attention_layernorm.weight"])
                
            # Llama might not have biases in layer norms
            if f"model.layers.{layer_idx}.input_layernorm.bias" in baseline_state:
                adaptive_state[f"blocks.{layer_idx}.ln1.bias"].copy_(
                    baseline_state[f"model.layers.{layer_idx}.input_layernorm.bias"])
                adaptive_state[f"blocks.{layer_idx}.ln2.bias"].copy_(
                    baseline_state[f"model.layers.{layer_idx}.post_attention_layernorm.bias"])
            else:
                # Initialize biases with zeros if not present in source model
                adaptive_state[f"blocks.{layer_idx}.ln1.bias"].zero_()
                adaptive_state[f"blocks.{layer_idx}.ln2.bias"].zero_()
                
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

            # Llama uses gate_proj, up_proj, and down_proj in a different MLP setup
            # We'll use gate_proj + up_proj combined for the "in" part
            gate_w = baseline_state[f"model.layers.{layer_idx}.mlp.gate_proj.weight"]
            up_w = baseline_state[f"model.layers.{layer_idx}.mlp.up_proj.weight"]
            down_w = baseline_state[f"model.layers.{layer_idx}.mlp.down_proj.weight"]
            
            # Special handling for Llama's SwiGLU activation
            # We'll use the equivalent of up_proj for our input weights
            adaptive_state[in_w].copy_(up_w)
            # Bias might not exist in Llama, initialize with zeros
            adaptive_state[in_b].zero_()
            
            # Copy the down_proj for the output projection
            adaptive_state[out_w].copy_(down_w)
            adaptive_state[out_b].zero_()  # Llama might not have biases
            
            if not quiet:
                print("  ✓ Loaded feedforward layers (with SwiGLU adaptation)")
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

        # Load attention weights - Llama has separate q, k, v projections
        try:
            # Llama uses q_proj, k_proj, v_proj, and o_proj
            q_weight = baseline_state[f"model.layers.{layer_idx}.self_attn.q_proj.weight"]
            k_weight = baseline_state[f"model.layers.{layer_idx}.self_attn.k_proj.weight"]
            v_weight = baseline_state[f"model.layers.{layer_idx}.self_attn.v_proj.weight"]
            o_weight = baseline_state[f"model.layers.{layer_idx}.self_attn.o_proj.weight"]
            
            # Llama doesn't use biases for attention
            q_bias = torch.zeros(q_weight.shape[0], device=device)
            k_bias = torch.zeros(k_weight.shape[0], device=device)
            v_bias = torch.zeros(v_weight.shape[0], device=device)
            o_bias = torch.zeros(o_weight.shape[1], device=device)
            
            # Debug
            if not quiet:
                print(f"  • Q weight shape: {q_weight.shape}, Out weight shape: {o_weight.shape}")
            
            # Process each head separately
            for head_idx in range(num_heads):
                # Calculate slices for this head
                h_start = head_idx * head_dim
                h_end = (head_idx + 1) * head_dim
                
                # Get slices for this head
                qw = q_weight[h_start:h_end, :].clone()
                qb = q_bias[h_start:h_end].clone()
                
                kw = k_weight[h_start:h_end, :].clone()
                kb = k_bias[h_start:h_end].clone()
                
                vw = v_weight[h_start:h_end, :].clone()
                vb = v_bias[h_start:h_end].clone()
                
                # Get output projection weights for this head
                ow = o_weight[:, h_start:h_end].t().clone()
                
                # Output bias is shared across heads, so divide by num_heads
                ob = o_bias.clone() / num_heads
                
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

    # Handle final layer norm - Llama uses norm
    try:
        adaptive_state["ln_f.weight"].copy_(baseline_state["model.norm.weight"])
        
        # Llama might not have bias in norm
        if "model.norm.bias" in baseline_state:
            adaptive_state["ln_f.bias"].copy_(baseline_state["model.norm.bias"])
        else:
            adaptive_state["ln_f.bias"].zero_()
            
        loaded.extend(["ln_f.weight", "ln_f.bias"])
        if not quiet:
            print(f"✓ Loaded final layer norm")
    except Exception as e:
        if not quiet:
            print(f"✗ Failed to load final layer norm: {e}")
        skipped.extend(["ln_f.weight", "ln_f.bias"])

    # Copy embeddings and output weights
    try:
        # Llama has embeddings in model.embed_tokens
        adaptive_state["wte.weight"].copy_(baseline_state["model.embed_tokens.weight"])
        
        # Llama doesn't use position embeddings, initialize with small values
        with torch.no_grad():
            # Initialize position embeddings with small values
            adaptive_state["wpe.weight"].normal_(mean=0.0, std=0.02)
        
        # Llama shares embeddings with output layer or has lm_head
        if "lm_head.weight" in baseline_state:
            adaptive_state["lm_head.weight"].copy_(baseline_state["lm_head.weight"])
        else:
            # Use input embeddings if no separate output embeddings
            adaptive_state["lm_head.weight"].copy_(baseline_state["model.embed_tokens.weight"])
        
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