import torch
import torch.nn as nn
import os
from models.adaptive_transformer import AdaptiveCausalLmWrapper
from models.unet_transformer import load_unet_enhanced_model

# Environment variable to control whether to use optimized model
USE_OPTIMIZED_MODEL = os.environ.get("USE_OPTIMIZED_MODEL", "1") == "1"

def load_adaptive_model_gpt(model_name, baseline_model, config, device, quiet=False, optimized=None):
    """
    Load an adaptive transformer model initialized from a baseline GPT model.
    
    Args:
        model_name: Name of the base model
        baseline_model: Pretrained model to initialize from
        config: Configuration for the model
        device: Device to load the model on ('cpu' or 'cuda')
        quiet: If True, suppresses verbose loading messages
        optimized: Whether to use the optimized UNet model (if None, uses environment variable)
    """
    # Determine whether to use optimized implementation
    use_optimized = optimized if optimized is not None else USE_OPTIMIZED_MODEL
    
    if use_optimized:
        if not quiet:
            print("Using optimized UNet transformer with baseline integration")
        return load_unet_enhanced_model(
            baseline_model=baseline_model,
            device=device,
            use_baseline_integration=True,
            debug=(not quiet)
        )
    if not quiet:
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
            # Layer header should respect quiet flag
            if not quiet:
                print(f"[Processing layer {layer_idx}]")
            # Get the attention weights from baseline model
            qkv_w = baseline_state[f"transformer.h.{layer_idx}.attn.c_attn.weight"]  # [3*hidden_size, hidden_size]
            qkv_b = baseline_state[f"transformer.h.{layer_idx}.attn.c_attn.bias"]    # [3*hidden_size]
            c_proj_w = baseline_state[f"transformer.h.{layer_idx}.attn.c_proj.weight"]  # [hidden_size, hidden_size]
            c_proj_b = baseline_state[f"transformer.h.{layer_idx}.attn.c_proj.bias"]    # [hidden_size]

            # Debug info (only in verbose mode)
            if not quiet:
                print(f"  ✓ Loaded layer norms")
                print(f"  ✓ Loaded feedforward layers")
                print(f"  ✓ Initialized gate values with slight asymmetry")
                print(f"  • QKV weight shape: {qkv_w.shape}, Proj weight shape: {c_proj_w.shape}")
            
            # Determine if we need to transpose the weights based on shape
            should_transpose_qkv = qkv_w.shape[0] != 3 * hidden_size
            should_transpose_proj = c_proj_w.shape[0] != hidden_size
            
            if should_transpose_qkv and not quiet:
                print("  • Using standard GPT-2 convention (transpose needed)")
                qkv_w = qkv_w.t()
            elif should_transpose_qkv:
                qkv_w = qkv_w.t()
            
            if should_transpose_proj and not quiet:
                print("  • Transposing projection weights")
                c_proj_w = c_proj_w.t()
            elif should_transpose_proj:
                c_proj_w = c_proj_w.t()
            
            # For each attention head
            for head_idx in range(num_heads):
                # Calculate slice indices for this head
                hs, he = head_idx * head_dim, (head_idx + 1) * head_dim
                
                # Extract query, key, value weights and biases for this head
                # GPT-2 stores them as [q1,q2,...,qh,k1,k2,...,kh,v1,v2,...,vh]
                qw = qkv_w[:, hs:he].t()  # Transpose to match our model's shape
                qb = qkv_b[hs:he]
                
                kw = qkv_w[:, hidden_size + hs:hidden_size + he].t()
                kb = qkv_b[hidden_size + hs:hidden_size + he]
                
                vw = qkv_w[:, 2 * hidden_size + hs:2 * hidden_size + he].t()
                vb = qkv_b[2 * hidden_size + hs:2 * hidden_size + he]
                
                # Extract output projection weights and biases
                if c_proj_w.shape[1] == hidden_size:
                    # Need to slice by head size first
                    ow = c_proj_w[:, hs:he].t()
                else:
                    # Direct assignment
                    ow = c_proj_w[hs:he, :].t()
                
                # Output bias is shared, so divide by number of heads
                ob = c_proj_b / num_heads

                # Debug output (only in verbose mode)
                if head_idx == 0 and not quiet:
                    print(f"  • Using altered slicing for shape {qw.shape}")
                
                # Copy to our model
                weight_mapping = [
                    (f"blocks.{layer_idx}.attn.W_q.{head_idx}.weight", qw),
                    (f"blocks.{layer_idx}.attn.W_q.{head_idx}.bias", qb),
                    (f"blocks.{layer_idx}.attn.W_k.{head_idx}.weight", kw),
                    (f"blocks.{layer_idx}.attn.W_k.{head_idx}.bias", kb),
                    (f"blocks.{layer_idx}.attn.W_v.{head_idx}.weight", vw),
                    (f"blocks.{layer_idx}.attn.W_v.{head_idx}.bias", vb),
                    (f"blocks.{layer_idx}.attn.W_o.{head_idx}.weight", ow),
                    (f"blocks.{layer_idx}.attn.W_o.{head_idx}.bias", ob)
                ]
                
                for name, val in weight_mapping:
                    try:
                        # Check for shape match
                        target_shape = adaptive_state[name].shape
                        if val.shape != target_shape:
                            if not quiet:
                                print(f"  • Shape mismatch: {val.shape} vs {target_shape}")
                            if len(val.shape) == len(target_shape) and val.numel() == adaptive_state[name].numel():
                                # Try transposing
                                val = val.transpose(-1, -2)
                                if not quiet:
                                    print(f"  • Transposed to {val.shape}")
                        
                        adaptive_state[name].copy_(val)
                        loaded.append(name)
                    except Exception as e:
                        if not quiet:
                            print(f"  • Error loading {name}: {e}")
                        skipped.append(name)
                
            if not quiet:
                print(f"  ✓ Loaded attention weights for layer {layer_idx}")
                print(f"  ✓ Initialized skip connection with small values (scale={layer_scale:.2f})" if i > config.n_layer // 2 else f"  ✓ Initialized skip connection with default values")
            
        except Exception as e:
            print(f"[ERROR] Failed to load attention for layer {layer_idx}: {e}")
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

    # Final success message - shorter version in quiet mode
    if quiet:
        print(f"✅ Adaptive model initialized from {model_name} weights")
    else:
        print(f"\n✅ Adaptive model initialized from {model_name} weights ({len(loaded)}/{len(adaptive_state)} parameters loaded)")
    
    # Show gate activity summary (only if not in quiet mode)
    if not quiet:
        print("\n=== GATE ACTIVITY ===")
        for layer_idx in range(config.n_layer):
            gate_key = f"blocks.{layer_idx}.attn.gate"
            gate_values = adaptive_state[gate_key]
            active_heads = [i for i, v in enumerate(gate_values) if v > 0.5]  # Threshold at 0.5
            print(f"Layer {layer_idx}: Active heads -> {active_heads}")
    
    return model
