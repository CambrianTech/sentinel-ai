"""
Improved GPT2 weight loader to handle weight tensor shape issues
"""
import torch
import torch.nn as nn
from models.adaptive_transformer import AdaptiveCausalLmWrapper

def load_adaptive_model_gpt(model_name, baseline_model, config, device):
    """
    Load an adaptive transformer model initialized from a baseline GPT model.
    Properly handles GPT-2's specific weight storage format.
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

    # Get the token and position embeddings
    token_embeddings = baseline_model.get_input_embeddings()
    
    # For GPT-2, position embeddings are stored in transformer.wpe
    if hasattr(baseline_model, 'transformer') and hasattr(baseline_model.transformer, 'wpe'):
        position_embeddings = baseline_model.transformer.wpe
    else:
        position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        print("Warning: Could not find position embeddings, created new ones")
    
    # Create our adaptive model
    model = AdaptiveCausalLmWrapper(config, token_embeddings, position_embeddings).to(device)
    model.eval()
    
    # Get state dictionaries
    baseline_state = baseline_model.state_dict()
    adaptive_state = model.state_dict()
    
    # Track successfully loaded and skipped parameters
    loaded, skipped = [], []
    
    # Extract dimensions
    num_heads = config.n_head if hasattr(config, "n_head") else config.num_attention_heads
    hidden_size = config.n_embd if hasattr(config, "n_embd") else config.hidden_size
    head_dim = hidden_size // num_heads
    
    # Copy layernorms
    try:
        # Final layer norm
        adaptive_state["ln_f.weight"].copy_(baseline_state["transformer.ln_f.weight"])
        adaptive_state["ln_f.bias"].copy_(baseline_state["transformer.ln_f.bias"])
        loaded.extend(["ln_f.weight", "ln_f.bias"])
    except Exception as e:
        print(f"Error loading final layernorm: {e}")
        skipped.extend(["ln_f.weight", "ln_f.bias"])
    
    # Process each transformer layer
    for layer_idx in range(config.n_layer):
        print(f"\n[Processing layer {layer_idx}]")
        
        # 1. Load layer norms
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
            print("  ✓ Loaded layer norms")
        except Exception as e:
            print(f"  ✗ Error loading layer norms: {e}")
            skipped.extend([
                f"blocks.{layer_idx}.ln1.weight", f"blocks.{layer_idx}.ln1.bias",
                f"blocks.{layer_idx}.ln2.weight", f"blocks.{layer_idx}.ln2.bias"
            ])
        
        # 2. Load feedforward layers
        try:
            # Get weights from baseline
            mlp_fc_weight = baseline_state[f"transformer.h.{layer_idx}.mlp.c_fc.weight"] 
            mlp_fc_bias = baseline_state[f"transformer.h.{layer_idx}.mlp.c_fc.bias"]
            mlp_proj_weight = baseline_state[f"transformer.h.{layer_idx}.mlp.c_proj.weight"]
            mlp_proj_bias = baseline_state[f"transformer.h.{layer_idx}.mlp.c_proj.bias"]
            
            # Check and handle transposition if needed
            if mlp_fc_weight.shape[1] != adaptive_state[f"blocks.{layer_idx}.ffn.dense_in.weight"].shape[1]:
                mlp_fc_weight = mlp_fc_weight.t()
            if mlp_proj_weight.shape[1] != adaptive_state[f"blocks.{layer_idx}.ffn.dense_out.weight"].shape[1]:
                mlp_proj_weight = mlp_proj_weight.t()
                
            # Copy weights to our model
            adaptive_state[f"blocks.{layer_idx}.ffn.dense_in.weight"].copy_(mlp_fc_weight)
            adaptive_state[f"blocks.{layer_idx}.ffn.dense_in.bias"].copy_(mlp_fc_bias)
            adaptive_state[f"blocks.{layer_idx}.ffn.dense_out.weight"].copy_(mlp_proj_weight)
            adaptive_state[f"blocks.{layer_idx}.ffn.dense_out.bias"].copy_(mlp_proj_bias)
            
            loaded.extend([
                f"blocks.{layer_idx}.ffn.dense_in.weight", f"blocks.{layer_idx}.ffn.dense_in.bias",
                f"blocks.{layer_idx}.ffn.dense_out.weight", f"blocks.{layer_idx}.ffn.dense_out.bias"
            ])
            print("  ✓ Loaded feedforward layers")
        except Exception as e:
            print(f"  ✗ Error loading feedforward layers: {e}")
            skipped.extend([
                f"blocks.{layer_idx}.ffn.dense_in.weight", f"blocks.{layer_idx}.ffn.dense_in.bias",
                f"blocks.{layer_idx}.ffn.dense_out.weight", f"blocks.{layer_idx}.ffn.dense_out.bias"
            ])
        
        # 3. Initialize gates to reasonable values that create slight asymmetry
        try:
            gate_key = f"blocks.{layer_idx}.attn.gate"
            # Initialize with slightly different values to break symmetry
            num_heads = adaptive_state[gate_key].shape[0]
            gate_values = torch.ones_like(adaptive_state[gate_key])
            
            for h in range(num_heads):
                # Apply slight reduction based on position (0.9-1.0 range)
                pos_factor = 1.0 - 0.1 * abs(h - num_heads//2) / (num_heads//2)
                gate_values[h] = pos_factor
                
            adaptive_state[gate_key].copy_(gate_values)
            loaded.append(gate_key)
            print(f"  ✓ Initialized gate values with slight asymmetry")
        except Exception as e:
            print(f"  ✗ Error initializing gates: {e}")
            skipped.append(gate_key)
        
        # 4. Load attention weights - this is the tricky part
        try:
            # Get the combined QKV weights from GPT-2
            qkv_weight = baseline_state[f"transformer.h.{layer_idx}.attn.c_attn.weight"]
            qkv_bias = baseline_state[f"transformer.h.{layer_idx}.attn.c_attn.bias"]
            attn_proj_weight = baseline_state[f"transformer.h.{layer_idx}.attn.c_proj.weight"]
            attn_proj_bias = baseline_state[f"transformer.h.{layer_idx}.attn.c_proj.bias"]
            
            print(f"  • QKV weight shape: {qkv_weight.shape}, Proj weight shape: {attn_proj_weight.shape}")
            
            # GPT-2's attention weights might need transposing
            gpt2_convention = (qkv_weight.shape[0] == hidden_size and qkv_weight.shape[1] == 3 * hidden_size)
            if gpt2_convention:
                print("  • Using standard GPT-2 convention (transpose needed)")
                qkv_weight = qkv_weight.t()  # [3*hidden, hidden] after transpose
            else:
                print("  • Using non-standard weight format")
                
            # GPT-2 concatenates Q, K, V for all heads
            # Our model separates them by head
            for head_idx in range(num_heads):
                # Calculate slices for this head in the combined weights
                h_start = head_idx * head_dim
                h_end = (head_idx + 1) * head_dim
                
                # Extract Q, K, V slices for this head
                # In GPT-2: First hidden_size values are Q, next hidden_size are K, last hidden_size are V
                q_slice = slice(h_start, h_end)
                k_slice = slice(hidden_size + h_start, hidden_size + h_end)
                v_slice = slice(2 * hidden_size + h_start, 2 * hidden_size + h_end)
                
                # Extract weight matrices for Q, K, V
                # These need to be transposed to match our model's convention
                q_weight = qkv_weight[q_slice, :].clone()  # [head_dim, hidden_size]
                k_weight = qkv_weight[k_slice, :].clone()  # [head_dim, hidden_size]
                v_weight = qkv_weight[v_slice, :].clone()  # [head_dim, hidden_size]
                
                # Extract bias vectors
                q_bias = qkv_bias[q_slice].clone()  # [head_dim]
                k_bias = qkv_bias[k_slice].clone()  # [head_dim]
                v_bias = qkv_bias[v_slice].clone()  # [head_dim]
                
                # Extract output projection for this head
                # GPT-2 has a single output projection matrix; we need to slice it
                if attn_proj_weight.shape[0] == hidden_size:
                    # [hidden_size, hidden_size] -> slice to [hidden_size, head_dim]
                    o_weight = attn_proj_weight[:, h_start:h_end].clone()
                else:
                    # [head_dim, hidden_size] for each head
                    o_weight = attn_proj_weight[h_start:h_end, :].clone()
                    
                # Output bias is shared, so divide by number of heads
                o_bias = attn_proj_bias.clone() / num_heads
                
                # Transpose to match our model's expected dimensions if needed
                if q_weight.shape[1] != adaptive_state[f"blocks.{layer_idx}.attn.W_q.{head_idx}.weight"].shape[1]:
                    q_weight = q_weight.t()
                    k_weight = k_weight.t()
                    v_weight = v_weight.t()
                
                if o_weight.shape != adaptive_state[f"blocks.{layer_idx}.attn.W_o.{head_idx}.weight"].shape:
                    o_weight = o_weight.t()
                
                # Copy to our model's per-head parameters
                param_mapping = [
                    (f"blocks.{layer_idx}.attn.W_q.{head_idx}.weight", q_weight),
                    (f"blocks.{layer_idx}.attn.W_q.{head_idx}.bias", q_bias),
                    (f"blocks.{layer_idx}.attn.W_k.{head_idx}.weight", k_weight),
                    (f"blocks.{layer_idx}.attn.W_k.{head_idx}.bias", k_bias),
                    (f"blocks.{layer_idx}.attn.W_v.{head_idx}.weight", v_weight),
                    (f"blocks.{layer_idx}.attn.W_v.{head_idx}.bias", v_bias),
                    (f"blocks.{layer_idx}.attn.W_o.{head_idx}.weight", o_weight),
                    (f"blocks.{layer_idx}.attn.W_o.{head_idx}.bias", o_bias)
                ]
                
                # Copy the weights to our model
                for name, value in param_mapping:
                    try:
                        if value.shape != adaptive_state[name].shape:
                            print(f"  • Shape mismatch for {name}: {value.shape} vs {adaptive_state[name].shape}")
                            continue
                            
                        adaptive_state[name].copy_(value)
                        loaded.append(name)
                    except Exception as e:
                        print(f"  • Error loading {name}: {e}")
                        skipped.append(name)
                        
            print(f"  ✓ Loaded attention weights for layer {layer_idx}")
            
        except Exception as e:
            print(f"  ✗ Error loading attention weights: {e}")
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
        
        # 5. Initialize skip connections
        skip_fuse_weight = f"blocks.{layer_idx}.skip_fuse.weight"
        skip_fuse_bias = f"blocks.{layer_idx}.skip_fuse.bias"
        if skip_fuse_weight in adaptive_state:
            with torch.no_grad():
                # Initialize with small random values for stable gradient flow
                adaptive_state[skip_fuse_weight].normal_(mean=0.0, std=0.01)
                adaptive_state[skip_fuse_bias].zero_()
                
                # Apply layer-dependent scaling
                layer_pos = layer_idx - config.n_layer // 2
                if layer_pos > 0:
                    # Later layers get smaller initial values for stability
                    scaling = 1.0 - 0.5 * (layer_pos / (config.n_layer // 2))
                    adaptive_state[skip_fuse_weight].mul_(scaling)
                    
            loaded.extend([skip_fuse_weight, skip_fuse_bias])
            print(f"  ✓ Initialized skip connection with small values (scale={scaling if layer_pos > 0 else 1.0:.2f})")
    
    # Copy embeddings and output weights
    adaptive_state["wte.weight"].copy_(baseline_state["transformer.wte.weight"])
    adaptive_state["wpe.weight"][:baseline_state["transformer.wpe.weight"].shape[0]].copy_(
        baseline_state["transformer.wpe.weight"]
    )
    adaptive_state["lm_head.weight"].copy_(baseline_state["transformer.wte.weight"])
    loaded.extend(["wte.weight", "wpe.weight", "lm_head.weight"])
    print("✓ Loaded embeddings and output weights")
    
    # Initialize causal mask
    if "bias" in adaptive_state:
        with torch.no_grad():
            adaptive_state["bias"].zero_()
        loaded.append("bias")
    
    print(f"\n✅ Adaptive model initialized from {model_name} weights ({len(loaded)}/{len(adaptive_state)} parameters loaded)")
    print(f"   Skipped {len(skipped)} parameters")
    
    return model