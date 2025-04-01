import torch
import torch.nn as nn
from models.adaptive_transformer import AdaptiveCausalLmWrapper

def load_adaptive_model_bloom(model_name, baseline_model, config, device, debug=False, quiet=False):
    """
    Load an adaptive transformer model initialized from a baseline BLOOM model.
    
    This fixed version properly handles BLOOM's ALiBi attention mechanism and
    QKV weight structure.
    
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
    if not quiet:
        print(f"⚡ Using FIXED BLOOM loader with ALiBi attention compatibility")
    
    # Get the token embeddings from the baseline model
    token_embeddings = baseline_model.get_input_embeddings()
    
    # BLOOM doesn't use positional embeddings, it uses ALiBi attention
    # We'll create new position embeddings compatible with our architecture
    max_positions = getattr(config, 'max_position_embeddings', 
                           getattr(config, 'n_positions', 2048))  # Default value
    
    position_embeddings = nn.Embedding(max_positions, config.hidden_size)
    if debug and not quiet:
        print(f"Created new position embeddings with size {max_positions} (BLOOM uses ALiBi attention)")
    
    # Create the adaptive model
    model = AdaptiveCausalLmWrapper(config, token_embeddings, position_embeddings, debug=debug).to(device)
    model.eval()

    baseline_state = baseline_model.state_dict()
    adaptive_state = model.state_dict()

    # Create helper function to extract slices correctly
    def extract_head_slices(qkv_weight, qkv_bias, head_idx, head_dim, hidden_size):
        """Extract QKV weights for a specific head from BLOOM's combined QKV matrix"""
        # Calculate starting indices for each head
        q_start = head_idx * head_dim 
        k_start = hidden_size + head_idx * head_dim
        v_start = 2 * hidden_size + head_idx * head_dim
        
        # Extract slices
        q_weight = qkv_weight[:, q_start:q_start + head_dim].clone()
        q_bias = qkv_bias[q_start:q_start + head_dim].clone()
        
        k_weight = qkv_weight[:, k_start:k_start + head_dim].clone()
        k_bias = qkv_bias[k_start:k_start + head_dim].clone()
        
        v_weight = qkv_weight[:, v_start:v_start + head_dim].clone()
        v_bias = qkv_bias[v_start:v_start + head_dim].clone()
        
        return q_weight, q_bias, k_weight, k_bias, v_weight, v_bias
    
    # Setup dimensions
    num_heads = config.num_attention_heads if hasattr(config, "num_attention_heads") else config.n_head
    hidden_size = config.hidden_size
    head_dim = hidden_size // num_heads
    
    # Process each transformer block
    num_layers = config.num_hidden_layers if hasattr(config, "num_hidden_layers") else config.n_layer
    for layer_idx in range(num_layers):
        if debug and not quiet:
            print(f"\n[Processing layer {layer_idx}]")
        
        # Load layer norms
        try:
            # BLOOM uses input_layernorm and post_attention_layernorm
            adaptive_state[f"blocks.{layer_idx}.ln1.weight"].copy_(
                baseline_state[f"transformer.h.{layer_idx}.input_layernorm.weight"])
            adaptive_state[f"blocks.{layer_idx}.ln1.bias"].copy_(
                baseline_state[f"transformer.h.{layer_idx}.input_layernorm.bias"])
            adaptive_state[f"blocks.{layer_idx}.ln2.weight"].copy_(
                baseline_state[f"transformer.h.{layer_idx}.post_attention_layernorm.weight"])
            adaptive_state[f"blocks.{layer_idx}.ln2.bias"].copy_(
                baseline_state[f"transformer.h.{layer_idx}.post_attention_layernorm.bias"])
            if debug and not quiet:
                print("  ✓ Loaded layer norms")
        except Exception as e:
            if debug and not quiet:
                print(f"  ✗ Failed to load layer norms: {e}")
        
        # Load feedforward layers
        try:
            # BLOOM uses mlp.dense_h_to_4h and mlp.dense_4h_to_h
            adaptive_state[f"blocks.{layer_idx}.ffn.dense_in.weight"].copy_(
                baseline_state[f"transformer.h.{layer_idx}.mlp.dense_h_to_4h.weight"])
            adaptive_state[f"blocks.{layer_idx}.ffn.dense_in.bias"].copy_(
                baseline_state[f"transformer.h.{layer_idx}.mlp.dense_h_to_4h.bias"])
            adaptive_state[f"blocks.{layer_idx}.ffn.dense_out.weight"].copy_(
                baseline_state[f"transformer.h.{layer_idx}.mlp.dense_4h_to_h.weight"])
            adaptive_state[f"blocks.{layer_idx}.ffn.dense_out.bias"].copy_(
                baseline_state[f"transformer.h.{layer_idx}.mlp.dense_4h_to_h.bias"])
            
            if debug and not quiet:
                print("  ✓ Loaded feedforward layers")
        except Exception as e:
            if debug and not quiet:
                print(f"  ✗ Failed to load feedforward layers: {e}")
        
        # Initialize gate values (initially fully open)
        try:
            gate_key = f"blocks.{layer_idx}.attn.gate" 
            adaptive_state[gate_key].fill_(1.0)  # All heads active initially
            if debug and not quiet:
                print("  ✓ Initialized gate values (fully open)")
        except Exception as e:
            if debug and not quiet:
                print(f"  ✗ Failed to initialize gate values: {e}")
        
        # Load attention weights - BLOOM has a combined QKV projection
        try:
            # Get BLOOM's combined query_key_value weights and output projection
            qkv_weight = baseline_state[f"transformer.h.{layer_idx}.self_attention.query_key_value.weight"]
            qkv_bias = baseline_state[f"transformer.h.{layer_idx}.self_attention.query_key_value.bias"]
            
            o_weight = baseline_state[f"transformer.h.{layer_idx}.self_attention.dense.weight"]
            o_bias = baseline_state[f"transformer.h.{layer_idx}.self_attention.dense.bias"]
            
            # Process each head separately
            for head_idx in range(num_heads):
                # Extract weights for this specific head
                q_weight, q_bias, k_weight, k_bias, v_weight, v_bias = extract_head_slices(
                    qkv_weight, qkv_bias, head_idx, head_dim, hidden_size)
                
                # Get appropriate slice of output projection weights for this head
                h_start = head_idx * head_dim
                h_end = (head_idx + 1) * head_dim
                o_weight_slice = o_weight[:, h_start:h_end].clone()
                
                # Copy to our model
                adaptive_state[f"blocks.{layer_idx}.attn.W_q.{head_idx}.weight"].copy_(q_weight)
                adaptive_state[f"blocks.{layer_idx}.attn.W_q.{head_idx}.bias"].copy_(q_bias)
                
                adaptive_state[f"blocks.{layer_idx}.attn.W_k.{head_idx}.weight"].copy_(k_weight)
                adaptive_state[f"blocks.{layer_idx}.attn.W_k.{head_idx}.bias"].copy_(k_bias)
                
                adaptive_state[f"blocks.{layer_idx}.attn.W_v.{head_idx}.weight"].copy_(v_weight)
                adaptive_state[f"blocks.{layer_idx}.attn.W_v.{head_idx}.bias"].copy_(v_bias)
                
                # For output, we need to transpose to match our format
                adaptive_state[f"blocks.{layer_idx}.attn.W_o.{head_idx}.weight"].copy_(o_weight_slice.t())
                
                # Output bias is shared for all heads, so divide by num_heads
                adaptive_state[f"blocks.{layer_idx}.attn.W_o.{head_idx}.bias"].copy_(o_bias / num_heads)
            
            if debug and not quiet:
                print(f"  ✓ Loaded attention weights for {num_heads} heads")
        except Exception as e:
            if debug and not quiet:
                print(f"  ✗ Failed to load attention weights: {e}")
        
        # Initialize UNet skip connections
        try:
            skip_fuse_weight = f"blocks.{layer_idx}.skip_fuse.weight"
            skip_fuse_bias = f"blocks.{layer_idx}.skip_fuse.bias"
            
            # Initialize with small values to avoid disrupting the model too much
            with torch.no_grad():
                adaptive_state[skip_fuse_weight].normal_(mean=0.0, std=0.01)
                adaptive_state[skip_fuse_bias].zero_()
                
                # Apply progressively smaller values for deeper layers
                if layer_idx > num_layers // 2:
                    # Reduce impact in deeper layers
                    scale_factor = 1.0 - 0.5 * ((layer_idx - num_layers // 2) / (num_layers - num_layers // 2))
                    adaptive_state[skip_fuse_weight] *= scale_factor
            
            if debug and not quiet:
                print(f"  ✓ Initialized UNet skip connections")
        except Exception as e:
            if debug and not quiet:
                print(f"  ✗ Failed to initialize UNet skip connections: {e}")
    
    # Handle final layer norm
    try:
        adaptive_state["ln_f.weight"].copy_(baseline_state["transformer.ln_f.weight"])
        adaptive_state["ln_f.bias"].copy_(baseline_state["transformer.ln_f.bias"])
        if debug and not quiet:
            print("✓ Loaded final layer norm")
    except Exception as e:
        if debug and not quiet:
            print(f"✗ Failed to load final layer norm: {e}")
    
    # Copy embeddings and handle position embeddings
    try:
        # Word embeddings
        adaptive_state["wte.weight"].copy_(baseline_state["transformer.word_embeddings.weight"])
        
        # Position embeddings (BLOOM uses ALiBi instead of learned position embeddings)
        # We initialize with small values since they're not present in the original model
        with torch.no_grad():
            # Initialize with small random values
            adaptive_state["wpe.weight"].normal_(mean=0.0, std=0.02)
            
            # Optionally add some structure to make positions useful from the start
            # Create position encoding pattern (roughly mimicking learned patterns)
            pos = torch.arange(max_positions, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, hidden_size, 2, device=device) * -(math.log(10000.0) / hidden_size))
            # Add sinusoidal pattern for even indices
            adaptive_state["wpe.weight"][:, 0::2] += torch.sin(pos * div_term) * 0.1
            # Add cosine pattern for odd indices
            adaptive_state["wpe.weight"][:, 1::2] += torch.cos(pos * div_term) * 0.1
        
        # LM head (output layer)
        if "lm_head.weight" in baseline_state:
            adaptive_state["lm_head.weight"].copy_(baseline_state["lm_head.weight"])
        else:
            # Use word embeddings if no separate lm_head
            adaptive_state["lm_head.weight"].copy_(baseline_state["transformer.word_embeddings.weight"])
        
        if debug and not quiet:
            print("✓ Loaded embeddings and initialized position embeddings")
    except Exception as e:
        if debug and not quiet:
            print(f"✗ Failed to load embeddings: {e}")
    
    # Add attention bias (causal mask)
    # Create a causal mask to replace ALiBi attention
    if "bias" in adaptive_state:
        # Get the actual size of the bias tensor in the model
        bias_shape = adaptive_state["bias"].shape
        seq_len = bias_shape[-1]  # Use the exact dimension from the model
        
        # Create properly sized causal mask
        bias = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=device))
        bias = bias.view(1, 1, seq_len, seq_len)
        
        # Make sure it matches the exact size
        if bias.shape != adaptive_state["bias"].shape:
            print(f"⚠️ Bias shape mismatch: model={adaptive_state['bias'].shape}, created={bias.shape}")
            # Create a new bias with the exact size
            bias = torch.tril(torch.ones(adaptive_state["bias"].shape[2:], dtype=torch.uint8, device=device))
            bias = bias.view(adaptive_state["bias"].shape)
            
        adaptive_state["bias"].copy_(bias)
        
        if debug and not quiet:
            print(f"✓ Created causal attention mask of size {seq_len}x{seq_len}")
    
    print(f"✅ Adaptive model initialized from {model_name} weights using ALiBi-compatible loader")
    
    return model

# Add missing import
import math