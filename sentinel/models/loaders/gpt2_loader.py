import torch
import torch.nn as nn
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

from sentinel.models.adaptive_transformer import AdaptiveCausalLmWrapper
from sentinel.models.unet_transformer import load_unet_enhanced_model
from sentinel.models.unet_transformer_optimized import load_unet_enhanced_model_optimized

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
        return load_unet_enhanced_model_optimized(
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
        
    # Extract token and position embeddings from baseline model
    if hasattr(baseline_model, "get_input_embeddings"):
        token_embeddings = baseline_model.get_input_embeddings()
    else:
        token_embeddings = None
    
    # Get position embeddings correctly
    if hasattr(baseline_model, 'transformer') and hasattr(baseline_model.transformer, 'wpe'):
        position_embeddings = baseline_model.transformer.wpe
    else:
        # Create new position embeddings if needed
        max_pos_embeddings = getattr(config, 'max_position_embeddings', 1024)
        hidden_size = getattr(config, 'hidden_size', config.n_embd)
        position_embeddings = nn.Embedding(max_pos_embeddings, hidden_size)
    
    # Create adaptive transformer
    transformer = AdaptiveCausalLmWrapper(
        base_model=baseline_model,
        config=config,
        token_embeddings=token_embeddings,
        position_embeddings=position_embeddings
    )
    
    # Move to device
    transformer = transformer.to(device)
    
    if not quiet:
        print(f"Loaded adaptive transformer with gated attention")
        print(f"Using {len(transformer.transformer.blocks)} layers with "
              f"{transformer.transformer.blocks[0].attn.num_heads} heads each")
    
    return transformer

def load_gpt2_with_adaptive_transformer(model_name="gpt2", device="cuda", adaptive_layers=None, quiet=False, optimized=None):
    """
    Load a GPT-2 model with adaptive transformer architecture.
    
    Args:
        model_name: Name of the GPT-2 model variant (e.g., "gpt2", "distilgpt2", "gpt2-large")
        device: Device to load the model on ('cpu' or 'cuda')
        adaptive_layers: List of layer indices to make adaptive (if None, all layers)
        quiet: If True, suppresses verbose loading messages
        optimized: Whether to use the optimized implementation
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if not quiet:
        print(f"Loading GPT-2 model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load baseline model
    baseline_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Get configuration
    config = baseline_model.config
    
    # Create adaptive model
    adaptive_model = load_adaptive_model_gpt(
        model_name=model_name,
        baseline_model=baseline_model,
        config=config,
        device=device,
        quiet=quiet,
        optimized=optimized
    )
    
    return adaptive_model, tokenizer

def load_gpt2_with_sentinel_gates(model_name="gpt2", device="cuda", gate_init=1.0, 
                                 connection_init=0.0, norm_attn_output=True, debug=False):
    """
    Load a GPT-2 model with sentinel gates and agency capabilities.
    
    Args:
        model_name: Name of the GPT-2 model variant
        device: Device to load the model on
        gate_init: Initial value for gates
        connection_init: Initial value for U-Net connections
        norm_attn_output: Whether to normalize attention outputs
        debug: Whether to print debug information
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load model with adaptive transformer
    model, tokenizer = load_gpt2_with_adaptive_transformer(
        model_name=model_name,
        device=device,
        quiet=not debug,
        optimized=True  # Always use optimized version for this function
    )
    
    # Set initial gate values
    with torch.no_grad():
        for block in model.transformer.blocks:
            # Set gate values
            block.attn.gate.fill_(gate_init)
            
            # Set U-Net connection scale
            if hasattr(block, 'skip_scale'):
                block.skip_scale = connection_init
    
    if debug:
        print(f"Initialized model with gate value: {gate_init}")
        if connection_init > 0:
            print(f"Enabled U-Net connections with scale: {connection_init}")
    
    return model, tokenizer