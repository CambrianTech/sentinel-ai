"""Model management utilities for pruning experiments

This module provides utilities for loading and managing transformer models.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name, device=None):
    """Load a transformer model and tokenizer.
    
    Args:
        model_name: Name of the model to load (e.g., 'distilgpt2')
        device: Device to load the model onto (e.g., 'cuda', 'cpu')
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Set device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Load model and tokenizer
    print(f"Loading {model_name} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Successfully loaded {model_name}")
    return model, tokenizer


def add_head_gates(model):
    """Add gates to attention heads for pruning.
    
    This function adds a 'gate' parameter to each attention module in the model.
    Gates are initialized to 1.0 and can be set to 0.0 to prune heads.
    
    Args:
        model: A HuggingFace transformer model
        
    Returns:
        The model with gates added
    """
    # Get transformer blocks
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        blocks = model.transformer.h  # GPT-2 structure
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'blocks'):
        blocks = model.transformer.blocks
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        blocks = model.encoder.layer  # BERT structure
    elif hasattr(model, 'layers'):
        blocks = model.layers  # OPT, BLOOM structure
    else:
        print(f"Warning: Could not find blocks in model structure. Gates not added.")
        return model
    
    # Add gates to each attention module
    for i, block in enumerate(blocks):
        # Find attention module
        attn_module = None
        if hasattr(block, 'attn'):
            attn_module = block.attn
        elif hasattr(block, 'attention') and hasattr(block.attention, 'self'):
            attn_module = block.attention.self
        elif hasattr(block, 'self_attn'):
            attn_module = block.self_attn
        else:
            # Try common attribute names
            for attr in ['mha', 'self_attention', 'attention']:
                if hasattr(block, attr):
                    attn_module = getattr(block, attr)
                    break
        
        # If attention module found, add gate
        if attn_module is not None:
            # Determine number of heads
            num_heads = None
            # First, try direct attributes
            for attr in ['num_heads', 'n_head', 'num_attention_heads']:
                if hasattr(attn_module, attr):
                    num_heads = getattr(attn_module, attr)
                    break
            
            # If not found, look in parent block
            if num_heads is None and hasattr(block, 'num_heads'):
                num_heads = block.num_heads
            elif num_heads is None and hasattr(block, 'n_head'):
                num_heads = block.n_head
            
            # Default based on model architecture
            if num_heads is None:
                if 'distilgpt2' in model.config.name_or_path:
                    num_heads = 12
                elif 'gpt2' in model.config.name_or_path:
                    if 'medium' in model.config.name_or_path:
                        num_heads = 16
                    elif 'large' in model.config.name_or_path:
                        num_heads = 20
                    elif 'xl' in model.config.name_or_path:
                        num_heads = 25
                    else:
                        num_heads = 12
                else:
                    # Use a safe default
                    num_heads = 12
            
            # Create gate parameter if it doesn't exist
            if not hasattr(attn_module, 'gate'):
                attn_module.register_parameter(
                    'gate',
                    torch.nn.Parameter(torch.ones(num_heads, device=model.device))
                )
                print(f"Added gate for block {i} with {num_heads} heads")
            else:
                print(f"Block {i} already has a gate parameter")
    
    # Monkey patch the forward pass of attention modules (if needed)
    # This is model-specific and would be implemented based on the model architecture
    
    return model
