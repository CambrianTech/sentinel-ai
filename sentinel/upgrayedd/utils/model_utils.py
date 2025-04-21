"""
Model utilities for loading, saving, and manipulating transformer models.
"""

import os
import torch
import logging
from typing import Tuple, Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

def load_model_and_tokenizer(
    model_name: str,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
    **model_kwargs
) -> Tuple[torch.nn.Module, Any]:
    """
    Load model and tokenizer from HuggingFace.
    
    Args:
        model_name: Model name or path
        cache_dir: Directory to cache models
        device: Device to load model on
        **model_kwargs: Additional keyword arguments to pass to the model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")
    
    try:
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        logger.info(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            **model_kwargs
        )
        
        # Move model to device
        device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        model = model.to(device_obj)
        
        # Log model size
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded with {param_count/1e6:.2f}M parameters on {device_obj}")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def save_model_and_tokenizer(
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: str,
    save_name: str = "model"
) -> str:
    """
    Save model and tokenizer to disk.
    
    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        output_dir: Directory to save to
        save_name: Name for the saved model directory
        
    Returns:
        Path to saved model
    """
    save_dir = os.path.join(output_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"Saving model to: {save_dir}")
    
    # Save model
    model.save_pretrained(save_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(save_dir)
    
    return save_dir

def get_model_info(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Get information about a model's architecture.
    
    Args:
        model: The model to inspect
        
    Returns:
        Dictionary with model information
    """
    info = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "layers": None,
        "heads": None,
        "hidden_size": None,
        "architecture": model.__class__.__name__,
        "device": next(model.parameters()).device.type
    }
    
    # Try to get more info from config
    if hasattr(model, "config"):
        config = model.config
        # For GPT-2 style configs
        if hasattr(config, "n_layer"):
            info["layers"] = config.n_layer
        # For BERT style configs
        elif hasattr(config, "num_hidden_layers"):
            info["layers"] = config.num_hidden_layers
            
        # Get head count
        if hasattr(config, "n_head"):
            info["heads"] = config.n_head
        elif hasattr(config, "num_attention_heads"):
            info["heads"] = config.num_attention_heads
            
        # Get hidden size
        if hasattr(config, "n_embd"):
            info["hidden_size"] = config.n_embd
        elif hasattr(config, "hidden_size"):
            info["hidden_size"] = config.hidden_size
    
    return info