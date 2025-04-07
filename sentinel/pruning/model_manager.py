"""
Model Manager for Pruning Experiments

This module handles loading, saving, and managing transformer models for pruning experiments.
"""

import os
import torch
from typing import Tuple, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_name: str, device: str = "cuda") -> Tuple[torch.nn.Module, Any]:
    """
    Load model and tokenizer from HuggingFace.
    
    Args:
        model_name: The name of the model (e.g., "distilgpt2")
        device: The device to load the model onto ("cuda" or "cpu")
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Move model to device
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(device)
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {param_count/1e6:.2f}M parameters")
    
    return model, tokenizer


def save_model(model: torch.nn.Module, tokenizer: Any, path: str) -> None:
    """
    Save model and tokenizer to disk.
    
    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        path: The path to save to
        
    Returns:
        None
    """
    print(f"Saving model to: {path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(path)
    
    # Save tokenizer
    tokenizer.save_pretrained(path)
    
    print(f"Model saved successfully to: {path}")


def load_pruned_model(path: str, device: str = "cuda") -> Tuple[torch.nn.Module, Any]:
    """
    Load a previously pruned model and tokenizer from disk.
    
    Args:
        path: The path to load from
        device: The device to load onto
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading pruned model from: {path}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Move model to device
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(device)
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Pruned model loaded with {param_count/1e6:.2f}M parameters")
    
    return model, tokenizer