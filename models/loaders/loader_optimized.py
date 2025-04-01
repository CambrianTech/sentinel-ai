"""
Optimized model loader with performance improvements

This module provides optimized model loading functionality with
performance improvements based on profiling results. It includes
configurable optimization levels to experiment with different
approaches to performance improvement.
"""

import os
import torch
from transformers import AutoModelForCausalLM

def load_optimized_baseline_model(model_name, device):
    """Load a baseline model with the specified name."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"✅ Loaded baseline model: {model_name} with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model

def load_optimized_adaptive_model(model_name, baseline_model, device, debug=True, quiet=False, optimization_level=None):
    """
    Load an adaptive transformer model with optimizations based on profiling results.
    
    Args:
        model_name: Name of the base model
        baseline_model: Pretrained model to initialize from
        device: Device to load the model on ('cpu' or 'cuda')
        debug: Enable debug output
        quiet: Suppress verbose output
        optimization_level: Override the default optimization level (0-3)
        
    Returns:
        An optimized adaptive transformer model
    """
    config = baseline_model.config
    
    # Determine model loading approach based on model type
    if "gpt" in model_name.lower():
        from models.loaders.gpt2_loader_optimized import load_optimized_adaptive_model_gpt
        return load_optimized_adaptive_model_gpt(
            model_name, 
            baseline_model, 
            config, 
            device, 
            quiet=quiet,
            optimization_level=optimization_level
        )
    else:
        # Default fallback to original loader for unsupported models
        from models.loaders.loader import load_adaptive_model
        print("Warning: Using original loader for unsupported model type")
        return load_adaptive_model(model_name, baseline_model, device, debug, quiet)