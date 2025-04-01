"""
Optimized GPT2 Loader with Enhanced Integration

This module provides an optimized loader for GPT2-based models with
improved performance characteristics. It incorporates lessons learned
from profiling the original implementation to address key bottlenecks.
"""

import torch
import torch.nn as nn
import os
from models.optimized_attention import OptimizedGatedMultiHeadAttention
from models.unet_transformer_optimized import load_optimized_unet_model
from models.unet_transformer import load_unet_enhanced_model

# Environment variable to control optimization level
OPTIMIZATION_LEVEL = int(os.environ.get("OPTIMIZATION_LEVEL", "2"))

def load_optimized_adaptive_model_gpt(model_name, baseline_model, config, device, quiet=False, optimization_level=None):
    """
    Load an optimized adaptive transformer model initialized from a baseline GPT model.
    
    Args:
        model_name: Name of the base model
        baseline_model: Pretrained model to initialize from
        config: Configuration for the model
        device: Device to load the model on ('cpu' or 'cuda')
        quiet: If True, suppresses verbose loading messages
        optimization_level: Override the default optimization level (0-3)
            0: Use original implementation
            1: Use optimized attention only
            2: Use optimized attention + efficient integration (default)
            3: Use all optimizations + caching
    """
    # Determine optimization level
    opt_level = optimization_level if optimization_level is not None else OPTIMIZATION_LEVEL
    
    if not quiet:
        print(f"Loading adaptive model with optimization level {opt_level}")
    
    # Optimization level 0: Use original implementation
    if opt_level == 0:
        from models.loaders.gpt2_loader import load_adaptive_model_gpt
        return load_adaptive_model_gpt(model_name, baseline_model, config, device, quiet)
    
    # Optimization level 1+: Use at least optimized attention
    if opt_level >= 1:
        use_baseline_integration = True  # Used for level 2+
        
        if opt_level >= 2:
            # Use optimized UNet implementation with efficient integration
            return load_optimized_unet_model(
                baseline_model=baseline_model,
                device=device,
                use_baseline_integration=use_baseline_integration,
                debug=(not quiet)
            )
        else:
            # Level 1: Just use optimized attention in original UNet
            from models.unet_transformer import load_unet_enhanced_model
            return load_unet_enhanced_model(
                baseline_model=baseline_model,
                device=device,
                use_baseline_integration=use_baseline_integration,
                debug=(not quiet)
            )
    
    # Should never reach here
    raise ValueError(f"Invalid optimization level: {opt_level}")