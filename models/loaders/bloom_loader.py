"""
BLOOM Model Loader for Sentinel-AI - DEPRECATED MODULE

This module is a backwards compatibility layer for the old models.loaders.bloom_loader module.
The functionality has been moved to sentinel.models.loaders.bloom_loader.

Please update your imports to use the new module path.
"""

import torch
import torch.nn as nn
import warnings
from models.adaptive_transformer import AdaptiveCausalLmWrapper

# Emit deprecation warning
warnings.warn(
    "The models.loaders.bloom_loader module is deprecated. "
    "Please use sentinel.models.loaders.bloom_loader instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from the new location if available
try:
    from sentinel.models.loaders.bloom_loader import load_adaptive_model_bloom as _load_adaptive_model_bloom
    from sentinel.models.loaders.bloom_loader import load_bloom_with_adaptive_transformer as _load_bloom_with_adaptive_transformer
    
    # If imports succeed, redefine the functions to use the new implementations
    def load_adaptive_model_bloom(model_name, baseline_model, config, device, debug=False, quiet=False):
        return _load_adaptive_model_bloom(model_name, baseline_model, config, device, debug, quiet)
    
    def load_bloom_with_adaptive_transformer(model_name, baseline_model, config, device, debug=False, quiet=False):
        return _load_bloom_with_adaptive_transformer(model_name, baseline_model, config, device, debug, quiet)
except ImportError:
    # If imports fail, use the original implementation
    # This is a redirect only, so it will always use the function from the sentinel package
    # The original implementation would be here, but we're just providing the interface
    
    def load_bloom_with_adaptive_transformer(model_name, baseline_model, config, device, debug=False, quiet=False):
        """
        Load an adaptive transformer model initialized from a baseline BLOOM model.
        This is a stub that should use the implementation from sentinel.models.loaders.bloom_loader.
        """
        raise ImportError(
            "The bloom_loader module has been moved to sentinel.models.loaders.bloom_loader. "
            "Please update your imports to use the new module path."
        )
    
    def load_adaptive_model_bloom(model_name, baseline_model, config, device, debug=False, quiet=False):
        """
        Load an adaptive transformer model initialized from a baseline BLOOM model.
        This is a stub that should use the implementation from sentinel.models.loaders.bloom_loader.
        """
        raise ImportError(
            "The bloom_loader module has been moved to sentinel.models.loaders.bloom_loader. "
            "Please update your imports to use the new module path."
        )