# Legacy module - redirects to sentinel.models.loaders
"""
DEPRECATED: This module will be removed in a future version.
Please use sentinel.models.loaders instead.
"""

import warnings

# Emit deprecation warning
warnings.warn(
    "The models.loaders module is deprecated. Please use sentinel.models.loaders instead.",
    DeprecationWarning, 
    stacklevel=2
)

# Import from the new location for backward compatibility
try:
    from sentinel.models.loaders import *
    from sentinel.models.loaders.loader import ModelLoader
    from sentinel.models.loaders.gpt2_loader import load_gpt2_with_adaptive_transformer
    from sentinel.models.loaders.bloom_loader import load_bloom_with_adaptive_transformer
    from sentinel.models.loaders.opt_loader import load_opt_with_adaptive_transformer
    from sentinel.models.loaders.llama_loader import load_llama_with_adaptive_transformer
except ImportError as e:
    if "No module named 'sentinel.models'" in str(e):
        # Allow original imports to work if sentinel package is not available
        from .loader import ModelLoader
        from .gpt2_loader import load_gpt2_with_adaptive_transformer, load_gpt2_with_sentinel_gates
        from .bloom_loader import load_bloom_with_adaptive_transformer
        from .opt_loader import load_opt_with_adaptive_transformer
        from .llama_loader import load_llama_with_adaptive_transformer
    else:
        # Re-raise other import errors
        raise