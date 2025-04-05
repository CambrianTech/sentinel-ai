"""
Adaptive Transformer Implementation

This module provides the implementation of the AdaptiveCausalLmWrapper, 
which wraps transformer models with adaptive capabilities like per-head gating.

For backward compatibility, this initially imports from the legacy code path.
This will be replaced with a native implementation in the future.
"""

import warnings

# Import from legacy location for now, will be replaced with native implementation
try:
    from models.adaptive_transformer import AdaptiveCausalLmWrapper
    
    warnings.warn(
        "Using legacy import path for AdaptiveCausalLmWrapper. "
        "This will be replaced with a native implementation in sentinel.models.adaptive.transformer.",
        DeprecationWarning,
        stacklevel=2
    )
    
except ImportError:
    # Fallback implementation - will be replaced with an actual implementation
    # that doesn't depend on the legacy code
    import sys
    
    class AdaptiveCausalLmWrapper:
        """
        Placeholder implementation for AdaptiveCausalLmWrapper.
        
        This class is a stub that provides clear error messages when the original implementation
        cannot be imported. The actual implementation will be added in a future release.
        
        IMPORTANT: Currently, you must have the original models.adaptive_transformer module 
        available in your Python path. This stub exists only to provide clear error messages
        and documentation.
        """
        def __init__(self, config, token_embeddings, position_embeddings, debug=False):
            message = (
                "ERROR: Cannot initialize AdaptiveCausalLmWrapper.\n\n"
                "The sentinel.models.adaptive.transformer module is currently a compatibility layer "
                "that requires the original models.adaptive_transformer module.\n\n"
                "To resolve this issue:\n"
                "1. Ensure the original module is in your Python path\n"
                "2. Import from models.adaptive_transformer directly until the migration is complete\n\n"
                "This stub will be replaced with a full implementation in a future release."
            )
            print(message, file=sys.stderr)
            raise ImportError(message)
