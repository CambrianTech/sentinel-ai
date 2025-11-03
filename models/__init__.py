# Legacy module - redirects to sentinel.models
"""
DEPRECATED: This module will be removed in a future version.
Please use sentinel.models instead.
"""

import warnings

# Emit deprecation warning
warnings.warn(
    "The models module is deprecated. Please use sentinel.models instead.",
    DeprecationWarning, 
    stacklevel=2
)

# Import from the new location for backward compatibility
try:
    from sentinel.models import *
    from sentinel.models.adaptive_transformer import AdaptiveCausalLmWrapper
    from sentinel.models.agency_specialization import AgencySpecialization
    from sentinel.models.specialization_registry import SpecializationRegistry
    from sentinel.models.unet_transformer import load_unet_enhanced_model
    from sentinel.models.unet_transformer_optimized import load_unet_enhanced_model_optimized
except ImportError as e:
    if "No module named 'sentinel.models'" in str(e):
        # Allow original imports to work if sentinel package is not available
        from .adaptive_transformer import AdaptiveCausalLmWrapper
        from .agency_specialization import AgencySpecialization
        from .specialization_registry import SpecializationRegistry
        from .unet_transformer import load_unet_enhanced_model
        from .unet_transformer_optimized import load_unet_enhanced_model_optimized
    else:
        # Re-raise other import errors
        raise