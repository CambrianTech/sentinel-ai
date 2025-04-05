"""
Adaptive Neural Plasticity System - DEPRECATED MODULE

This module is a backwards compatibility layer for the old utils.adaptive.adaptive_plasticity module.
The functionality has been moved to sentinel.utils.adaptive.adaptive_plasticity.

Please update your imports to use the new module path:
from sentinel.utils.adaptive.adaptive_plasticity import AdaptivePlasticitySystem, run_adaptive_system
"""

import warnings
import sys

# Emit deprecation warning
warnings.warn(
    "The module utils.adaptive.adaptive_plasticity is deprecated. "
    "Please use sentinel.utils.adaptive.adaptive_plasticity instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location
from sentinel.utils.adaptive.adaptive_plasticity import (
    AdaptivePlasticitySystem,
    run_adaptive_system
)

# Add all imported symbols to __all__
__all__ = ["AdaptivePlasticitySystem", "run_adaptive_system"]