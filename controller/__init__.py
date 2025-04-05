# controller/__init__.py
"""
Deprecated: This module is being migrated to sentinel.controller

This module will continue to work for backward compatibility,
but new code should use sentinel.controller directly.
"""

import warnings

# Emit deprecation warning
warnings.warn(
    "The controller module is deprecated. Please use sentinel.controller instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location to maintain backward compatibility
try:
    from sentinel.controller.controller_ann import ANNController
    from sentinel.controller.controller_manager import ControllerManager
    from sentinel.controller.metrics.head_metrics import collect_head_metrics
except ImportError:
    # If sentinel package is not available, fall back to local imports
    from .controller_ann import ANNController
    from .controller_manager import ControllerManager
    from .metrics.head_metrics import collect_head_metrics