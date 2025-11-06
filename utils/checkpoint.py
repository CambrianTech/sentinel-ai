# utils/checkpoint.py - DEPRECATED MODULE

"""
Checkpoint utilities - DEPRECATED MODULE

This module is a backwards compatibility layer for the old utils.checkpoint module.
The functionality has been moved to sentinel.utils.checkpoint.

Please update your imports to use the new module path.
"""

import warnings

# Emit deprecation warning
warnings.warn(
    "The module utils.checkpoint is deprecated. "
    "Please use sentinel.utils.checkpoint instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location
from sentinel.utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    get_latest_checkpoint
)

# Add all imported symbols to __all__
__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "get_latest_checkpoint"
]