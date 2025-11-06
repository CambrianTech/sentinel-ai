# sentinel_data/__init__.py
"""
Compatibility module for HuggingFace datasets library.

This module provides compatibility for HuggingFace datasets to avoid import conflicts.
It should not be used directly in your code - use the sdata module instead.

WARNING: This is just a compatibility layer. For actual functionality, use the sdata module.
"""

import warnings

warnings.warn(
    "The sentinel_data module is a compatibility layer for HuggingFace datasets. "
    "For actual functionality, please use the sdata module instead.",
    DeprecationWarning,
    stacklevel=2
)

# Create a features module and class for compatibility
class features:
    """Stub features module"""
    class Features:
        """Stub Features class"""
        def __init__(self, *args, **kwargs):
            pass
    
    class Value:
        """Stub Value class"""
        def __init__(self, *args, **kwargs):
            pass

# Create a utils module for compatibility
class utils:
    """Stub utils module"""
    class logging:
        """Stub logging module"""
        @staticmethod
        def get_logger(name):
            """Get a logger with the given name"""
            import logging
            return logging.getLogger(name)

# Import packaged_modules
from sentinel_data.packaged_modules import *
from sentinel_data.table import table_cast