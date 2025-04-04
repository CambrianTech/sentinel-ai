"""
DEPRECATED: This module has moved to sentinel.data.loaders.dataset_loader
This import stub will be removed in a future version.
"""
import warnings
warnings.warn(
    "Importing from sentinel_data.dataset_loader is deprecated. "
    "Use sentinel.data.loaders.dataset_loader instead.",
    DeprecationWarning, 
    stacklevel=2
)

from sentinel.data.loaders.dataset_loader import *
