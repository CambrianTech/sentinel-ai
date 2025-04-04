"""
Legacy sentinel_data module 

This module provides backward compatibility with the old import structure.
In the new structure, this is moved to sentinel.data.
"""

import warnings

# Direct users to the new import location
warnings.warn(
    "Importing from sentinel_data is deprecated. "
    "Use sentinel.data instead.",
    DeprecationWarning,
    stacklevel=2
)

# Do not import from sentinel.data to avoid circular imports
# Instead, define stubs that forward to the new location

def load_dataset(*args, **kwargs):
    from sentinel.data.loaders.dataset_loader import load_dataset as _load_dataset
    return _load_dataset(*args, **kwargs)

def load_dataset_for_testing(*args, **kwargs):
    from sentinel.data.loaders.dataset_loader import load_dataset_for_testing as _load_dataset_for_testing
    return _load_dataset_for_testing(*args, **kwargs)

def create_dataloader(*args, **kwargs):
    from sentinel.data.loaders.dataset_loader import create_dataloader as _create_dataloader
    return _create_dataloader(*args, **kwargs)

__all__ = [
    "load_dataset",
    "load_dataset_for_testing",
    "create_dataloader"
]