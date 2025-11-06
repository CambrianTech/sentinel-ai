"""
Sentinel-AI pruning module.

This module contains the implementation of various pruning strategies and pruning-related utilities.
"""

from .entropy_magnitude import (
    compute_attention_entropy,
    collect_attention_distributions,
    entropy_based_pruning,
    magnitude_based_pruning,
    update_mask,
)

__all__ = [
    'compute_attention_entropy',
    'collect_attention_distributions',
    'entropy_based_pruning',
    'magnitude_based_pruning',
    'update_mask',
]