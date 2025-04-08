"""
Pruning strategies for transformer models.

This package provides implementations of different pruning strategies for
transformer models, including entropy-based, magnitude-based, and random pruning.
"""

from .entropy import entropy_based_pruning
from .magnitude import magnitude_based_pruning
from .random import random_pruning

__all__ = [
    "entropy_based_pruning",
    "magnitude_based_pruning",
    "random_pruning"
]