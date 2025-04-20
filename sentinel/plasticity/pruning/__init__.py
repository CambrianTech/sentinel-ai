"""
Pruning Strategy Package for Neural Plasticity

This package provides implementations of various pruning strategies for
attention heads in transformer models.

Version: v0.0.34 (2025-04-20 17:00:00)
"""

from .base import PruningStrategy

# Import specific pruning strategies
from sentinel.plasticity.pruning.entropy import EntropyPruner
from sentinel.plasticity.pruning.magnitude import MagnitudePruner