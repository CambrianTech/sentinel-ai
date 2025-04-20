"""
Pruning module for Neural Plasticity.

This module provides pruning strategies and related utilities for
neural plasticity experiments.

Version: v0.0.34 (2025-04-20 14:30:00)
"""

from .base import PruningStrategy
from .entropy import EntropyPruner
from .magnitude import MagnitudePruner