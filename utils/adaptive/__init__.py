"""
Adaptive neural plasticity utilities.

This module provides tools for implementing neural plasticity in transformer models,
allowing them to dynamically adjust their structure through cycles of
pruning, measuring, growing, and learning.
"""

from utils.adaptive.adaptive_plasticity import (
    AdaptivePlasticitySystem,
    run_adaptive_system
)

__all__ = [
    'AdaptivePlasticitySystem',
    'run_adaptive_system'
]