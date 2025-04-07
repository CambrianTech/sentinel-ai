"""
Upgrayedd - Adaptive optimization for transformer models.

This package contains tools for pruning, regrowth, and adaptive optimization
of transformer models with a focus on enhancing model performance while reducing size.
"""

from .optimizer.adaptive_optimizer import AdaptiveOptimizer

__version__ = "0.1.0"
__all__ = ["AdaptiveOptimizer"]