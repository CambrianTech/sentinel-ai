"""
Adaptive Neural Plasticity System

This module implements a self-improving neural network architecture
that uses degeneration detection as a feedback mechanism to guide
pruning, growth, and learning cycles.
"""

from .adaptive_plasticity import AdaptivePlasticitySystem, run_adaptive_system

__all__ = ["AdaptivePlasticitySystem", "run_adaptive_system"]