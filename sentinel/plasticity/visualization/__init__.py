"""
Visualization components for Neural Plasticity.

This module provides visualization utilities for the neural plasticity system
that are completely decoupled from the core system. This separation ensures
that the core neural plasticity system can run in production environments 
without any visualization dependencies.

Version: v0.0.34 (2025-04-20 14:30:00)
"""

# Re-export key visualization components
from .metrics_visualizer import MetricsVisualizer
from .entropy_visualizer import EntropyVisualizer
from .pruning_visualizer import PruningVisualizer
from .head_recovery_visualizer import HeadRecoveryVisualizer