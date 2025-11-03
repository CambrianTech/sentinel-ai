"""
Visualization utilities for Sentinel-AI.

This module provides visualization functions for attention patterns,
pruning analysis, and neural plasticity experiments.
"""

from .heatmaps import (
    plot_entropy_heatmap,
    plot_entropy_deltas_heatmap,
    plot_attention_pattern,
    plot_gate_activity,
    plot_regrowth_heatmap
)

__all__ = [
    'plot_entropy_heatmap',
    'plot_entropy_deltas_heatmap',
    'plot_attention_pattern',
    'plot_gate_activity',
    'plot_regrowth_heatmap'
]