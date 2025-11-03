"""
Utility functions for calculating attention head metrics - DEPRECATED MODULE

This module is a backwards compatibility layer for the old utils.head_metrics module.
The functionality has been moved to sentinel.utils.head_metrics.

Please update your imports to use the new module path.
"""

import warnings

# Emit deprecation warning
warnings.warn(
    "The module utils.head_metrics is deprecated. "
    "Please use sentinel.utils.head_metrics instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from the new location
from sentinel.utils.head_metrics import (
    compute_attention_entropy,
    compute_head_importance,
    compute_gradient_norms,
    visualize_head_metrics,
    plot_heatmap,
    analyze_head_clustering,
    visualize_head_similarities,
    recommend_pruning_growth
)

# Add all imported symbols to __all__
__all__ = [
    "compute_attention_entropy",
    "compute_head_importance",
    "compute_gradient_norms",
    "visualize_head_metrics",
    "plot_heatmap",
    "analyze_head_clustering",
    "visualize_head_similarities",
    "recommend_pruning_growth"
]