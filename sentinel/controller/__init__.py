"""
Sentinel-AI Controller Module

This module provides functionality for implementing the dynamic controller system 
that manages attention head gates, monitors head metrics, and implements the feedback loop
for adaptive transformer models.
"""

# Import key components for easy access
from sentinel.controller.controller_ann import ANNController
from sentinel.controller.controller_manager import ControllerManager
from sentinel.controller.metrics import collect_head_metrics

# For backward compatibility
try:
    from controller.controller_ann import ANNController as _DeprecatedANNController
    from controller.controller_manager import ControllerManager as _DeprecatedControllerManager
    from controller.metrics.head_metrics import collect_head_metrics as _deprecated_collect_head_metrics
except ImportError:
    # Module not available, will use the new location
    pass