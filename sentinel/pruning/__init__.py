"""
Sentinel-AI Pruning Module

This module provides functionality for pruning, measuring, and regrowth in transformer models,
implementing the full neural plasticity cycle.
"""

# Import key components for easy access
from sentinel.pruning.fixed_pruning_module import FixedPruningModule
from sentinel.pruning.fixed_pruning_module_jax import PruningModule