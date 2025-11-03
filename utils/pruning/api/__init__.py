"""
Pruning API for transformer models.

This package provides a simple API for pruning and fine-tuning transformer models.
"""

# Import key functions for easier access
from .pruning import compute_head_importance, prune_heads, fine_tune, evaluate_model
from .data import load_wikitext, prepare_data, prepare_test_data