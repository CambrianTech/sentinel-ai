# sdata/__init__.py
"""
Sentinel AI Data Module (renamed from sdata to avoid import conflicts)

This module provides data loading and processing functionality for Sentinel AI.
"""

from sdata.dataset_loader import load_dataset, prepare_dataset
from sdata.eval import evaluate_model, calculate_perplexity

__all__ = [
    'load_dataset',
    'prepare_dataset',
    'evaluate_model',
    'calculate_perplexity'
]