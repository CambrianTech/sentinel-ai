"""
Neural Plasticity Utilities (v0.0.1 2025-04-19)

This module provides utilities for implementing neural plasticity in transformer models:
- Dynamic pruning and growth of attention heads 
- Tracking of head utility metrics
- Visualization of plasticity dynamics

These utilities enable transformer models to become more efficient through
adaptively modifying their structure during training.
"""

__version__ = "0.0.1"
__date__ = "2025-04-19"

from .core import (
    calculate_head_entropy,
    calculate_head_gradients,
    generate_pruning_mask,
    apply_pruning_mask,
    evaluate_model
)

from .visualization import (
    visualize_head_entropy,
    visualize_head_gradients,
    visualize_pruning_decisions,
    visualize_training_metrics,
    visualize_attention_patterns
)

from .training import (
    create_plasticity_trainer,
    run_plasticity_loop,
    train_with_plasticity,
    get_plasticity_optimizer
)