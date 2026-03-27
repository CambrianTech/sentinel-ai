"""
Sentinel-AI Utilities Module

Re-exports utility functions from their canonical locations.
Modules that have been migrated to sentinel/utils/ are imported directly.
Modules still in utils/ are re-exported for convenience.
"""

# --- Modules that exist in sentinel/utils/ ---
try:
    from sentinel.utils.checkpoint import save_checkpoint, load_checkpoint
except ImportError:
    pass

try:
    from sentinel.utils.metrics import (
        calculate_metrics,
        log_metrics,
        calculate_perplexity,
        calculate_diversity,
        calculate_repetition
    )
except ImportError:
    pass

try:
    from sentinel.utils.metrics_logger import MetricsLogger
except ImportError:
    pass

try:
    from sentinel.utils.head_metrics import (
        compute_attention_entropy,
        compute_head_importance,
        compute_gradient_norms,
        visualize_head_metrics,
        recommend_pruning_growth
    )
except ImportError:
    pass

# --- Modules that still live in utils/ (root) ---
try:
    from utils.model_wrapper import ModelWrapper
except ImportError:
    pass

try:
    from utils.progress_tracker import ProgressTracker
except ImportError:
    pass

try:
    from utils.generation_wrapper import GenerationWrapper
except ImportError:
    pass

try:
    from utils.training import Trainer
except ImportError:
    pass

try:
    from utils.dynamic_architecture import DynamicArchitecture
except ImportError:
    pass

try:
    from utils.head_lr_manager import HeadLRManager
except ImportError:
    pass

try:
    from sentinel.utils.adaptive import AdaptivePlasticitySystem, run_adaptive_system
except ImportError:
    pass
