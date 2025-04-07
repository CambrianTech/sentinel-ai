"""
JAX-based pruning library for transformer models

This module provides tools for pruning and fine-tuning transformer models,
with a focus on attention head pruning and recovery through fine-tuning.
"""

from .environment import Environment
from .results_manager import ResultsManager
from .pruning_module import PruningModule
# Original fine tuner implementations - using one or both depending on what's available
# Skip the consolidated version to avoid datasets circular import
try:
    # Use original implementations
    from .fine_tuner import FineTuner
    from .fine_tuner_improved import ImprovedFineTuner
except ImportError:
    # Define placeholders if imports fail
    class FineTuner:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("FineTuner not available")
    
    class ImprovedFineTuner:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("ImprovedFineTuner not available")
from .strategies import (
    PruningStrategy,
    RandomStrategy,
    MagnitudeStrategy,
    AttentionEntropyStrategy,
    get_strategy
)
from .benchmark import PruningBenchmark
from .experiment import PruningExperiment, PruningFineTuningExperiment

# Import visualization module if available
try:
    from .visualization import (
        plot_experiment_summary,
        plot_strategy_comparison,
        plot_recovery_comparison,
        visualize_head_importance
    )
    has_visualization = True
except ImportError:
    has_visualization = False

# Export all public interfaces
__all__ = [
    'Environment',
    'ResultsManager',
    'PruningModule',
    'FineTuner',
    'ImprovedFineTuner',
    'PruningStrategy',
    'RandomStrategy',
    'MagnitudeStrategy',
    'AttentionEntropyStrategy',
    'get_strategy',
    'PruningBenchmark',
    'PruningExperiment',
    'PruningFineTuningExperiment',
]

# Add visualization functions if available
if has_visualization:
    __all__.extend([
        'plot_experiment_summary',
        'plot_strategy_comparison',
        'plot_recovery_comparison',
        'visualize_head_importance'
    ])