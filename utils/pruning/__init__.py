"""
JAX-based pruning library for transformer models

This module provides tools for pruning and fine-tuning transformer models,
with a focus on attention head pruning and recovery through fine-tuning.
"""

from .environment import Environment
from .results_manager import ResultsManager
from .pruning_module import PruningModule
from .fine_tuner import FineTuner
from .fine_tuner_improved import ImprovedFineTuner
from .strategies import (
    PruningStrategy,
    RandomStrategy,
    MagnitudeStrategy,
    AttentionEntropyStrategy,
    get_strategy
)
from .benchmark import PruningBenchmark
from .experiment import PruningExperiment, PruningFineTuningExperiment

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