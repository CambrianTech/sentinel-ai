"""
Neural Plasticity Framework

A modular, object-oriented framework for running neural plasticity experiments
with transformer models.

The framework provides:
- Core experiment abstractions
- Pruning strategy implementations
- Visualization components
- Metrics tracking and analysis
- Configuration management

All experiment outputs are stored in a standardized format in the /output directory.
"""

from sentinel.neural_plasticity.core.base_experiment import BaseExperiment
from sentinel.neural_plasticity.experiment.neural_plasticity_experiment import NeuralPlasticityExperiment

__version__ = "0.1.0 (2025-04-20 16:30:00)"