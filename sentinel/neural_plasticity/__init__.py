"""
Neural Plasticity Framework

A modular, object-oriented framework for running neural plasticity experiments
with transformer models.
"""

# Re-export from actual locations
try:
    from utils.neural_plasticity.experiment import NeuralPlasticityExperiment
except ImportError:
    pass

__version__ = "0.1.0"
