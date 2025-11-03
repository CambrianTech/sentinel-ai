"""
Neural Plasticity Dashboard Package

This package provides dashboard visualization and reporting for neural
plasticity experiments, including real-time metrics tracking, pruning
visualizations, and integration with Weights & Biases.

Version: v0.0.3 (2025-04-20 25:50:00)
"""

# Import key components for easier access
from .reporter import DashboardReporter
from .dashboard import LiveDashboard, DashboardServer

# Attempt to import wandb integration (allowing graceful fallback if wandb is not installed)
try:
    from .wandb_integration import WandbDashboard, setup_wandb_in_colab
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Attempt to import Colab integration (will fail gracefully if not in Colab)
try:
    from .colab_integration import (
        setup_colab_environment,
        create_experiment_dashboard_cell,
        get_experiment_parameters_cell,
        get_run_experiment_cell,
        create_colab_quickstart_notebook,
        monitor_experiment_in_colab
    )
    COLAB_INTEGRATION_AVAILABLE = True
except ImportError:
    COLAB_INTEGRATION_AVAILABLE = False

# Constants
DASHBOARD_VERSION = "0.0.3"

__all__ = [
    'DashboardReporter',
    'LiveDashboard',
    'DashboardServer',
    'DASHBOARD_VERSION',
    'WANDB_AVAILABLE',
    'COLAB_INTEGRATION_AVAILABLE'
]

if WANDB_AVAILABLE:
    __all__.extend(['WandbDashboard', 'setup_wandb_in_colab'])

if COLAB_INTEGRATION_AVAILABLE:
    __all__.extend([
        'setup_colab_environment',
        'create_experiment_dashboard_cell',
        'get_experiment_parameters_cell',
        'get_run_experiment_cell',
        'create_colab_quickstart_notebook',
        'monitor_experiment_in_colab'
    ])