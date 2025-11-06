"""
Plasticity Controller Package

This package contains controllers for neural plasticity, including
reinforcement learning controllers that learn to optimize pruning
and plasticity decisions.
"""

from sentinel.plasticity.controller.rl_controller import (
    RLController,
    RLControllerConfig,
    create_controller
)

__all__ = [
    'RLController',
    'RLControllerConfig',
    'create_controller',
]