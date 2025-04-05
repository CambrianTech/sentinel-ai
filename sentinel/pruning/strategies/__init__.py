"""
Strategies for pruning attention heads in transformer models.
"""

from sentinel.pruning.strategies.base import PruningStrategy
from sentinel.pruning.strategies.random import RandomPruningStrategy
from sentinel.pruning.strategies.magnitude import MagnitudePruningStrategy
from sentinel.pruning.strategies.entropy import EntropyPruningStrategy

def get_strategy(strategy_name, pruning_module):
    """
    Get a pruning strategy by name.
    
    Args:
        strategy_name: Name of the strategy (random, magnitude, entropy)
        pruning_module: PruningModule instance
        
    Returns:
        A pruning strategy instance
    """
    strategies = {
        "random": RandomPruningStrategy,
        "magnitude": MagnitudePruningStrategy,
        "entropy": EntropyPruningStrategy
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown pruning strategy: {strategy_name}")
    
    return strategies[strategy_name](pruning_module)