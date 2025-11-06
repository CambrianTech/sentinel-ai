"""
Pruning and growth strategies for the Upgrayedd system.

This package contains different strategies for pruning and regrowing 
neural network components during the Upgrayedd optimization process.
"""

__all__ = [
    "BasePruningStrategy",
    "RandomPruningStrategy",
    "EntropyPruningStrategy", 
    "MagnitudePruningStrategy",
    "get_strategy"
]

from upgrayedd.strategies.base import BasePruningStrategy
from upgrayedd.strategies.random import RandomPruningStrategy
from upgrayedd.strategies.entropy import EntropyPruningStrategy
from upgrayedd.strategies.magnitude import MagnitudePruningStrategy

def get_strategy(strategy_name: str, **kwargs) -> BasePruningStrategy:
    """
    Factory function to get the appropriate pruning strategy.
    
    Args:
        strategy_name: Name of the strategy to use
        **kwargs: Additional arguments to pass to the strategy constructor
        
    Returns:
        An instance of the requested pruning strategy
        
    Raises:
        ValueError: If the strategy name is not recognized
    """
    strategies = {
        "random": RandomPruningStrategy,
        "entropy": EntropyPruningStrategy,
        "magnitude": MagnitudePruningStrategy,
    }
    
    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Available strategies: {', '.join(strategies.keys())}"
        )
    
    return strategies[strategy_name](**kwargs)