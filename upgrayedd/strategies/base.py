"""
Base class for all pruning strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Any, Optional

import torch
import numpy as np


class BasePruningStrategy(ABC):
    """
    Abstract base class for pruning strategies.
    
    All pruning strategies should derive from this class and implement
    the required methods for selecting which heads to prune and which
    to regrow.
    """
    
    def __init__(
        self,
        pruning_ratio: float = 0.3,
        growth_ratio: float = 0.1,
        min_heads: int = 1,
        seed: Optional[int] = None
    ):
        """
        Initialize the strategy.
        
        Args:
            pruning_ratio: Fraction of heads to prune (0.0 to 1.0)
            growth_ratio: Fraction of pruned heads to regrow (0.0 to 1.0)
            min_heads: Minimum number of heads to keep in each layer
            seed: Random seed for reproducibility
        """
        if not 0 <= pruning_ratio < 1:
            raise ValueError("pruning_ratio must be between 0 and 1")
        if not 0 <= growth_ratio <= 1:
            raise ValueError("growth_ratio must be between 0 and 1")
        if min_heads < 1:
            raise ValueError("min_heads must be at least 1")
            
        self.pruning_ratio = pruning_ratio
        self.growth_ratio = growth_ratio
        self.min_heads = min_heads
        
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            torch.manual_seed(seed)
        else:
            self.rng = np.random.RandomState()
    
    @abstractmethod
    def select_heads_to_prune(
        self, 
        model: Any,
        head_importances: Dict[str, torch.Tensor]
    ) -> Dict[str, List[int]]:
        """
        Select which attention heads to prune.
        
        Args:
            model: The transformer model to prune
            head_importances: Dict mapping layer names to head importance scores
            
        Returns:
            Dictionary mapping layer names to lists of head indices to prune
        """
        pass
    
    @abstractmethod
    def select_heads_to_grow(
        self,
        model: Any,
        pruned_heads: Dict[str, List[int]],
        metrics: Dict[str, float]
    ) -> Dict[str, int]:
        """
        Select where to grow new attention heads.
        
        Args:
            model: The transformer model
            pruned_heads: Dict mapping layer names to lists of currently pruned heads
            metrics: Dict of performance metrics for the current model state
            
        Returns:
            Dictionary mapping layer names to number of heads to grow in that layer
        """
        pass
    
    def calculate_heads_per_layer(
        self, 
        model: Any
    ) -> Dict[str, int]:
        """
        Calculate the number of attention heads in each layer.
        
        Args:
            model: The transformer model
            
        Returns:
            Dictionary mapping layer names to number of heads
        """
        # This is a generic implementation that should be overridden
        # for specific model architectures if needed
        heads_per_layer = {}
        
        # Iterate through named modules to find attention layers
        for name, module in model.named_modules():
            if hasattr(module, "num_heads"):
                heads_per_layer[name] = module.num_heads
            elif hasattr(module, "num_attention_heads"):
                heads_per_layer[name] = module.num_attention_heads
                
        return heads_per_layer