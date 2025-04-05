"""
Random pruning strategy implementation.
"""

import random
from typing import List, Tuple, Any
from sentinel.pruning.strategies.base import PruningStrategy


class RandomPruningStrategy(PruningStrategy):
    """
    Random pruning strategy.
    
    This strategy assigns random importance scores to each attention head,
    resulting in random pruning when the least important heads are selected.
    """
    
    def get_head_importance(self, params: Any) -> List[Tuple[int, int, float]]:
        """
        Assign random importance scores to heads.
        
        Args:
            params: Model parameters
            
        Returns:
            List of (layer_idx, head_idx, importance_score) tuples with random scores
        """
        all_head_importance = []
        
        for layer_idx in range(self.pruning_module.num_layers):
            for head_idx in range(self.pruning_module.num_heads):
                # Random importance score
                score = random.random()
                all_head_importance.append((layer_idx, head_idx, score))
        
        return all_head_importance