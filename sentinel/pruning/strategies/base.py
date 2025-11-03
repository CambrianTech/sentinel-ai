"""
Base class for pruning strategies.
"""
from typing import List, Tuple, Any


class PruningStrategy:
    """
    Base class for pruning strategies.
    
    This provides the interface that all pruning strategies must implement.
    """
    
    def __init__(self, pruning_module):
        """
        Initialize the pruning strategy.
        
        Args:
            pruning_module: A PruningModule instance
        """
        self.pruning_module = pruning_module
    
    def get_head_importance(self, params: Any) -> List[Tuple[int, int, float]]:
        """
        Get importance scores for all attention heads.
        
        Args:
            params: Model parameters
            
        Returns:
            List of (layer_idx, head_idx, importance_score) tuples
        """
        raise NotImplementedError("Subclasses must implement get_head_importance")