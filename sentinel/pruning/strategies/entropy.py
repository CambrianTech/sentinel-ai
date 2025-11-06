"""
Entropy-based pruning strategy.
"""
from typing import List, Tuple, Any, Optional
from sentinel.pruning.strategies.base import PruningStrategy
from sentinel.pruning.strategies.magnitude import MagnitudePruningStrategy


class EntropyPruningStrategy(PruningStrategy):
    """
    Entropy-based pruning strategy.
    
    This strategy evaluates the importance of attention heads based on the entropy
    of their attention patterns. Lower entropy (more focused attention) is generally
    considered more important.
    
    For compatibility across models, this implementation currently uses a magnitude-based
    fallback, but could be extended with actual entropy calculations.
    """
    
    def __init__(self, pruning_module, sample_text: Optional[List[str]] = None):
        """
        Initialize the entropy pruning strategy.
        
        Args:
            pruning_module: A PruningModule instance
            sample_text: Sample texts to use for evaluating attention entropy
        """
        super().__init__(pruning_module)
        
        # Sample text for evaluating attention entropy
        if sample_text is None:
            self.sample_text = [
                "The quick brown fox jumps over the lazy dog",
                "Artificial intelligence is transforming the world",
                "Machine learning models can process large amounts of data",
                "The future of technology depends on sustainable practices",
                "Researchers are working on new methods to improve efficiency"
            ]
        else:
            self.sample_text = sample_text if isinstance(sample_text, list) else [sample_text]
    
    def get_head_importance(self, params: Any) -> List[Tuple[int, int, float]]:
        """
        Calculate head importance based on attention entropy.
        
        Currently falls back to magnitude-based importance for compatibility.
        
        Args:
            params: Model parameters
            
        Returns:
            List of (layer_idx, head_idx, importance_score) tuples
        """
        # For simplicity and compatibility across model types,
        # we'll just use magnitude-based pruning as a proxy
        magnitude_strategy = MagnitudePruningStrategy(self.pruning_module)
        return magnitude_strategy.get_head_importance(params)