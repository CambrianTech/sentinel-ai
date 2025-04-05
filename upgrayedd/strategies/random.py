"""
Random pruning strategy implementation.

This strategy randomly selects heads to prune and grow without 
considering importance or performance metrics.
"""

from typing import Dict, List, Any

import torch
import numpy as np

from upgrayedd.strategies.base import BasePruningStrategy


class RandomPruningStrategy(BasePruningStrategy):
    """
    Strategy that randomly selects heads to prune and grow.
    
    This serves as a baseline for comparison with more sophisticated
    pruning strategies.
    """
    
    def select_heads_to_prune(
        self, 
        model: Any,
        head_importances: Dict[str, torch.Tensor]
    ) -> Dict[str, List[int]]:
        """
        Randomly select heads to prune.
        
        Args:
            model: The transformer model to prune
            head_importances: Dict mapping layer names to head importance scores
                              (unused in random strategy)
            
        Returns:
            Dictionary mapping layer names to lists of head indices to prune
        """
        heads_per_layer = self.calculate_heads_per_layer(model)
        heads_to_prune = {}
        
        for layer_name, num_heads in heads_per_layer.items():
            # Calculate number of heads to prune, ensuring we keep min_heads
            num_to_prune = max(0, int(num_heads * self.pruning_ratio))
            num_to_prune = min(num_to_prune, num_heads - self.min_heads)
            
            if num_to_prune > 0:
                # Randomly select indices to prune
                prune_indices = self.rng.choice(
                    num_heads, 
                    size=num_to_prune, 
                    replace=False
                ).tolist()
                heads_to_prune[layer_name] = prune_indices
        
        return heads_to_prune
    
    def select_heads_to_grow(
        self,
        model: Any,
        pruned_heads: Dict[str, List[int]],
        metrics: Dict[str, float]
    ) -> Dict[str, int]:
        """
        Randomly select layers where heads should be grown.
        
        Args:
            model: The transformer model
            pruned_heads: Dict mapping layer names to lists of currently pruned heads
            metrics: Dict of performance metrics (unused in random strategy)
            
        Returns:
            Dictionary mapping layer names to number of heads to grow in that layer
        """
        heads_to_grow = {}
        
        # Filter layers with pruned heads
        layers_with_pruned_heads = {
            layer: len(heads) 
            for layer, heads in pruned_heads.items() 
            if heads
        }
        
        if not layers_with_pruned_heads:
            return {}
            
        # Calculate total number of heads to grow
        total_pruned = sum(layers_with_pruned_heads.values())
        total_to_grow = int(total_pruned * self.growth_ratio)
        
        if total_to_grow == 0:
            return {}
            
        # Randomly distribute heads to grow among layers
        layer_names = list(layers_with_pruned_heads.keys())
        
        # Weights based on how many heads were pruned in each layer
        weights = np.array([layers_with_pruned_heads[layer] for layer in layer_names])
        weights = weights / weights.sum()
        
        # Randomly assign growth to layers based on weights
        for _ in range(total_to_grow):
            layer = self.rng.choice(layer_names, p=weights)
            heads_to_grow[layer] = heads_to_grow.get(layer, 0) + 1
            
            # Ensure we don't grow more heads than we pruned in a layer
            if heads_to_grow[layer] >= layers_with_pruned_heads[layer]:
                # Remove this layer from consideration for further growth
                idx = layer_names.index(layer)
                layer_names.pop(idx)
                weights = np.delete(weights, idx)
                if len(layer_names) > 0:
                    weights = weights / weights.sum()
                else:
                    break
        
        return heads_to_grow