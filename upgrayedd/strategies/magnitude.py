"""
Magnitude-based pruning strategy implementation.

This strategy selects heads based on the magnitude of their weights,
assuming that heads with smaller weight norms are less important.
"""

from typing import Dict, List, Any, Tuple

import torch
import numpy as np

from upgrayedd.strategies.base import BasePruningStrategy


class MagnitudePruningStrategy(BasePruningStrategy):
    """
    Strategy that prunes heads based on weight magnitudes.
    
    This strategy assumes that heads with smaller weight norms
    contribute less to the model's performance and can be pruned.
    
    For head growth, it focuses on layers with the highest average
    magnitude of remaining heads, which likely benefit from additional capacity.
    """
    
    def __init__(
        self,
        pruning_ratio: float = 0.3,
        growth_ratio: float = 0.1,
        min_heads: int = 1,
        seed: int = None,
        weight_threshold: float = 0.01,
        use_cached_magnitudes: bool = True
    ):
        """
        Initialize magnitude-based pruning strategy.
        
        Args:
            pruning_ratio: Fraction of heads to prune (0.0 to 1.0)
            growth_ratio: Fraction of pruned heads to regrow (0.0 to 1.0) 
            min_heads: Minimum number of heads to keep in each layer
            seed: Random seed for reproducibility
            weight_threshold: Threshold below which heads are considered unimportant
            use_cached_magnitudes: Whether to reuse magnitude values if provided
        """
        super().__init__(pruning_ratio, growth_ratio, min_heads, seed)
        self.weight_threshold = weight_threshold
        self.use_cached_magnitudes = use_cached_magnitudes
        self.cached_magnitudes = {}
        
    def calculate_head_magnitudes(
        self,
        model: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the magnitudes of attention head weights.
        
        Args:
            model: The transformer model
            
        Returns:
            Dictionary mapping layer names to tensors of magnitude values per head
        """
        # If we have cached magnitudes and are allowed to use them, do so
        if self.use_cached_magnitudes and self.cached_magnitudes:
            return self.cached_magnitudes
            
        magnitudes = {}
        
        # This implementation will vary based on model architecture
        # For simplicity, we'll implement a generic approach:
        
        # 1. Identify attention modules
        heads_per_layer = self.calculate_heads_per_layer(model)
        
        for name, module in model.named_modules():
            # Skip if not an attention layer
            if name not in heads_per_layer:
                continue
                
            num_heads = heads_per_layer[name]
            head_mags = []
            
            # 2. Extract query, key, value weights
            # Note: This is a simplified approach - real implementation would need
            # to handle different model architectures (e.g., GPT-2, BLOOM, etc.)
            for param_name, param in module.named_parameters():
                if any(x in param_name for x in ["query", "key", "value"]):
                    # Check if weights are already separated by head
                    if param.dim() >= 3 and param.size(0) == num_heads:
                        # If weights are [num_heads, ...], compute norm per head
                        head_norms = torch.norm(param.view(num_heads, -1), dim=1)
                        head_mags.append(head_norms)
                    elif param.dim() >= 2:
                        # If weights are [out_dim, in_dim], assume out_dim = num_heads * head_dim
                        # Reshape to [num_heads, head_dim, in_dim]
                        head_dim = param.size(0) // num_heads
                        reshaped = param.view(num_heads, head_dim, -1)
                        head_norms = torch.norm(reshaped.view(num_heads, -1), dim=1)
                        head_mags.append(head_norms)
            
            # 3. Combine Q, K, V magnitudes if available
            if head_mags:
                # Average the magnitudes across Q, K, V
                combined_mags = torch.stack(head_mags).mean(dim=0)
                magnitudes[name] = combined_mags
            else:
                # Fallback: create placeholder magnitudes
                # In practice, this would need a model-specific implementation
                place_holder = torch.ones(num_heads) * 0.5
                # Add some randomness
                noise = torch.randn(num_heads) * 0.1
                magnitudes[name] = (place_holder + noise).abs()
                
        # Cache the computed magnitudes
        self.cached_magnitudes = magnitudes
        return magnitudes
    
    def select_heads_to_prune(
        self, 
        model: Any,
        head_importances: Dict[str, torch.Tensor]
    ) -> Dict[str, List[int]]:
        """
        Select heads to prune based on weight magnitudes.
        
        Args:
            model: The transformer model to prune
            head_importances: Dict mapping layer names to head importance scores
                              (can contain magnitude values if pre-computed)
            
        Returns:
            Dictionary mapping layer names to lists of head indices to prune
        """
        # Use provided head importances if they contain magnitude information,
        # otherwise calculate magnitudes
        if not head_importances or self.use_cached_magnitudes is False:
            magnitude_values = self.calculate_head_magnitudes(model)
        else:
            magnitude_values = head_importances
            
        heads_to_prune = {}
        heads_per_layer = self.calculate_heads_per_layer(model)
        
        for layer_name, num_heads in heads_per_layer.items():
            if layer_name not in magnitude_values:
                continue
                
            # Calculate number of heads to prune, ensuring we keep min_heads
            num_to_prune = max(0, int(num_heads * self.pruning_ratio))
            num_to_prune = min(num_to_prune, num_heads - self.min_heads)
            
            if num_to_prune > 0:
                # Get magnitude values for this layer
                layer_magnitudes = magnitude_values[layer_name]
                
                # Select heads with lowest magnitudes
                values, indices = torch.topk(
                    layer_magnitudes, 
                    k=num_to_prune, 
                    largest=False
                )
                
                # Only prune heads below the threshold
                prune_indices = []
                for i, value in enumerate(values):
                    if value < self.weight_threshold:
                        prune_indices.append(indices[i].item())
                
                # If we don't have enough heads below the threshold,
                # take the smallest ones regardless
                if len(prune_indices) < num_to_prune:
                    prune_indices = indices.tolist()
                
                heads_to_prune[layer_name] = prune_indices
        
        return heads_to_prune
    
    def select_heads_to_grow(
        self,
        model: Any,
        pruned_heads: Dict[str, List[int]],
        metrics: Dict[str, float]
    ) -> Dict[str, int]:
        """
        Select layers for head growth based on weight magnitudes.
        
        Prioritizes layers with high average magnitude in remaining heads,
        as these layers are actively learning important features.
        
        Args:
            model: The transformer model
            pruned_heads: Dict mapping layer names to lists of currently pruned heads
            metrics: Dict of performance metrics
            
        Returns:
            Dictionary mapping layer names to number of heads to grow in that layer
        """
        # Filter layers with pruned heads
        layers_with_pruned_heads = {
            layer: heads 
            for layer, heads in pruned_heads.items() 
            if heads
        }
        
        if not layers_with_pruned_heads:
            return {}
            
        # Calculate total number of heads to grow
        total_pruned = sum(len(heads) for heads in layers_with_pruned_heads.values())
        total_to_grow = int(total_pruned * self.growth_ratio)
        
        if total_to_grow == 0:
            return {}
            
        # Get magnitude values
        magnitude_values = self.cached_magnitudes
        if not magnitude_values:
            magnitude_values = self.calculate_head_magnitudes(model)
            
        # Calculate average magnitude of remaining heads for each layer
        layer_avg_magnitude = {}
        heads_per_layer = self.calculate_heads_per_layer(model)
        
        for layer_name, num_heads in heads_per_layer.items():
            if layer_name not in layers_with_pruned_heads:
                continue
                
            if layer_name not in magnitude_values:
                continue
                
            # Get magnitude values for this layer
            layer_magnitude = magnitude_values[layer_name]
            
            # Create mask for non-pruned heads
            mask = torch.ones(num_heads, dtype=torch.bool)
            for idx in layers_with_pruned_heads[layer_name]:
                if idx < num_heads:
                    mask[idx] = False
                    
            # Calculate average magnitude of remaining heads
            if mask.sum() > 0:
                avg_magnitude = layer_magnitude[mask].mean().item()
                layer_avg_magnitude[layer_name] = avg_magnitude
        
        # Prioritize layers with highest average magnitude for growth
        heads_to_grow = {}
        
        if layer_avg_magnitude:
            # Sort layers by decreasing average magnitude
            sorted_layers = sorted(
                layer_avg_magnitude.items(), 
                key=lambda x: x[1],
                reverse=True
            )
            
            # Allocate heads to grow, more to layers with higher magnitude
            remaining = total_to_grow
            for layer_name, _ in sorted_layers:
                # Determine how many heads to grow in this layer
                # More heads to layers with higher magnitude (earlier in list)
                growth_share = max(1, remaining // (len(sorted_layers) // 2 + 1))
                growth_share = min(growth_share, len(layers_with_pruned_heads[layer_name]))
                
                heads_to_grow[layer_name] = growth_share
                remaining -= growth_share
                
                if remaining <= 0:
                    break
        
        return heads_to_grow