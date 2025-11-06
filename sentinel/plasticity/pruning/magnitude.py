"""
Magnitude-based Pruning Strategy

This module provides an implementation of magnitude-based pruning for
attention heads in transformer models.

Version: v0.0.34 (2025-04-20 17:00:00)
"""

import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm

from .base import PruningStrategy

logger = logging.getLogger(__name__)

class MagnitudePruner(PruningStrategy):
    """
    Prunes attention heads based on weight magnitudes.
    
    This class implements magnitude-based pruning by:
    1. Calculating the L2 norm of attention weights
    2. Ranking heads by their weight magnitudes
    3. Pruning heads with smallest weights
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the magnitude pruner.
        
        Args:
            device: Device to run on
        """
        super().__init__(device)
        
    def collect_distributions(
        self,
        model,
        dataloader
    ) -> Dict[int, torch.Tensor]:
        """
        Collect weight magnitude distributions.
        
        Args:
            model: The model to analyze
            dataloader: DataLoader for input data (unused for magnitude pruning)
            
        Returns:
            Dictionary mapping layer indices to magnitude values
        """
        logger.info("Collecting weight magnitudes")
        
        # Detect model structure
        num_layers, num_heads = self.detect_model_structure(model)
        
        # Store magnitudes for each layer and head
        # Magnitude tensor shape: [num_layers, num_heads]
        magnitudes = torch.zeros((num_layers, num_heads), device=self.device)
        
        # Flag to track if we found any weights
        found_weights = False
        
        # Check model parameters
        for name, param in model.named_parameters():
            # Look for query, key, value weights in attention modules
            if 'query' in name or 'key' in name or 'value' in name:
                # Try to extract layer index from parameter name
                layer_idx = None
                for part in name.split('.'):
                    if 'layer' in part and '_' not in part:
                        # Try to find a number after "layer"
                        for i in range(len(part)):
                            if part[i:].isdigit():
                                layer_idx = int(part[i:])
                                break
                    elif part.isdigit():
                        # Try to parse standalone digit as layer index
                        layer_idx = int(part)
                        
                if layer_idx is None or layer_idx >= num_layers:
                    continue
                    
                # Extract head dimension from parameter shape
                if len(param.shape) >= 2:
                    # Different models organize heads differently
                    # Try to infer head dimension
                    head_dim = None
                    
                    # Common case: weights are organized by heads
                    if param.shape[0] == num_heads:
                        head_dim = 0
                    elif param.shape[0] % num_heads == 0:
                        # Weights might be flattened across heads
                        head_dim = 0
                        head_size = param.shape[0] // num_heads
                        
                        # Calculate magnitude for each head
                        for head_idx in range(num_heads):
                            start_idx = head_idx * head_size
                            end_idx = (head_idx + 1) * head_size
                            head_weights = param[start_idx:end_idx]
                            
                            # Calculate L2 norm
                            magnitude = torch.norm(head_weights)
                            magnitudes[layer_idx, head_idx] += magnitude.item()
                            
                        found_weights = True
                        continue
                        
                    # Try other dimensions if head_dim is still None
                    if head_dim is None:
                        # Skip this parameter
                        continue
                        
                    # Calculate magnitude for each head
                    for head_idx in range(min(param.shape[head_dim], num_heads)):
                        if head_dim == 0:
                            head_weights = param[head_idx]
                        elif head_dim == 1:
                            head_weights = param[:, head_idx]
                        else:
                            continue
                            
                        # Calculate L2 norm
                        magnitude = torch.norm(head_weights)
                        magnitudes[layer_idx, head_idx] += magnitude.item()
                        
                    found_weights = True
        
        # If we couldn't find weights using parameter names, try module inspection
        if not found_weights:
            for name, module in model.named_modules():
                if 'attention' in name.lower():
                    # Extract layer index from module name
                    layer_idx = None
                    for part in name.split('.'):
                        if 'layer' in part and '_' not in part:
                            # Try to find a number after "layer"
                            for i in range(len(part)):
                                if part[i:].isdigit():
                                    layer_idx = int(part[i:])
                                    break
                        elif part.isdigit():
                            # Try to parse standalone digit as layer index
                            layer_idx = int(part)
                            
                    if layer_idx is None or layer_idx >= num_layers:
                        continue
                        
                    # Look for attention-related parameters in the module
                    for param_name, param in module.named_parameters():
                        if 'query' in param_name or 'key' in param_name or 'value' in param_name:
                            # Similar logic as above to extract head-wise magnitudes
                            if len(param.shape) >= 2:
                                # Try to infer head dimension
                                if param.shape[0] == num_heads:
                                    for head_idx in range(num_heads):
                                        magnitude = torch.norm(param[head_idx])
                                        magnitudes[layer_idx, head_idx] += magnitude.item()
                                elif param.shape[0] % num_heads == 0:
                                    head_size = param.shape[0] // num_heads
                                    for head_idx in range(num_heads):
                                        start_idx = head_idx * head_size
                                        end_idx = (head_idx + 1) * head_size
                                        magnitude = torch.norm(param[start_idx:end_idx])
                                        magnitudes[layer_idx, head_idx] += magnitude.item()
                                        
                            found_weights = True
        
        # If we still didn't find weights, use a simple heuristic
        if not found_weights:
            logger.warning("Could not find attention weights, using random magnitudes for testing")
            magnitudes = torch.rand((num_layers, num_heads), device=self.device)
                
        # Convert to dictionary
        magnitude_dict = {i: magnitudes[i] for i in range(num_layers)}
        
        logger.info("Completed weight magnitude calculation")
        
        return magnitude_dict
        
    def prune(
        self,
        model,
        prune_percent: float = 0.2,
        **kwargs
    ) -> List[Tuple[int, int, float]]:
        """
        Prune heads with smallest weight magnitudes.
        
        Args:
            model: The model to prune
            prune_percent: Percentage of heads to prune (0.0 to 1.0)
            
        Returns:
            List of (layer_idx, head_idx, magnitude) for pruned heads
        """
        logger.info(f"Starting magnitude-based pruning with prune_percent={prune_percent}")
        
        # Collect magnitude distributions
        distributions = self.collect_distributions(model, None)
        
        # Detect model structure
        num_layers, num_heads = self.detect_model_structure(model)
        
        # Calculate total number of heads
        total_heads = num_layers * num_heads
        
        # Calculate number of heads to prune
        num_to_prune = int(total_heads * prune_percent)
        
        if num_to_prune <= 0:
            logger.warning(f"No heads to prune with prune_percent={prune_percent}")
            return []
            
        # Create flat list of (layer, head, magnitude) tuples
        head_magnitudes = []
        for layer_idx, layer_magnitude in distributions.items():
            for head_idx, magnitude in enumerate(layer_magnitude):
                if isinstance(magnitude, torch.Tensor):
                    magnitude_value = magnitude.item()
                else:
                    magnitude_value = float(magnitude)
                    
                head_magnitudes.append((int(layer_idx), int(head_idx), magnitude_value))
                
        # Skip if no valid magnitudes
        if not head_magnitudes:
            logger.warning("No valid magnitude values found")
            return []
            
        # Sort by magnitude (lowest first)
        head_magnitudes.sort(key=lambda x: x[2])
        
        # Select the heads to prune (lowest magnitude)
        pruned_heads = head_magnitudes[:num_to_prune]
        
        logger.info(f"Selected {len(pruned_heads)} heads for pruning")
        
        # Apply pruning mask to the model
        self.apply_pruning_mask(model, pruned_heads)
        
        return pruned_heads