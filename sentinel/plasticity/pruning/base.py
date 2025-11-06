"""
Base Pruning Strategy

This module provides the abstract base class for all attention head pruning strategies.

Version: v0.0.34 (2025-04-20 17:00:00)
"""

import torch
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any, Set

logger = logging.getLogger(__name__)

class PruningStrategy(ABC):
    """
    Abstract base class for attention head pruning strategies.
    
    This class defines the interface for all pruning strategies.
    Each strategy must implement methods for:
    - Collecting relevant data from the model
    - Pruning based on that data
    - Detecting model structure
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the pruning strategy.
        
        Args:
            device: Device to run on (autodetected if None)
        """
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
    @abstractmethod
    def collect_distributions(
        self,
        model,
        dataloader
    ) -> Dict[int, torch.Tensor]:
        """
        Collect attention distributions from the model.
        
        Args:
            model: The model to analyze
            dataloader: DataLoader for input data
            
        Returns:
            Dictionary mapping layer indices to distributions
        """
        pass
        
    @abstractmethod
    def prune(
        self,
        model,
        prune_percent: float = 0.2,
        **kwargs
    ) -> List[Tuple[int, int, float]]:
        """
        Prune attention heads in the model.
        
        Args:
            model: The model to prune
            prune_percent: Percentage of heads to prune (0.0 to 1.0)
            
        Returns:
            List of (layer_idx, head_idx, score) for pruned heads
        """
        pass
        
    def detect_model_structure(self, model) -> Tuple[int, int]:
        """
        Detect the structure of the model (number of layers and heads).
        
        Args:
            model: The model to analyze
            
        Returns:
            Tuple of (num_layers, num_heads)
        """
        # Default implementation relies on model structure
        num_layers = 0
        num_heads = 0
        
        # Try to infer from model attributes
        try:
            # Check for conventional structure
            if hasattr(model, "config"):
                if hasattr(model.config, "num_hidden_layers"):
                    num_layers = model.config.num_hidden_layers
                    
                if hasattr(model.config, "num_attention_heads"):
                    num_heads = model.config.num_attention_heads
            
            # Fallback for other model structures
            if num_layers == 0 or num_heads == 0:
                # Try to find attention modules
                for name, module in model.named_modules():
                    if 'attention' in name.lower():
                        if 'layers' in name.lower():
                            # Count layers
                            parts = name.split('.')
                            for i, part in enumerate(parts):
                                if 'layer' in part.lower() and i + 1 < len(parts):
                                    try:
                                        layer_idx = int(parts[i + 1])
                                        num_layers = max(num_layers, layer_idx + 1)
                                    except ValueError:
                                        pass
                                        
                        # Try to get num_heads from attention module
                        if hasattr(module, 'num_heads'):
                            num_heads = max(num_heads, module.num_heads)
                        elif hasattr(module, 'num_attention_heads'):
                            num_heads = max(num_heads, module.num_attention_heads)
            
            # Fallback for adaptive transformer specific structure
            if hasattr(model, "adaptive_layers"):
                num_layers = len(model.adaptive_layers)
                
                # Try to get num_heads from first layer
                if num_layers > 0 and hasattr(model.adaptive_layers[0], "num_heads"):
                    num_heads = model.adaptive_layers[0].num_heads
        except Exception as e:
            logger.warning(f"Error detecting model structure: {e}")
            
        # If still not detected, use default values
        if num_layers == 0:
            logger.warning("Could not detect number of layers, using default value 12")
            num_layers = 12
            
        if num_heads == 0:
            logger.warning("Could not detect number of heads, using default value 12")
            num_heads = 12
            
        logger.info(f"Detected model structure: {num_layers} layers with {num_heads} heads each")
        
        return num_layers, num_heads
        
    def apply_pruning_mask(
        self,
        model,
        pruned_heads: List[Tuple[int, int, float]]
    ) -> None:
        """
        Apply pruning mask to the model.
        
        Args:
            model: The model to modify
            pruned_heads: List of (layer_idx, head_idx, score) tuples
        """
        # Set of pruned heads
        pruned_set = set((int(l), int(h)) for l, h, _ in pruned_heads)
        
        # Apply pruning
        if hasattr(model, "prune_heads"):
            # Group pruned heads by layer
            heads_by_layer = {}
            for layer_idx, head_idx, _ in pruned_heads:
                layer_idx = int(layer_idx)
                head_idx = int(head_idx)
                
                if layer_idx not in heads_by_layer:
                    heads_by_layer[layer_idx] = []
                    
                heads_by_layer[layer_idx].append(head_idx)
                
            # Prune heads layer by layer
            for layer_idx, heads in heads_by_layer.items():
                try:
                    model.prune_heads({layer_idx: heads})
                    logger.info(f"Pruned {len(heads)} heads in layer {layer_idx}")
                except Exception as e:
                    logger.error(f"Error pruning heads in layer {layer_idx}: {e}")
        
        # For adaptive transformer models
        elif hasattr(model, "set_pruned_heads"):
            try:
                model.set_pruned_heads(pruned_set)
                logger.info(f"Set {len(pruned_set)} pruned heads")
            except Exception as e:
                logger.error(f"Error setting pruned heads: {e}")
                
        # For models with adaptive layers
        elif hasattr(model, "adaptive_layers"):
            success = 0
            for layer_idx, head_idx, _ in pruned_heads:
                try:
                    layer = model.adaptive_layers[int(layer_idx)]
                    if hasattr(layer, "set_head_pruned"):
                        layer.set_head_pruned(int(head_idx), True)
                        success += 1
                except (IndexError, AttributeError) as e:
                    logger.error(f"Error pruning head {head_idx} in layer {layer_idx}: {e}")
                    
            logger.info(f"Successfully pruned {success} out of {len(pruned_heads)} heads")
                    
        else:
            logger.warning("Model does not support pruning, no heads were pruned")
            
        # Store pruned heads in model for reference
        model.pruned_heads = pruned_set