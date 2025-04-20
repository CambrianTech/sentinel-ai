"""
Base pruning strategy for Neural Plasticity.

This module defines the base interface for all pruning strategies.

Version: v0.0.34 (2025-04-20 14:30:00)
"""

import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod

class PruningStrategy(ABC):
    """
    Abstract base class for pruning strategies.
    
    All pruning strategies must implement these methods:
    1. prune - Apply pruning to a model
    2. detect_model_structure - Detect number of layers and heads
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the pruning strategy.
        
        Args:
            device: Device to run on (auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def prune(
        self,
        model: torch.nn.Module,
        prune_percent: float,
        **kwargs
    ) -> List[Tuple[int, int, float]]:
        """
        Apply pruning to the model.
        
        Args:
            model: The model to prune
            prune_percent: Percentage of heads to prune (0.0 to 1.0)
            **kwargs: Additional strategy-specific arguments
            
        Returns:
            List of (layer_idx, head_idx, score) tuples for pruned heads
        """
        pass
        
    def detect_model_structure(
        self,
        model: torch.nn.Module
    ) -> Tuple[int, int]:
        """
        Detect number of layers and heads in a transformer model.
        
        Args:
            model: The transformer model
            
        Returns:
            Tuple of (num_layers, num_heads)
        """
        # Try various model architectures
        num_layers, num_heads = 0, 0
        
        # Check for direct blocks access (Sentinel-AI adaptive model)
        if hasattr(model, 'blocks'):
            num_layers = len(model.blocks)
            if hasattr(model.blocks[0].attn, 'num_heads'):
                num_heads = model.blocks[0].attn.num_heads
        
        # Try HuggingFace architectures
        else:
            # Get transformer component
            transformer = None
            if hasattr(model, 'transformer'):
                transformer = model.transformer
            elif hasattr(model, 'model') and hasattr(model.model, 'transformer'):
                transformer = model.model.transformer
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'transformer'):
                transformer = model.base_model.transformer
            
            # Find layers and heads
            if transformer:
                if hasattr(transformer, 'h'):  # GPT-2 style
                    num_layers = len(transformer.h)
                    if hasattr(transformer.h[0].attn, 'num_heads'):
                        num_heads = transformer.h[0].attn.num_heads
                elif hasattr(transformer, 'layer'):  # BERT style
                    num_layers = len(transformer.layer)
                    if hasattr(transformer.layer[0].attention.self, 'num_attention_heads'):
                        num_heads = transformer.layer[0].attention.self.num_attention_heads
        
        # If still not found, check config
        if num_heads == 0 and hasattr(model, 'config'):
            if hasattr(model.config, 'num_heads'):
                num_heads = model.config.num_heads
            elif hasattr(model.config, 'num_attention_heads'):
                num_heads = model.config.num_attention_heads
            
            if hasattr(model.config, 'num_hidden_layers'):
                num_layers = model.config.num_hidden_layers
            elif hasattr(model.config, 'n_layer'):
                num_layers = model.config.n_layer
        
        if num_layers == 0 or num_heads == 0:
            raise ValueError("Could not detect model structure. Please specify num_layers and num_heads manually.")
        
        return num_layers, num_heads