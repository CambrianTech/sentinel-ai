"""
Base class for model loaders in Sentinel AI.

This module provides the foundation for loading and configuring
different model architectures to work with Sentinel's adaptive features.
"""

import torch
import os

class ModelLoader:
    """
    Base class for model loaders.
    
    This class provides common functionality for loading models with
    adaptive features like attention head gating and agency.
    """
    
    def __init__(self, cache_dir=None, debug=False):
        """
        Initialize the model loader.
        
        Args:
            cache_dir: Directory to cache downloaded models
            debug: Whether to print debug information
        """
        self.cache_dir = cache_dir
        self.debug = debug
        
        # Check for environment variables
        self.use_optimized = os.environ.get("USE_OPTIMIZED_MODEL", "1") == "1"
        self.use_integrated = os.environ.get("USE_INTEGRATED_MODEL", "1") == "1"
        
    def load_model(self, model_name, device=None):
        """
        Load a model with adaptive architecture.
        
        Args:
            model_name: The name or path of the model to load
            device: Device to load the model on (default: auto-detect)
            
        Returns:
            Tuple of (model, tokenizer)
        """
        # Determine device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # This method should be implemented by subclasses
        raise NotImplementedError(
            "Subclasses must implement the load_model method."
        )
    
    def configure_gates(self, model, gate_values=None, pattern=None):
        """
        Configure gate values for attention heads.
        
        Args:
            model: The model to configure
            gate_values: Value to set for all gates, or dictionary mapping (layer, head) to value
            pattern: Predefined pattern name to apply
            
        Returns:
            Updated model
        """
        if not hasattr(model, "transformer") or not hasattr(model.transformer, "blocks"):
            raise ValueError("Model does not have the expected structure for gating")
        
        # Set gates based on pattern or gate_values
        with torch.no_grad():
            if pattern is not None:
                # Apply predefined pattern
                self._apply_gate_pattern(model, pattern)
            elif gate_values is not None:
                # Apply specific gate values
                self._set_gate_values(model, gate_values)
        
        return model
    
    def _apply_gate_pattern(self, model, pattern):
        """
        Apply a predefined gating pattern.
        
        Args:
            model: The model to configure
            pattern: Name of the pattern to apply
        """
        # Default patterns
        patterns = {
            "all_active": 1.0,
            "all_inactive": 0.0,
            "alternate": lambda l, h: 1.0 if (l + h) % 2 == 0 else 0.0,
            "upper_half": lambda l, h: 1.0 if l < len(model.transformer.blocks) // 2 else 0.0,
            "lower_half": lambda l, h: 1.0 if l >= len(model.transformer.blocks) // 2 else 0.0
        }
        
        if pattern not in patterns:
            raise ValueError(f"Unknown pattern: {pattern}. Available patterns: {list(patterns.keys())}")
        
        # Get pattern value (function or constant)
        pattern_value = patterns[pattern]
        
        # Apply pattern to each head
        for layer_idx, block in enumerate(model.transformer.blocks):
            for head_idx in range(block.attn.num_heads):
                # Calculate value for this head (either call function or use constant)
                if callable(pattern_value):
                    value = pattern_value(layer_idx, head_idx)
                else:
                    value = pattern_value
                
                # Set gate value
                block.attn.gate[head_idx] = value
    
    def _set_gate_values(self, model, gate_values):
        """
        Set specific gate values.
        
        Args:
            model: The model to configure
            gate_values: Value to set for all gates, or dictionary mapping (layer, head) to value
        """
        if isinstance(gate_values, (int, float)):
            # Set all gates to the same value
            for block in model.transformer.blocks:
                block.attn.gate.fill_(float(gate_values))
        elif isinstance(gate_values, dict):
            # Set specific gates
            for (layer_idx, head_idx), value in gate_values.items():
                if 0 <= layer_idx < len(model.transformer.blocks):
                    if 0 <= head_idx < model.transformer.blocks[layer_idx].attn.num_heads:
                        model.transformer.blocks[layer_idx].attn.gate[head_idx] = value
                    else:
                        if self.debug:
                            print(f"Warning: Invalid head index {head_idx} for layer {layer_idx}")
                else:
                    if self.debug:
                        print(f"Warning: Invalid layer index {layer_idx}")
        else:
            raise ValueError("gate_values must be a number or a dictionary")