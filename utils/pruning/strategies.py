"""
Pruning strategies for selecting which heads to prune.

This module is maintained for backward compatibility.
New code should import from sentinel.pruning.strategies instead.
"""

import warnings
import random
import jax
import jax.numpy as jnp

# Emit deprecation warning
warnings.warn(
    "The utils.pruning.strategies module is deprecated. "
    "Please use sentinel.pruning.strategies instead.",
    DeprecationWarning,
    stacklevel=2
)

try:
    # Import from the new location
    from sentinel.pruning.strategies.base import PruningStrategy
    from sentinel.pruning.strategies.random import RandomPruningStrategy as RandomStrategy
    from sentinel.pruning.strategies.magnitude import MagnitudePruningStrategy as MagnitudeStrategy
    from sentinel.pruning.strategies.entropy import EntropyPruningStrategy as AttentionEntropyStrategy
    from sentinel.pruning.strategies import get_strategy
except ImportError:
    # Fallback to original implementation if the new location is not available
    class PruningStrategy:
        """Base class for pruning strategies"""
        
        def __init__(self, pruning_module):
            self.pruning_module = pruning_module
        
        def get_head_importance(self, params):
            """Calculate importance for all heads"""
            raise NotImplementedError("Subclasses must implement get_head_importance")
        
        def prune_heads(self, params, head_indices):
            """Prune specified heads"""
            pruned_params = jax.tree_util.tree_map(lambda x: x, params)  # Deep copy
            
            for layer_idx, head_idx in head_indices:
                pruned_params = self.pruning_module.prune_head(pruned_params, layer_idx, head_idx)
                
            return pruned_params

    class RandomStrategy(PruningStrategy):
        """Random pruning strategy"""
        
        def get_head_importance(self, params):
            """Assign random importance to heads"""
            all_head_importance = []
            
            for layer_idx in range(self.pruning_module.num_layers):
                for head_idx in range(self.pruning_module.num_heads):
                    # Random importance score
                    score = random.random()
                    all_head_importance.append((layer_idx, head_idx, score))
            
            return all_head_importance

    class MagnitudeStrategy(PruningStrategy):
        """Magnitude-based pruning strategy"""
        
        def get_head_importance(self, params):
            """Calculate importance based on weight magnitudes"""
            all_head_importance = []
            model_type = self.pruning_module.model_type
            
            for layer_idx in range(self.pruning_module.num_layers):
                for head_idx in range(self.pruning_module.num_heads):
                    # Get head weights based on model type
                    if model_type == "gpt2":
                        # Access attention output projection
                        transformer_path = "transformer"
                        layer_path = "h"
                        layer_key = str(layer_idx)
                        attn_path = "attn"
                        
                        attn_block = params[transformer_path][layer_path][layer_key][attn_path]
                        output_proj = attn_block["c_proj"]["kernel"]
                        
                        # Calculate head dimensions
                        head_size = output_proj.shape[0] // self.pruning_module.num_heads
                        
                        # Get weights for this head
                        start_idx = head_idx * head_size
                        end_idx = (head_idx + 1) * head_size
                        head_weights = output_proj[start_idx:end_idx, :]
                        
                    elif model_type == "opt":
                        # For OPT models
                        model_path = "model"
                        decoder_path = "decoder"
                        layers_path = "layers"
                        layer_key = str(layer_idx)
                        attn_path = "self_attn"
                        
                        attn_block = params[model_path][decoder_path][layers_path][layer_key][attn_path]
                        output_proj = attn_block["out_proj"]["kernel"]
                        
                        # Calculate head dimensions
                        head_size = output_proj.shape[0] // self.pruning_module.num_heads
                        
                        # Get weights for this head
                        start_idx = head_idx * head_size
                        end_idx = (head_idx + 1) * head_size
                        head_weights = output_proj[start_idx:end_idx, :]
                        
                    elif model_type == "pythia":
                        # For Pythia models
                        transformer_path = "transformer"
                        layer_path = "h"
                        layer_key = str(layer_idx)
                        attn_path = "attn"
                        
                        attn_block = params[transformer_path][layer_path][layer_key][attn_path]
                        output_proj = attn_block["proj"]["kernel"]
                        
                        # Calculate head dimensions
                        head_size = output_proj.shape[0] // self.pruning_module.num_heads
                        
                        # Get weights for this head
                        start_idx = head_idx * head_size
                        end_idx = (head_idx + 1) * head_size
                        head_weights = output_proj[start_idx:end_idx, :]
                    
                    # Calculate importance as L2 norm of weights
                    importance = jnp.linalg.norm(head_weights).item()
                    all_head_importance.append((layer_idx, head_idx, importance))
            
            return all_head_importance

    class AttentionEntropyStrategy(PruningStrategy):
        """Entropy-based pruning strategy using attention patterns"""
        
        def __init__(self, pruning_module, sample_text=None):
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
        
        def get_head_importance(self, params):
            """Calculate importance based on fallback to magnitude"""
            # For simplicity and compatibility across model types,
            # we'll just use magnitude-based pruning as a proxy
            magnitude_strategy = MagnitudeStrategy(self.pruning_module)
            return magnitude_strategy.get_head_importance(params)

    # Factory function to get strategy by name
    def get_strategy(name, pruning_module, sample_text=None):
        """Get pruning strategy by name"""
        if name.lower() == "random":
            return RandomStrategy(pruning_module)
        elif name.lower() == "magnitude":
            return MagnitudeStrategy(pruning_module)
        elif name.lower() == "entropy":
            return AttentionEntropyStrategy(pruning_module, sample_text)
        else:
            raise ValueError(f"Unknown strategy: {name}")