"""
Magnitude-based pruning strategy.
"""

import jax.numpy as jnp
from typing import List, Tuple, Any
from sentinel.pruning.strategies.base import PruningStrategy


class MagnitudePruningStrategy(PruningStrategy):
    """
    Magnitude-based pruning strategy.
    
    This strategy calculates the importance of attention heads based on the L2 norm
    of their weight matrices. Heads with smaller weight magnitudes are considered
    less important and are candidates for pruning.
    """
    
    def get_head_importance(self, params: Any) -> List[Tuple[int, int, float]]:
        """
        Calculate importance based on weight magnitudes.
        
        Args:
            params: Model parameters
            
        Returns:
            List of (layer_idx, head_idx, importance_score) tuples based on weight magnitudes
        """
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