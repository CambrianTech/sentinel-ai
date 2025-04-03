"""
Head growth implementation for expanding pruned model architectures.

This module provides strategies and utilities for strategically growing 
attention heads in transformer models that have previously been pruned.
"""

import random
import copy
import jax
import jax.numpy as jnp
from collections import defaultdict
import torch
import numpy as np

class GrowthStrategy:
    """Base class for head growth strategies"""
    
    def __init__(self, pruning_module):
        self.pruning_module = pruning_module
    
    def get_head_candidates(self, params, active_heads, growth_percentage=0.05):
        """Calculate which heads to add based on the strategy"""
        raise NotImplementedError("Subclasses must implement get_head_candidates")

class GradientSensitivityStrategy(GrowthStrategy):
    """Gradient-based head growth strategy that adds heads where they would have most impact"""
    
    def calculate_gradient_sensitivity(self, params, active_heads, eval_batch=None):
        """
        Calculate gradient-based sensitivity scores for possible head positions.
        Higher scores indicate positions where adding a head would have more impact.
        """
        sensitivity_scores = []
        model_type = self.pruning_module.model_type
        
        # Get a small batch of data if not provided
        if eval_batch is None:
            if hasattr(self.pruning_module, 'tokenizer'):
                tokenizer = self.pruning_module.tokenizer
                texts = [
                    "The model needs to process this text efficiently.",
                    "This is a sample for measuring gradient sensitivity.",
                    "Artificial intelligence systems should be adaptive."
                ]
                eval_batch = tokenizer(texts, return_tensors="jax", padding=True, truncation=True)
            else:
                # Cannot proceed without data
                raise ValueError("No tokenizer available and no eval_batch provided")
        
        # Convert params to PyTorch for gradient tracking
        # For a real implementation, we would ideally do this in JAX/Flax
        # but using PyTorch for simplicity since it handles autograd well
        try:
            import torch
            
            # Get model dimensions
            num_layers = self.pruning_module.num_layers
            num_heads = self.pruning_module.num_heads
            
            # Create a simple proxy model to measure gradient sensitivity
            # This is a placeholder - in practice, we'd construct a model 
            # that matches the architecture of the pruned model
            class SimpleTransformerProxy(torch.nn.Module):
                def __init__(self, num_layers, num_heads, active_heads):
                    super().__init__()
                    self.num_layers = num_layers
                    self.num_heads = num_heads
                    self.active_heads = active_heads
                    
                    # Create parameters representing each potential head
                    self.head_params = torch.nn.ParameterDict()
                    for layer in range(num_layers):
                        for head in range(num_heads):
                            # Small parameter tensor to represent each head
                            # In a real implementation, this would be the actual head weights
                            param_name = f"layer_{layer}_head_{head}"
                            is_active = (layer, head) in active_heads
                            # Create parameter with requires_grad=True only for inactive heads
                            # that we're considering for growth
                            self.head_params[param_name] = torch.nn.Parameter(
                                torch.ones(1), 
                                requires_grad=not is_active
                            )
                    
                def forward(self, x):
                    # Simple forward pass that incorporates all head parameters
                    # In a real implementation, this would be the actual model forward pass
                    result = torch.zeros(1, requires_grad=True)
                    for layer in range(self.num_layers):
                        for head in range(self.num_heads):
                            param_name = f"layer_{layer}_head_{head}"
                            if (layer, head) in self.active_heads or self.head_params[param_name].requires_grad:
                                result = result + self.head_params[param_name]
                    return result
            
            # Create proxy model
            proxy_model = SimpleTransformerProxy(num_layers, num_heads, active_heads)
            
            # Prepare input tensor
            # In a real implementation, convert the eval_batch to PyTorch tensors
            input_tensor = torch.ones(1, requires_grad=True)
            
            # Forward pass
            output = proxy_model(input_tensor)
            
            # Create a simple loss function
            loss = output.sum()
            
            # Backward pass
            loss.backward()
            
            # Calculate sensitivity scores based on gradients
            for layer in range(num_layers):
                for head in range(num_heads):
                    if (layer, head) not in active_heads:
                        param_name = f"layer_{layer}_head_{head}"
                        param = proxy_model.head_params[param_name]
                        if param.grad is not None:
                            # Higher gradient magnitude indicates higher sensitivity
                            sensitivity = param.grad.abs().item()
                            sensitivity_scores.append((layer, head, sensitivity))
            
        except ImportError:
            # Fallback if PyTorch not available - use random scores with layer weighting
            print("Warning: PyTorch not available, using randomized gradient sensitivity approximation")
            for layer_idx in range(self.pruning_module.num_layers):
                for head_idx in range(self.pruning_module.num_heads):
                    if (layer_idx, head_idx) not in active_heads:
                        # Add layer bias to prioritize earlier layers
                        layer_weight = 1.0 - (layer_idx / self.pruning_module.num_layers * 0.5)
                        score = random.random() * layer_weight
                        sensitivity_scores.append((layer_idx, head_idx, score))
        
        return sensitivity_scores
    
    def get_head_candidates(self, params, active_heads, growth_percentage=0.05):
        """
        Get head candidates to grow based on gradient sensitivity
        """
        # Calculate sensitivity scores for all possible head positions
        sensitivity_scores = self.calculate_gradient_sensitivity(params, active_heads)
        
        # Sort by sensitivity score (highest first)
        sensitivity_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Calculate total number of heads
        total_heads = self.pruning_module.num_layers * self.pruning_module.num_heads
        
        # Calculate number of heads to add
        heads_to_add = max(1, int(growth_percentage * total_heads))
        
        # Return top candidates based on sensitivity scores
        return [(layer_idx, head_idx) for layer_idx, head_idx, _ in sensitivity_scores[:heads_to_add]]

class EntropyGapStrategy(GrowthStrategy):
    """Adds heads where there's a significant entropy gap in the attention patterns"""
    
    def calculate_attention_entropy(self, attention_pattern):
        """Calculate entropy of an attention pattern"""
        # attention_pattern shape: [batch_size, num_heads, seq_len, seq_len]
        # Ensure attention sums to 1 across the last dimension
        attention_pattern = attention_pattern / (jnp.sum(attention_pattern, axis=-1, keepdims=True) + 1e-12)
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -jnp.sum(attention_pattern * jnp.log(attention_pattern + 1e-12), axis=-1)
        
        # Average across sequence length and batch dimensions
        return jnp.mean(entropy, axis=(-1, -2))
    
    def calculate_entropy_gaps(self, params, active_heads, eval_batch=None):
        """
        Calculate entropy gaps for possible head positions.
        Higher gaps indicate positions where adding a head could provide more diverse attention.
        """
        entropy_gaps = []
        model_type = self.pruning_module.model_type
        
        # Get a small batch of data if not provided
        if eval_batch is None:
            if hasattr(self.pruning_module, 'tokenizer'):
                tokenizer = self.pruning_module.tokenizer
                texts = [
                    "The model needs to process this text efficiently.",
                    "This is a sample for measuring attention entropy.",
                    "Artificial intelligence systems should be adaptive."
                ]
                eval_batch = tokenizer(texts, return_tensors="jax", padding=True, truncation=True)
            else:
                # Cannot proceed without data
                raise ValueError("No tokenizer available and no eval_batch provided")
        
        # In a real implementation, we would do a forward pass with attention capture
        # Here we're using a simplified approach with randomization 
        
        # Calculate per-layer entropy gaps
        layer_entropy_gaps = {}
        
        try:
            # Get model
            model = self.pruning_module.model
            
            # Forward pass with attention capture
            if model is not None:
                # This requires model support for outputting attentions
                outputs = model(**eval_batch, params=params, output_attentions=True)
                if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    attentions = outputs.attentions
                    
                    # Calculate entropy for each layer
                    for layer_idx, layer_attention in enumerate(attentions):
                        layer_entropy = self.calculate_attention_entropy(layer_attention)
                        
                        # Calculate optimal entropy (maximum when attention is uniform)
                        seq_len = layer_attention.shape[-1]
                        optimal_entropy = jnp.log(seq_len)
                        
                        # Calculate entropy gap (how far from optimal)
                        gap = optimal_entropy - jnp.mean(layer_entropy)
                        layer_entropy_gaps[layer_idx] = gap.item()
            
            # For positions without active heads, calculate entropy gaps
            for layer_idx in range(self.pruning_module.num_layers):
                # Count active heads in this layer
                active_in_layer = sum(1 for l, h in active_heads if l == layer_idx)
                
                # Skip layers with all heads active
                if active_in_layer >= self.pruning_module.num_heads:
                    continue
                
                # Get entropy gap for this layer
                gap = layer_entropy_gaps.get(layer_idx, random.random())
                
                # For each inactive head in this layer
                for head_idx in range(self.pruning_module.num_heads):
                    if (layer_idx, head_idx) not in active_heads:
                        entropy_gaps.append((layer_idx, head_idx, gap))
        
        except Exception as e:
            # Fallback if entropy calculation fails
            print(f"Warning: Error in entropy calculation ({e}), using weighted random approach")
            for layer_idx in range(self.pruning_module.num_layers):
                # Count active heads in this layer
                active_in_layer = sum(1 for l, h in active_heads if l == layer_idx)
                
                # Skip layers with all heads active
                if active_in_layer >= self.pruning_module.num_heads:
                    continue
                
                # Add layer bias to prioritize earlier layers with fewer active heads
                layer_weight = 1.0 - (layer_idx / self.pruning_module.num_layers * 0.5)
                head_density = active_in_layer / self.pruning_module.num_heads
                score = random.random() * layer_weight * (1.0 - head_density)
                
                # For each inactive head in this layer
                for head_idx in range(self.pruning_module.num_heads):
                    if (layer_idx, head_idx) not in active_heads:
                        entropy_gaps.append((layer_idx, head_idx, score))
        
        return entropy_gaps
    
    def get_head_candidates(self, params, active_heads, growth_percentage=0.05):
        """
        Get head candidates to grow based on entropy gaps
        """
        # Calculate entropy gaps
        entropy_gaps = self.calculate_entropy_gaps(params, active_heads)
        
        # Sort by gap size (highest first)
        entropy_gaps.sort(key=lambda x: x[2], reverse=True)
        
        # Calculate total number of heads
        total_heads = self.pruning_module.num_layers * self.pruning_module.num_heads
        
        # Calculate number of heads to add
        heads_to_add = max(1, int(growth_percentage * total_heads))
        
        # Return top candidates based on entropy gaps
        return [(layer_idx, head_idx) for layer_idx, head_idx, _ in entropy_gaps[:heads_to_add]]

class BalancedStrategy(GrowthStrategy):
    """Ensures heads are distributed evenly across layers"""
    
    def get_head_candidates(self, params, active_heads, growth_percentage=0.05):
        """
        Get head candidates to grow based on balanced distribution across layers
        """
        # Count active heads per layer
        layer_counts = defaultdict(int)
        for layer_idx, _ in active_heads:
            layer_counts[layer_idx] += 1
        
        # Identify layers with fewest active heads
        layer_ranking = [(layer_idx, count) for layer_idx, count in layer_counts.items()]
        layer_ranking.sort(key=lambda x: x[1])
        
        # Create candidate list starting with layers having fewest heads
        candidates = []
        for layer_idx, _ in layer_ranking:
            for head_idx in range(self.pruning_module.num_heads):
                if (layer_idx, head_idx) not in active_heads:
                    candidates.append((layer_idx, head_idx))
        
        # Calculate total number of heads
        total_heads = self.pruning_module.num_layers * self.pruning_module.num_heads
        
        # Calculate number of heads to add
        heads_to_add = max(1, int(growth_percentage * total_heads))
        
        # Return top candidates based on layer balance
        return candidates[:heads_to_add]

class RandomStrategy(GrowthStrategy):
    """Random head growth strategy"""
    
    def get_head_candidates(self, params, active_heads, growth_percentage=0.05):
        """
        Get random head candidates to grow
        """
        # Create list of all inactive heads
        inactive_heads = []
        for layer_idx in range(self.pruning_module.num_layers):
            for head_idx in range(self.pruning_module.num_heads):
                if (layer_idx, head_idx) not in active_heads:
                    inactive_heads.append((layer_idx, head_idx))
        
        # Shuffle the list
        random.shuffle(inactive_heads)
        
        # Calculate total number of heads
        total_heads = self.pruning_module.num_layers * self.pruning_module.num_heads
        
        # Calculate number of heads to add
        heads_to_add = max(1, int(growth_percentage * total_heads))
        
        # Return random candidates
        return inactive_heads[:heads_to_add]

def grow_attention_heads_gradually(pruning_module, params=None, active_heads=None, 
                                growth_percentage=0.05, strategy="gradient_sensitivity", 
                                initial_scale=0.01, warmup_steps=100):
    """
    Gradually grow new attention heads to prevent performance collapse.
    
    Args:
        pruning_module: The pruning module containing the model
        params: Model parameters (if None, uses pruning_module.model.params)
        active_heads: List of (layer_idx, head_idx) tuples of active heads
                     (if None, determined from params)
        growth_percentage: Percentage of new heads to add
        strategy: Strategy to determine where to add heads
                  ("gradient_sensitivity", "entropy_gap", "balanced", "random")
        initial_scale: Initial scaling factor for new head weights (small to start)
        warmup_steps: Number of steps to linearly increase head influence
        
    Returns:
        new_params: Model parameters with new heads added
        added_count: Number of heads added
        added_heads: List of (layer, head) tuples where heads were added
        warmup_schedule: Function to update head scaling during warmup
    """
    # Get model parameters if not provided
    if params is None:
        if hasattr(pruning_module, 'model') and hasattr(pruning_module.model, 'params'):
            params = pruning_module.model.params
        else:
            raise ValueError("No params provided and couldn't find model params")
    
    # Determine active heads if not provided
    if active_heads is None:
        active_heads = determine_active_heads(pruning_module, params)
    
    # Create strategy based on name
    if strategy == "gradient_sensitivity":
        growth_strategy = GradientSensitivityStrategy(pruning_module)
    elif strategy == "entropy_gap":
        growth_strategy = EntropyGapStrategy(pruning_module)
    elif strategy == "balanced":
        growth_strategy = BalancedStrategy(pruning_module)
    else:  # Default to random
        growth_strategy = RandomStrategy(pruning_module)
    
    # Get head candidates to grow
    head_candidates = growth_strategy.get_head_candidates(params, active_heads, growth_percentage)
    
    if not head_candidates:
        print("No heads to add - model may already have all heads active")
        return params, 0, [], lambda step: 1.0
    
    # Create new parameters with added heads
    new_params = add_attention_heads(pruning_module, params, head_candidates, initial_scale)
    
    # Create warmup schedule function
    def warmup_schedule(step):
        if step >= warmup_steps:
            return 1.0
        return min(1.0, initial_scale + (1.0 - initial_scale) * (step / warmup_steps))
    
    return new_params, len(head_candidates), head_candidates, warmup_schedule

def determine_active_heads(pruning_module, params):
    """
    Determine which attention heads are currently active in the model
    """
    active_heads = set()
    model_type = pruning_module.model_type
    
    for layer_idx in range(pruning_module.num_layers):
        for head_idx in range(pruning_module.num_heads):
            # Check if this head is active by examining its output projection weights
            weights = get_head_output_weights(pruning_module, params, layer_idx, head_idx)
            
            # If weights aren't all zeros, the head is active
            if weights is not None and not jnp.allclose(weights, 0.0, atol=1e-5):
                active_heads.add((layer_idx, head_idx))
    
    return active_heads

def get_head_output_weights(pruning_module, params, layer_idx, head_idx):
    """
    Extract the output projection weights for a specific attention head
    """
    model_type = pruning_module.model_type
    
    if model_type == "gpt2":
        # Access attention output projection for GPT-2
        transformer_path = "transformer"
        layer_path = "h"
        layer_key = str(layer_idx)
        attn_path = "attn"
        
        try:
            attn_block = params[transformer_path][layer_path][layer_key][attn_path]
            output_proj = attn_block["c_proj"]["kernel"]
            
            # Calculate head dimensions
            head_size = output_proj.shape[0] // pruning_module.num_heads
            
            # Get weights for this head
            start_idx = head_idx * head_size
            end_idx = (head_idx + 1) * head_size
            return output_proj[start_idx:end_idx, :]
        except (KeyError, IndexError):
            return None
    
    elif model_type == "opt":
        # For OPT models
        model_path = "model"
        decoder_path = "decoder"
        layers_path = "layers"
        layer_key = str(layer_idx)
        attn_path = "self_attn"
        
        try:
            attn_block = params[model_path][decoder_path][layers_path][layer_key][attn_path]
            output_proj = attn_block["out_proj"]["kernel"]
            
            # Calculate head dimensions
            head_size = output_proj.shape[0] // pruning_module.num_heads
            
            # Get weights for this head
            start_idx = head_idx * head_size
            end_idx = (head_idx + 1) * head_size
            return output_proj[start_idx:end_idx, :]
        except (KeyError, IndexError):
            return None
    
    elif model_type == "pythia":
        # For Pythia models
        transformer_path = "transformer"
        layer_path = "h"
        layer_key = str(layer_idx)
        attn_path = "attn"
        
        try:
            attn_block = params[transformer_path][layer_path][layer_key][attn_path]
            output_proj = attn_block["proj"]["kernel"]
            
            # Calculate head dimensions
            head_size = output_proj.shape[0] // pruning_module.num_heads
            
            # Get weights for this head
            start_idx = head_idx * head_size
            end_idx = (head_idx + 1) * head_size
            return output_proj[start_idx:end_idx, :]
        except (KeyError, IndexError):
            return None
    
    return None

def add_attention_heads(pruning_module, params, head_candidates, initial_scale=0.01):
    """
    Add new attention heads to the model with small initial weights
    """
    # Create a deep copy of the parameters
    new_params = jax.tree_util.tree_map(lambda x: x, params)
    model_type = pruning_module.model_type
    
    for layer_idx, head_idx in head_candidates:
        if model_type == "gpt2":
            # Access paths for GPT-2
            transformer_path = "transformer"
            layer_path = "h"
            layer_key = str(layer_idx)
            attn_path = "attn"
            
            # Get attention block
            attn_block = new_params[transformer_path][layer_path][layer_key][attn_path]
            
            # Calculate head dimensions
            if "c_attn" in attn_block:
                # Get sizes from combined QKV projection
                qkv_kernel = attn_block["c_attn"]["kernel"]
                hidden_size = qkv_kernel.shape[1] // 3
            else:
                # Fallback using output projection
                output_proj = attn_block["c_proj"]["kernel"]
                hidden_size = output_proj.shape[0]
                
            head_size = hidden_size // pruning_module.num_heads
            
            # Calculate indices for this head
            start_idx = head_idx * head_size
            end_idx = (head_idx + 1) * head_size
            
            # For output projection, initialize with small non-zero weights
            output_proj = attn_block["c_proj"]["kernel"]
            
            # Check if weights are currently zeros (pruned)
            head_weights = output_proj[start_idx:end_idx, :]
            
            if jnp.allclose(head_weights, 0.0, atol=1e-5):
                # Generate small random values for initialization
                small_weights = jnp.ones_like(head_weights) * initial_scale
                # Set the new weights
                output_proj = output_proj.at[start_idx:end_idx, :].set(small_weights)
                # Update in parameters
                new_params[transformer_path][layer_path][layer_key][attn_path]["c_proj"]["kernel"] = output_proj
        
        elif model_type == "opt":
            # For OPT models
            model_path = "model"
            decoder_path = "decoder"
            layers_path = "layers"
            layer_key = str(layer_idx)
            attn_path = "self_attn"
            
            # Get attention block
            attn_block = new_params[model_path][decoder_path][layers_path][layer_key][attn_path]
            
            # Calculate head dimensions
            output_proj = attn_block["out_proj"]["kernel"]
            hidden_size = output_proj.shape[0]
            head_size = hidden_size // pruning_module.num_heads
            
            # Calculate indices for this head
            start_idx = head_idx * head_size
            end_idx = (head_idx + 1) * head_size
            
            # Check if weights are currently zeros (pruned)
            head_weights = output_proj[start_idx:end_idx, :]
            
            if jnp.allclose(head_weights, 0.0, atol=1e-5):
                # Generate small random values for initialization
                small_weights = jnp.ones_like(head_weights) * initial_scale
                # Set the new weights
                output_proj = output_proj.at[start_idx:end_idx, :].set(small_weights)
                # Update in parameters
                new_params[model_path][decoder_path][layers_path][layer_key][attn_path]["out_proj"]["kernel"] = output_proj
        
        elif model_type == "pythia":
            # For Pythia models
            transformer_path = "transformer"
            layer_path = "h"
            layer_key = str(layer_idx)
            attn_path = "attn"
            
            # Get attention block
            attn_block = new_params[transformer_path][layer_path][layer_key][attn_path]
            
            # Calculate head dimensions
            output_proj = attn_block["proj"]["kernel"]
            hidden_size = output_proj.shape[0]
            head_size = hidden_size // pruning_module.num_heads
            
            # Calculate indices for this head
            start_idx = head_idx * head_size
            end_idx = (head_idx + 1) * head_size
            
            # Check if weights are currently zeros (pruned)
            head_weights = output_proj[start_idx:end_idx, :]
            
            if jnp.allclose(head_weights, 0.0, atol=1e-5):
                # Generate small random values for initialization
                small_weights = jnp.ones_like(head_weights) * initial_scale
                # Set the new weights
                output_proj = output_proj.at[start_idx:end_idx, :].set(small_weights)
                # Update in parameters
                new_params[transformer_path][layer_path][layer_key][attn_path]["proj"]["kernel"] = output_proj
    
    return new_params

# Factory function to get growth strategy by name
def get_strategy(name, pruning_module):
    """Get growth strategy by name"""
    if name.lower() == "gradient_sensitivity":
        return GradientSensitivityStrategy(pruning_module)
    elif name.lower() == "entropy_gap":
        return EntropyGapStrategy(pruning_module)
    elif name.lower() == "balanced":
        return BalancedStrategy(pruning_module)
    elif name.lower() == "random":
        return RandomStrategy(pruning_module)
    else:
        raise ValueError(f"Unknown strategy: {name}")