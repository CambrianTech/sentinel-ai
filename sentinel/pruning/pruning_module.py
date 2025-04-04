"""
Core pruning implementation using JAX/Flax
"""

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

class PruningModule:
    """Core pruning implementation using JAX/Flax"""
    
    def __init__(self, model_name="distilgpt2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.original_params = None
        self.model_type = self._get_model_type(model_name)
        self.num_layers = 0
        self.num_heads = 0
        self.head_dim = 0
        self.load_model()
        
    def _get_model_type(self, model_name):
        """Determine model type from name."""
        model_name = model_name.lower()
        if "gpt2" in model_name or "gpt-2" in model_name:
            return "gpt2"
        elif "gpt-j" in model_name or "gptj" in model_name:
            return "gptj"
        elif "gpt-neo" in model_name or "gptneo" in model_name:
            return "gptneo"
        elif "opt" in model_name:
            return "opt"
        elif "bloom" in model_name:
            return "bloom"
        elif "llama" in model_name:
            return "llama"
        else:
            # Default to GPT-2 architecture
            return "gpt2"
            
    def load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model {self.model_name} with JAX...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = FlaxAutoModelForCausalLM.from_pretrained(self.model_name)
        self.original_params = self.model.params
        
        # Get model dimensions based on type
        if self.model_type == "gpt2":
            config = self.model.config
            self.num_layers = config.n_layer
            self.num_heads = config.n_head
            self.head_dim = config.n_embd // config.n_head
        elif self.model_type in ["gptj", "gptneo", "opt", "bloom", "llama"]:
            config = self.model.config
            self.num_layers = config.num_hidden_layers
            self.num_heads = config.num_attention_heads
            self.head_dim = config.hidden_size // config.num_attention_heads
        
        print(f"Model loaded: {self.model_name}")
        print(f"Layers: {self.num_layers}, Heads per layer: {self.num_heads}, Head dimension: {self.head_dim}")
        
    def get_attention_weights(self):
        """Extract attention weights from model params."""
        attention_weights = {}
        
        if self.model_type == "gpt2":
            for layer in range(self.num_layers):
                key = f"transformer/h/{layer}/attn/c_attn"
                if key in self.model.params:
                    attention_weights[f"layer_{layer}"] = self.model.params[key]["kernel"]
        else:
            # Handle other model types
            print(f"Attention weight extraction not implemented for {self.model_type}")
            
        return attention_weights
        
    def prune_heads(self, head_mask):
        """
        Prune heads according to the provided mask.
        
        Args:
            head_mask: A binary mask of shape [num_layers, num_heads] where 1 means keep and 0 means prune
        
        Returns:
            Pruned model parameters
        """
        if head_mask.shape != (self.num_layers, self.num_heads):
            raise ValueError(f"Expected head mask shape ({self.num_layers}, {self.num_heads}), got {head_mask.shape}")
        
        # Clone the original parameters
        pruned_params = jax.tree_map(lambda x: x, self.original_params)
        
        # Apply pruning based on model type
        if self.model_type == "gpt2":
            for layer in range(self.num_layers):
                for head in range(self.num_heads):
                    if head_mask[layer, head] == 0:
                        # Prune this head
                        self._prune_gpt2_head(pruned_params, layer, head)
        else:
            # Handle other model types
            print(f"Pruning not implemented for {self.model_type}")
            
        return pruned_params
        
    def _prune_gpt2_head(self, params, layer, head):
        """Zero out a specific attention head in a GPT-2 model."""
        # Get head dimension offsets
        head_offset = head * self.head_dim
        
        # Prune query, key, value projections
        attn_key = f"transformer/h/{layer}/attn/c_attn"
        if attn_key in params:
            # Q, K, V projections are concatenated in GPT-2
            # Format is [Q1, Q2, ..., K1, K2, ..., V1, V2, ...]
            # Zero out the query projection
            params[attn_key]["kernel"] = params[attn_key]["kernel"].at[head_offset:head_offset+self.head_dim].set(0)
            params[attn_key]["bias"] = params[attn_key]["bias"].at[head_offset:head_offset+self.head_dim].set(0)
            
            # Zero out the key projection
            key_offset = self.num_heads * self.head_dim + head_offset
            params[attn_key]["kernel"] = params[attn_key]["kernel"].at[key_offset:key_offset+self.head_dim].set(0)
            params[attn_key]["bias"] = params[attn_key]["bias"].at[key_offset:key_offset+self.head_dim].set(0)
            
            # Zero out the value projection
            val_offset = 2 * self.num_heads * self.head_dim + head_offset
            params[attn_key]["kernel"] = params[attn_key]["kernel"].at[val_offset:val_offset+self.head_dim].set(0)
            params[attn_key]["bias"] = params[attn_key]["bias"].at[val_offset:val_offset+self.head_dim].set(0)
            
        # Also prune output projection
        out_key = f"transformer/h/{layer}/attn/c_proj"
        if out_key in params:
            # Output projection is also affected
            params[out_key]["kernel"] = params[out_key]["kernel"].at[:, head_offset:head_offset+self.head_dim].set(0)
            
        return params
        
    def evaluate_pruning(self, pruned_params, prompt="The quick brown fox jumps over the lazy dog"):
        """Evaluate the pruned model on a simple prompt."""
        print(f"Evaluating pruned model on prompt: {prompt}")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="jax")
        
        # Get original model output
        original_output = self.model(**inputs, params=self.original_params).logits
        
        # Get pruned model output
        pruned_output = self.model(**inputs, params=pruned_params).logits
        
        # Calculate difference
        output_diff = jnp.abs(original_output - pruned_output).mean().item()
        print(f"Average output difference: {output_diff}")
        
        return output_diff