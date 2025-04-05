"""
Fixed Pruning Module for Sentinel-AI

This module provides a consistent interface for pruning attention heads across
different model architectures. It implements core operations for pruning and 
parameter manipulation while ensuring performance and stability.

This module is maintained for backward compatibility.
New code should import from sentinel.pruning.fixed_pruning_module instead.
"""

import os
import torch
import numpy as np
import warnings
from typing import Dict, List, Tuple, Any, Optional

# Emit deprecation warning
warnings.warn(
    "The utils.pruning.fixed_pruning_module module is deprecated. "
    "Please use sentinel.pruning.fixed_pruning_module instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from the new location for backward compatibility
try:
    from sentinel.pruning.fixed_pruning_module import FixedPruningModule
except ImportError:
    # If the import fails, define the class here for backward compatibility
    # This allows existing code to continue working even if the new module isn't available
    class FixedPruningModule:
        """
        Core module for pruning attention heads in transformer-based models.
        
        This class provides a fixed (stable) interface for pruning operations across
        different model architectures, including GPT-2, BLOOM, OPT, Pythia, and Llama.
    """
    
    def __init__(
        self, 
        model_name: str,
        device: str = None,
        debug: bool = False,
        quiet: bool = False
    ):
        """
        Initialize the pruning module.
        
        Args:
            model_name: Name or path of HuggingFace model
            device: Device to use ('cuda' or 'cpu')
            debug: Enable debug output
            quiet: Reduce verbose output
        """
        self.model_name = model_name
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.debug = debug
        self.quiet = quiet
        
        # Fields to be initialized during model loading
        self.model = None
        self.tokenizer = None
        self.num_layers = 0
        self.num_heads = 0
        
    def load_model(self) -> bool:
        """
        Load the model and tokenizer.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Import loaders from sentinel-ai
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            
            from models.loaders.loader import load_baseline_model, load_adaptive_model
            
            # Load tokenizer
            if "llama" in self.model_name.lower():
                from transformers import LlamaTokenizer
                self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
            else:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load baseline model
            if not self.quiet:
                print(f"Loading baseline model: {self.model_name}")
                
            baseline_model = load_baseline_model(self.model_name, self.device)
            
            # Convert to adaptive model
            if not self.quiet:
                print(f"Converting to adaptive model")
                
            adaptive_model = load_adaptive_model(
                self.model_name, 
                baseline_model, 
                self.device, 
                debug=self.debug,
                quiet=self.quiet
            )
            
            # Store model
            self.model = adaptive_model
            
            # Initialize model parameters
            self.params = {
                "blocks": []
            }
            
            # Extract model dimensions
            if hasattr(adaptive_model, "blocks"):
                self.num_layers = len(adaptive_model.blocks)
                self.num_heads = adaptive_model.blocks[0]["attn"].num_heads
                
                # Create a parameter dict structure compatible with growth and pruning operations
                for layer_idx in range(self.num_layers):
                    block_params = {"attn": {}}
                    
                    # Get gate values
                    gate_values = adaptive_model.blocks[layer_idx]["attn"].gate.detach().clone()
                    block_params["attn"]["gate"] = gate_values
                    
                    # Add to params
                    self.params["blocks"].append(block_params)
                
                if not self.quiet:
                    print(f"Model structure: {self.num_layers} layers, {self.num_heads} heads per layer")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def prune_head(self, params: Any, layer_idx: int, head_idx: int) -> Any:
        """
        Prune a specific attention head.
        
        Args:
            params: Model parameters
            layer_idx: Layer index
            head_idx: Head index
            
        Returns:
            Updated model parameters
        """
        # Make a deep copy of parameters
        import copy
        pruned_params = copy.deepcopy(params)
        
        # Set gate value to near-zero (effectively pruning the head)
        with torch.no_grad():
            # Update gate value in the parameter tree
            if layer_idx < len(pruned_params["blocks"]):
                block_params = pruned_params["blocks"][layer_idx]
                if "attn" in block_params:
                    attn_params = block_params["attn"]
                    
                    if "gate" in attn_params:
                        # Update tensor directly
                        attn_params["gate"][head_idx] = 0.001
                        
                        # Also update the model directly
                        if hasattr(self.model, "blocks"):
                            self.model.blocks[layer_idx]["attn"].gate[head_idx] = 0.001
            
        return pruned_params
    
    def evaluate_perplexity(self, params: Any, text: str) -> float:
        """
        Evaluate perplexity of the model on a text sample.
        
        Args:
            params: Model parameters (not used directly - gate values are updated in the model)
            text: Text to evaluate
            
        Returns:
            Perplexity value
        """
        try:
            # Tokenize text
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            # Forward pass
            outputs = self.model(inputs.input_ids)
            
            # Get logits
            logits = outputs
            
            # Shift for next token prediction
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = inputs.input_ids[:, 1:].contiguous()
            
            # Calculate loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Perplexity = exp(loss)
            perplexity = torch.exp(loss).item()
            
            return perplexity
            
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            return float('nan')
    
    def generate_text(self, params: Any, prompt: str, max_length: int = 100) -> str:
        """
        Generate text using the model.
        
        Args:
            params: Model parameters (not used directly - gate values are already updated in the model)
            prompt: Text prompt
            max_length: Maximum length of generated text
            
        Returns:
            Generated text
        """
        try:
            # Tokenize prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode outputs
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return generated_text
            
        except Exception as e:
            print(f"Error generating text: {e}")
            return f"[Generation error: {str(e)}]"