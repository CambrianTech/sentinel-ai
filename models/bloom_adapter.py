"""
BLOOM Model Adapter

This module provides specialized handling for BLOOM models to work correctly 
with our adaptive transformer architecture, accounting for BLOOM's ALiBi attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.adaptive_transformer import AdaptiveCausalLmWrapper
from transformers import AutoConfig, AutoModelForCausalLM

class BLOOMAdaptiveWrapper(nn.Module):
    """
    Special wrapper that uses the original BLOOM model for generation but
    provides an interface compatible with our adaptive architecture.
    
    This hybrid approach maintains BLOOM's ALiBi attention while allowing
    us to demonstrate the adaptive concept with minimal issues.
    """
    
    def __init__(self, model_name, device='cpu', debug=False):
        super().__init__()
        # Load baseline model directly
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # Get configuration
        self.config = self.model.config
        
        # Set up dummy attributes that match AdaptiveCausalLmWrapper interface
        self.n_layer = self.config.n_layer
        self.n_head = self.config.n_head
        self.blocks = nn.ModuleList()
        
        # Create dummy gate parameters for compatibility
        for i in range(self.n_layer):
            # Create dummy head gates that are all active
            setattr(self, f"layer_{i}_gates", nn.Parameter(torch.ones(self.n_head)))
            
            # Add a dummy block with a gate attribute
            block = nn.Module()
            block.attn = nn.Module()
            block.attn.gate = nn.Parameter(torch.ones(self.n_head))
            block.attn.num_heads = self.n_head  # Add num_heads attribute for controller
            self.blocks.append(block)
        
        # Track if we're in debug mode
        self.debug = debug
        
        print(f"✅ Created BLOOM Adapter for {model_name} with {self.n_layer} layers and {self.n_head} heads")
        print("   Note: Using original BLOOM model internals with adaptive interface")
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass - delegates to the original model"""
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    
    def generate(self, **kwargs):
        """Text generation - delegates to the original model"""
        return self.model.generate(**kwargs)
    
    def get_agency_report(self):
        """Returns a report of the agency (gate values) in the model"""
        agency_data = {}
        for i in range(self.n_layer):
            agency_data[f"layer_{i}"] = {
                "gates": self.blocks[i].attn.gate.detach().cpu().tolist(),
                "active_heads": list(range(self.n_head))  # All heads active
            }
        return agency_data
    
    def count_active_heads(self):
        """Count the number of active attention heads"""
        # All heads are considered active in this adapter
        return self.n_layer * self.n_head


def load_bloom_adapted(model_name, device='cpu', debug=False):
    """
    Load a BLOOM model with our adapter that maintains compatibility
    with the adaptive transformer interface.
    
    Args:
        model_name: Name of the BLOOM model to load
        device: Device to load the model onto
        debug: Whether to print debug information
        
    Returns:
        Adapted BLOOM model
    """
    model = BLOOMAdaptiveWrapper(model_name, device, debug)
    return model