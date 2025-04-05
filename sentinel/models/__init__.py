"""
Sentinel-AI Models Module

This module provides model implementations and loading utilities for 
running adaptive transformer models with various features:
- Sentinel gating mechanism for each attention head
- Support for dynamic pruning and regrowth
- U-Net style skip connections between layers
- Attention normalization to prevent exploding activations
- Attention head agency with consent tracking and state-aware computation
"""

# Import key models and utilities for easy access
from sentinel.models.adaptive_transformer import AdaptiveCausalLmWrapper
from sentinel.models.agency_specialization import AgencySpecialization
from sentinel.models.specialization_registry import SpecializationRegistry

# Import available loaders
try:
    from sentinel.models.loaders.gpt2_loader import load_gpt2_with_adaptive_transformer
except ImportError:
    pass