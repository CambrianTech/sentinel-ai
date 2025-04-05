"""
Sentinel-AI Model Loaders

This module provides loading utilities for adapting pre-trained models
to work with Sentinel-AI's adaptive features.
"""

from sentinel.models.loaders.loader import ModelLoader

# Import what's available
try:
    from sentinel.models.loaders.gpt2_loader import load_gpt2_with_adaptive_transformer
except ImportError:
    pass

try:
    from sentinel.models.loaders.bloom_loader import load_adaptive_model_bloom, load_bloom_with_adaptive_transformer
except ImportError:
    pass

try:
    from sentinel.models.loaders.opt_loader import load_opt_with_adaptive_transformer
except ImportError:
    pass

try:
    from sentinel.models.loaders.llama_loader import load_llama_with_adaptive_transformer
except ImportError:
    pass