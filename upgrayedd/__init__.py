"""
Upgrayedd Module - Transform static models into adaptive, self-optimizing networks

The Upgrayedd system allows any HuggingFace transformer model to be transformed
into a self-optimizing neural network through:

1. Controller-guided pruning
2. Strategic regrowth
3. Differential learning rates
4. Compression options

Named after the great Upgrayedd from Idiocracy:
"Spelled with two D's for a double dose of adaptive optimization."
"""

__version__ = "0.1.0"

from upgrayedd.core import (
    UpgrayeddPipeline,
    transform_model,
    evaluate_model,
    compress_model
)

from upgrayedd.config import (
    UpgrayeddConfig,
    load_config,
    save_config
)

__all__ = [
    "UpgrayeddPipeline",
    "transform_model",
    "evaluate_model",
    "compress_model",
    "UpgrayeddConfig", 
    "load_config",
    "save_config"
]