"""
Model Zoo - Curated pre-trained models for Continuum PersonaUsers.

Provides easy access to production-ready models optimized for different use cases.
"""
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

# Curated models for Continuum
CONTINUUM_MODELS = {
    # Tier 1: Small, fast, consumer-friendly
    'tinyllama-chat': {
        'hf_id': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'size': '1.1B',
        'layers': 22,
        'heads': 32,
        'context': 2048,
        'ram_requirement_gb': 4,
        'inference_speed': 'fast',
        'use_cases': ['general chat', 'personal assistant', 'quick tasks'],
        'description': 'Fast 1.1B chat model, perfect for personal assistants'
    },

    'phi-2': {
        'hf_id': 'microsoft/phi-2',
        'size': '2.7B',
        'layers': 32,
        'heads': 32,
        'context': 2048,
        'ram_requirement_gb': 8,
        'inference_speed': 'fast',
        'use_cases': ['reasoning', 'math', 'code understanding'],
        'description': 'Microsoft Phi-2, strong reasoning and code skills'
    },

    # Tier 2: Code-specialized
    'codellama-7b': {
        'hf_id': 'codellama/CodeLlama-7b-Instruct-hf',
        'size': '7B',
        'layers': 32,
        'heads': 32,
        'context': 4096,
        'ram_requirement_gb': 16,
        'inference_speed': 'medium',
        'use_cases': ['code generation', 'debugging', 'architecture'],
        'description': 'CodeLlama 7B, specialized for software engineering'
    },

    # Testing: Tiny model for quick experiments
    'distilgpt2': {
        'hf_id': 'distilgpt2',
        'size': '82M',
        'layers': 6,
        'heads': 12,
        'context': 1024,
        'ram_requirement_gb': 1,
        'inference_speed': 'very fast',
        'use_cases': ['testing', 'development', 'prototyping'],
        'description': 'Tiny GPT-2 for testing adaptive features'
    }
}


def list_models() -> Dict[str, Dict[str, Any]]:
    """Get all available models with metadata."""
    return CONTINUUM_MODELS


def get_model_info(model_key: str) -> Dict[str, Any]:
    """Get metadata for a specific model."""
    if model_key not in CONTINUUM_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(CONTINUUM_MODELS.keys())}")
    return CONTINUUM_MODELS[model_key]


def load_base_model(
    model_key: str,
    device: str = 'cpu',
    torch_dtype: Optional[torch.dtype] = None
) -> tuple[AutoModelForCausalLM, AutoTokenizer, AutoConfig]:
    """
    Load a base pre-trained model from HuggingFace.

    Args:
        model_key: Model identifier from CONTINUUM_MODELS
        device: 'cpu' or 'cuda'
        torch_dtype: Optional dtype (e.g., torch.float16 for quantization)

    Returns:
        (model, tokenizer, config)

    Example:
        model, tokenizer, config = load_base_model('tinyllama-chat')
    """
    if model_key not in CONTINUUM_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(CONTINUUM_MODELS.keys())}")

    info = CONTINUUM_MODELS[model_key]
    hf_id = info['hf_id']

    print(f"Loading {info['description']}...")
    print(f"  HuggingFace ID: {hf_id}")
    print(f"  Size: {info['size']}")
    print(f"  RAM requirement: {info['ram_requirement_gb']}GB")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load config
    config = AutoConfig.from_pretrained(hf_id)

    # Load model
    load_kwargs = {'torch_dtype': torch_dtype} if torch_dtype else {}
    model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
    model = model.to(device)

    print(f"✅ Model loaded successfully!")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Device: {device}")

    return model, tokenizer, config


def load_adaptive_model(
    model_key: str,
    device: str = 'cpu',
    torch_dtype: Optional[torch.dtype] = None
) -> tuple:
    """
    Load a pre-trained model and convert to adaptive architecture.

    Args:
        model_key: Model identifier from CONTINUUM_MODELS
        device: 'cpu' or 'cuda'
        torch_dtype: Optional dtype

    Returns:
        (adaptive_model, tokenizer, config)

    Example:
        model, tokenizer, config = load_adaptive_model('tinyllama-chat')
    """
    from sentinel.models.loaders.adaptive_converter import convert_to_adaptive

    # Load base model
    base_model, tokenizer, config = load_base_model(model_key, device, torch_dtype)

    # Convert to adaptive
    print(f"\nConverting to adaptive architecture...")
    adaptive_model = convert_to_adaptive(
        base_model,
        config,
        device=device
    )

    return adaptive_model, tokenizer, config


# Convenience: Direct model loaders
def load_tinyllama(device: str = 'cpu'):
    """Load TinyLlama-1.1B-Chat with adaptive capabilities."""
    return load_adaptive_model('tinyllama-chat', device=device)


def load_phi2(device: str = 'cpu'):
    """Load Phi-2 with adaptive capabilities."""
    return load_adaptive_model('phi-2', device=device)


def load_codellama(device: str = 'cpu'):
    """Load CodeLlama-7B with adaptive capabilities."""
    return load_adaptive_model('codellama-7b', device=device)
