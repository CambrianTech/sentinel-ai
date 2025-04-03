"""
Memory management utilities for stable model training.

This module provides functions to manage memory efficiently when training large
language models, especially in constrained environments like Google Colab.
"""

import logging
import math
from typing import Dict, Any, Optional, Tuple, Union

# Set up logger
logger = logging.getLogger(__name__)

def estimate_model_memory(
    model_name: str, 
    sequence_length: int = 128, 
    batch_size: int = 4
) -> Dict[str, float]:
    """
    Estimate memory requirements for a model during training.
    
    Args:
        model_name: Name or path of the model
        sequence_length: Length of input sequences
        batch_size: Batch size for training
        
    Returns:
        Dict with estimated memory requirements in GB
    """
    # Model sizes in millions of parameters
    model_sizes = {
        "gpt2": 124,
        "gpt2-medium": 355,
        "gpt2-large": 774,
        "gpt2-xl": 1558,
        "distilgpt2": 82,
        "facebook/opt-125m": 125,
        "facebook/opt-350m": 350,
        "facebook/opt-1.3b": 1300,
        "facebook/opt-2.7b": 2700,
        "facebook/opt-6.7b": 6700,
        "EleutherAI/pythia-70m": 70,
        "EleutherAI/pythia-160m": 160,
        "EleutherAI/pythia-410m": 410,
        "EleutherAI/pythia-1b": 1000,
        "EleutherAI/pythia-1.4b": 1400,
    }
    
    # Extract model size
    model_size_millions = None
    
    # Check for exact match
    if model_name in model_sizes:
        model_size_millions = model_sizes[model_name]
    else:
        # Try to find by partial match
        for key, size in model_sizes.items():
            if key in model_name:
                model_size_millions = size
                break
        
        # If still not found, try to extract from common patterns
        if model_size_millions is None:
            if "125m" in model_name.lower():
                model_size_millions = 125
            elif "350m" in model_name.lower():
                model_size_millions = 350
            elif "1.3b" in model_name.lower() or "1b" in model_name.lower():
                model_size_millions = 1300
            elif "2.7b" in model_name.lower() or "2b" in model_name.lower():
                model_size_millions = 2700
            elif "6.7b" in model_name.lower() or "6b" in model_name.lower():
                model_size_millions = 6700
    
    # Default to a medium-sized model if unknown
    if model_size_millions is None:
        logger.warning(f"Unknown model size for {model_name}, defaulting to 350M parameters")
        model_size_millions = 350
    
    # Calculate memory in GB (parameters * 4 bytes for float32)
    params_gb = (model_size_millions * 1000000 * 4) / (1024**3)
    
    # Estimate activation memory (varies by model architecture)
    # Sequence length increases activation memory linearly 
    activation_multiplier = 3.0
    seq_length_factor = sequence_length / 512
    activations_gb = params_gb * activation_multiplier * batch_size * seq_length_factor
    
    # Optimizer states (typically 2x model size for Adam)
    optimizer_gb = 2 * params_gb
    
    # Gradient accumulation (typically 1x model size)
    gradients_gb = params_gb
    
    # Extra for workspace, cached computations, etc.
    workspace_gb = 1.0 + (model_size_millions / 1000)
    
    # Total estimate
    total_gb = params_gb + activations_gb + optimizer_gb + gradients_gb + workspace_gb
    
    return {
        "parameters_gb": params_gb,
        "activations_gb": activations_gb,
        "optimizer_gb": optimizer_gb,
        "gradients_gb": gradients_gb,
        "workspace_gb": workspace_gb,
        "total_gb": total_gb,
        "model_size_millions": model_size_millions
    }

def recommend_batch_size(
    model_name: str, 
    available_memory_gb: float,
    sequence_length: int = 128,
    min_batch_size: int = 1
) -> int:
    """
    Recommend a suitable batch size given available memory constraints.
    
    Args:
        model_name: Name or path of the model
        available_memory_gb: Available GPU memory in GB
        sequence_length: Length of input sequences
        min_batch_size: Minimum allowable batch size
        
    Returns:
        Recommended batch size
    """
    # Start with batch size 16 and work down
    batch_size = 16
    
    while batch_size >= min_batch_size:
        # Estimate memory for this batch size
        memory_estimate = estimate_model_memory(
            model_name,
            sequence_length,
            batch_size
        )
        
        # If it fits, return this batch size
        if memory_estimate["total_gb"] <= available_memory_gb * 0.9:  # 90% of available
            return batch_size
        
        # Try a smaller batch size
        batch_size = batch_size // 2
    
    # If even the minimum batch size doesn't fit, return it anyway
    return min_batch_size

def recommend_sequence_length(
    model_name: str,
    available_memory_gb: float,
    batch_size: int = 4,
    min_length: int = 64
) -> int:
    """
    Recommend a suitable sequence length given memory constraints.
    
    Args:
        model_name: Name or path of the model
        available_memory_gb: Available GPU memory in GB
        batch_size: Batch size for training
        min_length: Minimum allowable sequence length
        
    Returns:
        Recommended sequence length
    """
    # Start with a generous sequence length and work down
    sequence_length = 512
    
    while sequence_length >= min_length:
        # Estimate memory for this sequence length
        memory_estimate = estimate_model_memory(
            model_name,
            sequence_length,
            batch_size
        )
        
        # If it fits, return this sequence length
        if memory_estimate["total_gb"] <= available_memory_gb * 0.9:  # 90% of available
            return sequence_length
        
        # Try a smaller sequence length
        sequence_length = sequence_length // 2
    
    # If even the minimum sequence length doesn't fit, return it anyway
    return min_length

def get_default_gpu_memory() -> float:
    """
    Get the default GPU memory for common environments.
    
    Returns:
        Estimated GPU memory in GB
    """
    try:
        # Try to detect GPU memory with torch
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            memory_bytes = torch.cuda.get_device_properties(device).total_memory
            return memory_bytes / (1024**3)  # Convert to GB
    except:
        pass
        
    try:
        # Try with JAX
        import jax
        devices = jax.devices()
        if devices and hasattr(devices[0], 'memory_stats'):
            memory_stats = devices[0].memory_stats()
            if 'bytes_limit' in memory_stats:
                return memory_stats['bytes_limit'] / (1024**3)
    except:
        pass
    
    try:
        # Try with NVIDIA management library
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return memory_info.total / (1024**3)
    except:
        pass
    
    # Default values for common environments
    try:
        import os
        if 'COLAB_GPU' in os.environ:
            # Google Colab with T4 GPU typically has 16GB
            return 15.0
    except:
        pass
        
    # Conservative default if detection fails
    return 8.0

def optimize_training_parameters(
    model_name: str,
    available_memory_gb: Optional[float] = None
) -> Dict[str, Any]:
    """
    Get optimized training parameters based on model and available memory.
    
    Args:
        model_name: Name or path of the model
        available_memory_gb: Available GPU memory in GB, auto-detected if None
        
    Returns:
        Dict with optimized training parameters
    """
    if available_memory_gb is None:
        available_memory_gb = get_default_gpu_memory()
        
    logger.info(f"Optimizing training parameters for {model_name} with {available_memory_gb:.1f}GB memory")
    
    # Detect model size
    memory_estimate = estimate_model_memory(model_name, 128, 1)
    model_size_m = memory_estimate["model_size_millions"]
    
    # Set batch size based on model size and available memory
    batch_size = recommend_batch_size(model_name, available_memory_gb)
    
    # Set sequence length based on model size and available memory
    sequence_length = recommend_sequence_length(model_name, available_memory_gb, batch_size)
    
    # Determine gradient accumulation steps
    gradient_accumulation_steps = max(1, 16 // batch_size)
    
    # Scale learning rate based on model size
    if model_size_m < 200:
        learning_rate = 5e-5
    elif model_size_m < 800:
        learning_rate = 3e-5
    else:
        learning_rate = 1e-5
        
    # Scale weight decay based on model size
    if model_size_m < 500:
        weight_decay = 0.01
    else:
        weight_decay = 0.1
        
    # Scale optimizer steps
    warmup_steps = max(100, int(500 * (model_size_m / 1000)))
    
    return {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "memory_estimate_gb": memory_estimate["total_gb"],
        "available_memory_gb": available_memory_gb,
        "model_size_millions": model_size_m
    }

def optimize_fine_tuner(fine_tuner, model_name: str, gpu_memory_gb: Optional[float] = None):
    """
    Optimize a fine-tuner with memory-efficient settings.
    
    Args:
        fine_tuner: The fine-tuner instance to optimize
        model_name: Name or path of the model
        gpu_memory_gb: Available GPU memory in GB, auto-detected if None
        
    Returns:
        Optimized fine-tuner instance
    """
    # Get optimized parameters
    params = optimize_training_parameters(model_name, gpu_memory_gb)
    
    # Apply parameters to fine-tuner
    if hasattr(fine_tuner, 'batch_size'):
        old_batch_size = fine_tuner.batch_size
        fine_tuner.batch_size = params["batch_size"]
        logger.info(f"Optimized batch size: {old_batch_size} → {fine_tuner.batch_size}")
        
    if hasattr(fine_tuner, 'max_seq_length'):
        old_seq_length = fine_tuner.max_seq_length
        fine_tuner.max_seq_length = params["sequence_length"]
        logger.info(f"Optimized sequence length: {old_seq_length} → {fine_tuner.max_seq_length}")
    
    # For XL models or large models, force synthetic data with small batch size
    model_name_lower = model_name.lower()
    is_xl_model = ("xl" in model_name_lower or 
                  "1.3b" in model_name_lower or 
                  "2.7b" in model_name_lower or
                  "large" in model_name_lower or
                  "1b" in model_name_lower)
    
    # Force synthetic data for large models
    if is_xl_model:
        # Force smallest possible batch size for large models
        fine_tuner.batch_size = 1
        fine_tuner.max_seq_length = min(64, fine_tuner.max_seq_length)
        
        # Add flag for synthetic data if supported
        if hasattr(fine_tuner, 'use_synthetic_data'):
            fine_tuner.use_synthetic_data = True
        
        logger.info(f"Using minimal parameters for large model {model_name}")
    
    # Add dropout RNG key handling if supported
    if hasattr(fine_tuner, 'use_rng_keys_for_dropout'):
        fine_tuner.use_rng_keys_for_dropout = True
        logger.info("Enabled RNG keys for dropout layers")
        
    return fine_tuner


if __name__ == "__main__":
    # Enable logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test memory estimation for different models
    for model in ["gpt2", "gpt2-medium", "gpt2-large", "facebook/opt-125m", "facebook/opt-1.3b"]:
        print(f"\nMemory estimates for {model}:")
        for batch_size in [1, 2, 4, 8]:
            estimate = estimate_model_memory(model, 128, batch_size)
            print(f"  Batch size {batch_size}: {estimate['total_gb']:.2f}GB")
            
        # Get optimized parameters
        gpu_mem = 16.0  # Assume 16GB GPU
        params = optimize_training_parameters(model, gpu_mem)
        print(f"Optimized parameters for {gpu_mem}GB GPU:")
        for k, v in params.items():
            print(f"  {k}: {v}")