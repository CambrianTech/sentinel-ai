"""
Training stability utilities for language models
"""

import torch
import numpy as np
import logging
from typing import Dict, Optional, List, Union, Tuple
import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def check_nan_gradients(model) -> bool:
    """
    Check if model has NaN gradients.
    
    Args:
        model: The model to check
        
    Returns:
        True if NaN gradients are found
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                logger.warning(f"NaN gradient detected in {name}")
                return True
    return False

def check_nan_loss(loss) -> bool:
    """
    Check if loss value is NaN.
    
    Args:
        loss: Loss value
        
    Returns:
        True if loss is NaN
    """
    if isinstance(loss, torch.Tensor):
        return torch.isnan(loss).any().item()
    return math.isnan(loss)

def reset_nan_grads(model):
    """
    Set NaN gradients to zero to prevent optimizer failure.
    
    Args:
        model: The model to fix
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            nan_mask = torch.isnan(param.grad)
            if nan_mask.any():
                param.grad[nan_mask] = 0
                logger.info(f"Reset NaN gradients in {name}")

def gradient_clipping(model, max_grad_norm: float = 1.0):
    """
    Clips gradients to prevent explosion.
    
    Args:
        model: The model
        max_grad_norm: Maximum allowed gradient norm
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

def convert_model_to_fp32(model):
    """
    Convert model to float32 if it was loaded in lower precision.
    This helps prevent numerical instability during training.
    
    Args:
        model: The model to convert
        
    Returns:
        Model with weights in float32
    """
    for param in model.parameters():
        if param.data.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
    
    return model

def mixed_precision_setup(enable_amp: bool = True):
    """
    Set up mixed precision training.
    
    Args:
        enable_amp: Whether to enable automatic mixed precision
        
    Returns:
        scaler: Gradient scaler for AMP
    """
    if enable_amp and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    return scaler

def reduce_batch_size_for_nan(batch_size: int) -> int:
    """
    Reduce batch size to recover from NaN loss.
    
    Args:
        batch_size: Current batch size
        
    Returns:
        New, reduced batch size
    """
    new_batch_size = max(1, batch_size // 2)
    if new_batch_size != batch_size:
        logger.info(f"Reducing batch size from {batch_size} to {new_batch_size} to recover from NaN")
    return new_batch_size

def reduce_learning_rate_for_nan(learning_rate: float) -> float:
    """
    Reduce learning rate to recover from NaN loss.
    
    Args:
        learning_rate: Current learning rate
        
    Returns:
        New, reduced learning rate
    """
    new_lr = learning_rate * 0.1
    logger.info(f"Reducing learning rate from {learning_rate} to {new_lr} to recover from NaN")
    return new_lr

def initialize_optimizer_with_zero_grad(optimizer):
    """
    Initialize optimizer with zero gradients for stability.
    
    Args:
        optimizer: The optimizer to initialize
    """
    optimizer.zero_grad()

def update_optimizer_for_fp32(optimizer):
    """
    Update optimizer state to handle fp32 parameters.
    
    Args:
        optimizer: The optimizer
    """
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param.requires_grad:
                if optimizer.state.get(param) is not None:
                    # Update stored parameter type in optimizer state
                    for state_key in optimizer.state[param]:
                        if torch.is_tensor(optimizer.state[param][state_key]):
                            optimizer.state[param][state_key] = optimizer.state[param][state_key].to(
                                dtype=param.data.dtype,
                                device=param.data.device
                            )

def log_memory_usage(phase: str = ""):
    """
    Log current GPU memory usage.
    
    Args:
        phase: Name of the current phase for logging
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        
        if phase:
            logger.info(f"Memory usage ({phase}): Allocated: {allocated:.2f}GB, "
                      f"Max Allocated: {max_allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        else:
            logger.info(f"Memory usage: Allocated: {allocated:.2f}GB, "
                      f"Max Allocated: {max_allocated:.2f}GB, Reserved: {reserved:.2f}GB")

def get_model_memory_usage(model, batch_size: int = 1, sequence_length: int = 512) -> Dict[str, float]:
    """
    Estimate model memory usage during training.
    
    Args:
        model: The model
        batch_size: Batch size
        sequence_length: Length of input sequences
        
    Returns:
        Dictionary with memory usage in GB: 
        {
            "parameters": Parameters memory,
            "activations": Estimated activation memory,
            "total": Total estimated usage
        }
    """
    # Count parameters
    params_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 3)
    
    # Estimate activation memory (rule of thumb: 2-4x parameter memory for transformer models)
    # This is a simplification - actual usage depends on model architecture and implementation
    activation_multiplier = 3.0  # Conservative middle estimate
    activations_mem = params_mem * activation_multiplier * batch_size * (sequence_length / 512)
    
    # Add optimizer states - AdamW uses 8 bytes per parameter (2 states)
    optimizer_mem = 2 * 4 * sum(p.numel() for p in model.parameters()) / (1024 ** 3)
    
    # Total estimate
    total = params_mem + activations_mem + optimizer_mem
    
    return {
        "parameters": params_mem,
        "activations": activations_mem,
        "optimizer": optimizer_mem,
        "total": total
    }

def is_model_overflowing(model, batch_size: int, sequence_length: int, max_memory_gb: float):
    """
    Check if model is likely to overflow given memory constraints.
    
    Args:
        model: The model to check
        batch_size: Batch size
        sequence_length: Sequence length
        max_memory_gb: Maximum available memory in GB
        
    Returns:
        Tuple of (is_overflowing, memory_estimate)
    """
    estimate = get_model_memory_usage(model, batch_size, sequence_length)
    return estimate["total"] > max_memory_gb, estimate

def enable_tf32():
    """
    Enable TF32 for faster operations on A100 GPUs while maintaining precision.
    This is a good compromise between FP32 and FP16.
    """
    if torch.cuda.is_available():
        # Check if CUDA version supports TF32
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:  # A100 and newer GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 precision enabled for faster matrix operations")

def find_stable_batch_size(model, max_batch_size: int, sequence_length: int, 
                          device: torch.device, max_memory_gb: float) -> int:
    """
    Find a stable batch size that won't cause OOM errors.
    
    Args:
        model: The model
        max_batch_size: Maximum batch size to try
        sequence_length: Input sequence length
        device: The device to use
        max_memory_gb: Maximum available memory in GB
        
    Returns:
        Stable batch size
    """
    # Start with binary search
    low, high = 1, max_batch_size
    current_bs = min(16, max_batch_size)  # Start with reasonable default
    
    memory_estimates = {}
    
    while low <= high:
        # Check if current batch size would overflow
        is_overflow, estimate = is_model_overflowing(model, current_bs, sequence_length, max_memory_gb)
        memory_estimates[current_bs] = estimate
        
        if is_overflow:
            # Too large, try smaller
            high = current_bs - 1
            current_bs = (low + high) // 2
        else:
            # Try larger
            low = current_bs + 1
            current_bs = (low + high) // 2
            
        # Stop if we've converged
        if low > high:
            current_bs = high  # Use the largest non-overflowing value
    
    # Apply a safety factor to avoid edge cases
    safe_bs = max(1, int(current_bs * 0.9))
    
    logger.info(f"Determined stable batch size: {safe_bs} (from max: {max_batch_size})")
    return safe_bs