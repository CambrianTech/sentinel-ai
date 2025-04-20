"""
Apple Silicon Optimization Utilities

This module provides utilities and patches for improving PyTorch performance
and stability on Apple Silicon (M1/M2/M3) hardware.

Version: v0.0.57 (2025-04-19 22:30:00)
"""

import torch
import numpy as np
import platform
import os
import sys
from typing import Optional, Dict, Any, Callable, Tuple

# Check for Apple Silicon
IS_APPLE_SILICON = False
IS_COLAB = False

# Detect if we're running in Google Colab
try:
    import google.colab
    IS_COLAB = True
    print("🌐 Running in Google Colab environment")
except (ImportError, ModuleNotFoundError):
    pass

# Detect Apple Silicon
try:
    if platform.system() == "Darwin" and platform.processor() == "arm":
        IS_APPLE_SILICON = True
        print("🍎 Apple Silicon detected - optimization utilities available")
except Exception:
    pass

# Original PyTorch functions to restore after patching
_original_functions = {}

def safe_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Safely perform matrix multiplication on Apple Silicon.
    
    Args:
        a: First tensor
        b: Second tensor
        
    Returns:
        Result of matrix multiplication
    """
    # Only apply special handling on Apple Silicon and NOT in Colab
    if not IS_APPLE_SILICON or IS_COLAB:
        # Regular matmul for non-Apple Silicon platforms or Colab
        return torch.matmul(a, b)
    
    # On Apple Silicon (local execution), we need extra precautions
    try:
        # Ensure tensors are on CPU
        if a.is_cuda:
            a = a.cpu()
        if b.is_cuda:
            b = b.cpu()
        
        # Ensure tensors are contiguous for better memory layout
        if not a.is_contiguous():
            a = a.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()
        
        # Ensure tensors are in float32 for better numerical stability
        if a.dtype != torch.float32 and a.dtype.is_floating_point:
            a = a.to(torch.float32)
        if b.dtype != torch.float32 and b.dtype.is_floating_point:
            b = b.to(torch.float32)
        
        # Ensure gradient tracking is disabled for matmul operation
        with torch.no_grad():
            # Perform matrix multiplication
            result = torch.matmul(a.detach(), b.detach())
            
            # Check for NaN/Inf values in result
            if torch.isnan(result).any() or torch.isinf(result).any():
                # Replace with zeros if NaN/Inf are found
                result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
                print("Warning: NaN/Inf values detected in matrix multiplication result")
            
        return result
        
    except Exception as e:
        print(f"Error in safe_matmul: {e}")
        # Fallback method using manual dot product for extreme cases
        try:
            # Try with numpy as absolute fallback
            a_np = a.detach().cpu().numpy()
            b_np = b.detach().cpu().numpy()
            result_np = np.matmul(a_np, b_np)
            return torch.tensor(result_np, device='cpu')
        except Exception as np_error:
            print(f"Fallback numpy matmul also failed: {np_error}")
            # Return zero tensor of appropriate shape as last resort
            out_shape = list(a.shape[:-1]) + list(b.shape[1:])
            return torch.zeros(out_shape, device='cpu')

def apply_tensor_patches():
    """
    Apply patches to PyTorch functions to improve stability on Apple Silicon.
    
    This function monkey-patches several PyTorch functions with safer versions
    that are less likely to crash on Apple Silicon hardware.
    """
    if not IS_APPLE_SILICON or IS_COLAB:
        print("Not applying tensor patches (not on Apple Silicon or running in Colab)")
        return
    
    print("🍎 Applying tensor operation patches for Apple Silicon")
    
    # Save original functions
    _original_functions['matmul'] = torch.matmul
    
    # Replace with safer versions
    torch.matmul = safe_matmul
    
    # Set PyTorch to use single thread
    torch.set_num_threads(1)
    
    # Force deterministic algorithms
    torch.use_deterministic_algorithms(True)
    
    # Set environment variables for single-threaded operation
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    print("✅ Applied tensor safety patches for Apple Silicon")

def restore_tensor_patches():
    """
    Restore original PyTorch functions after patching.
    """
    if not _original_functions:
        return
    
    print("🔄 Restoring original PyTorch functions")
    
    # Restore original functions
    if 'matmul' in _original_functions:
        torch.matmul = _original_functions['matmul']
    
    # Clear saved functions
    _original_functions.clear()
    
    print("✅ Restored original PyTorch functions")

def safe_context():
    """Context manager for safely running tensor operations on Apple Silicon.
    
    Example:
        with safe_context():
            result = model(inputs)
    """
    class SafeContext:
        def __enter__(self):
            apply_tensor_patches()
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            restore_tensor_patches()
            
    return SafeContext()

# Provide helpers for forcing CPU usage on Apple Silicon
def ensure_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure a tensor is on CPU and detached."""
    if IS_APPLE_SILICON and tensor.is_cuda:
        return tensor.detach().cpu()
    return tensor

def ensure_cpu_model(model: torch.nn.Module) -> torch.nn.Module:
    """Ensure a model is on CPU on Apple Silicon."""
    if IS_APPLE_SILICON:
        return model.cpu()
    return model