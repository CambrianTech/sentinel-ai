"""
Helper functions for Google Colab integration.

These utilities assist with:
- Setting up the Colab environment
- Detecting Colab hardware (GPU, TPU)
- Managing memory for optimal training performance
- Monitoring available resources
- Visualizing tensors safely across CPU/GPU environments
"""

import os
import sys
import warnings
import subprocess
import re
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Tuple, Union, Optional, List, Any

# Configure logging
logger = logging.getLogger(__name__)

def setup_colab_environment(
    prefer_gpu: bool = True,
    force_hardware_change: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Set up the Colab environment, configure hardware accelerator preferences,
    and return environment information.
    
    Args:
        prefer_gpu: Whether to prefer GPU over TPU if both are available
        force_hardware_change: Whether to force hardware accelerator change
        verbose: Whether to print status messages
    
    Returns:
        Dictionary with environment information
    """
    env_info = {
        "is_colab": False,
        "hardware": "CPU",
        "gpu_info": None,
        "memory_info": None
    }
    
    # Check if running in Colab
    try:
        from google.colab import _message
        env_info["is_colab"] = True
        
        if verbose:
            print("🔍 Detected Google Colab environment")
        
        # Try to set hardware accelerator
        if force_hardware_change or prefer_gpu:
            try:
                from google.colab import runtime
                if prefer_gpu:
                    if verbose:
                        print("🔄 Attempting to enable GPU acceleration...")
                    runtime.change_runtime(runtime_type="GPU")
                    if verbose:
                        print("✅ GPU acceleration enabled!")
                else:
                    if verbose:
                        print("🔄 Attempting to enable TPU acceleration...")
                    runtime.change_runtime(runtime_type="TPU")
                    if verbose:
                        print("✅ TPU acceleration enabled!")
            except Exception as e:
                if verbose:
                    print(f"⚠️ Could not change hardware accelerator: {e}")
                    print("Please set hardware accelerator manually: Runtime > Change runtime type")
        
        # Detect current hardware
        env_info.update(get_colab_type(verbose=verbose))
        
        # Check GPU status if available
        if env_info["hardware"] == "GPU":
            env_info.update(check_gpu_status(verbose=verbose))
    
    except ImportError:
        if verbose:
            print("This is not running in Google Colab")
    
    return env_info


def get_colab_type(verbose: bool = True) -> Dict[str, str]:
    """
    Detect the type of Colab environment (CPU, GPU, TPU)
    
    Args:
        verbose: Whether to print information
        
    Returns:
        Dictionary with hardware type
    """
    result = {"hardware": "CPU"}
    
    try:
        # Check for GPU
        gpu_info = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                 universal_newlines=True, check=False)
        
        if gpu_info.returncode == 0 and gpu_info.stdout.strip():
            gpu_name = gpu_info.stdout.strip()
            result["hardware"] = "GPU"
            result["gpu_name"] = gpu_name
            
            if verbose:
                print(f"✅ GPU detected: {gpu_name}")
            
            return result
            
        # Check for TPU
        try:
            import tensorflow as tf
            tpu_devices = tf.config.list_logical_devices('TPU')
            if tpu_devices:
                result["hardware"] = "TPU"
                result["tpu_devices"] = len(tpu_devices)
                
                if verbose:
                    print(f"✅ TPU detected with {len(tpu_devices)} cores")
                
                return result
        except:
            pass
            
        # No accelerator found
        if verbose:
            print("⚠️ No GPU or TPU detected. Using CPU only.")
            print("For faster execution, enable GPU: Runtime > Change runtime type > Hardware accelerator > GPU")
            
    except Exception as e:
        if verbose:
            print(f"Error detecting hardware accelerator: {e}")
    
    return result


def check_gpu_status(verbose: bool = True) -> Dict[str, Any]:
    """
    Check GPU status and memory availability
    
    Args:
        verbose: Whether to print status information
        
    Returns:
        Dictionary with GPU information
    """
    result = {
        "gpu_info": None,
        "memory_info": None,
        "has_sufficient_memory": False
    }
    
    try:
        # Run nvidia-smi
        if verbose:
            # Print full nvidia-smi output
            subprocess.run(["nvidia-smi"], check=False)
            print("\nMemory Information:")
        
        # Get detailed memory information
        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total,memory.free,memory.used', 
                                          '--format=csv,nounits,noheader'], 
                                         universal_newlines=True)
        
        vals = re.findall(r'\d+', gpu_info)
        if len(vals) >= 3:
            total, free, used = [int(x) for x in vals[:3]]
            
            memory_info = {
                "total_mb": total,
                "free_mb": free,
                "used_mb": used,
                "free_pct": free/total if total > 0 else 0
            }
            
            result["memory_info"] = memory_info
            result["has_sufficient_memory"] = free >= 3000  # Consider 3GB+ sufficient
            
            if verbose:
                print(f"Total Memory: {total}MB")
                print(f"Free Memory: {free}MB")
                print(f"Used Memory: {used}MB")
                print(f"Available Memory: {free/total:.1%}")
                
                if free < 3000:
                    print("⚠️ Warning: Low GPU memory available. Some large models may cause OOM errors.")
                else:
                    print("✅ Sufficient GPU memory available for most models.")
    
    except Exception as e:
        if verbose:
            print(f"Could not check GPU status: {e}")
    
    return result


def optimize_for_colab(
    model_size: str = "medium",
    available_memory_mb: Optional[int] = None,
    prefer_stability: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Get optimized parameters for training or inference in Colab
    based on available resources and model size.
    
    Args:
        model_size: Size category of the model ("tiny", "small", "medium", "large", "xl")
        available_memory_mb: Available GPU memory in MB (auto-detected if None)
        prefer_stability: Whether to prefer stability over speed
        verbose: Whether to print optimization details
        
    Returns:
        Dictionary with optimized parameters
    """
    # Default conservative parameters
    params = {
        "batch_size": 4,
        "sequence_length": 128,
        "gradient_accumulation_steps": 1,
        "use_fp16": False,
        "optimize_memory_usage": True, 
        "stability_level": 1,
        "apple_silicon_fix": False
    }
    
    # Check if running on Apple Silicon (M1/M2/M3)
    is_apple_silicon = False
    try:
        import platform
        if platform.system() == "Darwin" and platform.processor() == "arm":
            is_apple_silicon = True
            params["apple_silicon_fix"] = True
            if verbose:
                print("🍎 Detected Apple Silicon (M1/M2/M3) - Enabling special optimizations")
    except ImportError:
        pass
    
    # Get GPU info if not provided
    if available_memory_mb is None:
        gpu_info = check_gpu_status(verbose=False)
        if gpu_info["memory_info"]:
            available_memory_mb = gpu_info["memory_info"]["free_mb"]
        else:
            # No GPU or couldn't detect memory, use conservative defaults
            available_memory_mb = 2000
    
    # Adjust parameters based on model size
    size_factor_map = {
        "tiny": 0.5,    # e.g., DistilGPT2, smallest models
        "small": 1.0,   # e.g., GPT2, OPT-125M
        "medium": 2.0,  # e.g., GPT2-Medium, OPT-350M
        "large": 4.0,   # e.g., GPT2-Large, OPT-1.3B
        "xl": 8.0       # e.g., GPT2-XL, larger models
    }
    
    size_factor = size_factor_map.get(model_size.lower(), 1.0)
    memory_factor = max(0.1, min(1.0, available_memory_mb / 15000))
    
    # Calculate optimized parameters
    if prefer_stability:
        # More conservative settings
        params["batch_size"] = max(1, min(8, int(4 * memory_factor / size_factor)))
        params["sequence_length"] = max(32, min(128, int(128 * memory_factor / size_factor)))
        params["stability_level"] = 2 if size_factor >= 2.0 else 1
    else:
        # More aggressive settings
        params["batch_size"] = max(1, min(16, int(8 * memory_factor / size_factor)))
        params["sequence_length"] = max(32, min(256, int(192 * memory_factor / size_factor)))
        params["stability_level"] = 1
    
    # Enable mixed precision for medium+ models if enough memory
    if available_memory_mb > 6000 and size_factor >= 1.0:
        params["use_fp16"] = True
    
    # Apply gradient accumulation for very limited memory
    if available_memory_mb < 3000 or size_factor >= 4.0:
        params["gradient_accumulation_steps"] = max(2, int(size_factor))
    
    if verbose:
        print(f"📊 Colab optimization for {model_size} model with {available_memory_mb}MB memory:")
        print(f"  - Batch size: {params['batch_size']}")
        print(f"  - Sequence length: {params['sequence_length']}")
        print(f"  - Gradient accumulation steps: {params['gradient_accumulation_steps']}")
        print(f"  - Mixed precision (FP16): {params['use_fp16']}")
        print(f"  - Stability level: {params['stability_level']}")
    
    return params


def safe_tensor_imshow(
    tensor: Union[torch.Tensor, np.ndarray], 
    title: str = "Tensor Visualization",
    cmap: str = "viridis",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    show_colorbar: bool = True,
    vmin: Optional[float] = None, 
    vmax: Optional[float] = None
) -> plt.Figure:
    """
    Safely visualize a tensor with proper detach/cpu/numpy handling.
    
    Args:
        tensor: The tensor to visualize (can be on any device)
        title: Title for the plot
        cmap: Colormap to use
        save_path: Optional path to save the visualization 
                  (default: /tmp/tensor_viz_{timestamp}.png)
        figsize: Figure size as (width, height) in inches
        show_colorbar: Whether to show a colorbar
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling
        
    Returns:
        The matplotlib figure object
    """
    # Ensure tensor is properly converted for visualization
    if isinstance(tensor, torch.Tensor):
        # Handle GPU tensors or tensors with gradients
        if tensor.requires_grad:
            tensor = tensor.detach()
        if tensor.is_cuda:
            tensor = tensor.cpu()
        tensor_np = tensor.numpy()
    elif isinstance(tensor, np.ndarray):
        tensor_np = tensor
    else:
        raise TypeError(f"Expected torch.Tensor or numpy.ndarray, got {type(tensor)}")
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle potential NaN or Inf values
    if np.isnan(tensor_np).any() or np.isinf(tensor_np).any():
        print("⚠️ Warning: Tensor contains NaN or Inf values, replacing with zeros")
        tensor_np = np.nan_to_num(tensor_np, nan=0.0, posinf=1.0, neginf=0.0)
    
    im = ax.imshow(tensor_np, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add colorbar if requested
    if show_colorbar:
        fig.colorbar(im, ax=ax)
    
    # Add title
    ax.set_title(title)
    
    # Save if path provided, otherwise use default path
    if save_path is None:
        # Create timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"/tmp/tensor_viz_{timestamp}.png"
    
    # Save figure
    plt.savefig(save_path)
    
    # Display useful info
    print(f"Tensor shape: {tensor_np.shape}")
    print(f"Value range: [{tensor_np.min():.4f}, {tensor_np.max():.4f}]")
    print(f"Visualization saved to: {save_path}")
    
    return fig