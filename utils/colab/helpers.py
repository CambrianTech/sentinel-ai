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
        # First check PyTorch's CUDA availability for GPU
        gpu_available = False
        gpu_name = "Unknown GPU"
        gpu_memory = 0
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                if torch.cuda.device_count() > 0:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        except (ImportError, Exception) as e:
            if verbose:
                print(f"Error checking PyTorch CUDA: {e}")
        
        # Also check with nvidia-smi (more detailed)
        gpu_info = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                 universal_newlines=True, check=False)
        
        if gpu_info.returncode == 0 and gpu_info.stdout.strip():
            gpu_available = True
            gpu_info_parts = gpu_info.stdout.strip().split(',', 1)
            if len(gpu_info_parts) > 0:
                gpu_name = gpu_info_parts[0].strip()
                if len(gpu_info_parts) > 1:
                    # Extract memory information if available
                    try:
                        mem_str = gpu_info_parts[1].strip()
                        mem_value = float(re.search(r'(\d+\.?\d*)', mem_str).group(1))
                        mem_unit = re.search(r'([A-Za-z]+)', mem_str).group(1).upper()
                        
                        if mem_unit == 'MB':
                            gpu_memory = mem_value / 1024
                        elif mem_unit == 'GB':
                            gpu_memory = mem_value
                        else:
                            gpu_memory = 0
                    except (AttributeError, ValueError):
                        pass
        
        # Update result with GPU information if available
        if gpu_available:
            result["hardware"] = "GPU"
            result["gpu_name"] = gpu_name
            result["gpu_memory_gb"] = gpu_memory
            
            if verbose:
                print(f"✅ GPU detected: {gpu_name}")
                if gpu_memory > 0:
                    print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
                
                # Try to show more detailed GPU info
                try:
                    subprocess.run(["nvidia-smi"], check=False)
                except Exception:
                    pass
                
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
        except Exception:
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
        "apple_silicon_fix": False,
        "device": "cpu",
        "use_gpu": False,
        "memory_efficient_attention": False
    }
    
    # Check for different hardware environments
    
    # Check if running on Apple Silicon (M1/M2/M3)
    is_apple_silicon = False
    try:
        import platform
        if platform.system() == "Darwin" and platform.processor() == "arm":
            is_apple_silicon = True
            params["apple_silicon_fix"] = True
            params["device"] = "cpu"  # Force CPU for Apple Silicon
            if verbose:
                print("🍎 Detected Apple Silicon (M1/M2/M3) - Using CPU with special optimizations")
    except ImportError:
        pass
    
    # Check for Colab with GPU if not on Apple Silicon
    if not is_apple_silicon:
        # Check if running in Colab with GPU
        try:
            import google.colab
            colab_info = get_colab_type(verbose=False)
            
            if colab_info.get("hardware") == "GPU":
                params["use_gpu"] = True
                params["device"] = "cuda"
                
                # Get GPU memory info for T4 GPU typically found in Colab
                if "gpu_memory_gb" in colab_info:
                    gpu_memory_gb = colab_info["gpu_memory_gb"]
                    available_memory_mb = int(gpu_memory_gb * 1024 * 0.85)  # Use 85% of available memory
                    
                    if verbose:
                        print(f"✅ GPU detected in Colab: {colab_info.get('gpu_name', 'Unknown GPU')}")
                        print(f"📊 Available GPU memory: {gpu_memory_gb:.1f} GB (using {available_memory_mb} MB)")
        except (ImportError, Exception):
            pass
    
    # Get GPU info if still not provided
    if available_memory_mb is None:
        try:
            # Try to directly get GPU memory from PyTorch
            import torch
            if torch.cuda.is_available():
                params["use_gpu"] = True
                params["device"] = "cuda"
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # MB
                allocated = torch.cuda.memory_allocated(0) / (1024**2)  # MB
                available_memory_mb = int((total_memory - allocated) * 0.85)  # Use 85% of free memory
                
                if verbose and not is_apple_silicon:
                    print(f"🔍 Auto-detected {available_memory_mb} MB available GPU memory")
            else:
                # Fallback to check_gpu_status
                gpu_info = check_gpu_status(verbose=False)
                if gpu_info.get("memory_info"):
                    available_memory_mb = gpu_info["memory_info"]["free_mb"]
                    params["use_gpu"] = True
                    params["device"] = "cuda"
                else:
                    # No GPU or couldn't detect memory, use conservative defaults for CPU
                    available_memory_mb = 2000
                    params["use_gpu"] = False
                    params["device"] = "cpu"
        except Exception:
            # If all detection fails, use conservative defaults
            available_memory_mb = 2000
            params["use_gpu"] = False
            params["device"] = "cpu"
    
    # Adjust parameters based on model size
    size_factor_map = {
        "tiny": 0.5,    # e.g., DistilGPT2, smallest models (< 100M params)
        "small": 1.0,   # e.g., GPT2, OPT-125M (100-200M params)
        "medium": 2.0,  # e.g., GPT2-Medium, OPT-350M (200-500M params)
        "large": 4.0,   # e.g., GPT2-Large, OPT-1.3B (0.5B-2B params)
        "xl": 8.0       # e.g., GPT2-XL, larger models (>2B params)
    }
    
    size_factor = size_factor_map.get(model_size.lower(), 1.0)
    
    # Different memory scaling based on whether using GPU or CPU
    if params["use_gpu"]:
        # For T4 GPU in Colab (16GB), scaling factor relative to that
        memory_factor = max(0.1, min(1.0, available_memory_mb / 15000))
        
        # Calculate optimized parameters for GPU
        if prefer_stability:
            # More conservative settings
            params["batch_size"] = max(1, min(16, int(8 * memory_factor / size_factor)))
            params["sequence_length"] = max(64, min(256, int(192 * memory_factor / size_factor)))
            params["stability_level"] = 2 if size_factor >= 2.0 else 1
        else:
            # More aggressive settings
            params["batch_size"] = max(1, min(32, int(16 * memory_factor / size_factor)))
            params["sequence_length"] = max(64, min(512, int(384 * memory_factor / size_factor)))
            params["stability_level"] = 1
        
        # Enable memory efficient attention for T4 GPUs
        params["memory_efficient_attention"] = True
        
        # Enable mixed precision for all GPU operations
        params["use_fp16"] = True
        
        # Apply gradient accumulation for large models
        if size_factor >= 4.0:
            params["gradient_accumulation_steps"] = max(2, int(size_factor / 2))
    else:
        # For CPU, be more conservative
        memory_factor = max(0.1, min(1.0, available_memory_mb / 8000))
        
        # Calculate optimized parameters for CPU
        if prefer_stability:
            # More conservative settings
            params["batch_size"] = max(1, min(4, int(2 * memory_factor / size_factor)))
            params["sequence_length"] = max(32, min(128, int(96 * memory_factor / size_factor)))
        else:
            # More aggressive settings
            params["batch_size"] = max(1, min(8, int(4 * memory_factor / size_factor)))
            params["sequence_length"] = max(32, min(192, int(128 * memory_factor / size_factor)))
        
        # Gradient accumulation for CPU to compensate for smaller batch sizes
        params["gradient_accumulation_steps"] = max(2, int(size_factor * 2))
    
    if verbose:
        print(f"📊 Optimized settings for {model_size} model:")
        print(f"  - Device: {params['device']}")
        print(f"  - Batch size: {params['batch_size']}")
        print(f"  - Sequence length: {params['sequence_length']}")
        print(f"  - Gradient accumulation steps: {params['gradient_accumulation_steps']}")
        print(f"  - Mixed precision (FP16): {params['use_fp16']}")
        print(f"  - Memory efficient attention: {params['memory_efficient_attention']}")
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
    vmax: Optional[float] = None,
    verbose: bool = False,  # Default to less verbose for notebook use
    return_tensor: bool = False  # Allow returning the cleaned tensor for chaining operations
) -> Union[plt.Figure, Tuple[plt.Figure, np.ndarray]]:
    """
    Safely visualize a tensor with proper detach/cpu/numpy handling.
    Works with GPU tensors, CPU tensors, and Apple Silicon environments.
    
    Args:
        tensor: The tensor to visualize (can be on any device)
        title: Title for the plot
        cmap: Colormap to use
        save_path: Optional path to save the visualization
        figsize: Figure size as (width, height) in inches
        show_colorbar: Whether to show a colorbar
        vmin: Minimum value for colormap scaling (auto-computed if None)
        vmax: Maximum value for colormap scaling (auto-computed if None)
        verbose: Whether to print tensor information
        return_tensor: Whether to also return the numpy tensor alongside the figure
        
    Returns:
        The matplotlib figure object, or (figure, numpy_tensor) if return_tensor=True
    """
    # Detect Apple Silicon environment
    is_apple_silicon = False
    try:
        import platform
        if platform.system() == "Darwin" and platform.processor() == "arm":
            is_apple_silicon = True
    except:
        pass
    
    # Get original device information for debugging
    original_device = "cpu"
    required_detach = False
    was_cuda = False
    
    # Ensure tensor is properly converted for visualization
    if isinstance(tensor, torch.Tensor):
        # Record original state for debugging
        if hasattr(tensor, 'device'):
            original_device = str(tensor.device)
        if tensor.requires_grad:
            required_detach = True
        if tensor.is_cuda:
            was_cuda = True
            
        # Special handling for different environments
        if is_apple_silicon:
            # Extra careful handling for Apple Silicon
            try:
                if tensor.requires_grad:
                    tensor = tensor.detach()
                if tensor.is_cuda:
                    tensor = tensor.cpu()
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                # Convert to numpy array
                tensor_np = tensor.numpy()
            except Exception as e:
                # Ultimate fallback for Apple Silicon: create zeros array of same shape
                if verbose:
                    print(f"⚠️ Error converting tensor on Apple Silicon: {e}")
                    print("Creating empty tensor with same shape for visualization")
                tensor_shape = tensor.shape
                tensor_np = np.zeros(tensor_shape)
        else:
            # Standard handling for other environments
            try:
                if tensor.requires_grad:
                    tensor = tensor.detach()
                if tensor.is_cuda:
                    tensor = tensor.cpu()
                # Convert to numpy array
                tensor_np = tensor.numpy()
            except Exception as e:
                if verbose:
                    print(f"⚠️ Error converting tensor: {e}")
                # Create empty array as fallback
                tensor_shape = tensor.shape
                tensor_np = np.zeros(tensor_shape)
                
    elif isinstance(tensor, np.ndarray):
        tensor_np = tensor
    else:
        # Try to convert other array-like objects
        try:
            tensor_np = np.array(tensor)
        except:
            raise TypeError(f"Expected torch.Tensor, numpy.ndarray, or array-like object, got {type(tensor)}")
    
    # Handle potential NaN or Inf values
    has_nans = np.isnan(tensor_np).any()
    has_infs = np.isinf(tensor_np).any()
    
    if has_nans or has_infs:
        if verbose:
            print("⚠️ Warning: Tensor contains NaN or Inf values, replacing with safe values")
        tensor_np = np.nan_to_num(tensor_np, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Automatically determine appropriate color limits if not provided
    if vmin is None:
        try:
            vmin = np.percentile(tensor_np, 1) if not has_nans else tensor_np.min()
        except:
            vmin = tensor_np.min() if not has_nans else 0.0
            
    if vmax is None:
        try:
            vmax = np.percentile(tensor_np, 99) if not has_nans else tensor_np.max()
        except:
            vmax = tensor_np.max() if not has_infs else 1.0
        
    # Ensure vmin and vmax are not equal (which would cause matplotlib warnings)
    if np.isclose(vmin, vmax) or vmin == vmax:
        vmin = vmin - 0.1 if vmin != 0 else -0.1
        vmax = vmax + 0.1 if vmax != 0 else 0.1
    
    # Plot the image
    try:
        im = ax.imshow(tensor_np, cmap=cmap, vmin=vmin, vmax=vmax)
    except Exception as e:
        if verbose:
            print(f"⚠️ Error in plotting: {e}")
            print("Attempting to flatten or reshape tensor for visualization")
        
        # Try to reshape or flatten if imshow fails
        orig_shape = tensor_np.shape
        try:
            # For 1D tensors or tensors with singleton dimensions
            if tensor_np.ndim == 1 or (tensor_np.ndim > 1 and 1 in tensor_np.shape):
                tensor_np = tensor_np.reshape(-1, 1)
            # For higher dimensional tensors, flatten to 2D
            elif tensor_np.ndim > 2:
                tensor_np = tensor_np.reshape(tensor_np.shape[0], -1)
                
            im = ax.imshow(tensor_np, cmap=cmap, vmin=vmin, vmax=vmax)
            
            # Note the reshaping in the title
            title = f"{title} (reshaped from {orig_shape})"
        except Exception as reshape_error:
            if verbose:
                print(f"⚠️ Reshaping failed: {reshape_error}")
            # Create a simple heatmap as fallback
            tensor_np = np.ones((5, 5))
            im = ax.imshow(tensor_np, cmap=cmap)
            title = f"{title} (visualization failed - showing placeholder)"
    
    # Add colorbar if requested
    if show_colorbar:
        fig.colorbar(im, ax=ax)
    
    # Add title with environment info
    if is_apple_silicon:
        ax.set_title(f"{title} (Apple Silicon)")
    elif was_cuda:
        ax.set_title(f"{title} (CUDA)")
    else:
        ax.set_title(title)
    
    # Add axis labels if tensor is 2D
    if tensor_np.ndim == 2:
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Row Index')
    
    # Save if path provided
    if save_path is not None:
        try:
            plt.savefig(save_path, bbox_inches='tight')
            if verbose:
                print(f"Visualization saved to: {save_path}")
        except Exception as save_error:
            if verbose:
                print(f"⚠️ Could not save figure: {save_error}")
    
    # Display useful info
    if verbose:
        print(f"✅ Tensor visualization:")
        print(f"  - Shape: {tensor_np.shape}")
        print(f"  - Range: [{tensor_np.min():.4f}, {tensor_np.max():.4f}]") 
        print(f"  - Original device: {original_device}")
        if required_detach:
            print(f"  - Required detach: Yes (had gradients)")
        if was_cuda:
            print(f"  - Required CPU transfer: Yes (was on CUDA)")
        if has_nans or has_infs:
            print(f"  - Had NaN/Inf values: Yes (replaced with safe values)")
    
    # Return figure or tuple based on return_tensor flag
    if return_tensor:
        return fig, tensor_np
    else:
        return fig