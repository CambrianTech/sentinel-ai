"""
Core integration utilities for neural plasticity in Colab.

This module provides functions for detecting the environment (Colab vs local),
managing file paths, and integrating with Google Drive when in Colab.
"""

import os
import sys
import platform
import json
from datetime import datetime
from typing import Dict, Any, Optional, Union, Tuple

def is_colab() -> bool:
    """
    Detect if code is running in Google Colab.
    
    Returns:
        bool: True if running in Google Colab, False otherwise
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False

def is_apple_silicon() -> bool:
    """
    Detect if code is running on Apple Silicon hardware.
    
    Returns:
        bool: True if running on Apple Silicon (M1/M2/M3), False otherwise
    """
    return platform.system() == 'Darwin' and platform.processor() == 'arm'

def has_gpu() -> bool:
    """
    Detect if a GPU is available.
    
    Returns:
        bool: True if a GPU is available, False otherwise
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def get_environment_info() -> Dict[str, Any]:
    """
    Get detailed information about the current execution environment.
    
    Returns:
        Dict[str, Any]: Dictionary with environment information
    """
    info = {
        "is_colab": is_colab(),
        "is_apple_silicon": is_apple_silicon(),
        "has_gpu": has_gpu(),
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "timestamp": datetime.now().isoformat()
    }
    
    # Add GPU information if available
    if info["has_gpu"]:
        try:
            import torch
            info["gpu_type"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["cuda_version"] = torch.version.cuda
        except (ImportError, Exception) as e:
            info["gpu_error"] = str(e)
    
    return info

def get_output_dir(experiment_name: str, base_dir: Optional[str] = None) -> str:
    """
    Get appropriate output directory for the current environment.
    
    Args:
        experiment_name: Name of the experiment
        base_dir: Optional base directory override
    
    Returns:
        str: Path to the output directory
    """
    if is_colab() and base_dir is None:
        try:
            from google.colab import drive
            # Mount Google Drive if not already mounted
            drive.mount('/content/drive', force_remount=False)
            base_dir = "/content/drive/MyDrive/neural_plasticity_experiments"
        except Exception as e:
            print(f"Warning: Could not mount Google Drive: {e}")
            # Fallback to Colab local storage
            base_dir = "/content/neural_plasticity_experiments"
    elif base_dir is None:
        # Default local base directory
        base_dir = os.path.join(os.getcwd(), "neural_plasticity_output")
    
    # Create timestamp-based directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save environment info
    env_info = get_environment_info()
    with open(os.path.join(output_dir, "environment_info.json"), "w") as f:
        json.dump(env_info, f, indent=2)
    
    return output_dir

def save_experiment_results(results: Dict[str, Any], name: str, output_dir: Optional[str] = None) -> str:
    """
    Save experiment results to a file in a format that can be loaded in either environment.
    
    Args:
        results: Dictionary of experiment results
        name: Name for the results file
        output_dir: Optional directory to save results in
    
    Returns:
        str: Path to the saved results file
    """
    if output_dir is None:
        output_dir = get_output_dir(name)
    
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results_path

def load_experiment_results(path: str) -> Dict[str, Any]:
    """
    Load experiment results from a file.
    
    Args:
        path: Path to the results file or directory
    
    Returns:
        Dict[str, Any]: Loaded results
    """
    # Check if path is a directory
    if os.path.isdir(path):
        path = os.path.join(path, "results.json")
    
    # Load results
    with open(path, "r") as f:
        results = json.load(f)
    
    return results

def install_requirements() -> None:
    """
    Install required packages for neural plasticity experiments.
    
    This is primarily used in Colab to set up the environment.
    """
    if is_colab():
        import subprocess
        
        # Basic dependencies
        print("Installing basic dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "torch", "transformers", "datasets", "matplotlib", "numpy", "tqdm"
        ])
        
        # Install development version from GitHub
        print("Installing neural plasticity module from GitHub...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "git+https://github.com/CambrianTech/sentinel-ai.git@feature/implement-adaptive-plasticity"
        ])
        
        print("Installation complete!")
    else:
        print("Not running in Colab - skipping package installation")