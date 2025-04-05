"""
Checkpoint utilities for saving and loading model states.

This module provides functions for saving and loading checkpoints during
training, to allow resuming from interruptions or implementing early stopping.
"""

import os
import pickle
import torch
from typing import Dict, Any, Optional, Tuple


def save_checkpoint(model, optimizer, step, epoch, head_lr_multipliers, path):
    """
    Save a model checkpoint to disk.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        step: Current training step
        epoch: Current epoch
        head_lr_multipliers: Learning rate multipliers for each head
        path: Path to save the checkpoint
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "head_lr_multipliers": head_lr_multipliers,
        "train_step": step,
        "epoch": epoch,
    }, path)
    print(f"[Checkpoint] Saved at {path}")


def load_checkpoint(model, optimizer, path, device, head_lr_multipliers) -> Tuple[int, int, Dict]:
    """
    Load a model checkpoint from disk.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        path: Path to the checkpoint file
        device: Device to load to
        head_lr_multipliers: Default multipliers if not in checkpoint
        
    Returns:
        Tuple of (step, epoch, head_lr_multipliers)
    """
    if not os.path.exists(path):
        print("[Checkpoint] No checkpoint found, starting fresh.")
        return 0, 0, head_lr_multipliers

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    step = checkpoint.get("train_step", 0)
    epoch = checkpoint.get("epoch", 0)
    head_lr_multipliers = checkpoint.get("head_lr_multipliers", head_lr_multipliers)
    print(f"[Checkpoint] Loaded from {path}, resuming at epoch {epoch}, step {step}")
    return step, epoch, head_lr_multipliers


# Additional functions for enhanced functionality

def save_checkpoint_extended(model_state: Dict[str, Any], path: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Extended version of save_checkpoint with more flexibility.
    
    Args:
        model_state: Dictionary containing model state
        path: Path to save the checkpoint
        metadata: Optional metadata to save with the checkpoint
        
    Returns:
        True if save was successful, False otherwise
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # Combine model state and metadata
    checkpoint = {
        "model_state": model_state,
        "metadata": metadata or {}
    }
    
    # Save to disk
    try:
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        return True
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        return False


def load_checkpoint_extended(path: str) -> Dict[str, Any]:
    """
    Extended version of load_checkpoint with more flexibility.
    
    Args:
        path: Path to the checkpoint file
        
    Returns:
        Dictionary containing model_state and metadata
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    
    try:
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Ensure expected structure
        if "model_state" not in checkpoint:
            raise ValueError("Invalid checkpoint format: missing 'model_state'")
        
        return checkpoint
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint: {e}")


def get_latest_checkpoint(directory: str, prefix: str = "") -> Optional[str]:
    """
    Find the latest checkpoint in a directory.
    
    Args:
        directory: Directory to search
        prefix: Optional prefix for checkpoint files
        
    Returns:
        Path to the latest checkpoint, or None if no checkpoint found
    """
    if not os.path.exists(directory) or not os.path.isdir(directory):
        return None
    
    # Find all checkpoint files
    checkpoints = []
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and (filename.endswith(".pt") or 
                                          filename.endswith(".pth") or 
                                          filename.endswith(".ckpt")):
            checkpoints.append(os.path.join(directory, filename))
    
    if not checkpoints:
        return None
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    
    return checkpoints[0]