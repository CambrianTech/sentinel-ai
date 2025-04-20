"""
Neural Plasticity Visualization

This module provides visualization utilities for neural plasticity experiments.
It visualizes head entropy, gradients, pruning decisions, training metrics,
and attention patterns.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from utils.colab.helpers import safe_tensor_imshow


def visualize_head_entropy(
    entropy_values: torch.Tensor,
    title: str = "Attention Entropy Heatmap",
    min_value: float = 0.0,
    annotate: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    cmap: str = "viridis",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize entropy values across all layers and heads as a heatmap.
    
    Args:
        entropy_values: Tensor of entropy values with shape [layers, heads]
        title: Title for the plot
        min_value: Minimum value for the colormap scale
        annotate: Whether to add value annotations to the cells
        figsize: Figure size (width, height) in inches
        cmap: Colormap to use
        save_path: Optional path to save the visualization
        
    Returns:
        matplotlib Figure object
    """
    # Convert to numpy if tensor
    if isinstance(entropy_values, torch.Tensor):
        entropy_data = entropy_values.detach().cpu().numpy()
    else:
        entropy_data = entropy_values
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap using safe_tensor_imshow
    im = safe_tensor_imshow(
        entropy_data, 
        title=title,
        cmap=cmap
    )
    
    # Set proper colormap limits with non-zero range
    plt.clim(min_value, max(0.1, np.max(entropy_data)))
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Entropy')
    
    # Add labels
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Layer Index')
    
    # Add text annotations if requested
    if annotate:
        for i in range(entropy_data.shape[0]):
            for j in range(entropy_data.shape[1]):
                ax.text(j, i, f'{entropy_data[i, j]:.2f}',
                        ha="center", va="center", color="w")
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


def visualize_head_gradients(
    grad_norm_values: torch.Tensor,
    pruned_heads: Optional[List[Tuple[int, int]]] = None,
    revived_heads: Optional[List[Tuple[int, int]]] = None,
    title: str = "Gradient Norms",
    figsize: Tuple[int, int] = (10, 5),
    cmap: str = "plasma",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a visualization of gradient norms with markers for pruned/revived heads.
    
    Args:
        grad_norm_values: Tensor of gradient norm values for all heads
        pruned_heads: List of (layer, head) tuples for pruned heads
        revived_heads: List of (layer, head) tuples for revived heads
        title: Title for the plot
        figsize: Figure size (width, height) in inches
        cmap: Colormap to use
        save_path: Optional path to save the visualization
        
    Returns:
        matplotlib Figure object
    """
    # Convert to numpy if tensor
    if isinstance(grad_norm_values, torch.Tensor):
        grad_data = grad_norm_values.detach().cpu().numpy()
    else:
        grad_data = grad_norm_values
        
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap using safe_tensor_imshow
    im = safe_tensor_imshow(
        grad_data, 
        title=title,
        cmap=cmap
    )
    
    plt.colorbar(im, ax=ax, label="Gradient Norm")
    
    # Mark pruned heads with 'P'
    if pruned_heads:
        for layer, head in pruned_heads:
            ax.text(head, layer, "P", ha="center", va="center",
                   color="white", weight="bold", bbox=dict(facecolor='red', alpha=0.5))
    
    # Mark revived heads with 'R'
    if revived_heads:
        for layer, head in revived_heads:
            ax.text(head, layer, "R", ha="center", va="center",
                   color="white", weight="bold", bbox=dict(facecolor='green', alpha=0.5))
    
    # Set labels
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


def visualize_pruning_decisions(
    grad_norm_values: torch.Tensor,
    pruning_mask: torch.Tensor,
    title: str = "Pruning Decisions",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a visualization highlighting pruning decisions.
    
    Args:
        grad_norm_values: Tensor of gradient norm values for all heads
        pruning_mask: Boolean tensor where True indicates a head should be pruned
        title: Title for the plot
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the visualization
        
    Returns:
        matplotlib Figure object
    """
    # Convert to numpy arrays
    if isinstance(grad_norm_values, torch.Tensor):
        grad_data = grad_norm_values.detach().cpu().numpy()
    else:
        grad_data = grad_norm_values
        
    if isinstance(pruning_mask, torch.Tensor):
        mask_data = pruning_mask.detach().cpu().numpy()
    else:
        mask_data = pruning_mask
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Base plot with all gradient values
    im = safe_tensor_imshow(
        grad_data, 
        title=title,
        cmap="YlOrRd"
    )
    
    # Create a masked array where pruned heads are highlighted
    masked_grads = np.ma.array(grad_data, mask=~mask_data)
    
    # Overlay plot with pruned heads highlighted
    plt.imshow(
        masked_grads, 
        cmap='Reds', 
        alpha=0.7,
        aspect='auto'
    )
    
    # Add colorbar
    plt.colorbar(im, label='Gradient Norm')
    
    # Add title and labels
    plt.title(title)
    plt.xlabel('Head Index')
    plt.ylabel('Layer Index')
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


def visualize_training_metrics(
    metrics_history: Dict[str, List[float]],
    title: str = "Training Progress",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize training metrics over time.
    
    Args:
        metrics_history: Dictionary with metrics history
        title: Title for the plot
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the visualization
        
    Returns:
        matplotlib Figure object
    """
    # Extract step information
    steps = metrics_history.get("step", list(range(len(next(iter(metrics_history.values()))))))
    
    # Create subplot layout based on available metrics
    num_plots = 0
    if any(m in metrics_history for m in ["train_loss", "eval_loss"]):
        num_plots += 1
    if "perplexity" in metrics_history:
        num_plots += 1
    if any(m in metrics_history for m in ["pruned_heads", "revived_heads", "sparsity"]):
        num_plots += 1
        
    num_plots = max(num_plots, 1)  # Ensure at least one plot
    
    # Create figure
    fig, axs = plt.subplots(num_plots, 1, figsize=figsize, sharex=True)
    
    # Convert to list if only one subplot
    if num_plots == 1:
        axs = [axs]
    
    plot_idx = 0
    
    # Plot loss metrics
    if any(m in metrics_history for m in ["train_loss", "eval_loss"]):
        ax = axs[plot_idx]
        if "train_loss" in metrics_history:
            ax.plot(steps, metrics_history["train_loss"], label="Train Loss")
        if "eval_loss" in metrics_history:
            ax.plot(steps, metrics_history["eval_loss"], label="Eval Loss")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot perplexity
    if "perplexity" in metrics_history:
        ax = axs[plot_idx]
        ax.plot(steps, metrics_history["perplexity"], label="Perplexity")
        ax.set_ylabel("Perplexity")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot pruning metrics
    if any(m in metrics_history for m in ["pruned_heads", "revived_heads", "sparsity"]):
        ax = axs[plot_idx]
        if "pruned_heads" in metrics_history:
            ax.plot(steps, metrics_history["pruned_heads"], label="Pruned Heads", color="red")
        if "revived_heads" in metrics_history:
            ax.plot(steps, metrics_history["revived_heads"], label="Revived Heads", color="green")
        if "sparsity" in metrics_history:
            # Plot on secondary axis
            ax2 = ax.twinx()
            ax2.plot(steps, metrics_history["sparsity"], label="Sparsity", color="blue", linestyle="--")
            ax2.set_ylabel("Sparsity")
            
        ax.set_ylabel("Head Count")
        ax.legend(loc="upper left")
        if "sparsity" in metrics_history:
            ax2.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Set common x-axis label
    axs[-1].set_xlabel("Steps")
    
    # Set overall title
    fig.suptitle(title, fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


def visualize_attention_patterns(
    attention_maps: torch.Tensor,
    layer_idx: int = 0,
    head_idx: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    num_heads: int = 4,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize attention patterns for one or more heads.
    
    Args:
        attention_maps: Attention tensor with shape [batch, heads, seq_len, seq_len]
        layer_idx: Index of layer to visualize
        head_idx: Index of attention head to visualize (if None, shows multiple heads)
        title: Title for the plot
        figsize: Figure size (width, height) in inches
        num_heads: Number of heads to visualize if head_idx is None
        save_path: Optional path to save the visualization
        
    Returns:
        matplotlib Figure object
    """
    # Handle tensor if it's a list of layers
    if isinstance(attention_maps, list):
        attention_maps = attention_maps[layer_idx]
    
    # Ensure tensor is on CPU and converted to numpy
    if isinstance(attention_maps, torch.Tensor):
        attn = attention_maps.detach().cpu()
    else:
        attn = attention_maps
    
    # Check tensor dimensions
    if attn.dim() != 4:
        raise ValueError(f"Expected 4D attention tensor [batch, heads, seq_len, seq_len], got shape {attn.shape}")
    
    # Create figure
    if head_idx is not None:
        # Single head visualization
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot attention pattern using safe_tensor_imshow
        attention_map = safe_tensor_imshow(
            attn[0, head_idx], 
            title=title or f'Attention pattern (layer {layer_idx}, head {head_idx})',
            cmap='viridis'
        )
        
        # Set proper limits for attention values (0 to 1)
        plt.clim(0, 1.0)
        
        # Add colorbar
        plt.colorbar(attention_map, label='Attention probability')
        
        # Add labels
        plt.xlabel('Sequence position (to)')
        plt.ylabel('Sequence position (from)')
        
    else:
        # Multiple heads visualization
        heads_to_show = min(num_heads, attn.shape[1])
        fig, axs = plt.subplots(1, heads_to_show, figsize=figsize)
        
        # Adjust title
        if title:
            fig.suptitle(title, fontsize=14)
        else:
            fig.suptitle(f'Attention patterns (layer {layer_idx})', fontsize=14)
        
        # Plot each head
        for i in range(heads_to_show):
            # Plot attention pattern
            im = axs[i].imshow(attn[0, i].numpy(), cmap='viridis')
            
            # Set proper limits for attention values (0 to 1)
            im.set_clim(0, 1.0)
            
            # Add labels only to the first subplot for cleaner display
            if i == 0:
                axs[i].set_ylabel('From position')
            
            axs[i].set_xlabel('To position')
            axs[i].set_title(f'Head {i}')
        
        # Add colorbar to the right of the last subplot
        fig.colorbar(im, ax=axs[-1], label='Attention probability')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig