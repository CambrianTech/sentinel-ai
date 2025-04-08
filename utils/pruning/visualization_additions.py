"""
Additional visualization utilities for neural plasticity demonstrations.
These functions support the NeuralPlasticityDemo notebook.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any, Union


def visualize_gradient_norms(
    grad_norm_values: torch.Tensor, 
    pruned_heads: Optional[List[Tuple[int, int]]] = None, 
    revived_heads: Optional[List[Tuple[int, int]]] = None, 
    title: str = "Gradient Norms", 
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a visualization of gradient norms with markers for pruned/revived heads.
    
    Args:
        grad_norm_values: Tensor of gradient norm values for all heads
        pruned_heads: List of (layer, head) tuples for pruned heads
        revived_heads: List of (layer, head) tuples for revived heads
        title: Title for the plot
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(grad_norm_values.detach().cpu().numpy(), cmap="plasma", aspect="auto")
    plt.colorbar(label="Gradient Norm")
        
    # Mark pruned heads with 'P'
    if pruned_heads:
        for layer, head in pruned_heads:
            plt.text(head, layer, "P", ha="center", va="center",
                    color="white", weight="bold", bbox=dict(facecolor='red', alpha=0.5))
        
    # Mark revived heads with 'R'
    if revived_heads:
        for layer, head in revived_heads:
            plt.text(head, layer, "R", ha="center", va="center",
                    color="white", weight="bold", bbox=dict(facecolor='green', alpha=0.5))
        
    plt.title(title)
    plt.xlabel("Head Index")
    plt.ylabel("Layer Index")
    plt.tight_layout()
        
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
    return fig


def visualize_attention_matrix(
    attn_matrix: torch.Tensor,
    layer_idx: int = 0,
    head_idx: int = 0,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize an attention matrix for a specific layer and head.
    
    Args:
        attn_matrix: The attention tensor with shape [batch, heads, seq_len, seq_len]
        layer_idx: Layer index to visualize
        head_idx: Head index to visualize
        title: Optional title for the plot
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(8, 6))
    attention_map = plt.imshow(attn_matrix[0, head_idx].cpu().numpy(), cmap='viridis')
    
    # Ensure proper scaling for attention values (0 to 1)
    plt.clim(0, 1.0)
    
    # Add colorbar and labels
    plt.colorbar(label='Attention probability')
    plt.title(title or f'Attention pattern (layer {layer_idx}, head {head_idx})')
    plt.xlabel('Sequence position (to)')
    plt.ylabel('Sequence position (from)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
    return fig


def visualize_entropy_heatmap(
    entropy_values: torch.Tensor,
    title: str = "Attention Entropy Heatmap",
    save_path: Optional[str] = None,
    min_value: float = 0.0,
    annotate: bool = True
) -> plt.Figure:
    """
    Visualize entropy values across all layers and heads as a heatmap.
    
    Args:
        entropy_values: Tensor of entropy values with shape [layers, heads]
        title: Title for the plot
        save_path: Optional path to save the figure
        min_value: Minimum value for the colormap scale
        annotate: Whether to add value annotations to the cells
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(10, 6))
    
    # Convert to numpy if it's a torch tensor
    if isinstance(entropy_values, torch.Tensor):
        entropy_data = entropy_values.detach().cpu().numpy()
    else:
        entropy_data = entropy_values
    
    # Create the heatmap
    plt.imshow(entropy_data, cmap='viridis', aspect='auto')
    
    # Set proper colormap limits
    max_value = np.max(entropy_data)
    plt.clim(min_value, max(0.1, max_value))
    
    # Add colorbar and labels
    plt.colorbar(label='Entropy')
    plt.title(title)
    plt.xlabel('Head Index')
    plt.ylabel('Layer Index')
    
    # Add text annotations for each cell if requested
    if annotate:
        for i in range(entropy_data.shape[0]):
            for j in range(entropy_data.shape[1]):
                plt.text(j, i, f'{entropy_data[i, j]:.2f}',
                        ha="center", va="center", color="w")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
    return fig


def visualize_normalized_entropy(
    entropy_values: torch.Tensor,
    max_entropy: torch.Tensor,
    title: str = "Normalized Entropy (% of Maximum)",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize normalized entropy values as percentage of maximum possible entropy.
    
    Args:
        entropy_values: Tensor of entropy values with shape [layers, heads]
        max_entropy: Tensor or scalar of maximum possible entropy values
        title: Title for the plot
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(10, 6))
    
    # Normalize and convert to percentage
    if isinstance(entropy_values, torch.Tensor) and isinstance(max_entropy, torch.Tensor):
        norm_entropies = (entropy_values / max_entropy.item() * 100).detach().cpu().numpy()
    elif isinstance(entropy_values, torch.Tensor):
        norm_entropies = (entropy_values / max_entropy * 100).detach().cpu().numpy()
    else:
        norm_entropies = (entropy_values / max_entropy * 100)
    
    # Create the heatmap
    plt.imshow(norm_entropies, cmap='viridis', aspect='auto', vmin=0, vmax=100)
    plt.colorbar(label='% of Max Entropy')
    plt.title(title)
    plt.xlabel('Head Index')
    plt.ylabel('Layer Index')
    
    # Add text annotations for each cell
    for i in range(norm_entropies.shape[0]):
        for j in range(norm_entropies.shape[1]):
            plt.text(j, i, f'{norm_entropies[i, j]:.1f}%',
                    ha="center", va="center", color="w")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
    return fig


def visualize_entropy_vs_gradient(
    entropy_values: torch.Tensor,
    grad_norm_values: torch.Tensor,
    title: str = "Entropy vs Gradient Relationship",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a scatter plot showing the relationship between entropy and gradient values.
    
    Args:
        entropy_values: Tensor of entropy values 
        grad_norm_values: Tensor of gradient norm values
        title: Title for the plot
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(8, 6))
    
    # Convert to numpy if they're torch tensors
    if isinstance(entropy_values, torch.Tensor):
        entropy_flat = entropy_values.flatten().cpu().numpy()
    else:
        entropy_flat = entropy_values.flatten()
        
    if isinstance(grad_norm_values, torch.Tensor):
        grad_flat = grad_norm_values.flatten().cpu().numpy()
    else:
        grad_flat = grad_norm_values.flatten()
    
    # Create scatter plot
    plt.scatter(entropy_flat, grad_flat, alpha=0.7)
    plt.xlabel('Entropy (higher = less focused)')
    plt.ylabel('Gradient Norm (higher = more impact)')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
    return fig


def visualize_training_progress(
    metrics_history: Dict[str, list],
    max_display_points: int = 100,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize training progress including loss, pruning metrics and sparsity.
    
    Args:
        metrics_history: Dictionary containing training metrics history
        max_display_points: Maximum number of points to display (downsamples if needed)
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=figsize,
        dpi=100,
        sharex=True
    )

    # Add extra space between subplots
    plt.subplots_adjust(hspace=0.4)

    # Set maximum display limit to prevent excessively large plots
    display_steps = metrics_history["step"]
    if len(display_steps) > max_display_points:
        # Downsample by selecting evenly spaced points
        indices = np.linspace(0, len(display_steps) - 1, max_display_points).astype(int)
        display_steps = [metrics_history["step"][i] for i in indices]
        display_train_loss = [metrics_history["train_loss"][i] for i in indices]
        display_eval_loss = [metrics_history["eval_loss"][i] for i in indices]
        display_pruned_heads = [metrics_history["pruned_heads"][i] for i in indices]
        display_revived_heads = [metrics_history["revived_heads"][i] for i in indices if i < len(metrics_history["revived_heads"])]
        display_sparsity = [metrics_history["sparsity"][i] for i in indices]
        display_epoch = [metrics_history["epoch"][i] for i in indices]
        display_perplexity = ([metrics_history["perplexity"][i] for i in indices] 
                             if "perplexity" in metrics_history and metrics_history["perplexity"] else [])
    else:
        display_train_loss = metrics_history["train_loss"]
        display_eval_loss = metrics_history["eval_loss"]
        display_pruned_heads = metrics_history["pruned_heads"]
        display_revived_heads = metrics_history.get("revived_heads", [])
        display_sparsity = metrics_history["sparsity"]
        display_epoch = metrics_history["epoch"]
        display_perplexity = metrics_history.get("perplexity", [])

    # Plot losses
    ax1.plot(display_steps, display_train_loss, label="Train Loss")
    ax1.plot(display_steps, display_eval_loss, label="Eval Loss")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Evaluation Loss")
    ax1.legend()
    ax1.grid(True)

    # Mark epoch boundaries if available
    if "epoch" in metrics_history and len(display_epoch) > 1:
        for i in range(1, len(display_epoch)):
            if display_epoch[i] != display_epoch[i-1]:
                # This is an epoch boundary
                for ax in [ax1, ax2, ax3]:
                    ax.axvline(x=display_steps[i], color="k", linestyle="--", alpha=0.3)
                    # Make sure ylim exists before using it
                    try:
                        y_pos = ax.get_ylim()[1] * 0.9
                    except:
                        y_pos = 1.0  # fallback position
                    ax.text(display_steps[i], y_pos, f"Epoch {display_epoch[i]}", rotation=90, alpha=0.7)

    # Plot pruning metrics
    if display_revived_heads and len(display_revived_heads) == len(display_pruned_heads):
        ax2.bar(display_steps, display_pruned_heads, alpha=0.5, label="Pruned Heads", color="blue")
        ax2.bar(display_steps, display_revived_heads, alpha=0.5, label="Revived Heads", color="green")
        ax2.set_ylabel("Count")
        ax2.set_title("Head Pruning and Revival")
        ax2.legend(loc="upper left")
        ax2.grid(True)
    else:
        ax2.bar(display_steps, display_pruned_heads, alpha=0.5, label="Pruned Heads", color="blue")
        ax2.set_ylabel("Count")
        ax2.set_title("Head Pruning")
        ax2.legend(loc="upper left")
        ax2.grid(True)

    # Plot sparsity and perplexity
    ax3.plot(display_steps, display_sparsity, "r-", label="Sparsity")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Sparsity")
    ax3.grid(True)

    # Add perplexity line on secondary axis if available
    if display_perplexity:
        ax4 = ax3.twinx()
        ax4.plot(display_steps, display_perplexity, "g-", label="Perplexity")
        ax4.set_ylabel("Perplexity")
        ax4.legend(loc="upper right")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
    return fig


# Export all functions
__all__ = [
    'visualize_gradient_norms',
    'visualize_attention_matrix',
    'visualize_entropy_heatmap',
    'visualize_normalized_entropy',
    'visualize_entropy_vs_gradient',
    'visualize_training_progress',
]