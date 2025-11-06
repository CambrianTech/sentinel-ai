"""
Heatmap visualization utilities for Sentinel-AI.

This module provides functions for visualizing attention entropy,
gate activity, and other metrics as heatmaps.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def plot_entropy_heatmap(
    entropy_data: Dict[int, torch.Tensor],
    title: str = "Attention Entropy",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis"
) -> plt.Figure:
    """
    Plot entropy values as a heatmap.
    
    Args:
        entropy_data: Dictionary mapping layer indices to entropy tensors
        title: Title for the plot
        save_path: Path to save the figure (if None, figure is not saved)
        figsize: Figure size
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        cmap: Colormap name
        
    Returns:
        Matplotlib figure
    """
    # Convert entropy data to a 2D array
    layers = sorted(entropy_data.keys())
    
    if not layers:
        logger.warning("No entropy data to plot")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No entropy data available", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        return fig
    
    # Find the maximum number of heads across all layers
    max_heads = max(len(entropy_data[layer]) for layer in layers)
    
    # Create a 2D array of entropy values (layers x heads)
    entropy_array = np.zeros((len(layers), max_heads))
    entropy_array.fill(np.nan)  # Fill with NaN so missing heads are not plotted
    
    for i, layer in enumerate(layers):
        entropy_tensor = entropy_data[layer]
        entropy_array[i, :len(entropy_tensor)] = entropy_tensor.cpu().numpy()
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set vmin and vmax if not provided
    if vmin is None:
        vmin = np.nanmin(entropy_array)
    if vmax is None:
        vmax = np.nanmax(entropy_array)
    
    # Create the heatmap
    sns.heatmap(
        entropy_array,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Entropy"},
        xticklabels=range(max_heads),
        yticklabels=layers
    )
    
    # Set labels and title
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")
    ax.set_title(title)
    
    # Add annotations
    for i in range(len(layers)):
        for j in range(max_heads):
            if not np.isnan(entropy_array[i, j]):
                value = entropy_array[i, j]
                # Add text annotation in white or black depending on the darkness of the cell
                text_color = "white" if (value - vmin) / (vmax - vmin) > 0.6 else "black"
                ax.text(j + 0.5, i + 0.5, f"{value:.2f}", 
                        ha="center", va="center", color=text_color, fontsize=8)
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    return fig


def plot_entropy_deltas_heatmap(
    entropy_deltas: Dict[int, torch.Tensor],
    title: str = "Entropy Change After Fine-tuning",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    center: float = 0.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "RdBu_r"  # Red-Blue colormap, reversed so blue is decrease, red is increase
) -> plt.Figure:
    """
    Plot entropy changes after fine-tuning as a heatmap.
    
    Args:
        entropy_deltas: Dictionary mapping layer indices to entropy delta tensors
        title: Title for the plot
        save_path: Path to save the figure (if None, figure is not saved)
        figsize: Figure size
        center: Center value for the diverging colormap
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        cmap: Colormap name (default: RdBu_r for diverging red-blue)
        
    Returns:
        Matplotlib figure
    """
    # Convert entropy data to a 2D array
    layers = sorted(entropy_deltas.keys())
    
    if not layers:
        logger.warning("No entropy delta data to plot")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No entropy delta data available", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        return fig
    
    # Find the maximum number of heads across all layers
    max_heads = max(len(entropy_deltas[layer]) for layer in layers)
    
    # Create a 2D array of entropy delta values (layers x heads)
    delta_array = np.zeros((len(layers), max_heads))
    delta_array.fill(np.nan)  # Fill with NaN so missing heads are not plotted
    
    for i, layer in enumerate(layers):
        delta_tensor = entropy_deltas[layer]
        delta_array[i, :len(delta_tensor)] = delta_tensor.cpu().numpy()
    
    # Calculate overall min/max for symmetric colorbar
    if vmin is None and vmax is None:
        abs_max = max(abs(np.nanmin(delta_array)), abs(np.nanmax(delta_array)))
        vmin = -abs_max
        vmax = abs_max
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the heatmap with a diverging colormap centered at 0
    sns.heatmap(
        delta_array,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        cbar_kws={"label": "Entropy Change"},
        xticklabels=range(max_heads),
        yticklabels=layers
    )
    
    # Set labels and title
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")
    ax.set_title(title)
    
    # Add annotations
    for i in range(len(layers)):
        for j in range(max_heads):
            if not np.isnan(delta_array[i, j]):
                value = delta_array[i, j]
                # Add text annotation in white or black depending on the darkness of the cell
                text_color = "white" if abs((value - center) / (vmax - vmin)) > 0.6 else "black"
                ax.text(j + 0.5, i + 0.5, f"{value:.2f}", 
                        ha="center", va="center", color=text_color, fontsize=8)
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    return fig


def plot_attention_pattern(
    attention_matrix: torch.Tensor,
    title: str = "Attention Pattern",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot an attention pattern as a heatmap.
    
    Args:
        attention_matrix: Attention matrix of shape (seq_len, seq_len)
        title: Title for the plot
        save_path: Path to save the figure (if None, figure is not saved)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy
    if isinstance(attention_matrix, torch.Tensor):
        attention_matrix = attention_matrix.cpu().numpy()
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        attention_matrix,
        ax=ax,
        cmap="viridis",
        vmin=0,
        vmax=attention_matrix.max(),
        cbar_kws={"label": "Attention Weight"}
    )
    
    # Set labels and title
    ax.set_xlabel("Token Position (Key)")
    ax.set_ylabel("Token Position (Query)")
    ax.set_title(title)
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    return fig


def plot_gate_activity(
    gate_history: Dict[int, Dict[int, torch.Tensor]],
    head_indices: Optional[List[Tuple[int, int]]] = None,
    title: str = "Gate Activity During Fine-tuning",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot gate activity over time for selected heads.
    
    Args:
        gate_history: Dictionary mapping steps to layer->gate mappings
        head_indices: List of (layer_idx, head_idx) tuples to plot (if None, plots all pruned heads)
        title: Title for the plot
        save_path: Path to save the figure (if None, figure is not saved)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Sort steps
    steps = sorted(gate_history.keys())
    
    if not steps:
        logger.warning("No gate history to plot")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No gate history available", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        return fig
    
    # If no head indices are provided, select all heads that were pruned
    if head_indices is None:
        head_indices = []
        initial_step = steps[0]
        for layer_idx, gate in gate_history[initial_step].items():
            for head_idx, value in enumerate(gate):
                if value.item() < 0.1:  # Assuming pruned heads have gate values near 0
                    head_indices.append((layer_idx, head_idx))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each head's gate activity
    for layer_idx, head_idx in head_indices:
        gate_values = []
        for step in steps:
            if layer_idx in gate_history[step] and head_idx < len(gate_history[step][layer_idx]):
                gate_values.append(gate_history[step][layer_idx][head_idx].item())
            else:
                gate_values.append(float('nan'))
        
        ax.plot(steps, gate_values, marker='o', label=f"Layer {layer_idx}, Head {head_idx}")
    
    # Set labels and title
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gate Value")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend (only if there aren't too many heads)
    if len(head_indices) <= 10:
        ax.legend()
    else:
        logger.info(f"Too many heads ({len(head_indices)}) to show legend")
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    return fig


def plot_regrowth_heatmap(
    regrowth_data: Dict[Tuple[int, int], Dict[str, float]],
    title: str = "Head Regrowth After Fine-tuning",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    metric: str = "regrowth_ratio"
) -> plt.Figure:
    """
    Plot head regrowth as a heatmap.
    
    Args:
        regrowth_data: Dictionary mapping (layer_idx, head_idx) to metrics
        title: Title for the plot
        save_path: Path to save the figure (if None, figure is not saved)
        figsize: Figure size
        metric: Which metric to use for the heatmap
        
    Returns:
        Matplotlib figure
    """
    if not regrowth_data:
        logger.warning("No regrowth data to plot")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No regrowth data available", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        return fig
    
    # Extract layer and head indices
    layer_indices = sorted(set(layer_idx for (layer_idx, _) in regrowth_data.keys()))
    head_indices = sorted(set(head_idx for (_, head_idx) in regrowth_data.keys()))
    
    # Create a 2D array for the heatmap
    regrowth_array = np.zeros((len(layer_indices), len(head_indices)))
    regrowth_array.fill(np.nan)  # Fill with NaN so missing heads are not plotted
    
    # Fill the array with regrowth values
    for (layer_idx, head_idx), data in regrowth_data.items():
        if metric in data:
            layer_pos = layer_indices.index(layer_idx)
            head_pos = head_indices.index(head_idx)
            regrowth_array[layer_pos, head_pos] = data[metric]
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the heatmap
    sns.heatmap(
        regrowth_array,
        ax=ax,
        cmap="viridis",
        vmin=0,
        vmax=np.nanmax(regrowth_array),
        cbar_kws={"label": metric.replace("_", " ").title()},
        xticklabels=head_indices,
        yticklabels=layer_indices
    )
    
    # Set labels and title
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")
    ax.set_title(title)
    
    # Add annotations
    for i in range(len(layer_indices)):
        for j in range(len(head_indices)):
            if not np.isnan(regrowth_array[i, j]):
                value = regrowth_array[i, j]
                # Add text annotation in white or black depending on the darkness of the cell
                # Add epsilon to avoid division by zero
                epsilon = 1e-10
                max_value = max(np.nanmax(regrowth_array), epsilon)
                text_color = "white" if value / max_value > 0.6 else "black"
                ax.text(j + 0.5, i + 0.5, f"{value:.2f}", 
                        ha="center", va="center", color=text_color, fontsize=8)
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    return fig