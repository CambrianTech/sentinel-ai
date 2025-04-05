#!/usr/bin/env python
"""
Entropy Rhythm Plot for Transformer Models

This module provides visualization tools for creating EEG-like
plots of attention entropy oscillations over time, revealing
rhythmic patterns in transformer attention adaptation.

Key applications:
1. Visualizing entropy changes across plasticity cycles
2. Detecting rhythmic patterns in attention head specialization
3. Creating temporal maps of model adaptation
4. Studying the "brain waves" of transformer adaptation
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import datetime

logger = logging.getLogger(__name__)


def load_entropy_journal(journal_path: str) -> pd.DataFrame:
    """
    Load entropy journal data from a JSONL file.
    
    Args:
        journal_path: Path to the entropy journal JSONL file
        
    Returns:
        DataFrame with entropy journal data
    """
    if not os.path.exists(journal_path):
        raise FileNotFoundError(f"Entropy journal not found at {journal_path}")
    
    # Read JSONL file
    entries = []
    with open(journal_path, 'r') as f:
        for line in f:
            entries.append(json.loads(line))
    
    # Convert to DataFrame
    df = pd.DataFrame(entries)
    
    logger.info(f"Loaded {len(df)} entropy journal entries")
    
    return df


def plot_entropy_rhythm(
    entropy_df: pd.DataFrame,
    save_path: Optional[str] = None,
    normalize: bool = True,
    smooth_window: int = 1,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = 'viridis',
    show_layer_boundaries: bool = True
) -> plt.Figure:
    """
    Create EEG-like rhythm plot of entropy over cycles.
    
    Args:
        entropy_df: DataFrame with entropy journal data
        save_path: Path to save the plot
        normalize: Whether to normalize entropy values
        smooth_window: Window size for smoothing entropy values
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        show_layer_boundaries: Whether to show layer boundaries
        
    Returns:
        Matplotlib figure
    """
    # Group by cycle, layer, and head
    grouped = entropy_df.groupby(['cycle_idx', 'layer_idx', 'head_idx'])['entropy'].mean().reset_index()
    
    # Get unique cycles, layers, and heads
    cycles = sorted(grouped['cycle_idx'].unique())
    layers = sorted(grouped['layer_idx'].unique())
    
    # Calculate total number of heads
    total_heads = 0
    head_counts = {}
    for layer in layers:
        num_heads = len(grouped[(grouped['cycle_idx'] == cycles[0]) & 
                               (grouped['layer_idx'] == layer)]['head_idx'].unique())
        head_counts[layer] = num_heads
        total_heads += num_heads
    
    # Create a matrix for each cycle's entropy values
    # Shape: (cycles, total_heads)
    entropy_matrix = np.zeros((len(cycles), total_heads))
    entropy_matrix.fill(np.nan)
    
    # Fill the matrix with entropy values
    for i, cycle in enumerate(cycles):
        cycle_data = grouped[grouped['cycle_idx'] == cycle]
        
        # Track position in flattened array
        head_pos = 0
        
        for layer in layers:
            layer_data = cycle_data[cycle_data['layer_idx'] == layer]
            for _, row in layer_data.iterrows():
                if head_pos < total_heads:  # Safety check
                    entropy_matrix[i, head_pos] = row['entropy']
                    head_pos += 1
    
    # Apply smoothing if requested
    if smooth_window > 1:
        from scipy.ndimage import uniform_filter1d
        for i in range(entropy_matrix.shape[1]):
            valid_mask = ~np.isnan(entropy_matrix[:, i])
            if np.sum(valid_mask) > smooth_window:
                valid_values = entropy_matrix[valid_mask, i]
                smoothed = uniform_filter1d(valid_values, size=smooth_window, mode='nearest')
                entropy_matrix[valid_mask, i] = smoothed
    
    # Normalize if requested
    if normalize:
        # Normalize each head's entropy to [0, 1] range
        for i in range(entropy_matrix.shape[1]):
            valid_mask = ~np.isnan(entropy_matrix[:, i])
            if np.sum(valid_mask) > 1:  # Need at least 2 points to normalize
                min_val = np.min(entropy_matrix[valid_mask, i])
                max_val = np.max(entropy_matrix[valid_mask, i])
                if max_val > min_val:  # Avoid division by zero
                    entropy_matrix[valid_mask, i] = (entropy_matrix[valid_mask, i] - min_val) / (max_val - min_val)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Plot the entropy matrix as an image
    plt.imshow(
        entropy_matrix,
        aspect='auto',
        cmap=cmap,
        interpolation='none'
    )
    
    # Set labels
    plt.xlabel('Head Index (flattened across layers)')
    plt.ylabel('Plasticity Cycle')
    
    # Set title
    if title:
        plt.title(title)
    else:
        plt.title('Entropy Rhythm Plot')
    
    # Add colorbar
    cbar = plt.colorbar()
    if normalize:
        cbar.set_label('Normalized Entropy')
    else:
        cbar.set_label('Entropy')
    
    # Set y-ticks to cycle indices
    plt.yticks(np.arange(len(cycles)), cycles)
    
    # Add layer boundaries if requested
    if show_layer_boundaries:
        # Calculate boundary positions
        boundaries = [0]
        cum_heads = 0
        for layer in layers[:-1]:  # Skip the last layer
            cum_heads += head_counts[layer]
            boundaries.append(cum_heads)
        
        # Add vertical lines for layer boundaries
        for boundary in boundaries:
            plt.axvline(x=boundary - 0.5, color='white', linestyle='--', alpha=0.5)
        
        # Add layer labels
        layer_centers = []
        cum_heads = 0
        for layer in layers:
            center = cum_heads + head_counts[layer] / 2
            layer_centers.append(center)
            cum_heads += head_counts[layer]
        
        plt.xticks(layer_centers, [f"L{layer}" for layer in layers])
    else:
        # Just use regular head indices
        plt.xticks(np.arange(0, total_heads, max(1, total_heads // 10)))
    
    plt.tight_layout()
    
    # Save the figure if requested
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved entropy rhythm plot to {save_path}")
    
    return fig


def create_animated_entropy_rhythm(
    entropy_df: pd.DataFrame,
    save_path: str,
    fps: int = 10,
    normalize: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = 'viridis'
) -> None:
    """
    Create an animated entropy rhythm plot showing the evolution over cycles.
    
    Args:
        entropy_df: DataFrame with entropy journal data
        save_path: Path to save the animation (mp4 or gif)
        fps: Frames per second for the animation
        normalize: Whether to normalize entropy values
        title: Animation title
        figsize: Figure size
        cmap: Colormap name
    """
    # Group by cycle, layer, and head
    grouped = entropy_df.groupby(['cycle_idx', 'layer_idx', 'head_idx'])['entropy'].mean().reset_index()
    
    # Get unique cycles, layers, and heads
    cycles = sorted(grouped['cycle_idx'].unique())
    layers = sorted(grouped['layer_idx'].unique())
    
    # Calculate total number of heads
    total_heads = 0
    head_counts = {}
    for layer in layers:
        num_heads = len(grouped[(grouped['cycle_idx'] == cycles[0]) & 
                               (grouped['layer_idx'] == layer)]['head_idx'].unique())
        head_counts[layer] = num_heads
        total_heads += num_heads
    
    # Create a 3D matrix for each cycle's entropy values
    # Shape: (cycles, layers, max_heads_per_layer)
    max_heads = max(head_counts.values())
    entropy_3d = np.zeros((len(cycles), len(layers), max_heads))
    entropy_3d.fill(np.nan)
    
    # Fill the 3D matrix with entropy values
    for i, cycle in enumerate(cycles):
        cycle_data = grouped[grouped['cycle_idx'] == cycle]
        
        for j, layer in enumerate(layers):
            layer_data = cycle_data[cycle_data['layer_idx'] == layer]
            
            for _, row in layer_data.iterrows():
                head_idx = row['head_idx']
                if head_idx < max_heads:  # Safety check
                    entropy_3d[i, j, head_idx] = row['entropy']
    
    # Normalize if requested
    if normalize:
        # Calculate min/max across all values
        valid_mask = ~np.isnan(entropy_3d)
        if np.sum(valid_mask) > 0:
            min_val = np.nanmin(entropy_3d)
            max_val = np.nanmax(entropy_3d)
            if max_val > min_val:  # Avoid division by zero
                entropy_3d = (entropy_3d - min_val) / (max_val - min_val)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    
    ax1 = plt.subplot(gs[0])  # Main heatmap
    ax2 = plt.subplot(gs[1])  # Time series line at bottom
    
    # Initial frame for animation
    im = ax1.imshow(
        np.zeros((len(layers), max_heads)),
        aspect='auto',
        cmap=cmap,
        interpolation='none',
        vmin=0,
        vmax=1 if normalize else np.nanmax(entropy_3d)
    )
    
    # Set title
    if title:
        ax1.set_title(f"{title} - Cycle: {cycles[0]}")
    else:
        ax1.set_title(f"Entropy Rhythm - Cycle: {cycles[0]}")
    
    # Layer and head labels
    ax1.set_ylabel('Layer')
    ax1.set_xlabel('Head Index')
    ax1.set_yticks(np.arange(len(layers)))
    ax1.set_yticklabels(layers)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax1)
    if normalize:
        cbar.set_label('Normalized Entropy')
    else:
        cbar.set_label('Entropy')
    
    # Initialize line plot
    # Plot average entropy per layer as a time series
    layer_avgs = np.nanmean(entropy_3d, axis=2)  # Average across heads
    lines = []
    for j, layer in enumerate(layers):
        line, = ax2.plot(cycles[:1], layer_avgs[:1, j], label=f"Layer {layer}")
        lines.append(line)
    
    ax2.set_xlim(cycles[0], cycles[-1])
    ax2.set_ylim(0, 1 if normalize else np.nanmax(layer_avgs))
    ax2.set_xlabel('Cycle')
    ax2.set_ylabel('Avg Entropy')
    ax2.legend(fontsize='small')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Vertical line to show current cycle
    vline = ax2.axvline(x=cycles[0], color='red', linestyle='--')
    
    # Update function for animation
    def update(frame):
        # Update heatmap
        im.set_array(entropy_3d[frame])
        
        # Update title
        if title:
            ax1.set_title(f"{title} - Cycle: {cycles[frame]}")
        else:
            ax1.set_title(f"Entropy Rhythm - Cycle: {cycles[frame]}")
        
        # Update time series lines
        for j, line in enumerate(lines):
            line.set_data(cycles[:frame+1], layer_avgs[:frame+1, j])
        
        # Update vertical line
        vline.set_xdata(cycles[frame])
        
        return [im, vline] + lines
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(cycles), 
        blit=True, interval=1000/fps
    )
    
    # Save animation
    if save_path.endswith('.gif'):
        ani.save(save_path, writer='pillow', fps=fps)
    else:
        # Default to mp4
        ani.save(save_path, writer='ffmpeg', fps=fps)
    
    logger.info(f"Saved animated entropy rhythm to {save_path}")
    
    plt.close(fig)


def create_entropy_delta_heatmap(
    entropy_df: pd.DataFrame,
    save_path: Optional[str] = None,
    cycle_pairs: Optional[List[Tuple[int, int]]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create a heatmap showing entropy changes between cycle pairs.
    
    Args:
        entropy_df: DataFrame with entropy journal data
        save_path: Path to save the plot
        cycle_pairs: List of (before, after) cycle pairs to compare
                     If None, consecutive cycles are used
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Group by cycle, layer, and head
    grouped = entropy_df.groupby(['cycle_idx', 'layer_idx', 'head_idx'])['entropy'].mean().reset_index()
    
    # Get unique cycles and determine pairs
    cycles = sorted(grouped['cycle_idx'].unique())
    
    if cycle_pairs is None:
        # Default to consecutive cycles
        cycle_pairs = [(cycles[i], cycles[i+1]) for i in range(len(cycles)-1)]
    
    # Get unique layers and counts of heads per layer
    layers = sorted(grouped['layer_idx'].unique())
    head_counts = {}
    for layer in layers:
        # Count heads in first cycle
        heads = grouped[(grouped['cycle_idx'] == cycles[0]) & 
                       (grouped['layer_idx'] == layer)]['head_idx'].unique()
        head_counts[layer] = len(heads)
    
    # Calculate total number of comparisons
    n_comparisons = len(cycle_pairs)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_comparisons, figsize=figsize, sharey=True)
    if n_comparisons == 1:
        axes = [axes]  # Make it iterable
    
    # Custom diverging colormap (blue for decrease, white for no change, red for increase)
    cmap = LinearSegmentedColormap.from_list(
        'delta_entropy',
        ['blue', 'white', 'red'],
        N=256
    )
    
    # Process each pair
    for i, (before, after) in enumerate(cycle_pairs):
        ax = axes[i]
        
        # Get data for each cycle
        before_data = grouped[grouped['cycle_idx'] == before]
        after_data = grouped[grouped['cycle_idx'] == after]
        
        # Create delta matrix
        # Rows are flattened (layer, head) pairs
        rows = []
        layer_boundaries = [0]
        current_row = 0
        
        for layer in layers:
            num_heads = head_counts[layer]
            
            for head_idx in range(num_heads):
                # Get entropy values
                before_val = before_data[(before_data['layer_idx'] == layer) & 
                                        (before_data['head_idx'] == head_idx)]['entropy'].values
                
                after_val = after_data[(after_data['layer_idx'] == layer) & 
                                      (after_data['head_idx'] == head_idx)]['entropy'].values
                
                if len(before_val) > 0 and len(after_val) > 0:
                    delta = after_val[0] - before_val[0]
                else:
                    delta = np.nan
                
                rows.append(delta)
            
            current_row += num_heads
            layer_boundaries.append(current_row)
        
        # Reshape to match layer structure (for visual clarity)
        total_heads = sum(head_counts.values())
        delta_matrix = np.array(rows).reshape(1, total_heads)
        
        # Plot delta matrix
        im = ax.imshow(
            delta_matrix,
            aspect='auto',
            cmap=cmap,
            interpolation='none'
        )
        
        # Set title and labels
        ax.set_title(f"Cycles {before} → {after}")
        
        # Show layer boundaries
        for boundary in layer_boundaries[:-1]:
            ax.axvline(x=boundary - 0.5, color='black', linestyle='--', alpha=0.3)
        
        # Set x-ticks to layer centers
        layer_centers = [(layer_boundaries[i] + layer_boundaries[i+1]) // 2 
                         for i in range(len(layer_boundaries)-1)]
        ax.set_xticks(layer_centers)
        ax.set_xticklabels([f"L{layer}" for layer in layers])
        
        # Remove y-ticks (only one row)
        ax.set_yticks([])
    
    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Entropy Change (Δ)')
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle("Entropy Change Between Cycles", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Save the figure if requested
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved entropy delta heatmap to {save_path}")
    
    return fig


def plot_entropy_rhythm_from_file(
    journal_path: str,
    save_path: Optional[str] = None,
    normalize: bool = True,
    smooth_window: int = 1,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = 'viridis',
    show_layer_boundaries: bool = True
) -> plt.Figure:
    """
    Create EEG-like rhythm plot of entropy from a journal file.
    
    Args:
        journal_path: Path to the entropy journal JSONL file
        save_path: Path to save the plot
        normalize: Whether to normalize entropy values
        smooth_window: Window size for smoothing entropy values
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        show_layer_boundaries: Whether to show layer boundaries
        
    Returns:
        Matplotlib figure
    """
    # Load entropy journal
    df = load_entropy_journal(journal_path)
    
    # Create plot
    return plot_entropy_rhythm(
        df, save_path, normalize, smooth_window,
        title, figsize, cmap, show_layer_boundaries
    )


def create_animated_entropy_rhythm_from_file(
    journal_path: str,
    save_path: str,
    fps: int = 10,
    normalize: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = 'viridis'
) -> None:
    """
    Create an animated entropy rhythm plot from a journal file.
    
    Args:
        journal_path: Path to the entropy journal JSONL file
        save_path: Path to save the animation (mp4 or gif)
        fps: Frames per second for the animation
        normalize: Whether to normalize entropy values
        title: Animation title
        figsize: Figure size
        cmap: Colormap name
    """
    # Load entropy journal
    df = load_entropy_journal(journal_path)
    
    # Create animation
    create_animated_entropy_rhythm(
        df, save_path, fps, normalize, title, figsize, cmap
    )


if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Create entropy rhythm plots for transformer models")
    parser.add_argument("journal_path", type=str, help="Path to the entropy journal JSONL file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to save the output")
    parser.add_argument("--animated", "-a", action="store_true", help="Create animated visualization")
    parser.add_argument("--normalize", "-n", action="store_true", help="Normalize entropy values")
    parser.add_argument("--smooth", "-s", type=int, default=1, help="Window size for smoothing")
    parser.add_argument("--title", "-t", type=str, default=None, help="Plot title")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for animation")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Determine output path if not specified
    if args.output is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.animated:
            args.output = f"entropy_rhythm_{timestamp}.mp4"
        else:
            args.output = f"entropy_rhythm_{timestamp}.png"
    
    # Create visualization
    if args.animated:
        create_animated_entropy_rhythm_from_file(
            args.journal_path,
            args.output,
            fps=args.fps,
            normalize=args.normalize,
            title=args.title
        )
    else:
        plot_entropy_rhythm_from_file(
            args.journal_path,
            args.output,
            normalize=args.normalize,
            smooth_window=args.smooth,
            title=args.title
        )
        
    logger.info(f"Visualization created successfully: {args.output}")