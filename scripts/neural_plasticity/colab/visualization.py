"""
Visualization utilities for neural plasticity in Colab.

This module provides functions for creating visualizations that work in both
Colab and local environments, handling the differences in display options.
"""

import os
import numpy as np
import json
from typing import Dict, Any, Optional, Union, List, Tuple

# Local imports
from scripts.neural_plasticity.colab.integration import is_colab, is_apple_silicon

def initialize_visualization():
    """
    Set up the visualization environment.
    
    This function configures matplotlib based on the current environment:
    - In Colab: Uses inline backend with widget support
    - On Apple Silicon: Uses Agg backend to avoid crashes
    - Otherwise: Uses interactive backend
    """
    import matplotlib
    
    if is_colab():
        # Colab-specific setup
        matplotlib.use('inline')
        try:
            # This code will only work in Colab with IPython
            get_ipython().run_line_magic('matplotlib', 'inline')
            from google.colab import output
            output.enable_custom_widget_manager()
            print("✅ Colab visualization environment initialized")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize Colab widgets: {e}")
    
    elif is_apple_silicon():
        # Use Agg backend for Apple Silicon to avoid crashes
        matplotlib.use('Agg')
        print("🍎 Apple Silicon detected - using Agg backend for stability")
    
    else:
        # Use TkAgg for standard environments
        try:
            matplotlib.use('TkAgg')
        except Exception:
            # Fall back to Agg if TkAgg is not available
            matplotlib.use('Agg')
            print("⚠️ Using Agg backend (TkAgg not available)")

def safe_imshow(data, **kwargs):
    """
    Safe wrapper for matplotlib imshow that works across environments.
    
    Args:
        data: Data to display as an image
        **kwargs: Additional arguments for imshow
        
    Returns:
        The imshow object
    """
    import matplotlib.pyplot as plt
    
    # Handle tensor inputs safely
    if hasattr(data, 'detach') and hasattr(data, 'cpu') and hasattr(data, 'numpy'):
        data = data.detach().cpu().numpy()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=kwargs.pop('figsize', (10, 8)))
    im = ax.imshow(data, **kwargs)
    
    # Add colorbar if not explicitly disabled
    if kwargs.pop('colorbar', True):
        plt.colorbar(im, ax=ax)
    
    # Set title if provided
    if 'title' in kwargs:
        ax.set_title(kwargs.pop('title'))
    
    # Set labels if provided
    if 'xlabel' in kwargs:
        ax.set_xlabel(kwargs.pop('xlabel'))
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs.pop('ylabel'))
    
    # In Colab, we need to show the plot
    if is_colab():
        plt.tight_layout()
        plt.show()
    
    return im

def plot_entropy_heatmap(entropy_values, title="Attention Entropy", layer_idx=None, 
                        figsize=(10, 6), save_path=None):
    """
    Plot entropy heatmap that works in both environments.
    
    Args:
        entropy_values: 2D array of entropy values
        title: Title for the plot
        layer_idx: Optional layer index to include in title
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        The figure object
    """
    import matplotlib.pyplot as plt
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate heatmap
    im = ax.imshow(entropy_values, cmap='viridis')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Set title with layer info if provided
    if layer_idx is not None:
        ax.set_title(f"{title} (Layer {layer_idx})")
    else:
        ax.set_title(title)
    
    # Axis labels
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index" if layer_idx is None else "Attention Position")
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show plot in Colab
    if is_colab():
        plt.tight_layout()
        plt.show()
    
    return fig

def plot_training_progress(metrics, title="Training Progress", figsize=(12, 6), 
                          save_path=None):
    """
    Plot training progress metrics.
    
    Args:
        metrics: Dictionary with metrics (keys are metric names, values are lists of values)
        title: Title for the plot
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        The figure object
    """
    import matplotlib.pyplot as plt
    
    # Create the figure
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    
    # Handle single metric case
    if len(metrics) == 1:
        axes = [axes]
    
    # Plot each metric
    for i, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[i]
        ax.plot(values)
        ax.set_title(metric_name)
        ax.set_xlabel("Step")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show plot in Colab
    if is_colab():
        plt.show()
    
    return fig

def visualize_pruned_heads(pruned_heads, model_name, pruning_strategy, 
                         num_layers=None, heads_per_layer=None, figsize=(10, 8),
                         save_path=None):
    """
    Visualize which heads were pruned across the model.
    
    Args:
        pruned_heads: List of (layer_idx, head_idx, score) tuples
        model_name: Name of the model
        pruning_strategy: Strategy used for pruning
        num_layers: Optional number of layers in the model
        heads_per_layer: Optional number of heads per layer
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        The figure object
    """
    import matplotlib.pyplot as plt
    
    # Determine model dimensions if not provided
    if num_layers is None or heads_per_layer is None:
        # Extract dimensions from pruned_heads
        if pruned_heads:
            max_layer = max(h[0] for h in pruned_heads) + 1
            max_head = max(h[1] for h in pruned_heads) + 1
            num_layers = num_layers or max_layer
            heads_per_layer = heads_per_layer or max_head
        else:
            # Default dimensions for common models
            if "gpt2" in model_name.lower():
                num_layers = 12
                heads_per_layer = 12
            elif "bloom" in model_name.lower():
                num_layers = 24
                heads_per_layer = 16
            else:
                num_layers = 12
                heads_per_layer = 12
    
    # Create matrix of pruned status (1 = pruned, 0 = kept)
    pruned_matrix = np.zeros((num_layers, heads_per_layer))
    
    for layer_idx, head_idx, _ in pruned_heads:
        if layer_idx < num_layers and head_idx < heads_per_layer:
            pruned_matrix[layer_idx, head_idx] = 1
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate heatmap
    im = ax.imshow(pruned_matrix, cmap='Reds', interpolation='none')
    
    # Set title
    ax.set_title(f"Pruned Heads ({model_name}, {pruning_strategy})")
    
    # Axis labels
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")
    
    # Add text to show 1s (pruned heads)
    for i in range(num_layers):
        for j in range(heads_per_layer):
            if pruned_matrix[i, j] == 1:
                ax.text(j, i, "X", ha="center", va="center", color="white")
    
    # Add grid
    ax.set_xticks(np.arange(-.5, heads_per_layer, 1), minor=True)
    ax.set_yticks(np.arange(-.5, num_layers, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1, alpha=0.2)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.set_ticklabels(['Kept', 'Pruned'])
    
    # Add text with pruning statistics
    total_heads = num_layers * heads_per_layer
    pruned_count = len(pruned_heads)
    pruning_percentage = (pruned_count / total_heads) * 100
    
    plt.figtext(0.5, -0.05, 
               f"Total: {pruned_count}/{total_heads} heads pruned ({pruning_percentage:.1f}%)",
               ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Show plot in Colab
    if is_colab():
        plt.show()
    
    return fig

def display_metrics_comparison(before_metrics, after_metrics, title="Before vs After Pruning"):
    """
    Display a comparison of metrics before and after pruning.
    
    Args:
        before_metrics: Dictionary of metrics before pruning
        after_metrics: Dictionary of metrics after pruning
        title: Title for the comparison
        
    Returns:
        In Colab: Displays HTML table
        Otherwise: Returns a formatted string
    """
    # Combine metrics
    all_metrics = set(list(before_metrics.keys()) + list(after_metrics.keys()))
    
    # Create formatted output
    if is_colab():
        from IPython.display import display, HTML
        
        # Create HTML table
        html = f"<h3>{title}</h3>"
        html += "<table style='border-collapse: collapse; width: 100%;'>"
        html += "<tr><th style='border: 1px solid black; padding: 8px; text-align: left;'>Metric</th>"
        html += "<th style='border: 1px solid black; padding: 8px; text-align: left;'>Before</th>"
        html += "<th style='border: 1px solid black; padding: 8px; text-align: left;'>After</th>"
        html += "<th style='border: 1px solid black; padding: 8px; text-align: left;'>Change</th></tr>"
        
        for metric in sorted(all_metrics):
            before = before_metrics.get(metric, "N/A")
            after = after_metrics.get(metric, "N/A")
            
            # Calculate change if both values exist and are numeric
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                abs_change = after - before
                rel_change = (abs_change / before) * 100 if before != 0 else float('inf')
                
                # Determine if change is positive (improvement) or negative
                is_improvement = (metric.lower() in ["accuracy", "f1", "precision", "recall"] and rel_change > 0) or \
                                (metric.lower() in ["loss", "perplexity", "error"] and rel_change < 0)
                
                change_color = "green" if is_improvement else "red"
                change_text = f"{abs_change:.4f} ({rel_change:+.2f}%)"
            else:
                change_text = "N/A"
                change_color = "black"
            
            html += f"<tr><td style='border: 1px solid black; padding: 8px;'>{metric}</td>"
            html += f"<td style='border: 1px solid black; padding: 8px;'>{before}</td>"
            html += f"<td style='border: 1px solid black; padding: 8px;'>{after}</td>"
            html += f"<td style='border: 1px solid black; padding: 8px; color: {change_color};'>{change_text}</td></tr>"
        
        html += "</table>"
        display(HTML(html))
        
        # Return HTML string for saving if needed
        return html
    
    else:
        # Create text table for non-Colab environments
        result = f"\n{title}\n" + "-" * 80 + "\n"
        result += f"{'Metric':<20} {'Before':<15} {'After':<15} {'Change':<20}\n"
        result += "-" * 80 + "\n"
        
        for metric in sorted(all_metrics):
            before = before_metrics.get(metric, "N/A")
            after = after_metrics.get(metric, "N/A")
            
            # Calculate change if both values exist and are numeric
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                abs_change = after - before
                rel_change = (abs_change / before) * 100 if before != 0 else float('inf')
                change_text = f"{abs_change:.4f} ({rel_change:+.2f}%)"
            else:
                change_text = "N/A"
            
            result += f"{metric:<20} {str(before):<15} {str(after):<15} {change_text:<20}\n"
        
        result += "-" * 80 + "\n"
        print(result)
        
        return result