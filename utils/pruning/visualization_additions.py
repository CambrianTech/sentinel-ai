"""
Additional visualization utilities for neural plasticity demonstrations.
These functions support the NeuralPlasticityDemo notebook with improved
visualizations for attention patterns, gradient norms, and entropy metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
from IPython.display import display, HTML, clear_output
import time

class PersistentDisplay:
    """
    A persistent display widget that stays in place and updates its content.
    
    This class creates a display area that can be updated in-place,
    avoiding the creation of new output cells and making notebooks cleaner.
    """
    
    def __init__(
        self, 
        title: str = "Status Display",
        show_timestamp: bool = True,
        display_id: Optional[str] = None
    ):
        """
        Initialize a persistent display.
        
        Args:
            title: Title for the display area
            show_timestamp: Whether to show timestamp with updates
            display_id: Optional unique ID for this display
        """
        self.title = title
        self.show_timestamp = show_timestamp
        self.display_id = display_id or f"display_{int(time.time())}"
        self.update_count = 0
        
        # Create the initial display
        self._create_display()
    
    def _create_display(self):
        """Create the initial display container"""
        container_style = """
        padding: 10px; 
        border: 1px solid #ddd; 
        border-radius: 5px; 
        margin: 10px 0;
        background-color: #f8f9fa;
        """
        
        html = f"""
        <div id="{self.display_id}" style="{container_style}">
            <h3>{self.title}</h3>
            <p><i>Display will update when data is available...</i></p>
        </div>
        """
        display(HTML(html))
    
    def update(
        self, 
        content: str,
        clear: bool = True,
        notify: bool = False
    ):
        """
        Update the display with new content.
        
        Args:
            content: HTML content to display
            clear: Whether to clear previous output before updating
            notify: Flash display briefly to indicate update
        """
        self.update_count += 1
        
        # Generate timestamp if enabled
        timestamp = ""
        if self.show_timestamp:
            timestamp = f'<div style="font-size: 0.8em; color: #666; text-align: right;">Updated: {time.strftime("%H:%M:%S")}</div>'
        
        # Create container with optional notify animation
        container_style = """
        padding: 10px; 
        border: 1px solid #ddd; 
        border-radius: 5px; 
        margin: 10px 0;
        background-color: #f8f9fa;
        """
        
        # Add animation if notify is True
        if notify:
            container_style += """
            animation: flash-update 0.5s 1;
            """
            # Define the animation
            animation_style = """
            <style>
            @keyframes flash-update {
                0% { background-color: #f8f9fa; }
                50% { background-color: #e2f0fb; }
                100% { background-color: #f8f9fa; }
            }
            </style>
            """
        else:
            animation_style = ""
        
        # Build the HTML
        html = f"""
        {animation_style}
        <div id="{self.display_id}" style="{container_style}">
            <h3>{self.title}</h3>
            {content}
            {timestamp}
        </div>
        """
        
        # Update the display
        if clear:
            clear_output(wait=True)
        display(HTML(html))
    
    def update_with_metrics(
        self, 
        metrics: Dict[str, Any],
        metrics_format: Optional[Dict[str, str]] = None,
        layout: str = "grid",
        clear: bool = True,
        notify: bool = False
    ):
        """
        Update display with formatted metrics.
        
        Args:
            metrics: Dictionary of metrics to display
            metrics_format: Optional format strings for each metric
            layout: Layout style ('grid', 'list', 'table')
            clear: Whether to clear previous output
            notify: Whether to flash to indicate update
        """
        formats = metrics_format or {}
        
        # Generate HTML based on layout
        if layout == "grid":
            items_html = ""
            # Create grid of metrics
            items_html += '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px;">'
            for key, value in metrics.items():
                format_str = formats.get(key, "")
                if format_str and isinstance(value, (int, float)):
                    formatted_value = format_str.format(value)
                elif isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                
                # Generate appropriate color for numerical values
                color_style = ""
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if value > 0:
                        color_style = 'color: #28a745;'  # Green for positive
                    elif value < 0:
                        color_style = 'color: #dc3545;'  # Red for negative
                
                items_html += f"""
                <div style="padding: 8px; border: 1px solid #eee; border-radius: 4px;">
                    <div style="font-weight: bold;">{key}</div>
                    <div style="font-size: 1.2em; {color_style}">{formatted_value}</div>
                </div>
                """
            items_html += '</div>'
        
        elif layout == "table":
            # Create a table of metrics
            items_html = """
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background-color: #f1f1f1;">
                    <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Metric</th>
                    <th style="padding: 8px; text-align: right; border: 1px solid #ddd;">Value</th>
                </tr>
            """
            for key, value in metrics.items():
                format_str = formats.get(key, "")
                if format_str and isinstance(value, (int, float)):
                    formatted_value = format_str.format(value)
                elif isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                
                items_html += f"""
                <tr>
                    <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">{key}</td>
                    <td style="padding: 8px; text-align: right; border: 1px solid #ddd;">{formatted_value}</td>
                </tr>
                """
            items_html += "</table>"
        
        else:  # list layout
            # Create a simple list of metrics
            items_html = "<ul style='list-style-type: none; padding-left: 0;'>"
            for key, value in metrics.items():
                format_str = formats.get(key, "")
                if format_str and isinstance(value, (int, float)):
                    formatted_value = format_str.format(value)
                elif isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                
                items_html += f"<li><b>{key}:</b> {formatted_value}</li>"
            items_html += "</ul>"
        
        self.update(items_html, clear=clear, notify=notify)

    def update_with_figure(
        self,
        fig_or_func,
        caption: Optional[str] = None,
        clear: bool = True,
        notify: bool = False,
        **kwargs
    ):
        """
        Update with a matplotlib figure or a function that produces a figure.
        
        Args:
            fig_or_func: Either a matplotlib figure or a function that returns a figure
            caption: Optional caption for the figure
            clear: Whether to clear the output
            notify: Whether to flash update
            **kwargs: Additional arguments to pass to the function
        """
        # If a function is provided, call it to get the figure
        if callable(fig_or_func):
            fig = fig_or_func(**kwargs)
        else:
            fig = fig_or_func
        
        # Display the figure
        if clear:
            clear_output(wait=True)
        
        # Display the base container first
        container_style = """
        padding: 10px; 
        border: 1px solid #ddd; 
        border-radius: 5px; 
        margin: 10px 0;
        background-color: #f8f9fa;
        """
        
        header_html = f"""
        <div id="{self.display_id}" style="{container_style}">
            <h3>{self.title}</h3>
        """
        display(HTML(header_html))
        
        # Display the figure
        display(fig)
        
        # Add caption and timestamp if provided
        footer_html = ""
        if caption:
            footer_html += f'<div style="text-align: center; font-style: italic; margin-top: 5px;">{caption}</div>'
        
        if self.show_timestamp:
            footer_html += f'<div style="font-size: 0.8em; color: #666; text-align: right;">Updated: {time.strftime("%H:%M:%S")}</div>'
        
        footer_html += "</div>"
        display(HTML(footer_html))
        
        # Close the figure to free memory if it's a matplotlib figure
        if hasattr(fig, 'clf'):
            plt.close(fig)


class TrainingMonitor(PersistentDisplay):
    """
    Specialized persistent display for monitoring training progress.
    
    Provides visualizations focused on model training metrics, with
    special support for common machine learning metrics.
    """
    
    def __init__(
        self,
        title="Training Progress",
        metrics_to_track=None,
        show_timestamp=True,
        display_id=None
    ):
        """
        Initialize training monitor.
        
        Args:
            title: Title for the display
            metrics_to_track: List of metrics to track and display
            show_timestamp: Whether to show timestamp
            display_id: Optional unique ID
        """
        super().__init__(title=title, show_timestamp=show_timestamp, display_id=display_id)
        self.metrics_to_track = metrics_to_track or ["loss", "accuracy", "step", "epoch"]
        self.history = {metric: [] for metric in self.metrics_to_track}
    
    def update_metrics(self, metrics, step=None, epoch=None, clear=True, plot=True, notify=False):
        """
        Update training metrics and optionally plot progress.
        
        Args:
            metrics: Dictionary of current metrics
            step: Current training step
            epoch: Current epoch
            clear: Whether to clear output
            plot: Whether to plot progress charts
            notify: Whether to flash the display to indicate update
        """
        # Add step and epoch if provided
        if step is not None:
            metrics["step"] = step
        if epoch is not None:
            metrics["epoch"] = epoch
        
        # Update history
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
        
        # Create display content
        metrics_html = self._format_metrics_html(metrics)
        
        # Basic update without plots
        if not plot:
            self.update(metrics_html, clear=clear, notify=notify)
            return
        
        # Update with plots
        if clear:
            clear_output(wait=True)
        
        # Display header and metrics
        container_style = """
        padding: 10px; 
        border: 1px solid #ddd; 
        border-radius: 5px; 
        margin: 10px 0;
        background-color: #f8f9fa;
        """
        
        header_html = f"""
        <div id="{self.display_id}" style="{container_style}">
            <h3>{self.title}</h3>
            {metrics_html}
        """
        display(HTML(header_html))
        
        # Generate and display plots
        self._plot_training_progress()
        
        # Display timestamp and closing tag
        if self.show_timestamp:
            footer_html = f'<div style="font-size: 0.8em; color: #666; text-align: right;">Updated: {time.strftime("%H:%M:%S")}</div>'
            footer_html += "</div>"
            display(HTML(footer_html))
        else:
            display(HTML("</div>"))
    
    def _format_metrics_html(self, metrics):
        """Format metrics as an HTML grid display"""
        items_html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; margin-bottom: 15px;">'
        
        for key, value in metrics.items():
            # Format the value nicely
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            # Generate appropriate color for loss or accuracy
            color_style = ""
            if key.lower() in ("loss", "error"):
                color_style = 'color: #dc3545;'  # Red for loss
            elif "acc" in key.lower() or key.lower() in ("precision", "recall", "f1"):
                color_style = 'color: #28a745;'  # Green for accuracy metrics
            
            items_html += f"""
            <div style="padding: 8px; border: 1px solid #eee; border-radius: 4px;">
                <div style="font-weight: bold; font-size: 0.9em;">{key}</div>
                <div style="font-size: 1.1em; {color_style}">{formatted_value}</div>
            </div>
            """
        
        items_html += '</div>'
        return items_html
    
    def _plot_training_progress(self):
        """Generate and display training progress plots"""
        # Only plot if we have enough data
        if not all(len(values) > 1 for values in self.history.values() if values):
            return
        
        # Plot primary metrics (loss, accuracy, etc.)
        metrics_to_plot = ["loss", "accuracy", "val_loss", "val_accuracy"]
        available_metrics = [m for m in metrics_to_plot if m in self.history and len(self.history[m]) > 0]
        
        if available_metrics:
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 4))
            
            for metric in available_metrics:
                values = self.history[metric]
                if values:
                    # Plot with appropriate style
                    if "val" in metric:
                        ax.plot(values, '--', label=metric)
                    else:
                        ax.plot(values, '-', label=metric)
            
            ax.set_title("Training Progress")
            ax.set_xlabel("Step")
            ax.set_ylabel("Metric Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Use constrained_layout for better spacing
            plt.tight_layout()
            
            # Display the plot
            display(fig)
            plt.close(fig)
        
        # Plot any other interesting metrics
        other_metrics = [m for m in self.history.keys() 
                        if m not in metrics_to_plot + ["step", "epoch"]
                        and len(self.history[m]) > 0
                        and isinstance(self.history[m][0], (int, float))]
        
        if other_metrics:
            fig, ax = plt.subplots(figsize=(10, 4))
            
            for metric in other_metrics[:5]:  # Limit to 5 metrics for clarity
                values = self.history[metric]
                if values:
                    ax.plot(values, '-', label=metric)
            
            ax.set_title("Additional Metrics")
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            display(fig)
            plt.close(fig)


def visualize_gradient_norms(
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
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    # Convert to numpy if tensor
    if isinstance(grad_norm_values, torch.Tensor):
        grad_data = grad_norm_values.detach().cpu().numpy()
    else:
        grad_data = grad_norm_values
        
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    # Plot heatmap
    im = ax.imshow(grad_data, cmap=cmap, aspect="auto")
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
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


def visualize_attention_matrix(
    attn_matrix: torch.Tensor,
    layer_idx: int = 0,
    head_idx: int = 0,
    title: Optional[str] = None,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize an attention matrix for a specific layer and head.
    
    Args:
        attn_matrix: The attention tensor with shape [batch, heads, seq_len, seq_len]
        layer_idx: Layer index to visualize
        head_idx: Head index to visualize
        title: Optional title for the plot
        cmap: Colormap to use
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    # Convert to numpy if tensor
    if isinstance(attn_matrix, torch.Tensor):
        attn_np = attn_matrix[0, head_idx].detach().cpu().numpy()
    else:
        attn_np = attn_matrix[0, head_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    # Plot heatmap
    im = ax.imshow(attn_np, cmap=cmap)
    
    # Ensure proper scaling for attention values (0 to 1)
    im.set_clim(0, 1.0)
    
    # Add colorbar and labels
    plt.colorbar(im, ax=ax, label='Attention probability')
    plt.title(title or f'Attention pattern (layer {layer_idx}, head {head_idx})')
    plt.xlabel('Sequence position (to)')
    plt.ylabel('Sequence position (from)')
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
    return fig


def visualize_entropy_heatmap(
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
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    # Convert to numpy if tensor
    if isinstance(entropy_values, torch.Tensor):
        entropy_data = entropy_values.detach().cpu().numpy()
    else:
        entropy_data = entropy_values
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    # Plot heatmap
    im = ax.imshow(entropy_data, cmap=cmap, aspect='auto')
    
    # Set proper colormap limits with non-zero range
    max_value = np.max(entropy_data)
    im.set_clim(min_value, max(0.1, max_value))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Entropy')
    
    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Layer Index')
    
    # Add text annotations if requested
    if annotate:
        for i in range(entropy_data.shape[0]):
            for j in range(entropy_data.shape[1]):
                ax.text(j, i, f'{entropy_data[i, j]:.2f}',
                        ha="center", va="center", color="w")
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
    return fig


def visualize_normalized_entropy(
    entropy_values: torch.Tensor,
    max_entropy: torch.Tensor,
    title: str = "Normalized Entropy (% of Maximum)",
    figsize: Tuple[int, int] = (10, 6),
    cmap: str = "viridis",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize normalized entropy values as percentage of maximum possible entropy.
    
    Args:
        entropy_values: Tensor of entropy values with shape [layers, heads]
        max_entropy: Tensor or scalar of maximum possible entropy values
        title: Title for the plot
        figsize: Figure size (width, height) in inches
        cmap: Colormap to use
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    # Normalize and convert to percentage
    if isinstance(entropy_values, torch.Tensor) and isinstance(max_entropy, torch.Tensor):
        norm_entropies = (entropy_values / max_entropy.item() * 100).detach().cpu().numpy()
    elif isinstance(entropy_values, torch.Tensor):
        norm_entropies = (entropy_values / max_entropy * 100).detach().cpu().numpy()
    else:
        norm_entropies = (entropy_values / max_entropy * 100)
    
    # Create the heatmap
    im = ax.imshow(norm_entropies, cmap=cmap, aspect='auto', vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label='% of Max Entropy')
    ax.set_title(title)
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Layer Index')
    
    # Add text annotations for each cell
    for i in range(norm_entropies.shape[0]):
        for j in range(norm_entropies.shape[1]):
            ax.text(j, i, f'{norm_entropies[i, j]:.1f}%',
                    ha="center", va="center", color="w")
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
    return fig


def visualize_entropy_vs_gradient(
    entropy_values: torch.Tensor,
    grad_norm_values: torch.Tensor,
    title: str = "Entropy vs Gradient Relationship",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a scatter plot showing the relationship between entropy and gradient values.
    
    Args:
        entropy_values: Tensor of entropy values 
        grad_norm_values: Tensor of gradient norm values
        title: Title for the plot
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
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
    ax.scatter(entropy_flat, grad_flat, alpha=0.7)
    ax.set_xlabel('Entropy (higher = less focused)')
    ax.set_ylabel('Gradient Norm (higher = more impact)')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
    return fig


def visualize_training_progress(
    metrics_history: Dict[str, list],
    max_display_points: int = 100,
    figsize: Tuple[int, int] = (10, 8),
    show_perplexity: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize training progress including loss, pruning metrics and sparsity.
    
    Args:
        metrics_history: Dictionary containing training metrics history
        max_display_points: Maximum number of points to display (downsamples if needed)
        figsize: Figure size as (width, height)
        show_perplexity: Whether to show perplexity on a secondary axis
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    # Create figure with subplots, using constrained_layout for better spacing
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=figsize,
        dpi=100,
        sharex=True,
        constrained_layout=True
    )

    # Downsample data if needed
    display_steps = metrics_history["step"]
    if len(display_steps) > max_display_points:
        # Select evenly spaced points
        indices = np.linspace(0, len(display_steps) - 1, max_display_points).astype(int)
        display_steps = [metrics_history["step"][i] for i in indices]
        display_train_loss = [metrics_history["train_loss"][i] for i in indices]
        display_eval_loss = [metrics_history["eval_loss"][i] for i in indices]
        display_pruned_heads = [metrics_history["pruned_heads"][i] for i in indices]
        display_revived_heads = [metrics_history.get("revived_heads", [])[i] if i < len(metrics_history.get("revived_heads", [])) else 0 for i in indices]
        display_sparsity = [metrics_history["sparsity"][i] for i in indices]
        display_epoch = [metrics_history["epoch"][i] for i in indices]
        display_perplexity = [metrics_history.get("perplexity", [])[i] for i in indices] if "perplexity" in metrics_history and metrics_history["perplexity"] else []
    else:
        display_train_loss = metrics_history["train_loss"]
        display_eval_loss = metrics_history["eval_loss"]
        display_pruned_heads = metrics_history["pruned_heads"]
        display_revived_heads = metrics_history.get("revived_heads", [0] * len(display_steps))
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

    # Set reasonable y-limits for loss plot to avoid extreme scaling
    if len(display_eval_loss) > 0:
        loss_max = max(max(display_train_loss), max(display_eval_loss))
        # Cap the loss at a reasonable value to prevent excessive vertical scaling
        ax1.set_ylim(0, min(loss_max * 1.1, 10))

    # Mark epoch boundaries using transforms for better positioning
    if "epoch" in metrics_history and len(display_epoch) > 1:
        for i in range(1, len(display_epoch)):
            if display_epoch[i] != display_epoch[i-1]:
                # This is an epoch boundary - use transforms to fix text position
                for ax in [ax1, ax2, ax3]:
                    ax.axvline(x=display_steps[i], color="k", linestyle="--", alpha=0.3)
                    # Use axis transform to position text at fixed percent of axis height
                    ax.text(display_steps[i], 0.9, f"Epoch {display_epoch[i]}", 
                            rotation=90, alpha=0.7, va='top', transform=ax.get_xaxis_transform())

    # Plot pruning metrics
    ax2.bar(display_steps, display_pruned_heads, alpha=0.5, label="Pruned Heads", color="blue")
    
    # Only add revived heads if the data exists
    if any(display_revived_heads):
        ax2.bar(display_steps, display_revived_heads, alpha=0.5, label="Revived Heads", color="green")
    
    ax2.set_ylabel("Count")
    ax2.set_title("Head Pruning and Revival")
    ax2.legend(loc="upper left")
    ax2.grid(True)

    # Set reasonable y-limits for pruning plot
    if len(display_pruned_heads) > 0:
        pruning_max = max(max(display_pruned_heads), max(display_revived_heads) if display_revived_heads else 0)
        # Add some headroom but keep it reasonable
        ax2.set_ylim(0, min(pruning_max * 1.2 + 1, 10))

    # Plot sparsity
    ax3.plot(display_steps, display_sparsity, "r-", label="Sparsity")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Sparsity")
    ax3.grid(True)
    ax3.set_ylim(0, 1.0)  # Sparsity is always between 0 and 1

    # Add perplexity line on secondary axis if available and requested
    if show_perplexity and "perplexity" in metrics_history and len(display_perplexity) > 0:
        ax4 = ax3.twinx()
        perp_line = ax4.plot(display_steps, display_perplexity, "g-", label="Perplexity")
        ax4.set_ylabel("Perplexity", color="green")
        
        # Add text label for perplexity instead of a legend
        if len(display_perplexity) > 0:
            ax4.text(display_steps[-1], display_perplexity[-1], "Perplexity", 
                     color="green", va="center", ha="left", fontsize=9)
        
        # Set reasonable y-limits for perplexity
        if len(display_perplexity) > 0:
            perp_min = min(display_perplexity)
            perp_max = max(display_perplexity)
            perp_range = perp_max - perp_min
            # Give it some padding but keep it reasonable
            ax4.set_ylim(max(0, perp_min - 0.1 * perp_range), min(perp_max + 0.2 * perp_range, 100))
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
    return fig


def create_pruning_monitor():
    """
    Create a specialized monitor for pruning experiments.
    
    Returns:
        TrainingMonitor object configured for pruning experiments
    """
    return TrainingMonitor(
        title="Neural Plasticity Training Progress",
        metrics_to_track=[
            "step", "epoch", "train_loss", "eval_loss", 
            "pruned_heads", "revived_heads", "sparsity", "perplexity"
        ]
    )


def calculate_proper_entropy(attn_tensor, eps=1e-8):
    """
    Calculate entropy with proper normalization and numerical stability.
    
    Args:
        attn_tensor: Attention tensor of shape [batch, heads, seq_len, seq_len]
        eps: Small epsilon value for numerical stability
        
    Returns:
        Entropy values of shape [layers, heads]
    """
    # Get attention shape
    batch_size, num_heads, seq_len, _ = attn_tensor.shape
    
    # Reshape for processing
    attn_flat = attn_tensor.view(batch_size * num_heads * seq_len, -1)
    
    # Handle numerical issues - ensure positive values and proper normalization
    attn_flat = attn_flat.clamp(min=eps)
    attn_flat = attn_flat / attn_flat.sum(dim=-1, keepdim=True)
    
    # Calculate entropy: -sum(p * log(p))
    entropy = -torch.sum(attn_flat * torch.log(attn_flat), dim=-1)
    
    # Reshape back to per-head format and average
    entropy = entropy.view(batch_size, num_heads, seq_len)
    entropy = entropy.mean(dim=(0, 2))  # Average over batch and sequence
    
    # Normalize by maximum possible entropy (log of sequence length)
    max_entropy = torch.log(torch.tensor(attn_tensor.size(-1), dtype=torch.float, device=attn_tensor.device))
    
    # View as layers x heads
    return entropy, max_entropy


# Export all functions
__all__ = [
    'PersistentDisplay',
    'TrainingMonitor',
    'visualize_gradient_norms',
    'visualize_attention_matrix',
    'visualize_entropy_heatmap',
    'visualize_normalized_entropy',
    'visualize_entropy_vs_gradient',
    'visualize_training_progress',
    'create_pruning_monitor',
    'calculate_proper_entropy',
]