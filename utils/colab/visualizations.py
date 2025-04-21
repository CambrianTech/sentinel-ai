"""
Visualization utilities for Colab notebooks.

These utilities provide interactive and efficient visualization
capabilities for Colab notebooks, including:
- Persistent visualization widgets
- Memory-efficient display methods
- Real-time updating visualizations
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from IPython.display import display, HTML, clear_output
import time
import re

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

    def update_with_multi_figs(
        self,
        figs_or_funcs,
        captions=None,
        layout="vertical",
        clear=True,
        notify=False,
        **kwargs
    ):
        """
        Update with multiple figures or plotting functions.
        
        Args:
            figs_or_funcs: List of figures or functions that return figures
            captions: Optional list of captions
            layout: 'vertical', 'horizontal', or 'grid'
            clear: Whether to clear output
            notify: Whether to flash update
            **kwargs: Additional arguments for plotting functions
        """
        if clear:
            clear_output(wait=True)
        
        # Display the header
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
        
        # Create layout container for figures
        if layout == "horizontal":
            display(HTML("<div style='display: flex; flex-wrap: wrap; gap: 10px;'>"))
        elif layout == "grid":
            display(HTML("<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 15px;'>"))
        
        # Process and display each figure
        for i, fig_func in enumerate(figs_or_funcs):
            # If layout requires, add figure container
            if layout != "vertical":
                display(HTML("<div style='flex: 1; min-width: 300px;'>"))
            
            # Get the figure
            if callable(fig_func):
                fig = fig_func(**kwargs)
            else:
                fig = fig_func
            
            # Display the figure
            display(fig)
            
            # Add caption if provided
            if captions and i < len(captions) and captions[i]:
                display(HTML(f'<div style="text-align: center; font-style: italic; margin-top: 5px;">{captions[i]}</div>'))
            
            # Close figure container if needed
            if layout != "vertical":
                display(HTML("</div>"))
            
            # Close the figure to free memory
            if hasattr(fig, 'clf'):
                plt.close(fig)
        
        # Close layout container
        if layout != "vertical":
            display(HTML("</div>"))
        
        # Add timestamp
        if self.show_timestamp:
            footer_html = f'<div style="font-size: 0.8em; color: #666; text-align: right;">Updated: {time.strftime("%H:%M:%S")}</div>'
            footer_html += "</div>"
            display(HTML(footer_html))
        else:
            display(HTML("</div>"))


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
    
    def update_metrics(self, metrics, step=None, epoch=None, clear=True, plot=True):
        """
        Update training metrics and optionally plot progress.
        
        Args:
            metrics: Dictionary of current metrics
            step: Current training step
            epoch: Current epoch
            clear: Whether to clear output
            plot: Whether to plot progress charts
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
            self.update(metrics_html, clear=clear)
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
            
            display(fig)
            plt.close(fig)


def visualize_attention_heatmap(
    attention_matrix: Union[torch.Tensor, np.ndarray],
    layer_idx: int = 0,
    head_idx: int = 0,
    title: Optional[str] = None,
    cmap: str = "viridis",
    show_colorbar: bool = True,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Create a heatmap visualization of attention patterns.
    
    Args:
        attention_matrix: Attention matrix tensor with shape [batch, heads, seq_len, seq_len]
        layer_idx: Index of layer to visualize
        head_idx: Index of attention head to visualize
        title: Title for the plot
        cmap: Colormap to use
        show_colorbar: Whether to show a colorbar
        figsize: Figure size (width, height) in inches
        
    Returns:
        matplotlib Figure object
    """
    # Convert to numpy if tensor
    if isinstance(attention_matrix, torch.Tensor):
        attn_np = attention_matrix[0, head_idx].detach().cpu().numpy()
    else:
        attn_np = attention_matrix[0, head_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(attn_np, cmap=cmap)
    
    # Set proper limits for attention values (0 to 1)
    im.set_clim(0, 1.0)
    
    # Add colorbar if requested
    if show_colorbar:
        plt.colorbar(im, ax=ax, label='Attention probability')
    
    # Add title and labels
    ax.set_title(title or f'Attention Pattern (Layer {layer_idx}, Head {head_idx})')
    ax.set_xlabel('Position (to)')
    ax.set_ylabel('Position (from)')
    
    fig.tight_layout()
    return fig


def visualize_gradient_norms(
    grad_norm_values: torch.Tensor, 
    pruned_heads: Optional[List[Tuple[int, int]]] = None, 
    revived_heads: Optional[List[Tuple[int, int]]] = None, 
    title: str = "Gradient Norms", 
    figsize: Tuple[int, int] = (10, 5),
    cmap: str = "plasma",
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
    
    fig.tight_layout()
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


def visualize_head_entropy(
    entropy_values: Union[torch.Tensor, np.ndarray],
    title: str = "Attention Entropy Heatmap",
    min_value: float = 0.0,
    annotate: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    cmap: str = "viridis",
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
    
    fig.tight_layout()
    return fig


def extract_complete_training_data(experiment):
    """
    Extract comprehensive training data from the experiment object.
    
    Args:
        experiment: NeuralPlasticityExperiment object or dictionary with experiment results
        
    Returns:
        Dictionary containing all necessary data for visualization:
        - warmup_losses: List of warmup phase losses 
        - warmup_steps: List of warmup phase steps
        - stabilization_point: Index where warmup stabilized
        - pruning_losses: List of losses during the pruning phase
        - pruning_steps: List of steps during the pruning phase
        - fine_tuning_losses: List of losses during fine-tuning
        - fine_tuning_steps: List of steps during fine-tuning
        - perplexity_scores: List of perplexity scores throughout all phases
        - sparsity_history: List of model sparsity values
        - pruned_head_counts: List of cumulative pruned head counts
        - phase_markers: Dictionary with indices for transitions between phases
    """
    data = {
        "warmup_losses": [],
        "warmup_steps": [],
        "stabilization_point": None,
        "pruning_losses": [],
        "pruning_steps": [],
        "fine_tuning_losses": [],
        "fine_tuning_steps": [],
        "perplexity_scores": [],
        "sparsity_history": [],
        "pruned_head_counts": [],
        "phase_markers": {}
    }
    
    # Extract from NeuralPlasticityExperiment object
    if hasattr(experiment, 'results') and isinstance(experiment.results, dict):
        results = experiment.results
    elif isinstance(experiment, dict):
        results = experiment
    else:
        # Try to access experiment data through other common attributes
        results = {}
        if hasattr(experiment, 'warmup_results'):
            results['warmup'] = experiment.warmup_results
        if hasattr(experiment, 'pruning_results'):
            results['pruning'] = experiment.pruning_results
        if hasattr(experiment, 'fine_tuning_results'):
            results['fine_tuning'] = experiment.fine_tuning_results
    
    # Extract warmup phase data
    warmup_data = results.get('warmup', {})
    if isinstance(warmup_data, dict):
        data['warmup_losses'] = warmup_data.get('losses', [])
        data['warmup_steps'] = list(range(len(data['warmup_losses'])))
        data['stabilization_point'] = warmup_data.get('stabilization_point')
        
        # If stabilization_point is not directly available, calculate from other fields
        if data['stabilization_point'] is None and warmup_data.get('is_stable'):
            steps_without_decrease = warmup_data.get('steps_without_decrease', 0)
            if steps_without_decrease > 0:
                data['stabilization_point'] = len(data['warmup_losses']) - steps_without_decrease
    
    # Extract pruning phase data
    pruning_data = results.get('pruning', {})
    if isinstance(pruning_data, dict):
        # Extract training metrics during pruning phase
        metrics = pruning_data.get('training_metrics', {})
        if metrics:
            data['pruning_losses'] = metrics.get('train_loss', [])
            data['pruning_steps'] = metrics.get('step', list(range(len(data['pruning_losses']))))
            pruning_perplexity = metrics.get('perplexity', [])
            data['perplexity_scores'].extend(pruning_perplexity)
            data['sparsity_history'] = metrics.get('sparsity', [])
            data['pruned_head_counts'] = metrics.get('pruned_heads', [])
    
    # Extract fine-tuning phase data
    fine_tuning_data = results.get('fine_tuning', {})
    if isinstance(fine_tuning_data, dict):
        # Extract training metrics during fine-tuning phase
        metrics = fine_tuning_data.get('training_metrics', {})
        if metrics:
            data['fine_tuning_losses'] = metrics.get('train_loss', [])
            data['fine_tuning_steps'] = metrics.get('step', list(range(len(data['fine_tuning_losses']))))
            ft_perplexity = metrics.get('perplexity', [])
            data['perplexity_scores'].extend(ft_perplexity)
    
    # Calculate phase markers for visualization
    # Start of warmup is always 0
    data['phase_markers']['warmup_start'] = 0
    
    # End of warmup / start of pruning
    warmup_length = len(data['warmup_losses'])
    data['phase_markers']['warmup_end'] = warmup_length
    data['phase_markers']['pruning_start'] = warmup_length
    
    # End of pruning / start of fine-tuning
    pruning_length = len(data['pruning_losses'])
    data['phase_markers']['pruning_end'] = warmup_length + pruning_length
    data['phase_markers']['fine_tuning_start'] = warmup_length + pruning_length
    
    # End of fine-tuning
    fine_tuning_length = len(data['fine_tuning_losses'])
    data['phase_markers']['fine_tuning_end'] = warmup_length + pruning_length + fine_tuning_length
    
    return data


def generate_inspirational_quote():
    """
    Generate an inspirational quote about neural networks and plasticity.
    
    Returns:
        A random inspirational quote
    """
    quotes = [
        "The brain's plasticity is its most fascinating feature - so is your model's.",
        "Adapt, prune, refine, learn. That is the path to AI excellence.",
        "In neural network pruning, less is often more.",
        "The most efficient networks are not built, they are grown and pruned like gardens.",
        "A carefully pruned network is like a well-written sentence - nothing extra, nothing missing.",
        "Plasticity is not an accident of nature, but its core design principle.",
        "The art of AI is knowing which connections to strengthen and which to prune.",
        "In the neural world, connections that fire together wire together, and those that don't, don't.",
        "The wise gardener prunes to promote growth. The wise AI researcher does the same.",
        "Every neural connection serves a purpose or makes way for one that does.",
        "Neural plasticity teaches us that adaptation is not just a response to the environment - it is survival.",
        "Complexity should be pruned, not pursued."
    ]
    
    # Choose random quote
    import random
    return random.choice(quotes)


def visualize_complete_training_process(
    experiment,
    output_dir: Optional[str] = None,
    filename: Optional[str] = None,
    title: str = "Complete Neural Plasticity Training Process",
    show_plot: bool = True,
    figsize: Tuple[int, int] = (14, 10),
    show_quote: bool = True
) -> plt.Figure:
    """
    Create a comprehensive visualization of the entire neural plasticity process,
    showing warmup, stabilization, pruning, and fine-tuning phases.
    
    Args:
        experiment: NeuralPlasticityExperiment object or dictionary with experiment results
        output_dir: Directory to save the visualization (created if it doesn't exist)
        filename: Filename for saved visualization
        title: Title for the visualization
        show_plot: Whether to display the plot (set to False for headless environments)
        figsize: Figure size (width, height) in inches
        show_quote: Whether to include an inspirational quote
        
    Returns:
        matplotlib Figure object
    """
    # Extract all training data from experiment
    training_data = extract_complete_training_data(experiment)
    
    # Concatenate all losses and steps for comprehensive visualization
    all_losses = []
    all_steps = []
    
    # Add warmup phase data
    warmup_losses = training_data['warmup_losses']
    warmup_steps = training_data['warmup_steps']
    all_losses.extend(warmup_losses)
    all_steps.extend(warmup_steps)
    
    # Add pruning phase data
    pruning_losses = training_data['pruning_losses']
    pruning_steps = [step + len(warmup_steps) for step in training_data['pruning_steps']]
    all_losses.extend(pruning_losses)
    all_steps.extend(pruning_steps)
    
    # Add fine-tuning phase data
    fine_tuning_losses = training_data['fine_tuning_losses']
    fine_tuning_steps = [step + len(warmup_steps) + len(pruning_steps) for step in training_data['fine_tuning_steps']]
    all_losses.extend(fine_tuning_losses)
    all_steps.extend(fine_tuning_steps)
    
    # Create a multi-panel figure
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(4, 3, figure=fig, height_ratios=[3, 2, 2, 1])
    
    # 1. Main Plot: Complete Training Process
    ax_main = fig.add_subplot(gs[0, :])
    
    # Plot all losses
    if all_losses:
        ax_main.plot(all_steps, all_losses, 'b-', label='Training Loss')
    
    # Add phase markers
    markers = training_data['phase_markers']
    
    # Warmup phase highlighting
    if 'warmup_start' in markers and 'warmup_end' in markers:
        ax_main.axvspan(markers['warmup_start'], markers['warmup_end'], 
                       alpha=0.2, color='blue', label='Warmup Phase')
    
    # Pruning phase highlighting
    if 'pruning_start' in markers and 'pruning_end' in markers:
        ax_main.axvspan(markers['pruning_start'], markers['pruning_end'], 
                       alpha=0.2, color='red', label='Pruning Phase')
    
    # Fine-tuning phase highlighting
    if 'fine_tuning_start' in markers and 'fine_tuning_end' in markers:
        ax_main.axvspan(markers['fine_tuning_start'], markers['fine_tuning_end'], 
                       alpha=0.2, color='green', label='Fine-tuning Phase')
    
    # Add stabilization point if available
    stabilization_point = training_data['stabilization_point']
    if stabilization_point is not None and stabilization_point < len(warmup_losses):
        ax_main.axvline(x=stabilization_point, color='green', linestyle='--', 
                       label='Stabilization Point')
        # Add marker at the stabilization point
        stab_y = warmup_losses[stabilization_point] if stabilization_point < len(warmup_losses) else None
        if stab_y is not None:
            ax_main.plot(stabilization_point, stab_y, 'go', markersize=8)
    
    # Configure main plot
    ax_main.set_title(title, fontsize=16)
    ax_main.set_xlabel('Steps')
    ax_main.set_ylabel('Loss')
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc='upper right')
    
    # 2. Perplexity Plot
    ax_perplexity = fig.add_subplot(gs[1, :2])
    
    # Plot perplexity if available
    perplexity_scores = training_data['perplexity_scores']
    if perplexity_scores:
        perplexity_steps = list(range(len(perplexity_scores)))
        ax_perplexity.plot(perplexity_steps, perplexity_scores, 'purple', label='Perplexity')
        ax_perplexity.set_title('Model Perplexity')
        ax_perplexity.set_xlabel('Evaluation Steps')
        ax_perplexity.set_ylabel('Perplexity')
        ax_perplexity.grid(True, alpha=0.3)
    else:
        ax_perplexity.text(0.5, 0.5, 'No perplexity data available', 
                          ha='center', va='center')
    
    # 3. Pruning Statistics
    ax_sparsity = fig.add_subplot(gs[1, 2])
    
    # Plot sparsity/pruning stats if available
    sparsity_history = training_data['sparsity_history']
    pruned_head_counts = training_data['pruned_head_counts']
    
    if sparsity_history:
        ax_sparsity.plot(sparsity_history, 'r-', label='Sparsity')
        ax_sparsity.set_title('Model Sparsity')
        ax_sparsity.set_xlabel('Pruning Steps')
        ax_sparsity.set_ylabel('Sparsity (%)')
        ax_sparsity.grid(True, alpha=0.3)
    elif pruned_head_counts:
        ax_sparsity.plot(pruned_head_counts, 'r-', label='Pruned Heads')
        ax_sparsity.set_title('Pruned Attention Heads')
        ax_sparsity.set_xlabel('Pruning Steps')
        ax_sparsity.set_ylabel('Head Count')
        ax_sparsity.grid(True, alpha=0.3)
    else:
        ax_sparsity.text(0.5, 0.5, 'No pruning statistics available', 
                        ha='center', va='center')
    
    # 4. Warmup Detail Plot
    ax_warmup = fig.add_subplot(gs[2, 0])
    
    # Plot warmup phase in detail
    if warmup_losses:
        ax_warmup.plot(warmup_steps, warmup_losses, 'b-')
        ax_warmup.set_title('Warmup Phase Detail')
        ax_warmup.set_xlabel('Steps')
        ax_warmup.set_ylabel('Loss')
        ax_warmup.grid(True, alpha=0.3)
        
        # Highlight stabilization point
        if stabilization_point is not None and stabilization_point < len(warmup_losses):
            ax_warmup.axvline(x=stabilization_point, color='green', linestyle='--')
            ax_warmup.plot(stabilization_point, warmup_losses[stabilization_point], 'go', markersize=8)
    else:
        ax_warmup.text(0.5, 0.5, 'No warmup data available', 
                      ha='center', va='center')
    
    # 5. Pruning Detail Plot
    ax_pruning = fig.add_subplot(gs[2, 1])
    
    # Plot pruning phase in detail
    if pruning_losses:
        pruning_relative_steps = list(range(len(pruning_losses)))
        ax_pruning.plot(pruning_relative_steps, pruning_losses, 'r-')
        ax_pruning.set_title('Pruning Phase Detail')
        ax_pruning.set_xlabel('Steps')
        ax_pruning.set_ylabel('Loss')
        ax_pruning.grid(True, alpha=0.3)
    else:
        ax_pruning.text(0.5, 0.5, 'No pruning data available', 
                       ha='center', va='center')
    
    # 6. Fine-tuning Detail Plot
    ax_finetune = fig.add_subplot(gs[2, 2])
    
    # Plot fine-tuning phase in detail
    if fine_tuning_losses:
        finetune_relative_steps = list(range(len(fine_tuning_losses)))
        ax_finetune.plot(finetune_relative_steps, fine_tuning_losses, 'g-')
        ax_finetune.set_title('Fine-tuning Phase Detail')
        ax_finetune.set_xlabel('Steps')
        ax_finetune.set_ylabel('Loss')
        ax_finetune.grid(True, alpha=0.3)
    else:
        ax_finetune.text(0.5, 0.5, 'No fine-tuning data available', 
                        ha='center', va='center')
    
    # 7. Summary Box
    ax_summary = fig.add_subplot(gs[3, :])
    ax_summary.axis('off')
    
    # Create summary text
    summary_parts = []
    
    # Add total steps per phase
    summary_parts.append(f"Warmup: {len(warmup_losses)} steps")
    
    if pruning_losses:
        summary_parts.append(f"Pruning: {len(pruning_losses)} steps")
    
    if fine_tuning_losses:
        summary_parts.append(f"Fine-tuning: {len(fine_tuning_losses)} steps")
    
    # Add stabilization info
    if stabilization_point is not None:
        summary_parts.append(f"Stabilization at step {stabilization_point}")
    
    # Add pruning stats
    if sparsity_history and len(sparsity_history) > 0:
        summary_parts.append(f"Final sparsity: {sparsity_history[-1]:.1f}%")
    
    if pruned_head_counts and len(pruned_head_counts) > 0:
        summary_parts.append(f"Total pruned heads: {pruned_head_counts[-1]}")
    
    # Add perplexity improvement if available
    if perplexity_scores and len(perplexity_scores) > 1:
        initial_perplexity = perplexity_scores[0]
        final_perplexity = perplexity_scores[-1]
        improvement = (initial_perplexity - final_perplexity) / initial_perplexity * 100
        summary_parts.append(f"Perplexity improvement: {improvement:.1f}%")
    
    # Format the summary
    summary_text = " | ".join(summary_parts)
    
    # Add an inspirational quote if enabled
    if show_quote:
        quote = generate_inspirational_quote()
        formatted_quote = f'"{quote}"'
        ax_summary.text(0.5, 0.2, formatted_quote, ha='center', va='center', 
                       fontsize=11, fontstyle='italic', color='darkblue')
    
    # Add the summary stats
    ax_summary.text(0.5, 0.7, summary_text, ha='center', va='center', 
                   fontsize=10, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    # Add timestamp
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax_summary.text(0.99, 0.05, f"Generated: {current_time}", ha='right', va='bottom', 
                   fontsize=8, fontweight='light', transform=ax_summary.transAxes)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save visualization if requested
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default filename if not provided
        if filename is None:
            current_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"neural_plasticity_process_{current_timestamp}.png"
        
        # Ensure filename has .png extension
        if not filename.endswith(".png"):
            filename += ".png"
            
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"✅ Visualization saved to: {save_path}")
    
    # In Colab, showing is handled by the notebook
    # In standalone mode, we'll control display differently
    # We won't call plt.show() here to avoid unwanted popups
    # The calling code should decide how to display the figure
    
    return fig