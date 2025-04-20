"""
Neural Plasticity Visualization

This module provides visualization utilities for neural plasticity experiments.
It visualizes head entropy, gradients, pruning decisions, training metrics,
and attention patterns.

Version: v0.0.65 (2025-04-20 23:45:00)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import platform
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from utils.colab.helpers import safe_tensor_imshow, get_colab_type

# Initialize environment variables
IS_APPLE_SILICON = False
IS_COLAB = False
HAS_GPU = False

# Detect if we're running in Google Colab
try:
    import google.colab
    IS_COLAB = True
    print("🌐 Running in Google Colab environment")
    
    # Check for GPU in Colab
    colab_info = get_colab_type(verbose=False)
    if colab_info.get("hardware") == "GPU":
        HAS_GPU = True
        print(f"✅ CUDA GPU detected in Colab: {colab_info.get('gpu_name', 'Unknown GPU')}")
        print("🚀 Visualizations will be optimized for GPU acceleration")
        
        # Display GPU info for confirmation
        try:
            import subprocess
            subprocess.run(["nvidia-smi"], check=False)
        except Exception:
            pass
    else:
        print("⚠️ No GPU detected in Colab - using CPU for visualizations")
except (ImportError, ModuleNotFoundError):
    pass

# Detect Apple Silicon and apply optimizations if needed
try:
    if platform.system() == "Darwin" and platform.processor() == "arm":
        IS_APPLE_SILICON = True
        print("🍎 Apple Silicon detected - enabling visualization crash prevention")
        
        # Skip Apple Silicon optimizations if running in Colab (shouldn't happen, but just in case)
        if not IS_COLAB:
            # Force single-threaded image processing
            import os
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
            os.environ["NUMEXPR_NUM_THREADS"] = "1"
            
            # Switch matplotlib backend to Agg (non-interactive) on Apple Silicon
            # This helps prevent some rendering issues
            try:
                import matplotlib
                matplotlib.use('Agg')
                print("🎨 Switching to Agg matplotlib backend for improved stability")
            except (ImportError, RuntimeError):
                pass
                
            # Configure PyTorch for Apple Silicon if available
            try:
                import torch
                # Disable parallel CPU operations
                torch.set_num_threads(1)
            except (ImportError, AttributeError):
                pass
except (ImportError, AttributeError):
    pass


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
                        or dictionary of layers to entropy tensors
        title: Title for the plot
        min_value: Minimum value for the colormap scale
        annotate: Whether to add value annotations to the cells
        figsize: Figure size (width, height) in inches
        cmap: Colormap to use
        save_path: Optional path to save the visualization
        
    Returns:
        matplotlib Figure object
    """
    # Handle dictionary of layers to entropy values
    if isinstance(entropy_values, dict):
        # Convert dictionary to tensor [layers, heads]
        layers = sorted(list(entropy_values.keys()))
        if len(layers) > 0:
            # Check if the first value is a tensor with multiple dimensions
            first_val = entropy_values[layers[0]]
            if isinstance(first_val, torch.Tensor) and first_val.dim() > 1:
                # Take mean across additional dimensions
                entropy_stacked = []
                for layer in layers:
                    layer_entropy = entropy_values[layer]
                    if layer_entropy.dim() > 1:
                        # Reduce to 1D by taking mean across extra dimensions
                        layer_entropy = layer_entropy.mean(dim=tuple(range(1, layer_entropy.dim())))
                    entropy_stacked.append(layer_entropy)
                entropy_values = torch.stack(entropy_stacked)
            else:
                # Standard case, just stack the tensors
                entropy_values = torch.stack([entropy_values[layer] for layer in layers])
    
    # Handle tensors with extra dimensions
    if isinstance(entropy_values, torch.Tensor) and entropy_values.dim() > 2:
        # Reduce extra dimensions by taking the mean
        # Keep only the first two dimensions [layers, heads]
        entropy_values = entropy_values.mean(dim=tuple(range(2, entropy_values.dim())))
    
    # Convert to numpy if tensor
    if isinstance(entropy_values, torch.Tensor):
        entropy_data = entropy_values.detach().cpu().numpy()
    else:
        entropy_data = entropy_values
    
    # Create figure (only once)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap using imshow directly since entropy_data is already numpy
    im = ax.imshow(entropy_data, cmap=cmap)
    ax.set_title(title)
    
    # Set proper colormap limits with non-zero range
    im.set_clim(min_value, max(0.1, np.max(entropy_data)))
    
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
        
    # Create figure (only once)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap directly
    im = ax.imshow(grad_data, cmap=cmap)
    ax.set_title(title)
    
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
    
    # Create figure (only once)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Base plot with all gradient values
    im = ax.imshow(grad_data, cmap="YlOrRd")
    ax.set_title(title)
    
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


def visualize_warmup_dashboard(
    warmup_results: Dict[str, Any],
    title: str = "Neural Plasticity Warmup Dashboard",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive dashboard for visualizing warmup results including
    the stabilization detection process.
    
    Args:
        warmup_results: Dictionary with warmup metrics from run_warmup_phase
        title: Title for the dashboard
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the dashboard
        
    Returns:
        matplotlib Figure object
    """
    # Extract key data from warmup results
    warmup_losses = warmup_results.get("losses", [])
    smoothed_losses = warmup_results.get("smoothed_losses", [])
    initial_loss = warmup_results.get("initial_loss", 0)
    final_loss = warmup_results.get("final_loss", 0)
    is_stable = warmup_results.get("is_stable", False)
    steps_without_decrease = warmup_results.get("steps_without_decrease", 0)
    
    # Get stabilization point if available
    stabilization_point = None
    if is_stable:
        # Stabilization occurred at the step when we decided to stop
        stabilization_point = len(warmup_losses) - steps_without_decrease
    
    # Create figure with grid layout
    fig = plt.figure(figsize=figsize)
    
    # Define grid layout
    gs = plt.GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1])
    
    # 1. Main plot: Raw loss with stabilization indicator
    ax_raw = fig.add_subplot(gs[0, :])
    ax_raw.plot(warmup_losses, color='blue', label='Training Loss')
    ax_raw.set_title('Warmup Phase Loss with Stability Detection')
    ax_raw.set_xlabel('Step')
    ax_raw.set_ylabel('Loss')
    ax_raw.grid(True, alpha=0.3)
    
    # Add stabilization point markers if available
    if stabilization_point is not None and stabilization_point < len(warmup_losses):
        # Add vertical line at stabilization point
        ax_raw.axvline(x=stabilization_point, color='green', linestyle='--', alpha=0.7, 
                   label='Stabilization detected')
        
        # Mark the point
        ax_raw.plot(stabilization_point, warmup_losses[stabilization_point], 
                'go', markersize=8, label=f'Loss: {warmup_losses[stabilization_point]:.4f}')
        
        # Add note about stabilization
        stabilization_text = f"Stabilization at step {stabilization_point}"
        ax_raw.text(stabilization_point + len(warmup_losses)*0.01, 
                warmup_losses[stabilization_point]*0.95, 
                stabilization_text, fontsize=9, color='green')
    
    ax_raw.legend()
    
    # 2. Smoothed loss with trend analysis
    ax_smooth = fig.add_subplot(gs[1, 0])
    
    # Plot smoothed loss if available
    if len(smoothed_losses) > 1:
        x = range(0, len(smoothed_losses)*5, 5)
        ax_smooth.plot(x, smoothed_losses, color='purple', label='Smoothed Loss')
        ax_smooth.set_title('Smoothed Loss (5-step Rolling Average)')
        ax_smooth.set_xlabel('Step')
        ax_smooth.set_ylabel('Loss')
        ax_smooth.grid(True, alpha=0.3)
        
        # Add trend line if scipy is available
        try:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(x, smoothed_losses)
            ax_smooth.plot(x, [slope*xi + intercept for xi in x], 'r--', 
                       label=f'Trend: slope={slope:.6f}')
            ax_smooth.legend()
            
            # Mark stabilization point on smoothed curve if it falls within the data
            if stabilization_point is not None:
                # Calculate which index in smoothed_losses corresponds to stabilization_point
                smooth_idx = stabilization_point // 5
                if smooth_idx < len(smoothed_losses):
                    smooth_x = smooth_idx * 5  # Convert back to original x scale
                    ax_smooth.axvline(x=smooth_x, color='green', linestyle='--', alpha=0.7)
                    ax_smooth.plot(smooth_x, smoothed_losses[smooth_idx], 'go', markersize=8)
        except (ImportError, ValueError) as e:
            # Skip the trend line if an error occurs
            pass
    else:
        ax_smooth.text(0.5, 0.5, 'Not enough data for smoothed visualization',
                   ha='center', va='center')
    
    # 3. Polynomial fit/stability analysis
    ax_poly = fig.add_subplot(gs[1, 1])
    
    # Try to add polynomial fit for stability analysis
    if len(warmup_losses) > 10:
        try:
            from scipy.optimize import curve_fit
            
            # Define polynomial function (degree 2 = quadratic)
            def poly_func(x, a, b, c):
                return a * x**2 + b * x + c
            
            # Use numpy array for x values
            x_data = np.array(range(len(warmup_losses)))
            y_data = np.array(warmup_losses)
            
            # Fit polynomial to the loss values
            params, _ = curve_fit(poly_func, x_data, y_data)
            a, b, c = params
            
            # Plot polynomial fit
            x_dense = np.linspace(min(x_data), max(x_data), 100)
            y_fit = poly_func(x_dense, a, b, c)
            ax_poly.plot(x_data, y_data, 'b.', alpha=0.3, label='Raw data')
            ax_poly.plot(x_dense, y_fit, 'g-', 
                      label=f'Poly fit: {a:.5f}x² + {b:.5f}x + {c:.5f}')
            
            # Calculate derivatives at the end point
            x_end = max(x_data)
            deriv_1 = 2 * a * x_end + b  # First derivative
            deriv_2 = 2 * a              # Second derivative
            
            # Show stabilization info
            if is_stable:
                status = "Loss stabilized"
            else:
                if a > 0:
                    status = "Upward curve (stabilizing)"
                elif abs(deriv_1) < 0.01:
                    status = "Flat curve (stabilizing)"
                else:
                    status = "Still decreasing (not stable)"
            
            ax_poly.set_title(f"Polynomial Curve Fitting - {status}")
            
            # Add stability indicators
            poly_stable = (a > 0.0005) or (abs(deriv_1) < 0.01 and abs(deriv_2) < 0.005)
            if poly_stable:
                ax_poly.text(0.05, 0.05, "Polynomial analysis suggests stability", 
                          transform=ax_poly.transAxes, color='green', fontsize=9)
            
            # Add markers for stabilization point
            if stabilization_point is not None:
                ax_poly.axvline(x=stabilization_point, color='green', linestyle='--', alpha=0.7)
                ax_poly.plot(stabilization_point, poly_func(stabilization_point, a, b, c), 
                          'go', markersize=8)
                
            ax_poly.set_xlabel('Step')
            ax_poly.set_ylabel('Loss')
            ax_poly.grid(True, alpha=0.3)
            ax_poly.legend()
        except Exception as e:
            ax_poly.text(0.5, 0.5, f'Could not perform polynomial fit\n{str(e)}',
                     ha='center', va='center')
    else:
        ax_poly.text(0.5, 0.5, 'Not enough data for polynomial analysis',
                 ha='center', va='center')
    
    # 4. Segment analysis
    ax_segments = fig.add_subplot(gs[2, 0])
    
    # Get segment analysis data
    segment_analysis = warmup_results.get("segment_analysis", {})
    
    if segment_analysis and segment_analysis.get("segment_size", 0) > 0:
        segment_size = segment_analysis.get("segment_size", 0)
        first_avg = segment_analysis.get("first_segment_avg", 0)
        last_avg = segment_analysis.get("last_segment_avg", 0)
        improvement = segment_analysis.get("improvement", 0)
        still_improving = segment_analysis.get("still_improving", False)
        
        # Create bar chart for segments
        segments = ['First Segment', 'Last Segment']
        values = [first_avg, last_avg]
        bars = ax_segments.bar(segments, values, color=['blue', 'green' if not still_improving else 'orange'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax_segments.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        ax_segments.set_title(f'Segment Analysis (Improvement: {improvement:.1f}%)')
        ax_segments.set_ylabel('Average Loss')
        
        # Add stability indicator
        if not still_improving:
            ax_segments.text(0.5, 0.05, "Loss stabilized (not significantly improving)", 
                        transform=ax_segments.transAxes, color='green', fontsize=9, ha='center')
        else:
            ax_segments.text(0.5, 0.05, "Still improving significantly", 
                        transform=ax_segments.transAxes, color='orange', fontsize=9, ha='center')
    else:
        ax_segments.text(0.5, 0.5, 'Not enough data for segment analysis',
                    ha='center', va='center')
    
    # 5. Summary statistics
    ax_stats = fig.add_subplot(gs[2, 1])
    ax_stats.axis('off')  # Turn off axis
    
    # Create text summary
    summary_text = [
        f"Total Steps: {len(warmup_losses)}",
        f"Initial Loss: {initial_loss:.4f}",
        f"Final Loss: {final_loss:.4f}",
        f"Improvement: {(1 - final_loss/initial_loss)*100:.1f}% reduction",
        f"Stabilization Detected: {'Yes' if is_stable else 'No'}",
    ]
    
    if stabilization_point is not None:
        summary_text.append(f"Stabilization Point: Step {stabilization_point}")
        summary_text.append(f"Loss at Stabilization: {warmup_losses[stabilization_point]:.4f}")
    
    if is_stable:
        summary_text.append(f"Steps without significant decrease: {steps_without_decrease}")
    
    # Add text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax_stats.text(0.05, 0.95, '\n'.join(summary_text), transform=ax_stats.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)
    
    ax_stats.set_title('Warmup Summary Statistics')
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for suptitle
    
    # Save if path provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
    
    return fig


class VisualizationReporter:
    """
    Reporter for neural plasticity visualizations and metrics.
    
    This class provides methods to display, save, and summarize
    neural plasticity experiment results consistently across different
    environments and contexts.
    """
    
    def __init__(self, model=None, tokenizer=None, output_dir=None, save_visualizations=False, verbose=True):
        """
        Initialize the visualization reporter.
        
        Args:
            model: Optional model for evaluation and generation
            tokenizer: Optional tokenizer for text generation
            output_dir: Directory to save visualizations (created if it doesn't exist)
            save_visualizations: Whether to save visualizations to disk
            verbose: Whether to print detailed information
        """
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.save_visualizations = save_visualizations
        self.verbose = verbose
        self.viz_paths = {}
        
        # Create output directory if needed
        if self.save_visualizations and self.output_dir:
            import os
            os.makedirs(self.output_dir, exist_ok=True)
            
    def display_warmup_results(self, warmup_results):
        """
        Display warmup results including visualizations and metrics.
        
        Args:
            warmup_results: Dictionary with warmup metrics from run_warmup_phase
            
        Returns:
            Dictionary with additional display information
        """
        # Extract key metrics
        warmup_losses = warmup_results.get("losses", [])
        
        # Create enhanced warmup dashboard
        dashboard_fig = visualize_warmup_dashboard(
            warmup_results,
            title="Neural Plasticity Warmup Dashboard",
            figsize=(12, 10),
            save_path=os.path.join(self.output_dir, "warmup_dashboard.png") if self.save_visualizations else None
        )
        
        # Show the dashboard
        plt.figure(dashboard_fig.number)
        plt.show()
        
        # Show the original visualization if available (for compatibility)
        warmup_visualization = warmup_results.get("visualization")
        if warmup_visualization and warmup_visualization != dashboard_fig:
            plt.figure(warmup_visualization.number)
            plt.show()

        # Display segment analysis
        segment_analysis = warmup_results.get("segment_analysis", {})
        if segment_analysis and segment_analysis.get("segment_size", 0) > 0:
            print(f"\nWarm-up Segment Analysis:")
            print(f"First segment average loss: {segment_analysis['first_segment_avg']:.4f}")
            print(f"Last segment average loss: {segment_analysis['last_segment_avg']:.4f}")
            print(f"Improvement during warm-up: {segment_analysis['improvement']:.1f}%")
            print(f"Is model still significantly improving? {'Yes' if segment_analysis.get('still_improving', False) else 'No'}")

        # Print warm-up summary
        print(f"\nWarm-up completed with {len(warmup_losses)} steps across {len(warmup_results.get('epochs', []))}")
        print(f"Initial loss: {warmup_results.get('initial_loss', 0):.4f}")
        print(f"Final loss: {warmup_results.get('final_loss', 0):.4f}")
        
        # Print improvement percentage if available
        if 'improvement_percent' in warmup_results:
            print(f"Overall loss reduction: {warmup_results['improvement_percent']:.1f}%")

        # Display saved visualization paths if available
        if "visualization_paths" in warmup_results and warmup_results["visualization_paths"]:
            print("\nSaved visualizations:")
            for name, path in warmup_results["visualization_paths"].items():
                print(f"- {name}: {path}")
                
        # Store the dashboard in the warmup results for future reference
        warmup_results["dashboard"] = dashboard_fig
        if self.save_visualizations and self.output_dir:
            warmup_results["dashboard_path"] = os.path.join(self.output_dir, "warmup_dashboard.png")
                
        return warmup_results
    
    def display_pruning_results(self, pruning_results):
        """
        Display pruning results including visualizations and metrics.
        
        Args:
            pruning_results: Dictionary with pruning metrics from run_pruning_cycle
            
        Returns:
            Dictionary with additional display information
        """
        # Extract key metrics
        baseline_metrics = pruning_results.get("baseline_metrics", {})
        pruned_metrics = pruning_results.get("pruned_metrics", {})
        final_metrics = pruning_results.get("final_metrics", {})
        
        # Print baseline, pruned and final metrics
        print("\nMetrics Comparison:")
        print(f"  Baseline:  Loss = {baseline_metrics.get('loss', 0):.4f}, Perplexity = {baseline_metrics.get('perplexity', 0):.2f}")
        print(f"  After Pruning: Loss = {pruned_metrics.get('loss', 0):.4f}, Perplexity = {pruned_metrics.get('perplexity', 0):.2f}")
        print(f"  After Training: Loss = {final_metrics.get('loss', 0):.4f}, Perplexity = {final_metrics.get('perplexity', 0):.2f}")
        
        # Print improvement metrics
        if 'perplexity_improvement' in pruning_results:
            print(f"\nPerplexity improvement: {pruning_results['perplexity_improvement']*100:.2f}%")
            
        if 'recovery_rate' in pruning_results:
            print(f"Recovery rate: {pruning_results['recovery_rate']*100:.2f}%")
        
        # Print pruning summary
        pruned_heads = pruning_results.get("pruned_heads", [])
        print(f"\nPruning Summary:")
        print(f"  Pruned {len(pruned_heads)} out of {pruning_results.get('total_heads', 0)} heads")
        print(f"  Pruning strategy: {pruning_results.get('strategy', 'unknown')}")
        print(f"  Pruning level: {pruning_results.get('pruning_level', 0)*100:.1f}%")
        
        # Display visualizations
        training_metrics = pruning_results.get("training_metrics", {})
        if training_metrics:
            try:
                metrics_fig = visualize_training_metrics(
                    metrics_history=training_metrics,
                    title="Training After Pruning"
                )
                plt.figure(metrics_fig.number)
                plt.show()
            except Exception as e:
                if self.verbose:
                    print(f"Could not display training metrics: {e}")
        
        # Display visualization directory if available
        if "visualization_dir" in pruning_results:
            print(f"\nVisualization directory: {pruning_results['visualization_dir']}")
            
        return pruning_results
    
    def report_model_stats(self, model, sparsity=None, pruned_heads=None):
        """
        Display summary statistics for a model.
        
        Args:
            model: The transformer model
            sparsity: Optional sparsity value to display
            pruned_heads: Optional list of pruned heads to display
            
        Returns:
            Dictionary with model statistics
        """
        # Estimate model size
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        model_size_mb = (param_size + buffer_size) / 1024**2
        
        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Print model statistics
        print(f"\nModel Statistics:")
        print(f"  Size: {model_size_mb:.2f} MB")
        print(f"  Parameters: {total_params:,}")
        
        if sparsity is not None:
            print(f"  Sparsity: {sparsity:.2%}")
            
        if pruned_heads is not None:
            print(f"  Pruned heads: {len(pruned_heads)}")
            
        return {
            "model_size_mb": model_size_mb,
            "total_params": total_params,
            "sparsity": sparsity,
            "pruned_heads_count": len(pruned_heads) if pruned_heads else 0
        }
    
    def save_figure(self, fig, name, subfolder=None):
        """
        Save a figure to disk.
        
        Args:
            fig: Matplotlib figure to save
            name: Name for the saved figure file
            subfolder: Optional subfolder within output_dir
            
        Returns:
            Path to saved figure or None if not saved
        """
        if not self.save_visualizations or not self.output_dir:
            return None
            
        try:
            import os
            
            # Create subfolder if needed
            if subfolder:
                save_dir = os.path.join(self.output_dir, subfolder)
                os.makedirs(save_dir, exist_ok=True)
            else:
                save_dir = self.output_dir
                
            # Ensure filename has .png extension
            if not name.endswith(".png"):
                name = f"{name}.png"
                
            # Save figure
            save_path = os.path.join(save_dir, name)
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            
            # Store path in visualization paths
            self.viz_paths[name] = save_path
            
            if self.verbose:
                print(f"✅ Saved visualization to {save_path}")
                
            return save_path
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error saving visualization: {e}")
            return None
            
    def get_visualization_paths(self):
        """
        Get dictionary of all saved visualization paths.
        
        Returns:
            Dictionary mapping visualization names to file paths
        """
        return self.viz_paths
        
    def evaluate_model(self, model=None, dataloader=None, device=None):
        """
        Evaluate model on the given dataloader.
        
        Args:
            model: Model to evaluate (defaults to self.model)
            dataloader: DataLoader for evaluation
            device: Device to run evaluation on (defaults to model's device)
            
        Returns:
            Tuple of (loss, perplexity)
        """
        # Use model from instance if not provided
        if model is None:
            if self.model is None:
                raise ValueError("No model provided for evaluation")
            model = self.model
            
        # Import directly from core to avoid circular imports
        from .core import evaluate_model
        
        # Determine device if not provided
        if device is None:
            device = next(model.parameters()).device
            
        # Evaluate the model directly
        eval_results = evaluate_model(
            model=model,
            dataloader=dataloader,
            device=device
        )
        
        # Print results if verbose
        if self.verbose:
            print(f"Evaluation: Loss = {eval_results['loss']:.4f}, Perplexity = {eval_results['perplexity']:.2f}")
            
        return eval_results["loss"], eval_results["perplexity"]
    
    def display_complete_training_process(self, experiment=None):
        """
        Display comprehensive visualization of the complete neural plasticity process.
        
        Args:
            experiment: NeuralPlasticityExperiment object or dictionary with experiment results
                       (defaults to current experiment if None)
            
        Returns:
            Dictionary with visualization information
        """
        import os
        
        # Use the visualize_complete_training_process function from colab helper module
        from utils.colab.visualizations import visualize_complete_training_process
        
        # Determine the experiment object
        if experiment is None:
            # Try to find experiment data from instance variables
            experiment_data = {}
            if hasattr(self, 'experiment'):
                experiment = self.experiment
            else:
                # Collect results from individual components if available
                for attr_name in ['warmup_results', 'pruning_results', 'fine_tuning_results']:
                    if hasattr(self, attr_name):
                        experiment_data[attr_name.split('_')[0]] = getattr(self, attr_name)
                
                if experiment_data:
                    experiment = experiment_data
                else:
                    raise ValueError("No experiment data available. Please provide an experiment object.")
        
        # Determine output directory if saving
        output_dir = None
        if self.save_visualizations and self.output_dir:
            output_dir = os.path.join(self.output_dir, "complete_process")
            os.makedirs(output_dir, exist_ok=True)
        
        # Generate the visualization
        complete_fig = visualize_complete_training_process(
            experiment=experiment,
            output_dir=output_dir,
            title="Complete Neural Plasticity Training Process",
            show_plot=False,  # Don't auto-show, we'll handle display
            show_quote=True
        )
        
        # Display the figure - detect if we're in a notebook
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                # In a notebook, display the figure with IPython
                from IPython.display import display
                display(complete_fig)
            else:
                # In a regular Python environment, just set a reference
                plt.figure(complete_fig.number)
        except (ImportError, AttributeError):
            # Not in IPython, set active figure
            plt.figure(complete_fig.number)
        
        # Store visualization path if saved
        if output_dir:
            viz_path = next((f for f in os.listdir(output_dir) if f.startswith("neural_plasticity_process")), None)
            if viz_path:
                full_path = os.path.join(output_dir, viz_path)
                self.viz_paths["complete_training_process"] = full_path
                
                if self.verbose:
                    print(f"\nComplete training process visualization saved to: {full_path}")
        
        # Return the figure and info
        return {
            "visualization": complete_fig,
            "saved_path": self.viz_paths.get("complete_training_process")
        }
        
    def generate_comprehensive_dashboard(self, experiment=None, output_dir=None, include_all=True):
        """
        Generate a comprehensive set of dashboards for the complete neural plasticity process.
        
        This method creates multiple visualizations showing different aspects of the
        neural plasticity process, including the complete overview, detailed warmup
        analysis, pruning analysis, and fine-tuning results.
        
        Args:
            experiment: NeuralPlasticityExperiment object or dictionary with experiment results
                       (defaults to current experiment if None)
            output_dir: Directory to save visualizations (defaults to self.output_dir/dashboards)
            include_all: Whether to generate all dashboards or just the comprehensive one
            
        Returns:
            Dictionary mapping dashboard names to figure objects
        """
        # Determine the experiment object
        if experiment is None:
            # Try to find experiment data from instance variables
            experiment_data = {}
            if hasattr(self, 'experiment'):
                experiment = self.experiment
            else:
                # Collect results from individual components if available
                for attr_name in ['warmup_results', 'pruning_results', 'fine_tuning_results']:
                    if hasattr(self, attr_name):
                        experiment_data[attr_name.split('_')[0]] = getattr(self, attr_name)
                
                if experiment_data:
                    experiment = experiment_data
                else:
                    raise ValueError("No experiment data available. Please provide an experiment object.")
        
        # Import the dashboard generator
        from scripts.neural_plasticity_dashboard import generate_dashboards
        
        # Determine output directory
        if output_dir is None and self.save_visualizations and self.output_dir:
            output_dir = os.path.join(self.output_dir, "dashboards")
            
        # Generate the dashboards
        figures = generate_dashboards(
            experiment=experiment,
            output_dir=output_dir,
            show=False  # Don't auto-show, we'll handle display
        )
        
        # Display each figure
        for name, fig in figures.items():
            try:
                plt.figure(fig.number)
                
                # Try to use IPython display if available
                try:
                    from IPython import get_ipython
                    if get_ipython() is not None:
                        from IPython.display import display
                        display(fig)
                        # Clear the figure to avoid duplicate display
                        plt.clf()
                except (ImportError, AttributeError):
                    pass
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Could not display {name} dashboard: {e}")
        
        # Update visualization paths
        if output_dir:
            for name in figures:
                path = os.path.join(output_dir, f"{name}_dashboard.png")
                if os.path.exists(path):
                    self.viz_paths[f"{name}_dashboard"] = path
            
            if self.verbose:
                print(f"\nGenerated dashboards saved to: {output_dir}")
        
        return figures
    
    def generate_text(self, prompt, model=None, tokenizer=None, max_length=100, 
                     temperature=0.7, top_k=50, top_p=0.95, device=None, 
                     save_to_file=None):
        """
        Generate text from the model.
        
        Args:
            prompt: Text prompt to generate from
            model: Model to use (defaults to self.model)
            tokenizer: Tokenizer to use (defaults to self.tokenizer)
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            device: Device to run generation on
            save_to_file: Optional file path to save generated text
            
        Returns:
            Generated text string
        """
        # Use model and tokenizer from instance if not provided
        if model is None:
            if self.model is None:
                raise ValueError("No model provided for text generation")
            model = self.model
            
        if tokenizer is None:
            if self.tokenizer is None:
                raise ValueError("No tokenizer provided for text generation")
            tokenizer = self.tokenizer
            
        # Determine device if not provided
        if device is None:
            device = next(model.parameters()).device
            
        # Set model to evaluation mode
        model.eval()
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generate text
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and return text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Save to file if requested
        if save_to_file:
            try:
                import os
                
                # Create parent directory if needed
                os.makedirs(os.path.dirname(save_to_file), exist_ok=True)
                
                with open(save_to_file, 'w') as f:
                    f.write(f"Prompt: {prompt}\n\n")
                    f.write(f"Generated text:\n{generated_text}")
                    
                if self.verbose:
                    print(f"✅ Generated text saved to {save_to_file}")
            except Exception as e:
                if self.verbose:
                    print(f"❌ Error saving generated text: {e}")
        
        return generated_text
        
    def run_text_generation_suite(self, prompts, model=None, tokenizer=None, 
                                 max_length=100, save_to_file=True, subfolder="generation"):
        """
        Run a suite of text generation examples and optionally save them.
        
        Args:
            prompts: List of prompts or dictionary of {name: prompt}
            model: Model to use (defaults to self.model)
            tokenizer: Tokenizer to use (defaults to self.tokenizer)
            max_length: Maximum length of generated text
            save_to_file: Whether to save generations to files
            subfolder: Subfolder within output_dir to save generations
            
        Returns:
            Dictionary of prompt to generated text
        """
        # Convert list of prompts to dictionary if needed
        if isinstance(prompts, list):
            prompts_dict = {f"prompt_{i}": prompt for i, prompt in enumerate(prompts)}
        else:
            prompts_dict = prompts
            
        results = {}
        
        # Print header
        if self.verbose:
            print("\n=== Running Text Generation Suite ===\n")
            
        # Run generation for each prompt
        for name, prompt in prompts_dict.items():
            if self.verbose:
                print(f"Prompt: {prompt}")
                
            # Determine save path
            save_path = None
            if save_to_file and self.save_visualizations and self.output_dir:
                import os
                save_dir = os.path.join(self.output_dir, subfolder)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{name}.txt")
                
            # Generate text
            generated = self.generate_text(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                max_length=max_length,
                save_to_file=save_path
            )
            
            # Store result
            results[prompt] = generated
            
            # Print result
            if self.verbose:
                print(f"Generated: {generated[:200]}...")
                if len(generated) > 200:
                    print("...")
                print("-" * 40)
                
        return results
        
    def save_metrics_to_csv(self, metrics, filename="metrics.csv", subfolder=None):
        """
        Save metrics dictionary to CSV file.
        
        Args:
            metrics: Dictionary of metrics to save
            filename: Name for the CSV file
            subfolder: Optional subfolder within output_dir
            
        Returns:
            Path to saved CSV file or None if not saved
        """
        if not self.save_visualizations or not self.output_dir:
            return None
            
        try:
            import os
            import csv
            
            # Create subfolder if needed
            if subfolder:
                save_dir = os.path.join(self.output_dir, subfolder)
                os.makedirs(save_dir, exist_ok=True)
            else:
                save_dir = self.output_dir
                
            # Ensure filename has .csv extension
            if not filename.endswith(".csv"):
                filename = f"{filename}.csv"
                
            # Save CSV
            save_path = os.path.join(save_dir, filename)
            
            # Prepare data for CSV
            # Assume each value in metrics is a list of the same length
            if not metrics:
                return None
                
            # Get list of metric names
            metric_names = list(metrics.keys())
            
            # Get length of first metric array
            first_metric = next(iter(metrics.values()))
            rows = len(first_metric) if isinstance(first_metric, list) else 1
            
            with open(save_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(metric_names)
                
                # Write each row of data
                if rows > 1:
                    for i in range(rows):
                        writer.writerow([metrics[name][i] if i < len(metrics[name]) else "" for name in metric_names])
                else:
                    # Just one row of data
                    writer.writerow([metrics[name] for name in metric_names])
            
            # Store path
            self.viz_paths[filename] = save_path
            
            if self.verbose:
                print(f"✅ Saved metrics to {save_path}")
                
            return save_path
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error saving metrics: {e}")
            return None


def create_pruning_state_heatmap(
    model: torch.nn.Module,
    cumulative_pruned: List[Tuple[int, int]],
    newly_pruned: List[Tuple[int, int]] = None,
    title: str = "Cumulative Pruning State",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a heatmap showing the cumulative state of pruned heads.
    
    Args:
        model: The transformer model
        cumulative_pruned: List of (layer, head) tuples of all pruned heads
        newly_pruned: List of (layer, head) tuples of newly pruned heads
        title: Title for the plot
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the visualization
        
    Returns:
        matplotlib Figure object
    """
    # Determine model structure
    # Attempt to detect model structure directly
    num_layers = 0
    num_heads = 0
    
    # Check for HF transformer models
    if hasattr(model, 'config'):
        if hasattr(model.config, 'num_hidden_layers'):
            num_layers = model.config.num_hidden_layers
        elif hasattr(model.config, 'n_layer'):
            num_layers = model.config.n_layer
            
        if hasattr(model.config, 'num_attention_heads'):
            num_heads = model.config.num_attention_heads
    
    # If we couldn't detect from config, try counting transformer blocks
    if num_layers == 0 or num_heads == 0:
        # Try to detect from model architecture
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT-2 style
            num_layers = len(model.transformer.h)
            if num_layers > 0 and hasattr(model.transformer.h[0].attn, 'num_heads'):
                num_heads = model.transformer.h[0].attn.num_heads
    
    # Fallback values if detection fails
    if num_layers == 0:
        num_layers = len(cumulative_pruned) // 2 if cumulative_pruned else 6
    if num_heads == 0:
        num_heads = 12  # Common default
    
    # Create empty matrix for pruning state
    pruning_state = np.zeros((num_layers, num_heads))
    
    # Fill in cumulative pruned heads with 1
    for layer, head in cumulative_pruned:
        if 0 <= layer < num_layers and 0 <= head < num_heads:
            pruning_state[layer, head] = 1
    
    # Fill in newly pruned heads with 2 (for different color)
    if newly_pruned:
        for layer, head in newly_pruned:
            if 0 <= layer < num_layers and 0 <= head < num_heads:
                pruning_state[layer, head] = 2
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create custom colormap to distinguish newly pruned and previously pruned
    import matplotlib.colors as mcolors
    colors = [(1,1,1), (0.8,0.2,0.2), (0.2,0.8,0.2)]  # white, red, green
    cmap_custom = mcolors.ListedColormap(colors)
    bounds = [0, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap_custom.N)
    
    # Plot heatmap
    im = ax.imshow(pruning_state, cmap=cmap_custom, norm=norm)
    
    # Add colorbar with custom labels
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.set_ticklabels(['Active', 'Previously Pruned', 'Newly Pruned'])
    
    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Layer Index')
    
    # Add text annotations showing pruned head counts
    previously_pruned_count = len(cumulative_pruned) - len(newly_pruned) if newly_pruned else len(cumulative_pruned)
    newly_pruned_count = len(newly_pruned) if newly_pruned else 0
    total_heads = num_layers * num_heads
    
    pruned_percent = (len(cumulative_pruned) / total_heads) * 100
    ax.text(0.05, -0.15, f"Total pruned: {len(cumulative_pruned)}/{total_heads} heads ({pruned_percent:.1f}%)",
            transform=ax.transAxes, fontsize=10)
            
    if newly_pruned:
        ax.text(0.05, -0.20, f"Newly pruned this cycle: {newly_pruned_count} heads",
                transform=ax.transAxes, fontsize=10)
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


def visualize_attention_patterns(
    attention_maps: Union[torch.Tensor, List[torch.Tensor]],
    layer_idx: int = 0,
    head_idx: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    num_heads: int = 4,
    save_path: Optional[str] = None,
    use_safe_tensor: bool = True
) -> plt.Figure:
    """
    Visualize attention patterns for one or more heads with cross-platform compatibility.
    
    Args:
        attention_maps: Attention tensor with shape [batch, heads, seq_len, seq_len]
                       or list of attention tensors per layer
        layer_idx: Index of layer to visualize
        head_idx: Index of attention head to visualize (if None, shows multiple heads)
        title: Title for the plot
        figsize: Figure size (width, height) in inches
        num_heads: Number of heads to visualize if head_idx is None
        save_path: Optional path to save the visualization
        use_safe_tensor: Whether to use the safe_tensor_imshow function (recommended for
                        cross-platform compatibility)
        
    Returns:
        matplotlib Figure object
    """
    from utils.colab.helpers import safe_tensor_imshow
    
    # Handle tensor if it's a list of layers
    if isinstance(attention_maps, list):
        if layer_idx >= len(attention_maps):
            print(f"Warning: Layer {layer_idx} out of bounds (max: {len(attention_maps)-1})")
            layer_idx = min(layer_idx, len(attention_maps)-1)
        attention_maps = attention_maps[layer_idx]
    
    # Check tensor dimensions
    if isinstance(attention_maps, torch.Tensor):
        if attention_maps.dim() not in [3, 4]:
            raise ValueError(f"Expected 3D or 4D attention tensor, got shape {attention_maps.shape}")
        
        # Handle 3D tensor (missing batch dimension)
        if attention_maps.dim() == 3:
            attention_maps = attention_maps.unsqueeze(0)
    
    # If using safe_tensor_imshow (recommended for cross-platform)
    if use_safe_tensor:
        if head_idx is not None:
            # Single head visualization
            # Extract the head attention tensor
            if isinstance(attention_maps, torch.Tensor):
                head_attention = attention_maps[0, head_idx]
            else:
                head_attention = attention_maps[0][head_idx]
                
            # Use safe_tensor_imshow for cross-platform visualization
            head_title = title or f'Attention Pattern (Layer {layer_idx}, Head {head_idx})'
            fig = safe_tensor_imshow(
                tensor=head_attention,
                title=head_title,
                cmap='viridis',
                figsize=figsize,
                save_path=save_path,
                vmin=0.0,
                vmax=1.0
            )
            
            # Access the created axis for additional customization
            ax = fig.axes[0]
            ax.set_xlabel('Sequence Position (To)')
            ax.set_ylabel('Sequence Position (From)')
            
            return fig
        else:
            # Multiple heads visualization
            heads_to_show = min(num_heads, attention_maps.shape[1])
            fig, axs = plt.subplots(1, heads_to_show, figsize=figsize)
            
            # Adjust title
            main_title = title or f'Attention Patterns (Layer {layer_idx})'
            fig.suptitle(main_title, fontsize=14)
            
            # Ensure axs is always an array even with single subplot
            if heads_to_show == 1:
                axs = [axs]
            
            # Plot each head
            for i in range(heads_to_show):
                # Extract head attention tensor
                head_attention = attention_maps[0, i]
                
                # Use numpy-safe processing that works across platforms
                if isinstance(head_attention, torch.Tensor):
                    # Process safely
                    try:
                        if head_attention.requires_grad:
                            head_attention = head_attention.detach()
                        if head_attention.is_cuda:
                            head_attention = head_attention.cpu()
                        head_data = head_attention.numpy()
                    except Exception:
                        # Fallback for any conversion issues
                        head_data = np.zeros((10, 10))  # Empty placeholder
                else:
                    head_data = head_attention
                
                # Clean up NaN/Inf values
                head_data = np.nan_to_num(head_data, nan=0.0, posinf=1.0, neginf=0.0)
                
                # Plot attention pattern
                im = axs[i].imshow(head_data, cmap='viridis', vmin=0.0, vmax=1.0)
                
                # Add labels only to the first subplot for cleaner display
                if i == 0:
                    axs[i].set_ylabel('From Position')
                
                axs[i].set_xlabel('To Position')
                axs[i].set_title(f'Head {i}')
            
            # Add colorbar to the rightmost subplot
            plt.colorbar(im, ax=axs[-1], label='Attention Probability')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if path provided
            if save_path:
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
            
            return fig
    else:
        # Legacy implementation without safe_tensor_imshow (less reliable across platforms)
        # Ensure tensor is properly prepared for visualization
        if isinstance(attention_maps, torch.Tensor):
            # Handle environment-specific processing
            if IS_APPLE_SILICON:
                # Force to CPU on Apple Silicon to avoid BLAS crashes
                if attention_maps.is_cuda:
                    attention_maps = attention_maps.detach().cpu()
                # Detach for safety
                if attention_maps.requires_grad:
                    attention_maps = attention_maps.detach()
            elif IS_COLAB and HAS_GPU:
                # In Colab with GPU, move to CPU for visualization
                if attention_maps.is_cuda:
                    # Only detach if needed
                    if attention_maps.requires_grad:
                        attention_maps = attention_maps.detach().cpu()
                    else:
                        attention_maps = attention_maps.cpu()
            else:
                # Standard environment handling
                if attention_maps.requires_grad:
                    attention_maps = attention_maps.detach()
                if attention_maps.is_cuda:
                    attention_maps = attention_maps.cpu()
            
            # Always ensure contiguous memory layout for efficient conversion
            if not attention_maps.is_contiguous():
                attention_maps = attention_maps.contiguous()
                
            attn = attention_maps
        else:
            attn = attention_maps
        
        # Create figure
        if head_idx is not None:
            # Single head visualization
            fig, ax = plt.subplots(figsize=figsize)
            
            # Extract the specific head attention and safely convert to numpy
            if isinstance(attn, torch.Tensor):
                # Handle potential NaN/Inf values during conversion
                try:
                    head_attention = attn[0, head_idx].cpu().numpy()
                    if np.isnan(head_attention).any() or np.isinf(head_attention).any():
                        head_attention = np.nan_to_num(head_attention, nan=0.0, posinf=1.0, neginf=0.0)
                except Exception as e:
                    print(f"⚠️ Warning: Error converting attention tensor: {e}")
                    head_attention = np.zeros((10, 10))  # Empty placeholder
            else:
                head_attention = attn[0, head_idx]
            
            # Plot attention pattern
            im = ax.imshow(head_attention, cmap='viridis')
            ax.set_title(title or f'Attention pattern (layer {layer_idx}, head {head_idx})')
            
            # Set proper limits for attention values (0 to 1)
            im.set_clim(0, 1.0)
            
            # Add colorbar
            plt.colorbar(im, label='Attention probability')
            
            # Add labels
            plt.xlabel('Sequence position (to)')
            plt.ylabel('Sequence position (from)')
            
        else:
            # Multiple heads visualization
            heads_to_show = min(num_heads, attn.shape[1])
            fig, axs = plt.subplots(1, heads_to_show, figsize=figsize)
            
            # Ensure axs is always an array even with single subplot
            if heads_to_show == 1:
                axs = [axs]
                
            # Adjust title
            if title:
                fig.suptitle(title, fontsize=14)
            else:
                fig.suptitle(f'Attention patterns (layer {layer_idx})', fontsize=14)
            
            # Plot each head
            for i in range(heads_to_show):
                # Extract head attention and safely convert to numpy
                if isinstance(attn, torch.Tensor):
                    try:
                        head_attention = attn[0, i].cpu().numpy()
                        # Check for NaN/Inf values
                        if np.isnan(head_attention).any() or np.isinf(head_attention).any():
                            head_attention = np.nan_to_num(head_attention, nan=0.0, posinf=1.0, neginf=0.0)
                    except Exception as e:
                        print(f"⚠️ Warning: Error converting attention tensor for head {i}: {e}")
                        head_attention = np.zeros((10, 10))  # Empty placeholder
                else:
                    head_attention = attn[0, i]
                    
                # Plot attention pattern
                im = axs[i].imshow(head_attention, cmap='viridis')
                
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