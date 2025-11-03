#!/usr/bin/env python
"""
Neural Plasticity Process Visualization Generator

This module provides real-time visualization of neural plasticity experiments,
generating images during experiment execution rather than HTML dashboards.

Version: v0.0.34 (2025-04-20 13:45:00)
"""

import os
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PlasticityVisualizer:
    """
    Real-time visualizer for neural plasticity experiments.
    
    This class generates visualizations during experiment execution and saves
    them to the output directory. It does not generate HTML dashboards.
    """
    
    def __init__(self, output_dir: str = './output', show_plots: bool = False):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            show_plots: Whether to display plots in addition to saving them
        """
        # Ensure we're using the correct output directory
        if output_dir.startswith('./'):
            # Convert relative path to absolute
            self.output_dir = Path(os.path.abspath(output_dir))
        elif not os.path.isabs(output_dir):
            # Ensure it's a full path within project root
            self.output_dir = Path(os.path.join(os.getcwd(), output_dir))
        else:
            self.output_dir = Path(output_dir)
            
        self.show_plots = show_plots
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configure matplotlib for reproducible outputs and stability
        plt.switch_backend('agg')  # Use non-interactive backend
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Log the output directory being used
        logger.info(f"PlasticityVisualizer initialized - saving images to: {self.output_dir}")
        
    def visualize_training_progress(self, 
                                   metrics: Dict[str, List[float]],
                                   experiment_id: str = None,
                                   title: str = "Training Progress",
                                   figsize: Tuple[int, int] = (10, 6),
                                   save: bool = True) -> plt.Figure:
        """
        Visualize training metrics over time.
        
        Args:
            metrics: Dictionary of metrics history
            experiment_id: Optional experiment identifier for file naming
            title: Title for the visualization
            figsize: Figure size (width, height) in inches
            save: Whether to save the visualization
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each metric
        for metric_name, values in metrics.items():
            if metric_name == 'step':
                continue  # Skip step metric as it's used for x-axis
                
            ax.plot(values, label=metric_name)
            
        ax.set_title(title)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Value')
        ax.legend()
        
        # Save figure if requested
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_str = f"{experiment_id}_" if experiment_id else ""
            filename = f"{experiment_str}training_progress_{timestamp}.png"
            
            # Ensure we're saving directly to the output directory
            save_path = os.path.join(self.output_dir, filename)
            
            # Create any needed parent directories
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save the figure with high quality
            fig.savefig(save_path, bbox_inches='tight', dpi=120)
            logger.info(f"Saved training progress visualization to {save_path}")
            
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
            
        return fig
        
    def visualize_entropy_heatmap(self,
                                 entropy_data: Union[Dict[str, List[float]], torch.Tensor, np.ndarray],
                                 experiment_id: str = None,
                                 title: str = "Attention Entropy Heatmap",
                                 figsize: Tuple[int, int] = (10, 6),
                                 save: bool = True) -> plt.Figure:
        """
        Create a heatmap visualization of attention entropy values.
        
        Args:
            entropy_data: Dictionary mapping layer indices to entropy tensors,
                         or direct tensor/array of shape [layers, heads]
            experiment_id: Optional experiment identifier for file naming
            title: Title for the visualization
            figsize: Figure size (width, height) in inches
            save: Whether to save the visualization
            
        Returns:
            matplotlib Figure object
        """
        # Convert dictionary format to numpy array if needed
        if isinstance(entropy_data, dict):
            # Convert string keys to integers and sort
            layers = sorted([int(k) if isinstance(k, str) else k for k in entropy_data.keys()])
            
            # Extract the entropy tensors for each layer
            entropy_arrays = []
            for layer in layers:
                layer_key = str(layer) if str(layer) in entropy_data else layer
                layer_data = entropy_data[layer_key]
                
                # Convert list to numpy array if needed
                if isinstance(layer_data, list):
                    layer_array = np.array(layer_data)
                elif isinstance(layer_data, torch.Tensor):
                    layer_array = layer_data.detach().cpu().numpy()
                else:
                    layer_array = layer_data
                    
                entropy_arrays.append(layer_array)
                
            # Stack layers to create 2D array [layers, heads]
            entropy_array = np.stack(entropy_arrays)
        elif isinstance(entropy_data, torch.Tensor):
            entropy_array = entropy_data.detach().cpu().numpy()
        else:
            entropy_array = entropy_data
            
        # Create figure and plot heatmap
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(entropy_array, cmap='viridis')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Entropy')
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel('Head Index')
        ax.set_ylabel('Layer Index')
        
        # Add text annotations for clarity
        for i in range(entropy_array.shape[0]):
            for j in range(entropy_array.shape[1]):
                ax.text(j, i, f'{entropy_array[i, j]:.2f}',
                       ha="center", va="center", color="w", fontsize=8)
        
        # Save figure if requested
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_str = f"{experiment_id}_" if experiment_id else ""
            filename = f"{experiment_str}entropy_heatmap_{timestamp}.png"
            
            # Ensure we're saving directly to the output directory
            save_path = os.path.join(self.output_dir, filename)
            
            # Create any needed parent directories
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save the figure with high quality
            fig.savefig(save_path, bbox_inches='tight', dpi=120)
            logger.info(f"Saved entropy heatmap to {save_path}")
            
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
            
        return fig
        
    def visualize_pruned_heads(self,
                              pruned_heads: List[Tuple[int, int, float]],
                              num_layers: int,
                              num_heads: int,
                              experiment_id: str = None,
                              title: str = "Pruned Attention Heads",
                              figsize: Tuple[int, int] = (10, 6),
                              save: bool = True) -> plt.Figure:
        """
        Visualize which heads have been pruned.
        
        Args:
            pruned_heads: List of (layer_idx, head_idx, score) tuples
            num_layers: Number of layers in the model
            num_heads: Number of heads per layer
            experiment_id: Optional experiment identifier for file naming
            title: Title for the visualization
            figsize: Figure size (width, height) in inches
            save: Whether to save the visualization
            
        Returns:
            matplotlib Figure object
        """
        # Create a mask of pruned heads (1 for pruned, 0 for active)
        pruned_mask = np.zeros((num_layers, num_heads))
        
        # Fill in pruned heads
        for layer_idx, head_idx, _ in pruned_heads:
            if 0 <= layer_idx < num_layers and 0 <= head_idx < num_heads:
                pruned_mask[layer_idx, head_idx] = 1
                
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap with custom colormap (white for active, red for pruned)
        import matplotlib.colors as mcolors
        cmap = mcolors.ListedColormap(['white', 'red'])
        im = ax.imshow(pruned_mask, cmap=cmap)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
        cbar.set_ticklabels(['Active', 'Pruned'])
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel('Head Index')
        ax.set_ylabel('Layer Index')
        
        # Add text showing pruned head count
        pruned_count = len(pruned_heads)
        total_count = num_layers * num_heads
        pruned_percentage = (pruned_count / total_count) * 100
        
        ax.text(0.02, -0.1, 
                f"Pruned {pruned_count}/{total_count} heads ({pruned_percentage:.1f}%)",
                transform=ax.transAxes, fontsize=10)
        
        # Save figure if requested
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_str = f"{experiment_id}_" if experiment_id else ""
            filename = f"{experiment_str}pruned_heads_{timestamp}.png"
            
            # Ensure we're saving directly to the output directory
            save_path = os.path.join(self.output_dir, filename)
            
            # Create any needed parent directories
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save the figure with high quality
            fig.savefig(save_path, bbox_inches='tight', dpi=120)
            logger.info(f"Saved pruned heads visualization to {save_path}")
            
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
            
        return fig
        
    def visualize_recovery_analysis(self,
                                   baseline_metrics: Dict[str, float],
                                   pruned_metrics: Dict[str, float],
                                   final_metrics: Dict[str, float],
                                   experiment_id: str = None,
                                   title: str = "Model Recovery Analysis",
                                   figsize: Tuple[int, int] = (10, 6),
                                   save: bool = True) -> plt.Figure:
        """
        Visualize model performance recovery after pruning.
        
        Args:
            baseline_metrics: Original model metrics
            pruned_metrics: Metrics after pruning
            final_metrics: Metrics after fine-tuning
            experiment_id: Optional experiment identifier for file naming
            title: Title for the visualization
            figsize: Figure size (width, height) in inches
            save: Whether to save the visualization
            
        Returns:
            matplotlib Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define metrics to visualize
        metrics_to_plot = ['loss', 'perplexity']
        stages = ['Baseline', 'After Pruning', 'After Fine-tuning']
        
        # Create x positions for grouped bars
        x = np.arange(len(stages))
        width = 0.35
        
        # Create bars for each metric
        for i, metric in enumerate(metrics_to_plot):
            if metric in baseline_metrics and metric in pruned_metrics and metric in final_metrics:
                values = [baseline_metrics[metric], pruned_metrics[metric], final_metrics[metric]]
                offset = i * width - width/2 if len(metrics_to_plot) > 1 else 0
                ax.bar(x + offset, values, width, label=metric)
        
        # Add labels and title
        ax.set_xlabel('Training Stage')
        ax.set_ylabel('Metric Value')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(stages)
        ax.legend()
        
        # Calculate recovery rate
        if 'loss' in baseline_metrics and 'loss' in pruned_metrics and 'loss' in final_metrics:
            pruning_impact = pruned_metrics['loss'] - baseline_metrics['loss']
            recovery = pruned_metrics['loss'] - final_metrics['loss']
            
            if pruning_impact > 0:
                recovery_rate = (recovery / pruning_impact) * 100
                ax.text(0.98, 0.02, f"Recovery Rate: {recovery_rate:.1f}%", 
                       transform=ax.transAxes, fontsize=12, ha='right')
        
        # Save figure if requested
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_str = f"{experiment_id}_" if experiment_id else ""
            filename = f"{experiment_str}recovery_analysis_{timestamp}.png"
            
            # Ensure we're saving directly to the output directory
            save_path = os.path.join(self.output_dir, filename)
            
            # Create any needed parent directories
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save the figure with high quality
            fig.savefig(save_path, bbox_inches='tight', dpi=120)
            logger.info(f"Saved recovery analysis to {save_path}")
            
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
            
        return fig
        
    def visualize_entropy_changes(self,
                                 pre_entropy: Dict[str, List[float]],
                                 post_entropy: Dict[str, List[float]],
                                 experiment_id: str = None,
                                 title: str = "Entropy Changes After Fine-tuning",
                                 figsize: Tuple[int, int] = (12, 8),
                                 save: bool = True) -> plt.Figure:
        """
        Visualize changes in attention entropy before and after fine-tuning.
        
        Args:
            pre_entropy: Pre-finetuning entropy values by layer
            post_entropy: Post-finetuning entropy values by layer
            experiment_id: Optional experiment identifier for file naming
            title: Title for the visualization
            figsize: Figure size (width, height) in inches
            save: Whether to save the visualization
            
        Returns:
            matplotlib Figure object
        """
        # Calculate entropy deltas
        entropy_deltas = {}
        for layer in pre_entropy:
            if layer in post_entropy:
                # Convert to numpy arrays for calculation
                pre_values = np.array(pre_entropy[layer])
                post_values = np.array(post_entropy[layer])
                
                # Calculate the change in entropy
                entropy_deltas[layer] = post_values - pre_values
        
        # Convert to 2D array for visualization
        layers = sorted([int(k) if isinstance(k, str) else k for k in entropy_deltas.keys()])
        delta_arrays = []
        
        for layer in layers:
            layer_key = str(layer) if str(layer) in entropy_deltas else layer
            layer_data = entropy_deltas[layer_key]
            delta_arrays.append(layer_data)
            
        delta_array = np.stack(delta_arrays)
        
        # Create figure
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        
        # Plot entropy delta heatmap - handle NaN values
        # Replace NaNs with zeros for visualization purposes
        delta_array_clean = np.nan_to_num(delta_array, nan=0.0)
        
        im = axs[0].imshow(delta_array_clean, cmap='coolwarm', vmin=-0.5, vmax=0.5)
        axs[0].set_title('Entropy Changes (Post - Pre)')
        axs[0].set_xlabel('Head Index')
        axs[0].set_ylabel('Layer Index')
        
        # Add text to indicate if data contains NaNs
        if np.isnan(delta_array).any():
            axs[0].text(0.5, 0.01, 'Note: NaN values displayed as zero',
                      transform=axs[0].transAxes, fontsize=8, ha='center', 
                      bbox=dict(facecolor='white', alpha=0.6))
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=axs[0])
        cbar.set_label('Entropy Change')
        
        # Plot histogram of entropy changes
        flat_deltas = delta_array.flatten()
        
        # Handle NaN values in the data
        valid_deltas = flat_deltas[~np.isnan(flat_deltas)]
        
        if len(valid_deltas) > 0:
            # Use only valid data for histogram
            axs[1].hist(valid_deltas, bins=20, edgecolor='black')
            axs[1].set_title('Distribution of Entropy Changes')
            axs[1].set_xlabel('Entropy Change')
            axs[1].set_ylabel('Count')
            
            # Draw vertical line at zero
            axs[1].axvline(x=0, color='r', linestyle='--')
            
            # Calculate stats using only valid data
            num_increased = (valid_deltas > 0).sum()
            num_decreased = (valid_deltas < 0).sum()
            increased_percent = (num_increased / len(valid_deltas)) * 100 if len(valid_deltas) > 0 else 0
            decreased_percent = (num_decreased / len(valid_deltas)) * 100 if len(valid_deltas) > 0 else 0
        else:
            # No valid data, show a message
            axs[1].text(0.5, 0.5, 'No valid entropy change data available',
                       ha='center', va='center', transform=axs[1].transAxes)
            axs[1].set_title('Distribution of Entropy Changes')
            
            # Set default values for stats
            num_increased = 0
            num_decreased = 0
            increased_percent = 0
            decreased_percent = 0
        
        # Add stats to the plot
        axs[1].text(0.05, 0.95, 
                   f"Increased: {num_increased} ({increased_percent:.1f}%)\nDecreased: {num_decreased} ({decreased_percent:.1f}%)", 
                   transform=axs[1].transAxes, fontsize=10, va='top')
        
        # Add overall title
        fig.suptitle(title)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_str = f"{experiment_id}_" if experiment_id else ""
            filename = f"{experiment_str}entropy_changes_{timestamp}.png"
            
            # Ensure we're saving directly to the output directory
            save_path = os.path.join(self.output_dir, filename)
            
            # Create any needed parent directories
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save the figure with high quality
            fig.savefig(save_path, bbox_inches='tight', dpi=120)
            logger.info(f"Saved entropy changes visualization to {save_path}")
            
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
            
        return fig
        
    def visualize_all_metrics(self,
                             experiment_dir: str,
                             output_subdir: str = None,
                             experiment_id: str = None) -> Dict[str, plt.Figure]:
        """
        Generate all visualizations for an experiment based on its output files.
        
        Args:
            experiment_dir: Directory containing experiment output files
            output_subdir: Optional subdirectory to save visualizations in (defaults to root output dir)
            experiment_id: Optional experiment identifier for file naming
            
        Returns:
            Dictionary mapping visualization names to figure objects
        """
        # Use the main output directory by default (no subdirectories)
        output_dir = self.output_dir
        
        # Only use a subdirectory if explicitly requested
        if output_subdir:
            output_dir = os.path.join(self.output_dir, output_subdir)
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a new visualizer for this specific output directory
            temp_visualizer = PlasticityVisualizer(output_dir=output_dir, show_plots=self.show_plots)
        else:
            # Use this visualizer directly
            temp_visualizer = self
        
        # Dictionary to store generated figures
        figures = {}
        
        # Load experiment data files
        try:
            # Load parameters
            params_path = os.path.join(experiment_dir, "params.json")
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    params = json.load(f)
            else:
                params = {}
                
            # Load metrics
            metrics_path = os.path.join(experiment_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
            else:
                metrics = {}
                
            # Load performance history
            perf_path = os.path.join(experiment_dir, "performance_history.json")
            if os.path.exists(perf_path):
                with open(perf_path, 'r') as f:
                    performance_history = json.load(f)
            else:
                performance_history = {}
                
            # Load pruned heads
            pruned_path = os.path.join(experiment_dir, "pruned_heads.json")
            if os.path.exists(pruned_path):
                with open(pruned_path, 'r') as f:
                    pruned_heads = json.load(f)
            else:
                pruned_heads = []
                
            # Load entropy data
            pre_entropy_path = os.path.join(experiment_dir, "pre_entropy.json")
            post_entropy_path = os.path.join(experiment_dir, "post_entropy.json")
            
            pre_entropy = {}
            if os.path.exists(pre_entropy_path):
                with open(pre_entropy_path, 'r') as f:
                    pre_entropy = json.load(f)
                    
            post_entropy = {}
            if os.path.exists(post_entropy_path):
                with open(post_entropy_path, 'r') as f:
                    post_entropy = json.load(f)
            
            # Generate all visualizations
            
            # 1. Recovery analysis
            if 'baseline' in metrics and 'post_pruning' in metrics and 'final' in metrics:
                figures['recovery'] = temp_visualizer.visualize_recovery_analysis(
                    baseline_metrics=metrics['baseline'],
                    pruned_metrics=metrics['post_pruning'],
                    final_metrics=metrics['final'],
                    experiment_id=experiment_id,
                    title="Model Recovery After Pruning"
                )
                
            # 2. Pruned heads visualization
            if pruned_heads and 'model_name' in params:
                # Determine number of layers and heads based on model
                num_layers = 12  # Default for small models
                num_heads = 12   # Default for small models
                
                # Extract from pruned_heads list to find max values
                if pruned_heads:
                    max_layer = max(head[0] for head in pruned_heads) + 1
                    max_head = max(head[1] for head in pruned_heads) + 1
                    num_layers = max(num_layers, max_layer)
                    num_heads = max(num_heads, max_head)
                
                figures['pruned_heads'] = temp_visualizer.visualize_pruned_heads(
                    pruned_heads=pruned_heads,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    experiment_id=experiment_id,
                    title=f"Pruned Heads ({params.get('pruning_strategy', 'unknown')} strategy)"
                )
                
            # 3. Entropy heatmap
            if pre_entropy:
                figures['pre_entropy'] = temp_visualizer.visualize_entropy_heatmap(
                    entropy_data=pre_entropy,
                    experiment_id=experiment_id,
                    title="Pre-Pruning Entropy Heatmap"
                )
                
            if post_entropy:
                figures['post_entropy'] = temp_visualizer.visualize_entropy_heatmap(
                    entropy_data=post_entropy,
                    experiment_id=experiment_id,
                    title="Post-Finetuning Entropy Heatmap"
                )
                
            # 4. Entropy changes
            if pre_entropy and post_entropy:
                figures['entropy_changes'] = temp_visualizer.visualize_entropy_changes(
                    pre_entropy=pre_entropy,
                    post_entropy=post_entropy,
                    experiment_id=experiment_id,
                    title="Entropy Changes After Fine-tuning"
                )
                
            # 5. Training progress
            if performance_history:
                # Prepare metrics dictionary
                metrics_dict = {}
                
                # Extract metrics from performance history
                for step, step_metrics in performance_history.items():
                    # Convert step to integer to ensure proper sorting
                    step_int = int(step)
                    
                    for metric_name, value in step_metrics.items():
                        if metric_name not in metrics_dict:
                            metrics_dict[metric_name] = []
                            
                        # Ensure metrics are added in order
                        while len(metrics_dict[metric_name]) < step_int:
                            metrics_dict[metric_name].append(None)
                            
                        metrics_dict[metric_name].append(value)
                
                if metrics_dict:
                    figures['training_progress'] = temp_visualizer.visualize_training_progress(
                        metrics=metrics_dict,
                        experiment_id=experiment_id,
                        title="Training Progress After Pruning"
                    )
                    
            # 6. Create a combined metrics grid
            if len(figures) >= 2:
                # Create a grid of all visualizations
                grid_cols = min(len(figures), 2)
                grid_rows = (len(figures) + grid_cols - 1) // grid_cols
                
                combined_fig, axs = plt.subplots(grid_rows, grid_cols, 
                                                figsize=(15, 8 * grid_rows))
                
                # Flatten axs if it's a 2D array
                if grid_rows > 1 and grid_cols > 1:
                    axs = axs.flatten()
                elif grid_rows == 1 and grid_cols > 1:
                    axs = axs.flatten()
                elif grid_rows > 1 and grid_cols == 1:
                    axs = axs.flatten()
                else:
                    axs = [axs]
                
                # Add each figure to the grid
                for i, (name, fig) in enumerate(figures.items()):
                    if i < len(axs):
                        # Extract the image data from the figure
                        fig.canvas.draw()
                        img = np.array(fig.canvas.renderer.buffer_rgba())
                        
                        # Display the image
                        axs[i].imshow(img)
                        axs[i].set_title(name.replace('_', ' ').title())
                        axs[i].axis('off')
                
                # Hide any unused subplots
                for i in range(len(figures), len(axs)):
                    axs[i].axis('off')
                
                # Add overall title
                model_name = params.get('model_name', 'Unknown')
                pruning_strategy = params.get('pruning_strategy', 'Unknown')
                pruning_level = params.get('pruning_level', 0)
                
                combined_fig.suptitle(
                    f"Neural Plasticity Experiment: {model_name}\n"
                    f"Strategy: {pruning_strategy}, Level: {pruning_level:.2f}",
                    fontsize=16
                )
                
                # Adjust layout
                combined_fig.tight_layout(rect=[0, 0, 1, 0.95])
                
                # Save the combined figure
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                experiment_str = f"{experiment_id}_" if experiment_id else ""
                filename = f"{experiment_str}metrics_grid_{timestamp}.png"
                
                # Ensure we're saving directly to the output directory 
                # (not using a subdirectory for this grid view)
                save_path = os.path.join(self.output_dir, filename)
                
                # Create any needed parent directories
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Save the figure with high quality
                combined_fig.savefig(save_path, bbox_inches='tight', dpi=120)
                logger.info(f"Saved combined metrics grid to {save_path}")
                
                # Add to figures dictionary
                figures['metrics_grid'] = combined_fig
                
                if self.show_plots:
                    plt.show()
                else:
                    plt.close(combined_fig)
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            
        return figures


# Example usage
if __name__ == "__main__":
    # Set up command line argument parsing
    import argparse
    
    parser = argparse.ArgumentParser(description="Process neural plasticity experiment visualizations")
    parser.add_argument("--experiment_dir", type=str, required=True, 
                       help="Path to experiment directory containing result files")
    parser.add_argument("--output_dir", type=str, default="./output", 
                       help="Output directory for visualizations")
    parser.add_argument("--show", action="store_true", 
                       help="Show plots in addition to saving them")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create visualizer
    visualizer = PlasticityVisualizer(output_dir=args.output_dir, show_plots=args.show)
    
    # Generate all visualizations
    visualizer.visualize_all_metrics(
        experiment_dir=args.experiment_dir,
        output_subdir="experiment_visuals"
    )
    
    logger.info("Visualization processing complete")