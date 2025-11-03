"""
Entropy visualization component for Neural Plasticity.

This module provides visualization tools for displaying entropy data
from neural plasticity experiments.

Version: v0.0.34 (2025-04-20 14:30:00)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Optional, Union, Any, Tuple

class EntropyVisualizer:
    """
    Visualizes entropy data for neural plasticity experiments.
    
    This class is responsible for:
    1. Generating heatmaps of attention entropy
    2. Visualizing entropy changes before/after pruning and fine-tuning
    3. Creating entropy distribution comparisons
    
    This visualization class is completely decoupled from the core
    plasticity system and can be omitted in production environments.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the entropy visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure matplotlib for non-interactive backend
        plt.switch_backend('agg')
    
    def visualize_entropy_heatmap(
        self, 
        entropy_data: Union[Dict[Union[str, int], torch.Tensor], torch.Tensor, np.ndarray],
        experiment_id: str,
        phase: str = "pre_pruning"
    ) -> str:
        """
        Generate heatmap visualization of attention entropy values.
        
        Args:
            entropy_data: Dictionary mapping layer indices to entropy tensors,
                          or a tensor/array of shape [layers, heads]
            experiment_id: Experiment identifier
            phase: Current phase (pre_pruning, post_pruning, post_finetuning)
            
        Returns:
            Path to saved visualization
        """
        # Create visualization directory
        viz_dir = os.path.join(self.output_dir, experiment_id, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Convert dictionary to numpy array if needed
        if isinstance(entropy_data, dict):
            # Convert string keys to integers and sort
            layers = sorted([int(k) if isinstance(k, str) else k for k in entropy_data.keys()])
            
            # Extract the entropy tensors for each layer
            entropy_arrays = []
            for layer in layers:
                layer_key = str(layer) if str(layer) in entropy_data else layer
                layer_data = entropy_data[layer_key]
                
                # Convert to numpy array if needed
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
        
        # Create a larger figure with better resolution
        plt.figure(figsize=(14, 10))
        
        # Create a more readable layout with subplots
        gs = plt.GridSpec(1, 20)
        ax = plt.subplot(gs[0, :18])  # Main heatmap
        
        # Handle all-NaN slices by setting reasonable default limits
        vmin = np.nanmin(entropy_array) if not np.all(np.isnan(entropy_array)) else 0
        vmax = np.nanmax(entropy_array) if not np.all(np.isnan(entropy_array)) else 1
        
        # Better color gradient for entropy visualization
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=vmin, vmax=max(vmin+0.1, vmax))  # Ensure min/max are distinct
        
        # Replace NaNs with masked array for better visualization
        masked_entropy = np.ma.array(entropy_array, mask=np.isnan(entropy_array))
        
        # Plot heatmap with improved settings
        im = ax.imshow(masked_entropy, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
        
        # Add colorbar with better formatting
        cax = plt.subplot(gs[0, 19:])  # Colorbar
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Entropy Value', fontsize=12, weight='bold')
        cbar.ax.tick_params(labelsize=10)
        
        # Set title based on phase with improved styling
        if phase == "pre_pruning":
            title = "Pre-Pruning Attention Entropy"
        elif phase == "post_pruning":
            title = "Post-Pruning Attention Entropy"
        elif phase == "post_finetuning":
            title = "Post-Finetuning Attention Entropy"
        else:
            title = "Attention Entropy Heatmap"
            
        ax.set_title(title, fontsize=16, weight='bold', pad=20)
        
        # Better axis labels
        ax.set_xlabel('Head Index', fontsize=14, labelpad=10)
        ax.set_ylabel('Layer Index', fontsize=14, labelpad=10)
        
        # Generate tick labels
        ax.set_xticks(np.arange(entropy_array.shape[1]))
        ax.set_yticks(np.arange(entropy_array.shape[0]))
        
        # Add text annotations for clarity, but only for smaller models
        if entropy_array.shape[0] * entropy_array.shape[1] <= 96:  # 12x8 grid or smaller
            # Custom text colors based on background brightness
            for i in range(entropy_array.shape[0]):
                for j in range(entropy_array.shape[1]):
                    val = entropy_array[i, j]
                    if not np.isnan(val):
                        # Determine text color based on cell darkness
                        color_val = cmap(norm(val))
                        brightness = 0.299 * color_val[0] + 0.587 * color_val[1] + 0.114 * color_val[2]
                        text_color = 'white' if brightness < 0.7 else 'black'
                        
                        ax.text(j, i, f'{val:.2f}', ha="center", va="center", 
                               color=text_color, fontsize=8, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        filename = f"entropy_heatmap_{phase}.png"
        save_path = os.path.join(viz_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return save_path
    
    def visualize_entropy_changes(
        self, 
        pre_entropy: Union[Dict[Union[str, int], torch.Tensor], torch.Tensor, np.ndarray],
        post_entropy: Union[Dict[Union[str, int], torch.Tensor], torch.Tensor, np.ndarray],
        experiment_id: str
    ) -> str:
        """
        Visualize changes in entropy before and after fine-tuning.
        
        Args:
            pre_entropy: Entropy values before fine-tuning
            post_entropy: Entropy values after fine-tuning
            experiment_id: Experiment identifier
            
        Returns:
            Path to saved visualization
        """
        # Create visualization directory
        viz_dir = os.path.join(self.output_dir, experiment_id, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Convert inputs to numpy arrays
        pre_array = self._ensure_numpy(pre_entropy)
        post_array = self._ensure_numpy(post_entropy)
        
        # Calculate entropy deltas
        if pre_array.shape == post_array.shape:
            delta_array = post_array - pre_array
        else:
            # Handle shape mismatch - use minimum common shape
            min_layers = min(pre_array.shape[0], post_array.shape[0])
            min_heads = min(pre_array.shape[1], post_array.shape[1])
            delta_array = post_array[:min_layers, :min_heads] - pre_array[:min_layers, :min_heads]
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot pre-pruning entropy
        self._plot_entropy_heatmap(pre_array, axes[0], "Pre-Fine-Tuning Entropy")
        
        # Plot post-pruning entropy
        self._plot_entropy_heatmap(post_array, axes[1], "Post-Fine-Tuning Entropy")
        
        # Plot delta entropy with different colormap centered at zero
        self._plot_entropy_delta(delta_array, axes[2], "Entropy Change")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        filename = "entropy_changes.png"
        save_path = os.path.join(viz_dir, filename)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        return save_path
    
    def _ensure_numpy(
        self, 
        data: Union[Dict[Union[str, int], torch.Tensor], torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        """
        Convert various input formats to numpy array.
        
        Args:
            data: Input data (dictionary, tensor, or array)
            
        Returns:
            Numpy array version of the data
        """
        if isinstance(data, dict):
            # Convert dictionary to numpy array
            layers = sorted([int(k) if isinstance(k, str) else k for k in data.keys()])
            arrays = []
            
            for layer in layers:
                layer_key = str(layer) if str(layer) in data else layer
                layer_data = data[layer_key]
                
                if isinstance(layer_data, list):
                    arrays.append(np.array(layer_data))
                elif isinstance(layer_data, torch.Tensor):
                    arrays.append(layer_data.detach().cpu().numpy())
                else:
                    arrays.append(layer_data)
            
            return np.stack(arrays)
        elif isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        else:
            return np.array(data)
    
    def _plot_entropy_heatmap(
        self, 
        entropy_array: np.ndarray, 
        ax: plt.Axes, 
        title: str
    ) -> None:
        """
        Plot entropy heatmap on the given axes.
        
        Args:
            entropy_array: Numpy array of entropy values
            ax: Matplotlib axes to plot on
            title: Plot title
        """
        # Choose colormap and normalization
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=np.nanmin(entropy_array), vmax=np.nanmax(entropy_array))
        
        # Plot heatmap
        im = ax.imshow(entropy_array, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add labels and title
        ax.set_xlabel('Head Index')
        ax.set_ylabel('Layer Index')
        ax.set_title(title)
        
        # Add tick marks
        ax.set_xticks(np.arange(entropy_array.shape[1]))
        ax.set_yticks(np.arange(entropy_array.shape[0]))
    
    def _plot_entropy_delta(
        self, 
        delta_array: np.ndarray, 
        ax: plt.Axes, 
        title: str
    ) -> None:
        """
        Plot entropy delta heatmap on the given axes.
        
        Args:
            delta_array: Numpy array of entropy changes
            ax: Matplotlib axes to plot on
            title: Plot title
        """
        # Choose colormap and normalization centered at zero
        cmap = plt.cm.RdBu_r
        
        # Handle potential all-NaN array
        if np.all(np.isnan(delta_array)):
            abs_max = 1.0  # Default value when all values are NaN
        else:
            abs_max = max(abs(np.nanmin(delta_array)), abs(np.nanmax(delta_array)))
            abs_max = max(abs_max, 0.01)  # Ensure we have a non-zero range
            
        norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)
        
        # Create masked array to properly handle NaNs
        masked_delta = np.ma.array(delta_array, mask=np.isnan(delta_array))
        
        # Plot heatmap
        im = ax.imshow(masked_delta, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add labels and title
        ax.set_xlabel('Head Index')
        ax.set_ylabel('Layer Index')
        ax.set_title(title)
        
        # Add tick marks
        ax.set_xticks(np.arange(delta_array.shape[1]))
        ax.set_yticks(np.arange(delta_array.shape[0]))