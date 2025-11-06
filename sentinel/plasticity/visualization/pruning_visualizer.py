"""
Pruning visualization component for Neural Plasticity.

This module provides visualization tools for displaying information about
pruned attention heads in neural plasticity experiments.

Version: v0.0.34 (2025-04-20 14:30:00)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any, Tuple

class PruningVisualizer:
    """
    Visualizes pruning data for neural plasticity experiments.
    
    This class is responsible for:
    1. Generating visualizations of which heads were pruned
    2. Creating heatmaps showing pruning probabilities
    3. Visualizing pruning decisions and their justifications
    
    This visualization class is completely decoupled from the core
    plasticity system and can be omitted in production environments.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the pruning visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure matplotlib for non-interactive backend
        plt.switch_backend('agg')
    
    def visualize_pruned_heads(
        self, 
        pruned_heads: List[Tuple[int, int, float]],
        num_layers: int,
        num_heads: int,
        experiment_id: str,
        pruning_strategy: str = "entropy"
    ) -> str:
        """
        Generate visualization of which attention heads were pruned.
        
        Args:
            pruned_heads: List of (layer_idx, head_idx, score) tuples
            num_layers: Number of layers in the model
            num_heads: Number of heads per layer
            experiment_id: Experiment identifier
            pruning_strategy: Strategy used for pruning (entropy, magnitude, etc.)
            
        Returns:
            Path to saved visualization
        """
        # Create visualization directory
        viz_dir = os.path.join(self.output_dir, experiment_id, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create a mask of pruned heads (1 for pruned, 0 for active)
        pruned_mask = np.zeros((num_layers, num_heads))
        pruning_scores = np.zeros((num_layers, num_heads)) - 1  # -1 means no score
        
        # Fill in the mask and scores
        for layer_idx, head_idx, score in pruned_heads:
            if 0 <= layer_idx < num_layers and 0 <= head_idx < num_heads:
                pruned_mask[layer_idx, head_idx] = 1
                pruning_scores[layer_idx, head_idx] = score
        
        # Create a figure with three plots
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Binary mask of pruned heads
        axes[0].set_title("Pruned Attention Heads", fontsize=16, pad=20)
        binary_heatmap = axes[0].imshow(pruned_mask, cmap='Reds', interpolation='nearest', aspect='auto')
        axes[0].set_xlabel("Head Index", fontsize=14, labelpad=10)
        axes[0].set_ylabel("Layer Index", fontsize=14, labelpad=10)
        axes[0].set_xticks(np.arange(num_heads))
        axes[0].set_yticks(np.arange(num_layers))
        
        # Add text labels for pruned heads
        for i in range(num_layers):
            for j in range(num_heads):
                if pruned_mask[i, j] > 0:
                    axes[0].text(j, i, "X", ha="center", va="center", 
                               color="black", fontweight="bold", fontsize=12)
        
        # Add a descriptive label showing pruning percentage
        total_heads = num_layers * num_heads
        pruned_count = len(pruned_heads)
        pruning_percentage = (pruned_count / total_heads) * 100
        axes[0].text(0.5, -0.1, 
                   f"Pruned: {pruned_count}/{total_heads} heads ({pruning_percentage:.1f}%)", 
                   transform=axes[0].transAxes, ha="center", fontsize=12)
        
        # Plot 2: Heatmap of pruning scores (if available)
        if np.any(pruning_scores > 0):
            masked_scores = np.ma.masked_where(pruning_scores < 0, pruning_scores)
            cmap = plt.cm.viridis
            cmap.set_bad('white', 1.0)
            
            score_heatmap = axes[1].imshow(
                masked_scores, cmap=cmap, interpolation='nearest', aspect='auto'
            )
            axes[1].set_title(f"{pruning_strategy.capitalize()} Pruning Scores", 
                            fontsize=16, pad=20)
            axes[1].set_xlabel("Head Index", fontsize=14, labelpad=10)
            axes[1].set_ylabel("Layer Index", fontsize=14, labelpad=10)
            axes[1].set_xticks(np.arange(num_heads))
            axes[1].set_yticks(np.arange(num_layers))
            
            # Add colorbar
            plt.colorbar(score_heatmap, ax=axes[1], label=f"{pruning_strategy.capitalize()} Score")
            
            # Add score labels for pruned heads if not too many
            if pruned_count <= 30:
                for layer_idx, head_idx, score in pruned_heads:
                    if 0 <= layer_idx < num_layers and 0 <= head_idx < num_heads:
                        axes[1].text(head_idx, layer_idx, f"{score:.2f}", 
                                   ha="center", va="center", color="black", 
                                   fontsize=8, fontweight="bold")
        else:
            # If scores aren't available, show a message
            axes[1].text(0.5, 0.5, "Pruning scores not available", 
                       ha="center", va="center", transform=axes[1].transAxes,
                       fontsize=14)
            axes[1].set_title("Pruning Scores", fontsize=16, pad=20)
            axes[1].set_xlabel("Head Index", fontsize=14, labelpad=10)
            axes[1].set_ylabel("Layer Index", fontsize=14, labelpad=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        filename = f"pruned_heads_{pruning_strategy}.png"
        save_path = os.path.join(viz_dir, filename)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        return save_path
        
    def visualize_pruning_distribution(
        self,
        head_scores: np.ndarray,
        threshold: float,
        experiment_id: str,
        pruning_strategy: str = "entropy"
    ) -> str:
        """
        Visualize the distribution of head scores with pruning threshold.
        
        Args:
            head_scores: Array of scores for each head
            threshold: Pruning threshold
            experiment_id: Experiment identifier
            pruning_strategy: Strategy used for pruning
            
        Returns:
            Path to saved visualization
        """
        # Create visualization directory
        viz_dir = os.path.join(self.output_dir, experiment_id, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Flatten scores if multi-dimensional
        if head_scores.ndim > 1:
            head_scores = head_scores.flatten()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        n, bins, patches = ax.hist(head_scores, bins=30, color='skyblue', 
                                   alpha=0.7, edgecolor='black')
        
        # Add vertical line for threshold
        ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                  label=f'Pruning Threshold: {threshold:.3f}')
        
        # Highlight pruned region
        if pruning_strategy == "entropy":
            # For entropy pruning, higher values get pruned
            ax.axvspan(threshold, max(bins), alpha=0.2, color='red', label='Pruned Region')
        else:
            # For magnitude pruning, lower values get pruned
            ax.axvspan(min(bins), threshold, alpha=0.2, color='red', label='Pruned Region')
        
        # Add labels and title
        ax.set_xlabel(f'{pruning_strategy.capitalize()} Score', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_title(f'Distribution of {pruning_strategy.capitalize()} Scores', 
                    fontsize=16, pad=20)
        
        # Add legend
        ax.legend()
        
        # Add text with pruning statistics
        if pruning_strategy == "entropy":
            pruned_count = (head_scores >= threshold).sum()
        else:
            pruned_count = (head_scores <= threshold).sum()
            
        total_count = len(head_scores)
        pruned_percent = (pruned_count / total_count) * 100
        
        ax.text(0.05, 0.95, 
               f"Pruned: {pruned_count}/{total_count} heads ({pruned_percent:.1f}%)", 
               transform=ax.transAxes, fontsize=12, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        filename = f"pruning_distribution_{pruning_strategy}.png"
        save_path = os.path.join(viz_dir, filename)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        return save_path