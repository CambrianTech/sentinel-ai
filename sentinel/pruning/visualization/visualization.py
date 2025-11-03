"""
Visualization for Pruning Experiments

This module provides visualization and progress tracking for pruning experiments.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from typing import Dict, Any, List, Tuple, Optional


class ProgressTracker:
    """Track metrics throughout the pruning and fine-tuning process."""
    
    def __init__(self, disable_plotting=False):
        """Initialize the progress tracker."""
        self.metrics = {
            "loss": [],
            "perplexity": [],
            "steps": [],
            "pruning_level": None,
            "pruned_heads": [],
            "generation_samples": []
        }
        
        self.disable_plotting = disable_plotting
        
        if not disable_plotting:
            try:
                # Create visualizations
                self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 5))
            except Exception as e:
                print(f"Warning: Could not create visualization plots: {e}")
                self.disable_plotting = True
        
    def update(self, step: int, loss: float, perplexity: float, generation_sample: Optional[str] = None) -> None:
        """
        Update metrics with new values.
        
        Args:
            step: The current step
            loss: The loss value
            perplexity: The perplexity value
            generation_sample: Optional sample of generated text
        """
        self.metrics["steps"].append(step)
        self.metrics["loss"].append(loss)
        self.metrics["perplexity"].append(perplexity)
        
        if generation_sample is not None:
            self.metrics["generation_samples"].append({
                "step": step,
                "text": generation_sample
            })
        
        # Update visualization
        if not self.disable_plotting:
            try:
                self._update_plots()
            except Exception as e:
                print(f"Warning: Failed to update plots: {e}")
                self.disable_plotting = True
        
    def set_pruning_info(self, level: float, pruned_heads: List[Tuple[int, int]]) -> None:
        """
        Set pruning information.
        
        Args:
            level: The pruning level (0-1)
            pruned_heads: List of pruned heads as (layer, head) tuples
        """
        self.metrics["pruning_level"] = level
        self.metrics["pruned_heads"] = [(int(layer), int(head)) for layer, head in pruned_heads]
        
    def _update_plots(self) -> None:
        """Update visualization plots."""
        if self.disable_plotting:
            return
            
        steps = self.metrics["steps"]
        loss = self.metrics["loss"]
        ppl = self.metrics["perplexity"]
        
        if not steps:
            return
            
        try:
            # Clear previous plots
            self.axes[0].clear()
            self.axes[1].clear()
            
            # Plot loss
            self.axes[0].plot(steps, loss, 'b-')
            self.axes[0].set_title('Loss')
            self.axes[0].set_xlabel('Step')
            self.axes[0].set_ylabel('Loss')
            self.axes[0].grid(True)
            
            # Plot perplexity
            self.axes[1].plot(steps, ppl, 'r-')
            self.axes[1].set_title('Perplexity (lower is better)')
            self.axes[1].set_xlabel('Step')
            self.axes[1].set_ylabel('Perplexity')
            self.axes[1].grid(True)
            
            self.fig.tight_layout()
            plt.draw()
            
            # Only pause if interactive mode is enabled
            if plt.isinteractive():
                plt.pause(0.001)
        except Exception as e:
            print(f"Warning: Error updating plots: {e}")
            self.disable_plotting = True
        
    def save_plots(self, path: str) -> None:
        """
        Save plots to file.
        
        Args:
            path: Path to save the plots to
        """
        if self.disable_plotting:
            print(f"Warning: Plotting is disabled, cannot save plots to {path}")
            return
            
        try:
            plt.savefig(path)
            print(f"Plots saved to: {path}")
        except Exception as e:
            print(f"Error saving plots to {path}: {e}")
        
    def save_metrics(self, path: str) -> None:
        """
        Save metrics to file.
        
        Args:
            path: Path to save the metrics to
        """
        try:
            # Convert any non-serializable objects to strings
            metrics_to_save = self.metrics.copy()
            
            # Convert tensors to regular Python values if needed
            if "loss" in metrics_to_save:
                metrics_to_save["loss"] = [float(x) if hasattr(x, "item") else float(x) for x in metrics_to_save["loss"]]
            if "perplexity" in metrics_to_save:
                metrics_to_save["perplexity"] = [float(x) if hasattr(x, "item") else float(x) for x in metrics_to_save["perplexity"]]
            
            # Save to file
            with open(path, 'w') as f:
                json.dump(metrics_to_save, f, indent=2)
            print(f"Metrics saved to: {path}")
        except Exception as e:
            print(f"Error saving metrics to {path}: {e}")
            
    def get_summary(self) -> Dict[str, Any]:
        """
        Return a summary of key metrics.
        
        Returns:
            Dictionary of summary metrics
        """
        if not self.metrics["perplexity"]:
            return {}
            
        return {
            "pruning_level": self.metrics["pruning_level"],
            "pruned_heads_count": len(self.metrics["pruned_heads"]),
            "initial_loss": self.metrics["loss"][0] if self.metrics["loss"] else None,
            "final_loss": self.metrics["loss"][-1] if self.metrics["loss"] else None,
            "initial_perplexity": self.metrics["perplexity"][0] if self.metrics["perplexity"] else None,
            "final_perplexity": self.metrics["perplexity"][-1] if self.metrics["perplexity"] else None,
            "improvement_percent": ((self.metrics["perplexity"][0] - self.metrics["perplexity"][-1]) / 
                                   self.metrics["perplexity"][0] * 100) 
                                   if (self.metrics["perplexity"] and len(self.metrics["perplexity"]) > 1) else None
        }


def visualize_head_importance(importance: torch.Tensor, pruned_heads: Optional[List[Tuple[int, int]]] = None) -> plt.Figure:
    """
    Visualize the importance of attention heads.
    
    Args:
        importance: Tensor of head importance scores
        pruned_heads: Optional list of pruned heads
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get dimensions
    num_layers, num_heads = importance.shape
    
    # Create a heatmap
    im = ax.imshow(importance, cmap="viridis")
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label="Importance")
    
    # Add labels
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("Attention Head Importance")
    
    # Set ticks
    ax.set_xticks(np.arange(num_heads))
    ax.set_yticks(np.arange(num_layers))
    ax.set_xticklabels([str(i) for i in range(num_heads)])
    ax.set_yticklabels([str(i) for i in range(num_layers)])
    
    # Rotate x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Mark pruned heads if provided
    if pruned_heads:
        for layer_idx, head_idx in pruned_heads:
            rect = plt.Rectangle((head_idx - 0.5, layer_idx - 0.5), 1, 1, fill=False, 
                                edgecolor='red', linewidth=2)
            ax.add_patch(rect)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig