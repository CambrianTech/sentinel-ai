"""
Visualization for Pruning Experiments

This module provides visualization and progress tracking for pruning experiments.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from typing import Dict, Any, List, Tuple, Optional


class ProgressTracker:
    """Track metrics throughout the pruning and fine-tuning process."""
    
    def __init__(self):
        """Initialize the progress tracker."""
        self.metrics = {
            "loss": [],
            "perplexity": [],
            "steps": [],
            "pruning_level": None,
            "pruned_heads": [],
            "generation_samples": []
        }
        
        # Create visualizations
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 5))
        
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
        self._update_plots()
        
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
        steps = self.metrics["steps"]
        loss = self.metrics["loss"]
        ppl = self.metrics["perplexity"]
        
        if not steps:
            return
            
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
        plt.pause(0.001)
        
    def save_plots(self, path: str) -> None:
        """
        Save plots to file.
        
        Args:
            path: Path to save the plots to
        """
        plt.savefig(path)
        print(f"Plots saved to: {path}")
        
    def save_metrics(self, path: str) -> None:
        """
        Save metrics to file.
        
        Args:
            path: Path to save the metrics to
        """
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to: {path}")
            
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