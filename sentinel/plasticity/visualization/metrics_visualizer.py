"""
Metrics visualization component for Neural Plasticity.

This module provides visualization tools for tracking and displaying
training metrics during neural plasticity experiments.

Version: v0.0.34 (2025-04-20 14:30:00)
"""

import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any

class MetricsVisualizer:
    """
    Visualizes training metrics for neural plasticity experiments.
    
    This class is responsible for:
    1. Generating training progress visualizations
    2. Saving metrics history plots
    3. Creating comparison visualizations between experiment phases
    
    This visualization class is completely decoupled from the core
    plasticity system and can be omitted in production environments.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the metrics visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure matplotlib for non-interactive backend
        plt.switch_backend('agg')
        
    def visualize_training_progress(
        self, 
        metrics: Dict[Union[str, int], Dict[str, float]], 
        experiment_id: str,
        phase: str = "complete"
    ) -> str:
        """
        Generate visualization of training progress metrics.
        
        Args:
            metrics: Dictionary mapping steps to metric dictionaries
            experiment_id: Experiment identifier
            phase: Current experiment phase (warmup, pruning, fine-tuning, complete)
            
        Returns:
            Path to saved visualization
        """
        # Create visualization directory
        viz_dir = os.path.join(self.output_dir, experiment_id, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data for visualization
        steps = []
        losses = []
        perplexities = []
        
        # Handle both string and integer keys
        int_keys = []
        for step in metrics.keys():
            try:
                if isinstance(step, str):
                    int_keys.append(int(step))
                else:
                    int_keys.append(step)
            except (ValueError, TypeError):
                # Skip keys that can't be converted to integers
                continue
                
        # Sort the integer keys
        sorted_steps = sorted(int_keys)
        
        for step_int in sorted_steps:
            # Get the metrics for this step - handle both string and integer keys
            if str(step_int) in metrics:
                step_metrics = metrics[str(step_int)]
            elif step_int in metrics:
                step_metrics = metrics[step_int]
            else:
                continue
                
            if isinstance(step_metrics, dict):
                steps.append(step_int)
                
                if "loss" in step_metrics:
                    losses.append(step_metrics["loss"])
                    
                if "perplexity" in step_metrics:
                    perplexities.append(step_metrics["perplexity"])
        
        # Make sure we have data to plot
        if steps and all(s is not None for s in steps) and len(steps) > 1:
            # Create a sequential range for the x-axis (0 to N for each step)
            x_range = range(len(steps))
            
            # Create primary y-axis for loss
            if losses and all(l is not None for l in losses):
                ax.plot(x_range, losses, 'b-', linewidth=2, label='Loss')
                ax.set_xlabel('Training Steps')
                ax.set_ylabel('Loss', color='b')
                ax.tick_params(axis='y', labelcolor='b')
                
                # Set x-tick positions to match step numbers
                ax.set_xticks(x_range)
                ax.set_xticklabels([str(s) for s in steps])
                
                # Add vertical line for pruning step if this is the complete visualization
                if phase == "complete" and len(steps) > 2:
                    # Assume pruning happens at 1/3 of the way through
                    pruning_idx = len(steps) // 3
                    ax.axvline(x=pruning_idx, color='r', linestyle='--', label='Pruning')
                    
                    # Add shaded regions for different phases
                    ax.axvspan(0, pruning_idx, alpha=0.2, color='blue', label='Warmup')
                    ax.axvspan(pruning_idx, len(steps)-1, alpha=0.2, color='orange', label='Fine-tuning')
            
            # Create secondary y-axis for perplexity if available
            if perplexities and all(p is not None for p in perplexities):
                ax2 = ax.twinx()
                
                # Set up the perplexity axis based on the range of values
                if max(perplexities) > 100:
                    # Use log scale for better visualization when values span multiple orders of magnitude
                    ax2.set_yscale('log')
                    ax2.plot(x_range, perplexities, 'g-', linewidth=2, label='Perplexity (log scale)')
                    ax2.set_ylabel('Perplexity (log scale)', color='g')
                else:
                    # Use linear scale for smaller, similar-magnitude values
                    ax2.plot(x_range, perplexities, 'g-', linewidth=2, label='Perplexity')
                    ax2.set_ylabel('Perplexity', color='g')
                    # Set reasonable upper limit with some headroom
                    ax2.set_ylim(0, max(perplexities) * 1.2)
                
                ax2.tick_params(axis='y', labelcolor='g')
                
                # Add legends for both axes
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            else:
                ax.legend()
        else:
            # No data yet, show placeholder
            ax.text(0.5, 0.5, 'No training progress data available yet', 
                   ha='center', va='center', transform=ax.transAxes)
        
        # Set title based on phase
        if phase == "warmup":
            title = "Neural Plasticity: Warmup Phase"
        elif phase == "pruning":
            title = "Neural Plasticity: After Pruning"
        elif phase == "fine-tuning":
            title = "Neural Plasticity: Fine-Tuning Progress"
        else:
            title = "Neural Plasticity: Full Training Process"
            
        ax.set_title(title)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        filename = f"training_progress_{phase}.png"
        save_path = os.path.join(viz_dir, filename)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        return save_path
    
    def save_metrics(self, metrics: Dict[str, Any], experiment_id: str) -> str:
        """
        Save metrics data to JSON file.
        
        Args:
            metrics: Dictionary of metrics data
            experiment_id: Experiment identifier
            
        Returns:
            Path to saved JSON file
        """
        # Create experiment directory
        experiment_dir = os.path.join(self.output_dir, experiment_id)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save metrics to JSON file
        metrics_path = os.path.join(experiment_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        return metrics_path