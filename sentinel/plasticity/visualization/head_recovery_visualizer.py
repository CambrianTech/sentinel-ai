"""
Head recovery visualization component for Neural Plasticity.

This module provides visualization tools for displaying information about
head recovery after pruning in neural plasticity experiments.

Version: v0.0.34 (2025-04-20 14:30:00)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any, Tuple

class HeadRecoveryVisualizer:
    """
    Visualizes head recovery information for neural plasticity experiments.
    
    This class is responsible for:
    1. Generating visualizations of recovery metrics
    2. Creating comparison visualizations between baseline, pruned, and final states
    3. Visualizing regrowth patterns across attention heads
    
    This visualization class is completely decoupled from the core
    plasticity system and can be omitted in production environments.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the recovery visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure matplotlib for non-interactive backend
        plt.switch_backend('agg')
    
    def visualize_recovery_analysis(
        self,
        baseline_metrics: Dict[str, float],
        pruned_metrics: Dict[str, float],
        final_metrics: Dict[str, float],
        experiment_id: str
    ) -> str:
        """
        Generate visualization comparing metrics across experiment phases.
        
        Args:
            baseline_metrics: Metrics from baseline model
            pruned_metrics: Metrics after pruning
            final_metrics: Metrics after fine-tuning
            experiment_id: Experiment identifier
            
        Returns:
            Path to saved visualization
        """
        # Create visualization directory
        viz_dir = os.path.join(self.output_dir, experiment_id, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Collect metrics to visualize
        metrics_to_plot = []
        
        # Identify metrics present in all phases
        for metric in baseline_metrics:
            if metric in pruned_metrics and metric in final_metrics:
                # Skip metrics that aren't numeric or are just metadata
                if not self._is_numeric(baseline_metrics[metric]):
                    continue
                
                if metric in ["original_perplexity", "perplexity_text", "perplexity_scaled"]:
                    continue
                    
                metrics_to_plot.append(metric)
        
        # If no common metrics found, use a default set
        if not metrics_to_plot:
            metrics_to_plot = ["loss", "perplexity"]
            
        # Create figure
        num_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5))
        
        # Handle single metric case
        if num_metrics == 1:
            axes = [axes]
        
        # For each metric, create a bar chart
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            # Extract values, handling missing metrics with NaN
            values = [
                float(baseline_metrics.get(metric, np.nan)),
                float(pruned_metrics.get(metric, np.nan)),
                float(final_metrics.get(metric, np.nan))
            ]
            
            # Labels for x-axis
            labels = ['Baseline', 'Post-Pruning', 'Post-Fine-Tuning']
            
            # Create bar chart
            bars = ax.bar(labels, values, color=['blue', 'red', 'green'], alpha=0.7)
            
            # Add title and labels
            ax.set_title(f'{metric.capitalize()}', fontsize=14)
            ax.tick_params(axis='x', labelrotation=25)
            
            # Calculate recovery percentage
            if np.isfinite(values[0]) and np.isfinite(values[1]) and np.isfinite(values[2]):
                drop = values[1] - values[0]  # Change from baseline to pruned
                recovery = values[1] - values[2]  # Change from pruned to final
                
                if abs(drop) > 1e-6:  # Avoid division by zero for very small changes
                    recovery_pct = (recovery / abs(drop)) * 100
                    
                    # For metrics where lower is better (like loss)
                    if metric in ["loss", "perplexity"]:
                        recovery_pct = -recovery_pct
                    
                    # Add recovery percentage text
                    ax.text(1.5, max(values) * 0.9, 
                          f'Recovery: {recovery_pct:.1f}%', 
                          ha='center', fontsize=10,
                          bbox=dict(facecolor='white', alpha=0.8))
            
            # Add value labels on top of bars
            for bar_idx, bar in enumerate(bars):
                height = bar.get_height()
                if np.isfinite(height):
                    # Format perplexity values for better readability
                    if metric == "perplexity":
                        # Use logarithmic scale value or text version if available
                        if bar_idx == 0 and "perplexity_text" in baseline_metrics:
                            label = baseline_metrics["perplexity_text"]
                        elif bar_idx == 1 and "perplexity_text" in pruned_metrics:
                            label = pruned_metrics["perplexity_text"]
                        elif bar_idx == 2 and "perplexity_text" in final_metrics:
                            label = final_metrics["perplexity_text"]
                        else:
                            # Format based on magnitude
                            if height > 1000000:
                                label = f'{height/1000000:.1f}M'
                            elif height > 1000:
                                label = f'{height/1000:.1f}K'
                            else:
                                label = f'{height:.1f}'
                    else:
                        # For other metrics like loss, show full precision
                        label = f'{height:.3f}'
                            
                    ax.text(bar.get_x() + bar.get_width()/2, height * 1.05,
                           label, ha='center', va='bottom', fontsize=9)
        
        # Adjust layout
        plt.tight_layout()
        
        # Add a note about logarithmic scale if applicable
        scale_note = None
        for metrics_dict in [baseline_metrics, pruned_metrics, final_metrics]:
            if "scale_note" in metrics_dict:
                scale_note = metrics_dict["scale_note"]
                break
                
        if scale_note:
            fig.text(0.5, 0.01, f"Note: {scale_note} used for better visualization", 
                    ha='center', fontsize=10, style='italic')
            fig.subplots_adjust(bottom=0.15)
        
        # Save figure
        filename = "recovery_analysis.png"
        save_path = os.path.join(viz_dir, filename)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        return save_path
    
    def visualize_regrown_heads(
        self,
        regrowth_data: Dict[Tuple[int, int], Dict[str, float]],
        num_layers: int,
        num_heads: int,
        experiment_id: str
    ) -> str:
        """
        Visualize which heads showed regrowth tendencies.
        
        Args:
            regrowth_data: Dict mapping (layer, head) to regrowth metrics
            num_layers: Number of layers in the model
            num_heads: Number of heads per layer
            experiment_id: Experiment identifier
            
        Returns:
            Path to saved visualization
        """
        # Create visualization directory
        viz_dir = os.path.join(self.output_dir, experiment_id, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create a mask of regrown heads and their regrowth ratios
        regrowth_mask = np.zeros((num_layers, num_heads))
        
        # Fill in the mask
        for (layer_idx, head_idx), data in regrowth_data.items():
            if 0 <= layer_idx < num_layers and 0 <= head_idx < num_heads:
                regrowth_mask[layer_idx, head_idx] = data.get('regrowth_ratio', 0)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Binary mask of regrown heads
        binary_mask = (regrowth_mask > 0).astype(float)
        axes[0].set_title("Regrown Attention Heads", fontsize=16, pad=20)
        binary_heatmap = axes[0].imshow(binary_mask, cmap='Greens', interpolation='nearest', aspect='auto')
        axes[0].set_xlabel("Head Index", fontsize=14, labelpad=10)
        axes[0].set_ylabel("Layer Index", fontsize=14, labelpad=10)
        axes[0].set_xticks(np.arange(num_heads))
        axes[0].set_yticks(np.arange(num_layers))
        
        # Add text labels for regrown heads
        for i in range(num_layers):
            for j in range(num_heads):
                if binary_mask[i, j] > 0:
                    axes[0].text(j, i, "R", ha="center", va="center", 
                               color="black", fontweight="bold", fontsize=12)
        
        # Add a descriptive label showing regrowth percentage
        regrown_count = len(regrowth_data)
        total_heads = num_layers * num_heads
        regrowth_percentage = (regrown_count / total_heads) * 100
        axes[0].text(0.5, -0.1, 
                   f"Regrown: {regrown_count}/{total_heads} heads ({regrowth_percentage:.1f}%)", 
                   transform=axes[0].transAxes, ha="center", fontsize=12)
        
        # Plot 2: Heatmap of regrowth ratios
        cmap = plt.cm.viridis
        cmap.set_bad('white', 1.0)
        masked_ratios = np.ma.masked_where(regrowth_mask <= 0, regrowth_mask)
        
        ratio_heatmap = axes[1].imshow(
            masked_ratios, cmap=cmap, interpolation='nearest', aspect='auto'
        )
        axes[1].set_title("Regrowth Magnitude", fontsize=16, pad=20)
        axes[1].set_xlabel("Head Index", fontsize=14, labelpad=10)
        axes[1].set_ylabel("Layer Index", fontsize=14, labelpad=10)
        axes[1].set_xticks(np.arange(num_heads))
        axes[1].set_yticks(np.arange(num_layers))
        
        # Add colorbar
        plt.colorbar(ratio_heatmap, ax=axes[1], label="Regrowth Ratio")
        
        # Add ratio labels for regrown heads if not too many
        if regrown_count <= 30:
            for (layer_idx, head_idx), data in regrowth_data.items():
                if 0 <= layer_idx < num_layers and 0 <= head_idx < num_heads:
                    ratio = data.get('regrowth_ratio', 0)
                    axes[1].text(head_idx, layer_idx, f"{ratio:.1f}", 
                               ha="center", va="center", color="black", 
                               fontsize=8, fontweight="bold")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        filename = "regrown_heads.png"
        save_path = os.path.join(viz_dir, filename)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        return save_path
        
    def _is_numeric(self, value: Any) -> bool:
        """
        Check if a value is numeric for visualization purposes.
        
        Args:
            value: Value to check
            
        Returns:
            True if numeric, False otherwise
        """
        try:
            float_val = float(value)
            return np.isfinite(float_val)
        except (ValueError, TypeError):
            return False