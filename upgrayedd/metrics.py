"""
Upgrayedd metrics module - Utilities for measuring and tracking model performance
"""

import os
import json
import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict

logger = logging.getLogger("Upgrayedd")

@dataclass
class MetricPoint:
    """A single metric measurement point."""
    phase: str
    timestamp: str
    cycle: Optional[int] = None
    perplexity: Optional[float] = None
    active_heads: Optional[int] = None
    total_heads: Optional[int] = None
    pruning_level: Optional[float] = None
    growth_ratio: Optional[float] = None
    head_reduction: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary, removing None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class MetricsTracker:
    """
    Tracks and logs metrics throughout the Upgrayedd process.
    
    This class handles:
    - Recording metrics at each stage of the process
    - Saving metrics to a JSONL file
    - Generating visualizations from metrics
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the metrics tracker.
        
        Args:
            output_dir: Directory to save metrics and visualizations
        """
        self.output_dir = output_dir
        self.metrics_dir = os.path.join(output_dir, "metrics")
        self.visualizations_dir = os.path.join(output_dir, "visualizations")
        
        # Create directories
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
        
        # Initialize metrics file
        self.metrics_file = os.path.join(self.metrics_dir, "integration_metrics.jsonl")
        
        # Clear metrics file if it exists
        with open(self.metrics_file, 'w') as f:
            pass
        
        # Initialize metrics list
        self.metrics: List[MetricPoint] = []
    
    def log_baseline(
        self,
        perplexity: float,
        active_heads: int,
        total_heads: int
    ) -> None:
        """
        Log baseline metrics.
        
        Args:
            perplexity: Baseline perplexity
            active_heads: Number of active heads
            total_heads: Total number of heads
        """
        metric = MetricPoint(
            phase="baseline",
            timestamp=datetime.now().isoformat(),
            perplexity=perplexity,
            active_heads=active_heads,
            total_heads=total_heads
        )
        
        self._log_metric(metric)
    
    def log_cycle(
        self,
        cycle: int,
        initial_perplexity: float,
        pruned_perplexity: float,
        grown_perplexity: float,
        final_perplexity: float,
        active_heads: int,
        total_heads: int,
        pruning_level: float,
        growth_ratio: float
    ) -> None:
        """
        Log metrics for a complete optimization cycle.
        
        Args:
            cycle: Cycle number
            initial_perplexity: Perplexity at the start of the cycle
            pruned_perplexity: Perplexity after pruning
            grown_perplexity: Perplexity after growth
            final_perplexity: Perplexity after fine-tuning
            active_heads: Number of active heads after the cycle
            total_heads: Total number of heads
            pruning_level: Pruning level applied
            growth_ratio: Growth ratio applied
        """
        # Calculate head reduction
        head_reduction = (total_heads - active_heads) / total_heads if total_heads > 0 else 0
        
        metric = MetricPoint(
            phase="cycle_complete",
            timestamp=datetime.now().isoformat(),
            cycle=cycle,
            perplexity=final_perplexity,  # For compatibility
            active_heads=active_heads,
            total_heads=total_heads,
            pruning_level=pruning_level,
            growth_ratio=growth_ratio,
            head_reduction=head_reduction
        )
        
        # Add extra fields manually (not in the dataclass)
        metric_dict = metric.to_dict()
        metric_dict["initial_perplexity"] = initial_perplexity
        metric_dict["pruned_perplexity"] = pruned_perplexity
        metric_dict["grown_perplexity"] = grown_perplexity
        metric_dict["final_perplexity"] = final_perplexity
        metric_dict["perplexity_improvement"] = (initial_perplexity - final_perplexity) / initial_perplexity if initial_perplexity > 0 else 0
        
        self._log_metric_dict(metric_dict)
    
    def log_pruning(
        self,
        cycle: int,
        initial_perplexity: float,
        pruned_perplexity: float,
        active_heads: int,
        total_heads: int,
        pruning_level: float
    ) -> None:
        """
        Log metrics after a pruning phase.
        
        Args:
            cycle: Cycle number
            initial_perplexity: Perplexity before pruning
            pruned_perplexity: Perplexity after pruning
            active_heads: Number of active heads after pruning
            total_heads: Total number of heads
            pruning_level: Pruning level applied
        """
        # Calculate head reduction
        head_reduction = (total_heads - active_heads) / total_heads if total_heads > 0 else 0
        
        metric = MetricPoint(
            phase="pruning",
            timestamp=datetime.now().isoformat(),
            cycle=cycle,
            perplexity=pruned_perplexity,
            active_heads=active_heads,
            total_heads=total_heads,
            pruning_level=pruning_level,
            head_reduction=head_reduction
        )
        
        # Add extra fields manually
        metric_dict = metric.to_dict()
        metric_dict["initial_perplexity"] = initial_perplexity
        metric_dict["pruned_perplexity"] = pruned_perplexity
        metric_dict["perplexity_change"] = (pruned_perplexity - initial_perplexity) / initial_perplexity if initial_perplexity > 0 else 0
        
        self._log_metric_dict(metric_dict)
    
    def log_growth(
        self,
        cycle: int,
        pruned_perplexity: float,
        grown_perplexity: float,
        active_heads: int,
        total_heads: int,
        growth_ratio: float
    ) -> None:
        """
        Log metrics after a growth phase.
        
        Args:
            cycle: Cycle number
            pruned_perplexity: Perplexity before growth
            grown_perplexity: Perplexity after growth
            active_heads: Number of active heads after growth
            total_heads: Total number of heads
            growth_ratio: Growth ratio applied
        """
        # Calculate head reduction
        head_reduction = (total_heads - active_heads) / total_heads if total_heads > 0 else 0
        
        metric = MetricPoint(
            phase="growth",
            timestamp=datetime.now().isoformat(),
            cycle=cycle,
            perplexity=grown_perplexity,
            active_heads=active_heads,
            total_heads=total_heads,
            growth_ratio=growth_ratio,
            head_reduction=head_reduction
        )
        
        # Add extra fields manually
        metric_dict = metric.to_dict()
        metric_dict["pruned_perplexity"] = pruned_perplexity
        metric_dict["grown_perplexity"] = grown_perplexity
        metric_dict["perplexity_change"] = (grown_perplexity - pruned_perplexity) / pruned_perplexity if pruned_perplexity > 0 else 0
        
        self._log_metric_dict(metric_dict)
    
    def log_fine_tuning(
        self,
        cycle: int,
        grown_perplexity: float,
        final_perplexity: float,
        active_heads: int,
        total_heads: int
    ) -> None:
        """
        Log metrics after a fine-tuning phase.
        
        Args:
            cycle: Cycle number
            grown_perplexity: Perplexity before fine-tuning
            final_perplexity: Perplexity after fine-tuning
            active_heads: Number of active heads
            total_heads: Total number of heads
        """
        # Calculate head reduction
        head_reduction = (total_heads - active_heads) / total_heads if total_heads > 0 else 0
        
        metric = MetricPoint(
            phase="fine_tuning",
            timestamp=datetime.now().isoformat(),
            cycle=cycle,
            perplexity=final_perplexity,
            active_heads=active_heads,
            total_heads=total_heads,
            head_reduction=head_reduction
        )
        
        # Add extra fields manually
        metric_dict = metric.to_dict()
        metric_dict["grown_perplexity"] = grown_perplexity
        metric_dict["final_perplexity"] = final_perplexity
        metric_dict["perplexity_change"] = (final_perplexity - grown_perplexity) / grown_perplexity if grown_perplexity > 0 else 0
        
        self._log_metric_dict(metric_dict)
    
    def _log_metric(self, metric: MetricPoint) -> None:
        """
        Log a metric point.
        
        Args:
            metric: Metric point to log
        """
        # Add to list
        self.metrics.append(metric)
        
        # Write to file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metric.to_dict()) + "\n")
    
    def _log_metric_dict(self, metric_dict: Dict[str, Any]) -> None:
        """
        Log a metric dictionary.
        
        Args:
            metric_dict: Metric dictionary to log
        """
        # Write to file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metric_dict) + "\n")
    
    def generate_visualizations(self) -> None:
        """Generate visualizations from the collected metrics."""
        # Load all metrics
        metrics = []
        with open(self.metrics_file, 'r') as f:
            for line in f:
                metrics.append(json.loads(line))
        
        # Extract baseline and cycle metrics
        baseline_metrics = [m for m in metrics if m.get('phase') == 'baseline']
        cycle_metrics = [m for m in metrics if m.get('phase') == 'cycle_complete']
        
        if not baseline_metrics or not cycle_metrics:
            logger.warning("Insufficient metrics for visualization")
            return
        
        # Set up visualization style
        sns.set_style("whitegrid")
        sns.set_palette("viridis")
        plt.rcParams.update({'font.size': 12, 'figure.figsize': (14, 8)})
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Perplexity Over Cycles
        cycles = [m['cycle'] for m in cycle_metrics]
        perplexities = [m.get('final_perplexity', 0) for m in cycle_metrics]
        active_heads = [m.get('active_heads', 0) for m in cycle_metrics]
        pruned_perplexities = [m.get('pruned_perplexity', 0) for m in cycle_metrics]
        grown_perplexities = [m.get('grown_perplexity', 0) for m in cycle_metrics]
        
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(cycles, perplexities, 'o-', linewidth=2, markersize=8, label='Final Perplexity')
        
        if all(p > 0 for p in pruned_perplexities) and all(p > 0 for p in grown_perplexities):
            ax1.plot(cycles, pruned_perplexities, 's--', alpha=0.7, label='After Pruning')
            ax1.plot(cycles, grown_perplexities, '^--', alpha=0.7, label='After Growth')
        
        ax1.set_title('Perplexity Across Optimization Cycles', fontsize=14)
        ax1.set_xlabel('Cycle', fontsize=12)
        ax1.set_ylabel('Perplexity', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Active Heads Over Cycles
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(cycles, active_heads, 'o-', color='green', linewidth=2, markersize=8)
        ax2.set_title('Active Attention Heads After Each Cycle', fontsize=14)
        ax2.set_xlabel('Cycle', fontsize=12)
        ax2.set_ylabel('Number of Active Heads', fontsize=12)
        
        # Add baseline head count as a horizontal line
        if baseline_metrics and 'active_heads' in baseline_metrics[0]:
            baseline_heads = baseline_metrics[0]['active_heads']
            ax2.axhline(y=baseline_heads, color='red', linestyle='--', alpha=0.7,
                       label=f'Baseline ({baseline_heads} heads)')
            ax2.legend()
        
        ax2.grid(True, alpha=0.3)
        
        # 3. Perplexity Changes Within Cycles
        if all(p > 0 for p in pruned_perplexities) and all(p > 0 for p in grown_perplexities):
            ax3 = plt.subplot(2, 2, 3)
            
            # Prepare data for grouped bar chart
            cycle_labels = [f'Cycle {c}' for c in cycles]
            x = np.arange(len(cycle_labels))
            width = 0.25
            
            # Plot bars for each phase
            initial_perplexities = [m.get('initial_perplexity', 0) for m in cycle_metrics]
            
            ax3.bar(x - width, initial_perplexities, width, label='Initial')
            ax3.bar(x, pruned_perplexities, width, label='After Pruning')
            ax3.bar(x + width, perplexities, width, label='Final')
            
            ax3.set_title('Perplexity Changes Within Each Cycle', fontsize=14)
            ax3.set_xlabel('Optimization Cycle', fontsize=12)
            ax3.set_ylabel('Perplexity', fontsize=12)
            ax3.set_xticks(x)
            ax3.set_xticklabels(cycle_labels)
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Head Reduction vs Perplexity Improvement
        if all(p > 0 for p in perplexities) and all(a > 0 for a in active_heads):
            ax4 = plt.subplot(2, 2, 4)
            
            # Calculate improvement percentages
            if baseline_metrics and 'perplexity' in baseline_metrics[0] and 'active_heads' in baseline_metrics[0]:
                baseline_perp = baseline_metrics[0]['perplexity']
                baseline_heads = baseline_metrics[0]['active_heads']
                
                perp_improvements = [(baseline_perp - p) / baseline_perp * 100 for p in perplexities]
                head_reductions = [(baseline_heads - a) / baseline_heads * 100 for a in active_heads]
                
                for i, cycle in enumerate(cycles):
                    ax4.annotate(f'Cycle {cycle}', (head_reductions[i], perp_improvements[i]),
                               xytext=(5, 5), textcoords='offset points')
                
                ax4.plot(head_reductions, perp_improvements, 'o-', linewidth=2, markersize=8, color='purple')
                ax4.set_title('Perplexity Improvement vs Head Reduction', fontsize=14)
                ax4.set_xlabel('Head Reduction (%)', fontsize=12)
                ax4.set_ylabel('Perplexity Improvement (%)', fontsize=12)
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, 'optimization_summary.png'), dpi=300)
        
        # Efficiency plot
        if baseline_metrics and 'active_heads' in baseline_metrics[0] and 'perplexity' in baseline_metrics[0]:
            baseline_perp = baseline_metrics[0]['perplexity']
            baseline_heads = baseline_metrics[0]['active_heads']
            
            # Calculate efficiency (perplexity per head)
            baseline_efficiency = baseline_perp / baseline_heads
            efficiencies = [p / a for p, a in zip(perplexities, active_heads)]
            
            plt.figure(figsize=(10, 6))
            plt.plot(cycles, efficiencies, 'o-', linewidth=2, markersize=8, color='teal')
            plt.axhline(y=baseline_efficiency, color='red', linestyle='--', alpha=0.7,
                       label=f'Baseline ({baseline_efficiency:.3f})')
            plt.title('Model Efficiency (Perplexity per Head)', fontsize=14)
            plt.xlabel('Cycle', fontsize=12)
            plt.ylabel('Efficiency (Lower is Better)', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.visualizations_dir, 'efficiency.png'), dpi=300)
        
        logger.info(f"Saved visualizations to {self.visualizations_dir}")


def calculate_perplexity(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: Union[str, torch.device]
) -> float:
    """
    Calculate perplexity of a model on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to use for evaluation
        
    Returns:
        float: Perplexity
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Shift tokens for causal language modeling
            labels = input_ids.clone()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Get loss
            loss = outputs.loss
            
            # Update totals
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += input_ids.size(0) * input_ids.size(1)
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return perplexity.item()