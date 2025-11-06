#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-Phase Dashboard for Neural Plasticity Visualization

This module implements a comprehensive dashboard for visualizing neural 
plasticity experiments across multiple phases and cycles, providing
detailed metrics tracking and visualizations.

Version: v0.1.1 (2025-04-20 23:15:00)
"""

import os
import time
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Set
from pathlib import Path
from datetime import datetime

# Import visualization components
from ..metrics_visualizer import MetricsVisualizer
from ..entropy_visualizer import EntropyVisualizer

logger = logging.getLogger(__name__)

class MultiPhaseDashboard:
    """
    Dashboard for visualizing multi-phase neural plasticity experiments.
    
    This dashboard provides comprehensive visualization of the entire
    neural plasticity process, including warmup, analysis, pruning,
    fine-tuning, and evaluation phases, with support for multiple cycles.
    """
    
    # Define constants for compatibility with WandbDashboard
    MODE_ONLINE = "online"
    MODE_OFFLINE = "offline"
    MODE_DISABLED = "disabled"
    
    # Bridging methods for compatibility with WandbDashboard interface
    def set_phase(self, phase):
        """Set current experiment phase (compatibility method)."""
        self.record_phase_transition(phase, self.current_step)
        
    def log_metrics(self, metrics, step=None):
        """Log metrics (compatibility method)."""
        self.record_step(metrics, step)
        
    def log_pruning_decision(self, pruning_info, step=None):
        """Log pruning decision (compatibility method)."""
        if step is None:
            step = self.current_step
        self.record_pruning_event(pruning_info, step)
    
    def log_text_sample(self, prompt, text, model_type="pruned"):
        """Log text sample (compatibility method)."""
        # Store text samples for later visualization
        if not hasattr(self, 'text_samples'):
            self.text_samples = []
        
        self.text_samples.append({
            "prompt": prompt,
            "output": text,
            "model_type": model_type,
            "step": self.current_step
        })
    
    def log_inference_comparison(self, prompt, baseline_text, pruned_text, metrics=None):
        """Log inference comparison (compatibility method)."""
        # Store comparison data for visualization
        if not hasattr(self, 'comparisons'):
            self.comparisons = []
            
        self.comparisons.append({
            "prompt": prompt,
            "baseline_text": baseline_text,
            "pruned_text": pruned_text,
            "metrics": metrics,
            "step": self.current_step
        })
    
    def log_inference_dashboard(self, perplexity_data=None, generation_samples=None, attention_data=None):
        """Log inference dashboard (compatibility method)."""
        # Store for final dashboard generation
        self.perplexity_data = perplexity_data
        self.generation_samples = generation_samples
        self.attention_data = attention_data
    
    def get_metrics_callback(self):
        """Return a metrics callback function (compatibility method)."""
        def metrics_callback(step=None, metrics=None):
            # Handle parameter order variations
            if isinstance(step, dict) and metrics is None:
                # Called as metrics_callback(metrics_dict)
                metrics = step
                step = None
            elif isinstance(metrics, int) and isinstance(step, dict):
                # Called as metrics_callback(metrics_dict, step)
                temp = step
                step = metrics
                metrics = temp
                
            self.record_step(metrics, step)
        return metrics_callback
    
    def get_sample_callback(self):
        """Return a sample callback function (compatibility method)."""
        def sample_callback(text, prompt=None, model_type="pruned"):
            self.log_text_sample(prompt, text, model_type)
        return sample_callback
    
    def finish(self):
        """Clean up resources (compatibility method)."""
        # No special cleanup needed for this implementation
        logger.info("Dashboard resources cleaned up.")
    
    def __init__(
        self,
        output_dir: str,
        experiment_name: str = "neural_plasticity",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the multi-phase dashboard.
        
        Args:
            output_dir: Directory to save dashboard files
            experiment_name: Name of the experiment
            config: Configuration parameters
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.config = config or {}
        
        # Phase tracking
        self.phases = []
        self.phase_transitions = []
        self.stabilization_points = []
        
        # Per-phase metrics
        self.warmup_metrics = {"steps": [], "loss": [], "perplexity": []}
        self.pruning_metrics = {"steps": [], "loss": [], "perplexity": []}
        self.finetuning_metrics = {"steps": [], "loss": [], "perplexity": []}
        
        # Global metrics
        self.all_steps = []
        self.all_loss = []
        self.all_perplexity = []
        self.all_sparsity = []
        
        # Evaluation metrics
        self.eval_steps = []
        self.eval_perplexity = []
        
        # Pruning information
        self.pruning_info = []
        self.current_step = 0
        self.current_phase = None
        
        # Controller integration metrics
        self.head_metrics = {}
        self.layer_metrics = {}
        self.controller_activations = []
        self.controller_decisions = []
        self.attention_entropy = {}
        self.attention_magnitude = {}
        
        # Head state tracking
        self.pruned_heads = set()
        self.active_heads = set()
        self.head_scores = {}
        self.head_recovery = {}
        
        # Initialize visualizers
        self.metrics_visualizer = MetricsVisualizer(self.output_dir)
        self.entropy_visualizer = EntropyVisualizer(self.output_dir)
        
        logger.info(f"Multi-phase dashboard initialized: {self.output_dir}")
    
    def record_step(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Record metrics for a training step.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number (uses internal counter if None)
        """
        if step is None:
            step = self.current_step
            self.current_step += 1
        else:
            self.current_step = step + 1
        
        # Capture current phase
        phase = metrics.get("phase", self.current_phase)
        if phase != self.current_phase:
            self.record_phase_transition(phase, step)
            self.current_phase = phase
        
        # Track global metrics
        self.all_steps.append(step)
        
        if "loss" in metrics:
            self.all_loss.append(metrics["loss"])
        
        if "perplexity" in metrics:
            self.all_perplexity.append(metrics["perplexity"])
        
        if "sparsity" in metrics:
            self.all_sparsity.append(metrics["sparsity"])
        
        # Track per-phase metrics
        if phase == "warmup":
            if "loss" in metrics:
                self.warmup_metrics["steps"].append(step)
                self.warmup_metrics["loss"].append(metrics["loss"])
            if "perplexity" in metrics:
                self.warmup_metrics["perplexity"].append(metrics["perplexity"])
        
        elif phase == "pruning":
            if "loss" in metrics:
                self.pruning_metrics["steps"].append(step)
                self.pruning_metrics["loss"].append(metrics["loss"])
            if "perplexity" in metrics:
                self.pruning_metrics["perplexity"].append(metrics["perplexity"])
        
        elif phase == "finetuning":
            if "loss" in metrics:
                self.finetuning_metrics["steps"].append(step)
                self.finetuning_metrics["loss"].append(metrics["loss"])
            if "perplexity" in metrics:
                self.finetuning_metrics["perplexity"].append(metrics["perplexity"])
        
        # Record evaluation metrics
        if "eval_perplexity" in metrics:
            self.eval_steps.append(step)
            self.eval_perplexity.append(metrics["eval_perplexity"])
        
        # Record stabilization point if indicated
        if metrics.get("stabilized", False):
            self.stabilization_points.append((step, phase))
    
    def record_phase_transition(self, phase: str, step: int):
        """
        Record a phase transition.
        
        Args:
            phase: New phase
            step: Step number of the transition
        """
        self.phases.append(phase)
        self.phase_transitions.append(step)
        self.current_phase = phase
        
        logger.info(f"Phase transition: {phase} at step {step}")
    
    def record_pruning_event(self, pruning_info: Dict[str, Any], step: int):
        """
        Record a pruning event.
        
        Args:
            pruning_info: Dictionary with pruning details
            step: Step number of the pruning event
        """
        pruning_info["step"] = step
        self.pruning_info.append(pruning_info)
        
        logger.info(f"Pruning event recorded at step {step}: {len(pruning_info.get('pruned_heads', []))} heads pruned")
        
        # Update pruned heads tracking
        if 'pruned_heads' in pruning_info:
            new_pruned_heads = set(pruning_info['pruned_heads'])
            self.pruned_heads.update(new_pruned_heads)
            
            # Remove pruned heads from active heads if they exist
            if hasattr(self, 'active_heads') and self.active_heads:
                self.active_heads = self.active_heads - new_pruned_heads
    
    def visualize_complete_process(self, save_path: Optional[str] = None):
        """
        Generate a comprehensive visualization of the complete training process.
        
        Args:
            save_path: Optional path to save the visualization
            
        Returns:
            The matplotlib figure
        """
        if not self.all_steps:
            logger.warning("No data to visualize.")
            return None
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 18))
        
        # Define grid layout
        gs = plt.GridSpec(4, 2, height_ratios=[3, 1.5, 1.5, 0.3])
        
        # 1. Complete training process (top spanning both columns)
        ax_main = fig.add_subplot(gs[0, :])
        self._plot_complete_training_process(ax_main)
        
        # 2. Perplexity over time (middle left)
        ax_perplexity = fig.add_subplot(gs[1, 0])
        self._plot_perplexity(ax_perplexity)
        
        # 3. Model sparsity (middle right)
        ax_sparsity = fig.add_subplot(gs[1, 1])
        self._plot_sparsity(ax_sparsity)
        
        # 4. Phase details (bottom row)
        ax_warmup = fig.add_subplot(gs[2, 0])
        self._plot_phase_detail(ax_warmup, "warmup")
        
        # Either pruning or fine-tuning detail based on what we have more data for
        if len(self.pruning_metrics["steps"]) > len(self.finetuning_metrics["steps"]):
            ax_phase = fig.add_subplot(gs[2, 1])
            self._plot_phase_detail(ax_phase, "pruning")
        else:
            ax_phase = fig.add_subplot(gs[2, 1])
            self._plot_phase_detail(ax_phase, "finetuning")
        
        # 5. Summary statistics (bottom spanning both columns)
        ax_summary = fig.add_subplot(gs[3, :])
        self._plot_summary_statistics(ax_summary)
        
        # Set title and adjust layout
        fig.suptitle("Neural Plasticity Complete Training Process", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
            logger.info(f"Visualization saved to {save_path}")
        
        return fig
    
    def _plot_complete_training_process(self, ax):
        """Plot the complete training process with phase highlighting."""
        # Plot the loss curve
        ax.plot(self.all_steps, self.all_loss, 'b-', linewidth=1.5, label="Training Loss")
        
        # Highlight phases with different colors
        if len(self.phase_transitions) > 0:
            # Add phase transitions
            for i, (phase, step) in enumerate(zip(self.phases, self.phase_transitions)):
                if i == 0:
                    # First phase starts from step 0
                    start_step = 0
                else:
                    start_step = self.phase_transitions[i-1]
                
                # End step is either the next phase transition or the last step
                if i < len(self.phase_transitions) - 1:
                    end_step = self.phase_transitions[i+1]
                else:
                    end_step = self.all_steps[-1] if len(self.all_steps) > 0 else start_step
                
                # Define colors for each phase
                colors = {
                    "warmup": "blue",
                    "pruning": "red",
                    "finetuning": "green",
                    "evaluation": "purple",
                    "setup": "gray",
                    "analysis": "orange",
                    "complete": "gray"
                }
                
                # Define alpha (transparency) for each phase
                alphas = {
                    "warmup": 0.2,
                    "pruning": 0.2,
                    "finetuning": 0.2,
                    "evaluation": 0.2,
                    "setup": 0.1,
                    "analysis": 0.2,
                    "complete": 0.1
                }
                
                # Highlight phase region
                color = colors.get(phase, "gray")
                alpha = alphas.get(phase, 0.1)
                
                # Find corresponding y-values for shading
                valid_steps = [s for s in self.all_steps if start_step <= s <= end_step]
                if valid_steps:
                    ax.axvspan(start_step, end_step, alpha=alpha, color=color)
                    # Add phase label
                    mid_step = (start_step + end_step) / 2
                    y_pos = max(self.all_loss) * 0.9  # Place near the top
                    ax.text(mid_step, y_pos, phase.capitalize(), 
                           ha='center', va='center', fontsize=10,
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Add stabilization points
        for step, phase in self.stabilization_points:
            ax.axvline(x=step, color='green', linestyle='--', alpha=0.7)
            ax.text(step, max(self.all_loss) * 0.8, "Stabilized",
                   ha='center', va='center', fontsize=8, rotation=90,
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Add pruning events
        for info in self.pruning_info:
            step = info.get("step", 0)
            heads = len(info.get("pruned_heads", []))
            ax.axvline(x=step, color='red', linestyle=':', alpha=0.7)
            ax.text(step, max(self.all_loss) * 0.7, f"Pruned {heads} heads",
                   ha='center', va='center', fontsize=8, rotation=90,
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Set labels and legend
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.set_title("Complete Training Process")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _plot_perplexity(self, ax):
        """Plot perplexity over time."""
        if not self.eval_perplexity:
            # If no evaluation perplexity, use training perplexity
            if self.all_perplexity:
                ax.plot(self.all_steps, self.all_perplexity, 'purple', linewidth=1.5)
        else:
            ax.plot(self.eval_steps, self.eval_perplexity, 'purple', linewidth=1.5)
        
        ax.set_xlabel("Evaluation Steps")
        ax.set_ylabel("Perplexity")
        ax.set_title("Model Perplexity")
        ax.grid(True, alpha=0.3)
    
    def _plot_sparsity(self, ax):
        """Plot model sparsity over time."""
        if not self.all_sparsity:
            return
        
        # Convert to percentage if in decimal form
        sparsity = [s * 100 if s <= 1.0 else s for s in self.all_sparsity]
        
        # Find steps where sparsity changes
        steps = []
        unique_sparsity = []
        
        for i, (step, s) in enumerate(zip(self.all_steps, sparsity)):
            if i == 0 or s != unique_sparsity[-1]:
                steps.append(step)
                unique_sparsity.append(s)
        
        # Plot as a step function
        ax.step(steps, unique_sparsity, 'r-', linewidth=1.5, where='post')
        
        ax.set_xlabel("Pruning Steps")
        ax.set_ylabel("Sparsity (%)")
        ax.set_title("Model Sparsity")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(sparsity) * 1.1)
    
    def _plot_phase_detail(self, ax, phase):
        """Plot detailed view of a specific phase."""
        if phase == "warmup" and self.warmup_metrics["steps"]:
            steps = self.warmup_metrics["steps"]
            loss = self.warmup_metrics["loss"]
            ax.plot(steps, loss, 'b-', linewidth=1.5)
            
            # Add stabilization points
            for step, p in self.stabilization_points:
                if p == "warmup":
                    ax.axvline(x=step, color='green', linestyle='--', alpha=0.7)
            
            title = "Warmup Phase Detail"
            color = 'blue'
            
        elif phase == "pruning" and self.pruning_metrics["steps"]:
            steps = self.pruning_metrics["steps"]
            loss = self.pruning_metrics["loss"]
            ax.plot(steps, loss, 'r-', linewidth=1.5)
            
            # Add pruning events
            for info in self.pruning_info:
                step = info.get("step", 0)
                if step in steps:
                    ax.axvline(x=step, color='red', linestyle=':', alpha=0.7)
            
            title = "Pruning Phase Detail"
            color = 'red'
            
        elif phase == "finetuning" and self.finetuning_metrics["steps"]:
            steps = self.finetuning_metrics["steps"]
            loss = self.finetuning_metrics["loss"]
            ax.plot(steps, loss, 'g-', linewidth=1.5)
            
            title = "Fine-tuning Phase Detail"
            color = 'green'
        
        else:
            ax.text(0.5, 0.5, f"No {phase} data available", 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f"{phase.capitalize()} Phase Detail")
            return
        
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    def _plot_summary_statistics(self, ax):
        """Plot summary statistics as a text box."""
        # Calculate summary statistics
        total_steps = self.all_steps[-1] if self.all_steps else 0
        
        # Phase steps
        warmup_steps = len(self.warmup_metrics["steps"])
        pruning_steps = len(self.pruning_metrics["steps"])
        finetuning_steps = len(self.finetuning_metrics["steps"])
        
        # Stabilization points
        warmup_stab = "N/A"
        for step, phase in self.stabilization_points:
            if phase == "warmup":
                warmup_stab = step
                break
        
        # Final sparsity
        final_sparsity = self.all_sparsity[-1] if self.all_sparsity else 0
        if final_sparsity <= 1.0:
            final_sparsity *= 100  # Convert to percentage
        
        # Pruned heads
        total_pruned_heads = 0
        for info in self.pruning_info:
            total_pruned_heads += len(info.get("pruned_heads", []))
        
        # Perplexity improvement
        if self.eval_perplexity and len(self.eval_perplexity) > 1:
            initial_perplexity = self.eval_perplexity[0]
            final_perplexity = self.eval_perplexity[-1]
            improvement = ((initial_perplexity - final_perplexity) / initial_perplexity) * 100
            perplexity_text = f"Perplexity improvement: {improvement:.1f}%"
        else:
            perplexity_text = "Perplexity improvement: N/A"
        
        # Create summary text
        summary = f"Warmup: {warmup_steps} steps | Pruning: {pruning_steps} steps | Fine-tuning: {finetuning_steps} steps | "
        summary += f"Stabilization at step {warmup_stab} | Final sparsity: {final_sparsity:.1f}% | "
        summary += f"Total pruned heads: {total_pruned_heads} | {perplexity_text}"
        
        # Remove axis elements
        ax.axis('off')
        
        # Add text box
        ax.text(0.5, 0.5, summary, ha='center', va='center', 
               bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round'),
               wrap=True, fontsize=10)
        
        # Add quote
        quote = '"In the neural world, connections that fire together wire together, and those that don\'t, don\'t."'
        ax.text(0.01, 0.01, quote, ha='left', va='bottom', fontsize=8, 
               style='italic', transform=ax.transAxes, alpha=0.7)
    
    def generate_multi_cycle_dashboard(self, save_path: Optional[str] = None):
        """
        Generate a dashboard for an experiment with multiple pruning cycles.
        
        Args:
            save_path: Optional path to save the visualization
            
        Returns:
            The matplotlib figure
        """
        if not self.all_steps:
            logger.warning("No data to visualize.")
            return None
        
        # Extract cycle info from pruning events
        cycles = []
        for info in self.pruning_info:
            if 'cycle' in info and info['cycle'] not in cycles:
                cycles.append(info['cycle'])
        
        if not cycles:
            logger.warning("No cycle information found in pruning events.")
            return None
        
        # Create figure
        fig = plt.figure(figsize=(15, 10 + len(cycles) * 3))
        
        # Define grid layout
        gs = plt.GridSpec(2 + len(cycles), 2, height_ratios=[3] + [1.5] * len(cycles) + [1])
        
        # 1. Complete training process (top spanning both columns)
        ax_main = fig.add_subplot(gs[0, :])
        self._plot_complete_training_process(ax_main)
        
        # 2. Cycle-specific plots
        # Find cycle boundaries from pruning events
        cycle_boundaries = {}
        for info in self.pruning_info:
            if 'cycle' in info and 'step' in info:
                cycle = info['cycle']
                step = info['step']
                if cycle not in cycle_boundaries:
                    cycle_boundaries[cycle] = {'start': step, 'end': None}
                # Each pruning event marks the end of previous cycle (if any)
                for c in cycle_boundaries:
                    if c < cycle and cycle_boundaries[c]['end'] is None:
                        cycle_boundaries[c]['end'] = step
        
        # Set end of last cycle to the last step
        if cycles and max(cycles) in cycle_boundaries and cycle_boundaries[max(cycles)]['end'] is None:
            cycle_boundaries[max(cycles)]['end'] = self.all_steps[-1] if self.all_steps else 0
        
        # Plot each cycle
        for i, cycle in enumerate(sorted(cycles)):
            if cycle in cycle_boundaries:
                start = cycle_boundaries[cycle]['start']
                end = cycle_boundaries[cycle]['end']
                
                # Filter metrics for this cycle
                cycle_steps = []
                cycle_loss = []
                cycle_perplexity = []
                cycle_sparsity = []
                
                for j, step in enumerate(self.all_steps):
                    if start <= step <= (end or float('inf')):
                        cycle_steps.append(step)
                        if j < len(self.all_loss):
                            cycle_loss.append(self.all_loss[j])
                        if j < len(self.all_perplexity):
                            cycle_perplexity.append(self.all_perplexity[j])
                        if j < len(self.all_sparsity):
                            cycle_sparsity.append(self.all_sparsity[j])
                
                # Create left plot (loss)
                ax_left = fig.add_subplot(gs[i+1, 0])
                if cycle_steps and cycle_loss:
                    ax_left.plot(cycle_steps, cycle_loss, 'b-', linewidth=1.5, label=f"Loss (Cycle {cycle})")
                    ax_left.set_title(f"Cycle {cycle} - Loss")
                    ax_left.set_xlabel("Steps")
                    ax_left.set_ylabel("Loss")
                    ax_left.grid(True, alpha=0.3)
                else:
                    ax_left.text(0.5, 0.5, f"No data for Cycle {cycle}", ha='center', va='center', fontsize=12)
                    ax_left.set_title(f"Cycle {cycle}")
                
                # Create right plot (perplexity or sparsity)
                ax_right = fig.add_subplot(gs[i+1, 1])
                if cycle_steps:
                    if cycle_perplexity:
                        ax_right.plot(cycle_steps, cycle_perplexity, 'purple', linewidth=1.5, label=f"Perplexity (Cycle {cycle})")
                        ax_right.set_title(f"Cycle {cycle} - Perplexity")
                        ax_right.set_ylabel("Perplexity")
                    elif cycle_sparsity:
                        # Convert to percentage if in decimal form
                        sparsity = [s * 100 if s <= 1.0 else s for s in cycle_sparsity]
                        ax_right.step(cycle_steps, sparsity, 'r-', linewidth=1.5, where='post', label=f"Sparsity (Cycle {cycle})")
                        ax_right.set_title(f"Cycle {cycle} - Sparsity")
                        ax_right.set_ylabel("Sparsity (%)")
                    
                    ax_right.set_xlabel("Steps")
                    ax_right.grid(True, alpha=0.3)
                else:
                    ax_right.text(0.5, 0.5, f"No data for Cycle {cycle}", ha='center', va='center', fontsize=12)
                    ax_right.set_title(f"Cycle {cycle}")
        
        # 3. Summary statistics (bottom spanning both columns)
        ax_summary = fig.add_subplot(gs[-1, :])
        
        # Calculate per-cycle statistics
        cycle_stats = []
        for cycle in sorted(cycles):
            if cycle in cycle_boundaries:
                start = cycle_boundaries[cycle]['start']
                end = cycle_boundaries[cycle]['end']
                
                # Find pruning events in this cycle
                heads_pruned = 0
                for info in self.pruning_info:
                    if info.get('cycle') == cycle:
                        heads_pruned += len(info.get('pruned_heads', []))
                
                # Find performance change in this cycle
                cycle_perf_change = "N/A"
                if self.eval_steps and self.eval_perplexity:
                    start_perp = None
                    end_perp = None
                    for eval_step, perp in zip(self.eval_steps, self.eval_perplexity):
                        if eval_step >= start and start_perp is None:
                            start_perp = perp
                        if eval_step <= (end or float('inf')):
                            end_perp = perp
                    
                    if start_perp is not None and end_perp is not None:
                        change = ((start_perp - end_perp) / start_perp) * 100
                        cycle_perf_change = f"{change:.1f}%"
                
                # Add cycle summary
                cycle_stats.append(f"Cycle {cycle}: {heads_pruned} heads pruned, performance change: {cycle_perf_change}")
        
        # Create summary text
        cycle_summary = " | ".join(cycle_stats)
        summary_text = f"Multi-Cycle Pruning Experiment\n{cycle_summary}"
        
        # Remove axis elements
        ax_summary.axis('off')
        
        # Add text box
        ax_summary.text(0.5, 0.5, summary_text, ha='center', va='center', 
                       bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round'),
                       wrap=True, fontsize=10)
        
        # Set title and adjust layout
        fig.suptitle("Neural Plasticity Multi-Cycle Training Process", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
            logger.info(f"Multi-cycle visualization saved to {save_path}")
        
        return fig
    
    def generate_standalone_dashboard(self, output_dir: Optional[str] = None) -> str:
        """
        Generate a standalone HTML dashboard with all visualizations.
        
        Args:
            output_dir: Directory to save dashboard files (defaults to self.output_dir)
            
        Returns:
            Path to the generated HTML dashboard
        """
        # Check for text samples and model comparisons to include in dashboard
        has_text_samples = hasattr(self, 'text_samples') and self.text_samples
        has_comparisons = hasattr(self, 'comparisons') and self.comparisons
        has_perf_data = hasattr(self, 'perplexity_data') and self.perplexity_data
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "dashboard")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations
        complete_process_path = os.path.join(output_dir, "complete_process.png")
        self.visualize_complete_process(complete_process_path)
        
        # Extract cycle information
        cycles = []
        for info in self.pruning_info:
            if 'cycle' in info and info['cycle'] not in cycles:
                cycles.append(info['cycle'])
        
        # Generate multi-cycle visualization if applicable
        multi_cycle_path = None
        if len(cycles) > 1:
            multi_cycle_path = os.path.join(output_dir, "multi_cycle_process.png")
            self.generate_multi_cycle_dashboard(multi_cycle_path)
        
        # Generate controller and head metrics visualizations if applicable
        controller_dashboard_path = None
        head_metrics_path = None
        
        if self.controller_decisions or self.head_metrics:
            # Controller dashboard
            if self.controller_decisions:
                controller_dashboard_path = os.path.join(output_dir, "controller_dashboard.png")
                self.generate_controller_dashboard(controller_dashboard_path)
            
            # Head metrics visualization
            if self.head_metrics:
                head_metrics_path = os.path.join(output_dir, "head_metrics.png")
                self.visualize_head_metrics(head_metrics_path)
        
        # Calculate summary statistics for the HTML
        total_steps = self.all_steps[-1] if self.all_steps else 0
        warmup_steps = len(self.warmup_metrics["steps"])
        pruning_steps = len(self.pruning_metrics["steps"])
        finetuning_steps = len(self.finetuning_metrics["steps"])
        
        # Get stabilization step
        warmup_stab = "N/A"
        for step, phase in self.stabilization_points:
            if phase == "warmup":
                warmup_stab = step
                break
        
        # Final sparsity
        final_sparsity = self.all_sparsity[-1] if self.all_sparsity else 0
        if final_sparsity <= 1.0:
            final_sparsity *= 100  # Convert to percentage
        
        # Pruned heads
        total_pruned_heads = 0
        for info in self.pruning_info:
            total_pruned_heads += len(info.get("pruned_heads", []))
        
        # Controller metrics if available
        controller_active_heads = 0
        if self.controller_decisions and len(self.controller_decisions) > 0:
            latest_decision = self.controller_decisions[-1]
            if "active_heads" in latest_decision:
                controller_active_heads = len(latest_decision["active_heads"])
        
        # Save dashboard data for later analysis
        self.save_dashboard_data(os.path.join(output_dir, "data"))
        
        # Create HTML dashboard
        html_path = os.path.join(output_dir, "dashboard.html")
        
        with open(html_path, 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Neural Plasticity Multi-Phase Dashboard</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                    h1, h2, h3 {{ color: #3f51b5; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .dashboard-section {{ 
                        background-color: white; 
                        border-radius: 8px; 
                        padding: 20px; 
                        margin: 20px 0; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .metrics {{ 
                        display: flex; 
                        flex-wrap: wrap; 
                        gap: 15px; 
                        margin: 15px 0;
                    }}
                    .metric-card {{
                        flex: 1;
                        min-width: 200px;
                        background-color: white;
                        padding: 15px;
                        border-radius: 5px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                        border-left: 5px solid #3f51b5;
                    }}
                    .metric-card.pruning {{
                        border-left-color: #e74c3c;
                    }}
                    .metric-card.controller {{
                        border-left-color: #2ecc71;
                    }}
                    .metric-title {{
                        font-size: 14px;
                        color: #7f8c8d;
                        margin-bottom: 5px;
                    }}
                    .metric-value {{
                        font-size: 24px;
                        font-weight: bold;
                        color: #2c3e50;
                    }}
                    img {{
                        max-width: 100%;
                        height: auto;
                        border-radius: 5px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        margin: 10px 0;
                    }}
                    .summary-box {{
                        background-color: #f1f1f1;
                        border-radius: 5px;
                        padding: 15px;
                        margin: 15px 0;
                        font-size: 14px;
                        border-left: 5px solid #3f51b5;
                    }}
                    .quote {{
                        font-style: italic;
                        color: #7f8c8d;
                        text-align: center;
                        margin: 30px 0;
                    }}
                    .tabs {{
                        display: flex;
                        margin-bottom: 20px;
                    }}
                    .tab {{
                        padding: 10px 20px;
                        cursor: pointer;
                        background-color: #f1f1f1;
                        border-radius: 5px 5px 0 0;
                        margin-right: 5px;
                    }}
                    .tab.active {{
                        background-color: white;
                        font-weight: bold;
                    }}
                    .tab-content {{
                        display: none;
                    }}
                    .tab-content.active {{
                        display: block;
                    }}
                </style>
                <script>
                    function openTab(evt, tabName) {{
                        var i, tabcontent, tablinks;
                        tabcontent = document.getElementsByClassName("tab-content");
                        for (i = 0; i < tabcontent.length; i++) {{
                            tabcontent[i].style.display = "none";
                        }}
                        tablinks = document.getElementsByClassName("tab");
                        for (i = 0; i < tablinks.length; i++) {{
                            tablinks[i].className = tablinks[i].className.replace(" active", "");
                        }}
                        document.getElementById(tabName).style.display = "block";
                        evt.currentTarget.className += " active";
                    }}
                    
                    // Initialize the first tab as active when page loads
                    document.addEventListener('DOMContentLoaded', function() {{
                        document.getElementsByClassName('tab')[0].click();
                    }});
                </script>
            </head>
            <body>
                <div class="container">
                    <h1>Neural Plasticity Multi-Phase Dashboard</h1>
                    <p>Experiment: {self.experiment_name}</p>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <!-- Summary Metrics -->
                    <div class="dashboard-section">
                        <h2>Experiment Summary</h2>
                        <div class="metrics">
                            <div class="metric-card">
                                <div class="metric-title">Warmup Steps</div>
                                <div class="metric-value">{warmup_steps}</div>
                            </div>
                            <div class="metric-card pruning">
                                <div class="metric-title">Pruning Steps</div>
                                <div class="metric-value">{pruning_steps}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-title">Fine-tuning Steps</div>
                                <div class="metric-value">{finetuning_steps}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-title">Stabilization Step</div>
                                <div class="metric-value">{warmup_stab}</div>
                            </div>
                            <div class="metric-card pruning">
                                <div class="metric-title">Final Sparsity</div>
                                <div class="metric-value">{final_sparsity:.1f}%</div>
                            </div>
                            <div class="metric-card pruning">
                                <div class="metric-title">Total Pruned Heads</div>
                                <div class="metric-value">{total_pruned_heads}</div>
                            </div>
                            {f'''
                            <div class="metric-card controller">
                                <div class="metric-title">Active Heads (Controller)</div>
                                <div class="metric-value">{controller_active_heads}</div>
                            </div>
                            ''' if controller_active_heads > 0 else ''}
                        </div>
                    </div>
                    
                    <!-- Tabs for different visualizations -->
                    <div class="dashboard-section">
                        <div class="tabs">
                            <div class="tab active" onclick="openTab(event, 'tab-process')">Process Overview</div>
                            {f'''<div class="tab" onclick="openTab(event, 'tab-cycles')">Pruning Cycles</div>''' if multi_cycle_path else ''}
                            {f'''<div class="tab" onclick="openTab(event, 'tab-controller')">Controller Metrics</div>''' if controller_dashboard_path else ''}
                            {f'''<div class="tab" onclick="openTab(event, 'tab-heads')">Head Metrics</div>''' if head_metrics_path else ''}
                            {f'''<div class="tab" onclick="openTab(event, 'tab-text-samples')">Text Samples</div>''' if has_comparisons else ''}
                        </div>
                        
                        <!-- Tab content for Process Overview -->
                        <div id="tab-process" class="tab-content active">
                            <h3>Complete Process Visualization</h3>
                            <p>This visualization shows the entire neural plasticity process including warmup, pruning, and fine-tuning phases.</p>
                            <img src="complete_process.png" alt="Complete Process Visualization">
                            
                            <div class="summary-box">
                                <strong>Process Summary:</strong> This experiment involved {warmup_steps} warmup steps, 
                                {pruning_steps} pruning steps, and {finetuning_steps} fine-tuning steps across 
                                {len(cycles) if cycles else 1} pruning cycle(s).
                                The training stabilized at step {warmup_stab} and achieved a final model sparsity of 
                                {final_sparsity:.1f}% by pruning {total_pruned_heads} attention heads.
                            </div>
                        </div>
                        
                        {f'''
                        <!-- Tab content for Pruning Cycles -->
                        <div id="tab-cycles" class="tab-content">
                            <h3>Multi-Cycle Analysis</h3>
                            <p>This visualization shows the detailed breakdown of each pruning cycle.</p>
                            <img src="multi_cycle_process.png" alt="Multi-Cycle Analysis">
                            
                            <div class="summary-box">
                                <strong>Cycle Analysis:</strong> The experiment ran for {len(cycles)} pruning cycles,
                                with each cycle focusing on identifying and removing less important attention heads.
                                The progressive pruning approach allows for gradual adaptation of the model.
                            </div>
                        </div>
                        ''' if multi_cycle_path else ''}
                        
                        {f'''
                        <!-- Tab content for Controller Metrics -->
                        <div id="tab-controller" class="tab-content">
                            <h3>Controller Integration Dashboard</h3>
                            <p>This visualization shows the ANN controller metrics and decisions throughout the experiment.</p>
                            <img src="controller_dashboard.png" alt="Controller Dashboard">
                            
                            <div class="summary-box">
                                <strong>Controller Summary:</strong> The Adaptive Neural Network (ANN) controller actively 
                                managed attention head gating throughout the experiment, dynamically adjusting which heads 
                                are active based on importance metrics.
                                Currently {controller_active_heads} heads are actively managed by the controller.
                            </div>
                        </div>
                        ''' if controller_dashboard_path else ''}
                        
                        {f'''
                        <!-- Tab content for Text Samples -->
                        <div id="tab-text-samples" class="tab-content">
                            <h3>Generated Text Samples</h3>
                            <p>This section shows text generated by the model before and after pruning.</p>
                            
                            <div class="dashboard-section">
                                <h3>Model Output Comparison</h3>
                                
                                {self._generate_sample_html() if hasattr(self, 'comparisons') and self.comparisons else ''}
                            </div>
                            
                            <div class="summary-box">
                                <strong>Text Generation Summary:</strong> These samples demonstrate the model's generation 
                                capabilities before and after the pruning process. Comparing the outputs helps evaluate 
                                whether the pruned model maintains similar linguistic abilities while using fewer resources.
                            </div>
                        </div>
                        ''' if has_comparisons else ''}
                        
                        {f'''
                        <!-- Tab content for Head Metrics -->
                        <div id="tab-heads" class="tab-content">
                            <h3>Attention Head Metrics Analysis</h3>
                            <p>This visualization shows detailed metrics for individual attention heads, including entropy, 
                            magnitude, and importance scores.</p>
                            <img src="head_metrics.png" alt="Head Metrics Visualization">
                            
                            <div class="summary-box">
                                <strong>Head Metrics Summary:</strong> Analysis of individual attention head metrics provides
                                insights into which heads are most important for the model's performance.
                                The visualizations show entropy distribution, importance scores, and recovery patterns for
                                attention heads throughout the experiment.
                            </div>
                        </div>
                        ''' if head_metrics_path else ''}
                    </div>
                    
                    <div class="dashboard-section">
                        <h2>Experiment Results</h2>
                        
                        <div class="metrics">
                            <div class="metric-card">
                                <div class="metric-title">Total Steps</div>
                                <div class="metric-value">{total_steps}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-title">Pruning Cycles</div>
                                <div class="metric-value">{len(cycles) if cycles else 1}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-title">Final Sparsity</div>
                                <div class="metric-value">{final_sparsity:.1f}%</div>
                            </div>
                        </div>
                        
                        <div class="summary-box">
                            <strong>Conclusion:</strong> The neural plasticity experiment demonstrates that transformer models
                            can be effectively pruned and fine-tuned for better efficiency while maintaining or improving performance.
                            {f"The integration with the adaptive controller provides dynamic management of attention heads based on their importance." 
                              if controller_dashboard_path else ""}
                        </div>
                    </div>
                    
                    <div class="quote">
                        "In the neural world, connections that fire together wire together, and those that don't, don't."
                    </div>
                    
                    <p style="text-align: center; font-size: 12px; color: #7f8c8d;">
                        Generated with Neural Plasticity Dashboard | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    </p>
                </div>
            </body>
            </html>
            """)
        
        logger.info(f"Standalone dashboard saved to: {html_path}")
        return html_path
        
    def record_head_metrics(self, head_metrics: Dict[str, Any], step: int):
        """
        Record metrics for individual attention heads.
        
        Args:
            head_metrics: Dictionary mapping head identifiers to metrics
            step: Current step number
        """
        for head_id, metrics in head_metrics.items():
            if head_id not in self.head_metrics:
                self.head_metrics[head_id] = {"steps": [], "metrics": {}}
            
            self.head_metrics[head_id]["steps"].append(step)
            
            for metric_name, value in metrics.items():
                if metric_name not in self.head_metrics[head_id]["metrics"]:
                    self.head_metrics[head_id]["metrics"][metric_name] = []
                
                self.head_metrics[head_id]["metrics"][metric_name].append(value)
        
        logger.debug(f"Recorded head metrics for {len(head_metrics)} heads at step {step}")
    
    def record_controller_decision(self, decision: Dict[str, Any], step: int):
        """
        Record a decision made by the ANN controller.
        
        Args:
            decision: Decision information from the controller
            step: Current step number
        """
        decision["step"] = step
        self.controller_decisions.append(decision)
        
        # Update active heads based on controller decision
        if "active_heads" in decision:
            self.active_heads = set(decision["active_heads"])
        
        logger.debug(f"Recorded controller decision at step {step}")
    
    def record_attention_entropy(self, entropy_data: Dict[str, np.ndarray], step: int):
        """
        Record attention entropy data for visualization.
        
        Args:
            entropy_data: Dictionary mapping head identifiers to entropy values
            step: Current step number
        """
        # Store a snapshot of entropy data at this step
        self.attention_entropy[step] = entropy_data
        
        # Update head scores with entropy values
        for head_id, entropy in entropy_data.items():
            if isinstance(entropy, np.ndarray):
                entropy_value = float(entropy.mean())
            else:
                entropy_value = float(entropy)
                
            if head_id not in self.head_scores:
                self.head_scores[head_id] = {"steps": [], "entropy": [], "magnitude": [], "importance": []}
            
            self.head_scores[head_id]["steps"].append(step)
            self.head_scores[head_id]["entropy"].append(entropy_value)
        
        logger.debug(f"Recorded attention entropy for {len(entropy_data)} heads at step {step}")
    
    def record_attention_magnitude(self, magnitude_data: Dict[str, np.ndarray], step: int):
        """
        Record attention magnitude data for visualization.
        
        Args:
            magnitude_data: Dictionary mapping head identifiers to magnitude values
            step: Current step number
        """
        # Store a snapshot of magnitude data at this step
        self.attention_magnitude[step] = magnitude_data
        
        # Update head scores with magnitude values
        for head_id, magnitude in magnitude_data.items():
            if isinstance(magnitude, np.ndarray):
                magnitude_value = float(magnitude.mean())
            else:
                magnitude_value = float(magnitude)
                
            if head_id not in self.head_scores:
                self.head_scores[head_id] = {"steps": [], "entropy": [], "magnitude": [], "importance": []}
            
            if "steps" in self.head_scores[head_id] and step not in self.head_scores[head_id]["steps"]:
                self.head_scores[head_id]["steps"].append(step)
            
            self.head_scores[head_id]["magnitude"].append(magnitude_value)
        
        logger.debug(f"Recorded attention magnitude for {len(magnitude_data)} heads at step {step}")
    
    def record_head_importance(self, importance_data: Dict[str, float], step: int):
        """
        Record head importance scores (combined metric).
        
        Args:
            importance_data: Dictionary mapping head identifiers to importance scores
            step: Current step number
        """
        # Update head scores with importance values
        for head_id, importance in importance_data.items():
            if head_id not in self.head_scores:
                self.head_scores[head_id] = {"steps": [], "entropy": [], "magnitude": [], "importance": []}
            
            if "steps" in self.head_scores[head_id] and step not in self.head_scores[head_id]["steps"]:
                self.head_scores[head_id]["steps"].append(step)
            
            self.head_scores[head_id]["importance"].append(float(importance))
        
        logger.debug(f"Recorded head importance for {len(importance_data)} heads at step {step}")
    
    def record_head_recovery(self, recovery_data: Dict[str, float], step: int):
        """
        Record head recovery information from controller-guided growth.
        
        Args:
            recovery_data: Dictionary mapping head identifiers to recovery metrics
            step: Current step number
        """
        for head_id, recovery_value in recovery_data.items():
            if head_id not in self.head_recovery:
                self.head_recovery[head_id] = {"steps": [], "recovery_score": []}
            
            self.head_recovery[head_id]["steps"].append(step)
            self.head_recovery[head_id]["recovery_score"].append(float(recovery_value))
        
        logger.debug(f"Recorded head recovery for {len(recovery_data)} heads at step {step}")
        
    def visualize_head_metrics(self, save_path: Optional[str] = None):
        """
        Generate visualizations of head-level metrics.
        
        Args:
            save_path: Optional path to save the visualization
            
        Returns:
            The matplotlib figure
        """
        if not self.head_metrics:
            logger.warning("No head metrics data to visualize.")
            return None
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 18))
        
        # Define grid layout
        gs = plt.GridSpec(3, 2)
        
        # 1. Entropy distribution over time
        ax_entropy = fig.add_subplot(gs[0, 0])
        self._plot_head_metric_distribution(ax_entropy, "entropy")
        
        # 2. Magnitude distribution over time
        ax_magnitude = fig.add_subplot(gs[0, 1])
        self._plot_head_metric_distribution(ax_magnitude, "magnitude")
        
        # 3. Top and bottom importance heads
        ax_top = fig.add_subplot(gs[1, 0])
        self._plot_top_bottom_heads(ax_top, "importance", top=True)
        
        ax_bottom = fig.add_subplot(gs[1, 1])
        self._plot_top_bottom_heads(ax_bottom, "importance", top=False)
        
        # 4. Head recovery visualization
        ax_recovery = fig.add_subplot(gs[2, :])
        self._plot_head_recovery(ax_recovery)
        
        # Set title and adjust layout
        fig.suptitle("Attention Head Metrics Analysis", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
            logger.info(f"Head metrics visualization saved to {save_path}")
        
        return fig
    
    def _plot_head_metric_distribution(self, ax, metric_name):
        """Plot the distribution of a metric across heads over time."""
        valid_steps = sorted(list(set([step for head_id in self.head_scores 
                                    for step in self.head_scores[head_id]["steps"]])))
        
        if not valid_steps:
            ax.text(0.5, 0.5, f"No {metric_name} data available", 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f"{metric_name.capitalize()} Distribution")
            return
        
        # Collect data for each step
        step_data = []
        for step in valid_steps:
            step_values = []
            for head_id in self.head_scores:
                score_data = self.head_scores[head_id]
                if metric_name in score_data and "steps" in score_data:
                    step_idx = None
                    try:
                        step_idx = score_data["steps"].index(step)
                    except ValueError:
                        continue
                    
                    if step_idx is not None and step_idx < len(score_data[metric_name]):
                        step_values.append(score_data[metric_name][step_idx])
            
            if step_values:
                step_data.append(step_values)
        
        if not step_data:
            ax.text(0.5, 0.5, f"No {metric_name} data available", 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f"{metric_name.capitalize()} Distribution")
            return
        
        # Create violin plot
        positions = range(len(valid_steps))
        violins = ax.violinplot(step_data, positions=positions, showmeans=True)
        
        # Customize violins
        for pc in violins['bodies']:
            pc.set_facecolor('blue')
            pc.set_alpha(0.7)
        
        # Set labels and title
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{s}" for s in valid_steps], rotation=45)
        ax.set_xlabel("Step")
        ax.set_ylabel(f"{metric_name.capitalize()} Value")
        ax.set_title(f"{metric_name.capitalize()} Distribution Across Heads")
        ax.grid(True, alpha=0.3)
    
    def _plot_top_bottom_heads(self, ax, metric_name, top=True):
        """Plot the top or bottom heads based on a metric."""
        # Collect all head scores at the last available step for each head
        head_scores = {}
        for head_id in self.head_scores:
            score_data = self.head_scores[head_id]
            if metric_name in score_data and len(score_data[metric_name]) > 0:
                head_scores[head_id] = score_data[metric_name][-1]
        
        if not head_scores:
            ax.text(0.5, 0.5, f"No {metric_name} data available", 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f"{'Top' if top else 'Bottom'} Heads by {metric_name.capitalize()}")
            return
        
        # Sort heads by score
        sorted_heads = sorted(head_scores.items(), key=lambda x: x[1], reverse=top)
        num_heads = min(10, len(sorted_heads))
        selected_heads = sorted_heads[:num_heads]
        
        # Create bar chart
        y_pos = range(num_heads)
        heights = [score for _, score in selected_heads]
        bars = ax.barh(y_pos, heights, align='center')
        
        # Color bars based on pruned/active status
        for i, (head_id, _) in enumerate(selected_heads):
            if head_id in self.pruned_heads:
                bars[i].set_color('red')
            elif head_id in self.active_heads:
                bars[i].set_color('green')
            else:
                bars[i].set_color('blue')
        
        # Set labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels([head_id for head_id, _ in selected_heads])
        ax.set_xlabel(f"{metric_name.capitalize()} Value")
        ax.set_title(f"{'Top' if top else 'Bottom'} {num_heads} Heads by {metric_name.capitalize()}")
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Active'),
            Patch(facecolor='red', label='Pruned'),
            Patch(facecolor='blue', label='Unknown')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_head_recovery(self, ax):
        """Plot head recovery information."""
        if not self.head_recovery:
            ax.text(0.5, 0.5, "No head recovery data available", 
                   ha='center', va='center', fontsize=12)
            ax.set_title("Head Recovery Analysis")
            return
        
        # Identify recovered heads (those that were pruned and then activated)
        recovered_heads = set()
        recovery_steps = {}
        
        for head_id in self.head_recovery:
            recovery_data = self.head_recovery[head_id]
            if len(recovery_data["recovery_score"]) > 0 and recovery_data["recovery_score"][-1] > 0.5:
                recovered_heads.add(head_id)
                recovery_steps[head_id] = recovery_data["steps"][-1]
        
        if not recovered_heads:
            ax.text(0.5, 0.5, "No recovered heads detected", 
                   ha='center', va='center', fontsize=12)
            ax.set_title("Head Recovery Analysis")
            return
        
        # Plot recovery timeline
        events = []
        for info in self.pruning_info:
            if "pruned_heads" in info and "step" in info:
                for head in info["pruned_heads"]:
                    if head in recovered_heads:
                        events.append((info["step"], head, "pruned"))
        
        for head in recovered_heads:
            if head in recovery_steps:
                events.append((recovery_steps[head], head, "recovered"))
        
        # Sort events by step
        events.sort(key=lambda x: x[0])
        
        # Plot events on timeline
        unique_heads = sorted(list(recovered_heads))
        y_pos = {head: i for i, head in enumerate(unique_heads)}
        
        for step, head, event_type in events:
            if event_type == "pruned":
                marker = 'x'
                color = 'red'
                marker_size = 100
            else:
                marker = 'o'
                color = 'green'
                marker_size = 100
            
            ax.scatter(step, y_pos[head], marker=marker, s=marker_size, 
                      color=color, alpha=0.7, label=f"{event_type}" if f"{event_type}" not in ax.get_legend_handles_labels()[1] else "")
        
        # Connect pruning and recovery events for each head
        for head in unique_heads:
            head_events = [(step, event_type) for step, h, event_type in events if h == head]
            head_events.sort(key=lambda x: x[0])
            
            steps = [step for step, _ in head_events]
            y_values = [y_pos[head]] * len(steps)
            
            ax.plot(steps, y_values, 'k--', alpha=0.3)
        
        # Set labels and title
        ax.set_yticks(range(len(unique_heads)))
        ax.set_yticklabels(unique_heads)
        ax.set_xlabel("Step")
        ax.set_ylabel("Head ID")
        ax.set_title("Head Recovery Timeline")
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        ax.grid(True, alpha=0.3)
    
    def generate_controller_dashboard(self, save_path: Optional[str] = None):
        """
        Generate a dashboard focused on controller metrics and decisions.
        
        Args:
            save_path: Optional path to save the visualization
            
        Returns:
            The matplotlib figure
        """
        if not self.controller_decisions:
            logger.warning("No controller decision data to visualize.")
            return None
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 18))
        
        # Define grid layout
        gs = plt.GridSpec(3, 2)
        
        # 1. Controller decision timeline
        ax_timeline = fig.add_subplot(gs[0, :])
        self._plot_controller_timeline(ax_timeline)
        
        # 2. Head activity heatmap
        ax_activity = fig.add_subplot(gs[1, 0])
        self._plot_head_activity_heatmap(ax_activity)
        
        # 3. Head importance heatmap
        ax_importance = fig.add_subplot(gs[1, 1])
        self._plot_head_importance_heatmap(ax_importance)
        
        # 4. Layer-wise metrics
        ax_layers = fig.add_subplot(gs[2, :])
        self._plot_layer_metrics(ax_layers)
        
        # Set title and adjust layout
        fig.suptitle("ANN Controller Integration Dashboard", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
            logger.info(f"Controller dashboard saved to {save_path}")
        
        return fig
    
    def _plot_controller_timeline(self, ax):
        """Plot controller decision timeline."""
        if not self.controller_decisions:
            ax.text(0.5, 0.5, "No controller decision data available", 
                   ha='center', va='center', fontsize=12)
            ax.set_title("Controller Decision Timeline")
            return
        
        # Extract data from controller decisions
        steps = [decision["step"] for decision in self.controller_decisions]
        active_count = []
        pruned_count = []
        
        for decision in self.controller_decisions:
            if "active_heads" in decision:
                active_count.append(len(decision["active_heads"]))
            else:
                active_count.append(0)
            
            # Count pruned heads at each step
            pruned_at_step = 0
            for info in self.pruning_info:
                if info["step"] <= decision["step"]:
                    pruned_at_step += len(info.get("pruned_heads", []))
            
            pruned_count.append(pruned_at_step)
        
        # Plot active and pruned head counts
        ax.plot(steps, active_count, 'g-', linewidth=2, label="Active Heads")
        ax.plot(steps, pruned_count, 'r-', linewidth=2, label="Cumulative Pruned Heads")
        
        # Add markers for pruning events
        for info in self.pruning_info:
            step = info.get("step", 0)
            heads = len(info.get("pruned_heads", []))
            ax.scatter(step, heads, marker='x', s=100, color='red', zorder=5)
            ax.axvline(x=step, color='red', linestyle=':', alpha=0.5)
        
        # Set labels and title
        ax.set_xlabel("Steps")
        ax.set_ylabel("Head Count")
        ax.set_title("Controller Decision Timeline")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _plot_head_activity_heatmap(self, ax):
        """Plot head activity heatmap based on controller decisions."""
        if not self.controller_decisions:
            ax.text(0.5, 0.5, "No controller decision data available", 
                   ha='center', va='center', fontsize=12)
            ax.set_title("Head Activity Heatmap")
            return
        
        # Collect all unique head IDs
        all_heads = set()
        for decision in self.controller_decisions:
            if "active_heads" in decision:
                all_heads.update(decision["active_heads"])
        
        if not all_heads:
            ax.text(0.5, 0.5, "No head activity data available", 
                   ha='center', va='center', fontsize=12)
            ax.set_title("Head Activity Heatmap")
            return
        
        # Sort heads by ID for consistent visualization
        sorted_heads = sorted(list(all_heads))
        
        # Create activity matrix
        steps = [decision["step"] for decision in self.controller_decisions]
        activity_matrix = np.zeros((len(sorted_heads), len(steps)))
        
        for i, decision in enumerate(self.controller_decisions):
            active_heads = decision.get("active_heads", [])
            for j, head in enumerate(sorted_heads):
                activity_matrix[j, i] = 1 if head in active_heads else 0
        
        # Plot heatmap
        im = ax.imshow(activity_matrix, aspect='auto', cmap='viridis')
        
        # Set labels and title
        ax.set_xlabel("Step Index")
        ax.set_ylabel("Head ID")
        ax.set_title("Head Activity Heatmap")
        
        # Set yticks to show head IDs
        num_heads = len(sorted_heads)
        if num_heads <= 20:
            ax.set_yticks(range(num_heads))
            ax.set_yticklabels(sorted_heads)
        else:
            # Show limited ticks to avoid overcrowding
            tick_indices = np.linspace(0, num_heads-1, 20, dtype=int)
            ax.set_yticks(tick_indices)
            ax.set_yticklabels([sorted_heads[i] for i in tick_indices])
        
        # Set xticks to show step numbers
        num_steps = len(steps)
        if num_steps <= 20:
            ax.set_xticks(range(num_steps))
            ax.set_xticklabels(steps, rotation=45)
        else:
            # Show limited ticks to avoid overcrowding
            tick_indices = np.linspace(0, num_steps-1, 10, dtype=int)
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([steps[i] for i in tick_indices], rotation=45)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label="Activity (1=active, 0=inactive)")
    
    def _plot_head_importance_heatmap(self, ax):
        """Plot head importance heatmap based on scores."""
        if not self.head_scores:
            ax.text(0.5, 0.5, "No head importance data available", 
                   ha='center', va='center', fontsize=12)
            ax.set_title("Head Importance Heatmap")
            return
        
        # Find heads with importance scores
        heads_with_importance = []
        for head_id, scores in self.head_scores.items():
            if "importance" in scores and len(scores["importance"]) > 0:
                heads_with_importance.append(head_id)
        
        if not heads_with_importance:
            ax.text(0.5, 0.5, "No head importance data available", 
                   ha='center', va='center', fontsize=12)
            ax.set_title("Head Importance Heatmap")
            return
        
        # Sort heads by ID for consistent visualization
        sorted_heads = sorted(heads_with_importance)
        
        # Collect all unique steps
        all_steps = set()
        for head_id in sorted_heads:
            score_data = self.head_scores[head_id]
            if "steps" in score_data:
                all_steps.update(score_data["steps"])
        
        sorted_steps = sorted(list(all_steps))
        
        # Create importance matrix
        importance_matrix = np.zeros((len(sorted_heads), len(sorted_steps)))
        for i, head_id in enumerate(sorted_heads):
            score_data = self.head_scores[head_id]
            if "importance" in score_data and "steps" in score_data:
                for j, step in enumerate(sorted_steps):
                    if step in score_data["steps"]:
                        step_idx = score_data["steps"].index(step)
                        if step_idx < len(score_data["importance"]):
                            importance_matrix[i, j] = score_data["importance"][step_idx]
        
        # Plot heatmap
        im = ax.imshow(importance_matrix, aspect='auto', cmap='plasma')
        
        # Set labels and title
        ax.set_xlabel("Step Index")
        ax.set_ylabel("Head ID")
        ax.set_title("Head Importance Heatmap")
        
        # Set yticks to show head IDs
        num_heads = len(sorted_heads)
        if num_heads <= 20:
            ax.set_yticks(range(num_heads))
            ax.set_yticklabels(sorted_heads)
        else:
            # Show limited ticks to avoid overcrowding
            tick_indices = np.linspace(0, num_heads-1, 20, dtype=int)
            ax.set_yticks(tick_indices)
            ax.set_yticklabels([sorted_heads[i] for i in tick_indices])
        
        # Set xticks to show step numbers
        num_steps = len(sorted_steps)
        if num_steps <= 20:
            ax.set_xticks(range(num_steps))
            ax.set_xticklabels(sorted_steps, rotation=45)
        else:
            # Show limited ticks to avoid overcrowding
            tick_indices = np.linspace(0, num_steps-1, 10, dtype=int)
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([sorted_steps[i] for i in tick_indices], rotation=45)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label="Importance Score")
    
    def _generate_sample_html(self) -> str:
        """Generate HTML for text samples."""
        if not hasattr(self, 'comparisons') or not self.comparisons:
            return ''
            
        html_parts = []
        for i, sample in enumerate(self.comparisons):
            html = f"""
            <div class="sample-container" style="margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px;">
                <h4>Sample {i+1}</h4>
                <div style="margin-bottom: 10px;">
                    <strong>Prompt:</strong> <span style="font-style: italic;">{sample['prompt']}</span>
                </div>
                <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                    <div style="flex: 1; min-width: 300px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
                        <h5 style="margin-top: 0; color: #3f51b5;">Baseline Model</h5>
                        <p style="white-space: pre-wrap; font-family: monospace;">{sample['baseline_text']}</p>
                    </div>
                    <div style="flex: 1; min-width: 300px; padding: 10px; background-color: #f0f4ff; border-radius: 5px;">
                        <h5 style="margin-top: 0; color: #4caf50;">Pruned Model</h5>
                        <p style="white-space: pre-wrap; font-family: monospace;">{sample['pruned_text']}</p>
                    </div>
                </div>
            </div>
            """
            html_parts.append(html)
        
        return ''.join(html_parts)
        
    def _plot_layer_metrics(self, ax):
        """Plot layer-wise metrics if available."""
        if not self.layer_metrics:
            ax.text(0.5, 0.5, "No layer metrics data available", 
                   ha='center', va='center', fontsize=12)
            ax.set_title("Layer-wise Metrics")
            return
        
        # Extract layer IDs and steps
        layer_ids = sorted(list(self.layer_metrics.keys()))
        steps = set()
        for layer_id in layer_ids:
            layer_data = self.layer_metrics[layer_id]
            if "steps" in layer_data:
                steps.update(layer_data["steps"])
        
        sorted_steps = sorted(list(steps))
        
        if not sorted_steps:
            ax.text(0.5, 0.5, "No step data available for layers", 
                   ha='center', va='center', fontsize=12)
            ax.set_title("Layer-wise Metrics")
            return
        
        # Plot line for each layer
        for layer_id in layer_ids:
            layer_data = self.layer_metrics[layer_id]
            if "steps" in layer_data and "importance" in layer_data:
                layer_steps = layer_data["steps"]
                layer_importance = layer_data["importance"]
                
                # Ensure data lengths match
                min_len = min(len(layer_steps), len(layer_importance))
                ax.plot(layer_steps[:min_len], layer_importance[:min_len], 
                       marker='o', markersize=4, linewidth=1.5, label=f"Layer {layer_id}")
        
        # Set labels and title
        ax.set_xlabel("Steps")
        ax.set_ylabel("Layer Importance")
        ax.set_title("Layer-wise Importance Over Time")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    def save_dashboard_data(self, output_dir: Optional[str] = None):
        """
        Save dashboard data to JSON files for later analysis.
        
        Args:
            output_dir: Directory to save data files (defaults to self.output_dir)
            
        Returns:
            Path to the saved data directory
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "dashboard_data")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert data to serializable format
        serializable_data = {
            "experiment_metadata": {
                "name": self.experiment_name,
                "timestamp": datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                "config": self.config
            },
            "phases": {
                "phase_list": self.phases,
                "transitions": self.phase_transitions,
                "stabilization_points": self.stabilization_points
            },
            "metrics": {
                "steps": self.all_steps,
                "loss": [float(x) for x in self.all_loss],
                "perplexity": [float(x) for x in self.all_perplexity] if self.all_perplexity else [],
                "sparsity": [float(x) for x in self.all_sparsity] if self.all_sparsity else [],
                "eval_steps": self.eval_steps,
                "eval_perplexity": [float(x) for x in self.eval_perplexity] if self.eval_perplexity else []
            },
            "pruning_events": self.pruning_info,
            "pruned_heads": list(self.pruned_heads)
        }
        
        # Serialize head metrics (avoiding NumPy values)
        head_metrics_serializable = {}
        for head_id, metrics in self.head_metrics.items():
            head_metrics_serializable[head_id] = {
                "steps": metrics["steps"]
            }
            
            for metric_name, values in metrics["metrics"].items():
                head_metrics_serializable[head_id][metric_name] = []
                for x in values:
                    if hasattr(x, 'item'):
                        head_metrics_serializable[head_id][metric_name].append(float(x.item()))
                    elif isinstance(x, (int, float)):
                        head_metrics_serializable[head_id][metric_name].append(float(x))
                    elif isinstance(x, list):
                        # If it's a list, calculate the mean if it contains numbers
                        if all(isinstance(item, (int, float)) for item in x):
                            head_metrics_serializable[head_id][metric_name].append(float(sum(x) / len(x)))
                        else:
                            head_metrics_serializable[head_id][metric_name].append(0.0)  # Fallback
                    else:
                        # Unknown type, use 0.0 as a fallback
                        head_metrics_serializable[head_id][metric_name].append(0.0)
        
        serializable_data["head_metrics"] = head_metrics_serializable
        
        # Write main dashboard data
        with open(os.path.join(output_dir, "dashboard_data.json"), 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        # Write head scores separately (can be large)
        head_scores_serializable = {}
        for head_id, scores in self.head_scores.items():
            head_scores_serializable[head_id] = {}
            for key, values in scores.items():
                if key == "steps" or all(isinstance(x, int) for x in values):
                    head_scores_serializable[head_id][key] = values
                else:
                    head_scores_serializable[head_id][key] = []
                    for x in values:
                        if hasattr(x, 'item'):
                            head_scores_serializable[head_id][key].append(float(x.item()))
                        elif isinstance(x, (int, float)):
                            head_scores_serializable[head_id][key].append(float(x))
                        elif isinstance(x, list):
                            # If it's a list, calculate the mean if it contains numbers
                            if all(isinstance(item, (int, float)) for item in x):
                                head_scores_serializable[head_id][key].append(float(sum(x) / len(x)))
                            else:
                                head_scores_serializable[head_id][key].append(0.0)  # Fallback
                        else:
                            # Unknown type, use 0.0 as a fallback
                            head_scores_serializable[head_id][key].append(0.0)
        
        with open(os.path.join(output_dir, "head_scores.json"), 'w') as f:
            json.dump(head_scores_serializable, f, indent=2)
        
        logger.info(f"Dashboard data saved to: {output_dir}")
        return output_dir