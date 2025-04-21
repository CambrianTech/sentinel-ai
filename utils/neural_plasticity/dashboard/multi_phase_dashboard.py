#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural Plasticity Multi-Phase Dashboard

This module provides extended dashboard capabilities for tracking neural plasticity
experiments across multiple phases and cycles, with comprehensive visualizations
for the entire training process. Supports both single and multi-cycle experiments
with detailed timeline, metrics tracking, and interactive visualizations.

Version: v0.0.2 (2025-04-20 21:30:00)
"""

import os
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from pathlib import Path
from datetime import datetime

# Conditional imports for Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import the base dashboard functionality
from .wandb_integration import WandbDashboard

logger = logging.getLogger(__name__)

class MultiPhaseDashboard(WandbDashboard):
    """
    Extended dashboard for tracking multi-phase neural plasticity experiments.
    
    This class builds on the base WandbDashboard to provide additional visualizations
    and tracking capabilities for experiments with multiple phases (warmup, pruning,
    fine-tuning) and potentially multiple cycles.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the multi-phase dashboard."""
        super().__init__(*args, **kwargs)
        
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
        
        # Add default tags
        if "multi-phase" not in self.tags:
            self.tags.append("multi-phase")
    
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
        
        # Log metrics to wandb
        self.log_metrics(metrics, step)
    
    def record_phase_transition(self, phase: str, step: int):
        """
        Record a phase transition.
        
        Args:
            phase: New phase
            step: Step number of the transition
        """
        self.phases.append(phase)
        self.phase_transitions.append(step)
        self.set_phase(phase, step)
    
    def record_pruning_event(self, pruning_info: Dict[str, Any], step: int):
        """
        Record a pruning event.
        
        Args:
            pruning_info: Dictionary with pruning details
            step: Step number of the pruning event
        """
        pruning_info["step"] = step
        self.pruning_info.append(pruning_info)
        self.log_pruning_decision(pruning_info, step)
    
    def visualize_complete_process(self, save_path: Optional[str] = None):
        """
        Generate a comprehensive visualization of the complete training process.
        
        Args:
            save_path: Optional path to save the visualization
        """
        if not self.all_steps:
            logger.warning("No data to visualize.")
            return
        
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
            
            # Log to wandb
            if self.initialized:
                wandb.log({"complete_process": wandb.Image(fig)})
        
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
        
        # Add source quote
        if hasattr(self, 'run') and self.run:
            entity = self.run.entity if hasattr(self.run, 'entity') else "user"
            project = self.run.project if hasattr(self.run, 'project') else "project"
            url = self.run.url if hasattr(self.run, 'url') else ""
            
            if url:
                source = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Dashboard: {url}"
            else:
                source = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                
            ax.text(0.99, 0.01, source, ha='right', va='bottom', fontsize=8,
                   transform=ax.transAxes, alpha=0.7)
        
        # Add quote
        quote = '"In the neural world, connections that fire together wire together, and those that don\'t, don\'t."'
        ax.text(0.01, 0.01, quote, ha='left', va='bottom', fontsize=8, 
               style='italic', transform=ax.transAxes, alpha=0.7)
    
    def generate_multi_cycle_dashboard(self, save_path: Optional[str] = None):
        """
        Generate a dashboard for an experiment with multiple pruning cycles.
        
        Args:
            save_path: Optional path to save the visualization
        """
        if not self.all_steps:
            logger.warning("No data to visualize.")
            return
        
        # Extract cycle info from pruning events
        cycles = []
        for info in self.pruning_info:
            if 'cycle' in info and info['cycle'] not in cycles:
                cycles.append(info['cycle'])
        
        if not cycles:
            logger.warning("No cycle information found in pruning events.")
            return
        
        # Create figure
        fig = plt.figure(figsize=(15, 10 + len(cycles) * 3))
        
        # Define grid layout
        gs = plt.GridSpec(2 + len(cycles), 2, height_ratios=[3] + [1.5] * len(cycles) + [1])
        
        # 1. Complete training process (top spanning both columns)
        ax_main = fig.add_subplot(gs[0, :])
        self._plot_complete_training_process(ax_main)
        
        # 2. Cycle-specific plots
        # Group data by cycles
        cycle_data = {}
        
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
            
            # Log to wandb
            if self.initialized:
                wandb.log({"multi_cycle_process": wandb.Image(fig)})
        
        return fig
    
    def generate_standalone_dashboard(self, output_dir: str):
        """
        Generate a standalone HTML dashboard with all visualizations.
        
        Args:
            output_dir: Directory to save the dashboard files
        
        Returns:
            Path to the HTML dashboard file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations
        complete_process_path = os.path.join(output_dir, "complete_process.png")
        self.visualize_complete_process(complete_process_path)
        
        # Create HTML dashboard
        html_path = os.path.join(output_dir, "dashboard.html")
        
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
            
        # Extract cycle information
        cycles = []
        for info in self.pruning_info:
            if 'cycle' in info and info['cycle'] not in cycles:
                cycles.append(info['cycle'])
        
        # Generate visualizations
        complete_process_path = os.path.join(output_dir, "complete_process.png")
        self.visualize_complete_process(complete_process_path)
        
        # Generate multi-cycle visualization if there are multiple cycles
        multi_cycle_path = None
        if len(cycles) > 1:
            multi_cycle_path = os.path.join(output_dir, "multi_cycle_process.png")
            self.generate_multi_cycle_dashboard(multi_cycle_path)
        
        # Generate the HTML content
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
                    h1, h2 {{ color: #3f51b5; }}
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
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Neural Plasticity Multi-Phase Dashboard</h1>
                    <p>Experiment visualization generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <!-- Summary Metrics -->
                    <div class="dashboard-section">
                        <h2>Experiment Summary</h2>
                        <div class="metrics">
                            <div class="metric-card">
                                <div class="metric-title">Warmup Steps</div>
                                <div class="metric-value">{warmup_steps}</div>
                            </div>
                            <div class="metric-card">
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
                            <div class="metric-card">
                                <div class="metric-title">Final Sparsity</div>
                                <div class="metric-value">{final_sparsity:.1f}%</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-title">Total Pruned Heads</div>
                                <div class="metric-value">{total_pruned_heads}</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Complete Process Visualization -->
                    <div class="dashboard-section">
                        <h2>Complete Process</h2>
                        <p>This visualization shows the entire neural plasticity process including warmup, pruning, and fine-tuning phases.</p>
                        <img src="complete_process.png" alt="Complete Process Visualization">
                    </div>
                    
                    {f'''
                    <!-- Multi-Cycle Visualization -->
                    <div class="dashboard-section">
                        <h2>Multi-Cycle Analysis</h2>
                        <p>This visualization shows the detailed breakdown of each pruning cycle.</p>
                        <img src="multi_cycle_process.png" alt="Multi-Cycle Analysis">
                    </div>
                    ''' if multi_cycle_path else ''}
                    
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
                            <strong>Summary:</strong> This experiment involved {warmup_steps} warmup steps, 
                            {pruning_steps} pruning steps, and {finetuning_steps} fine-tuning steps across 
                            {len(cycles) if cycles else 1} pruning cycle(s).
                            The training stabilized at step {warmup_stab} and achieved a final model sparsity of 
                            {final_sparsity:.1f}% by pruning {total_pruned_heads} attention heads.
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


# Demo function to test the multi-phase dashboard
def run_multi_phase_demo():
    """Run a demo of the multi-phase dashboard with mock data."""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", f"multi_phase_demo_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dashboard
    dashboard = MultiPhaseDashboard(
        project_name="neural-plasticity-demo",
        experiment_name=f"multi-phase-demo-{timestamp}",
        output_dir=output_dir,
        config={
            "model_name": "gpt2",
            "dataset": "wikitext/wikitext-2",
            "pruning_strategy": "entropy",
            "pruning_level": 0.2,
            "demo_mode": True
        },
        mode="offline",
        tags=["demo", "multi-phase"]
    )
    
    # Generate mock data for warmup phase
    dashboard.record_phase_transition("warmup", 0)
    
    # Warmup phase (150 steps)
    base_loss = 7.5 * np.exp(-0.01 * np.arange(150)) + 0.5
    noise = np.random.normal(0, 0.2, size=150)
    warmup_loss = base_loss + noise
    
    for step in range(150):
        perplexity = np.exp(warmup_loss[step])
        
        if step % 10 == 0:
            print(f"Warmup step {step}: Loss = {warmup_loss[step]:.4f}, Perplexity = {perplexity:.2f}")
        
        # Record metrics
        dashboard.record_step({
            "loss": warmup_loss[step],
            "perplexity": perplexity,
            "phase": "warmup",
            "stabilized": step == 75  # Stabilization point at step 75
        }, step)
    
    # Record stabilization
    dashboard.stabilization_points.append((75, "warmup"))
    
    # Analysis phase (10 steps)
    dashboard.record_phase_transition("analysis", 150)
    for step in range(150, 160):
        dashboard.record_step({
            "loss": warmup_loss[-1],  # Keep the last warmup loss
            "perplexity": np.exp(warmup_loss[-1]),
            "phase": "analysis"
        }, step)
    
    # Pruning phase (100 steps)
    dashboard.record_phase_transition("pruning", 160)
    
    # Create a decreasing loss curve for pruning
    base_loss = 6.0 * np.exp(-0.02 * np.arange(100)) + 1.0
    noise = np.random.normal(0, 0.2, size=100)
    pruning_loss = base_loss + noise
    
    # First pruning event
    pruning_info = {
        "strategy": "entropy",
        "pruning_level": 0.1,
        "pruned_heads": [(0, 2), (1, 3), (2, 1), (3, 4), (4, 2)],
        "cycle": 1
    }
    dashboard.record_pruning_event(pruning_info, 165)
    
    sparsity = 0.0
    for i, step in enumerate(range(160, 260)):
        # Increase sparsity at pruning events
        if step == 165:
            sparsity = 0.1
        elif step == 210:
            sparsity = 0.15
            # Second pruning event
            pruning_info = {
                "strategy": "entropy",
                "pruning_level": 0.05,
                "pruned_heads": [(5, 1), (6, 2), (7, 3), (8, 4)],
                "cycle": 2
            }
            dashboard.record_pruning_event(pruning_info, step)
        
        perplexity = np.exp(pruning_loss[i])
        
        if i % 10 == 0:
            print(f"Pruning step {step}: Loss = {pruning_loss[i]:.4f}, Perplexity = {perplexity:.2f}, Sparsity = {sparsity:.2f}")
        
        # Record metrics
        dashboard.record_step({
            "loss": pruning_loss[i],
            "perplexity": perplexity,
            "sparsity": sparsity,
            "phase": "pruning"
        }, step)
    
    # Fine-tuning phase (200 steps)
    dashboard.record_phase_transition("finetuning", 260)
    
    # Create a decreasing loss curve for fine-tuning
    base_loss = 1.0 * np.exp(-0.01 * np.arange(200)) + 0.5
    noise = np.random.normal(0, 0.1, size=200)
    finetuning_loss = base_loss + noise
    
    # Final pruning event to reach 20% sparsity
    pruning_info = {
        "strategy": "entropy",
        "pruning_level": 0.05,
        "pruned_heads": [(9, 1), (10, 2), (11, 3)],
        "cycle": 3
    }
    dashboard.record_pruning_event(pruning_info, 265)
    sparsity = 0.2
    
    for i, step in enumerate(range(260, 460)):
        perplexity = np.exp(finetuning_loss[i])
        
        if i % 20 == 0:
            print(f"Fine-tuning step {step}: Loss = {finetuning_loss[i]:.4f}, Perplexity = {perplexity:.2f}, Sparsity = {sparsity:.2f}")
        
        # Record metrics
        dashboard.record_step({
            "loss": finetuning_loss[i],
            "perplexity": perplexity,
            "sparsity": sparsity,
            "phase": "finetuning"
        }, step)
    
    # Evaluation phase
    dashboard.record_phase_transition("evaluation", 460)
    
    # Record evaluation metrics
    eval_steps = np.linspace(0, 460, 10).astype(int)
    initial_perplexity = np.exp(warmup_loss[0])
    final_perplexity = np.exp(finetuning_loss[-1])
    
    eval_perplexity = np.linspace(initial_perplexity, final_perplexity, len(eval_steps))
    
    for i, step in enumerate(eval_steps):
        dashboard.eval_steps.append(step)
        dashboard.eval_perplexity.append(eval_perplexity[i])
    
    # Generate visualizations
    print("Generating complete process visualization...")
    fig = dashboard.visualize_complete_process(os.path.join(output_dir, "complete_process.png"))
    
    # Generate standalone dashboard
    print("Generating standalone dashboard...")
    html_path = dashboard.generate_standalone_dashboard(output_dir)
    
    # Open dashboard in browser
    try:
        import webbrowser
        print(f"Opening dashboard at: file://{os.path.abspath(html_path)}")
        webbrowser.open(f"file://{os.path.abspath(html_path)}")
    except Exception as e:
        print(f"Could not open browser: {e}")
    
    print(f"Multi-phase dashboard demo completed. Results saved to: {output_dir}")
    return dashboard


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run demo
    run_multi_phase_demo()