#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Multi-Phase Dashboard Visualization

This script runs a standalone test of the multi-phase dashboard visualization
without requiring the full neural plasticity implementation.

Version: v0.0.1 (2025-04-20 21:45:00)
"""

import os
import sys
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("dashboard-test")

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Create a simplified MultiPhaseDashboard for testing
class SimpleDashboard:
    """Simple dashboard class for testing visualizations."""
    
    def __init__(self, output_dir):
        """Initialize dashboard with output directory."""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
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
        
    def record_phase_transition(self, phase, step):
        """Record a phase transition."""
        self.phases.append(phase)
        self.phase_transitions.append(step)
        self.current_phase = phase
        logger.info(f"Phase transition: {step} -> {phase}")
    
    def record_step(self, metrics, step=None):
        """Record metrics for a step."""
        if step is None:
            step = self.current_step
            self.current_step += 1
        else:
            self.current_step = step + 1
        
        # Capture current phase
        phase = metrics.get("phase", self.current_phase)
        
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
    
    def record_pruning_event(self, pruning_info, step):
        """Record a pruning event."""
        pruning_info["step"] = step
        self.pruning_info.append(pruning_info)
        logger.info(f"Pruning event at step {step}: {len(pruning_info.get('pruned_heads', []))} heads pruned")
    
    def visualize_complete_process(self, save_path=None):
        """Generate a visualization of the complete process."""
        if not self.all_steps:
            logger.warning("No data to visualize.")
            return
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 12))
        
        # Define grid layout
        gs = plt.GridSpec(3, 2, height_ratios=[3, 2, 1])
        
        # 1. Complete training process (top spanning both columns)
        ax_main = fig.add_subplot(gs[0, :])
        self._plot_complete_training_process(ax_main)
        
        # 2. Perplexity and sparsity plots
        ax_perplexity = fig.add_subplot(gs[1, 0])
        self._plot_perplexity(ax_perplexity)
        
        ax_sparsity = fig.add_subplot(gs[1, 1])
        self._plot_sparsity(ax_sparsity)
        
        # 3. Summary statistics
        ax_summary = fig.add_subplot(gs[2, :])
        self._plot_summary_statistics(ax_summary)
        
        # Set title and adjust layout
        fig.suptitle("Neural Plasticity - Complete Training Process", fontsize=16)
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
                    end_step = self.all_steps[-1] if self.all_steps else start_step
                
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
    
    def generate_standalone_dashboard(self, output_dir):
        """Generate a standalone HTML dashboard."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualizations
        complete_process_path = os.path.join(output_dir, "complete_process.png")
        self.visualize_complete_process(save_path=complete_process_path)
        
        # Get summary statistics
        total_steps = self.all_steps[-1] if self.all_steps else 0
        warmup_steps = len(self.warmup_metrics["steps"])
        pruning_steps = len(self.pruning_metrics["steps"])
        finetuning_steps = len(self.finetuning_metrics["steps"])
        
        # Final sparsity
        final_sparsity = self.all_sparsity[-1] if self.all_sparsity else 0
        if final_sparsity <= 1.0:
            final_sparsity *= 100  # Convert to percentage
        
        # Pruned heads
        total_pruned_heads = 0
        for info in self.pruning_info:
            total_pruned_heads += len(info.get("pruned_heads", []))
        
        # Create HTML dashboard
        html_path = os.path.join(output_dir, "dashboard.html")
        
        with open(html_path, 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Neural Plasticity Dashboard</title>
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
                    <h1>Neural Plasticity Dashboard</h1>
                    <p>Test dashboard generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <!-- Summary Metrics -->
                    <div class="dashboard-section">
                        <h2>Experiment Summary</h2>
                        <div class="metrics">
                            <div class="metric-card">
                                <div class="metric-title">Total Steps</div>
                                <div class="metric-value">{total_steps}</div>
                            </div>
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
                                <div class="metric-title">Final Sparsity</div>
                                <div class="metric-value">{final_sparsity:.1f}%</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-title">Pruned Heads</div>
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
                    
                    <div class="summary-box">
                        <strong>Summary:</strong> This test experiment demonstrated the multi-phase dashboard visualization
                        with synthetic data across multiple phases of neural plasticity training.
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


def run_dashboard_test():
    """Run a test of the multi-phase dashboard visualization with synthetic data."""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "output", f"dashboard_test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dashboard
    dashboard = SimpleDashboard(output_dir)
    
    # Generate synthetic data for a multi-phase experiment
    logger.info("Generating synthetic data for dashboard test")
    
    # Warmup phase (150 steps)
    dashboard.record_phase_transition("warmup", 0)
    
    # Create decreasing loss curve with noise for warmup
    base_loss = 5.0 * np.exp(-0.01 * np.arange(150)) + 1.0
    noise = np.random.normal(0, 0.2, size=150)
    warmup_loss = base_loss + noise
    
    # Record warmup data
    for step in range(150):
        perplexity = np.exp(warmup_loss[step])
        
        if step % 25 == 0:
            logger.info(f"Warmup step {step}: Loss = {warmup_loss[step]:.4f}, Perplexity = {perplexity:.2f}")
        
        dashboard.record_step({
            "loss": warmup_loss[step],
            "perplexity": perplexity,
            "sparsity": 0.0,
            "phase": "warmup",
            "stabilized": step == 125  # Stabilization point
        }, step)
    
    # Pruning phase (100 steps)
    dashboard.record_phase_transition("pruning", 150)
    
    # Create loss curve for pruning with spikes at pruning events
    base_loss = 1.0 * np.exp(-0.005 * np.arange(100)) + 0.8
    noise = np.random.normal(0, 0.1, size=100)
    pruning_loss = base_loss + noise
    
    # Add spikes at pruning events
    pruning_loss[5] += 0.5  # First pruning event
    pruning_loss[50] += 0.3  # Second pruning event
    
    # Record pruning phase data
    sparsity = 0.0
    for i, step in enumerate(range(150, 250)):
        # Record pruning events
        if i == 5:
            # First pruning event
            heads = [(0, 1), (1, 2), (2, 3), (3, 0)]
            sparsity = 0.1
            dashboard.record_pruning_event({
                "strategy": "entropy",
                "pruning_level": 0.1,
                "pruned_heads": heads,
                "cycle": 1
            }, step)
        elif i == 50:
            # Second pruning event
            heads = [(4, 1), (5, 2), (6, 0)]
            sparsity = 0.2
            dashboard.record_pruning_event({
                "strategy": "entropy",
                "pruning_level": 0.1,
                "pruned_heads": heads,
                "cycle": 2
            }, step)
        
        perplexity = np.exp(pruning_loss[i])
        
        if i % 25 == 0:
            logger.info(f"Pruning step {step}: Loss = {pruning_loss[i]:.4f}, Perplexity = {perplexity:.2f}, Sparsity = {sparsity:.2f}")
        
        dashboard.record_step({
            "loss": pruning_loss[i],
            "perplexity": perplexity,
            "sparsity": sparsity,
            "phase": "pruning"
        }, step)
    
    # Fine-tuning phase (200 steps)
    dashboard.record_phase_transition("finetuning", 250)
    
    # Create decreasing loss curve for fine-tuning
    base_loss = 0.8 * np.exp(-0.005 * np.arange(200)) + 0.5
    noise = np.random.normal(0, 0.05, size=200)
    finetuning_loss = base_loss + noise
    
    # Record fine-tuning phase data
    for i, step in enumerate(range(250, 450)):
        perplexity = np.exp(finetuning_loss[i])
        
        if i % 50 == 0:
            logger.info(f"Fine-tuning step {step}: Loss = {finetuning_loss[i]:.4f}, Perplexity = {perplexity:.2f}, Sparsity = {sparsity:.2f}")
        
        dashboard.record_step({
            "loss": finetuning_loss[i],
            "perplexity": perplexity,
            "sparsity": sparsity,
            "phase": "finetuning"
        }, step)
    
    # Evaluation phase
    dashboard.record_phase_transition("evaluation", 450)
    
    # Record evaluation data
    eval_steps = np.linspace(0, 450, 10).astype(int)
    eval_perplexity = np.linspace(200, 20, 10)  # Decreasing perplexity (improving)
    
    for i, step in enumerate(eval_steps):
        dashboard.eval_steps.append(step)
        dashboard.eval_perplexity.append(eval_perplexity[i])
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    dashboard.visualize_complete_process(os.path.join(output_dir, "complete_process.png"))
    
    # Generate standalone dashboard
    logger.info("Generating standalone dashboard...")
    html_path = dashboard.generate_standalone_dashboard(output_dir)
    
    # Open dashboard in browser
    try:
        import webbrowser
        logger.info(f"Opening dashboard at: file://{os.path.abspath(html_path)}")
        webbrowser.open(f"file://{os.path.abspath(html_path)}")
    except Exception as e:
        logger.error(f"Could not open browser: {e}")
    
    logger.info(f"Dashboard test completed. Results saved to: {output_dir}")
    return dashboard


if __name__ == "__main__":
    # Run dashboard test
    run_dashboard_test()