#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metrics Dashboard Visualization

This script creates an interactive dashboard to visualize metrics from
training or benchmarking runs, showing pruning patterns, performance metrics,
and head importance over time.

Usage:
    python scripts/visualize_metrics_dashboard.py --metrics_dir ./benchmark_results
"""

import os
import sys
import argparse
import json
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import time
from datetime import datetime

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)


def setup_args():
    """Set up command line arguments."""
    parser = argparse.ArgumentParser(description="Metrics dashboard visualization")
    
    parser.add_argument("--metrics_dir", type=str, required=True,
                       help="Directory containing metrics files (JSONL)")
    parser.add_argument("--refresh_interval", type=float, default=5.0,
                       help="Interval in seconds to refresh the dashboard")
    parser.add_argument("--save_video", action="store_true",
                       help="Save an animation of the dashboard")
    parser.add_argument("--selected_metrics", type=str, 
                        default="loss,perplexity,active_heads_ratio",
                       help="Comma-separated list of metrics to visualize")
    parser.add_argument("--max_frames", type=int, default=100,
                       help="Maximum number of frames for saved video")
    
    return parser.parse_args()


def load_metrics(metrics_dir):
    """Load metrics from JSONL files."""
    metrics_data = []
    
    # Find all JSONL files
    jsonl_files = glob.glob(os.path.join(metrics_dir, "**/*metrics*.jsonl"), recursive=True)
    
    if not jsonl_files:
        print(f"No metrics files found in {metrics_dir}")
        return None
    
    # Load data from all files
    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            metrics_data.append(data)
                        except json.JSONDecodeError:
                            print(f"Error parsing line in {jsonl_file}")
        except Exception as e:
            print(f"Error reading {jsonl_file}: {e}")
    
    if not metrics_data:
        print("No valid metrics data found")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics_data)
    
    # Sort by step if available
    if "step" in df.columns:
        df = df.sort_values("step")
    
    return df


def find_gate_value_files(metrics_dir):
    """Find gate value files in the metrics directory."""
    gate_files = glob.glob(os.path.join(metrics_dir, "**/*gate_values*.pt"), recursive=True)
    
    # Sort by step number (extract from filename)
    def extract_step(filename):
        try:
            step = int(filename.split("step_")[-1].split(".")[0])
            return step
        except:
            return 0
    
    gate_files = sorted(gate_files, key=extract_step)
    
    return gate_files


def create_dashboard(metrics_df, gate_files, selected_metrics, args):
    """Create a dashboard visualization of metrics and gate values."""
    if metrics_df is None or len(metrics_df) == 0:
        print("No valid metrics data available")
        return None
    
    # Create figure
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Sentinel-AI Metrics Dashboard", fontsize=16)
    
    gs = gridspec.GridSpec(3, 3, figure=fig)
    
    # Performance metrics (top left)
    ax_perf = fig.add_subplot(gs[0, 0])
    ax_perf.set_title("Performance Metrics")
    ax_perf.set_xlabel("Step")
    ax_perf.set_ylabel("Value")
    
    # Head activity (top center)
    ax_head = fig.add_subplot(gs[0, 1])
    ax_head.set_title("Head Activity")
    ax_head.set_xlabel("Step")
    ax_head.set_ylabel("Active Head Ratio")
    
    # Gate values heatmap (top right)
    ax_gate = fig.add_subplot(gs[0, 2])
    ax_gate.set_title("Gate Values")
    ax_gate.set_xlabel("Head Index")
    ax_gate.set_ylabel("Layer")
    
    # Pruning pattern (middle left)
    ax_pattern = fig.add_subplot(gs[1, 0])
    ax_pattern.set_title("Pruning Pattern")
    ax_pattern.set_xlabel("Head Index")
    ax_pattern.set_ylabel("Layer")
    
    # Head importance (middle center)
    ax_importance = fig.add_subplot(gs[1, 1])
    ax_importance.set_title("Head Importance")
    ax_importance.set_xlabel("Head Index")
    ax_importance.set_ylabel("Layer")
    
    # Strategy comparison (middle right)
    ax_strategy = fig.add_subplot(gs[1, 2])
    ax_strategy.set_title("Strategy Comparison")
    ax_strategy.set_xlabel("Strategy")
    ax_strategy.set_ylabel("Performance")
    
    # Timeline (bottom row)
    ax_timeline = fig.add_subplot(gs[2, :])
    ax_timeline.set_title("Timeline of Metrics")
    ax_timeline.set_xlabel("Step")
    ax_timeline.set_ylabel("Normalized Value")
    
    # Add text for status display
    status_text = fig.text(0.05, 0.02, "", fontsize=10)
    
    # Tighten layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Create update function for animation
    def update(frame):
        # Clear all axes
        for ax in [ax_perf, ax_head, ax_gate, ax_pattern, ax_importance, ax_strategy, ax_timeline]:
            ax.clear()
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set titles
        ax_perf.set_title("Performance Metrics")
        ax_head.set_title("Head Activity")
        ax_gate.set_title("Gate Values")
        ax_pattern.set_title("Pruning Pattern")
        ax_importance.set_title("Head Importance")
        ax_strategy.set_title("Strategy Comparison")
        ax_timeline.set_title("Timeline of Metrics")
        
        # Set labels
        ax_perf.set_xlabel("Step")
        ax_perf.set_ylabel("Value")
        ax_head.set_xlabel("Step")
        ax_head.set_ylabel("Active Head Ratio")
        ax_gate.set_xlabel("Head Index")
        ax_gate.set_ylabel("Layer")
        ax_pattern.set_xlabel("Head Index")
        ax_pattern.set_ylabel("Layer")
        ax_importance.set_xlabel("Head Index")
        ax_importance.set_ylabel("Layer")
        ax_strategy.set_xlabel("Strategy")
        ax_strategy.set_ylabel("Performance")
        ax_timeline.set_xlabel("Step")
        ax_timeline.set_ylabel("Normalized Value")
        
        # Load updated metrics
        current_metrics = load_metrics(args.metrics_dir)
        
        if current_metrics is None or len(current_metrics) == 0:
            status_text.set_text(f"No metrics data available. Last update: {datetime.now().strftime('%H:%M:%S')}")
            return
        
        current_gate_files = find_gate_value_files(args.metrics_dir)
        
        # Update status text
        status_text.set_text(f"Metrics: {len(current_metrics)} entries. Gate files: {len(current_gate_files)}. "
                           f"Last update: {datetime.now().strftime('%H:%M:%S')}")
        
        # Plot performance metrics
        metrics_to_plot = []
        for metric in selected_metrics:
            matching_cols = [col for col in current_metrics.columns if metric in col]
            metrics_to_plot.extend(matching_cols)
        
        # Limit to first few metrics to avoid crowding
        metrics_to_plot = metrics_to_plot[:5]
        
        for metric in metrics_to_plot:
            if metric in current_metrics.columns:
                valid_data = current_metrics[["step", metric]].dropna()
                if len(valid_data) > 0:
                    ax_perf.plot(valid_data["step"], valid_data[metric], label=metric)
        
        ax_perf.legend()
        
        # Plot head activity
        head_metrics = [col for col in current_metrics.columns if "active_heads" in col and "ratio" not in col]
        
        for metric in head_metrics:
            valid_data = current_metrics[["step", metric]].dropna()
            if len(valid_data) > 0:
                ax_head.plot(valid_data["step"], valid_data[metric], label=metric)
        
        if "active_heads_ratio" in current_metrics.columns:
            valid_data = current_metrics[["step", "active_heads_ratio"]].dropna()
            if len(valid_data) > 0:
                ax_head.plot(valid_data["step"], valid_data["active_heads_ratio"], 
                            label="Overall", linewidth=2, color='black')
        
        ax_head.legend()
        
        # Plot gate values heatmap
        if current_gate_files:
            try:
                # Load most recent gate values file
                latest_gate_file = current_gate_files[-1]
                gate_values = torch.load(latest_gate_file)
                
                # Create a matrix of gate values
                num_layers = len(gate_values)
                max_heads = max(gate.size(0) for gate in gate_values.values())
                
                gate_matrix = np.zeros((num_layers, max_heads))
                
                for i, (layer_name, gate) in enumerate(gate_values.items()):
                    gate_matrix[i, :gate.size(0)] = gate.cpu().numpy()
                
                # Plot heatmap
                im = sns.heatmap(gate_matrix, cmap="viridis", vmin=0, vmax=1, 
                                ax=ax_gate, cbar=True, cbar_kws={"shrink": 0.8})
                ax_gate.set_title(f"Gate Values (Step {os.path.basename(latest_gate_file).split('_')[-1].split('.')[0]})")
                
                # Also plot binary pruning pattern
                pruning_matrix = (gate_matrix > 0.5).astype(float)
                sns.heatmap(pruning_matrix, cmap="Blues", vmin=0, vmax=1, 
                          ax=ax_pattern, cbar=True, cbar_kws={"shrink": 0.8})
                ax_pattern.set_title("Pruning Pattern (White = Pruned)")
                
            except Exception as e:
                print(f"Error plotting gate values: {e}")
                ax_gate.text(0.5, 0.5, "Error loading gate values", 
                            ha='center', va='center', transform=ax_gate.transAxes)
                ax_pattern.text(0.5, 0.5, "Error loading pruning pattern", 
                               ha='center', va='center', transform=ax_pattern.transAxes)
        else:
            ax_gate.text(0.5, 0.5, "No gate values available", 
                        ha='center', va='center', transform=ax_gate.transAxes)
            ax_pattern.text(0.5, 0.5, "No pruning pattern available", 
                           ha='center', va='center', transform=ax_pattern.transAxes)
        
        # Plot head importance (placeholder)
        ax_importance.text(0.5, 0.5, "Head importance data not available", 
                         ha='center', va='center', transform=ax_importance.transAxes)
        
        # Plot strategy comparison
        strategies = [col for col in current_metrics.columns if "strategy" in col]
        if "eval/strategy" in current_metrics.columns:
            strategy_data = current_metrics.groupby("eval/strategy")["eval/perplexity"].mean().reset_index()
            
            if len(strategy_data) > 0:
                # Sort by performance (lower perplexity is better)
                strategy_data = strategy_data.sort_values("eval/perplexity")
                
                # Plot bar chart
                bars = ax_strategy.bar(strategy_data["eval/strategy"], strategy_data["eval/perplexity"])
                
                # Add values on top of bars
                for i, v in enumerate(strategy_data["eval/perplexity"]):
                    ax_strategy.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
                
                ax_strategy.set_title("Strategy Comparison (Perplexity)")
                ax_strategy.set_xticklabels(strategy_data["eval/strategy"], rotation=45, ha="right")
            else:
                ax_strategy.text(0.5, 0.5, "No strategy comparison data", 
                               ha='center', va='center', transform=ax_strategy.transAxes)
        else:
            ax_strategy.text(0.5, 0.5, "No strategy comparison data", 
                           ha='center', va='center', transform=ax_strategy.transAxes)
        
        # Plot timeline of normalized metrics
        timeline_metrics = metrics_to_plot[:3]  # Limit to 3 metrics for clarity
        
        if timeline_metrics:
            for metric in timeline_metrics:
                if metric in current_metrics.columns:
                    valid_data = current_metrics[["step", metric]].dropna()
                    if len(valid_data) > 0:
                        # Normalize to [0, 1] range for visualization
                        values = valid_data[metric]
                        min_val = values.min()
                        max_val = values.max()
                        if max_val > min_val:
                            normalized = (values - min_val) / (max_val - min_val)
                            ax_timeline.plot(valid_data["step"], normalized, label=metric)
                            
                            # Add trend line
                            if len(valid_data) > 2:
                                z = np.polyfit(valid_data["step"], normalized, 1)
                                p = np.poly1d(z)
                                x_range = np.linspace(valid_data["step"].min(), valid_data["step"].max(), 100)
                                ax_timeline.plot(x_range, p(x_range), linestyle='--', alpha=0.7)
            
            ax_timeline.legend()
            ax_timeline.set_title("Timeline of Normalized Metrics (with Trend Lines)")
        else:
            ax_timeline.text(0.5, 0.5, "No timeline data available", 
                          ha='center', va='center', transform=ax_timeline.transAxes)
    
    # Create animation
    if args.save_video:
        ani = FuncAnimation(fig, update, frames=min(args.max_frames, 100), interval=1000, blit=False)
        
        # Save animation
        output_file = os.path.join(args.metrics_dir, "metrics_dashboard.mp4")
        ani.save(output_file, writer='ffmpeg', fps=4)
        print(f"Saved animation to {output_file}")
    else:
        # Interactive mode
        plt.ion()  # Turn on interactive mode
        
        try:
            while True:
                update(0)
                plt.draw()
                plt.pause(args.refresh_interval)
        except KeyboardInterrupt:
            print("Dashboard stopped by user")
        finally:
            plt.ioff()  # Turn off interactive mode
    
    plt.close()
    return True


def main():
    """Main function."""
    args = setup_args()
    
    if not os.path.exists(args.metrics_dir):
        print(f"Error: Metrics directory '{args.metrics_dir}' does not exist")
        return 1
    
    # Load metrics data
    metrics_df = load_metrics(args.metrics_dir)
    
    if metrics_df is None:
        print("No metrics data found. Waiting for data...")
        # Wait for first metrics file to appear
        while metrics_df is None:
            time.sleep(args.refresh_interval)
            metrics_df = load_metrics(args.metrics_dir)
            if metrics_df is None:
                print("Still waiting for metrics data...")
    
    # Find gate value files
    gate_files = find_gate_value_files(args.metrics_dir)
    
    # Parse selected metrics
    selected_metrics = args.selected_metrics.split(",")
    
    # Create dashboard
    create_dashboard(metrics_df, gate_files, selected_metrics, args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())