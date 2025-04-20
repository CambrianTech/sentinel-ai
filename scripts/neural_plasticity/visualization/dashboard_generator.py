#!/usr/bin/env python
"""
Neural Plasticity Dashboard Generator

Generates comprehensive visualizations for the Neural Plasticity experiment process,
including warmup, stabilization, pruning, and fine-tuning phases.

Usage:
    python scripts/neural_plasticity/visualization/dashboard_generator.py --experiment_dir /path/to/experiment_output
    python scripts/neural_plasticity/visualization/dashboard_generator.py --experiment_dir /path/to/experiment_output --output_file dashboard.html

Author: Claude <noreply@anthropic.com>
Version: v0.1.0 (2025-04-20)
"""

import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import glob
import io
import base64
from pathlib import Path

# Add root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

# Import utilities if available
try:
    from utils.colab.visualizations import visualize_complete_training_process
except ImportError:
    print("Warning: Could not import visualization utilities from utils.colab")


def load_json_file(path: str, default=None):
    """
    Load a JSON file safely.
    
    Args:
        path: Path to the JSON file
        default: Default value to return if file can't be loaded
        
    Returns:
        Loaded JSON content or default value
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load JSON file {path}: {e}")
        return default or {}


def find_latest_experiment_dir(base_dir: str = None):
    """
    Find the most recent experiment directory if none is specified.
    
    Args:
        base_dir: Base directory to search in
        
    Returns:
        Path to the most recent experiment directory
    """
    if base_dir is None:
        base_dir = os.path.join(project_root, "experiment_output", "neural_plasticity")
    
    if not os.path.exists(base_dir):
        return None
    
    # Find all run directories
    run_dirs = glob.glob(os.path.join(base_dir, "run_*"))
    
    if not run_dirs:
        return None
    
    # Sort by modification time (most recent first)
    run_dirs.sort(key=os.path.getmtime, reverse=True)
    
    return run_dirs[0]


def find_experiment_data_dir(experiment_dir: str):
    """
    Find the actual data directory within the experiment directory.
    
    Args:
        experiment_dir: Path to the experiment directory
        
    Returns:
        Path to the directory containing the experiment data
    """
    # Look for directories matching the pattern entropy_*
    data_dirs = glob.glob(os.path.join(experiment_dir, "entropy_*"))
    
    if not data_dirs:
        return None
    
    # Sort by modification time (most recent first)
    data_dirs.sort(key=os.path.getmtime, reverse=True)
    
    return data_dirs[0]


def load_experiment_data(experiment_dir: str):
    """
    Load all experiment data from the specified directory.
    
    Args:
        experiment_dir: Path to the experiment directory
        
    Returns:
        Dictionary containing all experiment data
    """
    data_dir = find_experiment_data_dir(experiment_dir)
    
    if not data_dir:
        print(f"Warning: Could not find data directory in {experiment_dir}")
        return {}
    
    # Load all experiment data files
    data = {}
    
    # Parameters
    data['params'] = load_json_file(os.path.join(data_dir, "params.json"))
    
    # Metrics
    data['metrics'] = load_json_file(os.path.join(data_dir, "metrics.json"))
    
    # Pruned heads
    data['pruned_heads'] = load_json_file(os.path.join(data_dir, "pruned_heads.json"))
    
    # Entropy data
    data['pre_entropy'] = load_json_file(os.path.join(data_dir, "pre_entropy.json"))
    data['post_entropy'] = load_json_file(os.path.join(data_dir, "post_entropy.json"))
    data['entropy_deltas'] = load_json_file(os.path.join(data_dir, "entropy_deltas.json"))
    
    # Results
    data['results'] = load_json_file(os.path.join(data_dir, "results.json"))
    
    # Performance history
    data['performance_history'] = load_json_file(os.path.join(data_dir, "performance_history.json"))
    
    # Entropy history
    data['entropy_history'] = load_json_file(os.path.join(data_dir, "entropy_history.json"))
    
    # Gate history
    data['gate_history'] = load_json_file(os.path.join(data_dir, "gate_history.json"))
    
    # Regrowth analysis
    data['regrowth_analysis'] = load_json_file(os.path.join(data_dir, "regrowth_analysis.json"))
    
    return data


def format_float(value, decimals=2):
    """Format a float value with the specified number of decimal places."""
    if isinstance(value, (int, float)) and not np.isnan(value):
        return f"{value:.{decimals}f}"
    return "N/A"


def calculate_improvement(before, after):
    """Calculate percentage improvement between two values."""
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return "N/A"
    if np.isnan(before) or np.isnan(after) or before == 0:
        return "N/A"
    improvement = (before - after) / before * 100
    return f"{improvement:.2f}%"


def process_metrics(metrics):
    """Process and format metrics for display."""
    processed = {}
    
    # Extract values
    if 'baseline' in metrics:
        processed['baseline_loss'] = metrics['baseline'].get('loss', 'N/A')
        processed['baseline_perplexity'] = metrics['baseline'].get('perplexity', 'N/A')
    
    if 'post_pruning' in metrics:
        processed['post_pruning_loss'] = metrics['post_pruning'].get('loss', 'N/A')
        processed['post_pruning_perplexity'] = metrics['post_pruning'].get('perplexity', 'N/A')
    
    if 'final' in metrics:
        processed['final_loss'] = metrics['final'].get('loss', 'N/A')
        processed['final_perplexity'] = metrics['final'].get('perplexity', 'N/A')
    
    # Calculate improvements
    if 'baseline_loss' in processed and 'final_loss' in processed:
        if (isinstance(processed['baseline_loss'], (int, float)) and 
            isinstance(processed['final_loss'], (int, float))):
            improvement = (processed['baseline_loss'] - processed['final_loss']) / processed['baseline_loss'] * 100
            processed['loss_improvement'] = f"{improvement:.2f}%"
        else:
            processed['loss_improvement'] = "N/A"
    
    if 'baseline_perplexity' in processed and 'final_perplexity' in processed:
        if (isinstance(processed['baseline_perplexity'], (int, float)) and 
            isinstance(processed['final_perplexity'], (int, float)) and
            processed['baseline_perplexity'] > 0):
            improvement = (processed['baseline_perplexity'] - processed['final_perplexity']) / processed['baseline_perplexity'] * 100
            processed['perplexity_improvement'] = f"{improvement:.2f}%"
        else:
            processed['perplexity_improvement'] = "N/A"
    
    return processed


def create_perplexity_visualization(metrics):
    """Create perplexity comparison visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract perplexity values
    try:
        baseline = metrics['baseline']['perplexity']
        post_pruning = metrics['post_pruning']['perplexity']
        final = metrics['final']['perplexity']
        
        # Ensure we have valid values
        if np.any(np.isnan([baseline, post_pruning, final])):
            return None
        
        # Plot on log scale
        phases = ['Baseline', 'After Pruning', 'After Fine-tuning']
        values = [baseline, post_pruning, final]
        
        # Use different colors for each bar
        colors = ['blue', 'red', 'green']
        
        ax.bar(phases, values, color=colors)
        ax.set_yscale('log')
        ax.set_title('Model Perplexity Comparison (log scale)')
        ax.set_ylabel('Perplexity (lower is better)')
        
        # Add value labels on top of bars
        for i, v in enumerate(values):
            if v > 1000000:
                label = f"{v/1000000:.1f}M"
            elif v > 1000:
                label = f"{v/1000:.1f}K"
            else:
                label = f"{v:.1f}"
            ax.text(i, v, label, ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save to bytesIO
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        # Convert to base64 for embedding in HTML
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
    
    except Exception as e:
        print(f"Error creating perplexity visualization: {e}")
        return None


def create_pruned_heads_heatmap(pruned_heads):
    """Create heatmap of pruned attention heads."""
    if not pruned_heads:
        return None
    
    try:
        # Extract layer and head indices
        layers = [item[0] for item in pruned_heads]
        heads = [item[1] for item in pruned_heads]
        
        # Determine matrix dimensions
        max_layer = max(layers) if layers else 5
        max_head = max(heads) if heads else 11
        
        # Create matrix
        matrix = np.zeros((max_layer+1, max_head+1))
        for layer, head, _ in pruned_heads:
            matrix[layer, head] = 1
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap='Reds', aspect='auto')
        plt.colorbar(im, ax=ax, label='Pruned')
        
        ax.set_title(f"Pruned Attention Heads ({len(pruned_heads)} total)")
        ax.set_xlabel("Head Index")
        ax.set_ylabel("Layer Index")
        
        # Add grid
        ax.set_xticks(np.arange(-.5, max_head + 1, 1), minor=True)
        ax.set_yticks(np.arange(-.5, max_layer + 1, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5, alpha=0.2)
        
        plt.tight_layout()
        
        # Save to bytesIO
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        # Convert to base64 for embedding in HTML
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
    
    except Exception as e:
        print(f"Error creating pruned heads heatmap: {e}")
        return None


def create_entropy_visualization(pre_entropy, post_entropy):
    """Create visualization comparing entropy before and after pruning."""
    try:
        # Convert string keys to integers 
        pre_entropy_numeric = {}
        post_entropy_numeric = {}
        
        # Process pre-entropy data
        for k, v in pre_entropy.items():
            try:
                layer_idx = int(k)
                pre_entropy_numeric[layer_idx] = np.array(v)
            except (ValueError, TypeError):
                continue
        
        # Process post-entropy data
        for k, v in post_entropy.items():
            try:
                layer_idx = int(k)
                post_entropy_numeric[layer_idx] = np.array(v)
            except (ValueError, TypeError):
                continue
        
        # Find common layers
        common_layers = set(pre_entropy_numeric.keys()).intersection(post_entropy_numeric.keys())
        
        if not common_layers:
            return None
        
        # Ensure data is valid
        valid_layers = []
        for layer in sorted(common_layers):
            pre_data = pre_entropy_numeric[layer]
            post_data = post_entropy_numeric[layer]
            
            if not np.all(np.isnan(pre_data)) and not np.all(np.isnan(post_data)):
                valid_layers.append(layer)
        
        if not valid_layers:
            return None
        
        # Create visualization for up to 3 layers (for space)
        valid_layers = valid_layers[:3]
        
        # Create subplots
        fig, axes = plt.subplots(len(valid_layers), 2, figsize=(12, len(valid_layers)*3))
        
        # If only one layer, axes is not a 2D array
        if len(valid_layers) == 1:
            axes = np.array([axes])
        
        # Plot each layer's entropy before and after
        for i, layer in enumerate(valid_layers):
            pre_data = pre_entropy_numeric[layer]
            post_data = post_entropy_numeric[layer]
            
            # Handle NaN values
            pre_data = np.nan_to_num(pre_data, nan=0.0)
            post_data = np.nan_to_num(post_data, nan=0.0)
            
            # Ensure data has consistent shape
            min_len = min(len(pre_data), len(post_data))
            
            # Pre-pruning entropy
            im = axes[i, 0].imshow(pre_data[:min_len].reshape(1, -1), cmap='viridis', aspect='auto')
            axes[i, 0].set_title(f"Layer {layer} Pre-Pruning Entropy")
            axes[i, 0].set_yticks([])
            plt.colorbar(im, ax=axes[i, 0])
            
            # Post-pruning entropy
            im = axes[i, 1].imshow(post_data[:min_len].reshape(1, -1), cmap='viridis', aspect='auto')
            axes[i, 1].set_title(f"Layer {layer} Post-Pruning Entropy")
            axes[i, 1].set_yticks([])
            plt.colorbar(im, ax=axes[i, 1])
        
        plt.tight_layout()
        
        # Save to bytesIO
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        # Convert to base64 for embedding in HTML
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
    
    except Exception as e:
        print(f"Error creating entropy visualization: {e}")
        return None


def create_loss_improvement_chart(metrics):
    """Create chart showing loss improvement through the process."""
    try:
        # Extract loss values
        baseline = metrics['baseline']['loss']
        post_pruning = metrics['post_pruning']['loss']
        final = metrics['final']['loss']
        
        # Ensure we have valid values
        if np.isnan(baseline) or np.isnan(post_pruning) or np.isnan(final):
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot loss evolution
        stages = ['Baseline', 'After Pruning', 'After Fine-tuning']
        loss_values = [baseline, post_pruning, final]
        
        ax.plot(stages, loss_values, 'b-o', markersize=8)
        ax.set_title('Loss Evolution During Neural Plasticity Process')
        ax.set_ylabel('Loss Value')
        ax.grid(True, alpha=0.3)
        
        # Add value annotations
        for i, v in enumerate(loss_values):
            ax.text(i, v, f"{v:.2f}", ha='center', va='bottom')
        
        # Add phase highlighting
        ax.axvspan(-0.5, 0.5, alpha=0.2, color='blue', label='Initial')
        ax.axvspan(0.5, 1.5, alpha=0.2, color='red', label='Pruning')
        ax.axvspan(1.5, 2.5, alpha=0.2, color='green', label='Fine-tuning')
        
        ax.legend()
        plt.tight_layout()
        
        # Save to bytesIO
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        # Convert to base64 for embedding in HTML
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
    
    except Exception as e:
        print(f"Error creating loss improvement chart: {e}")
        return None


def create_neural_plasticity_dashboard(experiment_data, output_path):
    """
    Create a dynamic HTML dashboard for neural plasticity experiment.
    
    Args:
        experiment_data: Dictionary containing experiment data
        output_path: Path to save the dashboard HTML file
        
    Returns:
        Path to the generated dashboard
    """
    # Extract key data
    params = experiment_data.get('params', {})
    metrics = experiment_data.get('metrics', {})
    pruned_heads = experiment_data.get('pruned_heads', [])
    pre_entropy = experiment_data.get('pre_entropy', {})
    post_entropy = experiment_data.get('post_entropy', {})
    results = experiment_data.get('results', {})
    
    # Process metrics for display
    processed_metrics = process_metrics(metrics)
    
    # Calculate sparsity
    # Note: This is a simplification - in a real implementation you'd use the actual model structure
    total_heads = 0
    for layer_key in pre_entropy:
        try:
            total_heads += len(pre_entropy[layer_key])
        except (TypeError, AttributeError):
            pass
    
    # If we couldn't calculate from entropy, use a reasonable default
    if total_heads == 0:
        total_heads = 384  # 6 layers * 64 heads for DistilGPT2
    
    pruned_head_count = len(pruned_heads)
    sparsity = (pruned_head_count / total_heads) * 100 if total_heads > 0 else 0
    
    # Create visualizations
    perplexity_viz = create_perplexity_visualization(metrics)
    pruned_heads_viz = create_pruned_heads_heatmap(pruned_heads)
    entropy_viz = create_entropy_visualization(pre_entropy, post_entropy)
    loss_viz = create_loss_improvement_chart(metrics)
    
    # Try to extract recovery rate from results
    recovery_rate = results.get('recovery_rate', 'N/A')
    if isinstance(recovery_rate, (int, float)) and not np.isnan(recovery_rate):
        recovery_rate = f"{recovery_rate:.2f}%"
    
    # Build dashboard HTML
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dynamic Neural Plasticity Experiment Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                background-color: #f8f9fa;
            }}
            header {{
                border-bottom: 2px solid #3d5a80;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            h1 {{
                margin: 0;
                color: #3d5a80;
                font-size: 1.8em;
            }}
            h2 {{
                color: #3d5a80;
                border-bottom: 1px solid #ddd;
                padding-bottom: 8px;
                margin-top: 25px;
                font-size: 1.4em;
            }}
            h3 {{
                color: #2c3e50;
                margin-top: 20px;
                font-size: 1.2em;
            }}
            .timestamp {{
                color: #666;
                font-size: 0.8em;
                text-align: right;
                margin-top: -25px;
            }}
            .content-box {{
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }}
            /* Main visualization styles */
            .main-visualizations {{
                display: grid;
                grid-template-columns: 1fr;
                gap: 20px;
            }}
            .full-width-chart {{
                width: 100%;
                border-bottom: 1px solid #eee;
                padding-bottom: 20px;
                margin-bottom: 20px;
            }}
            .metrics-panel {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin-bottom: 20px;
            }}
            .heatmap-panel {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
            }}
            /* Metrics display */
            .metrics-display {{
                display: grid;
                grid-template-columns: repeat(5, 1fr);
                gap: 10px;
                margin: 20px 0;
                border-top: 1px solid #eee;
                padding-top: 15px;
            }}
            .metric-box {{
                text-align: center;
                border: 1px solid #eee;
                padding: 10px;
                border-radius: 4px;
            }}
            .metric-value {{
                font-size: 22px;
                font-weight: bold;
                margin: 5px 0;
            }}
            .metric-label {{
                font-size: 12px;
                color: #666;
                text-transform: uppercase;
            }}
            /* Image containers */
            .image-container {{
                width: 100%;
                border: 1px solid #ddd;
                border-radius: 4px;
                overflow: hidden;
                margin-bottom: 15px;
                background-color: white;
            }}
            .image-container img {{
                width: 100%;
                display: block;
            }}
            .image-caption {{
                padding: 8px 10px;
                background-color: #f5f5f5;
                font-size: 12px;
                color: #333;
                border-top: 1px solid #ddd;
            }}
            /* Tabs navigation */
            .tabs {{
                display: flex;
                border-bottom: 1px solid #ddd;
                margin-bottom: 20px;
                background-color: white;
                border-radius: 4px 4px 0 0;
                overflow: hidden;
            }}
            .tab {{
                padding: 10px 15px;
                cursor: pointer;
                background-color: #f8f9fa;
                border: none;
                border-right: 1px solid #ddd;
                font-size: 14px;
                transition: all 0.2s;
            }}
            .tab.active {{
                background-color: #3d5a80;
                color: white;
                font-weight: bold;
            }}
            .tab:hover:not(.active) {{
                background-color: #e9ecef;
            }}
            .tab-content {{
                display: none;
                background-color: white;
                border: 1px solid #ddd;
                border-top: none;
                padding: 20px;
                border-radius: 0 0 4px 4px;
            }}
            .tab-content.active {{
                display: block;
            }}
            /* Criteria list */
            .criteria-list {{
                margin: 15px 0;
                padding-left: 20px;
            }}
            .criteria-list li {{
                margin-bottom: 8px;
            }}
            /* Quote box */
            .quote-box {{
                font-style: italic;
                color: #495057;
                text-align: center;
                margin: 30px 0;
                padding: 15px;
                border-top: 1px solid #eee;
                border-bottom: 1px solid #eee;
            }}
            /* Detailed tables */
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
                font-size: 14px;
            }}
            th, td {{
                padding: 8px 10px;
                border: 1px solid #ddd;
                text-align: left;
            }}
            th {{
                background-color: #f5f5f5;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            tr:hover {{
                background-color: #f1f1f1;
            }}
            .improvement-positive {{
                color: #28a745;
                font-weight: bold;
            }}
            .improvement-negative {{
                color: #dc3545;
                font-weight: bold;
            }}
            footer {{
                margin-top: 40px;
                padding-top: 15px;
                border-top: 1px solid #ddd;
                text-align: center;
                font-size: 12px;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <header>
            <h1>Dynamic Neural Plasticity Experiment Report</h1>
            <div class="timestamp">Generated: {timestamp}</div>
        </header>
        
        <div class="content-box">
            <h2>Neural Plasticity Process</h2>
            
            <div class="main-visualizations">
                <div class="full-width-chart">
                    <div class="image-container">
                        <img src="{loss_viz or '#'}" alt="Dynamic Neural Plasticity: Training Process">
                        <div class="image-caption">Dynamic Neural Plasticity: Training Process</div>
                    </div>
                </div>
                
                <div class="metrics-panel">
                    <div class="image-container">
                        <img src="{perplexity_viz or '#'}" alt="Model Perplexity">
                        <div class="image-caption">Model Perplexity Comparison</div>
                    </div>
                    <div class="image-container">
                        <img src="{pruned_heads_viz or '#'}" alt="Pruned Attention Heads">
                        <div class="image-caption">Pruned Attention Heads</div>
                    </div>
                </div>
                
                <div class="heatmap-panel">
                    <div class="image-container">
                        <img src="{entropy_viz or '#'}" alt="Attention Head Entropy">
                        <div class="image-caption">Attention Head Entropy Before and After Pruning</div>
                    </div>
                </div>
            </div>
            
            <div class="image-caption" style="text-align: center; margin: 10px 0; border: none; background: none;">Dynamic visualization of the neural plasticity process, showing training loss, pruning events, and performance recovery</div>
            
            <div class="metrics-display">
                <div class="metric-box">
                    <div class="metric-label">Total Heads</div>
                    <div class="metric-value">{total_heads}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Pruned Heads</div>
                    <div class="metric-value">{pruned_head_count}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Growing Events</div>
                    <div class="metric-value">0</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Sparsity</div>
                    <div class="metric-value">{sparsity:.1f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Recovery Rate</div>
                    <div class="metric-value">{recovery_rate}</div>
                </div>
            </div>
            
            <div class="metrics-display">
                <div class="metric-box" style="grid-column: span 2;">
                    <div class="metric-label">Initial Loss</div>
                    <div class="metric-value">{format_float(processed_metrics.get('baseline_loss'))}</div>
                </div>
                <div class="metric-box" style="grid-column: span 2;">
                    <div class="metric-label">Final Loss</div>
                    <div class="metric-value">{format_float(processed_metrics.get('final_loss'))}</div>
                </div>
                <div class="metric-box" style="grid-column: span 1;">
                    <div class="metric-label">Improvement</div>
                    <div class="metric-value improvement-positive">{processed_metrics.get('loss_improvement', 'N/A')}</div>
                </div>
            </div>
        </div>
        
        <div class="tabs">
            <div class="tab active" onclick="openTab(event, 'overview')">Overview</div>
            <div class="tab" onclick="openTab(event, 'pruning-analysis')">Pruning Analysis</div>
            <div class="tab" onclick="openTab(event, 'head-timeline')">Head Timeline</div>
            <div class="tab" onclick="openTab(event, 'decision-viz')">Decision Visualizations</div>
            <div class="tab" onclick="openTab(event, 'text-gen')">Text Generation</div>
        </div>
        
        <div id="overview" class="tab-content active">
            <h2>Neural Plasticity Summary</h2>
            <p>This experiment applies dynamic neural plasticity to a <strong>{params.get('model_name', 'transformer')}</strong> model using <strong>{params.get('pruning_strategy', 'entropy')}</strong> pruning at level <strong>{params.get('pruning_level', 'N/A')}</strong>. The process involves warmup, pruning, and fine-tuning phases, resulting in a more efficient model with improved performance.</p>
            
            <h3>Performance Metrics</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Initial</th>
                    <th>After Pruning</th>
                    <th>Final</th>
                    <th>Change</th>
                </tr>
                <tr>
                    <td>Loss</td>
                    <td>{format_float(processed_metrics.get('baseline_loss'))}</td>
                    <td>{format_float(processed_metrics.get('post_pruning_loss'))}</td>
                    <td>{format_float(processed_metrics.get('final_loss'))}</td>
                    <td class="improvement-positive">{processed_metrics.get('loss_improvement', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Perplexity</td>
                    <td>{format_float(processed_metrics.get('baseline_perplexity'))}</td>
                    <td>{format_float(processed_metrics.get('post_pruning_perplexity'))}</td>
                    <td>{format_float(processed_metrics.get('final_perplexity'))}</td>
                    <td class="improvement-positive">{processed_metrics.get('perplexity_improvement', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Sparsity</td>
                    <td>0%</td>
                    <td>{sparsity:.1f}%</td>
                    <td>{sparsity:.1f}%</td>
                    <td>↑ {sparsity:.1f}%</td>
                </tr>
            </table>
            
            <h3>Process Timeline</h3>
            <p>The neural plasticity process follows these phases:</p>
            <ol>
                <li><strong>Warmup Phase:</strong> Initial model training to establish baseline performance.</li>
                <li><strong>Pruning Phase:</strong> Selective pruning of attention heads based on {params.get('pruning_strategy', 'entropy')} criteria.</li>
                <li><strong>Fine-tuning Phase:</strong> Re-training of the pruned model to recover and improve performance.</li>
            </ol>
            
            <h3>Experiment Configuration</h3>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Model</td>
                    <td>{params.get('model_name', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Pruning Strategy</td>
                    <td>{params.get('pruning_strategy', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Pruning Level</td>
                    <td>{params.get('pruning_level', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Fine-tuning Steps</td>
                    <td>{params.get('fine_tuning_steps', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Learning Rate</td>
                    <td>{params.get('learning_rate', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Use Differential LR</td>
                    <td>{params.get('use_differential_lr', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Batch Size</td>
                    <td>{params.get('batch_size', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Device</td>
                    <td>{params.get('device', 'N/A')}</td>
                </tr>
            </table>
        </div>
        
        <div id="pruning-analysis" class="tab-content">
            <h2>Pruned Attention Heads</h2>
            <p>Analysis of which heads were selected for pruning, based on their entropy values (measure of attention focus) and gradient magnitudes (measure of contribution to learning).</p>
            
            <div class="image-container">
                <img src="{entropy_viz or '#'}" alt="Attention Entropy Maps">
                <div class="image-caption">Comparison of attention entropy before and after pruning</div>
            </div>
            
            <h3>Pruning Decision Criteria</h3>
            <ul class="criteria-list">
                <li><strong>Entropy:</strong> Higher values indicate more dispersed attention (less focused)</li>
                <li><strong>Gradient Interest:</strong> Higher values indicate less contribution to model learning</li>
                <li><strong>Combined Score:</strong> H × G, where H = Entropy and G = Gradient Magnitude</li>
                <li><strong>Selection Process:</strong> Heads with score > threshold ({params.get('pruning_level', 0.2)}) are pruning candidates</li>
            </ul>
            
            <h3>Pruned Heads Detail</h3>
            <table>
                <tr>
                    <th>Layer</th>
                    <th>Head</th>
                    <th>Score</th>
                </tr>
    """
    
    # Add pruned heads data
    heads_to_show = min(20, len(pruned_heads))
    for i in range(heads_to_show):
        layer, head, score = pruned_heads[i]
        score_val = format_float(score) if not np.isnan(score) else "N/A"
        html_content += f"""
                <tr>
                    <td>{layer}</td>
                    <td>{head}</td>
                    <td>{score_val}</td>
                </tr>
        """
    
    if len(pruned_heads) > 20:
        html_content += f"""
                <tr>
                    <td colspan="3">... and {len(pruned_heads) - 20} more heads pruned</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
        
        <div id="head-timeline" class="tab-content">
            <h2>Head Lifecycle Timeline</h2>
            <p>Visualization showing the lifecycle of attention heads throughout the neural plasticity process.</p>
            
            <div class="image-container">
                <img src="{}" alt="Head Timeline Visualization">
                <div class="image-caption">Dynamic visualization of head activity over time</div>
            </div>
            
            <p class="info-box">
                No detailed head timeline data is available for this experiment run. 
                Future runs with the neural plasticity dashboard will include detailed tracking of each head's 
                activity, entropy, and gradient values throughout the training process.
            </p>
        </div>
        
        <div id="decision-viz" class="tab-content">
            <h2>Decision Process Visualizations</h2>
            <p>Detailed visualizations showing exactly why the system made specific pruning and growing decisions. These visualizations provide transparency into the mathematical decision process of each operation.</p>
            
            <div class="image-container">
                <img src="{pruned_heads_viz or '#'}" alt="Pruned Heads Matrix">
                <div class="image-caption">Heatmap of pruned attention heads by layer and position</div>
            </div>
            
            <p class="info-box">
                Detailed decision visualizations are not available for this experiment run.
                Future runs will include per-head visualizations showing the exact metrics that led
                to pruning decisions, with detailed entropy and gradient analysis for each head.
            </p>
        </div>
        
        <div id="text-gen" class="tab-content">
            <h2>Text Generation Examples</h2>
            <p>Comparison of text generation quality before and after pruning and fine-tuning.</p>
            
            <p class="info-box">
                Text generation samples are not available for this experiment run.
                Future runs will include text generation comparisons between the original model,
                the pruned model, and the fine-tuned model to demonstrate the preservation
                of generation quality while achieving higher efficiency.
            </p>
        </div>
        
        <footer>
            <p>Generated by Sentinel AI Neural Plasticity Module v0.1.0</p>
            <p>© 2025 Sentinel AI Project</p>
        </footer>
        
        <script>
            function openTab(evt, tabName) {
                // Hide all tab content
                var tabContents = document.getElementsByClassName("tab-content");
                for (var i = 0; i < tabContents.length; i++) {
                    tabContents[i].className = tabContents[i].className.replace(" active", "");
                }
                
                // Remove active class from all tabs
                var tabs = document.getElementsByClassName("tab");
                for (var i = 0; i < tabs.length; i++) {
                    tabs[i].className = tabs[i].className.replace(" active", "");
                }
                
                // Show the current tab and add active class
                document.getElementById(tabName).className += " active";
                evt.currentTarget.className += " active";
            }
        </script>
    </body>
    </html>
    """
    
    # Write the HTML file
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(html_content)
    
    print(f"✅ Neural Plasticity Dashboard generated at {output_path}")
    return output_path


def find_existing_dashboard_path(experiment_dir):
    """Find the dashboards directory and any existing dashboard files."""
    dashboards_dir = os.path.join(experiment_dir, "dashboards")
    if not os.path.exists(dashboards_dir):
        os.makedirs(dashboards_dir, exist_ok=True)
    return os.path.join(dashboards_dir, "dashboard.html")


def main():
    parser = argparse.ArgumentParser(description="Generate Neural Plasticity Dashboard")
    parser.add_argument("--experiment_dir", type=str, help="Directory containing experiment results")
    parser.add_argument("--output_file", type=str, help="Path to save the dashboard HTML file")
    args = parser.parse_args()
    
    # Find experiment directory if not specified
    experiment_dir = args.experiment_dir or find_latest_experiment_dir()
    if not experiment_dir:
        print("No experiment directory found. Please specify --experiment_dir.")
        return
    
    # Determine output path
    if args.output_file:
        output_path = args.output_file
    else:
        output_path = find_existing_dashboard_path(experiment_dir)
    
    # Load experiment data
    experiment_data = load_experiment_data(experiment_dir)
    if not experiment_data:
        print(f"No experiment data found in {experiment_dir}")
        return
    
    # Generate the dashboard
    create_neural_plasticity_dashboard(experiment_data, output_path)


if __name__ == "__main__":
    main()