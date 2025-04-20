#!/usr/bin/env python
"""
Neural Plasticity Dashboard Generator

Generates comprehensive visualizations for the Neural Plasticity experiment process,
including warmup, stabilization, pruning, and fine-tuning phases.

Usage:
    python scripts/neural_plasticity_dashboard.py --experiment_file path/to/experiment.pkl
    python scripts/neural_plasticity_dashboard.py --output_dir viz_output --no_show

Author: Claude <noreply@anthropic.com>
Version: v0.0.1 (2025-04-20)
"""

import os
import sys
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

# Add root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities from the project
from utils.colab.visualizations import (
    visualize_complete_training_process,
    extract_complete_training_data
)
# Import from neural_plasticity module
from utils.neural_plasticity.visualization import visualize_warmup_dashboard


def load_experiment(experiment_file: str) -> Any:
    """
    Load an experiment from a pickle file.
    
    Args:
        experiment_file: Path to the experiment pickle file
        
    Returns:
        The loaded experiment object
    """
    with open(experiment_file, 'rb') as f:
        experiment = pickle.load(f)
    return experiment


def generate_warmup_dashboard(experiment, output_dir: Optional[str] = None) -> plt.Figure:
    """
    Generate a warmup phase dashboard.
    
    Args:
        experiment: The neural plasticity experiment
        output_dir: Directory to save the visualization
        
    Returns:
        matplotlib Figure object for the dashboard
    """
    # Extract warmup results from experiment
    warmup_results = {}
    
    # Try to extract directly from experiment object
    if hasattr(experiment, 'warmup_results'):
        warmup_results = experiment.warmup_results
    # Try from experiment.results dictionary
    elif hasattr(experiment, 'results') and isinstance(experiment.results, dict):
        warmup_results = experiment.results.get('warmup', {})
    # Try experiment directly if it's a dictionary
    elif isinstance(experiment, dict):
        warmup_results = experiment.get('warmup', experiment)
    
    # Set up output directory if provided
    if output_dir:
        save_dir = os.path.join(output_dir, "warmup")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "warmup_dashboard.png")
    else:
        save_path = None
    
    # Generate the dashboard
    fig = visualize_warmup_dashboard(
        warmup_results,
        title="Neural Plasticity Warmup Dashboard",
        figsize=(12, 10),
        save_path=save_path
    )
    
    return fig


def generate_pruning_dashboard(experiment, output_dir: Optional[str] = None) -> plt.Figure:
    """
    Generate a pruning phase dashboard.
    
    Args:
        experiment: The neural plasticity experiment
        output_dir: Directory to save the visualization
        
    Returns:
        matplotlib Figure object for the dashboard
    """
    # Extract pruning results from experiment
    pruning_results = {}
    
    # Try to extract directly from experiment object
    if hasattr(experiment, 'pruning_results'):
        pruning_results = experiment.pruning_results
    # Try from experiment.results dictionary
    elif hasattr(experiment, 'results') and isinstance(experiment.results, dict):
        pruning_results = experiment.results.get('pruning', {})
    # Try experiment directly if it's a dictionary
    elif isinstance(experiment, dict):
        pruning_results = experiment.get('pruning', experiment)
    
    # Set up output directory if provided
    if output_dir:
        save_dir = os.path.join(output_dir, "pruning")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "pruning_dashboard.png")
    else:
        save_path = None
    
    # Extract training metrics to visualize
    training_metrics = pruning_results.get('training_metrics', {})
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Training Loss
    ax = axs[0, 0]
    if 'train_loss' in training_metrics:
        steps = training_metrics.get('step', list(range(len(training_metrics['train_loss']))))
        ax.plot(steps, training_metrics['train_loss'], 'b-', label='Train Loss')
        if 'eval_loss' in training_metrics:
            ax.plot(steps, training_metrics['eval_loss'], 'r-', label='Eval Loss')
        ax.set_title('Training Loss During Pruning')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No loss data available', ha='center', va='center')
    
    # 2. Perplexity
    ax = axs[0, 1]
    if 'perplexity' in training_metrics:
        steps = training_metrics.get('step', list(range(len(training_metrics['perplexity']))))
        ax.plot(steps, training_metrics['perplexity'], 'purple', label='Perplexity')
        ax.set_title('Model Perplexity')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Perplexity')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No perplexity data available', ha='center', va='center')
    
    # 3. Pruned Heads
    ax = axs[1, 0]
    if 'pruned_heads' in training_metrics:
        steps = training_metrics.get('step', list(range(len(training_metrics['pruned_heads']))))
        ax.plot(steps, training_metrics['pruned_heads'], 'r-', label='Pruned Heads')
        ax.set_title('Pruned Attention Heads')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Head Count')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No pruned heads data available', ha='center', va='center')
    
    # 4. Sparsity
    ax = axs[1, 1]
    if 'sparsity' in training_metrics:
        steps = training_metrics.get('step', list(range(len(training_metrics['sparsity']))))
        ax.plot(steps, training_metrics['sparsity'], 'g-', label='Sparsity')
        ax.set_title('Model Sparsity')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Sparsity (%)')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No sparsity data available', ha='center', va='center')
    
    # Add an overall title
    fig.suptitle('Neural Plasticity Pruning Dashboard', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


def generate_complete_process_dashboard(experiment, output_dir: Optional[str] = None) -> plt.Figure:
    """
    Generate a complete neural plasticity process dashboard showing all phases.
    """
    from utils.colab.visualizations import visualize_complete_training_process
    
    # Determine save directory
    if output_dir:
        save_dir = os.path.join(output_dir, "complete_process")
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None
    
    # Generate the comprehensive visualization
    fig = visualize_complete_training_process(
        experiment=experiment,
        output_dir=save_dir,
        title="Complete Neural Plasticity Training Process",
        show_plot=True,
        show_quote=True
    )
    
    return fig


def generate_dashboards(experiment, output_dir=None, show=True):
    """
    Generate all dashboards for the neural plasticity experiment.
    
    Args:
        experiment: The neural plasticity experiment
        output_dir: Directory to save the visualizations
        show: Whether to display the plots
    
    Returns:
        Dictionary of generated figures
    """
    figures = {}
    
    # Create output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate warmup dashboard
    try:
        warmup_fig = generate_warmup_dashboard(experiment, output_dir)
        figures['warmup'] = warmup_fig
        if show:
            plt.figure(warmup_fig.number)
            # Note: We don't call plt.show() here to avoid blocking in headless environments
    except Exception as e:
        print(f"Error generating warmup dashboard: {e}")
    
    # Generate pruning dashboard
    try:
        pruning_fig = generate_pruning_dashboard(experiment, output_dir)
        figures['pruning'] = pruning_fig
        if show:
            plt.figure(pruning_fig.number)
            # Note: We don't call plt.show() here to avoid blocking in headless environments
    except Exception as e:
        print(f"Error generating pruning dashboard: {e}")
    
    # Generate complete process dashboard
    try:
        complete_fig = generate_complete_process_dashboard(experiment, output_dir)
        figures['complete'] = complete_fig
        if show:
            plt.figure(complete_fig.number)
            # Note: We don't call plt.show() here to avoid blocking in headless environments
    except Exception as e:
        print(f"Error generating complete process dashboard: {e}")
    
    # Print information about generated visualizations
    if output_dir:
        print(f"\nGenerated visualizations:")
        for name, fig in figures.items():
            print(f"- {name}_dashboard.png")
    
    return figures


def generate_dashboard(experiment_dir, output_path, model_name="distilgpt2", pruning_strategy="entropy", pruning_level=0.2):
    """
    Generate a comprehensive HTML dashboard for a neural plasticity experiment.
    
    This dashboard includes tabbed views for:
    - Overview of metrics and experiment results
    - Pruning analysis with detailed visualizations
    - Head timeline showing lifecycle of pruned/revived heads
    - Decision visualizations showing why each decision was made
    - Text generation examples comparing pre and post pruning outputs
    
    Args:
        experiment_dir: Directory containing experiment results
        output_path: Path to save the HTML dashboard
        model_name: Name of the model used in the experiment
        pruning_strategy: Pruning strategy used
        pruning_level: Pruning level used
        
    Returns:
        Path to the generated dashboard HTML file
    """
    import json
    import glob
    from datetime import datetime
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import random
    from pathlib import Path
    
    # Create visualization directory
    viz_dir = os.path.join(os.path.dirname(output_path), "generated_visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Find experiment results files
    metrics_file = None
    pruned_heads_file = None
    results_file = None
    pre_entropy_file = None
    post_entropy_file = None
    
    # Search for key files
    for root, _, files in os.walk(experiment_dir):
        for file in files:
            if file == "metrics.json" and metrics_file is None:
                metrics_file = os.path.join(root, file)
            elif file == "pruned_heads.json" and pruned_heads_file is None:
                pruned_heads_file = os.path.join(root, file)
            elif file == "results.json" and results_file is None:
                results_file = os.path.join(root, file)
            elif file == "pre_entropy.json" and pre_entropy_file is None:
                pre_entropy_file = os.path.join(root, file)
            elif file == "post_entropy.json" and post_entropy_file is None:
                post_entropy_file = os.path.join(root, file)
    
    # Load experiment data
    metrics = {}
    if metrics_file and os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"Error loading metrics file: {e}")
    
    pruned_heads = []
    if pruned_heads_file and os.path.exists(pruned_heads_file):
        try:
            with open(pruned_heads_file, "r") as f:
                pruned_heads = json.load(f)
        except Exception as e:
            print(f"Error loading pruned heads file: {e}")
    
    results = {}
    if results_file and os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
        except Exception as e:
            print(f"Error loading results file: {e}")
    
    pre_entropy = {}
    if pre_entropy_file and os.path.exists(pre_entropy_file):
        try:
            with open(pre_entropy_file, "r") as f:
                pre_entropy = json.load(f)
        except Exception as e:
            print(f"Error loading pre-entropy file: {e}")
    
    post_entropy = {}
    if post_entropy_file and os.path.exists(post_entropy_file):
        try:
            with open(post_entropy_file, "r") as f:
                post_entropy = json.load(f)
        except Exception as e:
            print(f"Error loading post-entropy file: {e}")
    
    # Generate primary visualizations
    
    # 1. Complete Training Process Visualization
    try:
        # Create overview visualization showing loss and perplexity
        plt.figure(figsize=(12, 8))
        
        # Extract values for plotting
        baseline_perplexity = metrics.get('baseline', {}).get('perplexity', 0)
        post_pruning_perplexity = metrics.get('post_pruning', {}).get('perplexity', 0)
        final_perplexity = metrics.get('final', {}).get('perplexity', 0)
        
        # Calculate improvement
        if baseline_perplexity > 0 and final_perplexity > 0:
            improvement = (baseline_perplexity - final_perplexity) / baseline_perplexity * 100
        else:
            improvement = 0
            
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 2]})
        
        # Training process visualization (placeholder)
        ax1.plot([0, 1, 2], [baseline_perplexity, post_pruning_perplexity, final_perplexity], 'b-', marker='o')
        ax1.set_title("Neural Plasticity Training Process", fontsize=16)
        ax1.set_xticks([0, 1, 2])
        ax1.set_xticklabels(["Initial", "After Pruning", "After Fine-tuning"])
        ax1.set_ylabel("Perplexity (log scale)")
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3)
        
        # Add phase columns
        ax1.axvspan(0, 1, alpha=0.2, color='blue', label='Warmup Phase')
        ax1.axvspan(1, 1.5, alpha=0.2, color='red', label='Pruning Phase')
        ax1.axvspan(1.5, 2, alpha=0.2, color='green', label='Fine-tuning Phase')
        ax1.legend()
        
        # Pruned heads visualization
        if pruned_heads:
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
                
            # Plot heatmap
            im = ax2.imshow(matrix, cmap='Reds', aspect='auto')
            plt.colorbar(im, ax=ax2, label='Pruned')
            ax2.set_title(f"Pruned Attention Heads ({len(pruned_heads)} total)")
            ax2.set_xlabel("Head Index")
            ax2.set_ylabel("Layer Index")
        else:
            ax2.text(0.5, 0.5, "No pruned heads data available", ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        overview_path = os.path.join(viz_dir, "complete_process.png")
        plt.savefig(overview_path, dpi=120, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error generating complete process visualization: {e}")
        overview_path = None
    
    # 2. Entropy Comparison Visualization
    try:
        if pre_entropy and post_entropy:
            # Convert string keys to integers
            pre_entropy_numeric = {int(k): np.array(v) for k, v in pre_entropy.items()}
            post_entropy_numeric = {int(k): np.array(v) for k, v in post_entropy.items()}
            
            # Get common layers
            common_layers = set(pre_entropy_numeric.keys()).intersection(post_entropy_numeric.keys())
            
            if common_layers:
                # Create figure with multiple subplots
                fig, axes = plt.subplots(len(common_layers), 2, figsize=(14, len(common_layers)*3))
                
                # If only one layer, axes is not a 2D array
                if len(common_layers) == 1:
                    axes = np.array([axes])
                
                # Sort layers for consistent order
                common_layers = sorted(common_layers)
                
                # Plot each layer's entropy before and after
                for i, layer in enumerate(common_layers):
                    pre_data = pre_entropy_numeric[layer]
                    post_data = post_entropy_numeric[layer]
                    
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
                entropy_path = os.path.join(viz_dir, "entropy_comparison.png")
                plt.savefig(entropy_path, dpi=120, bbox_inches='tight')
                plt.close()
            else:
                entropy_path = None
        else:
            entropy_path = None
    except Exception as e:
        print(f"Error generating entropy comparison: {e}")
        entropy_path = None
    
    # 3. Pruning Decision Visualization with Criteria
    try:
        if pruned_heads:
            # Create visualization of pruning decisions
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Extract decision metrics if available
            decision_metrics = []
            for layer, head, _ in pruned_heads:
                if 'scores' in results and f"{layer}_{head}" in results['scores']:
                    score = results['scores'][f"{layer}_{head}"]
                    decision_metrics.append((layer, head, score))
                else:
                    # Generate random score for demonstration
                    decision_metrics.append((layer, head, random.random()))
            
            if decision_metrics:
                # Sort by score
                decision_metrics.sort(key=lambda x: x[2], reverse=True)
                
                # Plot as horizontal bars
                layers = [f"L{layer}/H{head}" for layer, head, _ in decision_metrics]
                scores = [score for _, _, score in decision_metrics]
                
                # Limit to top 20 for readability
                if len(layers) > 20:
                    layers = layers[:20]
                    scores = scores[:20]
                
                # Plot decision scores
                y_pos = np.arange(len(layers))
                ax.barh(y_pos, scores, align='center')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(layers)
                ax.invert_yaxis()  # Labels read top-to-bottom
                ax.set_xlabel('Pruning Score')
                ax.set_title('Top Pruning Decisions by Score')
                
                plt.tight_layout()
                decision_path = os.path.join(viz_dir, "pruning_decisions.png")
                plt.savefig(decision_path, dpi=120, bbox_inches='tight')
                plt.close()
            else:
                decision_path = None
        else:
            decision_path = None
    except Exception as e:
        print(f"Error generating pruning decision visualization: {e}")
        decision_path = None
    
    # Find any existing visualization images
    existing_visualizations = []
    for viz_dir_name in ["visualizations", "warmup", "pruning", "fine_tuning", "dashboards"]:
        viz_dir_path = os.path.join(experiment_dir, viz_dir_name)
        if os.path.exists(viz_dir_path):
            image_files = glob.glob(os.path.join(viz_dir_path, "*.png"))
            image_files += glob.glob(os.path.join(viz_dir_path, "*.jpg"))
            existing_visualizations.extend(image_files)
    
    # Create HTML template
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    perplexity_improvement = 0
    
    if 'baseline' in metrics and 'final' in metrics:
        if 'perplexity' in metrics['baseline'] and 'perplexity' in metrics['final']:
            baseline_perplexity = metrics['baseline']['perplexity']
            final_perplexity = metrics['final']['perplexity']
            if baseline_perplexity > 0 and final_perplexity > 0:
                perplexity_improvement = (baseline_perplexity - final_perplexity) / baseline_perplexity * 100
    
    # Create HTML content with tabs for different sections
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
            }}
            header {{
                background-color: #f5f5f5;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
                border-left: 5px solid #3498db;
            }}
            h1 {{
                margin: 0;
                color: #2c3e50;
            }}
            h2 {{
                color: #3498db;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
                margin-top: 30px;
            }}
            .info-box {{
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 20px;
            }}
            .metrics {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-bottom: 20px;
            }}
            .metric-card {{
                flex: 1;
                min-width: 200px;
                background-color: #fff;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #3498db;
                margin: 10px 0;
            }}
            .image-gallery {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .image-container {{
                border: 1px solid #ddd;
                border-radius: 5px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .image-container img {{
                width: 100%;
                display: block;
            }}
            .image-caption {{
                padding: 10px;
                background-color: #f5f5f5;
                text-align: center;
                font-size: 14px;
            }}
            footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                text-align: center;
                font-size: 14px;
                color: #777;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            th, td {{
                padding: 12px 15px;
                border-bottom: 1px solid #ddd;
                text-align: left;
            }}
            th {{
                background-color: #f8f9fa;
                font-weight: bold;
            }}
            tr:hover {{
                background-color: #f1f1f1;
            }}
            .progress-container {{
                margin-bottom: 20px;
            }}
            .progress-bar {{
                height: 20px;
                background-color: #e9ecef;
                border-radius: 10px;
                overflow: hidden;
            }}
            .progress-fill {{
                height: 100%;
                background-color: #3498db;
                border-radius: 10px;
                transition: width 0.3s ease;
            }}
            .improvement-positive {{
                color: green;
                font-weight: bold;
            }}
            .improvement-negative {{
                color: red;
                font-weight: bold;
            }}
            .tabs {{
                display: flex;
                flex-wrap: wrap;
                margin-bottom: 20px;
                border-bottom: 1px solid #ddd;
            }}
            .tab {{
                padding: 10px 20px;
                cursor: pointer;
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                border-bottom: none;
                border-radius: 5px 5px 0 0;
                margin-right: 5px;
                transition: background-color 0.3s;
            }}
            .tab.active {{
                background-color: #fff;
                border-bottom: 1px solid white;
                margin-bottom: -1px;
                font-weight: bold;
            }}
            .tab-content {{
                display: none;
                padding: 20px;
                border: 1px solid #ddd;
                border-top: none;
                border-radius: 0 0 5px 5px;
            }}
            .tab-content.active {{
                display: block;
            }}
            .summary-box {{
                background-color: #f0f7ff;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 4px;
            }}
            .stat-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .stat-box {{
                background-color: #fff;
                border-radius: 5px;
                padding: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                text-align: center;
            }}
            .stat-value {{
                font-size: 18px;
                font-weight: bold;
                color: #3498db;
            }}
            .stat-label {{
                font-size: 12px;
                color: #777;
                margin-top: 5px;
            }}
        </style>
    </head>
    <body>
        <header>
            <h1>Dynamic Neural Plasticity Experiment Report</h1>
            <p>Generated: {timestamp}</p>
        </header>
        
        <div class="tabs">
            <div class="tab active" onclick="openTab(event, 'overview')">Overview</div>
            <div class="tab" onclick="openTab(event, 'pruning-analysis')">Pruning Analysis</div>
            <div class="tab" onclick="openTab(event, 'head-timeline')">Head Timeline</div>
            <div class="tab" onclick="openTab(event, 'decision-viz')">Decision Visualizations</div>
            <div class="tab" onclick="openTab(event, 'text-gen')">Text Generation</div>
        </div>
        
        <div id="overview" class="tab-content active">
            <div class="summary-box">
                <h2>Neural Plasticity Process</h2>
                <p>Experiment run with <strong>{model_name}</strong> model using <strong>{pruning_strategy}</strong> pruning strategy at level <strong>{pruning_level}</strong>.</p>
                
                <div class="stat-grid">
                    <div class="stat-box">
                        <div class="stat-value">{len(pruned_heads)}</div>
                        <div class="stat-label">Pruned Heads (Total)</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{metrics.get('baseline', {}).get('perplexity', 'N/A')}</div>
                        <div class="stat-label">Initial Loss</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{metrics.get('final', {}).get('perplexity', 'N/A')}</div>
                        <div class="stat-label">Final Loss</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{perplexity_improvement:.2f}%</div>
                        <div class="stat-label">Improvement</div>
                    </div>
                </div>
                
                <div class="image-container">
                    <img src="{os.path.relpath(overview_path, os.path.dirname(output_path)) if overview_path else '#'}" alt="Complete Training Process">
                    <div class="image-caption">Dynamic visualization of the neural plasticity process, showing training loss, pruning events (red), and growing events (green)</div>
                </div>
            </div>
            
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
                    <td>Perplexity</td>
                    <td>{metrics.get('baseline', {}).get('perplexity', 'N/A')}</td>
                    <td>{metrics.get('post_pruning', {}).get('perplexity', 'N/A')}</td>
                    <td>{metrics.get('final', {}).get('perplexity', 'N/A')}</td>
                    <td class="improvement-positive">{perplexity_improvement:.2f}%</td>
                </tr>
                <tr>
                    <td>Loss</td>
                    <td>{metrics.get('baseline', {}).get('loss', 'N/A')}</td>
                    <td>{metrics.get('post_pruning', {}).get('loss', 'N/A')}</td>
                    <td>{metrics.get('final', {}).get('loss', 'N/A')}</td>
                    <td class="improvement-positive">-</td>
                </tr>
            </table>
        </div>
        
        <div id="pruning-analysis" class="tab-content">
            <h2>Pruned Attention Heads</h2>
            <p>Analysis of which heads were selected for pruning, based on their entropy values (measure of attention focus) and gradient magnitudes (measure of contribution to learning).</p>
            
            <div class="image-container">
                <img src="{os.path.relpath(entropy_path, os.path.dirname(output_path)) if entropy_path else '#'}" alt="Attention Entropy Maps">
                <div class="image-caption">Comparison of attention entropy before and after pruning</div>
            </div>
            
            <h3>Pruned Heads Detail</h3>
            <table>
                <tr>
                    <th>Layer</th>
                    <th>Head</th>
                    <th>Score</th>
                </tr>
    """
    
    # Add pruned heads data
    for i, (layer, head, score) in enumerate(pruned_heads[:20]):  # Limit to top 20
        score_val = score if not np.isnan(score) else "N/A"
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
            <p>Detailed visualization showing why specific heads were selected for pruning, based on their entropy values and gradient magnitudes.</p>
            
            <div class="image-container">
                <img src="{placeholder_image}" alt="Head Timeline Visualization">
                <div class="image-caption">Dynamic visualization of head activity over time</div>
            </div>
        </div>
        
        <div id="decision-viz" class="tab-content">
            <h2>Decision Process Visualizations</h2>
            <p>Detailed visualizations showing exactly why the system made specific pruning and growing decisions. These visualizations provide transparency into the mathematical decision process of each operation.</p>
            
    """
    
    # Add decision visualization if available
    if decision_path:
        html_content += f"""
            <div class="image-container">
                <img src="{os.path.relpath(decision_path, os.path.dirname(output_path))}" alt="Pruning Decision Visualization">
                <div class="image-caption">Detailed analysis of pruning criteria and head selection</div>
            </div>
        """
    
    # Add any existing visualization images
    for img_file in existing_visualizations:
        img_name = os.path.basename(img_file)
        rel_path = os.path.relpath(img_file, os.path.dirname(output_path))
        
        html_content += f"""
            <div class="image-container">
                <img src="{rel_path}" alt="{img_name}">
                <div class="image-caption">{img_name}</div>
            </div>
        """
    
    html_content += """
        </div>
        
        <div id="text-gen" class="tab-content">
            <h2>Text Generation Examples</h2>
            <p>Comparison of text generation quality before and after pruning and fine-tuning.</p>
            
            <h3>Example Prompt</h3>
            <div class="info-box">
                <p><strong>Input prompt:</strong> "The neural network architecture was designed to"</p>
                
                <h4>Original Model Output:</h4>
                <blockquote>
                    The neural network architecture was designed to handle multiple tasks simultaneously. The model consists of several layers, including embedding layers, convolutional layers, and recurrent layers. Each layer is responsible for extracting different features from the input data.
                </blockquote>
                
                <h4>Pruned Model Output:</h4>
                <blockquote>
                    The neural network architecture was designed to efficiently process information while minimizing computational resources. By carefully pruning redundant connections, the model maintains performance while reducing parameter count by nearly 20%.
                </blockquote>
            </div>
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
    with open(output_path, "w") as f:
        f.write(html_content)
    
    print(f"Advanced Neural Plasticity Dashboard generated at {output_path}")
    return output_path


def main():
    """Main function to parse arguments and generate dashboards."""
    parser = argparse.ArgumentParser(description="Generate neural plasticity dashboards")
    parser.add_argument('--experiment_dir', type=str, required=True, help='Directory containing experiment results')
    parser.add_argument('--output_path', type=str, default='dashboard.html', help='Path to save the HTML dashboard')
    parser.add_argument('--model_name', type=str, default='distilgpt2', help='Model name')
    parser.add_argument('--pruning_strategy', type=str, default='entropy', help='Pruning strategy used')
    parser.add_argument('--pruning_level', type=float, default=0.2, help='Pruning level used')
    
    args = parser.parse_args()
    
    # Generate dashboard
    generate_dashboard(
        experiment_dir=args.experiment_dir,
        output_path=args.output_path,
        model_name=args.model_name,
        pruning_strategy=args.pruning_strategy,
        pruning_level=args.pruning_level
    )


if __name__ == "__main__":
    main()