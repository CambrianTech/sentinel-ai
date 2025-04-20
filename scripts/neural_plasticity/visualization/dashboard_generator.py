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
    Generate a simple HTML dashboard for a neural plasticity experiment.
    
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
    
    # Create a simple HTML dashboard
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Neural Plasticity Dashboard</title>
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
                grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
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
        </style>
    </head>
    <body>
        <header>
            <h1>Neural Plasticity Experiment Dashboard</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
        
        <section class="info-box">
            <h2>Experiment Configuration</h2>
            <div class="metrics">
                <div class="metric-card">
                    <h3>Model</h3>
                    <div class="metric-value">{model_name}</div>
                </div>
                <div class="metric-card">
                    <h3>Pruning Strategy</h3>
                    <div class="metric-value">{pruning_strategy}</div>
                </div>
                <div class="metric-card">
                    <h3>Pruning Level</h3>
                    <div class="metric-value">{pruning_level}</div>
                </div>
            </div>
        </section>
    """
    
    # Try to find metrics.json file
    metrics_file = os.path.join(experiment_dir, "metrics.json")
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            
            # Add metrics section
            html_content += """
        <section>
            <h2>Performance Metrics</h2>
            <div class="metrics">
            """
            
            # Baseline metrics
            baseline = metrics.get("baseline", {})
            if baseline:
                html_content += f"""
                <div class="metric-card">
                    <h3>Baseline Perplexity</h3>
                    <div class="metric-value">{baseline.get("perplexity", "N/A")}</div>
                </div>
                """
            
            # Final metrics
            final = metrics.get("final", {})
            if final:
                html_content += f"""
                <div class="metric-card">
                    <h3>Final Perplexity</h3>
                    <div class="metric-value">{final.get("perplexity", "N/A")}</div>
                </div>
                """
            
            # Improvement
            if baseline and final and "perplexity" in baseline and "perplexity" in final:
                baseline_perplexity = baseline["perplexity"]
                final_perplexity = final["perplexity"]
                if baseline_perplexity > 0 and final_perplexity > 0:
                    improvement = (baseline_perplexity - final_perplexity) / baseline_perplexity * 100
                    html_content += f"""
                    <div class="metric-card">
                        <h3>Perplexity Improvement</h3>
                        <div class="metric-value">{improvement:.2f}%</div>
                    </div>
                    """
            
            html_content += """
            </div>
        </section>
            """
        except Exception as e:
            print(f"Error loading metrics file: {e}")
    
    # Find visualization images
    html_content += """
        <section>
            <h2>Visualizations</h2>
            <div class="image-gallery">
    """
    
    # Search for visualization images
    visualization_dirs = [
        os.path.join(experiment_dir, "visualizations"),
        os.path.join(experiment_dir, "warmup"),
        os.path.join(experiment_dir, "pruning"),
        os.path.join(experiment_dir, "fine_tuning"),
        os.path.join(experiment_dir, "dashboards")
    ]
    
    found_images = False
    for viz_dir in visualization_dirs:
        if os.path.exists(viz_dir):
            image_files = glob.glob(os.path.join(viz_dir, "*.png"))
            image_files += glob.glob(os.path.join(viz_dir, "*.jpg"))
            
            for img_file in image_files:
                img_name = os.path.basename(img_file)
                # Make a relative path from output_path to img_file
                rel_path = os.path.relpath(img_file, os.path.dirname(output_path))
                html_content += f"""
                <div class="image-container">
                    <img src="{rel_path}" alt="{img_name}">
                    <div class="image-caption">{img_name}</div>
                </div>
                """
                found_images = True
    
    if not found_images:
        html_content += """
            <p>No visualization images found.</p>
        """
    
    html_content += """
            </div>
        </section>
        
        <footer>
            <p>Generated by Sentinel AI Neural Plasticity Module v0.1.0</p>
            <p>© 2025 Sentinel AI Project</p>
        </footer>
    </body>
    </html>
    """
    
    # Write the HTML file
    with open(output_path, "w") as f:
        f.write(html_content)
    
    print(f"Dashboard generated at {output_path}")
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