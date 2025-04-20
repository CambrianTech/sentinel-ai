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


def main():
    """Main function to parse arguments and generate dashboards."""
    parser = argparse.ArgumentParser(description="Generate neural plasticity dashboards")
    parser.add_argument('--experiment_file', type=str, help='Path to experiment pickle file')
    parser.add_argument('--output_dir', type=str, default='viz_output', help='Directory to save visualizations')
    parser.add_argument('--no_show', action='store_true', help='Do not display plots (useful for headless environments)')
    
    args = parser.parse_args()
    
    # Load experiment
    if args.experiment_file:
        try:
            experiment = load_experiment(args.experiment_file)
        except Exception as e:
            print(f"Error loading experiment file: {e}")
            return
    else:
        # For testing: Create a dummy experiment with synthetic data
        experiment = {
            'warmup': {
                'losses': [5.0, 4.8, 4.5, 4.2, 4.0, 3.8, 3.7, 3.6, 3.55, 3.53, 3.52, 3.51, 3.50, 3.49, 3.485],
                'is_stable': True,
                'steps_without_decrease': 3,
                'initial_loss': 5.0,
                'final_loss': 3.485,
                'segment_analysis': {
                    'segment_size': 5,
                    'first_segment_avg': 4.5,
                    'last_segment_avg': 3.5,
                    'improvement': 22.2,
                    'still_improving': False
                }
            },
            'pruning': {
                'training_metrics': {
                    'train_loss': [3.5, 3.55, 3.6, 3.65, 3.7, 3.65, 3.6, 3.55, 3.5, 3.45],
                    'step': list(range(10)),
                    'perplexity': [30.0, 32.0, 33.0, 34.0, 35.0, 34.0, 33.0, 32.0, 31.0, 30.0],
                    'pruned_heads': [0, 2, 4, 6, 8, 10, 12, 14, 16, 16],
                    'sparsity': [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 40.0]
                }
            },
            'fine_tuning': {
                'training_metrics': {
                    'train_loss': [3.4, 3.3, 3.2, 3.1, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5],
                    'step': list(range(10)),
                    'perplexity': [30.0, 28.0, 26.0, 24.0, 22.0, 20.0, 18.0, 16.0, 14.0, 12.0]
                }
            }
        }
        print("No experiment file provided. Using synthetic data for demonstration.")
    
    # Generate dashboards
    figures = generate_dashboards(experiment, args.output_dir, show=not args.no_show)
    
    # Show plots if not in headless mode
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()