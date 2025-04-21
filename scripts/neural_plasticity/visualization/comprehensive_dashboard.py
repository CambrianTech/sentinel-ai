#!/usr/bin/env python
"""
Comprehensive Neural Plasticity Dashboard Generator

This script creates a rich dashboard that integrates all visualizations from the neural
plasticity system, including entropy heatmaps, pruning decisions, training metrics,
and attention patterns.

It uses the visualization utilities from utils/neural_plasticity/visualization.py
and dashboard generation from utils/neural_plasticity/dashboard.py to generate
comprehensive visual reports.

Usage:
    python scripts/neural_plasticity/visualization/comprehensive_dashboard.py
        --experiment_dir /path/to/experiment_output
        --output_dir /path/to/save/dashboard

Author: Claude <noreply@anthropic.com>
Version: v0.1.0 (2025-04-20)
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)

# Try to import visualization utilities
try:
    from utils.neural_plasticity.visualization import (
        visualize_head_entropy,
        visualize_head_gradients,
        visualize_pruning_decisions,
        visualize_training_metrics,
        create_pruning_state_heatmap,
        visualize_attention_patterns,
        VisualizationReporter
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Could not import neural plasticity visualization utilities. Using fallback implementations.")
    VISUALIZATION_AVAILABLE = False
    # Import from colab visualizations as fallback
    try:
        from utils.colab.visualizations import (
            visualize_head_entropy,
            visualize_gradient_norms as visualize_head_gradients,
            visualize_training_metrics
        )
    except ImportError:
        logger.warning("Could not import colab visualization utilities. Dashboards will be limited.")

# Try to import dashboard utilities
try:
    from utils.neural_plasticity.dashboard import create_dashboard, DashboardReporter
    DASHBOARD_AVAILABLE = True
except ImportError:
    logger.warning("Could not import neural plasticity dashboard utilities. Using fallback implementation.")
    DASHBOARD_AVAILABLE = False

# Try to import Colab visualization utilities
try:
    from utils.colab.visualizations import visualize_complete_training_process
    COLAB_VISUALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Could not import Colab visualization utilities. Complete process visualization will be limited.")
    COLAB_VISUALIZATION_AVAILABLE = False


def load_json_file(path, default=None):
    """Load JSON data from a file with error handling."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load JSON file {path}: {e}")
        return default or {}


def find_experiment_data_dir(experiment_dir):
    """Find the actual data directory within the experiment directory."""
    import glob
    
    # Look for directories matching the pattern entropy_*
    data_dirs = glob.glob(os.path.join(experiment_dir, "entropy_*"))
    
    if not data_dirs:
        logger.warning(f"No entropy_* directory found in {experiment_dir}")
        return None
    
    # Sort by modification time (most recent first)
    data_dirs.sort(key=os.path.getmtime, reverse=True)
    
    logger.info(f"Using data directory: {data_dirs[0]}")
    return data_dirs[0]


def load_experiment_data(experiment_dir):
    """Load all relevant experiment data from the given directory."""
    data_dir = find_experiment_data_dir(experiment_dir)
    if not data_dir:
        logger.error(f"Could not find data directory in {experiment_dir}")
        return {}
    
    logger.info(f"Loading experiment data from {data_dir}")
    
    # Initialize data dictionary
    data = {
        'experiment_dir': experiment_dir,
        'data_dir': data_dir
    }
    
    # Load JSON files
    json_files = [
        'params.json',
        'metrics.json',
        'pruned_heads.json',
        'pre_entropy.json',
        'post_entropy.json',
        'entropy_deltas.json',
        'entropy_history.json',
        'performance_history.json',
        'gate_history.json',
        'regrowth_analysis.json',
        'results.json'
    ]
    
    for filename in json_files:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            key = os.path.splitext(filename)[0]
            data[key] = load_json_file(file_path)
            logger.info(f"Loaded {filename}")
        else:
            logger.warning(f"File not found: {filename}")
    
    return data


def convert_dict_to_tensor(data_dict):
    """Convert a dictionary of lists to a tensor with shape [layers, heads]."""
    if not data_dict:
        return None
    
    try:
        # Convert keys to integers and sort
        keys = []
        for k in data_dict.keys():
            try:
                keys.append(int(k))
            except (ValueError, TypeError):
                pass
        
        keys.sort()
        
        # Extract values in order
        values = []
        for k in keys:
            values.append(data_dict[str(k)])
        
        # Convert to tensor
        if values:
            try:
                if all(isinstance(item, list) for item in values):
                    return torch.tensor(values)
                else:
                    # Handle case where values might be tensors or other complex objects
                    return torch.tensor(values) if all(isinstance(item, (int, float)) for item in values) else None
            except Exception as e:
                logger.warning(f"Could not convert to tensor: {e}")
                return None
    except Exception as e:
        logger.warning(f"Error converting dictionary to tensor: {e}")
        return None


def fallback_visualize_pruning_state(pruned_heads, title="Pruned Heads Map", figsize=(10, 6)):
    """Fallback implementation of create_pruning_state_heatmap when utils are unavailable."""
    # Determine dimensions from pruned_heads
    if not pruned_heads:
        # Default dimensions if no heads are pruned
        num_layers = 6
        num_heads = 12
    else:
        # Find max dimensions from pruned heads
        max_layer = max(head[0] for head in pruned_heads) if pruned_heads else 5
        max_head = max(head[1] for head in pruned_heads) if pruned_heads else 11
        num_layers = max_layer + 1
        num_heads = max_head + 1
    
    # Create empty matrix for pruning state
    pruning_state = np.zeros((num_layers, num_heads))
    
    # Fill in pruned heads
    for layer, head in pruned_heads:
        if 0 <= layer < num_layers and 0 <= head < num_heads:
            pruning_state[layer, head] = 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    import matplotlib.colors as mcolors
    colors = [(0.9, 0.9, 0.9), (0.8, 0.2, 0.2)]  # light gray, red
    cmap = mcolors.ListedColormap(colors)
    
    im = ax.imshow(pruning_state, cmap=cmap, aspect='auto')
    
    # Add colorbar with custom labels
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.set_ticklabels(['Active', 'Pruned'])
    
    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Layer Index')
    
    # Add text with pruned head count
    total_heads = num_layers * num_heads
    pruned_percent = (len(pruned_heads) / total_heads) * 100
    ax.text(0.05, -0.15, f"Total pruned: {len(pruned_heads)}/{total_heads} heads ({pruned_percent:.1f}%)",
            transform=ax.transAxes, fontsize=10)
    
    plt.tight_layout()
    return fig


def fallback_visualize_complete_process(experiment_data, output_dir):
    """Fallback implementation for visualize_complete_training_process when utils are unavailable."""
    # Extract metrics
    metrics = experiment_data.get('metrics', {})
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Extract loss values for different phases
    baseline_loss = metrics.get('baseline', {}).get('loss', 0)
    post_pruning_loss = metrics.get('post_pruning', {}).get('loss', 0)
    final_loss = metrics.get('final', {}).get('loss', 0)
    
    # Plot loss values
    steps = [0, 1, 2]
    losses = [baseline_loss, post_pruning_loss, final_loss]
    phases = ['Baseline', 'After Pruning', 'After Fine-tuning']
    
    ax1.plot(steps, losses, 'b-o', linewidth=2, markersize=8)
    ax1.set_title('Training Loss Through Neural Plasticity Process')
    ax1.set_ylabel('Loss')
    ax1.set_xticks(steps)
    ax1.set_xticklabels(phases)
    ax1.grid(True, alpha=0.3)
    
    # Add value annotations
    for i, v in enumerate(losses):
        ax1.text(i, v, f"{v:.4f}", ha='center', va='bottom')
    
    # Add phase highlighting
    ax1.axvspan(-0.5, 0.5, alpha=0.2, color='blue', label='Initial')
    ax1.axvspan(0.5, 1.5, alpha=0.2, color='red', label='Pruning')
    ax1.axvspan(1.5, 2.5, alpha=0.2, color='green', label='Fine-tuning')
    
    # Plot perplexity values
    baseline_perplexity = metrics.get('baseline', {}).get('perplexity', 0)
    post_pruning_perplexity = metrics.get('post_pruning', {}).get('perplexity', 0)
    final_perplexity = metrics.get('final', {}).get('perplexity', 0)
    
    perplexities = [baseline_perplexity, post_pruning_perplexity, final_perplexity]
    
    ax2.plot(steps, perplexities, 'g-o', linewidth=2, markersize=8)
    ax2.set_title('Perplexity Through Neural Plasticity Process')
    ax2.set_ylabel('Perplexity')
    ax2.set_xticks(steps)
    ax2.set_xticklabels(phases)
    ax2.grid(True, alpha=0.3)
    
    # Add value annotations
    for i, v in enumerate(perplexities):
        ax2.text(i, v, f"{v:.2f}", ha='center', va='bottom')
    
    # Add phase highlighting
    ax2.axvspan(-0.5, 0.5, alpha=0.2, color='blue')
    ax2.axvspan(0.5, 1.5, alpha=0.2, color='red')
    ax2.axvspan(1.5, 2.5, alpha=0.2, color='green')
    
    # Get pruning information
    pruned_heads = experiment_data.get('pruned_heads', [])
    pruned_count = len(pruned_heads)
    
    # Add summary text
    plt.figtext(0.5, 0.01, 
               f"Neural Plasticity Process Summary - Pruned Heads: {pruned_count} - Perplexity Improvement: {(baseline_perplexity - final_perplexity) / baseline_perplexity * 100:.1f}%",
               ha="center", fontsize=10, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "neural_plasticity_process.png")
    
    # Save the figure
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    logger.info(f"Saved complete process visualization to {output_path}")
    
    return fig


def generate_visualizations(experiment_data, output_dir):
    """Generate all visualizations for the experiment data."""
    logger.info(f"Generating visualizations in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    visualizations = {}
    
    # Extract key data
    params = experiment_data.get('params', {})
    metrics = experiment_data.get('metrics', {})
    pruned_heads = experiment_data.get('pruned_heads', [])
    pre_entropy = experiment_data.get('pre_entropy', {})
    post_entropy = experiment_data.get('post_entropy', {})
    
    # Convert dictionaries to tensors
    pre_entropy_tensor = convert_dict_to_tensor(pre_entropy)
    post_entropy_tensor = convert_dict_to_tensor(post_entropy)
    
    # Create entropy heatmap
    if pre_entropy_tensor is not None and VISUALIZATION_AVAILABLE:
        try:
            fig = visualize_head_entropy(
                entropy_values=pre_entropy_tensor,
                title="Pre-Pruning Entropy Heatmap"
            )
            file_path = os.path.join(output_dir, "pre_pruning_entropy.png")
            fig.savefig(file_path, dpi=100, bbox_inches='tight')
            visualizations['pre_pruning_entropy'] = file_path
            logger.info(f"Generated pre-pruning entropy visualization: {file_path}")
        except Exception as e:
            logger.error(f"Error creating pre-pruning entropy visualization: {e}")
    
    # Create post-pruning entropy heatmap
    if post_entropy_tensor is not None and VISUALIZATION_AVAILABLE:
        try:
            fig = visualize_head_entropy(
                entropy_values=post_entropy_tensor,
                title="Post-Pruning Entropy Heatmap"
            )
            file_path = os.path.join(output_dir, "post_pruning_entropy.png")
            fig.savefig(file_path, dpi=100, bbox_inches='tight')
            visualizations['post_pruning_entropy'] = file_path
            logger.info(f"Generated post-pruning entropy visualization: {file_path}")
        except Exception as e:
            logger.error(f"Error creating post-pruning entropy visualization: {e}")
    
    # Create pruning state heatmap
    try:
        if VISUALIZATION_AVAILABLE and 'create_pruning_state_heatmap' in globals():
            fig = create_pruning_state_heatmap(
                model=None,  # Not needed for visualization only
                cumulative_pruned=pruned_heads,
                title="Pruned Heads Map"
            )
        else:
            # Use fallback implementation
            fig = fallback_visualize_pruning_state(
                pruned_heads=pruned_heads,
                title="Pruned Heads Map"
            )
            
        file_path = os.path.join(output_dir, "pruned_heads_map.png")
        fig.savefig(file_path, dpi=100, bbox_inches='tight')
        visualizations['pruned_heads_map'] = file_path
        logger.info(f"Generated pruned heads map: {file_path}")
    except Exception as e:
        logger.error(f"Error creating pruned heads map: {e}")
    
    # Create training metrics visualization
    try:
        # Construct metrics history from final metrics
        metrics_history = {
            'step': [0, 1, 2],
            'train_loss': [
                metrics.get('baseline', {}).get('loss', 0),
                metrics.get('post_pruning', {}).get('loss', 0),
                metrics.get('final', {}).get('loss', 0)
            ],
            'perplexity': [
                metrics.get('baseline', {}).get('perplexity', 0),
                metrics.get('post_pruning', {}).get('perplexity', 0),
                metrics.get('final', {}).get('perplexity', 0)
            ]
        }
        
        fig = visualize_training_metrics(
            metrics_history=metrics_history,
            title="Training Metrics"
        )
        file_path = os.path.join(output_dir, "training_metrics.png")
        fig.savefig(file_path, dpi=100, bbox_inches='tight')
        visualizations['training_metrics'] = file_path
        logger.info(f"Generated training metrics visualization: {file_path}")
    except Exception as e:
        logger.error(f"Error creating training metrics visualization: {e}")
    
    return visualizations


def generate_comprehensive_dashboard(experiment_data, output_dir):
    """
    Generate a comprehensive dashboard using all available visualization systems.
    
    This function attempts to use:
    1. The core visualization system from utils/neural_plasticity/visualization.py
    2. The dashboard system from utils/neural_plasticity/dashboard.py
    3. The Colab visualization system from utils/colab/visualizations.py
    
    It falls back to basic functionality if any system is unavailable.
    
    Args:
        experiment_data: Dictionary with experiment data
        output_dir: Directory to save visualization outputs
    
    Returns:
        Dictionary with paths to generated visualizations and dashboards
    """
    logger.info(f"Generating comprehensive dashboard in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    outputs = {}
    
    # 1. Generate individual visualizations
    viz_dir = os.path.join(output_dir, "visualizations")
    visualizations = generate_visualizations(experiment_data, viz_dir)
    outputs['visualizations'] = visualizations
    
    # 2. Generate HTML dashboard if available
    if DASHBOARD_AVAILABLE:
        try:
            # Extract key data
            params = experiment_data.get('params', {})
            metrics = experiment_data.get('metrics', {})
            pruned_heads = experiment_data.get('pruned_heads', [])
            pre_entropy = experiment_data.get('pre_entropy', {})
            post_entropy = experiment_data.get('post_entropy', {})
            
            # Create metrics history
            metrics_history = {
                'step': [0, 1, 2],
                'train_loss': [
                    metrics.get('baseline', {}).get('loss', 0),
                    metrics.get('post_pruning', {}).get('loss', 0),
                    metrics.get('final', {}).get('loss', 0)
                ],
                'perplexity': [
                    metrics.get('baseline', {}).get('perplexity', 0),
                    metrics.get('post_pruning', {}).get('perplexity', 0),
                    metrics.get('final', {}).get('perplexity', 0)
                ],
                'sparsity': [0, 
                             len(pruned_heads) / 384 * 100 if pruned_heads else 0,
                             len(pruned_heads) / 384 * 100 if pruned_heads else 0]
            }
            
            # Convert entropy data to tensors
            pre_entropy_tensor = convert_dict_to_tensor(pre_entropy)
            post_entropy_tensor = convert_dict_to_tensor(post_entropy)
            
            # Generate the dashboard
            dashboard_path = os.path.join(output_dir, "dashboard.html")
            
            create_dashboard(
                metrics_history=metrics_history,
                entropy_values=pre_entropy_tensor,
                pruned_heads=pruned_heads,
                model_info={
                    'model_name': params.get('model_name', 'Unknown'),
                    'total_params': 82000000,  # Placeholder
                    'model_size_mb': 320.5  # Placeholder
                },
                save_path=dashboard_path
            )
            
            outputs['dashboard'] = dashboard_path
            logger.info(f"Generated HTML dashboard: {dashboard_path}")
        except Exception as e:
            logger.error(f"Error generating HTML dashboard: {e}")
    
    # 3. Generate comprehensive training process visualization
    try:
        process_viz_path = os.path.join(output_dir, "complete_process")
        os.makedirs(process_viz_path, exist_ok=True)
        
        if COLAB_VISUALIZATION_AVAILABLE:
            # Convert experiment_data to format expected by visualize_complete_training_process
            processed_data = {
                'warmup': {'metrics': experiment_data.get('metrics', {}).get('baseline', {})},
                'pruning': {'metrics': experiment_data.get('metrics', {}).get('post_pruning', {})},
                'fine_tuning': {'metrics': experiment_data.get('metrics', {}).get('final', {})},
                'pruned_heads': experiment_data.get('pruned_heads', []),
                'params': experiment_data.get('params', {})
            }
            
            # Generate the visualization
            fig = visualize_complete_training_process(
                experiment=processed_data,
                output_dir=process_viz_path,
                title="Neural Plasticity Training Process",
                show_plot=False
            )
        else:
            # Use fallback implementation
            fig = fallback_visualize_complete_process(
                experiment_data=experiment_data,
                output_dir=process_viz_path
            )
        
        # Save the visualization
        process_viz_file = os.path.join(process_viz_path, "neural_plasticity_process.png")
        fig.savefig(process_viz_file, dpi=100, bbox_inches='tight')
        
        outputs['complete_process'] = process_viz_file
        logger.info(f"Generated complete process visualization: {process_viz_file}")
    except Exception as e:
        logger.error(f"Error generating complete process visualization: {e}")
    
    # 4. Use VisualizationReporter if available
    if VISUALIZATION_AVAILABLE and 'VisualizationReporter' in globals():
        try:
            reporter = VisualizationReporter(output_dir=os.path.join(output_dir, "reporter_dashboards"), save_visualizations=True)
            
            # Add key metrics
            reporter_metrics = {
                'baseline': experiment_data.get('metrics', {}).get('baseline', {}),
                'post_pruning': experiment_data.get('metrics', {}).get('post_pruning', {}),
                'final': experiment_data.get('metrics', {}).get('final', {})
            }
            
            # Generate dashboards using the reporter
            reporter_outputs = reporter.generate_comprehensive_dashboard(
                experiment={
                    'metrics': reporter_metrics,
                    'pruned_heads': experiment_data.get('pruned_heads', []),
                    'params': experiment_data.get('params', {})
                },
                output_dir=os.path.join(output_dir, "reporter_dashboards")
            )
            
            # Add reporter outputs to the results
            outputs['reporter_dashboards'] = reporter_outputs
            logger.info(f"Generated reporter dashboards in {os.path.join(output_dir, 'reporter_dashboards')}")
        except Exception as e:
            logger.error(f"Error generating reporter dashboards: {e}")
    
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Generate Comprehensive Neural Plasticity Dashboard")
    parser.add_argument("--experiment_dir", type=str, help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, help="Directory to save dashboard files")
    args = parser.parse_args()
    
    # Find the most recent experiment directory if not specified
    if not args.experiment_dir:
        import glob
        experiment_dirs = glob.glob(os.path.join(project_root, "experiment_output", "neural_plasticity", "run_*"))
        experiment_dirs += glob.glob(os.path.join(project_root, "neural_plasticity_output", "run_*"))
        
        if experiment_dirs:
            # Sort by modification time (most recent first)
            experiment_dirs.sort(key=os.path.getmtime, reverse=True)
            args.experiment_dir = experiment_dirs[0]
            logger.info(f"Using latest experiment directory: {args.experiment_dir}")
        else:
            logger.error("No experiment directory found. Please specify --experiment_dir.")
            return
    
    # Set output directory if not specified
    if not args.output_dir:
        args.output_dir = os.path.join(args.experiment_dir, "dashboards", "comprehensive")
    
    # Load experiment data
    experiment_data = load_experiment_data(args.experiment_dir)
    if not experiment_data:
        logger.error(f"No experiment data found in {args.experiment_dir}")
        return
    
    # Generate the comprehensive dashboard
    outputs = generate_comprehensive_dashboard(experiment_data, args.output_dir)
    
    # Print summary
    logger.info("Dashboard generation completed")
    logger.info(f"Output directory: {args.output_dir}")
    
    print("\nDashboard Generation Summary:")
    print(f"- Experiment directory: {args.experiment_dir}")
    print(f"- Output directory: {args.output_dir}")
    
    if 'dashboard' in outputs:
        print(f"- Main dashboard: {outputs['dashboard']}")
    
    if 'complete_process' in outputs:
        print(f"- Complete process visualization: {outputs['complete_process']}")
    
    if 'visualizations' in outputs:
        print(f"- Individual visualizations: {len(outputs['visualizations'])} files in {os.path.join(args.output_dir, 'visualizations')}")
    
    print("\nTo view the main dashboard, open this file in a web browser:")
    if 'dashboard' in outputs:
        print(f"  {outputs['dashboard']}")
    else:
        print(f"  {os.path.join(args.output_dir, 'dashboard.html')} (if generated)")
        
    # Create symlink to make it easy to access the dashboard
    try:
        symlink_dir = os.path.join(project_root, "dashboards")
        os.makedirs(symlink_dir, exist_ok=True)
        
        if 'dashboard' in outputs:
            symlink_path = os.path.join(symlink_dir, "latest_comprehensive_dashboard.html")
            if os.path.exists(symlink_path):
                os.remove(symlink_path)
            os.symlink(outputs['dashboard'], symlink_path)
            print(f"\nQuick access symlink created: {symlink_path}")
    except Exception as e:
        logger.warning(f"Could not create symlink: {e}")


if __name__ == "__main__":
    main()