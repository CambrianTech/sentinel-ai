#!/usr/bin/env python
"""
Generate Rich Neural Plasticity Dashboard

This script creates a rich, dynamic dashboard that exactly matches the example-run.png
reference with proper tabbed navigation and detailed visualizations.

Usage:
    python scripts/neural_plasticity/visualization/generate_rich_dashboard.py
        --experiment_dir /path/to/experiment_output
        --output_path /path/to/save/dashboard.html

Author: Claude <noreply@anthropic.com>
Version: v0.1.0 (2025-04-20)
"""

import os
import sys
import json
import argparse
import glob
from datetime import datetime
from pathlib import Path

# Add project root to Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(project_root)

# Import dashboard generator from utils
from utils.neural_plasticity.dashboard import create_dashboard

def load_json_data(file_path):
    """Load JSON data from a file, handling errors gracefully."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load JSON file {file_path}: {e}")
        return {}

def find_experiment_data_dir(experiment_dir):
    """Find the actual data directory within the experiment directory."""
    # Look for directories matching the pattern entropy_*
    data_dirs = glob.glob(os.path.join(experiment_dir, "entropy_*"))
    
    if not data_dirs:
        return None
    
    # Sort by modification time (most recent first)
    data_dirs.sort(key=os.path.getmtime, reverse=True)
    
    return data_dirs[0]

def load_experiment_data(experiment_dir):
    """Load all experiment data from the specified directory."""
    data_dir = find_experiment_data_dir(experiment_dir)
    
    if not data_dir:
        print(f"Warning: Could not find data directory in {experiment_dir}")
        return {}
    
    # Load all experiment data files
    data = {}
    
    # Parameters
    params_path = os.path.join(data_dir, "params.json")
    if os.path.exists(params_path):
        data['params'] = load_json_data(params_path)
    
    # Metrics
    metrics_path = os.path.join(data_dir, "metrics.json")
    if os.path.exists(metrics_path):
        data['metrics'] = load_json_data(metrics_path)
    
    # Pruned heads
    pruned_heads_path = os.path.join(data_dir, "pruned_heads.json")
    if os.path.exists(pruned_heads_path):
        data['pruned_heads'] = load_json_data(pruned_heads_path)
    
    # Entropy data
    pre_entropy_path = os.path.join(data_dir, "pre_entropy.json")
    if os.path.exists(pre_entropy_path):
        data['pre_entropy'] = load_json_data(pre_entropy_path)
    
    post_entropy_path = os.path.join(data_dir, "post_entropy.json")
    if os.path.exists(post_entropy_path):
        data['post_entropy'] = load_json_data(post_entropy_path)
    
    # Results
    results_path = os.path.join(data_dir, "results.json")
    if os.path.exists(results_path):
        data['results'] = load_json_data(results_path)
    
    # Additional data files
    for filename in ["entropy_deltas.json", "performance_history.json", 
                    "gate_history.json", "regrowth_analysis.json"]:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            key = os.path.splitext(filename)[0]
            data[key] = load_json_data(file_path)
    
    return data

def prepare_data_for_dashboard(experiment_data):
    """
    Transform experiment data into the format expected by create_dashboard.
    
    Args:
        experiment_data: Dictionary with experiment data
        
    Returns:
        Dictionary with data ready for dashboard creation
    """
    # Extract key data
    params = experiment_data.get('params', {})
    metrics = experiment_data.get('metrics', {})
    results = experiment_data.get('results', {})
    pruned_heads_data = experiment_data.get('pruned_heads', [])
    
    # Convert pruned_heads to the expected format
    pruned_heads = []
    for item in pruned_heads_data:
        if isinstance(item, list) and len(item) >= 2:
            pruned_heads.append((item[0], item[1]))
    
    # Create a metrics history dictionary
    metrics_history = {}
    
    # Add basic metrics
    for key in ['loss', 'perplexity', 'sparsity']:
        if key in metrics:
            metrics_history[key] = [metrics[key]]
    
    # Add step data
    metrics_history['step'] = [0]
    
    # Add pruned_heads count
    metrics_history['pruned_heads'] = [len(pruned_heads)]
    
    # Process entropy data
    pre_entropy = experiment_data.get('pre_entropy', {})
    post_entropy = experiment_data.get('post_entropy', {})
    
    # Process pre_entropy into tensor format if available
    entropy_values = None
    if pre_entropy:
        try:
            import torch
            import numpy as np
            
            # Example conversion based on expected structure (adjust as needed)
            # Convert dictionary to 2D tensor [layers, heads]
            layers = sorted(list(pre_entropy.keys()))
            if layers:
                entropy_data = []
                for layer in layers:
                    if isinstance(pre_entropy[layer], list):
                        entropy_data.append(pre_entropy[layer])
                
                if entropy_data:
                    entropy_values = torch.tensor(entropy_data)
        except Exception as e:
            print(f"Warning: Could not process entropy data: {e}")
    
    # Process gradient data (if available)
    grad_norm_values = None
    if 'gradient_norms' in experiment_data:
        try:
            import torch
            grad_data = experiment_data['gradient_norms']
            grad_norm_values = torch.tensor(grad_data)
        except Exception:
            pass
    
    # Generate model info
    model_info = {
        'model_name': params.get('model_name', 'distilgpt2'),
        'total_params': 82000000,  # Placeholder value
        'model_size_mb': 320.5      # Placeholder value
    }
    
    return {
        'metrics_history': metrics_history,
        'entropy_values': entropy_values,
        'grad_norm_values': grad_norm_values,
        'pruning_mask': None,  # Not available in current data
        'pruned_heads': pruned_heads,
        'attention_maps': None,  # Not available in current data
        'sample_data': None,     # Not available in current data
        'model_info': model_info
    }

def generate_dashboard(experiment_data, output_path):
    """
    Generate a rich dashboard from experiment data.
    
    Args:
        experiment_data: Dictionary with experiment data
        output_path: Path to save the dashboard HTML
        
    Returns:
        Path to the generated dashboard
    """
    # Prepare data for dashboard creation
    dashboard_data = prepare_data_for_dashboard(experiment_data)
    
    # Create the dashboard
    dashboard_html = create_dashboard(
        metrics_history=dashboard_data['metrics_history'],
        entropy_values=dashboard_data['entropy_values'],
        grad_norm_values=dashboard_data['grad_norm_values'],
        pruning_mask=dashboard_data['pruning_mask'],
        pruned_heads=dashboard_data['pruned_heads'],
        attention_maps=dashboard_data['attention_maps'],
        sample_data=dashboard_data['sample_data'],
        model_info=dashboard_data['model_info'],
        save_path=output_path
    )
    
    print(f"✅ Rich Neural Plasticity Dashboard generated at {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Generate Rich Neural Plasticity Dashboard")
    parser.add_argument("--experiment_dir", type=str, help="Directory containing experiment results")
    parser.add_argument("--output_path", type=str, help="Path to save the dashboard HTML file")
    args = parser.parse_args()
    
    # Find latest experiment directory if not specified
    if not args.experiment_dir:
        experiment_dirs = glob.glob(os.path.join(project_root, "experiment_output", "neural_plasticity", "run_*"))
        experiment_dirs += glob.glob(os.path.join(project_root, "neural_plasticity_output", "run_*"))
        
        if experiment_dirs:
            # Sort by modification time (most recent first)
            experiment_dirs.sort(key=os.path.getmtime, reverse=True)
            args.experiment_dir = experiment_dirs[0]
            print(f"Using latest experiment directory: {args.experiment_dir}")
        else:
            print("No experiment directories found. Please specify --experiment_dir.")
            return
    
    # Set default output path if not specified
    if not args.output_path:
        if args.experiment_dir:
            # Create dashboards directory within experiment directory
            dashboards_dir = os.path.join(args.experiment_dir, "dashboards")
            os.makedirs(dashboards_dir, exist_ok=True)
            args.output_path = os.path.join(dashboards_dir, "rich_dashboard.html")
        else:
            args.output_path = "rich_dashboard.html"
    
    # Load experiment data
    experiment_data = load_experiment_data(args.experiment_dir)
    
    # Generate dashboard
    generate_dashboard(experiment_data, args.output_path)

if __name__ == "__main__":
    main()