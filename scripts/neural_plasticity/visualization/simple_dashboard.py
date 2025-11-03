#!/usr/bin/env python
"""
Simple Neural Plasticity Dashboard Generator

This script creates a very simple HTML dashboard showing experiment results
without any complex formatting or template issues.

Usage:
    python scripts/neural_plasticity/visualization/simple_dashboard.py
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

def generate_dashboard(experiment_data, output_path):
    """Generate a simple HTML dashboard from experiment data."""
    # Extract key data
    params = experiment_data.get('params', {})
    metrics = experiment_data.get('metrics', {})
    results = experiment_data.get('results', {})
    pruned_heads = experiment_data.get('pruned_heads', [])
    
    # Current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start building HTML
    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <title>Neural Plasticity Dashboard</title>",
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 20px; }",
        "        h1 { color: #333; }",
        "        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
        "        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "        th { background-color: #f2f2f2; }",
        "        .metric { display: inline-block; margin: 10px; padding: 15px; background-color: #f9f9f9; border-radius: 5px; text-align: center; }",
        "        .metric-value { font-size: 24px; font-weight: bold; color: #2980b9; }",
        "        .container { max-width: 1200px; margin: 0 auto; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <div class='container'>",
        f"        <h1>Neural Plasticity Dashboard</h1>",
        f"        <p>Generated on {timestamp}</p>",
        "",
        "        <h2>Experiment Metrics</h2>",
        "        <div>",
    ]
    
    # Add metrics
    for key, value in metrics.items():
        if isinstance(value, float):
            value = f"{value:.4f}"
        html.append(f"            <div class='metric'><div>{key}</div><div class='metric-value'>{value}</div></div>")
    
    # Add params table
    html.extend([
        "        </div>",
        "",
        "        <h2>Experiment Parameters</h2>",
        "        <table>",
        "            <tr><th>Parameter</th><th>Value</th></tr>",
    ])
    
    for key, value in params.items():
        html.append(f"            <tr><td>{key}</td><td>{value}</td></tr>")
    
    # Add pruned heads
    html.extend([
        "        </table>",
        "",
        "        <h2>Pruned Heads</h2>",
        f"        <p>Total pruned heads: {len(pruned_heads)}</p>",
        "        <table>",
        "            <tr><th>Layer</th><th>Head</th><th>Score</th></tr>",
    ])
    
    for i, head_data in enumerate(pruned_heads[:20]):  # Show at most 20 heads
        if isinstance(head_data, list) and len(head_data) >= 2:
            layer, head = head_data[0], head_data[1]
            score = head_data[2] if len(head_data) >= 3 else "N/A"
            html.append(f"            <tr><td>{layer}</td><td>{head}</td><td>{score}</td></tr>")
    
    # If there are more than 20 pruned heads, add a summary row
    if len(pruned_heads) > 20:
        html.append(f"            <tr><td colspan='3'>... and {len(pruned_heads) - 20} more heads</td></tr>")
    
    # Add results
    html.extend([
        "        </table>",
        "",
        "        <h2>Experiment Results</h2>",
        "        <pre>",
        json.dumps(results, indent=4),
        "        </pre>",
        "",
        "        <p>Dashboard version: v0.1.0 (2025-04-20)</p>",
        "    </div>",
        "</body>",
        "</html>",
    ])
    
    # Join all HTML lines
    dashboard_html = "\n".join(html)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Write HTML to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    
    print(f"✅ Simple Neural Plasticity Dashboard generated at {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Simple Neural Plasticity Dashboard Generator")
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
            args.output_path = os.path.join(dashboards_dir, "simple_dashboard.html")
        else:
            args.output_path = "simple_dashboard.html"
    
    # Load experiment data
    experiment_data = load_experiment_data(args.experiment_dir)
    
    # Generate dashboard
    generate_dashboard(experiment_data, args.output_path)

if __name__ == "__main__":
    main()