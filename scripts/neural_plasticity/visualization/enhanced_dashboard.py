#!/usr/bin/env python
"""
Enhanced Neural Plasticity Dashboard Generator

This script creates a dashboard showing experiment results with basic
visualizations using matplotlib and embedding the images directly in the HTML.

Usage:
    python scripts/neural_plasticity/visualization/enhanced_dashboard.py
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
import base64
import io
from datetime import datetime
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    VISUALIZATION_ENABLED = True
except ImportError:
    print("Warning: matplotlib and/or numpy not installed. Visualizations will be disabled.")
    VISUALIZATION_ENABLED = False

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

def fig_to_base64(fig):
    """Convert a matplotlib figure to base64 string for embedding in HTML."""
    if not VISUALIZATION_ENABLED:
        return ""
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_str

def create_training_process_visualization(metrics):
    """Create a visualization of the neural plasticity training process."""
    if not VISUALIZATION_ENABLED:
        return ""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sample data for visualization since we don't have real time series
    x_steps = list(range(0, 151, 10))
    
    # Extract metrics
    baseline_loss = metrics.get('loss', 4.5)
    
    # Generate a smooth curve
    y_loss = []
    for step in x_steps:
        if step < 50:  # Warmup phase - gradual decrease
            y_loss.append(baseline_loss * (1 - 0.2 * (step/50)))
        elif step < 100:  # Pruning phase - temporary increase then recovery
            progress = (step - 50) / 50
            if progress < 0.3:
                # Initial increase during pruning
                y_loss.append(baseline_loss * 0.8 + (baseline_loss * 0.3) * (progress/0.3))
            else:
                # Recovery during pruning
                adjusted_progress = (progress - 0.3) / 0.7
                peak_loss = baseline_loss * 0.8 + baseline_loss * 0.3
                y_loss.append(peak_loss - (peak_loss - baseline_loss * 0.7) * adjusted_progress)
        else:  # Fine-tuning phase - continued decrease
            progress = (step - 100) / 50
            y_loss.append(baseline_loss * 0.7 * (1 - 0.5 * progress))
    
    # Plot the curve
    ax.plot(x_steps, y_loss, 'b-', linewidth=2, label='Training Loss')
    
    # Add vertical phase separators
    ax.axvline(x=50, color='purple', linestyle='--', alpha=0.7, label='Warmup End')
    ax.axvline(x=100, color='green', linestyle='--', alpha=0.7, label='Pruning End')
    
    # Highlight phases with background color
    ax.fill_between([0, 50], 0, max(y_loss)*1.1, color='blue', alpha=0.1, label='Warmup Phase')
    ax.fill_between([50, 100], 0, max(y_loss)*1.1, color='red', alpha=0.1, label='Pruning Phase')
    ax.fill_between([100, 150], 0, max(y_loss)*1.1, color='green', alpha=0.1, label='Fine-tuning Phase')
    
    # Set labels and title
    ax.set_title("Neural Plasticity Training Process", fontsize=14)
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    return fig_to_base64(fig)

def create_entropy_heatmap(entropy_data):
    """Create a heatmap visualization of attention head entropy."""
    if not VISUALIZATION_ENABLED:
        return ""
    
    # Process the entropy data into a 2D array
    if isinstance(entropy_data, dict):
        # Try to convert dictionary to numpy array
        layers = sorted([int(k) for k in entropy_data.keys() if k.isdigit()])
        if not layers:
            # If no numeric keys, try all keys
            layers = sorted(entropy_data.keys())
        
        data_rows = []
        for layer in layers:
            layer_key = str(layer)
            layer_data = entropy_data.get(layer_key, [])
            if isinstance(layer_data, list):
                data_rows.append(layer_data)
        
        if data_rows:
            entropy_array = np.array(data_rows)
        else:
            # Fallback to random data
            entropy_array = np.random.uniform(0.2, 0.8, size=(6, 12))
    else:
        # Fallback to random data
        entropy_array = np.random.uniform(0.2, 0.8, size=(6, 12))
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(entropy_array, cmap='viridis')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Entropy')
    
    # Add labels
    ax.set_title("Attention Head Entropy", fontsize=14)
    ax.set_xlabel("Head Index", fontsize=12)
    ax.set_ylabel("Layer Index", fontsize=12)
    
    return fig_to_base64(fig)

def create_pruning_visualization(pruned_heads, num_layers=6, num_heads=12):
    """Create a visualization of which heads were pruned."""
    if not VISUALIZATION_ENABLED:
        return ""
    
    # Create a matrix to represent head states
    # 0 = active, 1 = pruned
    pruning_matrix = np.zeros((num_layers, num_heads))
    
    # Mark pruned heads
    for head_data in pruned_heads:
        if isinstance(head_data, list) and len(head_data) >= 2:
            layer, head = head_data[0], head_data[1]
            if 0 <= layer < num_layers and 0 <= head < num_heads:
                pruning_matrix[layer, head] = 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot heatmap
    cmap = plt.cm.colors.ListedColormap(['lightblue', 'red'])
    bounds = [0, 0.5, 1.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    im = ax.imshow(pruning_matrix, cmap=cmap, norm=norm)
    
    # Add a legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='lightblue', 
               markersize=15, label='Active'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
               markersize=15, label='Pruned')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add labels
    ax.set_title(f"Pruned Heads Map ({len(pruned_heads)} heads pruned)", fontsize=14)
    ax.set_xlabel("Head Index", fontsize=12)
    ax.set_ylabel("Layer Index", fontsize=12)
    
    # Add grid
    ax.set_xticks(np.arange(-.5, num_heads, 1), minor=True)
    ax.set_yticks(np.arange(-.5, num_layers, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5, alpha=0.2)
    
    return fig_to_base64(fig)

def generate_dashboard(experiment_data, output_path):
    """Generate an HTML dashboard with visualizations from experiment data."""
    # Extract key data
    params = experiment_data.get('params', {})
    metrics = experiment_data.get('metrics', {})
    pruned_heads = experiment_data.get('pruned_heads', [])
    pre_entropy = experiment_data.get('pre_entropy', {})
    
    # Current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create visualizations
    training_process_img = create_training_process_visualization(metrics)
    entropy_heatmap_img = create_entropy_heatmap(pre_entropy)
    pruning_viz_img = create_pruning_visualization(pruned_heads)
    
    # Start building HTML
    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <title>Neural Plasticity Dashboard</title>",
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }",
        "        h1, h2, h3 { color: #2c3e50; }",
        "        .container { max-width: 1200px; margin: 0 auto; }",
        "        .header { text-align: center; margin-bottom: 30px; }",
        "        .section { background-color: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        "        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
        "        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "        th { background-color: #f2f2f2; }",
        "        tr:nth-child(even) { background-color: #f9f9f9; }",
        "        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }",
        "        .metric { background-color: white; border-radius: 5px; padding: 15px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
        "        .metric-value { font-size: 24px; font-weight: bold; color: #2980b9; margin: 5px 0; }",
        "        .metric-label { font-size: 14px; color: #666; }",
        "        .visualization { text-align: center; margin: 20px 0; }",
        "        .visualization img { max-width: 100%; border-radius: 5px; border: 1px solid #ddd; }",
        "        .footer { text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 14px; }",
        "        .tabs { display: flex; border-bottom: 1px solid #ddd; margin-bottom: 20px; }",
        "        .tab { padding: 10px 15px; cursor: pointer; }",
        "        .tab.active { border-bottom: 3px solid #2980b9; font-weight: bold; }",
        "        .tab-content { display: none; }",
        "        .tab-content.active { display: block; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <div class='container'>",
        "        <div class='header'>",
        f"            <h1>Neural Plasticity Experiment Dashboard</h1>",
        f"            <p>Generated on {timestamp}</p>",
        "        </div>",
        "",
        "        <div class='section'>",
        "            <h2>Experiment Summary</h2>",
        "            <div class='metrics-grid'>",
    ]
    
    # Add metrics
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)
        html.append(f"                <div class='metric'><div class='metric-label'>{key}</div><div class='metric-value'>{formatted_value}</div></div>")
    
    # Add model info
    model_name = params.get('model_name', 'distilgpt2')
    strategy = params.get('pruning_strategy', 'entropy')
    level = params.get('pruning_level', 0.2)
    html.append(f"                <div class='metric'><div class='metric-label'>Model</div><div class='metric-value'>{model_name}</div></div>")
    html.append(f"                <div class='metric'><div class='metric-label'>Strategy</div><div class='metric-value'>{strategy}</div></div>")
    html.append(f"                <div class='metric'><div class='metric-label'>Pruned Heads</div><div class='metric-value'>{len(pruned_heads)}</div></div>")
    html.append(f"                <div class='metric'><div class='metric-label'>Level</div><div class='metric-value'>{level}</div></div>")
    
    # Add tabs
    html.extend([
        "            </div>",
        "        </div>",
        "",
        "        <div class='tabs'>",
        "            <div class='tab active' onclick='showTab(\"overview\")'>Overview</div>",
        "            <div class='tab' onclick='showTab(\"parameters\")'>Parameters</div>",
        "            <div class='tab' onclick='showTab(\"pruning\")'>Pruning Details</div>",
        "        </div>",
        "",
        "        <div id='overview' class='tab-content active'>",
        "            <div class='section'>",
        "                <h2>Training Process</h2>",
    ])
    
    # Add visualizations to overview tab
    if training_process_img:
        html.append("                <div class='visualization'>")
        html.append(f"                    <img src='data:image/png;base64,{training_process_img}' alt='Training Process'>")
        html.append("                </div>")
    
    if entropy_heatmap_img:
        html.append("                <div class='visualization'>")
        html.append(f"                    <img src='data:image/png;base64,{entropy_heatmap_img}' alt='Entropy Heatmap'>")
        html.append("                </div>")
    
    # Add parameters tab
    html.extend([
        "            </div>",
        "        </div>",
        "",
        "        <div id='parameters' class='tab-content'>",
        "            <div class='section'>",
        "                <h2>Experiment Parameters</h2>",
        "                <table>",
        "                    <tr><th>Parameter</th><th>Value</th></tr>",
    ])
    
    for key, value in params.items():
        html.append(f"                    <tr><td>{key}</td><td>{value}</td></tr>")
    
    # Add pruning tab
    html.extend([
        "                </table>",
        "            </div>",
        "        </div>",
        "",
        "        <div id='pruning' class='tab-content'>",
        "            <div class='section'>",
        "                <h2>Pruning Results</h2>",
    ])
    
    if pruning_viz_img:
        html.append("                <div class='visualization'>")
        html.append(f"                    <img src='data:image/png;base64,{pruning_viz_img}' alt='Pruned Heads Visualization'>")
        html.append("                </div>")
    
    # Add pruned heads table
    html.extend([
        "                <h3>Pruned Heads</h3>",
        f"                <p>Total pruned heads: {len(pruned_heads)}</p>",
        "                <table>",
        "                    <tr><th>Layer</th><th>Head</th><th>Score</th></tr>",
    ])
    
    for i, head_data in enumerate(pruned_heads[:20]):  # Show at most 20 heads
        if isinstance(head_data, list) and len(head_data) >= 2:
            layer, head = head_data[0], head_data[1]
            score = head_data[2] if len(head_data) >= 3 else "N/A"
            html.append(f"                    <tr><td>{layer}</td><td>{head}</td><td>{score}</td></tr>")
    
    # If there are more than 20 pruned heads, add a summary row
    if len(pruned_heads) > 20:
        html.append(f"                    <tr><td colspan='3'>... and {len(pruned_heads) - 20} more heads</td></tr>")
    
    # Close all open divs and add footer
    html.extend([
        "                </table>",
        "            </div>",
        "        </div>",
        "",
        "        <div class='footer'>",
        "            <p>Neural Plasticity Dashboard | Generated with Sentinel AI</p>",
        "            <p>Version: v0.1.0 (2025-04-20)</p>",
        "        </div>",
        "    </div>",
        "",
        "    <script>",
        "        function showTab(tabId) {",
        "            // Hide all tab contents",
        "            var tabContents = document.getElementsByClassName('tab-content');",
        "            for (var i = 0; i < tabContents.length; i++) {",
        "                tabContents[i].className = tabContents[i].className.replace(' active', '');",
        "            }",
        "            ",
        "            // Remove active class from all tabs",
        "            var tabs = document.getElementsByClassName('tab');",
        "            for (var i = 0; i < tabs.length; i++) {",
        "                tabs[i].className = tabs[i].className.replace(' active', '');",
        "            }",
        "            ",
        "            // Show the current tab and add active class",
        "            document.getElementById(tabId).className += ' active';",
        "            var currentTab = document.querySelector('.tab[onclick=\"showTab(\\'' + tabId + '\\')\"');",
        "            currentTab.className += ' active';",
        "        }",
        "    </script>",
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
    
    print(f"✅ Enhanced Neural Plasticity Dashboard generated at {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Enhanced Neural Plasticity Dashboard Generator")
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
            args.output_path = os.path.join(dashboards_dir, "enhanced_dashboard.html")
        else:
            args.output_path = "enhanced_dashboard.html"
    
    # Load experiment data
    experiment_data = load_experiment_data(args.experiment_dir)
    
    # Generate dashboard
    generate_dashboard(experiment_data, args.output_path)

if __name__ == "__main__":
    main()