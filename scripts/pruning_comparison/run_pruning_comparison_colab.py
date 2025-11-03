#!/usr/bin/env python3
"""
Colab-ready Script for Running Pure Pruning Benchmark

This script is designed to run the pure pruning benchmark in Google Colab,
with all the necessary setup and integration features.

Features:
- Automatic Google Drive integration
- GPU detection and optimization
- Repository cloning and setup
- Results visualization
- Progress tracking

Usage in Colab:
1. !git clone https://github.com/yourusername/sentinel-ai.git
2. %cd sentinel-ai
3. !python scripts/pruning_comparison/run_pruning_comparison_colab.py
"""

import os
import sys
import time
import subprocess
import warnings
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
warnings.filterwarnings('ignore')

# Check if running in Colab
IN_COLAB = 'google.colab' in sys.modules
if not IN_COLAB:
    print("This script is designed to be run in Google Colab")
    print("For local execution, use the pure_pruning_benchmark.py script directly")
    sys.exit(1)

# Check for GPU
if IN_COLAB:
    gpu_info = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    if 'T4' in gpu_info:
        print("T4 GPU detected")
    elif 'GPU' in gpu_info:
        print("GPU detected, but not T4. Script is optimized for T4.")
    else:
        print("No GPU detected! This script requires a GPU runtime.")
        print("Go to Runtime > Change runtime type and select GPU")
        raise SystemError("No GPU detected")

# Setup the environment
print("Setting up the environment...")
!pip install -q torch transformers matplotlib seaborn tqdm pandas numpy fvcore
!pip install -q datasets

# Add project to Python path
import sys
sys.path.insert(0, os.getcwd())

# Create all necessary directories
os.makedirs("pure_pruning_results", exist_ok=True)

# Define the Colab interface widgets
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

def create_ui():
    """Create the UI for the benchmark configuration."""
    title_html = HTML(
        """
        <h1 style="color:#4CAF50; text-align:center;">
            Pure Pruning Benchmark for Sentinel-AI
        </h1>
        <p style="text-align:center; font-size:1.2em;">
            This notebook runs a comprehensive benchmark to measure the efficiency 
            benefits of pruning in isolation from agency features
        </p>
        <hr>
        """
    )
    display(title_html)
    
    # Model Configuration
    model_dropdown = widgets.Dropdown(
        options=['distilgpt2', 'gpt2', 'gpt2-medium'],
        value='distilgpt2',
        description='Model:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    
    # Training Configuration
    pruning_strategy = widgets.Dropdown(
        options=['gradual', 'one_shot', 'iterative'],
        value='gradual',
        description='Pruning Strategy:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    
    pruning_method = widgets.Dropdown(
        options=['entropy', 'random', 'magnitude'],
        value='entropy',
        description='Pruning Method:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    
    target_sparsity = widgets.FloatSlider(
        value=0.3,
        min=0.1,
        max=0.8,
        step=0.1,
        description='Target Sparsity:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    
    epochs = widgets.IntSlider(
        value=10,
        min=5,
        max=30,
        step=5,
        description='Training Epochs:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    
    # Advanced options
    advanced_options = widgets.Accordion(
        children=[
            widgets.VBox([
                widgets.FloatText(
                    value=5e-5,
                    description='Learning Rate:',
                    style={'description_width': 'initial'},
                    layout=widgets.Layout(width='50%')
                ),
                widgets.FloatText(
                    value=1e-5,
                    description='Post-Pruning LR:',
                    style={'description_width': 'initial'},
                    layout=widgets.Layout(width='50%')
                ),
                widgets.IntText(
                    value=4,
                    description='Batch Size:',
                    style={'description_width': 'initial'},
                    layout=widgets.Layout(width='50%')
                ),
                widgets.Checkbox(
                    value=True,
                    description='Measure FLOPs (requires fvcore)',
                    style={'description_width': 'initial'}
                ),
                widgets.Checkbox(
                    value=True,
                    description='Compare with other pruning methods',
                    style={'description_width': 'initial'}
                )
            ])
        ],
        selected_index=None
    )
    advanced_options.set_title(0, 'Advanced Options')
    
    # Drive mounting options
    drive_options = widgets.Accordion(
        children=[
            widgets.VBox([
                widgets.Checkbox(
                    value=True,
                    description='Mount Google Drive',
                    style={'description_width': 'initial'}
                ),
                widgets.Text(
                    value='SentinelAI_Results',
                    description='Drive Folder:',
                    style={'description_width': 'initial'},
                    layout=widgets.Layout(width='50%')
                )
            ])
        ],
        selected_index=None
    )
    drive_options.set_title(0, 'Google Drive Integration')
    
    # Run button
    run_button = widgets.Button(
        description='Run Benchmark',
        button_style='success',
        layout=widgets.Layout(width='30%')
    )
    
    # Create the UI layout
    display(widgets.VBox([
        widgets.HTML("<h3>Model Configuration</h3>"),
        model_dropdown,
        widgets.HTML("<h3>Pruning Configuration</h3>"),
        pruning_strategy,
        pruning_method,
        target_sparsity,
        epochs,
        advanced_options,
        drive_options,
        widgets.HTML("<br>"),
        run_button
    ]))
    
    # Button click handler
    def on_run_button_clicked(b):
        # Extract all configuration values
        config = {
            'model_name': model_dropdown.value,
            'pruning_strategy': pruning_strategy.value,
            'pruning_method': pruning_method.value,
            'target_sparsity': target_sparsity.value,
            'epochs': epochs.value,
            'learning_rate': advanced_options.children[0].children[0].value,
            'post_pruning_lr': advanced_options.children[0].children[1].value,
            'batch_size': advanced_options.children[0].children[2].value,
            'measure_flops': advanced_options.children[0].children[3].value,
            'compare_methods': advanced_options.children[0].children[4].value,
            'mount_drive': drive_options.children[0].children[0].value,
            'drive_folder': drive_options.children[0].children[1].value
        }
        
        # Clear output and show configuration
        clear_output()
        display(HTML(
            f"""
            <h2 style="color:#4CAF50; text-align:center;">
                Running Pure Pruning Benchmark
            </h2>
            <div style="text-align:center; padding: 10px; background-color: #f5f5f5; border-radius: 10px; margin: 10px;">
                <p><b>Model:</b> {config['model_name']}</p>
                <p><b>Pruning Strategy:</b> {config['pruning_strategy']}</p>
                <p><b>Pruning Method:</b> {config['pruning_method']}</p>
                <p><b>Target Sparsity:</b> {config['target_sparsity']}</p>
                <p><b>Epochs:</b> {config['epochs']}</p>
            </div>
            <p style="text-align:center; color:#666;">
                Please wait while the benchmark runs...
            </p>
            """
        ))
        
        # Run the benchmark with this configuration
        run_benchmark(config)
    
    run_button.on_click(on_run_button_clicked)

def run_benchmark(config):
    """Run the actual benchmark with the provided configuration."""
    # Set up Google Drive if requested
    if config['mount_drive']:
        from google.colab import drive
        drive.mount('/content/drive')
        
        # Create folder if it doesn't exist
        drive_path = f"/content/drive/My Drive/{config['drive_folder']}"
        os.makedirs(drive_path, exist_ok=True)
        print(f"✅ Mounted Google Drive. Results will be saved to: {drive_path}")
    
    # Import the pure pruning benchmark
    print("Importing pure pruning benchmark module...")
    from scripts.pure_pruning_benchmark import PruningBenchmark, parse_args
    
    # Create mock args to initialize the benchmark
    mock_args = parse_args([
        '--model_name', config['model_name'],
        '--pruning_strategy', config['pruning_strategy'],
        '--pruning_method', config['pruning_method'],
        '--target_sparsity', str(config['target_sparsity']),
        '--epochs', str(config['epochs']),
        '--learning_rate', str(config['learning_rate']),
        '--post_pruning_lr', str(config['post_pruning_lr']),
        '--batch_size', str(config['batch_size']),
        '--dataset', 'wikitext',
        '--max_length', '128',
        '--device', 'cuda',
        '--output_dir', './pure_pruning_results'
    ])
    
    if config['measure_flops']:
        mock_args.measure_flops = True
    
    if config['compare_methods']:
        mock_args.compare_methods = True
    
    # Create timer for tracking
    start_time = time.time()
    
    # Create and run the benchmark
    print("Initializing benchmark...")
    benchmark = PruningBenchmark(mock_args)
    benchmark.setup()
    
    # Run the benchmark with progress tracking
    print("🚀 Starting benchmark. This may take a while...")
    benchmark.run()
    
    # Calculate total time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"✅ Benchmark completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Generate additional visualizations for Colab
    create_colab_visualizations(benchmark.output_dir)
    
    # Copy results to Drive if requested
    if config['mount_drive']:
        drive_path = f"/content/drive/My Drive/{config['drive_folder']}"
        benchmark_name = os.path.basename(benchmark.output_dir)
        target_path = os.path.join(drive_path, benchmark_name)
        
        print(f"Copying results to Google Drive: {target_path}")
        !cp -r {benchmark.output_dir}/* {target_path}/
        print("✅ Results successfully copied to Google Drive")
    
    # Display summary of results
    display_summary(benchmark.output_dir)

def create_colab_visualizations(output_dir):
    """Create additional visualizations specifically for Colab."""
    try:
        # Load benchmark report
        report_path = os.path.join(output_dir, "benchmark_report.md")
        with open(report_path, 'r') as f:
            report_content = f.read()
        
        # Load metrics
        metrics_dir = os.path.join(output_dir, "metrics")
        metrics_files = [f for f in os.listdir(metrics_dir) if f.endswith('.json')]
        
        if not metrics_files:
            print("No metrics found to visualize")
            return
            
        metrics_path = os.path.join(metrics_dir, metrics_files[0])
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Create interactive visualizations
        plt.figure(figsize=(14, 10))
        
        # 1. Pruning progress visualization
        epochs = sorted([int(e) for e in metrics.get("epochs", [])])
        
        if "active_heads_percentage" in metrics:
            active_heads = [metrics["active_heads_percentage"].get(str(e), None) for e in epochs]
            active_heads = [x for x in active_heads if x is not None]
            
            if active_heads:
                plt.subplot(2, 2, 1)
                plt.plot(epochs[:len(active_heads)], active_heads, 'o-', color='#2196F3', linewidth=2)
                plt.title("Pruning Progress", fontsize=14)
                plt.xlabel("Epoch", fontsize=12)
                plt.ylabel("Active Heads (%)", fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
        
        # 2. Perplexity over time
        if "perplexity" in metrics:
            perplexity = [metrics["perplexity"].get(str(e), None) for e in epochs]
            perplexity = [x for x in perplexity if x is not None]
            
            if perplexity:
                plt.subplot(2, 2, 2)
                plt.plot(epochs[:len(perplexity)], perplexity, 'o-', color='#FF5722', linewidth=2)
                plt.title("Perplexity Over Time", fontsize=14)
                plt.xlabel("Epoch", fontsize=12)
                plt.ylabel("Perplexity", fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
        
        # 3. Inference latency
        if "inference_latency" in metrics:
            latency = [metrics["inference_latency"].get(str(e), None) for e in epochs]
            latency = [x for x in latency if x is not None]
            
            if latency:
                plt.subplot(2, 2, 3)
                plt.plot(epochs[:len(latency)], latency, 'o-', color='#4CAF50', linewidth=2)
                plt.title("Inference Latency", fontsize=14)
                plt.xlabel("Epoch", fontsize=12)
                plt.ylabel("ms/token", fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
        
        # 4. Text quality metrics
        if "lexical_diversity" in metrics and "repetition_score" in metrics:
            diversity = [metrics["lexical_diversity"].get(str(e), None) for e in epochs]
            repetition = [metrics["repetition_score"].get(str(e), None) for e in epochs]
            
            diversity = [x for x in diversity if x is not None]
            repetition = [x for x in repetition if x is not None]
            
            if diversity and repetition:
                plt.subplot(2, 2, 4)
                
                # Align the arrays to the same length
                min_len = min(len(diversity), len(repetition))
                
                plt.plot(epochs[:min_len], diversity[:min_len], 'o-', color='#9C27B0', 
                         linewidth=2, label="Diversity (higher is better)")
                plt.plot(epochs[:min_len], repetition[:min_len], 'o-', color='#FFC107', 
                         linewidth=2, label="Repetition (lower is better)")
                
                plt.title("Text Quality Metrics", fontsize=14)
                plt.xlabel("Epoch", fontsize=12)
                plt.ylabel("Score", fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
        
        plt.tight_layout()
        
        # Save the interactive visualizations
        plt.savefig(os.path.join(output_dir, "colab_visualizations.png"), dpi=150)
        plt.close()
        
        print("✅ Created additional visualizations for Colab")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return None

def display_summary(output_dir):
    """Display a summary of the benchmark results."""
    # Load benchmark report
    report_path = os.path.join(output_dir, "benchmark_report.md")
    with open(report_path, 'r') as f:
        report_content = f.read()
    
    # Display report as HTML
    from IPython.display import Markdown
    display(Markdown(report_content))
    
    # Display visualizations
    dashboard_path = os.path.join(output_dir, "summary_dashboard.png")
    if os.path.exists(dashboard_path):
        from IPython.display import Image
        display(Image(dashboard_path))
    
    # Additional visualizations
    colab_viz_path = os.path.join(output_dir, "colab_visualizations.png")
    if os.path.exists(colab_viz_path):
        from IPython.display import Image
        display(Image(colab_viz_path))
    
    # If method comparison was done, show those results
    comparison_dir = os.path.join(output_dir, "method_comparison")
    if os.path.exists(comparison_dir):
        radar_path = os.path.join(comparison_dir, "radar_comparison.png")
        if os.path.exists(radar_path):
            display(HTML("<h3>Pruning Methods Comparison</h3>"))
            from IPython.display import Image
            display(Image(radar_path))

def main():
    """Main function to run when the script is executed."""
    # Display a welcome message
    print("="*80)
    print("Pure Pruning Benchmark for Sentinel-AI".center(80))
    print("Colab Integration Version".center(80))
    print("="*80)
    print("\nSetting up the interface...")
    
    # Create and display the UI
    create_ui()

if __name__ == "__main__":
    main()