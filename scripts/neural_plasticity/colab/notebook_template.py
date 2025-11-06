"""
Neural Plasticity Demo: Dynamic Pruning & Regrowth (v0.1.0 (2025-04-20 13:30:00))

This notebook demonstrates Sentinel AI's neural plasticity system, which allows transformer models 
to dynamically prune and regrow attention heads during training based on utility metrics.

This template uses the modular neural plasticity implementation from the Sentinel AI codebase,
ensuring consistent behavior between Colab and local environments.
"""

#@title Setup and Installation
# Install required packages
%%capture
!pip install -q torch transformers datasets matplotlib numpy tqdm ipywidgets

# Install from GitHub repository, using the proper branch
!pip install -q git+https://github.com/CambrianTech/sentinel-ai.git@feature/implement-adaptive-plasticity

# Imports for notebook
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from IPython.display import HTML, display

# Import our neural plasticity modules
from scripts.neural_plasticity.run_experiment import main as run_neural_plasticity
from scripts.neural_plasticity.colab.integration import (
    is_colab, 
    is_apple_silicon, 
    has_gpu, 
    get_environment_info,
    get_output_dir, 
    install_requirements
)
from scripts.neural_plasticity.colab.visualization import (
    initialize_visualization,
    plot_entropy_heatmap,
    visualize_pruned_heads,
    display_metrics_comparison
)
from scripts.neural_plasticity.colab.progress import ProgressTracker
from scripts.neural_plasticity.colab.interactive import (
    PruningStrategySelector,
    ModelSelector,
    create_experiment_config_widget
)
from scripts.neural_plasticity.visualization.dashboard_generator import (
    generate_dashboard,
    generate_dashboard_url
)

#@title Initialize Environment
# Set up the environment
initialize_visualization()

# Get environment info
env_info = get_environment_info()
print(f"🔍 Environment detected:")
print(f"  • Running in {'Google Colab' if env_info['is_colab'] else 'local environment'}")
print(f"  • {'Apple Silicon detected' if env_info['is_apple_silicon'] else 'Standard CPU architecture'}")
print(f"  • {'GPU available: ' + env_info.get('gpu_type', 'Unknown') if env_info['has_gpu'] else 'No GPU detected'}")

# Set up output directory
output_dir = get_output_dir("neural_plasticity_demo")
print(f"📂 Results will be saved to: {output_dir}")

#@title Configure Experiment
# Create configuration widget
config_widget = create_experiment_config_widget()

# Get configuration
config = config_widget["config"]

# Function to run when button is clicked
def run_experiment_on_click(b):
    # Disable button while running
    b.disabled = True
    b.description = "⏳ Running..."
    
    try:
        # Get current configuration
        model_path = config["model_path"]
        pruning_strategy = config["pruning_strategy"]
        pruning_level = config["pruning_level"]
        
        print(f"🚀 Running experiment with:")
        print(f"  • Model: {model_path}")
        print(f"  • Pruning Strategy: {pruning_strategy}")
        print(f"  • Pruning Level: {pruning_level}")
        
        # Run the experiment using our consolidated implementation
        results = run_neural_plasticity(
            model_name=model_path,
            pruning_strategy=pruning_strategy,
            pruning_level=pruning_level,
            output_dir=output_dir,
            quick_test=True  # For demo purposes, change to False for a full run
        )
        
        # Display results
        display(HTML("<h2>Experiment Results</h2>"))
        
        # Display metrics comparison
        baseline_metrics = results.get('metrics', {}).get('baseline', {})
        final_metrics = results.get('metrics', {}).get('final', {})
        display_metrics_comparison(baseline_metrics, final_metrics)
        
        # Generate and display dashboard
        dashboard_path = os.path.join(output_dir, "dashboards", "dashboard.html")
        generate_dashboard(
            experiment_dir=output_dir,
            output_path=dashboard_path,
            model_name=model_path,
            pruning_strategy=pruning_strategy,
            pruning_level=pruning_level
        )
        
        # Display link to dashboard
        dashboard_url = generate_dashboard_url(dashboard_path)
        if is_colab():
            print("Dashboard downloaded. Open the HTML file to view it.")
        else:
            print(f"Dashboard generated at: {dashboard_url}")
        
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
    
    # Re-enable button
    b.disabled = False
    b.description = "🚀 Run Experiment"

# Attach callback to button
config_widget["run_button"].on_click(run_experiment_on_click)

#@title Advanced: Run with Custom Parameters
# This cell allows running with custom parameters using code instead of widgets
def run_custom_experiment():
    # Define custom parameters
    params = {
        "model_name": "distilgpt2",  # Change as needed
        "pruning_strategy": "entropy",
        "pruning_level": 0.3,
        "batch_size": 2,
        "max_length": 64,
        "fine_tuning_steps": 100,
        "device": None,  # Auto-detect
        "quick_test": True  # Set to False for full experiment
    }
    
    # Create custom output directory
    custom_output_dir = get_output_dir("custom_experiment")
    
    # Run experiment
    with ProgressTracker("Running custom experiment", total=100) as tracker:
        # Run the experiment
        tracker.update(1, {"status": "Starting"})
        
        results = run_neural_plasticity(
            model_name=params["model_name"],
            pruning_strategy=params["pruning_strategy"],
            pruning_level=params["pruning_level"],
            batch_size=params["batch_size"],
            max_length=params["max_length"],
            fine_tuning_steps=params["fine_tuning_steps"],
            output_dir=custom_output_dir,
            device=params["device"],
            quick_test=params["quick_test"]
        )
        
        tracker.update(100, {"status": "Complete"})
    
    # Generate dashboard
    dashboard_path = os.path.join(custom_output_dir, "dashboards", "dashboard.html")
    generate_dashboard(
        experiment_dir=custom_output_dir,
        output_path=dashboard_path,
        model_name=params["model_name"],
        pruning_strategy=params["pruning_strategy"],
        pruning_level=params["pruning_level"]
    )
    
    return results, dashboard_path

# Uncomment to run with custom parameters
# results, dashboard_path = run_custom_experiment()

#@title View Results from Previous Run
# This cell allows loading and visualizing results from a previous run
def view_previous_results():
    from scripts.neural_plasticity.colab.integration import load_experiment_results
    
    # Ask for directory path
    directory_path = input("Enter the path to the experiment directory: ")
    
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return
    
    # Try to load results
    try:
        results = load_experiment_results(directory_path)
        
        # Display metrics
        if "metrics" in results:
            baseline = results["metrics"].get("baseline", {})
            final = results["metrics"].get("final", {})
            
            print(f"📊 Experiment Results:")
            display_metrics_comparison(baseline, final)
        
        # Generate dashboard
        dashboard_path = os.path.join(directory_path, "dashboard.html")
        generate_dashboard(
            experiment_dir=directory_path,
            output_path=dashboard_path,
            model_name=results.get("params", {}).get("model_name", "Unknown"),
            pruning_strategy=results.get("params", {}).get("pruning_strategy", "Unknown"),
            pruning_level=results.get("params", {}).get("pruning_level", 0.0)
        )
        
        # Display link to dashboard
        dashboard_url = generate_dashboard_url(dashboard_path)
        if is_colab():
            print("Dashboard downloaded. Open the HTML file to view it.")
        else:
            print(f"Dashboard generated at: {dashboard_url}")
            
    except Exception as e:
        print(f"Error loading results: {e}")

# Uncomment to view previous results
# view_previous_results()