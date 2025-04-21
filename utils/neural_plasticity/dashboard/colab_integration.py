"""
Neural Plasticity Colab Dashboard Integration

This module provides utilities for integrating the neural plasticity
experiment dashboard with Google Colab notebooks, including collaboration
features for sharing dashboards.

Version: v0.0.2 (2025-04-20 19:45:00)
"""

import sys
import os
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

# Check if running in Colab
IS_COLAB = False
try:
    import google.colab
    from IPython.display import display, HTML, Javascript
    IS_COLAB = True
except ImportError:
    pass

logger = logging.getLogger(__name__)

def setup_colab_environment(install_wandb: bool = True):
    """
    Set up the Colab environment for neural plasticity experiments.
    
    Args:
        install_wandb: Whether to install wandb if it's not already installed
        
    Returns:
        bool: True if setup was successful
    """
    if not IS_COLAB:
        logger.warning("Not running in Colab environment")
        return False
    
    try:
        # Display welcome message
        display(HTML("""
        <div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h2 style="margin-top: 0; color: #3f51b5;">Neural Plasticity Dashboard Setup</h2>
            <p>Setting up the environment for running neural plasticity experiments with real-time dashboard visualization.</p>
        </div>
        """))
        
        # Check if wandb is installed
        wandb_installed = False
        try:
            import wandb
            wandb_installed = True
        except ImportError:
            wandb_installed = False
        
        # Install wandb if needed
        if not wandb_installed and install_wandb:
            print("📦 Installing Weights & Biases for dashboard visualization...")
            from IPython.display import clear_output
            try:
                from google.colab import output
                output.clear()
            except:
                clear_output()
                
            # Install wandb (using subprocess instead of notebook magic)
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb", "-q"])
            
            # Import after installation
            import wandb
            wandb_installed = True
            
            print("✅ Weights & Biases (wandb) installed successfully!")
        
        # Prompt for wandb login if needed
        if wandb_installed:
            try:
                wandb.ensure_login()
                print("🔑 Already logged in to Weights & Biases")
            except:
                print("\n🔑 Please log in to Weights & Biases to enable the dashboard:")
                wandb.login()
        
        # Display successful setup message
        display(HTML("""
        <div style="background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 5px solid #4caf50;">
            <h3 style="margin-top: 0; color: #2e7d32;">Environment Ready!</h3>
            <p>Your Colab environment is ready for running neural plasticity experiments with real-time dashboard visualization.</p>
        </div>
        """))
        
        return True
    except Exception as e:
        logger.error(f"Error setting up Colab environment: {e}")
        print(f"⚠️ Error setting up Colab environment: {e}")
        return False

def create_experiment_dashboard_cell(project_name: str = "neural-plasticity"):
    """
    Create a cell with code to initialize a wandb dashboard for the experiment.
    
    Args:
        project_name: Name of the wandb project
        
    Returns:
        str: Code to initialize the dashboard
    """
    if not IS_COLAB:
        return "# Not running in Colab environment"
    
    code = f"""
# Initialize Neural Plasticity Dashboard with Weights & Biases
import wandb
from datetime import datetime
import os

# Create unique name for this experiment
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"colab-experiment-{timestamp}"

# Configuration parameters for the dashboard
config = {{
    "model_name": model_name,  # Use model_name from notebook variables
    "pruning_strategy": pruning_strategy,  # Use pruning_strategy from notebook
    "pruning_level": pruning_level,  # Use pruning_level from notebook
    "environment": "colab"
}}

# Initialize wandb with your project
run = wandb.init(
    project="{project_name}",
    name=experiment_name,
    config=config
)

print(f"📊 Dashboard initialized! View at: {{wandb.run.url}}")

# Create WandbDashboard instance
from utils.neural_plasticity.dashboard.wandb_integration import WandbDashboard

dashboard = WandbDashboard(
    project_name="{project_name}",
    experiment_name=experiment_name,
    config=config,
    mode="online"
)

# Set initial phase
dashboard.set_phase("setup")

# Get callbacks for later use
metrics_callback = dashboard.get_metrics_callback()
sample_callback = dashboard.get_sample_callback()
"""
    
    display(HTML(f"""
    <div style="background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 5px solid #2196f3;">
        <h3 style="margin-top: 0; color: #0d47a1;">Dashboard Initialization Code</h3>
        <p>Copy the following code into a new cell to initialize the neural plasticity dashboard:</p>
    </div>
    """))
    
    print(code)
    
    return code

def get_experiment_parameters_cell():
    """
    Returns code for a cell that sets default experiment parameters.
    """
    if not IS_COLAB:
        return "# Not running in Colab environment"
    
    code = """
# Set Neural Plasticity Experiment Parameters
# You can modify these parameters to customize your experiment

# Model parameters
model_name = "distilgpt2"  # Model to use (e.g., distilgpt2, gpt2, facebook/opt-125m)

# Pruning parameters
pruning_strategy = "entropy"  # Options: entropy, magnitude, random, combined
pruning_level = 0.2  # Percentage of heads to prune (0.0 to 1.0)
cycles = 3  # Number of pruning cycles to run

# Training parameters
learning_rate = 5e-5
batch_size = 4  # Use a small batch size for Colab
max_length = 128
training_steps = 100  # Use a smaller number for quicker experiments in Colab

# Dataset parameters
dataset = "wikitext"
dataset_config = "wikitext-2-raw-v1"

# Output parameters
use_dashboard = True  # Enable Weights & Biases dashboard

print("✅ Experiment parameters set!")
"""
    
    display(HTML(f"""
    <div style="background-color: #fff8e1; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 5px solid #ffc107;">
        <h3 style="margin-top: 0; color: #ff6f00;">Experiment Parameters</h3>
        <p>Copy the following code into a new cell to set up experiment parameters:</p>
    </div>
    """))
    
    print(code)
    
    return code

def get_run_experiment_cell():
    """
    Returns code for a cell that runs the neural plasticity experiment.
    """
    if not IS_COLAB:
        return "# Not running in Colab environment"
    
    code = """
# Run Neural Plasticity Experiment with Dashboard
from utils.neural_plasticity.experiment import NeuralPlasticityExperiment

# Create experiment output directory
import os
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"/content/neural_plasticity_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Initialize experiment
experiment = NeuralPlasticityExperiment(
    # Model parameters
    model_name=model_name,
    device="cuda" if torch.cuda.is_available() else "cpu",
    
    # Dataset parameters
    dataset=dataset,
    dataset_config=dataset_config,
    batch_size=batch_size,
    max_length=max_length,
    
    # Pruning parameters
    pruning_strategy=pruning_strategy,
    pruning_level=pruning_level,
    learning_rate=learning_rate,
    
    # Output parameters
    output_dir=output_dir,
    save_results=True,
    show_samples=True,
    
    # Dashboard parameters
    use_dashboard=use_dashboard,
    
    # Callbacks for wandb dashboard
    metrics_callback=metrics_callback if use_dashboard else None,
    sample_callback=sample_callback if use_dashboard else None
)

# Run full experiment
print("Starting neural plasticity experiment...")
dashboard.set_phase("warmup")  # Update dashboard phase

# Run the experiment
results = experiment.run_full_experiment(
    warmup_epochs=1,
    pruning_cycles=cycles,
    training_steps=training_steps
)

# Display results summary
print(f"\\nExperiment completed!")
print(f"Baseline perplexity: {results['baseline_metrics']['perplexity']:.2f}")
print(f"Final perplexity: {results['final_metrics']['perplexity']:.2f}")
print(f"Improvement: {results['improvement_percent']:.2f}%")
print(f"Pruned heads: {len(results['pruned_heads'])}")
print(f"\\nFull results saved to: {output_dir}")

# Finish wandb run
if use_dashboard:
    wandb.finish()
"""
    
    display(HTML(f"""
    <div style="background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 5px solid #4caf50;">
        <h3 style="margin-top: 0; color: #2e7d32;">Run Experiment</h3>
        <p>Copy the following code into a new cell to run the neural plasticity experiment:</p>
    </div>
    """))
    
    print(code)
    
    return code

def create_colab_quickstart_notebook():
    """
    Create a complete quickstart notebook for running neural plasticity
    experiments in Colab with dashboard visualization.
    """
    if not IS_COLAB:
        logger.warning("Not running in Colab environment")
        return False
    
    # Create cells for the notebook
    cells = [
        {
            "title": "Setup Environment",
            "description": "Install and set up necessary packages",
            "code": """
# Setup Neural Plasticity Environment
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "wandb", "transformers", "datasets", "torch"])

# Setup imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import wandb for dashboard
import wandb
"""
        },
        {
            "title": "Set Experiment Parameters",
            "description": "Configure the neural plasticity experiment",
            "code": get_experiment_parameters_cell()
        },
        {
            "title": "Initialize Dashboard",
            "description": "Set up real-time experiment dashboard",
            "code": create_experiment_dashboard_cell(project_name="neural-plasticity")
        },
        {
            "title": "Run Experiment",
            "description": "Execute the neural plasticity experiment with dashboard visualization",
            "code": get_run_experiment_cell()
        },
        {
            "title": "Visualize Results",
            "description": "Display additional visualizations of results",
            "code": """
# Visualize experiment results
import matplotlib.pyplot as plt

# Create a bar chart comparing baseline and final perplexity
plt.figure(figsize=(10, 6))
baseline = results['baseline_metrics']['perplexity']
final = results['final_metrics']['perplexity']
improvement = results['improvement_percent']

plt.bar(['Baseline', 'After Pruning'], [baseline, final], color=['blue', 'green'])
plt.title(f'Perplexity Improvement: {improvement:.2f}%')
plt.ylabel('Perplexity (lower is better)')
plt.grid(axis='y', alpha=0.3)

# Add values on top of bars
for i, v in enumerate([baseline, final]):
    plt.text(i, v + 0.5, f'{v:.2f}', ha='center')

plt.show()

# Log the figure to wandb if enabled
if use_dashboard:
    wandb.log({"final_perplexity_comparison": wandb.Image(plt)})
"""
        }
    ]
    
    # Display notebook creation message
    display(HTML(f"""
    <div style="background-color: #f3e5f5; padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h2 style="margin-top: 0; color: #6a1b9a;">Neural Plasticity Quickstart Notebook</h2>
        <p>Here's a complete notebook for running neural plasticity experiments with dashboard visualization:</p>
    </div>
    """))
    
    # Display each cell
    for i, cell in enumerate(cells):
        display(HTML(f"""
        <div style="background-color: #ede7f6; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 5px solid #673ab7;">
            <h3 style="margin-top: 0; color: #4527a0;">Cell {i+1}: {cell['title']}</h3>
            <p>{cell['description']}</p>
        </div>
        """))
        
        print(f"# {cell['title']}")
        print(cell['code'])
        print("\n" + "-"*80 + "\n")
    
    return True

def monitor_experiment_in_colab(experiment_url: Optional[str] = None):
    """
    Display a monitoring dashboard for a running neural plasticity experiment in Colab.
    
    Args:
        experiment_url: Optional URL to an existing wandb experiment
        
    Returns:
        bool: True if dashboard was displayed successfully
    """
    if not IS_COLAB:
        logger.warning("Not running in Colab environment")
        return False
    
    try:
        # Check if wandb is installed
        try:
            import wandb
        except ImportError:
            print("⚠️ Weights & Biases (wandb) is not installed. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb", "-q"])
            import wandb
        
        # Create dashboard UI
        if experiment_url:
            # Display iframe to existing experiment
            dashboard_html = f"""
            <div style="width: 100%; height: 800px; overflow: hidden; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <iframe src="{experiment_url}" width="100%" height="100%" frameborder="0"></iframe>
            </div>
            """
        else:
            # Display dashboard for current experiment
            # Check if there's an active run
            if wandb.run is None:
                print("⚠️ No active wandb run found. Please initialize wandb first.")
                return False
            
            dashboard_html = f"""
            <div style="width: 100%; height: 800px; overflow: hidden; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <iframe src="{wandb.run.url}" width="100%" height="100%" frameborder="0"></iframe>
            </div>
            """
        
        # Display the dashboard
        display(HTML(dashboard_html))
        
        dashboard_url = experiment_url or wandb.run.url
        print(f"🔗 Dashboard URL: {dashboard_url}")
        
        # Show sharing options
        display_sharing_options(wandb.run)
        
        return True
    except Exception as e:
        logger.error(f"Error displaying experiment dashboard: {e}")
        print(f"⚠️ Error displaying experiment dashboard: {e}")
        return False
        
# Collaboration functions

def create_shareable_link(run_url: str) -> str:
    """
    Create a shareable link for the wandb dashboard with appropriate permissions.
    
    Args:
        run_url: URL to the wandb run
        
    Returns:
        Shareable link with read-only permissions
    """
    if not run_url:
        return "No dashboard URL available to share"
    
    # Add parameters to make the link shareable
    return f"{run_url}?shareToken=view"

def display_sharing_options(wandb_run):
    """
    Display sharing options for the dashboard in Colab.
    
    Args:
        wandb_run: Active wandb run
    """
    if not IS_COLAB or not wandb_run or not hasattr(wandb_run, 'url'):
        return
    
    shareable_link = create_shareable_link(wandb_run.url)
    
    share_html = f"""
    <div style="background-color: #f9f9f9; border-left: 5px solid #2196F3; padding: 15px; margin: 20px 0; border-radius: 4px;">
        <h3 style="color: #2196F3; margin-top: 0;">Dashboard Sharing Options</h3>
        <p>Share your neural plasticity experiment dashboard with collaborators:</p>
        
        <div style="background-color: white; padding: 10px; border-radius: 4px; border: 1px solid #ddd; margin: 10px 0;">
            <div style="font-weight: bold; margin-bottom: 8px;">Shareable Link:</div>
            <input type="text" value="{shareable_link}" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;" readonly
                onclick="this.select(); document.execCommand('copy'); this.parentNode.querySelector('.copy-notice').style.display = 'inline-block';">
            <div class="copy-notice" style="display: none; color: green; margin-top: 5px;">✓ Link copied to clipboard</div>
        </div>
        
        <div style="margin-top: 15px;">
            <p><strong>To collaborate with anyone:</strong></p>
            <ol>
                <li>Share the link above with your collaborators</li>
                <li>They can view your dashboard without needing wandb accounts</li>
                <li>Dashboard updates in real-time as your experiment progresses</li>
            </ol>
        </div>
    </div>
    """
    
    display(HTML(share_html))
    
def setup_ngrok_tunnel(port=8080, dashboard_dir=None):
    """
    Set up an ngrok tunnel for local dashboard collaboration.
    
    Args:
        port: Port for the wandb server
        dashboard_dir: Directory containing the wandb data
        
    Returns:
        URL to the ngrok tunnel
    """
    try:
        # Check if ngrok is installed
        try:
            from pyngrok import ngrok
        except ImportError:
            if IS_COLAB:
                print("Installing ngrok for dashboard sharing...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
                from pyngrok import ngrok
        
        import subprocess
        import threading
        import time
        
        # Start wandb server in a separate thread
        def start_server():
            cmd = f"wandb server --port={port}"
            if dashboard_dir:
                cmd += f" --directory={dashboard_dir}"
            subprocess.Popen(cmd, shell=True)
        
        server_thread = threading.Thread(target=start_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Give the server time to start
        time.sleep(3)
        
        # Start ngrok tunnel
        public_url = ngrok.connect(port)
        
        # Display shareable link
        share_html = f"""
        <div style="background-color: #f9f9f9; border-left: 5px solid #2196F3; padding: 15px; margin: 20px 0; border-radius: 4px;">
            <h3 style="color: #2196F3; margin-top: 0;">Local Dashboard Tunnel (Temporary)</h3>
            <p>Share this temporary link with collaborators to view your local dashboard:</p>
            
            <div style="background-color: white; padding: 10px; border-radius: 4px; border: 1px solid #ddd; margin: 10px 0;">
                <div style="font-weight: bold; margin-bottom: 8px;">Temporary Tunnel URL:</div>
                <input type="text" value="{public_url}" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;" readonly
                    onclick="this.select(); document.execCommand('copy'); this.parentNode.querySelector('.copy-notice').style.display = 'inline-block';">
                <div class="copy-notice" style="display: none; color: green; margin-top: 5px;">✓ Link copied to clipboard</div>
            </div>
            
            <div style="margin-top: 15px;">
                <p><strong>Important Notes:</strong></p>
                <ul>
                    <li>This link is <em>temporary</em> and will stop working when this session ends</li>
                    <li>No wandb account is required to view this dashboard</li>
                    <li>This tunnel exposes your local wandb server to the internet</li>
                </ul>
            </div>
        </div>
        """
        
        if IS_COLAB:
            display(HTML(share_html))
        
        return public_url
        
    except Exception as e:
        print(f"Failed to set up ngrok tunnel: {e}")
        return None