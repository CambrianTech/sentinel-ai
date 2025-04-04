#!/usr/bin/env python
# coding: utf-8

# # Model Improvement Platform for Google Colab (v1.0.0)
# 
# This notebook provides a modular interface for configuring and running model improvement experiments in Google Colab.
# 
# ## Features
# - Interactive UI with dropdowns for parameter selection
# - Experiment configuration framework
# - Integration with existing improvement pipelines
# - Memory-efficient implementation for Colab environments
# - Real-time visualization of results
# 
# ## Usage
# 1. Upload to Colab using File > Upload notebook > Upload
# 2. Runtime > Change runtime type > Select GPU hardware accelerator
# 3. Run all cells to initialize the UI
# 4. Configure parameters and run experiments
# 
# ## Setup
# 
# First, let's install dependencies and clone the repository:

# %%
# Install required packages
print("🔧 Installing dependencies...")
!pip install -q jax jaxlib flax transformers matplotlib numpy pandas seaborn tqdm optax ipywidgets
# Install datasets explicitly with required version to ensure compatibility
!pip install -q 'datasets>=2.0.0' multiprocess

# %%
# Clone the repository 
# Use the feature/adaptive-plasticity branch which contains our improvements
print("📦 Cloning the repository...")
!git clone -b feature/adaptive-plasticity https://github.com/CambrianTech/sentinel-ai.git

# Create symlink for Colab compatibility
!ln -sf sentinel-ai refactor

# Change to the repository directory
print("📂 Changing to repository directory...")
%cd /content/sentinel-ai

# %%
# Import necessary libraries
import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import JAX/Flax
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

# Import Hugging Face libraries
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

# Add the current directory to path and import our modules
system_paths = [p for p in sys.path if '/usr/local' in p or 'python3' in p or 'site-packages' in p]
local_paths = ["."]  # Current directory first
other_paths = [p for p in sys.path if p not in system_paths and p != "."]
sys.path = system_paths + local_paths + other_paths

# Import core modules
from utils.pruning import (
    Environment,
    ResultsManager,
    PruningModule, 
    get_strategy,
    FineTuner,
    ImprovedFineTuner,
    PruningFineTuningExperiment
)
from utils.pruning.stability import patch_fine_tuner, optimize_fine_tuner
from utils.colab.helpers import setup_colab_environment, optimize_for_colab

# Set up plotting
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## Environment Detection and Setup
# 
# Let's detect our environment capabilities and setup the Colab environment:

# %%
# Initialize environment and detect capabilities
env_info = setup_colab_environment(prefer_gpu=True, verbose=True)
env = Environment()
env.print_info()

# Check JAX capabilities
print(f"\nJAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# %% [markdown]
# ## Model Improvement Experiment Runner
# 
# The ModularExperimentRunner class provides a consistent interface for configuring and running model improvement experiments.

# %%
class ModularExperimentRunner:
    """
    A modular framework for configuring and running model improvement experiments.
    
    This class provides a unified interface for different experiment types,
    making it easy to configure parameters through UI elements or programmatically.
    """
    
    def __init__(self):
        """Initialize the experiment runner with default configuration."""
        # Default configuration
        self.config = {
            # Model parameters
            "model": "distilgpt2",
            "model_size": "small",
            
            # Pruning parameters
            "enable_pruning": True,
            "pruning_strategy": "entropy",
            "pruning_level": 0.3,
            
            # Fine-tuning parameters
            "enable_fine_tuning": True,
            "fine_tuning_epochs": 2,
            "learning_rate": 5e-5,
            "batch_size": 4,
            "sequence_length": 128,
            
            # Adaptive parameters
            "enable_adaptive_plasticity": False,
            "plasticity_level": 0.5,
            "growth_rate": 0.1,
            
            # Stability parameters
            "stability_level": 1,
            "optimize_memory": True,
            
            # Environment parameters
            "detect_environment": True,
            
            # Experiment parameters
            "prompt": "Artificial intelligence will transform society by",
            "max_runtime": 3600,  # 1 hour default
            "results_dir": "improvement_results"
        }
        
        # Experiment instance
        self.experiment = None
        self.results = None
        
        # Model options categorized by size
        self.model_options = {
            "tiny": ["distilgpt2"],
            "small": ["gpt2", "facebook/opt-125m", "EleutherAI/pythia-160m"],
            "medium": ["gpt2-medium", "facebook/opt-350m", "EleutherAI/pythia-410m"],
            "large": ["gpt2-large", "facebook/opt-1.3b", "EleutherAI/pythia-1b"],
            "xl": ["gpt2-xl", "facebook/opt-2.7b"]
        }
        
        # Strategy options
        self.strategy_options = ["entropy", "magnitude", "random"]
        
        # Storage for previous configurations
        self.previous_configs = []
        
    def update_config(self, **kwargs):
        """Update configuration with new values."""
        # Save current config before updating
        self.previous_configs.append(self.config.copy())
        
        # Update config with new values
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
            else:
                print(f"Warning: Unknown configuration parameter '{key}'")
        
        return self.config
    
    def get_optimized_parameters(self):
        """Get optimized parameters based on model size and environment."""
        # Use the colab.helpers module to get optimized parameters
        model_size = self.config["model_size"]
        
        # Optimize parameters based on model size and available resources
        optimized_params = optimize_for_colab(
            model_size=model_size,
            prefer_stability=True if self.config["stability_level"] >= 2 else False,
            verbose=True
        )
        
        # Update configuration with optimized parameters
        self.config["batch_size"] = optimized_params["batch_size"]
        self.config["sequence_length"] = optimized_params["sequence_length"]
        self.config["stability_level"] = max(self.config["stability_level"], optimized_params["stability_level"])
        
        return optimized_params
    
    def create_experiment(self):
        """Create and configure the experiment instance."""
        # Create the experiment directory
        results_dir = self.config["results_dir"]
        os.makedirs(results_dir, exist_ok=True)
        
        # Determine experiment type based on configuration
        if self.config["enable_pruning"]:
            # Create pruning experiment
            self.experiment = PruningFineTuningExperiment(
                results_dir=results_dir,
                use_improved_fine_tuner=True,
                detect_environment=self.config["detect_environment"],
                optimize_memory=self.config["optimize_memory"],
                batch_size=self.config["batch_size"],
                sequence_length=self.config["sequence_length"],
                stability_level=self.config["stability_level"]
            )
            
            # Apply any adaptive plasticity configurations if enabled
            if self.config["enable_adaptive_plasticity"]:
                # Set adaptive parameters if the module exists
                try:
                    from utils.adaptive.adaptive_plasticity import configure_adaptive_plasticity
                    configure_adaptive_plasticity(
                        plasticity_level=self.config["plasticity_level"],
                        growth_rate=self.config["growth_rate"]
                    )
                    print(f"Configured adaptive plasticity with level {self.config['plasticity_level']}")
                except ImportError:
                    print("Warning: Adaptive plasticity module not found. This feature will be disabled.")
                    self.config["enable_adaptive_plasticity"] = False
        else:
            # Create a different type of experiment depending on configuration
            # For future expansion with other experiment types
            print("Currently only pruning experiments are supported")
            print("Creating a pruning experiment with minimal pruning")
            self.experiment = PruningFineTuningExperiment(
                results_dir=results_dir,
                use_improved_fine_tuner=True,
                detect_environment=self.config["detect_environment"],
                optimize_memory=self.config["optimize_memory"],
                batch_size=self.config["batch_size"],
                sequence_length=self.config["sequence_length"],
                stability_level=self.config["stability_level"]
            )
            
        return self.experiment
    
    def run_experiment(self):
        """Run the configured experiment."""
        if self.experiment is None:
            self.create_experiment()
        
        print(f"Running experiment with configuration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        
        # Define experiment parameters
        strategies = [self.config["pruning_strategy"]] if self.config["enable_pruning"] else []
        pruning_levels = [self.config["pruning_level"]] if self.config["enable_pruning"] else [0.0]
        fine_tuning_epochs = self.config["fine_tuning_epochs"] if self.config["enable_fine_tuning"] else 0
        
        # Run the experiment
        start_time = time.time()
        self.results = self.experiment.run_experiment(
            strategies=strategies,
            pruning_levels=pruning_levels,
            prompt=self.config["prompt"],
            fine_tuning_epochs=fine_tuning_epochs,
            max_runtime=self.config["max_runtime"],
            models=[self.config["model"]]
        )
        elapsed_time = time.time() - start_time
        
        print(f"Experiment completed in {elapsed_time/60:.2f} minutes")
        
        # Plot results
        self.experiment.plot_results(figsize=(16, 12))
        
        return self.results
    
    def create_ui(self):
        """Create an interactive UI for configuring and running experiments."""
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
            
            # Model selection widgets
            model_size_dropdown = widgets.Dropdown(
                options=list(self.model_options.keys()),
                value=self.config["model_size"],
                description='Model Size:',
                style={'description_width': 'initial'}
            )
            
            model_dropdown = widgets.Dropdown(
                options=self.model_options[self.config["model_size"]],
                value=self.config["model"],
                description='Model:',
                style={'description_width': 'initial'}
            )
            
            # Update model dropdown when model size changes
            def update_model_options(*args):
                model_dropdown.options = self.model_options[model_size_dropdown.value]
                model_dropdown.value = model_dropdown.options[0]
            
            model_size_dropdown.observe(update_model_options, names='value')
            
            # Pruning widgets
            pruning_checkbox = widgets.Checkbox(
                value=self.config["enable_pruning"],
                description='Enable Pruning',
                style={'description_width': 'initial'}
            )
            
            pruning_strategy_dropdown = widgets.Dropdown(
                options=self.strategy_options,
                value=self.config["pruning_strategy"],
                description='Pruning Strategy:',
                style={'description_width': 'initial'},
                disabled=not self.config["enable_pruning"]
            )
            
            pruning_level_slider = widgets.FloatSlider(
                value=self.config["pruning_level"],
                min=0.1,
                max=0.9,
                step=0.1,
                description='Pruning Level:',
                style={'description_width': 'initial'},
                disabled=not self.config["enable_pruning"]
            )
            
            # Update pruning widgets when checkbox changes
            def update_pruning_widgets(*args):
                pruning_strategy_dropdown.disabled = not pruning_checkbox.value
                pruning_level_slider.disabled = not pruning_checkbox.value
            
            pruning_checkbox.observe(update_pruning_widgets, names='value')
            
            # Fine-tuning widgets
            fine_tuning_checkbox = widgets.Checkbox(
                value=self.config["enable_fine_tuning"],
                description='Enable Fine-tuning',
                style={'description_width': 'initial'}
            )
            
            fine_tuning_epochs_slider = widgets.IntSlider(
                value=self.config["fine_tuning_epochs"],
                min=1,
                max=10,
                step=1,
                description='Epochs:',
                style={'description_width': 'initial'},
                disabled=not self.config["enable_fine_tuning"]
            )
            
            learning_rate_dropdown = widgets.Dropdown(
                options=[('1e-3', 1e-3), ('5e-4', 5e-4), ('1e-4', 1e-4), ('5e-5', 5e-5), ('1e-5', 1e-5)],
                value=self.config["learning_rate"],
                description='Learning Rate:',
                style={'description_width': 'initial'},
                disabled=not self.config["enable_fine_tuning"]
            )
            
            # Update fine-tuning widgets when checkbox changes
            def update_fine_tuning_widgets(*args):
                fine_tuning_epochs_slider.disabled = not fine_tuning_checkbox.value
                learning_rate_dropdown.disabled = not fine_tuning_checkbox.value
            
            fine_tuning_checkbox.observe(update_fine_tuning_widgets, names='value')
            
            # Adaptive plasticity widgets
            adaptive_checkbox = widgets.Checkbox(
                value=self.config["enable_adaptive_plasticity"],
                description='Enable Adaptive Plasticity',
                style={'description_width': 'initial'}
            )
            
            plasticity_level_slider = widgets.FloatSlider(
                value=self.config["plasticity_level"],
                min=0.1,
                max=1.0,
                step=0.1,
                description='Plasticity Level:',
                style={'description_width': 'initial'},
                disabled=not self.config["enable_adaptive_plasticity"]
            )
            
            growth_rate_slider = widgets.FloatSlider(
                value=self.config["growth_rate"],
                min=0.0,
                max=0.5,
                step=0.05,
                description='Growth Rate:',
                style={'description_width': 'initial'},
                disabled=not self.config["enable_adaptive_plasticity"]
            )
            
            # Update adaptive widgets when checkbox changes
            def update_adaptive_widgets(*args):
                plasticity_level_slider.disabled = not adaptive_checkbox.value
                growth_rate_slider.disabled = not adaptive_checkbox.value
            
            adaptive_checkbox.observe(update_adaptive_widgets, names='value')
            
            # Advanced settings widgets
            stability_level_dropdown = widgets.Dropdown(
                options=[(f"Level {i}", i) for i in range(1, 4)],
                value=self.config["stability_level"],
                description='Stability Level:',
                style={'description_width': 'initial'}
            )
            
            batch_size_dropdown = widgets.Dropdown(
                options=[1, 2, 4, 8, 16, 32],
                value=self.config["batch_size"],
                description='Batch Size:',
                style={'description_width': 'initial'}
            )
            
            sequence_length_dropdown = widgets.Dropdown(
                options=[32, 64, 128, 256, 512],
                value=self.config["sequence_length"],
                description='Sequence Length:',
                style={'description_width': 'initial'}
            )
            
            optimize_memory_checkbox = widgets.Checkbox(
                value=self.config["optimize_memory"],
                description='Optimize Memory Usage',
                style={'description_width': 'initial'}
            )
            
            # Experiment settings widgets
            prompt_text = widgets.Text(
                value=self.config["prompt"],
                description='Prompt:',
                style={'description_width': 'initial'}
            )
            
            max_runtime_dropdown = widgets.Dropdown(
                options=[
                    ('30 minutes', 1800),
                    ('1 hour', 3600),
                    ('2 hours', 7200),
                    ('4 hours', 14400),
                    ('8 hours', 28800),
                    ('12 hours', 43200),
                    ('24 hours', 86400)
                ],
                value=self.config["max_runtime"],
                description='Max Runtime:',
                style={'description_width': 'initial'}
            )
            
            results_dir_text = widgets.Text(
                value=self.config["results_dir"],
                description='Results Directory:',
                style={'description_width': 'initial'}
            )
            
            # Auto-optimize button
            auto_optimize_button = widgets.Button(
                description='🔧 Auto-Optimize Parameters',
                button_style='info',
                tooltip='Automatically optimize parameters based on model size and environment'
            )
            
            # Run button
            run_button = widgets.Button(
                description='🚀 Run Experiment',
                button_style='success',
                tooltip='Click to run the experiment with current settings'
            )
            
            # Output area
            output_area = widgets.Output()
            
            # Auto-optimize button handler
            def on_auto_optimize_clicked(b):
                # Update model size from dropdown
                self.update_config(model_size=model_size_dropdown.value)
                
                with output_area:
                    clear_output()
                    print("🔧 Auto-optimizing parameters...")
                    
                    # Get optimized parameters
                    optimized_params = self.get_optimized_parameters()
                    
                    # Update UI widgets
                    batch_size_dropdown.value = optimized_params["batch_size"]
                    sequence_length_dropdown.value = optimized_params["sequence_length"]
                    stability_level_dropdown.value = optimized_params["stability_level"]
                    
                    print("✅ Parameters optimized for current model and environment!")
                    print(f"  - Batch size: {optimized_params['batch_size']}")
                    print(f"  - Sequence length: {optimized_params['sequence_length']}")
                    print(f"  - Stability level: {optimized_params['stability_level']}")
                    
                    if optimized_params["use_fp16"]:
                        print("  - Mixed precision (FP16) enabled")
                    
                    if optimized_params["gradient_accumulation_steps"] > 1:
                        print(f"  - Gradient accumulation steps: {optimized_params['gradient_accumulation_steps']}")
            
            # Run button handler
            def on_run_button_clicked(b):
                # Update config from UI widgets
                self.update_config(
                    # Model parameters
                    model=model_dropdown.value,
                    model_size=model_size_dropdown.value,
                    
                    # Pruning parameters
                    enable_pruning=pruning_checkbox.value,
                    pruning_strategy=pruning_strategy_dropdown.value,
                    pruning_level=pruning_level_slider.value,
                    
                    # Fine-tuning parameters
                    enable_fine_tuning=fine_tuning_checkbox.value,
                    fine_tuning_epochs=fine_tuning_epochs_slider.value,
                    learning_rate=learning_rate_dropdown.value,
                    batch_size=batch_size_dropdown.value,
                    sequence_length=sequence_length_dropdown.value,
                    
                    # Adaptive parameters
                    enable_adaptive_plasticity=adaptive_checkbox.value,
                    plasticity_level=plasticity_level_slider.value,
                    growth_rate=growth_rate_slider.value,
                    
                    # Stability parameters
                    stability_level=stability_level_dropdown.value,
                    optimize_memory=optimize_memory_checkbox.value,
                    
                    # Experiment parameters
                    prompt=prompt_text.value,
                    max_runtime=max_runtime_dropdown.value,
                    results_dir=results_dir_text.value
                )
                
                with output_area:
                    clear_output()
                    print("🚀 Starting experiment...")
                    
                    # Create and run experiment
                    self.create_experiment()
                    self.run_experiment()
            
            # Connect buttons to handlers
            auto_optimize_button.on_click(on_auto_optimize_clicked)
            run_button.on_click(on_run_button_clicked)
            
            # Create tabs for different setting groups
            model_tab = widgets.VBox([
                widgets.HTML("<h3>Model Selection</h3>"),
                model_size_dropdown,
                model_dropdown
            ])
            
            pruning_tab = widgets.VBox([
                widgets.HTML("<h3>Pruning Settings</h3>"),
                pruning_checkbox,
                pruning_strategy_dropdown,
                pruning_level_slider
            ])
            
            fine_tuning_tab = widgets.VBox([
                widgets.HTML("<h3>Fine-tuning Settings</h3>"),
                fine_tuning_checkbox,
                fine_tuning_epochs_slider,
                learning_rate_dropdown
            ])
            
            adaptive_tab = widgets.VBox([
                widgets.HTML("<h3>Adaptive Plasticity Settings</h3>"),
                adaptive_checkbox,
                plasticity_level_slider,
                growth_rate_slider
            ])
            
            advanced_tab = widgets.VBox([
                widgets.HTML("<h3>Advanced Settings</h3>"),
                stability_level_dropdown,
                batch_size_dropdown,
                sequence_length_dropdown,
                optimize_memory_checkbox
            ])
            
            experiment_tab = widgets.VBox([
                widgets.HTML("<h3>Experiment Settings</h3>"),
                prompt_text,
                max_runtime_dropdown,
                results_dir_text
            ])
            
            # Create tabs
            tabs = widgets.Tab()
            tabs.children = [model_tab, pruning_tab, fine_tuning_tab, adaptive_tab, advanced_tab, experiment_tab]
            tabs.titles = ['Model', 'Pruning', 'Fine-tuning', 'Adaptive', 'Advanced', 'Experiment']
            
            # Assemble the full UI
            ui = widgets.VBox([
                widgets.HTML("<h2>Model Improvement Experiment</h2>"),
                tabs,
                widgets.HBox([auto_optimize_button, run_button]),
                output_area
            ])
            
            display(ui)
            
            # Display initial message
            with output_area:
                print("👋 Welcome to the Model Improvement Platform!")
                print("1. Configure your experiment using the tabs above")
                print("2. Click 'Auto-Optimize Parameters' to optimize for your environment")
                print("3. Click 'Run Experiment' to start the experiment")
            
            return ui
            
        except ImportError:
            print("Error: ipywidgets not available. Please install with 'pip install ipywidgets'")
            return None

# %% [markdown]
# ## Interactive Experiment Configuration UI
# 
# The UI below allows you to configure and run model improvement experiments interactively.

# %%
# Create the experiment runner
runner = ModularExperimentRunner()

# Create and display the UI
ui = runner.create_ui()

# %% [markdown]
# ## Manual Experiment Configuration
# 
# If you prefer to configure the experiment programmatically rather than using the UI, you can use the code below.
# 
# Uncomment and modify the parameters as needed:

# %%
# # Manual configuration
# runner.update_config(
#     # Model parameters
#     model="distilgpt2",
#     model_size="small",
#     
#     # Pruning parameters
#     enable_pruning=True,
#     pruning_strategy="entropy",
#     pruning_level=0.3,
#     
#     # Fine-tuning parameters
#     enable_fine_tuning=True,
#     fine_tuning_epochs=2,
#     
#     # Advanced parameters
#     batch_size=4,
#     sequence_length=128,
#     stability_level=2,
#     
#     # Experiment parameters
#     prompt="Artificial intelligence will transform society by",
#     max_runtime=3600,  # 1 hour
#     results_dir="manual_experiment_results"
# )
# 
# # Optimize parameters based on model size and environment
# runner.get_optimized_parameters()
# 
# # Create and run the experiment
# runner.create_experiment()
# runner.run_experiment()

# %% [markdown]
# ## Saving Results
# 
# After running experiments, you can save your configuration and results for future reference.

# %%
def save_experiment_config(runner, filename="experiment_config.json"):
    """Save the current experiment configuration to a JSON file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    
    # Create a serializable version of the config
    config_to_save = {k: (float(v) if isinstance(v, np.float32) else v) 
                     for k, v in runner.config.items()}
    
    # Add timestamp
    config_to_save["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save to file
    with open(filename, "w") as f:
        json.dump(config_to_save, f, indent=2)
    
    print(f"Configuration saved to {filename}")
    return filename

def load_experiment_config(runner, filename="experiment_config.json"):
    """Load an experiment configuration from a JSON file."""
    if not os.path.exists(filename):
        print(f"Configuration file {filename} not found")
        return None
    
    # Load from file
    with open(filename, "r") as f:
        config = json.load(f)
    
    # Remove timestamp if present
    if "timestamp" in config:
        del config["timestamp"]
    
    # Update runner config
    runner.update_config(**config)
    
    print(f"Configuration loaded from {filename}")
    return runner.config

# Example usage (uncomment to use):
# save_experiment_config(runner, "my_experiment_config.json")
# load_experiment_config(runner, "my_experiment_config.json")