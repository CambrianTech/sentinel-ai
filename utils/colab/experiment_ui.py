"""
Experiment UI for Google Colab

This module provides a modular interface for configuring and running model 
improvement experiments in Google Colab with interactive UI components.
"""

import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

# Local imports
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

# Import adaptive plasticity modules if available
try:
    from utils.adaptive.adaptive_plasticity import run_adaptive_system, AdaptivePlasticitySystem
    HAS_ADAPTIVE_PLASTICITY = True
except ImportError:
    HAS_ADAPTIVE_PLASTICITY = False
    print("Note: Adaptive plasticity modules not available in this version.")


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
            "plasticity_level": 0.2,
            "growth_ratio": 0.5,
            "training_steps": 100,
            "patience": 3,
            "memory_capacity": 10,
            "max_cycles": 5,
            "max_degeneration": 3.0,
            "max_perplexity_increase": 0.15,
            
            # Stability parameters
            "stability_level": 1,
            "optimize_memory": True,
            
            # Environment parameters
            "detect_environment": True,
            
            # Experiment parameters
            "prompt": "Artificial intelligence will transform society by",
            "max_runtime": 3600,  # 1 hour default
            "results_dir": "experiments/results/modular_experiment",
            "enable_visualization": True,
            "save_results": True
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
                        growth_rate=self.config["growth_ratio"]
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
        print(f"Running experiment with configuration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
            
        # Check if we should run adaptive plasticity experiment
        if self.config["enable_adaptive_plasticity"] and HAS_ADAPTIVE_PLASTICITY:
            return self._run_adaptive_experiment()
        else:
            # Run standard pruning/fine-tuning experiment
            if self.experiment is None:
                self.create_experiment()
                
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
            
            # Plot results if enabled
            if self.config["enable_visualization"]:
                self.experiment.plot_results(figsize=(16, 12))
            
            # Save results if enabled
            if self.config["save_results"]:
                results_file = os.path.join(self.config["results_dir"], f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                with open(results_file, 'w') as f:
                    # Handle non-serializable objects
                    import json
                    class NumpyEncoder(json.JSONEncoder):
                        def default(self, obj):
                            if isinstance(obj, np.integer):
                                return int(obj)
                            elif isinstance(obj, np.floating):
                                return float(obj)
                            elif isinstance(obj, np.ndarray):
                                return obj.tolist()
                            return super(NumpyEncoder, self).default(obj)
                            
                    json.dump({
                        "config": self.config,
                        "results": self.results if isinstance(self.results, (dict, list)) else str(self.results),
                        "timestamp": datetime.now().isoformat()
                    }, f, indent=2, cls=NumpyEncoder)
                    
                print(f"Results saved to {results_file}")
            
            return self.results
            
    def _run_adaptive_experiment(self):
        """Run an adaptive plasticity experiment."""
        print("Running adaptive plasticity experiment...")
        
        # Create output directory
        results_dir = self.config["results_dir"]
        os.makedirs(results_dir, exist_ok=True)
        
        # Prepare dataset
        print("Loading dataset...")
        try:
            from transformers import AutoTokenizer
            from sentinel_data.dataset_loader import load_dataset
            
            tokenizer = AutoTokenizer.from_pretrained(self.config["model"])
            dataset = load_dataset(
                dataset_name="tiny_shakespeare",  # Default dataset
                tokenizer=tokenizer,
                max_length=self.config["sequence_length"]
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Using dummy dataset for demonstration")
            # Create a simple dataset-like object
            class DummyDataset:
                def __getitem__(self, idx):
                    return {"input_ids": np.random.randint(0, 1000, size=128)}
                def __len__(self):
                    return 1000
            dataset = DummyDataset()
        
        # Start timing
        start_time = time.time()
        
        # Run adaptive system
        try:
            import torch
            system = run_adaptive_system(
                model_name=self.config["model"],
                dataset=dataset,
                output_dir=results_dir,
                max_cycles=self.config["max_cycles"],
                device="cuda" if torch.cuda.is_available() else "cpu",
                initial_pruning_level=self.config["plasticity_level"],
                initial_growth_ratio=self.config["growth_ratio"],
                initial_training_steps=self.config["training_steps"],
                patience=self.config["patience"],
                verbose=True
            )
            
            # Store results
            self.results = {
                "success": True,
                "output_dir": system.run_dir,
                "cycles_completed": system.metrics_logger.get_data_count("phase", "cycle_complete") if hasattr(system, "metrics_logger") else None,
                "final_head_count": len(system.current_active_heads) if hasattr(system, "current_active_heads") else None,
                "final_perplexity": system.current_perplexity if hasattr(system, "current_perplexity") else None,
            }
            
        except Exception as e:
            print(f"Error running adaptive plasticity experiment: {e}")
            import traceback
            traceback.print_exc()
            
            self.results = {
                "success": False,
                "error": str(e),
                "output_dir": results_dir
            }
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        print(f"Experiment completed in {elapsed_time/60:.2f} minutes")
        
        # Save experiment configuration
        if self.config["save_results"]:
            config_file = os.path.join(results_dir, f"adaptive_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(config_file, 'w') as f:
                import json
                json.dump({
                    "config": self.config,
                    "runtime_minutes": elapsed_time/60,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
                
            print(f"Configuration saved to {config_file}")
        
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
                max=0.9,
                step=0.1,
                description='Pruning Level:',
                style={'description_width': 'initial'},
                disabled=not self.config["enable_adaptive_plasticity"]
            )
            
            growth_ratio_slider = widgets.FloatSlider(
                value=self.config["growth_ratio"],
                min=0.0,
                max=0.9,
                step=0.1,
                description='Growth Ratio:',
                style={'description_width': 'initial'},
                disabled=not self.config["enable_adaptive_plasticity"]
            )
            
            training_steps_slider = widgets.IntSlider(
                value=self.config["training_steps"],
                min=50,
                max=500,
                step=50,
                description='Training Steps:',
                style={'description_width': 'initial'},
                disabled=not self.config["enable_adaptive_plasticity"]
            )
            
            max_cycles_slider = widgets.IntSlider(
                value=self.config["max_cycles"],
                min=1,
                max=20,
                step=1,
                description='Max Cycles:',
                style={'description_width': 'initial'},
                disabled=not self.config["enable_adaptive_plasticity"]
            )
            
            patience_slider = widgets.IntSlider(
                value=self.config["patience"],
                min=1,
                max=10,
                step=1,
                description='Patience:',
                style={'description_width': 'initial'},
                disabled=not self.config["enable_adaptive_plasticity"]
            )
            
            max_perplexity_slider = widgets.FloatSlider(
                value=self.config["max_perplexity_increase"],
                min=0.05,
                max=0.3,
                step=0.05,
                description='Max Perplexity Increase:',
                style={'description_width': 'initial'},
                disabled=not self.config["enable_adaptive_plasticity"]
            )
            
            # Update adaptive widgets when checkbox changes
            def update_adaptive_widgets(*args):
                adaptive_widgets = [
                    plasticity_level_slider,
                    growth_ratio_slider,
                    training_steps_slider,
                    max_cycles_slider,
                    patience_slider,
                    max_perplexity_slider
                ]
                for widget in adaptive_widgets:
                    widget.disabled = not adaptive_checkbox.value
            
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
                    growth_ratio=growth_ratio_slider.value,
                    training_steps=training_steps_slider.value,
                    max_cycles=max_cycles_slider.value,
                    patience=patience_slider.value,
                    max_perplexity_increase=max_perplexity_slider.value,
                    
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
                widgets.HTML("<h4>Pruning & Growth Parameters</h4>"),
                plasticity_level_slider,
                growth_ratio_slider,
                widgets.HTML("<h4>Training Parameters</h4>"),
                training_steps_slider,
                max_cycles_slider,
                patience_slider,
                widgets.HTML("<h4>Quality Control</h4>"),
                max_perplexity_slider
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


def launch_experiment_ui():
    """Launch the experiment UI."""
    # Setup environment
    print("Setting up environment...")
    env_info = setup_colab_environment(prefer_gpu=True, verbose=True)
    
    # Create runner
    runner = ModularExperimentRunner()
    
    # Create and display UI
    print("Launching UI...")
    ui = runner.create_ui()
    
    return runner