#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Upgrayedd API Notebook (v0.1.0)

This notebook demonstrates how to use the Upgrayedd API to optimize transformer models
through pruning and fine-tuning.

Features:
1. Load and prune transformer models using different strategies
2. Fine-tune pruned models to recover or improve performance
3. Visualize the process and results
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Install required packages
try:
    import transformers
    import datasets
    import tqdm
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([
        "pip", "install", "-q", 
        "transformers>=4.30.0", 
        "datasets>=2.14.0", 
        "torch>=2.0.0", 
        "matplotlib"
    ])

# Add project root to path for imports
if os.path.exists('/content'):
    # We're in Colab
    project_root = '/content/sentinel-ai'
    
    # Clone the repo if not already cloned
    if not os.path.exists(project_root):
        import subprocess
        print("Cloning repository...")
        subprocess.check_call([
            "git", "clone", 
            "https://github.com/yourusername/sentinel-ai.git", 
            project_root
        ])
else:
    # We're running locally
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir.endswith('colab_notebooks'):
        project_root = os.path.dirname(current_dir)
    else:
        project_root = os.getcwd()
        if not os.path.basename(project_root) == 'sentinel-ai':
            parent_dir = os.path.dirname(project_root)
            if os.path.basename(parent_dir) == 'sentinel-ai':
                project_root = parent_dir

# Add the project root to the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

# ==========================================================================
# Setup: Import the Upgrayedd API
# ==========================================================================

try:
    # Import Upgrayedd API
    from sentinel.upgrayedd.optimizer import AdaptiveOptimizer, AdaptiveOptimizerConfig
    print("✅ Successfully imported Upgrayedd API")
except ImportError as e:
    print(f"⚠️ Failed to import Upgrayedd API: {e}")
    print("Using fallback implementation...")
    
    # Fallback implementation
    class AdaptiveOptimizerConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class AdaptiveOptimizer:
        def __init__(self, config):
            self.config = config
            print("Using fallback AdaptiveOptimizer implementation!")
        
        def run_continuous_optimization(self, max_cycles=1):
            print(f"Would run {max_cycles} cycles of optimization")
            return {"improvement": 0.0}

# ==========================================================================
# Configuration
# ==========================================================================

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create configuration
config = {
    # Model
    "model_name": "distilgpt2",  # Change to your preferred model
    
    # Pruning
    "pruning_ratio": 0.3,      # Percentage of heads to prune (0-1)
    "strategy": "entropy",     # Pruning strategy (entropy, magnitude, random)
    
    # Fine-tuning
    "learning_rate": 5e-5,     # Learning rate
    "batch_size": 4,           # Batch size
    "epochs_per_cycle": 1,     # For quick demonstration
    
    # Dataset
    "dataset": "wikitext",     # Dataset name
    
    # Device
    "device": str(device),
    
    # Other
    "use_differential_lr": True,
    "output_dir": "./upgrayedd_output"
}

# Create directory for output
os.makedirs(config["output_dir"], exist_ok=True)

# Create the configuration object
optimizer_config = AdaptiveOptimizerConfig(**config)

# Show configuration
print("\nExperiment Configuration:")
for key, value in config.items():
    print(f"- {key}: {value}")

# ==========================================================================
# Run Optimization
# ==========================================================================

def run_experiment():
    """Run the optimization experiment"""
    # Create the optimizer
    optimizer = AdaptiveOptimizer(optimizer_config)
    
    # Run optimization (single cycle for demonstration)
    print("\nRunning optimization...")
    results = optimizer.run_continuous_optimization(max_cycles=1)
    
    # Show results
    if results.get("improvement") is not None:
        print(f"\nImprovement: {results['improvement']:.2f}%")
    
    # Return the optimizer for further inspection
    return optimizer

# This is the main execution point when running as a script
if __name__ == "__main__":
    optimizer = run_experiment()
else:
    # When running in a notebook, we also define a helper function
    def run_in_notebook():
        # Create the optimizer
        optimizer = AdaptiveOptimizer(optimizer_config)
        
        # Run optimization (single cycle for demonstration)
        print("\nRunning optimization...")
        results = optimizer.run_continuous_optimization(max_cycles=1)
        
        # Show results
        if results.get("improvement") is not None:
            print(f"\nImprovement: {results['improvement']:.2f}%")
        
        return optimizer, results