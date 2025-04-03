#!/usr/bin/env python
# coding: utf-8

# # Pruning and Fine-Tuning Benchmark for Google Colab (v0.0.27.0)
# 
# This is the Python script version of our notebook for Google Colab.
# Version 0.0.27.0 (April 2025) - Use PruningFineTuningExperiment from utils.pruning instead of custom class
# Version 0.0.26.3 (April 2025) - Added clarification for custom class
# Version 0.0.26.2 (April 2025) - Fixed sequence_length and stability_level assignments
# Version 0.0.26.1 (April 2025) - Fixed stability_level parameter inconsistency
# Version 0.0.26 (April 2025) - Added support for batch_size, sequence_length and stability_level parameters
# Version 0.0.25 (April 2025) - Verified fixed imports with HuggingFace datasets and data_modules
# Version 0.0.24 (April 2025) - Renamed internal module to fix HuggingFace datasets import
# 
# Instructions:
# 1. Upload to a new Colab notebook using File > Upload notebook > Upload
# 2. Runtime > Change runtime type > Select GPU or TPU hardware accelerator
# 3. Run cells to execute pruning and fine-tuning experiments
# 
# ## Overview
# 
# 1. **Baseline Evaluation**: Establish the initial model performance
# 2. **Pruning Phase**: Apply different pruning strategies and evaluate post-pruning performance
# 3. **Fine-Tuning Phase**: Fine-tune pruned models to recover or improve performance
# 4. **Analysis**: Compare performance across pruning levels and fine-tuning epochs
# 
# This experiment will run until interrupted, continuously improving the models and updating visualizations.
# 
# ## Setup
# 
# First, let's install dependencies and clone the repository:

# %%
# Install required packages (critical to do this before anything else)
print("🔧 Installing dependencies...")
!pip install -q jax jaxlib flax transformers matplotlib numpy pandas seaborn tqdm optax
# Install datasets explicitly with required version to ensure compatibility
!pip install -q 'datasets>=2.0.0' multiprocess

# %%
# Clone the repository 
# Note: We use the refactor/modular-experiment branch which contains our optimizations
print("📦 Cloning the repository...")
!git clone -b refactor/modular-experiment https://github.com/CambrianTech/sentinel-ai.git

# Create symlink for Colab compatibility
!ln -sf sentinel-ai refactor

# Change to the repository directory
print("📂 Changing to repository directory...")
%cd /content/sentinel-ai

# %%
# Import from HuggingFace datasets and our data_modules without conflicts
# Our internal module was renamed from 'datasets' to 'data_modules' to avoid namespace collisions
import datasets
from datasets import load_dataset
print(f"Using HuggingFace datasets from: {datasets.__file__}")

# Import rest of the libraries
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
# Prioritize local imports for our own modules, but keep system modules first
# This ensures we use our utils package but system's datasets
system_paths = [p for p in sys.path if '/usr/local' in p or 'python3' in p or 'site-packages' in p]
local_paths = ["."]  # Current directory first
other_paths = [p for p in sys.path if p not in system_paths and p != "."]
sys.path = system_paths + local_paths + other_paths
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

# Set up plotting
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## Environment Detection
# 
# Let's detect our environment capabilities:

# %%
# Initialize environment and detect capabilities
env = Environment()
env.print_info()

# Check JAX capabilities
print(f"\nJAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# %% [markdown]
# ## Experiment Manager
# 
# Let's create an experiment manager to run the full experiment:

# %%
# Define a custom PruningFineTuningExperiment class for the Colab notebook
# Note: This is different from the one in utils.pruning.experiment to support additional parameters
# # Note: We are now importing PruningFineTuningExperiment from utils.pruning instead of defining it here.
# The version in utils.pruning now supports all the same parameters (batch_size, sequence_length, stability_level).
# The code below is kept for reference but is commented out.

# Note: We use PruningFineTuningExperiment from utils.pruning (imported above)
# This ensures consistent behavior between Colab and local testing

# %% [markdown]

# ## Run the Experiment
# 
# Now we can run the full experiment:

# %%
# Initialize experiment with memory optimizations
experiment = PruningFineTuningExperiment(
    results_dir="pruning_finetuning_results",
    use_improved_fine_tuner=True,      # Use the improved fine-tuner with stability enhancements
    detect_environment=True,           # Automatically detect Colab environment
    optimize_memory=True,              # Optimize memory usage based on detected hardware
    batch_size=2,                      # Override batch size for fine-tuning
    sequence_length=64,                # Override sequence length for fine-tuning
    stability_level=2                  # Use enhanced stability measures (1-3, where 3 is max stability)
)

# %%
# Configuration
STRATEGIES = ["random", "magnitude", "entropy"]
# Updated based on optimization findings - 0.7 for max speed, 0.3 for balanced performance
PRUNING_LEVELS = [0.1, 0.3, 0.7]
PROMPT = "Artificial intelligence will transform society by"
FINE_TUNING_EPOCHS = 2  # Small number for quick iterations
MAX_RUNTIME = 6 * 3600  # 6 hours

# Start the experiment
results = experiment.run_experiment(
    strategies=STRATEGIES,
    pruning_levels=PRUNING_LEVELS,
    prompt=PROMPT,
    fine_tuning_epochs=FINE_TUNING_EPOCHS,
    max_runtime=MAX_RUNTIME
)

# %% [markdown]
# ## Longer Overnight Run
# 
# For an extended overnight run, uncomment and run this cell:

# %%
# Overnight Configuration
OVERNIGHT_STRATEGIES = ["random", "magnitude", "entropy"]
# Focused on most effective pruning levels based on optimization research
# 0.7 for maximum speed, 0.3 for balanced performance, other values for comparison
OVERNIGHT_PRUNING_LEVELS = [0.1, 0.2, 0.3, 0.5, 0.7]
OVERNIGHT_PROMPT = "Artificial intelligence will revolutionize industries by"
OVERNIGHT_FINE_TUNING_EPOCHS = 5  # More epochs for better recovery
OVERNIGHT_MAX_RUNTIME = 24 * 3600  # 24 hours

# Initialize experiment for overnight run
overnight_experiment = PruningFineTuningExperiment(
    results_dir="overnight_results",
    use_improved_fine_tuner=True,      # Use stability enhancements for overnight runs
    detect_environment=True,
    optimize_memory=True,
    batch_size=1,                      # Smaller batch for longer sequences
    sequence_length=128,               # Longer sequences for better quality
    stability_level=3                  # Maximum stability for overnight runs
)

# Run overnight experiment (uncomment to run)
# overnight_results = overnight_experiment.run_experiment(
#     strategies=OVERNIGHT_STRATEGIES,
#     pruning_levels=OVERNIGHT_PRUNING_LEVELS,
#     prompt=OVERNIGHT_PROMPT,
#     fine_tuning_epochs=OVERNIGHT_FINE_TUNING_EPOCHS,
#     max_runtime=OVERNIGHT_MAX_RUNTIME
# )

# %% [markdown]
# ## Optimization Recommendations
# 
# Based on our extensive profiling and research, we have the following recommendations:

# %%
print("\n===== Sentinel AI Optimization Recommendations =====")
print("1. For maximum throughput (pure speed):")
print("   - Use original model with 70% pruning (~28 tokens/sec on CPU)")
print("\n2. For models with agency features:")
print("   - On CPU: Use optimization level 2 with 30% pruning (~19-20 tokens/sec)")
print("   - On GPU: Use optimization level 3 with 30% pruning")
print("\n3. For balanced quality/performance:")
print("   - Use optimization level 2 with 30% pruning")
print("\nNote: Original model with heavy pruning (70%) often outperforms optimized models for pure throughput")

# %% [markdown]
# ## Comprehensive Analysis
# 
# After collecting results, run a comprehensive analysis:

# %%
# Plot results
experiment.plot_results(figsize=(16, 12))

# %%
# Create a summary table
if not experiment.results_df.empty:
    # Get data for different stages
    baseline_df = experiment.results_df[experiment.results_df["stage"] == "baseline"][["model", "strategy", "pruning_level", "perplexity"]]
    baseline_df = baseline_df.rename(columns={"perplexity": "baseline_perplexity"})
    
    pruned_df = experiment.results_df[experiment.results_df["stage"] == "pruned"][["model", "strategy", "pruning_level", "perplexity"]]
    pruned_df = pruned_df.rename(columns={"perplexity": "pruned_perplexity"})
    
    finetuned_df = experiment.results_df[experiment.results_df["stage"] == "fine_tuned"][["model", "strategy", "pruning_level", "perplexity"]]
    finetuned_df = finetuned_df.rename(columns={"perplexity": "finetuned_perplexity"})
    
    # Merge dataframes
    summary = pd.merge(baseline_df, pruned_df, on=["model", "strategy", "pruning_level"])
    summary = pd.merge(summary, finetuned_df, on=["model", "strategy", "pruning_level"])
    
    # Calculate changes
    summary["pruning_effect"] = summary["pruned_perplexity"] - summary["baseline_perplexity"]
    summary["finetuning_effect"] = summary["finetuned_perplexity"] - summary["pruned_perplexity"]
    summary["net_change"] = summary["finetuned_perplexity"] - summary["baseline_perplexity"]
    
    # Display summary
    summary.head()