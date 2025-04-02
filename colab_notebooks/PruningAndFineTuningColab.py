#!/usr/bin/env python
# coding: utf-8

# # Pruning and Fine-Tuning Benchmark for Google Colab (v0.0.3)
# 
# This is the Python script version of our notebook for Google Colab.
# Version 0.0.3 (April 2025) - Updated with ImprovedFineTuner for enhanced stability
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
# Install required packages and make sure HuggingFace datasets is properly installed
!pip install -q jax jaxlib flax transformers matplotlib numpy pandas seaborn tqdm optax
!pip install -q 'datasets>=2.0.0' multiprocess

# %%
# Clone the repository but make sure it's not in the Python path yet
!git clone https://github.com/CambrianTech/sentinel-ai.git
# Don't cd into it yet

# %%
# Import huggingface datasets directly before changing directory
# We want to make sure we're using the system package
from datasets import load_dataset
import datasets
print(f"Using datasets from: {datasets.__file__}")

# Now safely change to the repository directory
%cd sentinel-ai

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
sys.path.append(".")
from utils.pruning import (
    Environment,
    ResultsManager,
    PruningModule, 
    get_strategy,
    FineTuner,
    ImprovedFineTuner
)

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
class PruningFineTuningExperiment:
    """Manages the pruning + fine-tuning experiment"""
    
    def __init__(self, results_dir="pruning_finetuning_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.results = []
        self.current_experiment = {}
        
        # Initialize environment
        self.env = Environment()
        
        # Get suitable models for this environment
        self.available_models = self.env.get_suitable_models()
        print(f"Models available: {', '.join(self.available_models)}")
        
        # Setup Results Manager
        self.results_manager = ResultsManager(str(self.results_dir))
        self.results_df = pd.DataFrame()
    
    def run_experiment(self, strategies, pruning_levels, prompt, fine_tuning_epochs=1, max_runtime=3600):
        """Run the full experiment"""
        if not self.available_models:
            print("No suitable models found for this environment")
            return
        
        # Start time for runtime tracking
        start_time = time.time()
        
        # Generate all experiment combinations
        experiments = []
        for model in self.available_models:
            for strategy in strategies:
                for level in pruning_levels:
                    experiments.append({
                        "model": model,
                        "strategy": strategy,
                        "pruning_level": level,
                        "prompt": prompt,
                        "fine_tuning_epochs": fine_tuning_epochs
                    })
        
        # Shuffle to get more diverse results early
        random.shuffle(experiments)
        
        # Create progress bar
        pbar = tqdm(total=len(experiments), desc="Running experiments")
        
        # Run experiments
        for i, exp in enumerate(experiments):
            # Check if we've exceeded the runtime limit
            current_runtime = time.time() - start_time
            if max_runtime is not None and current_runtime > max_runtime:
                print(f"\nReached maximum runtime of {max_runtime/3600:.1f} hours")
                break
                
            # Update progress bar
            pbar.set_description(f"Testing {exp['model']}, {exp['strategy']}, {exp['pruning_level']:.2f}")
            
            # Run experiment
            try:
                result = self.run_single_experiment(**exp)
                if result is not None:
                    self.results.append(result)
                
                # Update progress bar
                pbar.update(1)
                
                # Plot intermediate results every few experiments
                if (i + 1) % 1 == 0 or i == len(experiments) - 1:
                    self.plot_results()
            except Exception as e:
                print(f"Error in experiment {exp['model']}, {exp['strategy']}, {exp['pruning_level']:.2f}: {e}")
                import traceback
                traceback.print_exc()
                # Still update progress bar
                pbar.update(1)
        
        # Close progress bar
        pbar.close()
        
        # Final results
        print(f"\nCompleted {len(self.results)} experiments out of {len(experiments)} attempted")
        runtime = time.time() - start_time
        print(f"Total runtime: {runtime/3600:.2f} hours ({runtime/60:.2f} minutes)")
        
        # Plot final results
        self.plot_results()
        
        return self.results
    
    def run_single_experiment(self, model, strategy, pruning_level, prompt, fine_tuning_epochs=1):
        """Run a single experiment with pruning and fine-tuning"""
        print(f"\n{'='*80}")
        print(f"Experiment: {model}, {strategy} strategy, {pruning_level:.2f} pruning level")
        print(f"{'='*80}")
        
        # Initialize pruning module
        pruning_module = PruningModule(model)
        if not pruning_module.load_model():
            print(f"Failed to load model {model}")
            return None
        
        # Setup experiment record
        self.current_experiment = {
            "model": model,
            "strategy": strategy,
            "pruning_level": pruning_level,
            "prompt": prompt,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stages": {}
        }
        
        # 1. Evaluate baseline model
        print("\n>> Stage 1: Evaluating baseline model")
        original_params = pruning_module.original_params
        
        # Evaluate perplexity and generation
        perplexity_baseline = pruning_module.evaluate_perplexity(original_params, prompt)
        print(f"Baseline perplexity: {perplexity_baseline:.4f}")
        
        generated_baseline = pruning_module.generate_text(original_params, prompt)
        print(f"Baseline generated: {generated_baseline}")
        
        # Record baseline results
        self.current_experiment["stages"]["baseline"] = {
            "perplexity": float(perplexity_baseline),
            "generated_text": generated_baseline
        }
        
        # 2. Apply pruning
        print("\n>> Stage 2: Applying pruning")
        pruning_strat = get_strategy(strategy, pruning_module, prompt)
        
        # Calculate importance scores
        print("Calculating head importance...")
        all_head_importance = pruning_strat.get_head_importance(original_params)
        
        # Sort by importance (ascending)
        all_head_importance.sort(key=lambda x: x[2])
        
        # Determine number of heads to prune
        total_heads = pruning_module.num_layers * pruning_module.num_heads
        heads_to_prune = int(total_heads * pruning_level)
        print(f"Pruning {heads_to_prune} out of {total_heads} heads")
        
        # Get head indices to prune (least important first)
        head_indices = [(l, h) for l, h, _ in all_head_importance[:heads_to_prune]]
        
        # Prune heads
        print("Pruning heads...")
        pruned_params = pruning_strat.prune_heads(original_params, head_indices)
        
        # Evaluate after pruning
        perplexity_pruned = pruning_module.evaluate_perplexity(pruned_params, prompt)
        print(f"Pruned perplexity: {perplexity_pruned:.4f}")
        
        generated_pruned = pruning_module.generate_text(pruned_params, prompt)
        print(f"Pruned generated: {generated_pruned}")
        
        # Record pruning results
        self.current_experiment["stages"]["pruned"] = {
            "perplexity": float(perplexity_pruned),
            "perplexity_change": float(perplexity_pruned - perplexity_baseline),
            "generated_text": generated_pruned,
            "pruned_heads": heads_to_prune,
            "total_heads": total_heads,
            "head_indices": head_indices
        }
        
        # 3. Fine-tune the pruned model
        print("\n>> Stage 3: Fine-tuning the pruned model")
        
        # Create fine-tuner - use specific wikitext config and OpenWebText as fallback
        dataset_name = "wikitext-2-v1"  # Specify the config name
        dataset_config = "wikitext"
        
        if self.env.in_colab and self.env.has_tpu:
            # TPUs can handle larger batch sizes
            batch_size = 16
        elif self.env.in_colab and self.env.has_gpu:
            batch_size = 8
        else:
            batch_size = 4
        
        # Check if model name indicates this might be a large model (OPT-1.3B, etc.)
        model_name = model.lower()
        use_improved_tuner = any(x in model_name for x in ['opt', 'large', '1.3b', 'bloom'])
        
        if use_improved_tuner:
            print(f"Using ImprovedFineTuner for model {model} to enhance stability")
            # Initialize improved fine-tuner with better stability for large models
            fine_tuner = ImprovedFineTuner(
                pruning_module, 
                dataset_name=dataset_config, 
                dataset_config=dataset_name,
                batch_size=batch_size
            )
            # Use lower learning rate for large models
            learning_rate = 1e-5
        else:
            # Use standard fine-tuner for smaller models
            fine_tuner = FineTuner(
                pruning_module, 
                dataset_name=dataset_config, 
                dataset_config=dataset_name,
                batch_size=batch_size
            )
            learning_rate = 5e-5
        
        # Fine-tune model
        try:
            tuned_params, metrics = fine_tuner.fine_tune(
                pruned_params, 
                num_epochs=fine_tuning_epochs,
                learning_rate=learning_rate,
                evaluate_interval=5
            )
        except Exception as e:
            print(f"Error during fine-tuning: {e}")
            # If standard tuner fails, fall back to improved tuner
            if not use_improved_tuner:
                print("Falling back to ImprovedFineTuner after error")
                fine_tuner = ImprovedFineTuner(
                    pruning_module, 
                    dataset_name=dataset_config, 
                    dataset_config=dataset_name,
                    batch_size=max(1, batch_size // 2)  # Reduce batch size
                )
                tuned_params, metrics = fine_tuner.fine_tune(
                    pruned_params,
                    num_epochs=fine_tuning_epochs,
                    learning_rate=1e-5,  # Lower learning rate for stability
                    evaluate_interval=5
                )
        
        # Plot training progress
        fine_tuner.plot_training_progress()
        
        # Evaluate fine-tuned model
        perplexity_tuned = pruning_module.evaluate_perplexity(tuned_params, prompt)
        print(f"Fine-tuned perplexity: {perplexity_tuned:.4f}")
        
        generated_tuned = pruning_module.generate_text(tuned_params, prompt)
        print(f"Fine-tuned generated: {generated_tuned}")
        
        # Record fine-tuning results
        self.current_experiment["stages"]["fine_tuned"] = {
            "perplexity": float(perplexity_tuned),
            "perplexity_change_from_baseline": float(perplexity_tuned - perplexity_baseline),
            "perplexity_change_from_pruned": float(perplexity_tuned - perplexity_pruned),
            "generated_text": generated_tuned,
            "training_epochs": fine_tuning_epochs,
            "training_metrics": metrics
        }
        
        # Compute recovery percentage
        if perplexity_pruned > perplexity_baseline:
            # Calculate how much of the perplexity increase was recovered
            perplexity_increase = perplexity_pruned - perplexity_baseline
            perplexity_recovery = perplexity_pruned - perplexity_tuned
            recovery_percentage = (perplexity_recovery / perplexity_increase) * 100 if perplexity_increase > 0 else 0
            
            self.current_experiment["stages"]["fine_tuned"]["recovery_percentage"] = float(recovery_percentage)
            print(f"Recovery percentage: {recovery_percentage:.2f}%")
        else:
            # Pruning improved perplexity, so we measure improvement from baseline
            improvement_percentage = ((perplexity_baseline - perplexity_tuned) / perplexity_baseline) * 100
            
            self.current_experiment["stages"]["fine_tuned"]["improvement_percentage"] = float(improvement_percentage)
            print(f"Improvement percentage: {improvement_percentage:.2f}%")
        
        # 4. Save results
        print("\n>> Stage 4: Saving results")
        
        # Save to disk
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        result_filename = f"{model.replace('/', '_')}_{strategy}_{pruning_level:.2f}_{timestamp}.json"
        result_path = self.results_dir / result_filename
        
        import json
        with open(result_path, "w") as f:
            json.dump(self.current_experiment, f, indent=2)
            
        print(f"Results saved to {result_path}")
        
        # Update DataFrame for plotting
        self._update_dataframe()
        
        return self.current_experiment
    
    def _update_dataframe(self):
        """Update DataFrame for visualization"""
        # Extract data for DataFrame
        data = []
        
        for result in self.results:
            # Extract model and strategy info
            model = result["model"]
            strategy = result["strategy"]
            pruning_level = result["pruning_level"]
            
            # Add baseline stage
            if "baseline" in result["stages"]:
                baseline = result["stages"]["baseline"]
                data.append({
                    "model": model,
                    "strategy": strategy,
                    "pruning_level": pruning_level,
                    "stage": "baseline",
                    "perplexity": baseline["perplexity"]
                })
            
            # Add pruned stage
            if "pruned" in result["stages"]:
                pruned = result["stages"]["pruned"]
                data.append({
                    "model": model,
                    "strategy": strategy,
                    "pruning_level": pruning_level,
                    "stage": "pruned",
                    "perplexity": pruned["perplexity"],
                    "perplexity_change": pruned.get("perplexity_change", 0)
                })
                
            # Add fine-tuned stage
            if "fine_tuned" in result["stages"]:
                fine_tuned = result["stages"]["fine_tuned"]
                data.append({
                    "model": model,
                    "strategy": strategy,
                    "pruning_level": pruning_level,
                    "stage": "fine_tuned",
                    "perplexity": fine_tuned["perplexity"],
                    "perplexity_change_from_baseline": fine_tuned.get("perplexity_change_from_baseline", 0),
                    "perplexity_change_from_pruned": fine_tuned.get("perplexity_change_from_pruned", 0),
                    "recovery_percentage": fine_tuned.get("recovery_percentage", None),
                    "improvement_percentage": fine_tuned.get("improvement_percentage", None)
                })
        
        self.results_df = pd.DataFrame(data)
    
    def plot_results(self, figsize=(15, 12)):
        """Plot comprehensive experiment results"""
        if not self.results:
            print("No results available yet")
            return
            
        # Update DataFrame
        self._update_dataframe()
            
        if self.results_df.empty:
            print("No data available for plotting")
            return
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # 1. Perplexity across stages by model and strategy
        plt.subplot(2, 2, 1)
        
        # Get unique models and strategies
        models = self.results_df["model"].unique()
        strategies = self.results_df["strategy"].unique()
        
        # Filter to main stages
        stages_df = self.results_df[self.results_df["stage"].isin(["baseline", "pruned", "fine_tuned"])]
        
        # Plot lines connecting stages for each experiment
        for model in models:
            model_df = stages_df[stages_df["model"] == model]
            
            for strategy in strategies:
                strategy_df = model_df[model_df["strategy"] == strategy]
                
                for pruning_level in strategy_df["pruning_level"].unique():
                    experiment_df = strategy_df[strategy_df["pruning_level"] == pruning_level]
                    
                    # Sort by stage to ensure correct order
                    stage_order = {"baseline": 0, "pruned": 1, "fine_tuned": 2}
                    experiment_df = experiment_df.sort_values(by="stage", key=lambda x: x.map(stage_order))
                    
                    # Plot if we have all stages
                    if len(experiment_df) >= 2:
                        label = f"{model}, {strategy}, {pruning_level:.2f}"
                        plt.plot(experiment_df["stage"], experiment_df["perplexity"], "o-", label=label)
        
        plt.title("Perplexity Across Stages")
        plt.xlabel("Stage")
        plt.ylabel("Perplexity")
        plt.xticks(rotation=45)
        plt.legend(fontsize=8)
        plt.grid(True)
        
        # 2. Recovery percentage vs pruning level
        plt.subplot(2, 2, 2)
        
        # Get data with recovery information
        recovery_df = self.results_df[self.results_df["stage"] == "fine_tuned"].copy()
        
        if not recovery_df.empty:
            # Create recovery column (combining both metrics)
            recovery_df["recovery"] = recovery_df["recovery_percentage"]
            # If improvement percentage exists and recovery is NaN, use negative of improvement
            mask = recovery_df["recovery"].isna() & recovery_df["improvement_percentage"].notna()
            recovery_df.loc[mask, "recovery"] = -recovery_df.loc[mask, "improvement_percentage"]
            
            # Plot by strategy
            for strategy in strategies:
                strategy_df = recovery_df[recovery_df["strategy"] == strategy]
                if not strategy_df.empty:
                    for model in models:
                        model_strategy_df = strategy_df[strategy_df["model"] == model]
                        if not model_strategy_df.empty:
                            # Sort by pruning level
                            model_strategy_df = model_strategy_df.sort_values("pruning_level")
                            plt.plot(model_strategy_df["pruning_level"], model_strategy_df["recovery"], 
                                    "o-", label=f"{model}, {strategy}")
            
            plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
            plt.axhline(y=100, color="g", linestyle="--", alpha=0.3)
            plt.text(0.01, 100, "Full Recovery", color="green", ha="left", va="bottom")
            plt.text(0.01, -5, "Improvement", color="blue", ha="left", va="top")
            
            plt.title("Recovery Percentage by Pruning Level")
            plt.xlabel("Pruning Level")
            plt.ylabel("Recovery % (negative means improvement)")
            plt.legend(fontsize=8)
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, "No recovery data available yet", 
                    ha="center", va="center", fontsize=12)
        
        # 3. Perplexity change: pruning vs fine-tuning effect
        plt.subplot(2, 2, 3)
        
        if "perplexity_change" in self.results_df.columns and "perplexity_change_from_pruned" in self.results_df.columns:
            # Get pruning change
            pruned_df = self.results_df[self.results_df["stage"] == "pruned"].copy()
            pruned_df = pruned_df[["model", "strategy", "pruning_level", "perplexity_change"]]
            
            # Get fine-tuning change
            finetuned_df = self.results_df[self.results_df["stage"] == "fine_tuned"].copy()
            finetuned_df = finetuned_df[["model", "strategy", "pruning_level", "perplexity_change_from_pruned"]]
            
            # Merge
            effects_df = pd.merge(
                pruned_df, finetuned_df,
                on=["model", "strategy", "pruning_level"],
                suffixes=("_pruning", "_finetuning")
            )
            
            if not effects_df.empty:
                # Plot scatter with size based on pruning level
                for strategy in strategies:
                    strategy_df = effects_df[effects_df["strategy"] == strategy]
                    if not strategy_df.empty:
                        for model in models:
                            model_df = strategy_df[strategy_df["model"] == model]
                            if not model_df.empty:
                                plt.scatter(
                                    model_df["perplexity_change"], 
                                    model_df["perplexity_change_from_pruned"],
                                    s=model_df["pruning_level"] * 500,  # Size based on pruning level
                                    label=f"{model}, {strategy}",
                                    alpha=0.7
                                )
                
                plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
                plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)
                
                # Add quadrant labels
                plt.text(-5, -5, "Both improved", fontsize=10, ha="center", va="center",
                        bbox=dict(facecolor="lightgreen", alpha=0.5))
                plt.text(5, -5, "Pruning hurt,\nFine-tuning fixed", fontsize=10, ha="center", va="center",
                        bbox=dict(facecolor="lightblue", alpha=0.5))
                plt.text(-5, 5, "Pruning helped,\nFine-tuning hurt", fontsize=10, ha="center", va="center",
                        bbox=dict(facecolor="lightyellow", alpha=0.5))
                plt.text(5, 5, "Both hurt", fontsize=10, ha="center", va="center",
                        bbox=dict(facecolor="lightcoral", alpha=0.5))
                
                plt.title("Effect of Pruning vs. Fine-tuning")
                plt.xlabel("Perplexity Change from Pruning")
                plt.ylabel("Perplexity Change from Fine-tuning")
                plt.legend(fontsize=8)
                plt.grid(True)
            else:
                plt.text(0.5, 0.5, "No effect data available yet", 
                        ha="center", va="center", fontsize=12)
        else:
            plt.text(0.5, 0.5, "No effect data available yet", 
                    ha="center", va="center", fontsize=12)
        
        # 4. Final results: perplexity reduction by pruning level and strategy
        plt.subplot(2, 2, 4)
        
        if "perplexity_change_from_baseline" in self.results_df.columns:
            # Get baseline and final results
            baseline_df = self.results_df[self.results_df["stage"] == "baseline"].copy()
            baseline_df = baseline_df[["model", "strategy", "pruning_level", "perplexity"]]
            baseline_df = baseline_df.rename(columns={"perplexity": "baseline_perplexity"})
            
            final_df = self.results_df[self.results_df["stage"] == "fine_tuned"].copy()
            final_df = final_df[["model", "strategy", "pruning_level", "perplexity", "perplexity_change_from_baseline"]]
            final_df = final_df.rename(columns={"perplexity": "final_perplexity"})
            
            # Merge
            final_results = pd.merge(
                baseline_df, final_df,
                on=["model", "strategy", "pruning_level"]
            )
            
            if not final_results.empty:
                # Plot as bar chart
                # Group by pruning level and strategy
                grouped = final_results.groupby(["pruning_level", "strategy"])["perplexity_change_from_baseline"].mean().reset_index()
                
                # Pivot for grouped bar chart
                pivot_df = grouped.pivot(index="pruning_level", columns="strategy", values="perplexity_change_from_baseline")
                
                # Plot
                pivot_df.plot(kind="bar", ax=plt.gca())
                
                plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
                plt.title("Final Perplexity Change from Baseline")
                plt.xlabel("Pruning Level")
                plt.ylabel("Perplexity Change")
                plt.legend(title="Strategy")
                plt.grid(True, axis="y")
            else:
                plt.text(0.5, 0.5, "No final results available yet", 
                        ha="center", va="center", fontsize=12)
        else:
            plt.text(0.5, 0.5, "No final results available yet", 
                    ha="center", va="center", fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        return fig

# %% [markdown]
# ## Run the Experiment
# 
# Now we can run the full experiment:

# %%
# Initialize experiment
experiment = PruningFineTuningExperiment("pruning_finetuning_results")

# %%
# Configuration
STRATEGIES = ["random", "magnitude", "entropy"]
PRUNING_LEVELS = [0.1, 0.3, 0.5]
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
OVERNIGHT_PRUNING_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
OVERNIGHT_PROMPT = "Artificial intelligence will revolutionize industries by"
OVERNIGHT_FINE_TUNING_EPOCHS = 5  # More epochs for better recovery
OVERNIGHT_MAX_RUNTIME = 24 * 3600  # 24 hours

# Initialize experiment for overnight run
overnight_experiment = PruningFineTuningExperiment("overnight_results")

# Run overnight experiment (uncomment to run)
# overnight_results = overnight_experiment.run_experiment(
#     strategies=OVERNIGHT_STRATEGIES,
#     pruning_levels=OVERNIGHT_PRUNING_LEVELS,
#     prompt=OVERNIGHT_PROMPT,
#     fine_tuning_epochs=OVERNIGHT_FINE_TUNING_EPOCHS,
#     max_runtime=OVERNIGHT_MAX_RUNTIME
# )

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