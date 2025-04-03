"""
Experiment framework for pruning and fine-tuning.

This module provides a modular, extensible framework for running pruning and 
fine-tuning experiments with language models, with support for different model
architectures, pruning strategies, and evaluation methods.
"""

import os
import json
import time
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

# Import visualization utilities when available
try:
    from .visualization import plot_experiment_summary
except ImportError:
    # Fallback if visualization module is not available
    plot_experiment_summary = None

from tqdm.auto import tqdm

# Set up logging
logger = logging.getLogger(__name__)

# Local imports
from .pruning_module import PruningModule
from .strategies import get_strategy
from .fine_tuner import FineTuner
from .fine_tuner_improved import ImprovedFineTuner
from .environment import Environment
from .results_manager import ResultsManager
from .stability import patch_fine_tuner, optimize_fine_tuner


class PruningExperiment:
    """
    Base class for pruning experiments.
    
    This class provides the core functionality for running pruning experiments,
    including model loading, pruning, and evaluation.
    """
    
    def __init__(self, 
                 results_dir: str = "pruning_results",
                 use_improved_fine_tuner: bool = True,
                 detect_environment: bool = True,
                 optimize_memory: bool = True):
        """
        Initialize the pruning experiment.
        
        Args:
            results_dir: Directory to save experiment results
            use_improved_fine_tuner: Whether to use the improved fine-tuner with stability enhancements
            detect_environment: Whether to detect the environment capabilities
            optimize_memory: Whether to optimize memory usage based on model size
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.results = []
        self.current_experiment = {}
        
        self.use_improved_fine_tuner = use_improved_fine_tuner
        self.optimize_memory = optimize_memory
        
        # Environment detection
        if detect_environment:
            self.env = Environment()
            self.detect_hardware_capabilities()
        else:
            self.env = None
            self.gpu_memory_gb = 0
            
        # Setup Results Manager
        self.results_manager = ResultsManager(str(self.results_dir))
        self.results_df = pd.DataFrame()
        
    def detect_hardware_capabilities(self):
        """Detect hardware capabilities like RAM and GPU memory."""
        # Add has_high_ram attribute if not present in Environment class
        if not hasattr(self.env, 'has_high_ram'):
            self.env.has_high_ram = False
            try:
                import psutil
                total_ram = psutil.virtual_memory().total / (1024**3)  # Convert to GB
                self.env.has_high_ram = total_ram > 12
                logger.info(f"Detected {total_ram:.1f}GB RAM, high RAM: {self.env.has_high_ram}")
            except:
                logger.warning("Could not detect RAM, assuming standard memory")
                
        # Detect GPU memory if possible
        self.gpu_memory_gb = 0
        if hasattr(self.env, 'has_gpu') and self.env.has_gpu:
            try:
                import torch
                gpu_props = torch.cuda.get_device_properties(0)
                self.gpu_memory_gb = gpu_props.total_memory / (1024**3)
                logger.info(f"Detected GPU with {self.gpu_memory_gb:.1f}GB VRAM")
            except:
                try:
                    # Alternative method for JAX
                    import jax
                    device = jax.devices()[0]
                    if hasattr(device, 'memory_stats') and callable(device.memory_stats):
                        memory_stats = device.memory_stats()
                        if 'bytes_limit' in memory_stats:
                            self.gpu_memory_gb = memory_stats['bytes_limit'] / (1024**3)
                            logger.info(f"Detected GPU with approximately {self.gpu_memory_gb:.1f}GB VRAM")
                except:
                    # Estimate based on environment
                    if hasattr(self.env, 'in_colab') and self.env.in_colab:
                        # T4 in Colab typically has 16GB
                        self.gpu_memory_gb = 16
                        logger.info(f"Estimating Colab GPU with {self.gpu_memory_gb}GB VRAM")
                        
        # Get suitable models for this environment if available
        if hasattr(self.env, 'get_suitable_models'):
            self.available_models = self.env.get_suitable_models()
            logger.info(f"Models available: {', '.join(self.available_models)}")
        else:
            self.available_models = []
            
    def create_fine_tuner(self, 
                         pruning_module: PruningModule, 
                         model_name: str, 
                         batch_size: int = 4,
                         dataset_name: str = "wikitext",
                         dataset_config: str = "wikitext-2-v1"):
        """
        Create a fine-tuner instance with appropriate settings.
        
        Args:
            pruning_module: The pruning module instance
            model_name: Name of the model (for detection)
            batch_size: Initial batch size (may be adjusted)
            dataset_name: Name of the dataset to use
            dataset_config: Configuration of the dataset
            
        Returns:
            A fine-tuner instance
        """
        # Check if model name indicates a model known to benefit from improved fine-tuner
        model_name_lower = model_name.lower()
        needs_improved = any(x in model_name_lower for x in [
            'opt', 'bloom', 'llama', 'falcon',  # Specific architectures
            '1.3b', '2.7b', '6.7b', '13b',      # Size indicators
            'large', 'xl', 'xxl'                 # Size indicators
        ])
        
        # Use custom parameters if provided through environment
        if batch_size is None and hasattr(self.env, 'batch_size') and self.env.batch_size is not None:
            batch_size = self.env.batch_size
            logger.info(f"Using configured batch size: {batch_size}")
        elif batch_size is None:
            # Use default batch size based on hardware
            batch_size = 4  # Default fallback
        
        # Get sequence length from environment if set
        sequence_length = getattr(self.env, 'seq_length', 64) if hasattr(self.env, 'seq_length') else 64
        
        # Get stability level from environment if set
        stability_level = getattr(self.env, 'stability_level', 2) if hasattr(self.env, 'stability_level') else 2
        
        # Determine which fine-tuner to use
        if self.use_improved_fine_tuner or needs_improved:
            logger.info(f"Using ImprovedFineTuner (stability level: {stability_level})")
            logger.info(f"Training parameters: batch_size={batch_size}, sequence_length={sequence_length}")
            
            fine_tuner = ImprovedFineTuner(
                pruning_module, 
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                batch_size=batch_size,
                sequence_length=sequence_length,
                stability_level=stability_level
            )
        else:
            logger.info(f"Using standard FineTuner")
            fine_tuner = FineTuner(
                pruning_module, 
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                batch_size=batch_size
            )
        
        # Apply stability patches
        logger.info("Installing NaN-safe loss function")
        fine_tuner = patch_fine_tuner(fine_tuner, model_name=model_name)
        
        # Apply memory optimization if enabled
        if self.optimize_memory:
            logger.info(f"Optimizing memory usage for {model_name}")
            fine_tuner = optimize_fine_tuner(fine_tuner, model_name=model_name, gpu_memory_gb=self.gpu_memory_gb)
        
        return fine_tuner
        
    def evaluate_baseline(self, 
                         pruning_module: PruningModule, 
                         prompt: str) -> Dict[str, Any]:
        """
        Evaluate the baseline model before pruning.
        
        Args:
            pruning_module: The pruning module instance
            prompt: Prompt for text generation evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        print("Evaluating baseline model")
        original_params = pruning_module.original_params
        
        # Evaluate perplexity and generation
        perplexity_baseline = pruning_module.evaluate_perplexity(original_params, prompt)
        print(f"Baseline perplexity: {perplexity_baseline:.4f}")
        
        generated_baseline = pruning_module.generate_text(original_params, prompt)
        print(f"Baseline generated: {generated_baseline}")
        
        # Return evaluation results
        return {
            "perplexity": float(perplexity_baseline),
            "generated_text": generated_baseline
        }
        
    def apply_pruning(self, 
                     pruning_module: PruningModule, 
                     strategy: str, 
                     pruning_level: float, 
                     prompt: str, 
                     baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply pruning to the model.
        
        Args:
            pruning_module: The pruning module instance
            strategy: Pruning strategy to use
            pruning_level: Proportion of heads to prune (0.0 to 1.0)
            prompt: Prompt for evaluation
            baseline_results: Results from baseline evaluation
            
        Returns:
            Dictionary with pruning results
        """
        print(f"Applying {strategy} pruning strategy at level {pruning_level:.2f}")
        
        # Get strategy instance
        pruning_strat = get_strategy(strategy, pruning_module, prompt)
        
        # Calculate importance scores
        print("Calculating head importance...")
        all_head_importance = pruning_strat.get_head_importance(pruning_module.original_params)
        
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
        pruned_params = pruning_strat.prune_heads(pruning_module.original_params, head_indices)
        
        # Evaluate after pruning
        perplexity_pruned = pruning_module.evaluate_perplexity(pruned_params, prompt)
        print(f"Pruned perplexity: {perplexity_pruned:.4f}")
        
        generated_pruned = pruning_module.generate_text(pruned_params, prompt)
        print(f"Pruned generated: {generated_pruned}")
        
        # Return pruning results
        return {
            "perplexity": float(perplexity_pruned),
            "perplexity_change": float(perplexity_pruned - baseline_results["perplexity"]),
            "generated_text": generated_pruned,
            "pruned_heads": heads_to_prune,
            "total_heads": total_heads,
            "head_indices": head_indices,
            "pruned_params": pruned_params  # Return params for fine-tuning
        }
        
    def fine_tune_model(self, 
                       pruning_module: PruningModule,
                       model_name: str,
                       pruned_params: Any,
                       prompt: str,
                       baseline_results: Dict[str, Any],
                       pruned_results: Dict[str, Any],
                       fine_tuning_epochs: int = 1,
                       batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Fine-tune the pruned model.
        
        Args:
            pruning_module: The pruning module instance
            model_name: Name of the model
            pruned_params: Parameters of the pruned model
            prompt: Prompt for evaluation
            baseline_results: Results from baseline evaluation
            pruned_results: Results from pruning
            fine_tuning_epochs: Number of epochs for fine-tuning
            batch_size: Batch size for fine-tuning
            
        Returns:
            Dictionary with fine-tuning results
        """
        print(f"Fine-tuning pruned model for {fine_tuning_epochs} epochs")
        
        # Create fine-tuner with dataset config
        fine_tuner = self.create_fine_tuner(
            pruning_module,
            model_name=model_name,
            batch_size=batch_size
        )
        
        # Adjust learning rate based on model size
        model_name_lower = model_name.lower()
        is_large_model = any(x in model_name_lower for x in [
            'opt', '1.3b', 'large', 'bloom', '2.7b', 'xl'
        ])
        learning_rate = 1e-5 if is_large_model else 5e-5
        
        # Fine-tune model
        try:
            # Proceed with fine-tuning
            tuned_params, metrics = fine_tuner.fine_tune(
                pruned_params, 
                num_epochs=fine_tuning_epochs,
                learning_rate=learning_rate,
                evaluate_interval=5
            )
            
            # Plot training progress if available
            if hasattr(fine_tuner, 'plot_training_progress'):
                fine_tuner.plot_training_progress()
            
            # Evaluate fine-tuned model
            perplexity_tuned = pruning_module.evaluate_perplexity(tuned_params, prompt)
            print(f"Fine-tuned perplexity: {perplexity_tuned:.4f}")
            
            generated_tuned = pruning_module.generate_text(tuned_params, prompt)
            print(f"Fine-tuned generated: {generated_tuned}")
            
            # Calculate recovery or improvement metrics
            perplexity_baseline = baseline_results["perplexity"]
            perplexity_pruned = pruned_results["perplexity"]
            
            # Initialize fine-tuning results
            fine_tuned = {
                "perplexity": float(perplexity_tuned),
                "perplexity_change_from_baseline": float(perplexity_tuned - perplexity_baseline),
                "perplexity_change_from_pruned": float(perplexity_tuned - perplexity_pruned),
                "generated_text": generated_tuned,
                "training_epochs": fine_tuning_epochs,
                "training_metrics": metrics
            }
            
            # Compute recovery or improvement percentage
            if perplexity_pruned > perplexity_baseline:
                # Calculate how much of the perplexity increase was recovered
                perplexity_increase = perplexity_pruned - perplexity_baseline
                perplexity_recovery = perplexity_pruned - perplexity_tuned
                recovery_percentage = (perplexity_recovery / perplexity_increase) * 100 if perplexity_increase > 0 else 0
                
                fine_tuned["recovery_percentage"] = float(recovery_percentage)
                print(f"Recovery percentage: {recovery_percentage:.2f}%")
            else:
                # Pruning improved perplexity, so we measure improvement from baseline
                improvement_percentage = ((perplexity_baseline - perplexity_tuned) / perplexity_baseline) * 100
                
                fine_tuned["improvement_percentage"] = float(improvement_percentage)
                print(f"Improvement percentage: {improvement_percentage:.2f}%")
            
            return fine_tuned
            
        except Exception as e:
            print(f"Error during fine-tuning: {e}")
            # Continue with partial results
            import traceback
            traceback.print_exc()
            
            return {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
    def run_single_experiment(self, 
                            model: str, 
                            strategy: str, 
                            pruning_level: float, 
                            prompt: str, 
                            fine_tuning_epochs: int = 1,
                            save_results: bool = True) -> Dict[str, Any]:
        """
        Run a single experiment with pruning and optional fine-tuning.
        
        Args:
            model: Name of the model to use
            strategy: Pruning strategy to use
            pruning_level: Proportion of heads to prune (0.0 to 1.0)
            prompt: Prompt for evaluation
            fine_tuning_epochs: Number of epochs for fine-tuning
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with experiment results
        """
        print(f"\n{'='*80}")
        print(f"Experiment: {model}, {strategy} strategy, {pruning_level:.2f} pruning level")
        print(f"{'='*80}")
        
        # Initialize pruning module
        pruning_module = PruningModule(model)
        if not pruning_module.load_model():
            print(f"Failed to load model {model}")
            return None
        
        # Store model name for architecture detection
        pruning_module.model_name = model
        
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
        baseline_results = self.evaluate_baseline(pruning_module, prompt)
        self.current_experiment["stages"]["baseline"] = baseline_results
        
        # 2. Apply pruning
        print("\n>> Stage 2: Applying pruning")
        pruned_results = self.apply_pruning(
            pruning_module, strategy, pruning_level, prompt, baseline_results
        )
        
        # Remove pruned_params from results to avoid serialization issues
        pruned_params = pruned_results.pop("pruned_params", None)
        self.current_experiment["stages"]["pruned"] = pruned_results
        
        # 3. Fine-tune the pruned model if epochs > 0
        if fine_tuning_epochs > 0:
            print("\n>> Stage 3: Fine-tuning the pruned model")
            fine_tuned_results = self.fine_tune_model(
                pruning_module, model, pruned_params, prompt, 
                baseline_results, pruned_results, fine_tuning_epochs
            )
            self.current_experiment["stages"]["fine_tuned"] = fine_tuned_results
        
        # 4. Save results if requested
        if save_results:
            print("\n>> Stage 4: Saving results")
            self._save_results()
        
        return self.current_experiment
                
    def _save_results(self):
        """Save current experiment results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        model = self.current_experiment.get("model", "unknown")
        strategy = self.current_experiment.get("strategy", "unknown")
        pruning_level = self.current_experiment.get("pruning_level", 0.0)
        
        result_filename = f"{model.replace('/', '_')}_{strategy}_{pruning_level:.2f}_{timestamp}.json"
        result_path = self.results_dir / result_filename
        
        with open(result_path, "w") as f:
            json.dump(self.current_experiment, f, indent=2)
            
        print(f"Results saved to {result_path}")
        
        # Update DataFrame for plotting
        self._update_dataframe()
        
    def _update_dataframe(self):
        """Update DataFrame for visualization."""
        # Extract data for DataFrame
        data = []
        
        for result in self.results + [self.current_experiment]:
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
                
                # Skip if this stage contains an error
                if "error" in fine_tuned:
                    continue
                    
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
        """
        Plot comprehensive experiment results.
        
        Args:
            figsize: Figure size (width, height) in inches
            
        Returns:
            Matplotlib figure object
        """
        if not self.results and not self.current_experiment:
            print("No results available yet")
            return
            
        # Update DataFrame
        self._update_dataframe()
            
        if self.results_df.empty:
            print("No data available for plotting")
            return
        
        # Use the modular visualization if available, otherwise fall back to inline implementation
        if plot_experiment_summary is not None:
            # Use the external implementation
            fig = plot_experiment_summary(self.results_df, figsize=figsize)
            plt.show()
            return fig
        
        # Fall back to built-in implementation if visualization module is not available
        try:
            import seaborn as sns
            sns.set_theme(style="whitegrid")
        except ImportError:
            print("Seaborn not available for visualization, install with 'pip install seaborn'")
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
                    
                    # Plot if we have at least two stages
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
            
            plt.title("Recovery/Improvement by Pruning Level")
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


class PruningFineTuningExperiment(PruningExperiment):
    """
    Extended experiment class with full pruning and fine-tuning pipeline.
    
    This class adds functionality for running multiple experiments with
    different models, strategies, and pruning levels.
    """
    
    def __init__(self, 
                 results_dir: str = "pruning_finetuning_results", 
                 use_improved_fine_tuner: bool = True,
                 detect_environment: bool = True,
                 optimize_memory: bool = True,
                 batch_size: Optional[int] = None,
                 sequence_length: Optional[int] = None,
                 stability_level: Optional[int] = None):
        """
        Initialize the pruning and fine-tuning experiment.
        
        Args:
            results_dir: Directory to save experiment results
            use_improved_fine_tuner: Whether to use the improved fine-tuner with stability enhancements
            detect_environment: Whether to detect the environment capabilities
            optimize_memory: Whether to optimize memory usage based on model size
            batch_size: Optional override for batch size during fine-tuning
            sequence_length: Optional override for sequence length during fine-tuning
            stability_level: Optional override for stability level during fine-tuning (1-3)
        """
        super().__init__(
            results_dir, 
            use_improved_fine_tuner, 
            detect_environment,
            optimize_memory
        )
        
        # Store custom fine-tuning parameters
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.stability_level = stability_level
        
        # Apply custom parameters to environment if provided
        if hasattr(self, 'env') and self.env is not None:
            if self.batch_size is not None:
                self.env.batch_size = self.batch_size
            
            if self.sequence_length is not None:
                self.env.seq_length = self.sequence_length
                
            if self.stability_level is not None:
                self.env.stability_level = self.stability_level
        
        # Model size limits based on environment and GPU memory
        self.update_model_size_limits()
        
    def update_model_size_limits(self):
        """
        Update model size limits based on environment and GPU memory.
        
        This determines which models and pruning levels can be run safely
        without running out of memory.
        """
        # Start with base models that should be safe on most hardware
        self.model_size_limits = {
            "gpt2": 1.0,  # Always allow GPT-2 (124M params)
            "gpt2-medium": 1.0,  # Always allow GPT-2 Medium (355M params)
            "opt-350m": 1.0,  # Always allow OPT-350M
            "opt-125m": 1.0,  # Always allow OPT-125M
            "facebook/opt-125m": 1.0,  # Always allow OPT-125M
            "facebook/opt-350m": 1.0,  # Always allow OPT-350M
            "EleutherAI/pythia-160m": 1.0,  # Always allow Pythia-160M
            "EleutherAI/pythia-410m": 1.0,  # Always allow Pythia-410M
        }
        
        # Add larger models only if we have enough resources
        if hasattr(self.env, 'has_gpu') and self.env.has_gpu and self.gpu_memory_gb >= 8:
            # If we have a GPU with 8+ GB, allow medium-sized models
            self.model_size_limits.update({
                "gpt2-large": 1.0,  # Allow GPT-2 Large (774M params) with sufficient GPU
                "EleutherAI/pythia-1b": 0.5,  # Allow Pythia-1B with pruning
                "facebook/opt-1.3b": 0.3,  # Allow OPT-1.3B with significant pruning only
            })
            
            if self.gpu_memory_gb >= 16:
                # If we have 16+ GB VRAM, allow larger models
                self.model_size_limits.update({
                    "gpt2-xl": 0.3,  # Allow GPT-2 XL with pruning
                    "facebook/opt-1.3b": 0.5,  # Allow OPT-1.3B with moderate pruning
                    "facebook/opt-2.7b": 0.2,  # Allow OPT-2.7B with heavy pruning
                })
                
        # Add special case to fix distilgpt2 issue
        if "distilgpt2" not in self.model_size_limits:
            self.model_size_limits["distilgpt2"] = 1.0
            
        print(f"Model size limits updated based on available resources")
        
    def run_experiment(self, 
                      strategies: List[str], 
                      pruning_levels: List[float], 
                      prompt: str, 
                      fine_tuning_epochs: int = 1, 
                      max_runtime: Optional[int] = 3600,
                      models: Optional[List[str]] = None):
        """
        Run the full experiment with multiple models, strategies, and pruning levels.
        
        Args:
            strategies: List of pruning strategies to use
            pruning_levels: List of pruning levels to use
            prompt: Prompt for evaluation
            fine_tuning_epochs: Number of epochs for fine-tuning
            max_runtime: Maximum runtime in seconds, or None for no limit
            models: List of models to use, or None to use available_models
            
        Returns:
            List of experiment results
        """
        # Use provided models or available models
        if models is None:
            if not self.available_models:
                print("No suitable models found for this environment")
                return []
            models_to_use = self.available_models
        else:
            models_to_use = models
        
        print(f"Running experiments with models: {', '.join(models_to_use)}")
        print(f"Strategies: {', '.join(strategies)}")
        print(f"Pruning levels: {', '.join([str(p) for p in pruning_levels])}")
        
        # Start time for runtime tracking
        start_time = time.time()
        
        # Generate all experiment combinations
        experiments = []
        for model in models_to_use:
            for strategy in strategies:
                for level in pruning_levels:
                    # Skip model/pruning combinations that would exceed memory limits
                    model_key = model.split('/')[-1] if '/' in model else model
                    model_size_limit = self.model_size_limits.get(model, self.model_size_limits.get(model_key, 0.0))
                    
                    if level > model_size_limit:
                        print(f"Skipping {model} with pruning level {level:.2f} - exceeds memory limits")
                        continue
                        
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
        runtime = time.time() - start_time
        print(f"\nCompleted {len(self.results)} experiments out of {len(experiments)} attempted")
        print(f"Total runtime: {runtime/3600:.2f} hours ({runtime/60:.2f} minutes)")
        
        # Plot final results
        self.plot_results()
        
        return self.results