#!/usr/bin/env python
# coding: utf-8

# # Pruning and Fine-Tuning Benchmark for Google Colab
# 
# This is the Python script version of our notebook for Google Colab.
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
# Install required packages
!pip install -q jax jaxlib flax transformers datasets matplotlib numpy pandas seaborn tqdm optax

# %%
# Clone the repository
!git clone -b feature/colab-overnight https://github.com/CambrianTech/sentinel-ai.git
%cd sentinel-ai

# %%
# Fix dataset import in Colab - IMPORTANT!
# When in Colab, we need to make sure we import from huggingface datasets,
# not our local datasets package

import sys
import os

# Completely remove sentinel-ai directory from sys.path
sys.path = [p for p in sys.path if 'sentinel-ai' not in p and p != '']

# Force install the huggingface datasets library to be safe
!pip install -q datasets

# Now, import dependencies from the HF datasets library first
from datasets import load_dataset

# Check that we're using the correct datasets package
import datasets
print(f"Using datasets from: {datasets.__file__}")

# Now we can add our repository to the path
sys.path.append(".")  # Make sure the repo root is in the path

# %%
# Import libraries
import os
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

# Import our pruning library
sys.path.append(".")  # Make sure the repo root is in the path
from utils.pruning import (
    Environment,
    ResultsManager,
    PruningModule, 
    get_strategy
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
# ## Fine-Tuning Implementation
# 
# First, let's implement the fine-tuning functionality:

# %%
class FineTuner:
    """Fine-tunes a pruned model to recover performance"""
    
    def __init__(self, pruning_module, dataset_name="openwebtext", batch_size=4):
        self.pruning_module = pruning_module
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.max_seq_length = 128  # Modest sequence length for faster training
        self.train_state = None
        self.metrics_history = []
        
        # Detect number of devices
        self.devices = jax.devices()
        self.n_devices = len(self.devices)
        if self.n_devices > 1:
            print(f"Using {self.n_devices} devices for training")
            self.batch_size = max(self.batch_size, self.n_devices)
            # Make batch size divisible by device count
            self.batch_size = (self.batch_size // self.n_devices) * self.n_devices
            print(f"Adjusted batch size to {self.batch_size} for multi-device training")
    
    def _prepare_dataset(self):
        """Load and prepare the dataset for fine-tuning"""
        try:
            # Try to load a small portion of the dataset for faster loading
            dataset = load_dataset(self.dataset_name, split="train[:5000]")
            
            # Process dataset
            tokenizer = self.pruning_module.tokenizer
            
            def tokenize_function(examples):
                # Tokenize the texts
                tokenized = tokenizer(examples["text"])
                return tokenized
            
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                num_proc=1,
                remove_columns=["text"]
            )
            
            # Create data loader
            def create_batch(samples):
                # Prepare batch of appropriate shape
                batch = {k: np.array(v) for k, v in samples.items()}
                
                # Create 'labels' for the causal language modeling task
                batch["labels"] = batch["input_ids"].copy()
                
                # Get sequence lengths
                seq_lengths = (batch["input_ids"] != tokenizer.pad_token_id).sum(axis=1)
                
                # Loop through samples and pad/truncate as needed
                for i, length in enumerate(seq_lengths):
                    # Ensure we have at least 2 tokens (can't shift with just 1)
                    if length < 2:
                        # Add padding to have at least 2 tokens
                        padding = np.array([tokenizer.pad_token_id] * (2 - length))
                        batch["input_ids"][i] = np.concatenate([batch["input_ids"][i][:length], padding])
                        batch["attention_mask"][i] = np.concatenate([batch["attention_mask"][i][:length], 
                                                                    np.ones_like(padding)])
                        batch["labels"][i] = np.concatenate([batch["labels"][i][:length], padding])
                        seq_lengths[i] = 2
                    
                    # Truncate to max sequence length if needed
                    if length > self.max_seq_length:
                        batch["input_ids"][i] = batch["input_ids"][i][:self.max_seq_length]
                        batch["attention_mask"][i] = batch["attention_mask"][i][:self.max_seq_length]
                        batch["labels"][i] = batch["labels"][i][:self.max_seq_length]
                        seq_lengths[i] = self.max_seq_length
                
                return batch
            
            # Create data loader
            dataloader = tokenized_dataset.batch(self.batch_size)
            dataloader = dataloader.map(create_batch, batched=True)
            
            return dataloader
        
        except Exception as e:
            print(f"Error preparing dataset: {e}")
            print("Falling back to synthetic data for training")
            return self._prepare_synthetic_dataset()
    
    def _prepare_synthetic_dataset(self):
        """Create synthetic data for training when dataset loading fails"""
        tokenizer = self.pruning_module.tokenizer
        
        # Generate random token IDs (avoid special tokens)
        vocab_size = tokenizer.vocab_size
        special_tokens = set([tokenizer.pad_token_id, tokenizer.eos_token_id, 
                             tokenizer.bos_token_id, tokenizer.unk_token_id])
        
        # Create 100 samples of random token sequences
        samples = []
        for _ in range(100):
            # Generate random length between 10 and max_seq_length
            length = np.random.randint(10, self.max_seq_length)
            
            # Generate random token IDs
            token_ids = np.random.randint(0, vocab_size, size=length)
            
            # Replace special tokens with normal tokens
            for i, token_id in enumerate(token_ids):
                if token_id in special_tokens:
                    token_ids[i] = (token_id + 1) % vocab_size
            
            # Create sample
            sample = {
                "input_ids": token_ids,
                "attention_mask": np.ones_like(token_ids),
                "labels": token_ids.copy()
            }
            samples.append(sample)
        
        # Create batches
        batches = []
        for i in range(0, len(samples), self.batch_size):
            batch_samples = samples[i:i+self.batch_size]
            
            # Pad to the same length within batch
            max_len = max(len(s["input_ids"]) for s in batch_samples)
            
            batch = {
                "input_ids": [],
                "attention_mask": [],
                "labels": []
            }
            
            for sample in batch_samples:
                pad_len = max_len - len(sample["input_ids"])
                batch["input_ids"].append(np.pad(sample["input_ids"], (0, pad_len), 
                                                constant_values=tokenizer.pad_token_id))
                batch["attention_mask"].append(np.pad(sample["attention_mask"], (0, pad_len), 
                                                    constant_values=0))
                batch["labels"].append(np.pad(sample["labels"], (0, pad_len), 
                                            constant_values=tokenizer.pad_token_id))
            
            # Convert to arrays
            batch = {k: np.array(v) for k, v in batch.items()}
            batches.append(batch)
        
        return batches
    
    def _create_train_state(self, params, learning_rate=5e-5):
        """Create a training state for the fine-tuning process"""
        # Create optimizer
        optimizer = optax.adam(learning_rate)
        
        # Create train state
        model = self.pruning_module.model
        return TrainState.create(
            apply_fn=model.__call__,
            params=params,
            tx=optimizer
        )
    
    def _loss_fn(self, params, batch):
        """Loss function for the language modeling task"""
        model = self.pruning_module.model
        
        # Get logits from model
        outputs = model(**batch, params=params, train=True)
        logits = outputs.logits
        
        # Get labels and create masks
        labels = batch["labels"]
        
        # Create loss mask (don't compute loss for padding tokens)
        loss_mask = (labels != self.pruning_module.tokenizer.pad_token_id)
        
        # Shift logits and labels for next token prediction
        shift_logits = logits[:, :-1]
        shift_labels = labels[:, 1:]
        shift_mask = loss_mask[:, 1:]
        
        # Calculate cross entropy loss
        loss = optax.softmax_cross_entropy_with_integer_labels(
            shift_logits, shift_labels
        )
        
        # Apply mask and calculate mean
        loss = (loss * shift_mask).sum() / shift_mask.sum()
        
        return loss
    
    def _train_step(self, state, batch):
        """Single training step"""
        grad_fn = jax.value_and_grad(self._loss_fn)
        loss, grads = grad_fn(state.params, batch)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss
    
    def fine_tune(self, pruned_params, num_epochs=1, learning_rate=5e-5, evaluate_interval=5):
        """Fine-tune the pruned model"""
        print(f"\nFine-tuning model with {self.dataset_name} dataset for {num_epochs} epochs...")
        
        # Prepare dataset
        dataset = self._prepare_dataset()
        
        # Create training state
        self.train_state = self._create_train_state(pruned_params, learning_rate)
        self.metrics_history = []
        
        # Training loop
        total_steps = 0
        perplexity_history = []
        
        for epoch in range(num_epochs):
            # Shuffled dataset for each epoch (if it's a list of batches)
            if isinstance(dataset, list):
                np.random.shuffle(dataset)
                epoch_dataset = dataset
            else:
                # If it's a datasets.Dataset, shuffle
                epoch_dataset = dataset.shuffle()
            
            # Create progress bar
            epoch_desc = f"Epoch {epoch+1}/{num_epochs}"
            batch_count = len(epoch_dataset) if hasattr(epoch_dataset, "__len__") else "?"
            progress_bar = tqdm(enumerate(epoch_dataset), desc=epoch_desc, 
                               total=batch_count if batch_count != "?" else None)
            
            epoch_losses = []
            
            for step, batch in progress_bar:
                # Train step
                self.train_state, loss = self._train_step(self.train_state, batch)
                total_steps += 1
                epoch_losses.append(loss.item())
                
                # Update progress bar
                progress_bar.set_description(f"{epoch_desc} - Loss: {loss.item():.4f}")
                
                # Evaluate periodically
                if total_steps % evaluate_interval == 0:
                    # Generate dummy text to check progress
                    prompt = "Artificial intelligence will transform"
                    try:
                        generated = self.pruning_module.generate_text(
                            self.train_state.params, prompt, max_length=30
                        )
                        perplexity = self.pruning_module.evaluate_perplexity(
                            self.train_state.params, prompt
                        )
                        perplexity_history.append((total_steps, perplexity))
                        print(f"\nStep {total_steps} - Perplexity: {perplexity:.4f}")
                        print(f"Generated: {generated}")
                    except Exception as e:
                        print(f"Error evaluating model: {e}")
            
            # End of epoch metrics
            epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            print(f"\nEpoch {epoch+1} completed. Average loss: {epoch_loss:.4f}")
            
            self.metrics_history.append({
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "perplexity_history": perplexity_history
            })
        
        print("\nFine-tuning completed!")
        return self.train_state.params, self.metrics_history
    
    def plot_training_progress(self, figsize=(12, 6)):
        """Plot training progress"""
        if not self.metrics_history:
            print("No training metrics available yet")
            return
        
        # Extract epoch losses
        epochs = [m["epoch"] for m in self.metrics_history]
        losses = [m["loss"] for m in self.metrics_history]
        
        # Extract perplexity history
        steps = []
        perplexities = []
        for m in self.metrics_history:
            for step, perplexity in m.get("perplexity_history", []):
                steps.append(step)
                perplexities.append(perplexity)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot losses
        ax1.plot(epochs, losses, "o-", color="blue")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.grid(True, linestyle="--", alpha=0.7)
        
        # Plot perplexities
        if steps and perplexities:
            ax2.plot(steps, perplexities, "o-", color="green")
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Perplexity")
            ax2.set_title("Perplexity During Training")
            ax2.grid(True, linestyle="--", alpha=0.7)
        else:
            ax2.text(0.5, 0.5, "No perplexity data available",
                    ha="center", va="center", fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        return fig

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
        
        # Create fine-tuner
        dataset_name = "wikitext"
        if self.env.in_colab and self.env.has_tpu:
            # TPUs can handle larger batch sizes
            batch_size = 16
        elif self.env.in_colab and self.env.has_gpu:
            batch_size = 8
        else:
            batch_size = 4
            
        fine_tuner = FineTuner(pruning_module, dataset_name=dataset_name, batch_size=batch_size)
        
        # Fine-tune model
        tuned_params, metrics = fine_tuner.fine_tune(
            pruned_params, 
            num_epochs=fine_tuning_epochs,
            learning_rate=5e-5,
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