#!/usr/bin/env python
# coding: utf-8

# # Pruning and Fine-Tuning Benchmark for Google Colab (v0.0.23)
# 
# This is the Python script version of our notebook for Google Colab.
# Version 0.0.23 (April 2025) - Critical fix for datasets import in Colab environment
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
# CRITICAL: This cell contains a robust fix for the datasets import conflict

# First, utility function to make the fix more maintainable
def fix_datasets_import():
    """Comprehensively fix the datasets module conflict in Colab"""
    import os
    import sys
    import shutil
    
    print("\n🧩 CRITICAL FIX: Resolving datasets import conflict...")
    print("This fix resolves conflicts between the HuggingFace datasets library and local datasets directory")
    
    # Approach 1: Move any existing code repository out of the path first
    # This ensures the right datasets module is imported first
    
    # Clean module cache first to be safe
    for k in list(sys.modules.keys()):
        if k == 'datasets' or k.startswith('datasets.'):
            del sys.modules[k]
            print(f"  ✓ Removed {k} from module cache")
    
    # Explicitly install the correct datasets again
    print("  ✓ Reinstalling datasets package...")
    !pip install --force-reinstall -q datasets>=2.0.0
    
    # Safe import of datasets before touching the repository
    # This ensures Python caches the right module first
    print("  ✓ Preloading correct datasets module...")
    try:
        # Import the correct datasets module
        print("  ✓ Importing datasets library from pip installation...")
        import datasets as ds_correct
        from datasets import load_dataset as ld_correct
        print(f"  ✓ Successfully preloaded datasets from: {ds_correct.__file__}")
    except ImportError as e:
        print(f"  ✗ Error preloading datasets: {e}")
        print("  ✓ Will try alternative approach")
        
    # Approach 2: Clone the repository only after the correct dataset module is cached
    print("\n🔄 Cloning the repository...")
    !git clone -b refactor/modular-experiment https://github.com/CambrianTech/sentinel-ai.git
    
    # Create symlink for Colab compatibility
    !ln -sf sentinel-ai refactor
    
    # CRITICAL: Now we need to handle the local datasets directory to prevent conflicts
    # It's safe to physically remove it since we've preloaded the correct module
    if os.path.exists('/content/sentinel-ai/datasets'):
        print("\n🧹 Backing up and removing local datasets directory...")
        
        # First, backup the directory content
        try:
            backup_dir = '/content/datasets_backup'
            os.makedirs(backup_dir, exist_ok=True)
            
            # Simple directory copy
            for item in os.listdir('/content/sentinel-ai/datasets'):
                src = os.path.join('/content/sentinel-ai/datasets', item)
                dst = os.path.join(backup_dir, item)
                if os.path.isdir(src):
                    # Use shell copy for reliability in Colab
                    !cp -r "{src}" "{dst}"
                else:
                    !cp "{src}" "{dst}"
            
            print("  ✓ Backed up datasets directory")
            
            # Now remove the directory using a shell command (most reliable in Colab)
            !rm -rf /content/sentinel-ai/datasets
            print("  ✓ Removed datasets directory")
            
        except Exception as e:
            print(f"  ✗ Error handling datasets directory: {e}")
            print("  ✓ Trying direct shell commands...")
            
            # Fallback to pure shell commands
            !mkdir -p /content/datasets_backup
            !cp -r /content/sentinel-ai/datasets/* /content/datasets_backup/ 2>/dev/null || true
            !rm -rf /content/sentinel-ai/datasets
            print("  ✓ Used shell commands to handle directory")
    
    # Change to the repository directory now that it's safe
    print("\n📂 Changing to repository directory...")
    %cd /content/sentinel-ai
    
    # Final verification - make sure we have the right datasets module
    print("\n🔍 Verifying datasets module...")
    try:
        import datasets
        from datasets import load_dataset
        
        # Verify content
        if hasattr(datasets, 'load_dataset'):
            print(f"✅ SUCCESS! Using correct datasets module from: {datasets.__file__}")
            # Add file to make sure our directory is never accidentally imported
            if not os.path.exists('datasets'):
                os.makedirs('datasets', exist_ok=True)
            with open('datasets/__init__.py', 'w') as f:
                f.write('# This is a placeholder to prevent accidental imports\n')
                f.write('# The real datasets module is from HuggingFace\n')
                f.write('raise ImportError("Please import datasets from HuggingFace, not this local directory")\n')
            print("✅ Created safety placeholder in local datasets directory")
        else:
            print(f"⚠️ Warning: datasets module loaded but missing load_dataset function")
            
    except ImportError as e:
        print(f"❌ ERROR importing datasets after fix: {e}")
        print("Please restart the runtime and try again")
    
    # Restore the saved version of the datasets directory
    # This is safe to do after the correct module is cached by Python
    if os.path.exists('/content/datasets_backup'):
        if os.path.exists('datasets'):
            print("  ✓ Local datasets directory already exists, not restoring backup")
        else:
            print("  ✓ Restoring datasets directory content...")
            !cp -r /content/datasets_backup/* datasets/ 2>/dev/null || true
            
    print("\n✅ IMPORT FIX COMPLETE - you can now run the rest of the notebook")
    return True

# Run the fix function
fix_datasets_import()

# %%
# At this point we should have the correct datasets module imported
# Now import the rest of what we need
import datasets
from datasets import load_dataset
print(f"Using datasets from: {datasets.__file__}")

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
    ImprovedFineTuner
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
class PruningFineTuningExperiment:
    """Manages the pruning + fine-tuning experiment"""
    
    def __init__(self, results_dir="pruning_finetuning_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.results = []
        self.current_experiment = {}
        
        # Initialize environment
        self.env = Environment()
        
        # Add has_high_ram attribute if not present in Environment class
        # This ensures backward compatibility while also providing better memory detection
        if not hasattr(self.env, 'has_high_ram'):
            self.env.has_high_ram = False
            try:
                import psutil
                total_ram = psutil.virtual_memory().total / (1024**3)  # Convert to GB
                self.env.has_high_ram = total_ram > 12
                print(f"Detected {total_ram:.1f}GB RAM, high RAM: {self.env.has_high_ram}")
            except:
                print("Could not detect RAM, assuming standard memory")
                
        # Detect GPU memory if possible
        self.gpu_memory_gb = 0
        if self.env.has_gpu:
            try:
                import torch
                gpu_props = torch.cuda.get_device_properties(0)
                self.gpu_memory_gb = gpu_props.total_memory / (1024**3)
                print(f"Detected GPU with {self.gpu_memory_gb:.1f}GB VRAM")
            except:
                try:
                    # Alternative method for JAX
                    device = jax.devices()[0]
                    if hasattr(device, 'memory_stats') and callable(device.memory_stats):
                        memory_stats = device.memory_stats()
                        if 'bytes_limit' in memory_stats:
                            self.gpu_memory_gb = memory_stats['bytes_limit'] / (1024**3)
                            print(f"Detected GPU with approximately {self.gpu_memory_gb:.1f}GB VRAM")
                except:
                    # Estimate based on common GPU types
                    if self.env.in_colab:
                        # T4 in Colab typically has 16GB
                        self.gpu_memory_gb = 16
                        print(f"Estimating Colab GPU with {self.gpu_memory_gb}GB VRAM")
        
        # Get suitable models for this environment
        self.available_models = self.env.get_suitable_models()
        print(f"Models available: {', '.join(self.available_models)}")
        
        # Model size limits based on environment - adapt based on detected resources
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
        if self.env.has_gpu and self.gpu_memory_gb >= 8:
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
        
        # 3. Fine-tune the pruned model - with improved stability
        print("\n>> Stage 3: Fine-tuning the pruned model")
        
        # Create fine-tuner with dataset config
        dataset_name = "wikitext"
        dataset_config = "wikitext-2-v1"
        
        # Determine batch size based on model size and environment
        if self.env.in_colab and self.env.has_tpu:
            # TPUs can handle larger batch sizes
            batch_size = 16
            # But still reduce for large models
            if "1.3b" in model.lower() or "large" in model.lower():
                batch_size = 8
        elif self.env.in_colab and self.env.has_gpu:
            batch_size = 8
            # Reduce batch size for larger models
            if "1.3b" in model.lower() or "large" in model.lower():
                batch_size = 4
            elif "2.7b" in model.lower() or "xl" in model.lower():
                batch_size = 2
        else:
            # CPU-only case
            batch_size = 4
            # Even smaller for large models on CPU
            if "1.3b" in model.lower() or "large" in model.lower():
                batch_size = 2
            elif "2.7b" in model.lower() or "xl" in model.lower():
                batch_size = 1
        
        # Use ImprovedFineTuner by default for all models to enhance stability
        print(f"Using ImprovedFineTuner for enhanced stability")
        fine_tuner = ImprovedFineTuner(
            pruning_module, 
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            batch_size=batch_size
        )
        
        # Adjust learning rate based on model size
        model_name = model.lower()
        is_large_model = any(x in model_name for x in ['opt', '1.3b', 'large', 'bloom', '2.7b', 'xl'])
        learning_rate = 1e-5 if is_large_model else 5e-5
        
        # Fine-tune model
        try:
            # Install a emergency NaN detection system
            def emergency_fix_loss_fn(params, batch):
                """Safe loss function that prevents NaN values"""
                model = pruning_module.model
                
                # Extract labels from batch but don't pass them to the model
                labels = batch.pop("labels", None)
                
                # Check for NaNs in input
                for k, v in batch.items():
                    if jnp.isnan(v).any() or jnp.isinf(v).any():
                        # Replace NaNs with zeros
                        batch[k] = jnp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                        print(f"Warning: Found NaN in {k}, replaced with zeros")
                
                try:
                    # Handle different model architectures
                    is_opt_model = 'opt' in model_name
                    
                    if is_opt_model:
                        # OPT models don't accept 'train' parameter
                        outputs = model(**batch, params=params)
                    else:
                        # Other models might need the 'train' parameter
                        outputs = model(**batch, params=params, train=True)
                        
                    logits = outputs.logits
                    
                    # Add labels back to batch
                    batch["labels"] = labels
                    
                    # Create mask for padding tokens
                    pad_token_id = pruning_module.tokenizer.pad_token_id  
                    loss_mask = (labels != pad_token_id)
                    
                    # Shift logits and labels
                    shift_logits = logits[:, :-1]
                    shift_labels = labels[:, 1:]
                    shift_mask = loss_mask[:, 1:]
                    
                    # Check for NaNs in logits
                    if jnp.isnan(shift_logits).any() or jnp.isinf(shift_logits).any():
                        print("Warning: Found NaN/Inf in logits - replacing with finite values")
                        shift_logits = jnp.nan_to_num(shift_logits, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    # Safe cross entropy computation
                    loss = optax.softmax_cross_entropy_with_integer_labels(
                        shift_logits, shift_labels
                    )
                    
                    # Check and fix NaNs in loss
                    if jnp.isnan(loss).any() or jnp.isinf(loss).any():
                        print("Warning: Found NaN/Inf in loss - replacing with finite values")
                        loss = jnp.nan_to_num(loss, nan=1.0, posinf=1.0, neginf=1.0)
                    
                    # Safe mean calculation
                    masked_loss = loss * shift_mask
                    mask_sum = shift_mask.sum()
                    
                    # Safe division
                    computed_loss = jnp.where(
                        mask_sum > 0,
                        masked_loss.sum() / mask_sum,
                        jnp.array(0.0, dtype=loss.dtype)
                    )
                    
                    if jnp.isnan(computed_loss) or jnp.isinf(computed_loss):
                        print("CRITICAL: NaN/Inf in final loss - using fallback value")
                        return jnp.array(1.0, dtype=loss.dtype)
                        
                    return computed_loss
                    
                except Exception as e:
                    print(f"Error in loss function: {e}")
                    # Return a safe default value
                    batch["labels"] = labels  # Restore labels
                    return jnp.array(1.0)  # Safe fallback
            
            # Patch the fine tuner with our stability utilities
            print("⚠️ Installing NaN-safe loss function")
            fine_tuner = patch_fine_tuner(fine_tuner, model_name=model)
            
            # Optimize memory usage based on model size and available GPU memory
            print(f"🧠 Optimizing memory usage for {model}")
            fine_tuner = optimize_fine_tuner(fine_tuner, model_name=model, gpu_memory_gb=self.gpu_memory_gb)
            
            # Proceed with fine-tuning
            tuned_params, metrics = fine_tuner.fine_tune(
                pruned_params, 
                num_epochs=fine_tuning_epochs,
                learning_rate=learning_rate,
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
        
        except Exception as e:
            print(f"Error during fine-tuning: {e}")
            # Continue with partial results
            import traceback
            traceback.print_exc()
        
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