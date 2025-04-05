#!/usr/bin/env python
"""
Complete Neural Plasticity Cycle Demonstration

This script demonstrates the full neural plasticity cycle (train → prune → measure → grow → learn)
with automatic tuning of pruning levels and growth strategies for optimal performance.

Example usage:
    python scripts/neural_plasticity_cycle.py --model_name distilgpt2 --dataset tiny_shakespeare
    python scripts/neural_plasticity_cycle.py --model_name gpt2 --initial_pruning 0.3 --growth_ratio 0.5
    python scripts/neural_plasticity_cycle.py --model_name facebook/opt-125m --cycles 3 --eval_every 100
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Sentinel-AI modules
from utils.pruning.fixed_pruning_module_jax import PruningModule
from utils.pruning.strategies import get_strategy as get_pruning_strategy
from utils.pruning.growth import (
    grow_attention_heads_gradually, 
    determine_active_heads,
    get_strategy as get_growth_strategy
)
from utils.pruning.head_lr_manager import HeadLRManager
from utils.train_utils import FineTuner
from utils.metrics_logger import MetricsLogger
from utils.charting import plot_head_distribution, plot_metrics_comparison

# Add missing function for plotting cycle comparison
def plot_cycle_comparison(metrics_by_cycle, metric="perplexity", title="Model Performance by Cycle", save_path=None):
    """
    Plot a comparison of metrics across cycles.
    
    Args:
        metrics_by_cycle: Dictionary of metrics by cycle
        metric: Metric to plot (perplexity, active_heads, etc.)
        title: Plot title
        save_path: Path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
        
        cycles = sorted(metrics_by_cycle.keys())
        values = [metrics_by_cycle[cycle].get(metric, 0) for cycle in cycles]
        
        plt.figure(figsize=(10, 6))
        plt.plot(cycles, values, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Cycle')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    except Exception as e:
        print(f"Error creating cycle comparison plot: {e}")
        # Continue execution even if visualization fails
from sdata.dataset_loader import load_dataset

# Fix torch import for FineTuner
import torch

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Complete Neural Plasticity Cycle Demonstration")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                      help="Model name (default: distilgpt2)")
    parser.add_argument("--dataset", type=str, default="tiny_shakespeare",
                      help="Dataset name (default: tiny_shakespeare)")
    parser.add_argument("--save_dir", type=str, default="./output/plasticity_experiments",
                      help="Directory to save experiment results (default: ./output/plasticity_experiments)")
    
    # Cycle parameters
    parser.add_argument("--cycles", type=int, default=1,
                      help="Number of complete plasticity cycles to run (default: 1)")
    parser.add_argument("--initial_training_steps", type=int, default=500,
                      help="Initial training steps before first pruning (default: 500)")
    parser.add_argument("--initial_pruning", type=float, default=0.3,
                      help="Initial pruning level as fraction of total heads (default: 0.3)")
    parser.add_argument("--growth_ratio", type=float, default=0.33,
                      help="Ratio of pruned heads to grow back (default: 0.33)")
    parser.add_argument("--learning_steps", type=int, default=300,
                      help="Learning steps after each growth phase (default: 300)")
    
    # Strategy parameters
    parser.add_argument("--pruning_strategy", type=str, default="entropy",
                      choices=["random", "magnitude", "entropy"],
                      help="Strategy for pruning (default: entropy)")
    parser.add_argument("--growth_strategy", type=str, default="gradient_sensitivity",
                      choices=["gradient_sensitivity", "entropy_gap", "balanced", "random"],
                      help="Strategy for growth (default: gradient_sensitivity)")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Batch size for training (default: 8)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                      help="Base learning rate (default: 5e-5)")
    parser.add_argument("--new_head_lr_multiplier", type=float, default=5.0,
                      help="Learning rate multiplier for new heads (default: 5.0)")
    parser.add_argument("--sequence_length", type=int, default=128,
                      help="Sequence length for training (default: 128)")
    
    # Evaluation parameters
    parser.add_argument("--eval_samples", type=int, default=5,
                      help="Number of samples for evaluation (default: 5)")
    parser.add_argument("--eval_every", type=int, default=50,
                      help="Evaluate every N steps (default: 50)")
    parser.add_argument("--generate_length", type=int, default=100,
                      help="Length of text to generate during evaluation (default: 100)")
    
    # Output parameters
    parser.add_argument("--save_model", action="store_true",
                      help="Save model checkpoints at each stage")
    parser.add_argument("--save_visualizations", action="store_true",
                      help="Save visualizations as PNG files")
    parser.add_argument("--experiment_name", type=str, default=None,
                      help="Name for this experiment (default: auto-generated)")
    parser.add_argument("--device", type=str, default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
                      help="Device to use (default: cuda if available, else cpu)")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose output")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed (default: 42)")
    
    return parser.parse_args()

def setup_experiment(args):
    """Set up the experiment directory and configuration"""
    # Create experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_short = args.model_name.split("/")[-1]
        args.experiment_name = f"plasticity_{model_short}_{timestamp}"
    
    # Create experiment directory
    experiment_dir = os.path.join(args.save_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    vis_dir = os.path.join(experiment_dir, "visualizations")
    metrics_dir = os.path.join(experiment_dir, "metrics")
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save experiment config
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, 'w') as f:
        config = vars(args)
        config["timestamp"] = datetime.now().isoformat()
        json.dump(config, f, indent=2)
    
    # Create metrics logger
    metrics_file = os.path.join(metrics_dir, "metrics.jsonl")
    metrics_logger = MetricsLogger(metrics_file)
    
    return {
        "experiment_dir": experiment_dir,
        "checkpoints_dir": checkpoints_dir,
        "vis_dir": vis_dir,
        "metrics_dir": metrics_dir,
        "metrics_logger": metrics_logger
    }

def load_or_create_model(args):
    """Load or create a pruning module with the specified model"""
    print(f"Loading model {args.model_name}...")
    
    # Create pruning module
    pruning_module = PruningModule(args.model_name)
    
    # Load model
    success = pruning_module.load_model()
    if not success:
        raise RuntimeError(f"Failed to load model {args.model_name}")
    
    # Get model information
    print(f"Model loaded successfully. Architecture: {pruning_module.model_type}")
    print(f"Layers: {pruning_module.num_layers}, Heads per layer: {pruning_module.num_heads}")
    print(f"Total heads: {pruning_module.num_layers * pruning_module.num_heads}")
    
    return pruning_module

def load_dataset_for_experiment(args, tokenizer=None):
    """Load the dataset for training and evaluation"""
    print(f"Loading dataset {args.dataset}...")
    
    try:
        # Load the dataset directly
        dataset_wrapper = load_dataset(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            max_length=args.sequence_length
        )
        
        # Our load_dataset now directly returns a DatasetWrapper
        return dataset_wrapper
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def evaluate_model(pruning_module, params, dataset, num_samples=5, generate_length=100):
    """Evaluate model performance on sample text"""
    # Get evaluation samples from dataset
    eval_samples = []
    
    # Get prompts from dataset if possible
    try:
        if hasattr(dataset, 'get_evaluation_samples'):
            eval_samples = dataset.get_evaluation_samples(num_samples)
        else:
            # Use default samples
            eval_samples = [
                "The transformer model processes data through multiple layers of computation.",
                "Artificial intelligence systems can learn from experience and improve over time.",
                "The neural network was trained to recognize patterns in complex datasets.",
                "Language models predict the next token based on previous context.",
                "The attention mechanism allows the model to focus on relevant parts of the input."
            ][:num_samples]
    except Exception as e:
        print(f"Error getting evaluation samples: {e}")
        # Fallback to default samples
        eval_samples = [
            "The transformer model processes data through multiple layers of computation.",
            "Artificial intelligence systems can learn from experience and improve over time."
        ]
    
    # Ensure we have enough samples
    if len(eval_samples) < num_samples:
        # Duplicate samples if necessary
        eval_samples = (eval_samples * ((num_samples // len(eval_samples)) + 1))[:num_samples]
    
    results = []
    perplexities = []
    
    for sample in eval_samples:
        # Calculate perplexity
        perplexity = pruning_module.evaluate_perplexity(params, sample)
        if not np.isnan(perplexity) and not np.isinf(perplexity):
            perplexities.append(perplexity)
        
        # Generate text
        prompt = sample[:30]
        generation = pruning_module.generate_text(params, prompt, max_length=generate_length)
        
        results.append({
            "prompt": prompt,
            "perplexity": float(perplexity) if not np.isnan(perplexity) and not np.isinf(perplexity) else None,
            "generation": generation
        })
    
    # Calculate average perplexity
    avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else float('nan')
    
    return {
        "samples": results,
        "average_perplexity": float(avg_perplexity),
        "perplexities": [float(p) for p in perplexities]
    }

def train_model(pruning_module, dataset, params=None, num_steps=500, lr=5e-5, 
                head_lr_manager=None, eval_every=50, metrics_logger=None):
    """Train the model for a specified number of steps"""
    print(f"Training model for {num_steps} steps...")
    
    # Initialize trainer
    trainer = FineTuner(
        pruning_module=pruning_module,
        dataset=dataset,
        learning_rate=lr,
        head_lr_manager=head_lr_manager
    )
    
    # Set initial parameters if provided
    if params is not None:
        trainer.set_params(params)
    
    # Training loop
    for step in tqdm(range(num_steps)):
        # Train for one step
        train_loss = trainer.train_step()
        
        # Log metrics
        if metrics_logger is not None:
            metrics_logger.log({
                "step": step,
                "phase": "training",
                "train_loss": float(train_loss)
            })
        
        # Evaluate periodically
        if step % eval_every == 0 or step == num_steps - 1:
            # Evaluate model
            eval_results = evaluate_model(
                pruning_module=pruning_module,
                params=trainer.get_params(),
                dataset=dataset
            )
            
            # Log evaluation metrics
            if metrics_logger is not None:
                metrics_logger.log({
                    "step": step,
                    "phase": "evaluation",
                    "perplexity": eval_results["average_perplexity"],
                    "sample_generation": eval_results["samples"][0]["generation"][:100]
                })
            
            # Print progress
            print(f"Step {step}: Loss = {train_loss:.4f}, Perplexity = {eval_results['average_perplexity']:.4f}")
    
    # Return final parameters
    return trainer.get_params()

def prune_model(pruning_module, params, pruning_level, strategy_name, metrics_logger=None):
    """Prune the model using the specified strategy and level"""
    print(f"Pruning model with {strategy_name} strategy at {pruning_level*100:.1f}% level...")
    
    # Get pruning strategy
    strategy = get_pruning_strategy(strategy_name, pruning_module)
    
    # Calculate importance scores for all heads
    head_importance = strategy.get_head_importance(params)
    
    # Sort by importance (ascending, so least important first)
    head_importance.sort(key=lambda x: x[2])
    
    # Calculate total heads and number to prune
    total_heads = pruning_module.num_layers * pruning_module.num_heads
    heads_to_prune = int(total_heads * pruning_level)
    
    # Select heads to prune (least important first)
    heads_to_prune = [(layer_idx, head_idx) for layer_idx, head_idx, _ in head_importance[:heads_to_prune]]
    
    # Prune the selected heads
    pruned_params = params.copy()
    for layer_idx, head_idx in heads_to_prune:
        pruned_params = pruning_module.prune_head(pruned_params, layer_idx, head_idx)
    
    # Determine active heads after pruning
    active_heads = determine_active_heads(pruning_module, pruned_params)
    
    # Log metrics
    if metrics_logger is not None:
        metrics_logger.log({
            "phase": "pruning",
            "pruning_level": pruning_level,
            "pruning_strategy": strategy_name,
            "heads_pruned": len(heads_to_prune),
            "active_heads_after_pruning": len(active_heads),
            "pruned_heads": [{"layer": int(l), "head": int(h)} for l, h in heads_to_prune]
        })
    
    return pruned_params, heads_to_prune, active_heads

def grow_model(pruning_module, params, active_heads, growth_ratio, strategy_name, metrics_logger=None):
    """Grow new attention heads using the specified strategy"""
    print(f"Growing heads with {strategy_name} strategy...")
    
    # Calculate number of pruned heads
    total_heads = pruning_module.num_layers * pruning_module.num_heads
    pruned_heads_count = total_heads - len(active_heads)
    
    # Calculate growth percentage (as fraction of total heads)
    growth_percentage = growth_ratio * (pruned_heads_count / total_heads)
    
    print(f"Growth percentage: {growth_percentage*100:.2f}% ({int(growth_percentage*total_heads)} heads)")
    
    # Grow attention heads
    grown_params, added_count, added_heads, warmup_schedule = grow_attention_heads_gradually(
        pruning_module,
        params=params,
        active_heads=active_heads,
        growth_percentage=growth_percentage,
        strategy=strategy_name,
        initial_scale=0.01,
        warmup_steps=100
    )
    
    # Determine active heads after growth
    grown_active_heads = determine_active_heads(pruning_module, grown_params)
    
    # Log metrics
    if metrics_logger is not None:
        metrics_logger.log({
            "phase": "growth",
            "growth_strategy": strategy_name,
            "growth_percentage": growth_percentage,
            "heads_added": added_count,
            "active_heads_after_growth": len(grown_active_heads),
            "added_heads": [{"layer": int(l), "head": int(h)} for l, h in added_heads]
        })
    
    return grown_params, added_count, added_heads, grown_active_heads, warmup_schedule

def neural_plasticity_cycle(args, experiment_dirs, pruning_module, dataset):
    """Run a complete neural plasticity cycle"""
    print("\n==== Starting Neural Plasticity Cycle ====\n")
    
    # Unpack experiment directories
    experiment_dir = experiment_dirs["experiment_dir"]
    checkpoints_dir = experiment_dirs["checkpoints_dir"]
    vis_dir = experiment_dirs["vis_dir"]
    metrics_logger = experiment_dirs["metrics_logger"]
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Step 1: Initial training
    print("\n=== Initial Training Phase ===\n")
    
    # Get initial model parameters
    initial_params = pruning_module.model.params
    
    # Visualize initial head distribution
    original_active_heads = determine_active_heads(pruning_module, initial_params)
    
    # Save head distribution visualization
    if args.save_visualizations:
        initial_vis_path = os.path.join(vis_dir, "initial_head_distribution.png")
        plot_head_distribution(
            pruning_module, 
            original_active_heads, 
            title="Initial Head Distribution",
            save_path=initial_vis_path
        )
    
    # Initial evaluation
    print("Evaluating initial model...")
    initial_eval = evaluate_model(
        pruning_module=pruning_module,
        params=initial_params,
        dataset=dataset,
        num_samples=args.eval_samples,
        generate_length=args.generate_length
    )
    
    print(f"Initial model average perplexity: {initial_eval['average_perplexity']:.4f}")
    
    # Log initial metrics
    metrics_logger.log({
        "phase": "initial",
        "model_name": args.model_name,
        "total_heads": pruning_module.num_layers * pruning_module.num_heads,
        "active_heads": len(original_active_heads),
        "perplexity": initial_eval["average_perplexity"]
    })
    
    # Train the initial model
    if args.initial_training_steps > 0:
        print(f"Training initial model for {args.initial_training_steps} steps...")
        initial_params = train_model(
            pruning_module=pruning_module,
            dataset=dataset,
            params=initial_params,
            num_steps=args.initial_training_steps,
            lr=args.learning_rate,
            eval_every=args.eval_every,
            metrics_logger=metrics_logger
        )
        
        # Evaluate trained model
        trained_eval = evaluate_model(
            pruning_module=pruning_module,
            params=initial_params,
            dataset=dataset,
            num_samples=args.eval_samples,
            generate_length=args.generate_length
        )
        
        print(f"Trained model average perplexity: {trained_eval['average_perplexity']:.4f}")
        
        # Save trained model
        if args.save_model:
            trained_model_path = os.path.join(checkpoints_dir, "model_trained.pth")
            with open(trained_model_path, 'wb') as f:
                import pickle
                pickle.dump(initial_params, f)
    
    # Run multiple cycles if requested
    for cycle in range(args.cycles):
        cycle_num = cycle + 1
        print(f"\n=== Neural Plasticity Cycle {cycle_num}/{args.cycles} ===\n")
        
        # Create cycle-specific directories
        cycle_vis_dir = os.path.join(vis_dir, f"cycle_{cycle_num}")
        os.makedirs(cycle_vis_dir, exist_ok=True)
        
        # Set current parameters
        current_params = initial_params
        
        # Step 2: Pruning Phase
        print("\n--- Pruning Phase ---\n")
        
        # Adjust pruning level based on cycle (optional)
        pruning_level = args.initial_pruning
        if cycle > 0:
            # Potentially increase pruning level in later cycles
            pruning_level = min(0.7, args.initial_pruning * (1 + 0.1 * cycle))
        
        # Prune the model
        pruned_params, pruned_heads, pruned_active_heads = prune_model(
            pruning_module=pruning_module,
            params=current_params,
            pruning_level=pruning_level,
            strategy_name=args.pruning_strategy,
            metrics_logger=metrics_logger
        )
        
        print(f"Pruned {len(pruned_heads)} heads, {len(pruned_active_heads)} active heads remaining")
        
        # Visualize pruned head distribution
        if args.save_visualizations:
            pruned_vis_path = os.path.join(cycle_vis_dir, "pruned_head_distribution.png")
            plot_head_distribution(
                pruning_module, 
                pruned_active_heads, 
                title=f"Pruned Head Distribution (Cycle {cycle_num})",
                save_path=pruned_vis_path
            )
        
        # Save pruned model
        if args.save_model:
            pruned_model_path = os.path.join(checkpoints_dir, f"model_pruned_cycle{cycle_num}.pth")
            with open(pruned_model_path, 'wb') as f:
                import pickle
                pickle.dump(pruned_params, f)
        
        # Step 3: Measurement Phase
        print("\n--- Measurement Phase ---\n")
        
        # Evaluate pruned model
        pruned_eval = evaluate_model(
            pruning_module=pruning_module,
            params=pruned_params,
            dataset=dataset,
            num_samples=args.eval_samples,
            generate_length=args.generate_length
        )
        
        print(f"Pruned model average perplexity: {pruned_eval['average_perplexity']:.4f}")
        
        # Calculate perplexity change
        perplexity_change = pruned_eval['average_perplexity'] - initial_eval['average_perplexity']
        perplexity_change_pct = (perplexity_change / initial_eval['average_perplexity']) * 100
        
        print(f"Perplexity change after pruning: {perplexity_change:+.4f} ({perplexity_change_pct:+.2f}%)")
        
        # Log metrics
        metrics_logger.log({
            "phase": "measurement",
            "cycle": cycle_num,
            "pruned_perplexity": pruned_eval["average_perplexity"],
            "perplexity_change": perplexity_change,
            "perplexity_change_percent": perplexity_change_pct
        })
        
        # Step 4: Growth Phase
        print("\n--- Growth Phase ---\n")
        
        # Grow new heads
        grown_params, added_count, added_heads, grown_active_heads, warmup_schedule = grow_model(
            pruning_module=pruning_module,
            params=pruned_params,
            active_heads=pruned_active_heads,
            growth_ratio=args.growth_ratio,
            strategy_name=args.growth_strategy,
            metrics_logger=metrics_logger
        )
        
        print(f"Added {added_count} heads, now have {len(grown_active_heads)} active heads")
        
        # Visualize grown head distribution
        if args.save_visualizations:
            grown_vis_path = os.path.join(cycle_vis_dir, "grown_head_distribution.png")
            plot_head_distribution(
                pruning_module, 
                grown_active_heads, 
                title=f"Grown Head Distribution (Cycle {cycle_num})",
                save_path=grown_vis_path
            )
        
        # Evaluate grown model with initial scaling
        grown_eval_initial = evaluate_model(
            pruning_module=pruning_module,
            params=grown_params,
            dataset=dataset,
            num_samples=args.eval_samples,
            generate_length=args.generate_length
        )
        
        print(f"Grown model (initial) average perplexity: {grown_eval_initial['average_perplexity']:.4f}")
        
        # Save grown model
        if args.save_model:
            grown_model_path = os.path.join(checkpoints_dir, f"model_grown_cycle{cycle_num}.pth")
            with open(grown_model_path, 'wb') as f:
                import pickle
                pickle.dump(grown_params, f)
        
        # Step 5: Learning Phase
        print("\n--- Learning Phase ---\n")
        
        # Create head learning rate manager
        head_lr_manager = HeadLRManager(
            base_lr=args.learning_rate,
            new_head_multiplier=args.new_head_lr_multiplier,
            new_heads=added_heads
        )
        
        # Train the grown model
        learned_params = train_model(
            pruning_module=pruning_module,
            dataset=dataset,
            params=grown_params,
            num_steps=args.learning_steps,
            lr=args.learning_rate,
            head_lr_manager=head_lr_manager,
            eval_every=args.eval_every,
            metrics_logger=metrics_logger
        )
        
        # Evaluate final model
        final_eval = evaluate_model(
            pruning_module=pruning_module,
            params=learned_params,
            dataset=dataset,
            num_samples=args.eval_samples,
            generate_length=args.generate_length
        )
        
        print(f"Final model average perplexity: {final_eval['average_perplexity']:.4f}")
        
        # Calculate final perplexity change
        final_perplexity_change = final_eval['average_perplexity'] - initial_eval['average_perplexity']
        final_perplexity_change_pct = (final_perplexity_change / initial_eval['average_perplexity']) * 100
        
        print(f"Final perplexity change: {final_perplexity_change:+.4f} ({final_perplexity_change_pct:+.2f}%)")
        
        # Log final metrics
        metrics_logger.log({
            "phase": "final",
            "cycle": cycle_num,
            "final_perplexity": final_eval["average_perplexity"],
            "active_heads": len(grown_active_heads),
            "active_heads_percentage": len(grown_active_heads) / (pruning_module.num_layers * pruning_module.num_heads) * 100,
            "perplexity_change_from_initial": final_perplexity_change,
            "perplexity_change_from_initial_percent": final_perplexity_change_pct,
            "perplexity_recovery": (pruned_eval["average_perplexity"] - final_eval["average_perplexity"]) / 
                                 (pruned_eval["average_perplexity"] - initial_eval["average_perplexity"]) * 100
                                 if pruned_eval["average_perplexity"] != initial_eval["average_perplexity"] else 0
        })
        
        # Save final model
        if args.save_model:
            final_model_path = os.path.join(checkpoints_dir, f"model_final_cycle{cycle_num}.pth")
            with open(final_model_path, 'wb') as f:
                import pickle
                pickle.dump(learned_params, f)
        
        # Create metrics visualization
        if args.save_visualizations:
            metrics_vis_path = os.path.join(cycle_vis_dir, "metrics_comparison.png")
            
            # Build metrics dictionary with checks for missing values
            perplexity_dict = {
                "Original": initial_eval["average_perplexity"],
                "Pruned": pruned_eval["average_perplexity"],
                "Final": final_eval["average_perplexity"]
            }
            
            # Only add Grown (Initial) if it's available and different from final
            if grown_eval_initial and "average_perplexity" in grown_eval_initial:
                perplexity_dict["Grown (Initial)"] = grown_eval_initial["average_perplexity"]
                
            metrics_data = {
                "perplexity": perplexity_dict,
                "active_heads_percentage": {
                    "Original": len(original_active_heads) / (pruning_module.num_layers * pruning_module.num_heads) * 100,
                    "Pruned": len(pruned_active_heads) / (pruning_module.num_layers * pruning_module.num_heads) * 100,
                    "Grown": len(grown_active_heads) / (pruning_module.num_layers * pruning_module.num_heads) * 100,
                    "Final": len(grown_active_heads) / (pruning_module.num_layers * pruning_module.num_heads) * 100
                }
            }
            
            try:
                plot_metrics_comparison(
                    metrics_data,
                    title=f"Neural Plasticity Cycle {cycle_num} Results",
                    save_path=metrics_vis_path
                )
            except Exception as e:
                print(f"Error creating metrics comparison visualization: {e}")
                # Continue execution even if visualization fails
        
        # Update initial parameters for next cycle
        initial_params = learned_params
    
    # Create summary visualization across all cycles
    if args.cycles > 1 and args.save_visualizations:
        summary_vis_path = os.path.join(vis_dir, "cycles_summary.png")
        
        # Load metrics for each cycle
        metrics_by_cycle = {}
        cycle_metrics = metrics_logger.load_metrics()
        
        # Extract final metrics for each cycle
        for metric in cycle_metrics:
            if metric.get("phase") == "final":
                cycle_num = metric.get("cycle")
                if cycle_num is not None:
                    metrics_by_cycle[cycle_num] = {
                        "perplexity": metric.get("final_perplexity", 0),
                        "active_heads": metric.get("active_heads", 0),
                        "perplexity_recovery": metric.get("perplexity_recovery", 0)
                    }
        
        # Create visualization if we have data
        if metrics_by_cycle:
            plot_cycle_comparison(
                metrics_by_cycle, 
                metric="perplexity",
                title=f"{args.model_name} Neural Plasticity Cycles",
                save_path=summary_vis_path
            )
            
            # Also create active heads visualization
            active_heads_vis_path = os.path.join(vis_dir, "active_heads_by_cycle.png")
            plot_cycle_comparison(
                metrics_by_cycle, 
                metric="active_heads",
                title=f"{args.model_name} Active Heads by Cycle",
                save_path=active_heads_vis_path
            )
            
            # Create recovery visualization
            recovery_vis_path = os.path.join(vis_dir, "perplexity_recovery_by_cycle.png")
            plot_cycle_comparison(
                metrics_by_cycle, 
                metric="perplexity_recovery",
                title=f"{args.model_name} Perplexity Recovery by Cycle",
                save_path=recovery_vis_path
            )
        
    return initial_params

def main():
    """Main function"""
    args = parse_args()
    
    # Set up experiment
    experiment_dirs = setup_experiment(args)
    
    # Load model once
    pruning_module = load_or_create_model(args)
    
    # Load dataset
    dataset = load_dataset_for_experiment(args, tokenizer=pruning_module.tokenizer)
    
    # Run neural plasticity cycle
    final_params = neural_plasticity_cycle(args, experiment_dirs, pruning_module, dataset)
    
    # Final evaluation
    final_eval = evaluate_model(
        pruning_module=pruning_module,
        params=final_params,
        dataset=dataset,
        num_samples=args.eval_samples,
        generate_length=args.generate_length
    )
    
    # Create summary report
    summary = {
        "experiment_name": args.experiment_name,
        "model_name": args.model_name,
        "dataset": args.dataset,
        "cycles": args.cycles,
        "final_perplexity": final_eval["average_perplexity"],
        "final_active_heads": len(determine_active_heads(pruning_module, final_params)),
        "total_heads": pruning_module.num_layers * pruning_module.num_heads,
        "pruning_strategy": args.pruning_strategy,
        "growth_strategy": args.growth_strategy,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save summary
    summary_path = os.path.join(experiment_dirs["experiment_dir"], "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nExperiment completed! Results saved to {experiment_dirs['experiment_dir']}")
    print(f"Final perplexity: {final_eval['average_perplexity']:.4f}")
    print(f"Active heads: {len(determine_active_heads(pruning_module, final_params))} / {pruning_module.num_layers * pruning_module.num_heads}")
    
if __name__ == "__main__":
    main()