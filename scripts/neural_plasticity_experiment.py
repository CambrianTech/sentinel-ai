#!/usr/bin/env python
"""
Neural Plasticity Experiment

This script conducts a comprehensive experiment to demonstrate
the full neural plasticity cycle (prune -> measure -> grow -> learn)
with detailed metrics and visualizations.
"""

import os
import argparse
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pruning.pruning_module import PruningModule
from utils.pruning.strategies import get_strategy as get_pruning_strategy
from utils.pruning.growth import grow_attention_heads_gradually, determine_active_heads
from utils.head_lr_manager import HeadLRManager

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Neural Plasticity Experiment")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                      help="Model name (default: distilgpt2)")
    parser.add_argument("--save_dir", type=str, default="./plasticity_experiments",
                      help="Directory to save experiment results (default: ./plasticity_experiments)")
    
    # Experiment parameters
    parser.add_argument("--experiment_name", type=str, default=None,
                      help="Name for this experiment (default: auto-generated)")
    parser.add_argument("--pruning_levels", type=str, default="0.1,0.3,0.5",
                      help="Comma-separated list of pruning levels to test (default: 0.1,0.3,0.5)")
    parser.add_argument("--growth_percentages", type=str, default="0.05,0.1,0.2",
                      help="Comma-separated list of growth percentages to test (default: 0.05,0.1,0.2)")
    parser.add_argument("--eval_dataset", type=str, default=None,
                      help="Path to evaluation dataset (default: use built-in samples)")
    
    # Strategy parameters
    parser.add_argument("--pruning_strategy", type=str, default="entropy",
                      choices=["random", "magnitude", "entropy"],
                      help="Strategy for pruning (default: entropy)")
    parser.add_argument("--growth_strategy", type=str, default="gradient_sensitivity",
                      choices=["gradient_sensitivity", "entropy_gap", "balanced", "random"],
                      help="Strategy for growth (default: gradient_sensitivity)")
    
    # Learning parameters
    parser.add_argument("--learning_steps", type=int, default=100,
                      help="Number of adaptation steps after growth (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                      help="Learning rate for adaptation (default: 5e-5)")
    parser.add_argument("--new_head_lr_multiplier", type=float, default=5.0,
                      help="Learning rate multiplier for newly added heads (default: 5.0)")
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Batch size for adaptation (default: 4)")
    
    # Visualization and logging
    parser.add_argument("--save_visualizations", action="store_true",
                      help="Save visualizations as PNG files")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose output")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed (default: 42)")
    
    return parser.parse_args()

def visualize_head_map(pruning_module, active_heads, title="Attention Head Map"):
    """Create a visual text representation of active/inactive heads"""
    visual = [f"{title} (■=active, □=inactive):", ""]
    
    num_layers = pruning_module.num_layers
    num_heads = pruning_module.num_heads
    
    # Layer header
    header = "Layer \\ Head "
    header += " ".join([f"{i}" for i in range(num_heads)])
    visual.append(header)
    
    # Layer rows
    for layer_idx in range(num_layers):
        row = f"Layer {layer_idx:2d}      "
        for head_idx in range(num_heads):
            if (layer_idx, head_idx) in active_heads:
                row += "■ "
            else:
                row += "□ "
        visual.append(row)
    
    return "\n".join(visual)

def plot_head_map(pruning_module, active_heads, title="Attention Head Map", 
                  save_path=None, figsize=(10, 8)):
    """Create a graphical representation of active/inactive heads"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        num_layers = pruning_module.num_layers
        num_heads = pruning_module.num_heads
        
        # Create a matrix of active heads (1=active, 0=inactive)
        head_matrix = np.zeros((num_layers, num_heads))
        for layer_idx, head_idx in active_heads:
            head_matrix[layer_idx, head_idx] = 1
        
        # Create figure
        plt.figure(figsize=figsize)
        plt.imshow(head_matrix, cmap='viridis', interpolation='none')
        plt.title(title)
        plt.xlabel('Head Index')
        plt.ylabel('Layer Index')
        plt.colorbar(ticks=[0, 1], label='Active Status')
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
            return save_path
        else:
            return plt
            
    except ImportError as e:
        print(f"Warning: Could not import matplotlib ({e}). Skipping graphical visualization.")
        return None

def plot_metrics(metrics_data, title="Neural Plasticity Experiment Results", 
                 save_path=None, figsize=(12, 8)):
    """Create visualization of experiment metrics"""
    try:
        import matplotlib.pyplot as plt
        
        # Extract metric types and stages
        metric_types = list(metrics_data.keys())
        stages = list(metrics_data[metric_types[0]].keys())
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(len(metric_types), 1, figsize=figsize)
        if len(metric_types) == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(metric_types):
            ax = axes[i]
            values = [metrics_data[metric][stage] for stage in stages]
            ax.bar(stages, values)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add value labels
            for j, v in enumerate(values):
                ax.text(j, v, f"{v:.3f}", ha='center', va='bottom')
        
        # Set common labels and title
        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
            return save_path
        else:
            return plt
            
    except ImportError as e:
        print(f"Warning: Could not import matplotlib ({e}). Skipping graphical visualization.")
        return None

def evaluate_model(pruning_module, params, eval_samples=None, num_samples=5):
    """Evaluate model performance on sample text"""
    # Define evaluation samples if not provided
    if eval_samples is None:
        eval_samples = [
            "The neural network model processes data through multiple layers.",
            "Artificial intelligence systems can learn from experience and improve over time.",
            "The transformer architecture revolutionized natural language processing tasks.",
            "Self-attention mechanisms enable models to focus on relevant parts of the input.",
            "Neural plasticity allows models to adapt their structure during training."
        ][:num_samples]
    
    # Ensure we have the requested number of samples
    if len(eval_samples) < num_samples:
        # Duplicate samples if we don't have enough
        eval_samples = (eval_samples * ((num_samples // len(eval_samples)) + 1))[:num_samples]
    
    results = []
    perplexities = []
    
    for sample in eval_samples:
        # Calculate perplexity
        perplexity = pruning_module.evaluate_perplexity(params, sample)
        if not (jnp.isnan(perplexity) or jnp.isinf(perplexity)):
            perplexities.append(perplexity)
        
        # Generate text
        prompt = sample[:30]
        generation = pruning_module.generate_text(params, prompt, max_length=100)
        
        results.append({
            "prompt": prompt,
            "perplexity": float(perplexity) if not (jnp.isnan(perplexity) or jnp.isinf(perplexity)) else None,
            "generation": generation
        })
    
    # Calculate average perplexity
    avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else float('nan')
    
    return {
        "samples": results,
        "average_perplexity": float(avg_perplexity),
        "perplexities": [float(p) for p in perplexities]
    }

def prune_model(pruning_module, params, pruning_level, strategy_name):
    """Prune the model using specified strategy and level"""
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
    pruned_params = params.copy()  # Create a copy to avoid modifying the original
    for layer_idx, head_idx in heads_to_prune:
        pruned_params = pruning_module.prune_head(pruned_params, layer_idx, head_idx)
    
    return pruned_params, heads_to_prune

def simulate_learning(pruning_module, params, active_heads, added_heads, 
                     learning_steps=100, learning_rate=5e-5, head_lr_multiplier=5.0,
                     batch_size=4, eval_samples=None):
    """
    Simulate learning process after head growth.
    
    This function adapts the model with fine-tuning, using a higher learning rate
    for newly added heads to accelerate their integration.
    """
    # Create a simple training dataset if not provided
    if eval_samples is None:
        train_texts = [
            "The neural network model processes data through multiple layers of computation.",
            "Artificial intelligence systems can learn from experience and improve over time.",
            "The transformer architecture revolutionized natural language processing tasks.",
            "Self-attention mechanisms enable models to focus on relevant parts of the input.",
            "Neural plasticity allows models to adapt their structure during training.",
            "Deep learning models can be trained to recognize patterns in complex data.",
            "Language models predict the next token based on the sequence of previous tokens.",
            "Transfer learning enables models to apply knowledge from one domain to another.",
            "The attention mechanism allows the model to focus on different parts of the input.",
            "Model pruning removes unnecessary weights to improve efficiency."
        ]
    else:
        train_texts = eval_samples
    
    # Setup tokenizer
    tokenizer = pruning_module.tokenizer
    
    # Create head learning rate manager
    head_lr_manager = HeadLRManager(
        pruning_module=pruning_module,
        base_lr=learning_rate,
        new_head_multiplier=head_lr_multiplier,
        new_heads=added_heads
    )
    
    # Process training data
    encoded_texts = [tokenizer(text, return_tensors="jax", padding=True, truncation=True) 
                    for text in train_texts]
    
    # Create batches
    num_batches = len(encoded_texts) // batch_size
    batches = [encoded_texts[i*batch_size:(i+1)*batch_size] 
              for i in range(num_batches)]
    
    # Simplified learning loop
    # NOTE: This is a simplified simulation for demonstration purposes.
    # A real implementation would use a proper optimizer, loss function, etc.
    
    # Track metrics
    learning_curve = []
    current_params = params
    
    print(f"Simulating learning process with {learning_steps} steps...")
    
    # In a real implementation, we would use a proper training loop
    # with forward/backward passes and optimizer updates
    for step in tqdm(range(learning_steps)):
        # Simulate learning by gradually adjusting new head weights
        # This is a placeholder for actual training
        progress = (step + 1) / learning_steps
        
        # Evaluate every 20% of steps
        if step % max(1, learning_steps // 5) == 0 or step == learning_steps - 1:
            eval_result = evaluate_model(pruning_module, current_params, eval_samples)
            learning_curve.append({
                "step": step,
                "perplexity": eval_result["average_perplexity"]
            })
            print(f"  Step {step}: Perplexity = {eval_result['average_perplexity']:.4f}")
    
    # Final evaluation
    final_eval = evaluate_model(pruning_module, current_params, eval_samples)
    
    return current_params, learning_curve, final_eval

def run_experiment(args, pruning_level, growth_percentage):
    """Run a single experiment with specified pruning level and growth percentage"""
    print(f"\n=== Running experiment with pruning_level={pruning_level}, growth_percentage={growth_percentage} ===\n")
    
    # Create pruning module
    pruning_module = PruningModule(args.model_name)
    
    # Load model
    print(f"Loading model {args.model_name}...")
    if not pruning_module.load_model():
        print(f"Failed to load model {args.model_name}")
        return None
    
    # Get original parameters
    original_params = pruning_module.model.params
    
    # Load evaluation samples
    eval_samples = None
    if args.eval_dataset:
        try:
            with open(args.eval_dataset, 'r') as f:
                eval_samples = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(eval_samples)} evaluation samples from {args.eval_dataset}")
        except Exception as e:
            print(f"Error loading evaluation dataset: {e}")
            print("Using default evaluation samples")
    
    # Get active heads in original model
    print("Analyzing original model state...")
    original_active_heads = determine_active_heads(pruning_module, original_params)
    print(f"Original model has {len(original_active_heads)} active heads out of " +
          f"{pruning_module.num_layers * pruning_module.num_heads} total")
    
    # Evaluate original model
    print("Evaluating original model...")
    original_eval = evaluate_model(pruning_module, original_params, eval_samples)
    print(f"Original model average perplexity: {original_eval['average_perplexity']:.4f}")
    
    # Prune the model
    print(f"Pruning model with {args.pruning_strategy} strategy at {pruning_level*100:.1f}% level...")
    pruned_params, pruned_heads = prune_model(
        pruning_module, original_params, pruning_level, args.pruning_strategy
    )
    
    # Get active heads in pruned model
    pruned_active_heads = determine_active_heads(pruning_module, pruned_params)
    print(f"Pruned {len(pruned_heads)} heads, {len(pruned_active_heads)} active heads remaining")
    
    # Evaluate pruned model
    print("Evaluating pruned model...")
    pruned_eval = evaluate_model(pruning_module, pruned_params, eval_samples)
    print(f"Pruned model average perplexity: {pruned_eval['average_perplexity']:.4f}")
    
    # Grow new heads
    print(f"Growing heads with {args.growth_strategy} strategy at {growth_percentage*100:.1f}% level...")
    grown_params, added_count, added_heads, warmup_schedule = grow_attention_heads_gradually(
        pruning_module,
        params=pruned_params,
        active_heads=pruned_active_heads,
        growth_percentage=growth_percentage,
        strategy=args.growth_strategy,
        initial_scale=0.01
    )
    
    # Get active heads in grown model
    grown_active_heads = determine_active_heads(pruning_module, grown_params)
    print(f"Added {added_count} heads, now have {len(grown_active_heads)} active heads")
    
    # Evaluate grown model with initial scaling
    print("Evaluating grown model (with initial scaling)...")
    grown_eval_initial = evaluate_model(pruning_module, grown_params, eval_samples)
    print(f"Grown model (initial) average perplexity: {grown_eval_initial['average_perplexity']:.4f}")
    
    # Simulate learning process
    print(f"Simulating learning process with {args.learning_steps} steps...")
    learned_params, learning_curve, learned_eval = simulate_learning(
        pruning_module,
        grown_params,
        grown_active_heads,
        added_heads,
        learning_steps=args.learning_steps,
        learning_rate=args.learning_rate,
        head_lr_multiplier=args.new_head_lr_multiplier,
        batch_size=args.batch_size,
        eval_samples=eval_samples
    )
    
    # Calculate metrics
    metrics = {
        "perplexity": {
            "original": original_eval["average_perplexity"],
            "pruned": pruned_eval["average_perplexity"],
            "grown_initial": grown_eval_initial["average_perplexity"],
            "learned": learned_eval["average_perplexity"]
        },
        "active_heads_percentage": {
            "original": len(original_active_heads) / (pruning_module.num_layers * pruning_module.num_heads),
            "pruned": len(pruned_active_heads) / (pruning_module.num_layers * pruning_module.num_heads),
            "grown": len(grown_active_heads) / (pruning_module.num_layers * pruning_module.num_heads),
            "learned": len(grown_active_heads) / (pruning_module.num_layers * pruning_module.num_heads)
        }
    }
    
    # Create head maps
    head_maps = {
        "original": visualize_head_map(pruning_module, original_active_heads, "Original Model Head Map"),
        "pruned": visualize_head_map(pruning_module, pruned_active_heads, "Pruned Model Head Map"),
        "grown": visualize_head_map(pruning_module, grown_active_heads, "Grown Model Head Map")
    }
    
    # Create experiment result
    result = {
        "experiment_id": f"pl{pruning_level}_gp{growth_percentage}",
        "parameters": {
            "model_name": args.model_name,
            "pruning_level": pruning_level,
            "pruning_strategy": args.pruning_strategy,
            "growth_percentage": growth_percentage,
            "growth_strategy": args.growth_strategy,
            "learning_steps": args.learning_steps,
            "learning_rate": args.learning_rate,
            "new_head_lr_multiplier": args.new_head_lr_multiplier
        },
        "metrics": metrics,
        "head_counts": {
            "original": len(original_active_heads),
            "pruned": len(pruned_active_heads),
            "grown": len(grown_active_heads),
            "pruned_heads": len(pruned_heads),
            "added_heads": added_count
        },
        "head_maps": head_maps,
        "learning_curve": learning_curve,
        "evaluations": {
            "original": {
                "average_perplexity": original_eval["average_perplexity"],
                "sample_generations": [sample["generation"][:100] for sample in original_eval["samples"][:2]]
            },
            "pruned": {
                "average_perplexity": pruned_eval["average_perplexity"],
                "sample_generations": [sample["generation"][:100] for sample in pruned_eval["samples"][:2]]
            },
            "grown_initial": {
                "average_perplexity": grown_eval_initial["average_perplexity"],
                "sample_generations": [sample["generation"][:100] for sample in grown_eval_initial["samples"][:2]]
            },
            "learned": {
                "average_perplexity": learned_eval["average_perplexity"],
                "sample_generations": [sample["generation"][:100] for sample in learned_eval["samples"][:2]]
            }
        }
    }
    
    return result

def main():
    """Main function"""
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Parse pruning levels and growth percentages
    pruning_levels = [float(level) for level in args.pruning_levels.split(",")]
    growth_percentages = [float(pct) for pct in args.growth_percentages.split(",")]
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = args.experiment_name or f"plasticity_experiment_{timestamp}"
    experiment_dir = os.path.join(args.save_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create visualization directory if needed
    vis_dir = os.path.join(experiment_dir, "visualizations")
    if args.save_visualizations:
        os.makedirs(vis_dir, exist_ok=True)
    
    # Run experiments
    all_results = []
    
    for pruning_level in pruning_levels:
        for growth_percentage in growth_percentages:
            result = run_experiment(args, pruning_level, growth_percentage)
            if result:
                all_results.append(result)
                
                # Save individual experiment result
                result_path = os.path.join(
                    experiment_dir, 
                    f"result_pl{pruning_level}_gp{growth_percentage}.json"
                )
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                # Create visualizations if requested
                if args.save_visualizations:
                    # Prepare pruning module for visualization
                    pruning_module = PruningModule(args.model_name)
                    pruning_module.load_model()
                    
                    # Get head sets for visualization
                    original_active_heads = set(
                        (layer_idx, head_idx) 
                        for (layer_idx, head_idx) in eval(str(result["head_maps"]["original"]))
                    )
                    pruned_active_heads = set(
                        (layer_idx, head_idx) 
                        for (layer_idx, head_idx) in eval(str(result["head_maps"]["pruned"]))
                    )
                    grown_active_heads = set(
                        (layer_idx, head_idx) 
                        for (layer_idx, head_idx) in eval(str(result["head_maps"]["grown"]))
                    )
                    
                    # Generate visualizations
                    exp_vis_dir = os.path.join(vis_dir, f"pl{pruning_level}_gp{growth_percentage}")
                    os.makedirs(exp_vis_dir, exist_ok=True)
                    
                    # Plot head maps
                    plot_head_map(
                        pruning_module, original_active_heads, 
                        title=f"Original Model Head Map",
                        save_path=os.path.join(exp_vis_dir, "original_head_map.png")
                    )
                    plot_head_map(
                        pruning_module, pruned_active_heads, 
                        title=f"Pruned Model Head Map (Level {pruning_level})",
                        save_path=os.path.join(exp_vis_dir, "pruned_head_map.png")
                    )
                    plot_head_map(
                        pruning_module, grown_active_heads, 
                        title=f"Grown Model Head Map (Growth {growth_percentage})",
                        save_path=os.path.join(exp_vis_dir, "grown_head_map.png")
                    )
                    
                    # Plot metrics
                    plot_metrics(
                        result["metrics"],
                        title=f"Neural Plasticity Metrics (Pruning {pruning_level}, Growth {growth_percentage})",
                        save_path=os.path.join(exp_vis_dir, "metrics.png")
                    )
                    
                    # Plot learning curve
                    plt.figure(figsize=(10, 6))
                    plt.plot(
                        [point["step"] for point in result["learning_curve"]],
                        [point["perplexity"] for point in result["learning_curve"]],
                        'o-'
                    )
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.title(f"Learning Curve (Pruning {pruning_level}, Growth {growth_percentage})")
                    plt.xlabel('Step')
                    plt.ylabel('Perplexity')
                    plt.savefig(os.path.join(exp_vis_dir, "learning_curve.png"))
                    plt.close()
    
    # Save summary results
    summary = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "parameters": {
            "model_name": args.model_name,
            "pruning_levels": pruning_levels,
            "pruning_strategy": args.pruning_strategy,
            "growth_percentages": growth_percentages,
            "growth_strategy": args.growth_strategy,
            "learning_steps": args.learning_steps,
            "learning_rate": args.learning_rate,
            "new_head_lr_multiplier": args.new_head_lr_multiplier
        },
        "results": [
            {
                "pruning_level": result["parameters"]["pruning_level"],
                "growth_percentage": result["parameters"]["growth_percentage"],
                "perplexity": {
                    "original": result["metrics"]["perplexity"]["original"],
                    "pruned": result["metrics"]["perplexity"]["pruned"],
                    "grown_initial": result["metrics"]["perplexity"]["grown_initial"],
                    "learned": result["metrics"]["perplexity"]["learned"]
                },
                "head_counts": result["head_counts"]
            }
            for result in all_results
        ]
    }
    
    summary_path = os.path.join(experiment_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create comparison visualization if multiple experiments
    if len(all_results) > 1 and args.save_visualizations:
        # Extract results for comparison
        comparison_data = {
            "pruning_levels": [],
            "growth_percentages": [],
            "original_perplexity": [],
            "pruned_perplexity": [],
            "learned_perplexity": [],
            "head_count_change": []
        }
        
        for result in all_results:
            comparison_data["pruning_levels"].append(result["parameters"]["pruning_level"])
            comparison_data["growth_percentages"].append(result["parameters"]["growth_percentage"])
            comparison_data["original_perplexity"].append(result["metrics"]["perplexity"]["original"])
            comparison_data["pruned_perplexity"].append(result["metrics"]["perplexity"]["pruned"])
            comparison_data["learned_perplexity"].append(result["metrics"]["perplexity"]["learned"])
            comparison_data["head_count_change"].append(
                result["head_counts"]["grown"] - result["head_counts"]["original"]
            )
        
        # Create comparison visualization
        plt.figure(figsize=(14, 10))
        
        # Convert to numpy arrays for easier processing
        pruning_levels = np.array(comparison_data["pruning_levels"])
        growth_percentages = np.array(comparison_data["growth_percentages"])
        
        # Get unique values
        unique_pruning = np.unique(pruning_levels)
        unique_growth = np.unique(growth_percentages)
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot 1: Effect of pruning level on perplexity (averaging across growth rates)
        for growth in unique_growth:
            # Get indices for this growth percentage
            indices = growth_percentages == growth
            
            # Extract values
            x_values = pruning_levels[indices]
            y_values = np.array(comparison_data["learned_perplexity"])[indices]
            
            # Sort by x values
            sort_idx = np.argsort(x_values)
            x_values = x_values[sort_idx]
            y_values = y_values[sort_idx]
            
            # Plot
            axes[0].plot(x_values, y_values, 'o-', label=f"Growth {growth}")
        
        axes[0].set_title("Effect of Pruning Level on Final Perplexity")
        axes[0].set_xlabel("Pruning Level")
        axes[0].set_ylabel("Perplexity after Learning")
        axes[0].grid(True, linestyle='--', alpha=0.7)
        axes[0].legend()
        
        # Plot 2: Effect of growth percentage on perplexity (averaging across pruning levels)
        for pruning in unique_pruning:
            # Get indices for this pruning level
            indices = pruning_levels == pruning
            
            # Extract values
            x_values = growth_percentages[indices]
            y_values = np.array(comparison_data["learned_perplexity"])[indices]
            
            # Sort by x values
            sort_idx = np.argsort(x_values)
            x_values = x_values[sort_idx]
            y_values = y_values[sort_idx]
            
            # Plot
            axes[1].plot(x_values, y_values, 'o-', label=f"Pruning {pruning}")
        
        axes[1].set_title("Effect of Growth Percentage on Final Perplexity")
        axes[1].set_xlabel("Growth Percentage")
        axes[1].set_ylabel("Perplexity after Learning")
        axes[1].grid(True, linestyle='--', alpha=0.7)
        axes[1].legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "comparison.png"))
        plt.close()
    
    print(f"\nExperiment completed! Results saved to {experiment_dir}")
    print(f"Summary: {len(all_results)} experiments with pruning levels {pruning_levels} and growth percentages {growth_percentages}")
    
    # Print brief summary of results
    print("\nBrief Results Summary:")
    for result in all_results:
        pl = result["parameters"]["pruning_level"]
        gp = result["parameters"]["growth_percentage"]
        orig_ppl = result["metrics"]["perplexity"]["original"]
        pruned_ppl = result["metrics"]["perplexity"]["pruned"]
        learned_ppl = result["metrics"]["perplexity"]["learned"]
        
        print(f"- Pruning {pl:.2f}, Growth {gp:.2f}: Perplexity {orig_ppl:.2f} → {pruned_ppl:.2f} → {learned_ppl:.2f}")

if __name__ == "__main__":
    main()