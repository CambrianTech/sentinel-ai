#!/usr/bin/env python
"""
Model Optimization through Neural Plasticity

This script runs multiple neural plasticity cycles to improve a model's performance 
while reducing its parameter count. It systematically:

1. Prunes less important attention heads
2. Measures performance impact
3. Strategically regrows critical heads
4. Uses differential learning rates for faster adaptation
5. Compares performance before and after optimization

Usage:
    python scripts/optimize_model_plasticity.py --model_name distilgpt2 --dataset wikitext
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
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
from sentinel_data.dataset_loader import load_dataset

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Model Optimization through Neural Plasticity")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                      help="Model name (default: distilgpt2)")
    parser.add_argument("--dataset", type=str, default="tiny_shakespeare",
                      help="Dataset name (default: tiny_shakespeare)")
    parser.add_argument("--output_dir", type=str, default="./output/plasticity_optimization",
                      help="Directory to save results (default: ./output/plasticity_optimization)")
    
    # Target-based optimization parameters
    parser.add_argument("--optimization_targets", type=str, default="perf,efficiency,size",
                      help="Optimization targets, comma-separated (options: perf,efficiency,size) (default: perf,efficiency,size)")
    parser.add_argument("--target_improvement", type=float, default=0.05,
                      help="Target relative improvement in perplexity (default: 0.05 = 5%% better)")
    parser.add_argument("--target_head_reduction", type=float, default=0.3,
                      help="Target head count reduction (default: 0.3 = 30%% fewer heads)")
    parser.add_argument("--max_cycles", type=int, default=5,
                      help="Maximum number of plasticity cycles to run (default: 5)")
    parser.add_argument("--perplexity_threshold", type=float, default=1.1,
                      help="Maximum allowed perplexity ratio compared to baseline (default: 1.1 = 10%% worse allowed)")
    parser.add_argument("--target_efficiency_improvement", type=float, default=0.15,
                      help="Target efficiency improvement (perplexity/head) (default: 0.15 = 15%% better)")
    parser.add_argument("--dynamic_targets", action="store_true",
                      help="Dynamically adjust targets based on progress")
    
    # Training parameters
    parser.add_argument("--initial_training_steps", type=int, default=100,
                      help="Initial training steps before optimization (default: 100)")
    parser.add_argument("--cycle_training_steps", type=int, default=200,
                      help="Base training steps per cycle, will scale with cycle number (default: 200)")
    parser.add_argument("--eval_samples", type=int, default=10,
                      help="Number of text samples for evaluation (default: 10)")
    parser.add_argument("--adaptive_training", action="store_true",
                      help="Adaptively adjust training steps based on performance")
    parser.add_argument("--memory_based_training", action="store_true",
                      help="Adjust training steps based on memory of past improvements")
    parser.add_argument("--early_stopping_patience", type=int, default=2,
                      help="Number of cycles without improvement before early stopping (default: 2)")
    
    # Pruning and growth parameters
    parser.add_argument("--initial_pruning_level", type=float, default=0.2,
                      help="Initial pruning level (default: 0.2 = 20%% of heads)")
    parser.add_argument("--aggressive_pruning", action="store_true",
                      help="Progressively increase pruning levels each cycle")
    parser.add_argument("--pruning_strategy", type=str, default="entropy",
                      choices=["random", "magnitude", "entropy", "auto"],
                      help="Strategy for pruning (default: entropy, 'auto' tries all)")
    parser.add_argument("--adaptive_pruning", action="store_true",
                      help="Adaptively adjust pruning based on previous cycle results")
    parser.add_argument("--growth_ratio", type=float, default=0.5,
                      help="Ratio of pruned heads to grow back (default: 0.5)")
    parser.add_argument("--growth_strategy", type=str, default="balanced",
                      choices=["gradient_sensitivity", "entropy_gap", "balanced", "random", "auto"],
                      help="Strategy for growth (default: balanced, 'auto' tries all)")
    parser.add_argument("--new_head_lr_multiplier", type=float, default=5.0,
                      help="Learning rate multiplier for new heads (default: 5.0)")
    parser.add_argument("--adaptive_growth", action="store_true",
                      help="Adaptively adjust growth ratio based on previous results")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size for training (default: 16)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                      help="Base learning rate (default: 5e-5)")
    parser.add_argument("--sequence_length", type=int, default=128,
                      help="Sequence length for training (default: 128)")
    parser.add_argument("--eval_every", type=int, default=50,
                      help="Evaluate every N steps (default: 50)")
    
    # Output and device parameters
    parser.add_argument("--save_model", action="store_true",
                      help="Save model checkpoints at each stage")
    parser.add_argument("--save_visualizations", action="store_true",
                      help="Save visualizations as PNG files")
    parser.add_argument("--track_head_importance", action="store_true",
                      help="Track importance of individual heads across cycles")
    parser.add_argument("--device", type=str, default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
                      help="Device to use (default: cuda if available, else cpu)")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose output")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed (default: 42)")
    parser.add_argument("--resume_from", type=str, default="",
                      help="Resume from a previous experiment directory")
    
    return parser.parse_args()

def optimize_model(args):
    """Run model optimization through neural plasticity with adaptive targets"""
    # Allow resuming from previous experiments
    if args.resume_from:
        experiment_dir = args.resume_from
        if not os.path.exists(experiment_dir):
            print(f"Error: Resume directory {experiment_dir} does not exist")
            return False
            
        # Load previous configuration
        config_path = os.path.join(experiment_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                prev_config = json.load(f)
                print(f"Loaded previous configuration from {config_path}")
                # Merge with current args, prioritizing command line args
                for k, v in prev_config.items():
                    if not hasattr(args, k) or getattr(args, k) is None:
                        setattr(args, k, v)
        else:
            print(f"Warning: No previous configuration found at {config_path}")
    else:
        # Set up new experiment directories
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_short = args.model_name.split("/")[-1]
        experiment_name = f"optimize_{model_short}_{timestamp}"
        experiment_dir = os.path.join(args.output_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    vis_dir = os.path.join(experiment_dir, "visualizations")
    metrics_dir = os.path.join(experiment_dir, "metrics")
    analysis_dir = os.path.join(experiment_dir, "analysis")
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set up metrics logging
    metrics_file = os.path.join(metrics_dir, "metrics.jsonl")
    metrics_logger = MetricsLogger(metrics_file)
    
    # Parse optimization targets
    targets = args.optimization_targets.split(",")
    print(f"Optimization targets: {', '.join(targets)}")
    
    # Set random seed for reproducibility
    if args.seed:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        try:
            import torch
            torch.manual_seed(args.seed)
            if args.device == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
        except ImportError:
            pass  # No torch

    # Load model
    print(f"Loading model {args.model_name}...")
    pruning_module = PruningModule(args.model_name)
    success = pruning_module.load_model()
    
    if not success:
        print(f"Failed to load model {args.model_name}")
        return False
    
    # Load dataset
    print(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(args.dataset, pruning_module.tokenizer, max_length=args.sequence_length)
    
    # Total heads in model
    total_heads = pruning_module.num_layers * pruning_module.num_heads
    
    # Initialize memory for tracking pruning and growth success
    pruning_strategy_results = {
        "entropy": {"success": 0, "trials": 0, "perplexity_change": []},
        "magnitude": {"success": 0, "trials": 0, "perplexity_change": []},
        "random": {"success": 0, "trials": 0, "perplexity_change": []}
    }
    
    growth_strategy_results = {
        "balanced": {"success": 0, "trials": 0, "perplexity_change": []},
        "gradient_sensitivity": {"success": 0, "trials": 0, "perplexity_change": []},
        "entropy_gap": {"success": 0, "trials": 0, "perplexity_change": []},
        "random": {"success": 0, "trials": 0, "perplexity_change": []}
    }
    
    # Initialize head importance tracking if requested
    head_importance_history = {}
    if args.track_head_importance:
        for layer_idx in range(pruning_module.num_layers):
            for head_idx in range(pruning_module.num_heads):
                head_importance_history[(layer_idx, head_idx)] = {
                    "importance_scores": [],
                    "pruned_count": 0,
                    "grown_count": 0,
                    "cycles_active": 0
                }
    
    # Check for existing baseline model
    baseline_model_path = os.path.join(checkpoints_dir, "model_baseline.pth")
    if args.resume_from and os.path.exists(baseline_model_path):
        print(f"Loading baseline model from {baseline_model_path}...")
        with open(baseline_model_path, 'rb') as f:
            import pickle
            baseline_params = pickle.load(f)
    else:
        baseline_params = pruning_module.params  # Access the params from the pruning module itself
    
    # Determine active heads
    baseline_active_heads = determine_active_heads(pruning_module, baseline_params)
    original_head_count = len(baseline_active_heads)
    
    print(f"Initial active heads: {original_head_count} of {total_heads} ({original_head_count/total_heads:.1%})")
    
    # Run initial evaluation
    print("Evaluating baseline model...")
    baseline_eval = evaluate_model(
        pruning_module=pruning_module,
        params=baseline_params,
        dataset=dataset,
        num_samples=args.eval_samples
    )
    
    baseline_perplexity = baseline_eval["average_perplexity"]
    print(f"Baseline model - Perplexity: {baseline_perplexity:.2f}, Heads: {original_head_count}")
    
    # Log baseline metrics
    metrics_logger.log({
        "phase": "baseline",
        "total_heads": total_heads,
        "active_heads": original_head_count,
        "perplexity": baseline_perplexity
    })
    
    # Create visualization of initial head distribution
    if args.save_visualizations:
        initial_vis_path = os.path.join(vis_dir, "initial_head_distribution.png")
        plot_head_distribution(
            pruning_module, 
            baseline_active_heads, 
            title="Initial Head Distribution",
            save_path=initial_vis_path
        )
    
    # Track head importance if requested
    if args.track_head_importance:
        print("Analyzing initial head importance...")
        # Get importance from different strategies
        for strategy_name in ["entropy", "magnitude"]:
            pruning_strategy = get_pruning_strategy(strategy_name, pruning_module)
            head_importance = pruning_strategy.get_head_importance(baseline_params)
            
            # Record initial importance scores
            for layer_idx, head_idx, score in head_importance:
                if (layer_idx, head_idx) in head_importance_history:
                    head_importance_history[(layer_idx, head_idx)]["importance_scores"].append({
                        "cycle": 0,
                        "strategy": strategy_name,
                        "score": float(score) if not isinstance(score, (int, float)) else score
                    })
                    
                    # Mark initially active heads
                    if (layer_idx, head_idx) in baseline_active_heads:
                        head_importance_history[(layer_idx, head_idx)]["cycles_active"] += 1
        
        # Save head importance data - convert tuple keys to strings for JSON
        head_importance_path = os.path.join(analysis_dir, "head_importance_baseline.json")
        # Convert tuple keys to strings for JSON serialization
        serializable_importance = {}
        for (layer_idx, head_idx), data in head_importance_history.items():
            serializable_importance[f"{layer_idx}_{head_idx}"] = data
            
        with open(head_importance_path, 'w') as f:
            json.dump(serializable_importance, f, indent=2)
    
    # Initial training to establish baseline
    if args.initial_training_steps > 0 and not args.resume_from:
        print(f"\n=== Initial Training Phase ===")
        print(f"Training for {args.initial_training_steps} steps to establish baseline...")
        
        trainer = FineTuner(
            pruning_module=pruning_module,
            dataset=dataset,
            learning_rate=args.learning_rate
        )
        
        # Train for specified steps
        for step in range(args.initial_training_steps):
            train_loss = trainer.train_step()
            
            # Log metrics periodically
            if step % args.eval_every == 0 or step == args.initial_training_steps - 1:
                # Evaluate model
                current_params = trainer.get_params()
                eval_results = evaluate_model(
                    pruning_module=pruning_module,
                    params=current_params,
                    dataset=dataset,
                    num_samples=args.eval_samples
                )
                
                # Log evaluation metrics
                metrics_logger.log({
                    "phase": "initial_training",
                    "step": step,
                    "train_loss": float(train_loss),
                    "perplexity": eval_results["average_perplexity"]
                })
                
                # Print progress
                print(f"Training step {step}/{args.initial_training_steps}: " +
                      f"Loss = {train_loss:.4f}, " +
                      f"Perplexity = {eval_results['average_perplexity']:.4f}")
        
        # Get updated model parameters
        baseline_params = trainer.get_params()
        
        # Re-evaluate after training
        baseline_eval = evaluate_model(
            pruning_module=pruning_module,
            params=baseline_params,
            dataset=dataset,
            num_samples=args.eval_samples
        )
        
        baseline_perplexity = baseline_eval["average_perplexity"]
        print(f"After initial training - Perplexity: {baseline_perplexity:.2f}")
        
        # Verify active heads didn't change during training
        baseline_active_heads = determine_active_heads(pruning_module, baseline_params)
        if len(baseline_active_heads) != original_head_count:
            print(f"Warning: Active head count changed during training: {original_head_count} -> {len(baseline_active_heads)}")
            original_head_count = len(baseline_active_heads)
        
        # Save the baseline model
        if args.save_model:
            with open(baseline_model_path, 'wb') as f:
                import pickle
                pickle.dump(baseline_params, f)
    
    # Initialize optimization variables
    best_params = baseline_params
    best_perplexity = baseline_perplexity
    best_head_count = original_head_count
    best_efficiency = baseline_perplexity / original_head_count
    
    # Define target metrics based on baseline
    target_perplexity = baseline_perplexity * (1.0 - args.target_improvement)
    target_head_count = max(1, int(original_head_count * (1.0 - args.target_head_reduction)))
    target_efficiency = best_efficiency * (1.0 - args.target_efficiency_improvement)
    max_acceptable_perplexity = baseline_perplexity * args.perplexity_threshold
    
    print(f"\n=== Optimization Targets ===")
    print(f"Performance target: {target_perplexity:.2f} perplexity (baseline: {baseline_perplexity:.2f})")
    print(f"Size target: {target_head_count} heads (baseline: {original_head_count})")
    print(f"Efficiency target: {target_efficiency:.5f} perplexity/head (baseline: {best_efficiency:.5f})")
    print(f"Maximum acceptable perplexity: {max_acceptable_perplexity:.2f}")
    
    # Set up historical metrics for dynamic target adjustment
    perplexity_history = [baseline_perplexity]
    head_count_history = [original_head_count]
    efficiency_history = [best_efficiency]
    
    # Track metrics for each cycle
    cycle_metrics = []
    
    # Current model parameters
    current_params = baseline_params
    
    # Tracking variables for progress and early stopping
    consecutive_failures = 0  # Count of cycles without improvement
    target_reached = False
    best_strategy_pair = None
    
    # Function to select best strategy based on historical performance
    def select_best_strategy(strategy_type, cycle):
        if strategy_type == "pruning":
            results = pruning_strategy_results
            default = "entropy"
        else:  # growth
            results = growth_strategy_results
            default = "balanced"
            
        # If no history or auto not selected, return configured strategy
        if cycle < 2 or (strategy_type == "pruning" and args.pruning_strategy != "auto") or \
           (strategy_type == "growth" and args.growth_strategy != "auto"):
            if strategy_type == "pruning":
                return args.pruning_strategy if args.pruning_strategy != "auto" else default
            else:
                return args.growth_strategy if args.growth_strategy != "auto" else default
                
        # Find strategy with best success rate
        best_strategy = None
        best_score = float('-inf')
        
        for strategy, stats in results.items():
            if stats["trials"] > 0:
                # Score based on success rate and average perplexity improvement
                success_rate = stats["success"] / stats["trials"]
                avg_perplexity_change = 0
                if stats["perplexity_change"]:
                    avg_perplexity_change = sum(stats["perplexity_change"]) / len(stats["perplexity_change"])
                    
                # Score formula: balance success rate with perplexity improvement
                score = success_rate - avg_perplexity_change * 0.5  # negative perplexity change is good
                
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
        
        # Return best strategy or default if no data
        return best_strategy if best_strategy else default
                
    # Run optimization cycles until targets are met or max cycles reached
    for cycle in range(args.max_cycles):
        print(f"\n=== Neural Plasticity Cycle {cycle+1}/{args.max_cycles} ===")
        cycle_dir = os.path.join(vis_dir, f"cycle_{cycle+1}")
        os.makedirs(cycle_dir, exist_ok=True)
        
        # Get current active heads
        active_heads_before = determine_active_heads(pruning_module, current_params)
        current_head_count = len(active_heads_before)
        
        # Dynamically adjust targets if enabled
        if args.dynamic_targets and cycle > 0:
            # Calculate progress toward targets
            perplexity_progress = (baseline_perplexity - min(perplexity_history)) / (baseline_perplexity - target_perplexity)
            head_count_progress = (original_head_count - min(head_count_history)) / (original_head_count - target_head_count)
            
            # If we're making more progress on one target, shift focus to the other
            if "perf" in targets and "size" in targets:
                if perplexity_progress > head_count_progress * 1.5:
                    # We're making good progress on perplexity, focus more on head reduction
                    target_head_count = max(1, int(target_head_count * 0.9))
                    print(f"Adjusting target head count to {target_head_count} (increased focus on size)")
                elif head_count_progress > perplexity_progress * 1.5:
                    # We're making good progress on head count, focus more on perplexity
                    target_perplexity = target_perplexity * 0.95
                    print(f"Adjusting target perplexity to {target_perplexity:.2f} (increased focus on performance)")
        
        # Determine pruning level based on targets and current progress
        if args.adaptive_pruning and cycle > 0:
            # Base pruning on recent success and proximity to target
            if consecutive_failures > 0:
                # Reduce pruning level if we're failing
                pruning_level = max(0.05, args.initial_pruning_level * (1.0 - 0.2 * consecutive_failures))
            elif current_head_count <= target_head_count * 1.1:
                # Close to target, be conservative
                pruning_level = 0.1
            else:
                # Increase pruning based on previous cycle success
                improvement_rate = 1.0
                if len(perplexity_history) >= 2:
                    # Calculate relative improvement from previous cycle
                    prev_perplexity = perplexity_history[-2]
                    curr_perplexity = perplexity_history[-1]
                    if prev_perplexity > 0:
                        improvement_rate = max(0.5, min(1.5, prev_perplexity / curr_perplexity))
                
                # Scale pruning level by improvement rate
                pruning_level = min(0.5, args.initial_pruning_level * improvement_rate)
        elif args.aggressive_pruning:
            # Increase pruning level each cycle
            pruning_level = min(0.7, args.initial_pruning_level * (1.0 + 0.2 * cycle))
        else:
            pruning_level = args.initial_pruning_level
        
        # Adjust pruning level based on how far we are from target head count
        if "size" in targets and current_head_count > target_head_count:
            # Calculate how much we need to prune to reach the target
            heads_to_remove = current_head_count - target_head_count
            target_pruning = heads_to_remove / current_head_count
            # Use the more aggressive of the two approaches, but cap at 0.5
            pruning_level = min(0.5, max(pruning_level, target_pruning))
            
        print(f"Pruning level for this cycle: {pruning_level:.2f}")
        
        # 1. Pruning Phase
        print("--- Pruning Phase ---")
        
        # Determine pruning strategy (either from args or auto-selection)
        if args.pruning_strategy == "auto":
            strategy_name = select_best_strategy("pruning", cycle)
            print(f"Auto-selected pruning strategy: {strategy_name}")
        else:
            strategy_name = args.pruning_strategy
        
        pruning_strategy = get_pruning_strategy(strategy_name, pruning_module)
        
        # Get head importance scores
        head_importance = pruning_strategy.get_head_importance(current_params)
        
        # Track head importance if requested
        if args.track_head_importance:
            for layer_idx, head_idx, score in head_importance:
                if (layer_idx, head_idx) in head_importance_history:
                    head_importance_history[(layer_idx, head_idx)]["importance_scores"].append({
                        "cycle": cycle + 1,
                        "strategy": strategy_name,
                        "score": float(score) if not isinstance(score, (int, float)) else score
                    })
        
        # Sort by importance (ascending, so least important first)
        head_importance.sort(key=lambda x: x[2])
        
        # Calculate heads to prune
        heads_to_prune = max(1, int(current_head_count * pruning_level))
        print(f"Pruning {heads_to_prune} of {current_head_count} heads...")
        
        # Select heads to prune (least important first)
        heads_to_prune_list = [(layer_idx, head_idx) for layer_idx, head_idx, _ in head_importance[:heads_to_prune]]
        
        # Prune the selected heads
        pruned_params = current_params.copy()
        for layer_idx, head_idx in heads_to_prune_list:
            pruned_params = pruning_module.prune_head(pruned_params, layer_idx, head_idx)
            
            # Track pruned heads
            if args.track_head_importance and (layer_idx, head_idx) in head_importance_history:
                head_importance_history[(layer_idx, head_idx)]["pruned_count"] += 1
        
        # Determine active heads after pruning
        active_heads_after_pruning = determine_active_heads(pruning_module, pruned_params)
        heads_actually_pruned = current_head_count - len(active_heads_after_pruning)
        
        print(f"Pruned {heads_actually_pruned} heads, " +
              f"{len(active_heads_after_pruning)} active heads remaining " +
              f"({len(active_heads_after_pruning)/total_heads:.1%} of total)")
        
        # Save head distribution visualization
        if args.save_visualizations:
            pruned_vis_path = os.path.join(cycle_dir, "pruned_head_distribution.png")
            plot_head_distribution(
                pruning_module, 
                active_heads_after_pruning, 
                title=f"Pruned Head Distribution (Cycle {cycle+1})",
                save_path=pruned_vis_path
            )
        
        # 2. Measurement Phase
        print("--- Measurement Phase ---")
        
        # Evaluate pruned model
        pruned_eval = evaluate_model(
            pruning_module=pruning_module,
            params=pruned_params,
            dataset=dataset,
            num_samples=args.eval_samples
        )
        
        pruned_perplexity = pruned_eval["average_perplexity"]
        print(f"Pruned model - Perplexity: {pruned_perplexity:.2f}")
        
        # Check if pruning was too aggressive
        if pruned_perplexity > max_acceptable_perplexity:
            print(f"WARNING: Pruning resulted in perplexity above threshold " + 
                  f"({pruned_perplexity:.2f} > {max_acceptable_perplexity:.2f})")
            print("Consider reducing pruning level for better results")
        
        # Log metrics
        metrics_logger.log({
            "phase": "pruning",
            "cycle": cycle + 1,
            "pruning_level": pruning_level,
            "pruning_strategy": strategy_name,
            "active_heads_before": current_head_count,
            "active_heads_after": len(active_heads_after_pruning),
            "pruned_perplexity": pruned_perplexity
        })
        
        # 3. Growth Phase
        print("--- Growth Phase ---")
        
        # Determine growth strategy
        if args.growth_strategy == "auto":
            # Use strategy selection function
            growth_strategy_name = select_best_strategy("growth", cycle)
            print(f"Auto-selected growth strategy: {growth_strategy_name}")
        else:
            growth_strategy_name = args.growth_strategy
        
        # Calculate growth percentage based on history and targets
        if args.adaptive_growth and cycle > 0:
            # Calculate distance to targets
            perplexity_gap = (perplexity_history[-1] - target_perplexity) / baseline_perplexity
            head_count_gap = (target_head_count - len(active_heads_after_pruning)) / original_head_count
            
            if perplexity_gap > 0.1 and "perf" in targets:
                # Far from performance target, grow more heads
                growth_ratio = min(0.8, args.growth_ratio * 1.2)
            elif head_count_gap < 0 and "size" in targets:
                # Already below head count target, be conservative with growth
                growth_ratio = args.growth_ratio * 0.5
            elif consecutive_failures > 0:
                # If we're failing to improve, try more growth to see if it helps
                growth_ratio = min(0.9, args.growth_ratio * (1.0 + 0.2 * consecutive_failures))
            else:
                growth_ratio = args.growth_ratio
                
            # If efficiency is good but perplexity is bad, grow more
            if len(efficiency_history) >= 2 and len(perplexity_history) >= 2:
                if efficiency_history[-1] < efficiency_history[-2] and perplexity_history[-1] > perplexity_history[-2]:
                    growth_ratio = min(0.9, growth_ratio * 1.3)
                    print(f"Increasing growth due to efficiency improvement but perplexity degradation")
        elif "efficiency" in targets:
            # Grow back fewer heads if we're targeting efficiency
            growth_ratio = min(args.growth_ratio, 0.4)
        else:
            growth_ratio = args.growth_ratio
            
        growth_percentage = growth_ratio * pruning_level
        
        # Adjust growth based on number of remaining active heads
        min_active_head_percent = 0.3  # Don't go below 30% of total heads
        current_head_percent = len(active_heads_after_pruning) / total_heads
        
        if current_head_percent < min_active_head_percent:
            # If too few heads remain, increase growth
            growth_percentage = max(growth_percentage, 0.2)
            print(f"Increasing growth due to low head count ({current_head_percent:.1%} < {min_active_head_percent:.1%})")
                
        print(f"Growth ratio: {growth_ratio:.2f}, Growth percentage: {growth_percentage:.2f}")
            
        # Save current parameters in case growing fails
        params_before_growth = pruned_params.copy()
        
        # Grow attention heads
        try:
            growth_params, added_count, added_heads, warmup_schedule = grow_attention_heads_gradually(
                pruning_module,
                params=pruned_params,
                active_heads=active_heads_after_pruning,
                growth_percentage=growth_percentage,
                strategy=growth_strategy_name,
                initial_scale=0.01,
                warmup_steps=min(50, args.cycle_training_steps // 4)
            )
            
            # Track grown heads
            if args.track_head_importance:
                for layer_idx, head_idx in added_heads:
                    if (layer_idx, head_idx) in head_importance_history:
                        head_importance_history[(layer_idx, head_idx)]["grown_count"] += 1
            
            growth_success = True
        except Exception as e:
            print(f"Error during growth phase: {e}")
            growth_params = params_before_growth
            added_count = 0
            added_heads = []
            warmup_schedule = lambda step: 1.0
            growth_success = False
        
        # Determine active heads after growth
        active_heads_after_growth = determine_active_heads(pruning_module, growth_params)
        
        if growth_success:
            print(f"Added {added_count} heads, now have {len(active_heads_after_growth)} active heads " +
                f"({len(active_heads_after_growth)/total_heads:.1%} of total)")
        else:
            print(f"Growth failed, continuing with pruned model ({len(active_heads_after_growth)} heads)")
            
        # Update active head tracking for importance history
        if args.track_head_importance:
            for layer_idx, head_idx in active_heads_after_growth:
                if (layer_idx, head_idx) in head_importance_history:
                    head_importance_history[(layer_idx, head_idx)]["cycles_active"] += 1
        
        # Save head distribution visualization
        if args.save_visualizations:
            grown_vis_path = os.path.join(cycle_dir, "grown_head_distribution.png")
            plot_head_distribution(
                pruning_module, 
                active_heads_after_growth, 
                title=f"Grown Head Distribution (Cycle {cycle+1})",
                save_path=grown_vis_path
            )
        
        # Evaluate model after growth (before learning)
        grown_eval = evaluate_model(
            pruning_module=pruning_module,
            params=growth_params,
            dataset=dataset,
            num_samples=args.eval_samples
        )
        
        grown_perplexity = grown_eval["average_perplexity"]
        print(f"After growth (before learning) - Perplexity: {grown_perplexity:.2f}")
        
        # Log metrics
        metrics_logger.log({
            "phase": "growth",
            "cycle": cycle + 1,
            "growth_ratio": growth_ratio,
            "growth_strategy": growth_strategy_name,
            "growth_percentage": growth_percentage,
            "heads_added": added_count,
            "active_heads_after_growth": len(active_heads_after_growth),
            "grown_perplexity": grown_perplexity
        })
        
        # 4. Learning Phase
        print("--- Learning Phase ---")
        
        # Determine number of learning steps based on various factors
        if args.memory_based_training and cycle > 0:
            # Use historical improvement rates to determine training needs
            
            # Look at previous cycles to estimate training needs
            avg_improvements = []
            
            # Collect improvement rates from previous cycles
            for i in range(min(3, len(cycle_metrics))):
                if i < len(cycle_metrics):
                    # Calculate improvement from pruned to final perplexity
                    if cycle_metrics[i]["pruned_perplexity"] > 0 and cycle_metrics[i]["final_perplexity"] > 0:
                        improvement_rate = (cycle_metrics[i]["pruned_perplexity"] - cycle_metrics[i]["final_perplexity"]) / cycle_metrics[i]["pruned_perplexity"]
                        avg_improvements.append(improvement_rate)
            
            # Default multiplier
            training_multiplier = 1.0
            
            if avg_improvements:
                # Calculate average improvement rate
                avg_improvement_rate = sum(avg_improvements) / len(avg_improvements)
                
                # Estimate needed training based on degradation and historical improvement rates
                if pruned_perplexity > baseline_perplexity:
                    # Current degradation from pruning
                    current_degradation = (pruned_perplexity - baseline_perplexity) / baseline_perplexity
                    
                    # Estimate training needed based on historical improvement
                    if avg_improvement_rate > 0:
                        # Scale by how much we need to recover vs. our historical ability to recover
                        training_multiplier = min(3.0, max(1.0, current_degradation / (avg_improvement_rate + 1e-6)))
                        
                        print(f"Memory-based training adjustment: historical improvement rate = {avg_improvement_rate:.2%}, " +
                              f"current degradation = {current_degradation:.2%}, multiplier = {training_multiplier:.2f}")
                    else:
                        # If we have no positive improvement history, be more aggressive
                        training_multiplier = min(3.0, 1.0 + current_degradation * 2)
            
            # Factor in current perplexity degradation
            perplexity_degradation = max(0, (grown_perplexity - baseline_perplexity) / baseline_perplexity)
            
            # Add more training if needed based on head addition
            if added_count > 0:
                # Add more training proportional to number of new heads
                head_ratio = added_count / len(active_heads_after_growth)
                training_multiplier += head_ratio  # More training for more new heads
            
            # Final calculation
            learning_steps = int(args.cycle_training_steps * training_multiplier)
            print(f"Memory-based training: {learning_steps} steps (base: {args.cycle_training_steps}, multiplier: {training_multiplier:.2f})")
            
        elif args.adaptive_training:
            # Scale based on perplexity degradation - more training if performance is worse
            perplexity_degradation = max(0, (grown_perplexity - baseline_perplexity) / baseline_perplexity)
            training_multiplier = 1.0 + 2.0 * perplexity_degradation
            learning_steps = int(args.cycle_training_steps * training_multiplier)
            
            # Additional training for later cycles
            cycle_factor = min(2.0, 1.0 + 0.25 * cycle)  # Up to 2x more training in later cycles
            learning_steps = int(learning_steps * cycle_factor)
            
            print(f"Adaptive training: {learning_steps} steps (base: {args.cycle_training_steps}, multiplier: {training_multiplier:.2f}, cycle factor: {cycle_factor:.2f})")
        else:
            learning_steps = args.cycle_training_steps
        
        # Create head learning rate manager for differential learning
        head_lr_manager = HeadLRManager(
            base_lr=args.learning_rate,
            new_head_multiplier=args.new_head_lr_multiplier,
            new_heads=added_heads
        )
        
        # Adjust learning rate based on cycle
        cycle_lr = args.learning_rate
        if cycle > 1:
            # Gradually reduce learning rate in later cycles
            cycle_lr = args.learning_rate * (1.0 - min(0.5, 0.1 * (cycle - 1)))
            print(f"Adjusted learning rate for cycle {cycle+1}: {cycle_lr:.6f} (base: {args.learning_rate:.6f})")
        
        # Initialize trainer
        trainer = FineTuner(
            pruning_module=pruning_module,
            dataset=dataset,
            learning_rate=cycle_lr,
            head_lr_manager=head_lr_manager
        )
        
        # Set initial parameters
        trainer.set_params(growth_params)
        
        # Track best perplexity during training to detect plateaus
        best_training_perplexity = grown_perplexity
        best_training_params = None
        steps_without_improvement = 0
        
        # Train for specified steps
        for step in range(learning_steps):
            train_loss = trainer.train_step()
            
            # Log metrics periodically
            if step % args.eval_every == 0 or step == learning_steps - 1:
                # Evaluate model
                current_params = trainer.get_params()
                eval_results = evaluate_model(
                    pruning_module=pruning_module,
                    params=current_params,
                    dataset=dataset,
                    num_samples=args.eval_samples
                )
                
                current_perplexity = eval_results["average_perplexity"]
                
                # Check for improvements
                if current_perplexity < best_training_perplexity:
                    improvement = best_training_perplexity - current_perplexity
                    relative_improvement = improvement / best_training_perplexity
                    best_training_perplexity = current_perplexity
                    best_training_params = current_params
                    steps_without_improvement = 0
                    
                    print(f"Learning step {step}/{learning_steps}: " +
                          f"Loss = {train_loss:.4f}, " +
                          f"Perplexity = {current_perplexity:.4f} " +
                          f"(-{relative_improvement*100:.2f}%)")
                else:
                    steps_without_improvement += args.eval_every
                    print(f"Learning step {step}/{learning_steps}: " +
                          f"Loss = {train_loss:.4f}, " +
                          f"Perplexity = {current_perplexity:.4f} (no improvement for {steps_without_improvement} steps)")
                
                # Log evaluation metrics
                metrics_logger.log({
                    "phase": "learning",
                    "cycle": cycle + 1,
                    "step": step,
                    "train_loss": float(train_loss),
                    "perplexity": current_perplexity
                })
                
                # Early stopping if no improvement for a long time
                patience = max(2, min(10, learning_steps // (args.eval_every * 4)))  # Adjust patience based on learning steps
                if (args.adaptive_training or args.memory_based_training) and steps_without_improvement >= patience * args.eval_every:
                    print(f"Early stopping: No improvement for {steps_without_improvement} steps (patience: {patience} evaluations)")
                    break
        
        # Use best parameters found during training
        if best_training_params is not None:
            learned_params = best_training_params
            print(f"Using best model found during training (perplexity: {best_training_perplexity:.4f})")
        else:
            # Get updated model parameters from last training step
            learned_params = trainer.get_params()
        
        # Evaluate final model after learning
        final_eval = evaluate_model(
            pruning_module=pruning_module,
            params=learned_params,
            dataset=dataset,
            num_samples=args.eval_samples
        )
        
        final_perplexity = final_eval["average_perplexity"]
        active_heads_final = determine_active_heads(pruning_module, learned_params)
        final_head_count = len(active_heads_final)
        
        print(f"After learning - Perplexity: {final_perplexity:.2f}, " +
              f"Heads: {final_head_count} ({final_head_count/total_heads:.1%} of total)")
        
        # Calculate perplexity change from baseline
        perplexity_change = final_perplexity - baseline_perplexity
        perplexity_change_pct = (perplexity_change / baseline_perplexity) * 100
        
        # Calculate recovery from pruning
        if pruned_perplexity != baseline_perplexity:
            recovery_pct = ((pruned_perplexity - final_perplexity) / 
                          (pruned_perplexity - baseline_perplexity)) * 100
        else:
            recovery_pct = 0
        
        print(f"Perplexity change from baseline: {perplexity_change:+.2f} ({perplexity_change_pct:+.2f}%)")
        print(f"Recovery from pruning: {recovery_pct:.2f}%")
        
        # Calculate efficiency (perplexity / head count ratio, normalized to baseline)
        baseline_efficiency = baseline_perplexity / original_head_count
        current_efficiency = final_perplexity / final_head_count
        efficiency_improvement = (baseline_efficiency / current_efficiency - 1.0) * 100
        
        print(f"Efficiency change: {efficiency_improvement:+.2f}% (lower perplexity per head)")
        
        # Log final cycle metrics
        metrics_logger.log({
            "phase": "cycle_complete",
            "cycle": cycle + 1,
            "final_perplexity": final_perplexity,
            "active_heads": final_head_count,
            "active_head_percentage": final_head_count / total_heads * 100,
            "perplexity_change": perplexity_change,
            "perplexity_change_pct": perplexity_change_pct,
            "recovery_pct": recovery_pct,
            "efficiency_improvement": efficiency_improvement
        })
        
        # Store cycle metrics for visualization
        cycle_metrics.append({
            "cycle": cycle + 1,
            "pruning_level": pruning_level,
            "pruning_strategy": strategy_name,
            "growth_strategy": growth_strategy_name,
            "pruned_perplexity": pruned_perplexity,
            "grown_perplexity": grown_perplexity,
            "final_perplexity": final_perplexity,
            "active_heads": final_head_count,
            "head_percentage": final_head_count / total_heads * 100,
            "efficiency_improvement": efficiency_improvement
        })
        
        # Create cycle metrics visualization
        if args.save_visualizations:
            metrics_vis_path = os.path.join(cycle_dir, "metrics_comparison.png")
            
            # Build metrics dictionary
            metrics_data = {
                "perplexity": {
                    "Baseline": baseline_perplexity,
                    "Pruned": pruned_perplexity,
                    "Grown": grown_perplexity,
                    "Final": final_perplexity
                },
                "active_heads_percentage": {
                    "Baseline": original_head_count / total_heads * 100,
                    "Pruned": len(active_heads_after_pruning) / total_heads * 100,
                    "Grown": len(active_heads_after_growth) / total_heads * 100,
                    "Final": final_head_count / total_heads * 100
                }
            }
            
            # Create visualization
            plot_metrics_comparison(
                metrics_data,
                title=f"Neural Plasticity Cycle {cycle+1} Results",
                save_path=metrics_vis_path
            )
        
        # Save model if requested
        if args.save_model:
            model_path = os.path.join(checkpoints_dir, f"model_cycle{cycle+1}.pth")
            with open(model_path, 'wb') as f:
                import pickle
                pickle.dump(learned_params, f)
                
        # Run inference with test prompts to demonstrate actual outputs
        print("\n=== Generation Test ===")
        test_prompts = [
            "The future of artificial intelligence depends on",
            "The most interesting aspect of neural networks is",
            "Transformers have revolutionized natural language processing by"
        ]
        
        for i, prompt in enumerate(test_prompts):
            try:
                print(f"\nPrompt {i+1}: {prompt}")
                generation = pruning_module.generate_text(
                    params=learned_params,
                    prompt=prompt,
                    max_length=100
                )
                
                # Simple check for degeneration
                words = generation.split()
                unique_ratio = len(set(words)) / len(words) if words else 0
                
                print(f"Generated ({len(words)} words, {unique_ratio:.1%} unique):")
                print("-" * 40)
                print(generation)
                print("-" * 40)
                
                # Warning for potentially degenerated text
                if unique_ratio < 0.3 and len(words) > 20:
                    print("⚠️ WARNING: Low word diversity detected, possible degeneration")
            except Exception as e:
                print(f"Generation failed: {e}")
        
        # Record cycle results in history
        perplexity_history.append(final_perplexity)
        head_count_history.append(final_head_count)
        efficiency_history.append(current_efficiency)
        
        # Update strategy performance tracking
        pruning_success = final_perplexity < pruned_perplexity  # Did we recover from pruning?
        if strategy_name in pruning_strategy_results:
            pruning_strategy_results[strategy_name]["trials"] += 1
            if pruning_success:
                pruning_strategy_results[strategy_name]["success"] += 1
            # Record perplexity change (negative is good)
            perplexity_change = (final_perplexity - baseline_perplexity) / baseline_perplexity
            pruning_strategy_results[strategy_name]["perplexity_change"].append(perplexity_change)
        
        if growth_strategy_name in growth_strategy_results:
            growth_strategy_results[growth_strategy_name]["trials"] += 1
            if pruning_success:
                growth_strategy_results[growth_strategy_name]["success"] += 1
            # Record perplexity change
            perplexity_change = (final_perplexity - baseline_perplexity) / baseline_perplexity
            growth_strategy_results[growth_strategy_name]["perplexity_change"].append(perplexity_change)
            
        # Check if this is the best model so far
        is_best_model = False
        improvement_type = []
        
        # Check performance improvement
        if final_perplexity < best_perplexity:
            is_best_model = True
            improvement_type.append("perplexity")
            print(f"New best perplexity: {final_perplexity:.2f} (previous: {best_perplexity:.2f})")
            
            # Record the best strategy pair if we found a significant improvement
            if (best_perplexity - final_perplexity) / best_perplexity > 0.02:  # 2% improvement
                best_strategy_pair = (strategy_name, growth_strategy_name)
                print(f"Recording best strategy pair: {best_strategy_pair[0]} pruning + {best_strategy_pair[1]} growth")
        
        # Check for best efficiency
        current_efficiency_score = current_efficiency
        if current_efficiency_score < best_efficiency:
            best_efficiency = current_efficiency_score
            
            if "efficiency" in targets:
                is_best_model = True
                improvement_type.append("efficiency")
                print(f"New best efficiency: {current_efficiency_score:.4f} (previous: {best_efficiency:.4f})")
        
        # Check for head count improvement (if size is a target)
        if "size" in targets and final_head_count <= best_head_count and final_perplexity <= max_acceptable_perplexity:
            is_best_model = True
            improvement_type.append("size")
            print(f"New best head count: {final_head_count} (previous: {best_head_count})")
        
        # Update best model if improvements found
        if is_best_model:
            best_params = learned_params
            best_perplexity = final_perplexity
            best_head_count = final_head_count
            consecutive_failures = 0
            
            # Save as best model
            if args.save_model:
                best_model_path = os.path.join(checkpoints_dir, "model_best.pth")
                with open(best_model_path, 'wb') as f:
                    import pickle
                    pickle.dump(best_params, f)
                    
                # Also save a tagged version with the improvement type
                tagged_name = f"model_best_{'-'.join(improvement_type)}.pth"
                tagged_path = os.path.join(checkpoints_dir, tagged_name)
                with open(tagged_path, 'wb') as f:
                    import pickle
                    pickle.dump(best_params, f)
                
                print(f"Saved best model to {best_model_path} and {tagged_path}")
                
            # Save head importance data if tracking
            if args.track_head_importance:
                importance_path = os.path.join(analysis_dir, f"head_importance_cycle{cycle+1}.json")
                # Convert tuple keys to strings for JSON serialization
                serializable_importance = {}
                for (layer_idx, head_idx), data in head_importance_history.items():
                    serializable_importance[f"{layer_idx}_{head_idx}"] = data
                
                with open(importance_path, 'w') as f:
                    json.dump(serializable_importance, f, indent=2)
        else:
            consecutive_failures += 1
            print(f"No improvement in this cycle. Consecutive cycles without improvement: {consecutive_failures}")
        
        # Check if we've reached our targets
        performance_target_met = final_perplexity <= target_perplexity
        size_target_met = final_head_count <= target_head_count
        efficiency_target_met = current_efficiency <= target_efficiency
        
        targets_met = []
        if "perf" in targets and performance_target_met:
            targets_met.append("performance")
        if "size" in targets and size_target_met:
            targets_met.append("size")
        if "efficiency" in targets and efficiency_target_met:
            targets_met.append("efficiency")
        
        # Criteria for early stopping
        if targets_met:
            print(f"\n=== Target{'s' if len(targets_met) > 1 else ''} reached! ===")
            for target in targets_met:
                if target == "performance":
                    print(f"Performance target: {target_perplexity:.2f}, achieved: {final_perplexity:.2f}")
                elif target == "size":
                    print(f"Size target: {target_head_count} heads, achieved: {final_head_count} heads")
                elif target == "efficiency":
                    print(f"Efficiency target: {target_efficiency:.5f}, achieved: {current_efficiency:.5f}")
            
            # Only stop if all priority targets are met
            priority_targets = [t for t in targets_met if t in targets]
            if set(priority_targets) == set(targets):
                print(f"All specified targets reached!")
                target_reached = True
                break
            else:
                print(f"Some targets reached, but continuing to optimize for remaining targets")
        
        # Early stopping if too many failures in a row
        if consecutive_failures >= args.early_stopping_patience:
            print(f"\n=== Early stopping due to lack of improvement ===")
            print(f"No improvements for {consecutive_failures} consecutive cycles (patience: {args.early_stopping_patience})")
            break
            
        # Save strategy results for adaptive strategy selection
        strategy_results_path = os.path.join(analysis_dir, "strategy_results.json")
        with open(strategy_results_path, 'w') as f:
            json.dump({
                "pruning_strategies": pruning_strategy_results,
                "growth_strategies": growth_strategy_results,
                "best_pair": best_strategy_pair
            }, f, indent=2)
        
        # Update current params for next cycle
        current_params = learned_params
    
    # Create summary visualization
    if args.max_cycles > 1 and args.save_visualizations:
        # Create cycle comparison chart
        plt.figure(figsize=(12, 10))
        
        # Plot perplexity
        plt.subplot(2, 1, 1)
        plt.plot(
            [m["cycle"] for m in cycle_metrics],
            [m["pruned_perplexity"] for m in cycle_metrics],
            'r--', label="After Pruning", marker='o'
        )
        plt.plot(
            [m["cycle"] for m in cycle_metrics],
            [m["final_perplexity"] for m in cycle_metrics],
            'g-', label="After Learning", marker='s'
        )
        plt.axhline(y=baseline_perplexity, color='b', linestyle='-', label="Baseline")
        plt.title("Perplexity Through Optimization Cycles")
        plt.xlabel("Cycle")
        plt.ylabel("Perplexity")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot head percentage
        plt.subplot(2, 1, 2)
        plt.plot(
            [m["cycle"] for m in cycle_metrics],
            [m["head_percentage"] for m in cycle_metrics],
            'b-', marker='o'
        )
        plt.title("Active Heads Through Optimization Cycles")
        plt.xlabel("Cycle")
        plt.ylabel("Active Heads (%)")
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        summary_vis_path = os.path.join(vis_dir, "optimization_summary.png")
        plt.tight_layout()
        plt.savefig(summary_vis_path)
        plt.close()
    
    # Create comprehensive final summary
    # Extract best strategy pair if one was found
    best_pruning_strategy = best_strategy_pair[0] if best_strategy_pair else "none determined"
    best_growth_strategy = best_strategy_pair[1] if best_strategy_pair else "none determined"
    
    # Calculate final metrics
    final_perplexity = best_perplexity
    final_head_count = best_head_count
    perplexity_change = final_perplexity - baseline_perplexity
    perplexity_change_pct = (perplexity_change / baseline_perplexity) * 100 if baseline_perplexity > 0 else 0
    head_reduction_pct = (1 - final_head_count / original_head_count) * 100 if original_head_count > 0 else 0
    efficiency_improvement_pct = ((baseline_perplexity / original_head_count) / (final_perplexity / final_head_count) - 1) * 100
    
    # Create target achievement metrics
    targets_achieved = {}
    if "perf" in targets:
        targets_achieved["performance"] = {
            "target": float(target_perplexity),
            "achieved": float(final_perplexity),
            "met": final_perplexity <= target_perplexity
        }
    if "size" in targets:
        targets_achieved["size"] = {
            "target": int(target_head_count),
            "achieved": int(final_head_count),
            "met": final_head_count <= target_head_count
        }
    if "efficiency" in targets:
        targets_achieved["efficiency"] = {
            "target": float(target_efficiency),
            "achieved": float(final_perplexity / final_head_count),
            "met": (final_perplexity / final_head_count) <= target_efficiency
        }
    
    # Create strategy effectiveness metrics
    strategy_effectiveness = {}
    for strategy, stats in pruning_strategy_results.items():
        if stats["trials"] > 0:
            strategy_effectiveness[f"pruning_{strategy}"] = {
                "success_rate": stats["success"] / stats["trials"] if stats["trials"] > 0 else 0,
                "trials": stats["trials"],
                "avg_perplexity_change": sum(stats["perplexity_change"]) / len(stats["perplexity_change"]) if stats["perplexity_change"] else 0
            }
            
    for strategy, stats in growth_strategy_results.items():
        if stats["trials"] > 0:
            strategy_effectiveness[f"growth_{strategy}"] = {
                "success_rate": stats["success"] / stats["trials"] if stats["trials"] > 0 else 0,
                "trials": stats["trials"],
                "avg_perplexity_change": sum(stats["perplexity_change"]) / len(stats["perplexity_change"]) if stats["perplexity_change"] else 0
            }
    
    # Create detailed cycle metrics
    detailed_cycle_metrics = []
    for i, metrics in enumerate(cycle_metrics):
        cycle_num = i + 1
        metrics_with_targets = metrics.copy()
        metrics_with_targets["target_perplexity"] = target_perplexity
        metrics_with_targets["target_head_count"] = target_head_count
        detailed_cycle_metrics.append(metrics_with_targets)
        
    # Build the final summary
    summary = {
        "experiment_info": {
            "model_name": args.model_name,
            "dataset": args.dataset,
            "optimization_targets": targets,
            "cycles_completed": cycle + 1,
            "cycles_max": args.max_cycles,
            "target_reached": target_reached,
            "timestamp": datetime.now().isoformat(),
            "experiment_dir": experiment_dir
        },
        "performance_metrics": {
            "baseline_perplexity": float(baseline_perplexity),
            "final_perplexity": float(final_perplexity),
            "perplexity_change": float(perplexity_change),
            "perplexity_change_pct": float(perplexity_change_pct),
            "baseline_heads": int(original_head_count),
            "final_heads": int(final_head_count),
            "head_reduction_pct": float(head_reduction_pct),
            "baseline_efficiency": float(baseline_perplexity / original_head_count),
            "final_efficiency": float(final_perplexity / final_head_count),
            "efficiency_improvement_pct": float(efficiency_improvement_pct)
        },
        "target_achievement": targets_achieved,
        "best_strategies": {
            "pruning": best_pruning_strategy,
            "growth": best_growth_strategy
        },
        "strategy_effectiveness": strategy_effectiveness,
        "cycle_metrics": detailed_cycle_metrics,
        "configuration": {
            "pruning_strategy": args.pruning_strategy,
            "growth_strategy": args.growth_strategy,
            "growth_ratio": args.growth_ratio,
            "initial_pruning_level": args.initial_pruning_level,
            "adaptive_pruning": args.adaptive_pruning,
            "adaptive_growth": args.adaptive_growth,
            "memory_based_training": args.memory_based_training,
            "dynamic_targets": args.dynamic_targets
        }
    }
    
    # Save summary
    summary_path = os.path.join(experiment_dir, "optimization_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print("\n=== Optimization Summary ===")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Optimization cycles completed: {cycle + 1} of {args.max_cycles}")
    print(f"Targets: {', '.join(targets)}")
    print("\nPerformance metrics:")
    print(f"  Baseline perplexity: {baseline_perplexity:.2f}")
    print(f"  Final perplexity: {final_perplexity:.2f}")
    print(f"  Perplexity change: {perplexity_change:+.2f} ({perplexity_change_pct:+.2f}%)")
    print(f"  Baseline heads: {original_head_count}")
    print(f"  Final heads: {final_head_count}")
    print(f"  Head reduction: {head_reduction_pct:.1f}%")
    print(f"  Efficiency improvement: {efficiency_improvement_pct:+.1f}%")
    
    print("\nTarget achievement:")
    for target, data in targets_achieved.items():
        status = "✓" if data["met"] else "✗"
        print(f"  {target}: {status} (target: {data['target']:.4f}, achieved: {data['achieved']:.4f})")
    
    if best_strategy_pair:
        print(f"\nBest strategy combination:")
        print(f"  Pruning: {best_strategy_pair[0]}")
        print(f"  Growth: {best_strategy_pair[1]}")
    
    # Generate a recommended command for the next run based on findings
    if cycle + 1 >= args.max_cycles:
        print("\nRecommended next run:")
        recommended_cmd = f"python scripts/optimize_model_plasticity.py --model_name {args.model_name} --dataset {args.dataset}"
        
        # Add the best strategies if found
        if best_strategy_pair:
            recommended_cmd += f" --pruning_strategy {best_strategy_pair[0]} --growth_strategy {best_strategy_pair[1]}"
        
        # Adjust targets based on what was achieved
        new_targets = []
        if "perf" in targets and not (targets_achieved.get("performance", {}).get("met", False)):
            new_targets.append("perf")
        if "size" in targets and not (targets_achieved.get("size", {}).get("met", False)):
            new_targets.append("size")
        if "efficiency" in targets and not (targets_achieved.get("efficiency", {}).get("met", False)):
            new_targets.append("efficiency")
        
        if new_targets:
            recommended_cmd += f" --optimization_targets {','.join(new_targets)}"
        
        # Add adaptive options that likely help
        recommended_cmd += " --memory_based_training --adaptive_pruning --adaptive_growth"
        
        # Add improved targets
        if targets_achieved.get("performance", {}).get("met", False):
            # If we met performance, aim for 5% better
            new_target = targets_achieved["performance"]["achieved"] * 0.95
            recommended_cmd += f" --target_improvement {0.05}"
        
        if targets_achieved.get("size", {}).get("met", False):
            # If we met size, aim for 10% more reduction
            current_reduction = (original_head_count - final_head_count) / original_head_count
            recommended_cmd += f" --target_head_reduction {current_reduction + 0.1:.2f}"
            
        print(f"  {recommended_cmd}")
        
    print(f"\nExperiment saved to: {experiment_dir}")
    
    # Final inference test - compare baseline and optimized model
    print("\n=== Final Model Comparison ===")
    comparison_prompts = [
        "The most effective way to implement neural networks is to",
        "Recent advancements in language models have focused on",
        "The future of machine learning research will likely emphasize"
    ]
    
    # Display results in side-by-side format
    for i, prompt in enumerate(comparison_prompts):
        try:
            # Generate with baseline model
            baseline_generation = pruning_module.generate_text(
                params=baseline_params,
                prompt=prompt,
                max_length=100
            )
            
            # Generate with optimized model
            optimized_generation = pruning_module.generate_text(
                params=best_params,
                prompt=prompt,
                max_length=100
            )
            
            # Use the shared display function from inference_utils
            from utils.pruning.inference_utils import display_side_by_side
            display_side_by_side(prompt, baseline_generation, optimized_generation)
            
        except Exception as e:
            print(f"Comparison failed: {e}")
    
    # Print optimization results summary
    print("\n=== Results Summary ===")
    print(f"Perplexity: {baseline_perplexity:.2f} → {final_perplexity:.2f} ({perplexity_change_pct:+.2f}%)")
    print(f"Head count: {original_head_count} → {final_head_count} ({head_reduction_pct:.1f}%)")
    print(f"Efficiency: {baseline_perplexity/original_head_count:.4f} → {final_perplexity/final_head_count:.4f} perplexity/head ({efficiency_improvement_pct:+.1f}%)")
    
    return True

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
    
    # Evaluate using dataset batches if available
    try:
        if hasattr(dataset, 'eval_dataloader'):
            # Use dataloader for a more reliable perplexity calculation
            total_loss = 0.0
            total_tokens = 0
            
            # Get a few batches
            batch_count = 0
            max_batches = 5
            
            # Setup token distribution tracking to detect degeneration
            token_distributions = []
            
            for batch in dataset.eval_dataloader:
                if batch_count >= max_batches:
                    break
                
                # Forward pass
                outputs = pruning_module.model(**batch, params=params)
                
                # Get logits and labels
                logits = outputs.logits
                input_ids = batch["input_ids"]
                
                # Track token distribution to check for degeneration
                # Get most likely tokens for each position
                top_tokens = jnp.argmax(logits, axis=-1)
                
                # Count occurrences of each token in top predictions
                unique_tokens, counts = np.unique(np.array(top_tokens), return_counts=True)
                token_distributions.append((unique_tokens, counts))
                
                # Shift for next token prediction
                shift_logits = logits[:, :-1]
                shift_labels = input_ids[:, 1:]
                
                # Calculate token-level losses
                losses = jax.nn.log_softmax(shift_logits)
                one_hot_labels = jax.nn.one_hot(shift_labels, shift_logits.shape[-1])
                token_losses = -jnp.sum(losses * one_hot_labels, axis=-1)
                
                # Sum losses for all tokens
                batch_loss = jnp.sum(token_losses)
                batch_tokens = jnp.sum(jnp.ones_like(shift_labels))
                
                total_loss += float(batch_loss)
                total_tokens += float(batch_tokens)
                
                batch_count += 1
            
            if total_tokens > 0:
                # Calculate perplexity over all tokens
                avg_loss = total_loss / total_tokens
                dataset_perplexity = jnp.exp(avg_loss).item()
                perplexities.append(dataset_perplexity)
                print(f"Dataset-based perplexity: {dataset_perplexity:.2f} (over {total_tokens} tokens)")
                
                # Check for degeneration into stop tokens or repetitive patterns
                degeneration_detected = False
                
                # Combine all token distributions
                all_tokens = np.concatenate([dist[0] for dist in token_distributions if len(dist[0]) > 0])
                all_counts = np.concatenate([dist[1] for dist in token_distributions if len(dist[1]) > 0])
                
                if len(all_tokens) > 0:
                    # Sort by frequency
                    sort_indices = np.argsort(-all_counts)
                    sorted_tokens = all_tokens[sort_indices]
                    sorted_counts = all_counts[sort_indices]
                    
                    # Check if any token dominates predictions (>40% of outputs)
                    total_tokens_analyzed = sum(all_counts)
                    if total_tokens_analyzed > 0:
                        top_token_ratio = sorted_counts[0] / total_tokens_analyzed if len(sorted_counts) > 0 else 0
                        
                        if top_token_ratio > 0.40:  # 40% threshold for degeneration warning
                            try:
                                top_token = sorted_tokens[0]
                                if hasattr(pruning_module, 'tokenizer'):
                                    token_str = pruning_module.tokenizer.decode([top_token])
                                    print(f"⚠️ WARNING: Possible degeneration detected! Token '{token_str}' appears in {top_token_ratio:.1%} of outputs")
                                else:
                                    print(f"⚠️ WARNING: Possible degeneration detected! Token ID {top_token} appears in {top_token_ratio:.1%} of outputs")
                                degeneration_detected = True
                            except Exception as decode_error:
                                print(f"Error decoding token: {decode_error}")
                
                if degeneration_detected:
                    # Penalize the perplexity score to discourage this behavior
                    # Add a penalty factor to the perplexity
                    penalty_factor = 1.5  # 50% penalty for degeneration
                    adjusted_perplexity = dataset_perplexity * penalty_factor
                    print(f"Applying degeneration penalty: {dataset_perplexity:.2f} → {adjusted_perplexity:.2f}")
                    
                    # Replace the previously added perplexity with the penalized version
                    perplexities[-1] = adjusted_perplexity
    except Exception as e:
        print(f"Error in dataset evaluation: {e}")
        
    # Always evaluate with individual samples for consistent comparison
    for sample in eval_samples:
        # Calculate perplexity
        perplexity = pruning_module.evaluate_perplexity(params, sample)
        if not np.isnan(perplexity) and not np.isinf(perplexity):
            perplexities.append(perplexity)
        
        # Generate text if requested
        if generate_length > 0:
            try:
                prompt = sample[:30]
                generation = pruning_module.generate_text(params, prompt, max_length=generate_length)
                
                # Check for degeneration in generated text
                degeneration_score = 0
                degeneration_issues = []
                
                # 1. Check for repetition of same token sequences
                if generation and len(generation) > 50:
                    # Look for repeated substrings (3+ tokens repeating 3+ times)
                    for substr_len in range(3, 8):  # Check 3-7 token sequences
                        for i in range(len(generation) - substr_len * 3):
                            substr = generation[i:i+substr_len]
                            if substr.strip():  # Skip empty sequences
                                # Count occurrences with overlap
                                count = 0
                                pos = i
                                while True:
                                    pos = generation.find(substr, pos + 1)
                                    if pos == -1:
                                        break
                                    count += 1
                                
                                if count >= 3:  # 3+ repetitions is suspicious
                                    degeneration_score += 0.2 * min(count, 5)  # Cap at 5 repetitions for scoring
                                    degeneration_issues.append(f"Repeated sequence: '{substr}' ({count}x)")
                                    break  # Only count one repeating sequence per length
                
                # 2. Check if generation has very few unique tokens (only count first 3 issues)
                if generation and len(generation) > 20:
                    words = generation.split()
                    if len(words) > 10:
                        unique_ratio = len(set(words)) / len(words)
                        if unique_ratio < 0.3:  # Less than 30% unique words
                            degeneration_score += 0.5
                            degeneration_issues.append(f"Low word diversity: {unique_ratio:.1%} unique")
                
                # 3. Check for excessive punctuation or special tokens
                if generation:
                    punct_count = sum(1 for c in generation if c in '.,!?;:()[]{}"\'')
                    if len(generation) > 0:
                        punct_ratio = punct_count / len(generation)
                        if punct_ratio > 0.2:  # More than 20% punctuation
                            degeneration_score += 0.3
                            degeneration_issues.append(f"High punctuation ratio: {punct_ratio:.1%}")
                
                # Add degeneration info to results
                if degeneration_score > 0:
                    issues_str = "; ".join(degeneration_issues[:3])  # Limit to first 3 issues for brevity
                    degeneration_info = f"[⚠️ Degeneration score: {degeneration_score:.1f} - {issues_str}]"
                    
                    # If severe degeneration, penalize perplexity
                    if degeneration_score >= 1.0 and perplexity is not None:
                        # Apply penalty proportional to degeneration severity
                        penalty = 1.0 + min(degeneration_score, 3.0) * 0.2  # Max 60% penalty
                        adjusted_perplexity = perplexity * penalty
                        print(f"Sample perplexity degeneration penalty: {perplexity:.2f} → {adjusted_perplexity:.2f}")
                        perplexity = adjusted_perplexity
                else:
                    degeneration_info = ""
                
                results.append({
                    "prompt": prompt,
                    "perplexity": float(perplexity) if not np.isnan(perplexity) and not np.isinf(perplexity) else None,
                    "generation": generation,
                    "degeneration_info": degeneration_info if degeneration_score > 0 else ""
                })
            except Exception as e:
                print(f"Error in text generation: {e}")
                results.append({
                    "prompt": sample[:30],
                    "perplexity": float(perplexity) if not np.isnan(perplexity) and not np.isinf(perplexity) else None,
                    "generation": f"[Generation failed: {str(e)}]",
                    "degeneration_info": ""
                })
    
    # Calculate average perplexity
    avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else float('nan')
    
    return {
        "samples": results,
        "average_perplexity": float(avg_perplexity),
        "perplexities": [float(p) for p in perplexities]
    }

def main():
    """Main function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run optimization
    success = optimize_model(args)
    
    if success:
        print("\nModel optimization completed successfully!")
    else:
        print("\nModel optimization failed.")

if __name__ == "__main__":
    main()