#!/usr/bin/env python
"""
Test script for head growth functionality.

This script loads a model, prunes it, then grows new heads to demonstrate
the full neural plasticity cycle.
"""

import os
import argparse
import json
import time
import jax
import jax.numpy as jnp
import random
import pickle

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pruning.pruning_module import PruningModule
from utils.pruning.strategies import get_strategy as get_pruning_strategy
from utils.pruning.growth import grow_attention_heads_gradually, determine_active_heads

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test the neural plasticity cycle with pruning and growth")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                      help="Model name (default: distilgpt2)")
    parser.add_argument("--save_dir", type=str, default="./plasticity_test",
                      help="Directory to save test results (default: ./plasticity_test)")
    
    # Pruning parameters
    parser.add_argument("--pruning_level", type=float, default=0.3,
                      help="Pruning level (percentage of heads to prune, default: 0.3)")
    parser.add_argument("--pruning_strategy", type=str, default="entropy",
                      choices=["random", "magnitude", "entropy"],
                      help="Strategy for pruning (default: entropy)")
    
    # Growth parameters
    parser.add_argument("--growth_percentage", type=float, default=0.1,
                      help="Percentage of heads to grow (default: 0.1)")
    parser.add_argument("--growth_strategy", type=str, default="gradient_sensitivity",
                      choices=["gradient_sensitivity", "entropy_gap", "balanced", "random"],
                      help="Strategy for growth (default: gradient_sensitivity)")
    parser.add_argument("--initial_scale", type=float, default=0.01,
                      help="Initial scale for new head weights (default: 0.01)")
    
    # Test parameters
    parser.add_argument("--eval_samples", type=int, default=3,
                      help="Number of samples for evaluation (default: 3)")
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

def evaluate_model(pruning_module, params, eval_text=None):
    """Evaluate model on sample text"""
    # Define evaluation texts if not provided
    if eval_text is None:
        eval_texts = [
            "The neural network model has been trained to recognize patterns in data.",
            "Artificial intelligence systems can learn from experience and improve over time.",
            "The transformer architecture uses self-attention mechanisms to process sequences."
        ]
    elif isinstance(eval_text, str):
        eval_texts = [eval_text]
    else:
        eval_texts = eval_text
    
    results = []
    for text in eval_texts:
        # Calculate perplexity
        perplexity = pruning_module.evaluate_perplexity(params, text)
        
        # Generate text
        prompt = text[:30]
        generation = pruning_module.generate_text(params, prompt, max_length=100)
        
        results.append({
            "prompt": prompt,
            "perplexity": perplexity,
            "generation": generation
        })
    
    # Calculate average perplexity
    avg_perplexity = sum(r["perplexity"] for r in results if not (jnp.isnan(r["perplexity"]) or jnp.isinf(r["perplexity"])))
    avg_perplexity /= len(results)
    
    return {
        "samples": results,
        "average_perplexity": avg_perplexity
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
    pruned_params = strategy.prune_heads(params, heads_to_prune)
    
    return pruned_params, heads_to_prune

def main():
    """Main function"""
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create pruning module
    pruning_module = PruningModule(args.model_name)
    
    # Load model
    print(f"Loading model {args.model_name}...")
    if not pruning_module.load_model():
        print(f"Failed to load model {args.model_name}")
        return
    
    # Get original parameters
    original_params = pruning_module.model.params
    
    # Get active heads in original model
    print("Analyzing original model state...")
    original_active_heads = determine_active_heads(pruning_module, original_params)
    print(f"Original model has {len(original_active_heads)} active heads out of " +
          f"{pruning_module.num_layers * pruning_module.num_heads} total")
    
    # Visualize original model's head map
    original_head_map = visualize_head_map(pruning_module, original_active_heads, 
                                       "Original Model Head Map")
    print(original_head_map)
    
    # Evaluate original model
    print("\nEvaluating original model...")
    original_eval = evaluate_model(pruning_module, original_params)
    print(f"Original model average perplexity: {original_eval['average_perplexity']:.2f}")
    print(f"Sample generation: {original_eval['samples'][0]['generation'][:100]}...")
    
    # Prune the model
    print(f"\nPruning model with {args.pruning_strategy} strategy at {args.pruning_level*100:.1f}% level...")
    pruned_params, pruned_heads = prune_model(pruning_module, original_params, 
                                          args.pruning_level, args.pruning_strategy)
    
    # Get active heads in pruned model
    print("Analyzing pruned model state...")
    pruned_active_heads = determine_active_heads(pruning_module, pruned_params)
    print(f"Pruned {len(pruned_heads)} heads, {len(pruned_active_heads)} active heads remaining")
    
    # Visualize pruned model's head map
    pruned_head_map = visualize_head_map(pruning_module, pruned_active_heads, 
                                     "Pruned Model Head Map")
    print(pruned_head_map)
    
    # Evaluate pruned model
    print("\nEvaluating pruned model...")
    pruned_eval = evaluate_model(pruning_module, pruned_params)
    print(f"Pruned model average perplexity: {pruned_eval['average_perplexity']:.2f}")
    print(f"Sample generation: {pruned_eval['samples'][0]['generation'][:100]}...")
    
    # Grow new heads
    print(f"\nGrowing heads with {args.growth_strategy} strategy at {args.growth_percentage*100:.1f}% level...")
    grown_params, added_count, added_heads, warmup_schedule = grow_attention_heads_gradually(
        pruning_module,
        params=pruned_params,
        active_heads=pruned_active_heads,
        growth_percentage=args.growth_percentage,
        strategy=args.growth_strategy,
        initial_scale=args.initial_scale
    )
    
    # Get active heads in grown model
    print("Analyzing grown model state...")
    grown_active_heads = determine_active_heads(pruning_module, grown_params)
    print(f"Added {added_count} heads, now have {len(grown_active_heads)} active heads")
    
    # Visualize grown model's head map
    grown_head_map = visualize_head_map(pruning_module, grown_active_heads, 
                                    "Grown Model Head Map")
    print(grown_head_map)
    
    # Evaluate grown model with initial scaling
    print("\nEvaluating grown model (with initial scaling)...")
    grown_eval_initial = evaluate_model(pruning_module, grown_params)
    print(f"Grown model (initial) average perplexity: {grown_eval_initial['average_perplexity']:.2f}")
    print(f"Sample generation: {grown_eval_initial['samples'][0]['generation'][:100]}...")
    
    # Simulate warmup completion
    print("\nSimulating warmup completion for new heads...")
    # In a real training loop, you would use warmup_schedule(step) to determine
    # the scaling factor for each step and update the weights accordingly.
    # Here we just simulate the end state (scale=1.0) without actually training.
    warmup_complete_params = grown_params  # In a real implementation, this would be updated during warmup
    
    # Evaluate model with simulated warmup completion
    print("Evaluating model after warmup simulation...")
    grown_eval_final = evaluate_model(pruning_module, warmup_complete_params)
    print(f"Grown model (final) average perplexity: {grown_eval_final['average_perplexity']:.2f}")
    print(f"Sample generation: {grown_eval_final['samples'][0]['generation'][:100]}...")
    
    # Save test results
    print("\nSaving test results...")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_path = os.path.join(args.save_dir, f"plasticity_test_{timestamp}.json")
    
    test_results = {
        "timestamp": timestamp,
        "model_name": args.model_name,
        "pruning": {
            "strategy": args.pruning_strategy,
            "level": args.pruning_level,
            "heads_pruned": len(pruned_heads),
            "active_heads_after": len(pruned_active_heads)
        },
        "growth": {
            "strategy": args.growth_strategy,
            "percentage": args.growth_percentage,
            "heads_added": added_count,
            "active_heads_after": len(grown_active_heads)
        },
        "metrics": {
            "original_perplexity": float(original_eval["average_perplexity"]),
            "pruned_perplexity": float(pruned_eval["average_perplexity"]),
            "grown_initial_perplexity": float(grown_eval_initial["average_perplexity"]),
            "grown_final_perplexity": float(grown_eval_final["average_perplexity"])
        },
        "head_maps": {
            "original": original_head_map,
            "pruned": pruned_head_map,
            "grown": grown_head_map
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Save model parameters
    models_dir = os.path.join(args.save_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    with open(os.path.join(models_dir, f"pruned_{timestamp}.pth"), 'wb') as f:
        pickle.dump(pruned_params, f)
    
    with open(os.path.join(models_dir, f"grown_{timestamp}.pth"), 'wb') as f:
        pickle.dump(grown_params, f)
    
    print(f"Results saved to {results_path}")
    print(f"Model parameters saved to {models_dir}")
    
    # Print summary
    print("\nNeural Plasticity Cycle Test Summary:")
    print(f"- Original model: {len(original_active_heads)} active heads, " +
          f"perplexity {original_eval['average_perplexity']:.2f}")
    print(f"- Pruned model: {len(pruned_active_heads)} active heads, " +
          f"perplexity {pruned_eval['average_perplexity']:.2f}")
    print(f"- Grown model: {len(grown_active_heads)} active heads, " +
          f"perplexity {grown_eval_final['average_perplexity']:.2f}")
    
    perplexity_change = grown_eval_final['average_perplexity'] - original_eval['average_perplexity']
    print(f"- Net perplexity change: {perplexity_change:+.2f}")
    
    head_change = len(grown_active_heads) - len(original_active_heads)
    print(f"- Net head count change: {head_change:+d}")
    
    print("\nNeural plasticity cycle completed successfully!")

if __name__ == "__main__":
    main()