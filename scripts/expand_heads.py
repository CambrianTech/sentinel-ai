#!/usr/bin/env python
"""
Expansion script to grow attention heads in pruned models.

This script loads a pruned model, identifies which heads to grow,
and saves a new model with additional attention heads.
"""

import os
import argparse
import time
import json
import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pruning.pruning_module import PruningModule
from utils.pruning.growth import grow_attention_heads_gradually

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Grow attention heads in a pruned transformer model")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, required=True, 
                      help="Path to the pruned model checkpoint")
    parser.add_argument("--output_path", type=str, default=None,
                      help="Path to save the expanded model (default: model_path with _expanded suffix)")
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                      help="Base model name (default: distilgpt2)")
    
    # Growth parameters
    parser.add_argument("--growth_percentage", type=float, default=0.05,
                      help="Percentage of total heads to add (default: 0.05 = 5%%)")
    parser.add_argument("--growth_strategy", type=str, default="gradient_sensitivity",
                      choices=["gradient_sensitivity", "entropy_gap", "balanced", "random"],
                      help="Strategy for selecting which heads to grow (default: gradient_sensitivity)")
    parser.add_argument("--initial_scale", type=float, default=0.01,
                      help="Initial weight scale for new heads (default: 0.01)")
    
    # Evaluation parameters
    parser.add_argument("--eval_text", type=str, default=None,
                      help="Sample text for evaluation (default: None)")
    parser.add_argument("--warmup_steps", type=int, default=100,
                      help="Steps for linear warmup of new heads (default: 100)")
    
    # Output options
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose output")
    
    return parser.parse_args()

def determine_active_heads_percent(pruning_module, params):
    """Determine what percentage of heads are currently active"""
    from utils.pruning.growth import determine_active_heads
    
    active_heads = determine_active_heads(pruning_module, params)
    total_heads = pruning_module.num_layers * pruning_module.num_heads
    
    return len(active_heads) / total_heads

def summarize_model_state(pruning_module, params=None):
    """Summarize the current state of the model"""
    if params is None:
        params = pruning_module.model.params
    
    from utils.pruning.growth import determine_active_heads
    
    # Get active heads
    active_heads = determine_active_heads(pruning_module, params)
    total_heads = pruning_module.num_layers * pruning_module.num_heads
    
    # Count active heads per layer
    layer_counts = {}
    for layer_idx in range(pruning_module.num_layers):
        active_in_layer = sum(1 for l, h in active_heads if l == layer_idx)
        layer_counts[layer_idx] = active_in_layer
    
    # Prepare summary
    summary = {
        "model_name": pruning_module.model_name,
        "total_heads": total_heads,
        "active_heads": len(active_heads),
        "active_percent": f"{len(active_heads) / total_heads * 100:.1f}%",
        "layer_distribution": layer_counts
    }
    
    return summary

def visualize_head_distribution(pruning_module, active_heads):
    """Create a visual representation of head distribution"""
    num_layers = pruning_module.num_layers
    num_heads = pruning_module.num_heads
    
    # Create a layer-by-layer visualization
    visual = ["Head Distribution (■=active, □=inactive):"]
    
    for layer_idx in range(num_layers):
        layer_visual = [f"Layer {layer_idx}: "]
        
        for head_idx in range(num_heads):
            if (layer_idx, head_idx) in active_heads:
                layer_visual.append("■")
            else:
                layer_visual.append("□")
        
        visual.append("".join(layer_visual))
    
    return "\n".join(visual)

def save_growth_info(output_path, growth_info):
    """Save growth information to a JSON file"""
    info_path = output_path.replace(".pth", "_growth_info.json")
    
    with open(info_path, 'w') as f:
        json.dump(growth_info, f, indent=2)
    
    return info_path

def evaluate_model(pruning_module, params, eval_text=None):
    """Evaluate model with specified parameters"""
    if eval_text is None:
        eval_text = (
            "The transformer model architecture has gained popularity due to its "
            "effectiveness in natural language processing tasks. It has been applied "
            "to various domains including machine translation, text generation, and "
            "sentiment analysis."
        )
    
    # Generate text
    generated_text = pruning_module.generate_text(params, eval_text[:50], max_length=100)
    
    # Calculate perplexity
    perplexity = pruning_module.evaluate_perplexity(params, eval_text)
    
    return {
        "perplexity": perplexity,
        "generated_text": generated_text
    }

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Set output path if not specified
    if args.output_path is None:
        args.output_path = args.model_path.replace(".pth", "_expanded.pth")
        # If no .pth extension, add _expanded suffix
        if args.output_path == args.model_path:
            args.output_path = args.model_path + "_expanded"
    
    # Create pruning module
    pruning_module = PruningModule(args.model_name)
    
    # Load model
    if not pruning_module.load_model():
        print(f"Failed to load model {args.model_name}")
        return
    
    # Load checkpoint parameters
    import pickle
    try:
        with open(args.model_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # If checkpoint is a dictionary with 'model_params', use those
        if isinstance(checkpoint, dict) and 'model_params' in checkpoint:
            params = checkpoint['model_params']
        else:
            # Assume checkpoint is the model parameters directly
            params = checkpoint
            
        print(f"Loaded checkpoint from {args.model_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Analyze initial model state
    print("Analyzing initial model state...")
    initial_state = summarize_model_state(pruning_module, params)
    
    if args.verbose:
        print(f"Initial model state: {json.dumps(initial_state, indent=2)}")
        from utils.pruning.growth import determine_active_heads
        active_heads = determine_active_heads(pruning_module, params)
        print(visualize_head_distribution(pruning_module, active_heads))
    
    # Evaluate initial model
    print("Evaluating initial model...")
    initial_eval = evaluate_model(pruning_module, params, args.eval_text)
    
    if args.verbose:
        print(f"Initial perplexity: {initial_eval['perplexity']:.2f}")
        print(f"Initial generation sample: {initial_eval['generated_text'][:100]}...")
    
    # Grow attention heads
    print(f"Growing heads using {args.growth_strategy} strategy...")
    start_time = time.time()
    
    new_params, added_count, added_heads, warmup_schedule = grow_attention_heads_gradually(
        pruning_module,
        params=params,
        growth_percentage=args.growth_percentage,
        strategy=args.growth_strategy,
        initial_scale=args.initial_scale,
        warmup_steps=args.warmup_steps
    )
    
    growth_time = time.time() - start_time
    
    # If no heads were added, exit
    if added_count == 0:
        print("No heads were added - model may already be fully active")
        return
    
    # Analyze expanded model state
    print("Analyzing expanded model state...")
    expanded_state = summarize_model_state(pruning_module, new_params)
    
    if args.verbose:
        print(f"Expanded model state: {json.dumps(expanded_state, indent=2)}")
        from utils.pruning.growth import determine_active_heads
        new_active_heads = determine_active_heads(pruning_module, new_params)
        print(visualize_head_distribution(pruning_module, new_active_heads))
    
    # Evaluate expanded model
    print("Evaluating expanded model...")
    expanded_eval = evaluate_model(pruning_module, new_params, args.eval_text)
    
    if args.verbose:
        print(f"Expanded perplexity: {expanded_eval['perplexity']:.2f}")
        print(f"Expanded generation sample: {expanded_eval['generated_text'][:100]}...")
    
    # Calculate change in metrics
    perplexity_change = expanded_eval['perplexity'] - initial_eval['perplexity']
    
    # Prepare growth info
    growth_info = {
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "model_name": args.model_name,
        "growth_strategy": args.growth_strategy,
        "growth_percentage": args.growth_percentage,
        "initial_scale": args.initial_scale,
        "warmup_steps": args.warmup_steps,
        "initial_state": initial_state,
        "expanded_state": expanded_state,
        "added_count": added_count,
        "added_heads": [(int(l), int(h)) for l, h in added_heads],  # Convert to int for JSON
        "growth_time_seconds": growth_time,
        "metrics": {
            "initial_perplexity": float(initial_eval['perplexity']),
            "expanded_perplexity": float(expanded_eval['perplexity']),
            "perplexity_change": float(perplexity_change)
        }
    }
    
    # Save expanded model
    print(f"Saving expanded model to {args.output_path}...")
    try:
        # Create checkpoint with expanded parameters
        if isinstance(checkpoint, dict) and 'model_params' in checkpoint:
            checkpoint['model_params'] = new_params
            checkpoint['growth_info'] = {
                "timestamp": time.strftime("%Y%m%d-%H%M%S"),
                "strategy": args.growth_strategy,
                "added_count": added_count,
                "added_heads": [(int(l), int(h)) for l, h in added_heads]
            }
            save_data = checkpoint
        else:
            # Just save the parameters
            save_data = new_params
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        
        # Save checkpoint
        with open(args.output_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Saved expanded model to {args.output_path}")
        
        # Save growth info
        info_path = save_growth_info(args.output_path, growth_info)
        print(f"Saved growth info to {info_path}")
    except Exception as e:
        print(f"Error saving expanded model: {e}")
    
    # Summary
    print("\nSummary:")
    print(f"- Added {added_count} heads ({args.growth_percentage*100:.1f}% of total)")
    print(f"- Active heads: {initial_state['active_heads']} → {expanded_state['active_heads']} " +
          f"({initial_state['active_percent']} → {expanded_state['active_percent']})")
    print(f"- Perplexity: {initial_eval['perplexity']:.2f} → {expanded_eval['perplexity']:.2f} " +
          f"({perplexity_change:+.2f})")
    
    print(f"\nExpanded model saved to {args.output_path}")
    print(f"To use this model in training, load the checkpoint and initialize with the new parameters")
    print(f"For gradual integration, use the warmup schedule function over {args.warmup_steps} steps")

if __name__ == "__main__":
    main()