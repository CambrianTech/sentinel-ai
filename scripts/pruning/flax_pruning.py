#!/usr/bin/env python3
"""
Pruning implementation using JAX/Flax instead of PyTorch to avoid BLAS issues

This script is designed to run on both M1/M2 Macs (where PyTorch has BLAS issues)
and on Google Colab (where it can utilize TPUs/GPUs when available).
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
import platform

# Determine if we're on a Mac
is_mac = platform.system() == "Darwin"
is_arm_mac = is_mac and platform.machine().startswith("arm")

# Check if required packages are installed, if not, install them
try:
    import jax
    import jax.numpy as jnp
    from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
    from flax.training.train_state import TrainState
    print("Required packages already installed")
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "jax", "flax", "transformers"])
    
    # Now import
    import jax
    import jax.numpy as jnp
    from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
    from flax.training.train_state import TrainState

# Set environment variables for better JAX performance
if is_arm_mac:
    # Mac-specific settings to avoid BLAS issues
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
else:
    # For Colab, let JAX use available accelerators
    pass

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")

def prune_head_in_params(params, layer_idx, head_idx, model_type="gpt2"):
    """Zero out weights for a specific attention head in Flax params"""
    if model_type == "gpt2":
        # Access path to transformer layers
        transformer_path = "transformer"
        # In Flax, layer indices are stored as strings
        layer_path = f"h"
        layer_key = str(layer_idx)
        attn_path = "attn"
        
        # Get attention block
        attn_block = params[transformer_path][layer_path][layer_key][attn_path]
        
        # Get head dimension and number of heads
        num_heads = 12  # Standard for GPT-2
        if "distil" in model_type:
            num_heads = 12  # DistilGPT-2 also has 12 heads
        
        hidden_size = attn_block["c_attn"]["kernel"].shape[1]
        head_size = hidden_size // num_heads
        
        # Calculate start and end indices for this head in query, key, value
        q_start = head_idx * head_size
        q_end = (head_idx + 1) * head_size
        
        # Zero out the output projection for this head
        output_proj = attn_block["c_proj"]["kernel"]
        # In Flax, c_proj.kernel has shape [hidden_size, hidden_size]
        # We need to zero out the rows corresponding to this head
        zeros = jnp.zeros_like(output_proj[q_start:q_end, :])
        output_proj = output_proj.at[q_start:q_end, :].set(zeros)
        
        # Update the parameters
        params[transformer_path][layer_path][layer_key][attn_path]["c_proj"]["kernel"] = output_proj
        
        print(f"Successfully pruned layer {layer_idx}, head {head_idx}")
    
    return params

def evaluate_perplexity(model, params, tokenizer, text):
    """Evaluate model perplexity on text"""
    # Tokenize input
    inputs = tokenizer(text, return_tensors="jax")
    
    # Get logits
    outputs = model(**inputs, params=params)
    logits = outputs.logits
    
    # Calculate loss
    input_ids = inputs["input_ids"]
    
    # Shift logits and labels for next token prediction
    shift_logits = logits[:, :-1]
    shift_labels = input_ids[:, 1:]
    
    # Calculate cross entropy loss
    loss = jnp.mean(
        -jnp.sum(
            jax.nn.log_softmax(shift_logits) * jax.nn.one_hot(shift_labels, shift_logits.shape[-1]),
            axis=-1
        )
    )
    
    # Return perplexity
    return jnp.exp(loss).item()

def generate_text(model, params, tokenizer, prompt, max_length=50):
    """Generate text using the model"""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="jax")
    
    # Generate text
    outputs = model.generate(
        **inputs,
        params=params,
        max_length=max_length,
        do_sample=True,
        top_k=40,
        top_p=0.95
    )
    
    # Decode output
    text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
    return text

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run pruning benchmark with JAX/Flax")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="Model name")
    parser.add_argument("--strategy", type=str, default="random", choices=["random", "entropy"], 
                        help="Pruning strategy")
    parser.add_argument("--pruning_level", type=float, default=0.3, 
                        help="Pruning level (0.0 to 1.0)")
    parser.add_argument("--prompt", type=str, default="Artificial intelligence is", 
                        help="Prompt for text generation")
    parser.add_argument("--output_dir", type=str, default="pruning_results", 
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Make output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nRunning JAX/Flax pruning benchmark with:")
    print(f"  Model: {args.model_name}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Pruning level: {args.pruning_level}")
    print(f"  Prompt: {args.prompt}\n")
    
    # Load model and tokenizer
    print(f"Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = FlaxAutoModelForCausalLM.from_pretrained(args.model_name)
    
    # Get model information
    if "gpt2" in args.model_name:
        model_type = "gpt2"
        num_layers = len(model.params["transformer"]["h"])
        num_heads = 12  # Standard for GPT-2
    else:
        raise ValueError(f"Unsupported model type: {args.model_name}")
    
    print(f"Model has {num_layers} layers with {num_heads} heads per layer")
    
    # Make a copy of the parameters (so we can keep original for comparison)
    original_params = model.params
    params = jax.tree_util.tree_map(lambda x: x, original_params)  # Deep copy
    
    # Evaluate model before pruning
    print("\nEvaluating model before pruning...")
    perplexity_before = evaluate_perplexity(model, params, tokenizer, args.prompt)
    print(f"Perplexity before pruning: {perplexity_before:.4f}")
    
    generated_before = generate_text(model, params, tokenizer, args.prompt)
    print(f"Generated (before pruning): {generated_before}")
    
    # Calculate head importance
    print("\nCalculating head importance...")
    all_head_importance = []
    
    # Different strategies for calculating importance
    for layer_idx in range(num_layers):
        if args.strategy == "random":
            # Random importance
            importance = np.random.rand(num_heads)
        else:  # entropy
            # Use simplified proxy: norm of output projection weights
            importance = np.zeros(num_heads)
            layer_params = params["transformer"]["h"][str(layer_idx)]["attn"]
            output_proj = layer_params["c_proj"]["kernel"]
            
            head_size = output_proj.shape[0] // num_heads
            for head_idx in range(num_heads):
                start_idx = head_idx * head_size
                end_idx = (head_idx + 1) * head_size
                importance[head_idx] = jnp.linalg.norm(output_proj[start_idx:end_idx, :]).item()
        
        # Normalize importance scores
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        
        # Add to all heads
        for head_idx, score in enumerate(importance):
            all_head_importance.append((layer_idx, head_idx, score))
    
    # Sort by importance (ascending)
    all_head_importance.sort(key=lambda x: x[2])
    
    # Calculate number of heads to prune
    total_heads = num_layers * num_heads
    heads_to_prune = int(total_heads * args.pruning_level)
    print(f"Pruning {heads_to_prune} out of {total_heads} heads")
    
    # Get heads to prune (least important first)
    pruned_heads = all_head_importance[:heads_to_prune]
    
    # Prune the heads
    print("\nPruning heads...")
    for layer_idx, head_idx, _ in pruned_heads:
        params = prune_head_in_params(params, layer_idx, head_idx, model_type)
    
    # Evaluate model after pruning
    print("\nEvaluating model after pruning...")
    perplexity_after = evaluate_perplexity(model, params, tokenizer, args.prompt)
    print(f"Perplexity after pruning: {perplexity_after:.4f}")
    print(f"Perplexity change: {perplexity_after - perplexity_before:.4f}")
    
    generated_after = generate_text(model, params, tokenizer, args.prompt)
    print(f"Generated (after pruning): {generated_after}")
    
    # Save results
    results = {
        "model": args.model_name,
        "strategy": args.strategy,
        "pruning_level": args.pruning_level,
        "pruned_heads": heads_to_prune,
        "total_heads": total_heads,
        "prompt": args.prompt,
        "perplexity_before": float(perplexity_before),  # Convert from JAX array
        "perplexity_after": float(perplexity_after),
        "perplexity_change": float(perplexity_after - perplexity_before),
        "generated_before": generated_before,
        "generated_after": generated_after,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{args.strategy}_{args.pruning_level}_{timestamp}.json"
    filepath = os.path.join(args.output_dir, filename)
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {filepath}")
    print("\nBenchmark completed successfully!")

if __name__ == "__main__":
    main()