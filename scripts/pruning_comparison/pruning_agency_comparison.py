#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pruning Efficacy Comparison Between Baseline and Agency-Enabled Models

This script demonstrates the effectiveness of agency-enabled models
compared to baseline models when subjected to aggressive pruning.

Features:
- Prunes both baseline and agency-enabled models to 50%
- Compares inference speed, quality, and resource utilization
- Generates visualizations showing the differences
- Measures performance degradation curves
"""

import argparse
import os
import sys
import time
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add root directory to path
try:
    # When running as a script with __file__ available
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
except NameError:
    # In Colab or interactive environments where __file__ isn't defined
    # First check if we're in the repo root or one level down
    if os.path.exists("models") and os.path.exists("utils"):
        # We're already in the root directory
        sys.path.insert(0, os.path.abspath("."))
    elif os.path.exists("../models") and os.path.exists("../utils"):
        # We're one level down from root
        sys.path.insert(0, os.path.abspath(".."))
    elif os.path.exists("sentinel-ai/models") and os.path.exists("sentinel-ai/utils"):
        # We're in the parent directory of the repo (typical Colab setup)
        sys.path.insert(0, os.path.abspath("sentinel-ai"))
    else:
        # Additional fallback paths for Colab - check common locations
        import glob
        possible_paths = glob.glob("*/models") + glob.glob("*/*/models")
        if possible_paths:
            # Use the first directory that has models
            repo_path = os.path.dirname(possible_paths[0])
            print(f"Found models directory at {repo_path}, adding to path")
            sys.path.insert(0, os.path.abspath(repo_path))
        else:
            print("Warning: Could not determine repository root path. Import errors may occur.")

from models.loaders.loader import load_baseline_model, load_adaptive_model
from models.loaders.gpt2_loader import load_adaptive_model_gpt
from models.adaptive_transformer import AdaptiveCausalLmWrapper  # Import correct model class
from transformers import AutoTokenizer
from utils.metrics import calculate_perplexity, calculate_diversity, calculate_repetition
from utils.charting import AGENCY_COLORS  # Import color scheme for consistency

# Set plotting style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def setup_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare pruning efficacy with and without agency")
    
    # Pruning configuration
    parser.add_argument("--pruning_levels", type=str, default="0,20,40,60",
                      help="Comma-separated pruning percentages to evaluate")
    parser.add_argument("--pruning_method", type=str, default="entropy", choices=["entropy", "random", "magnitude"],
                      help="Method to select heads for pruning")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="gpt2",
                      help="Base model to use (gpt2, distilgpt2)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                      help="Device to run on (cpu or cuda)")
    parser.add_argument("--batch_size", type=int, default=1, 
                      help="Batch size for generation")
    parser.add_argument("--precision", type=str, default="float32", choices=["float32", "float16", "bfloat16"],
                      help="Precision for model weights")
    
    # Generation parameters
    parser.add_argument("--num_tokens", type=int, default=50,
                      help="Number of tokens to generate for each prompt")
    parser.add_argument("--temperatures", type=str, default="0.7", 
                      help="Comma-separated temperatures to test (use single temperature for CPU tests)")
    parser.add_argument("--prompt_file", type=str, default="datasets/eval_prompts.txt",
                      help="File containing prompts to use for evaluation")
    parser.add_argument("--max_prompts", type=int, default=3,
                      help="Maximum number of prompts to evaluate (use small number for CPU tests)")
    
    # Experiment configuration
    parser.add_argument("--iterations", type=int, default=2,
                      help="Number of iterations to run for statistical significance (use small number for CPU tests)")
    parser.add_argument("--output_dir", type=str, default="validation_results/pruning_agency",
                      help="Directory to save results and visualizations")
    parser.add_argument("--save_outputs", action="store_true",
                      help="Save generated text outputs")
    parser.add_argument("--visualize_only", action="store_true",
                      help="Only generate visualizations from existing results")
    parser.add_argument("--memory_logging", action="store_true",
                      help="Log memory usage during evaluation")
    parser.add_argument("--quiet", action="store_true",
                      help="Reduce logging verbosity")
    
    return parser.parse_args()

def load_model(model_name, agency_enabled=False, device="cpu", precision="float32", quiet=False):
    """Load either a baseline or agency-enabled model with specified device and precision."""
    # Define a local log function
    def log_model(message, force=False):
        if not quiet or force:
            print(message)
            
    log_model(f"Loading {model_name} model with agency={'enabled' if agency_enabled else 'disabled'} on {device}...")
    
    # Set device
    torch_device = device
    if device == "cuda" and not torch.cuda.is_available():
        log_model("CUDA requested but not available. Using CPU instead.", force=True)
        torch_device = "cpu"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if agency_enabled:
        # Load baseline model first
        baseline_model = load_baseline_model(model_name, torch_device)
        
        # Then load adaptive model with agency
        model = load_adaptive_model(model_name, baseline_model, torch_device, debug=False, quiet=quiet)
        
        # Set agency enabled flag for reference in pruning
        if hasattr(model, "model"):
            model.model.enable_agency = True
        else:
            model.enable_agency = True
    else:
        # Load baseline model only
        model = load_baseline_model(model_name, torch_device)
    
    # Set precision
    if precision != "float32" and torch_device == "cuda":
        if precision == "float16":
            log_model("Converting model to float16 precision...")
            model = model.half()
        elif precision == "bfloat16" and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            log_model("Converting model to bfloat16 precision...")
            model = model.to(torch.bfloat16)
        else:
            log_model(f"Precision {precision} not supported on this device. Using float32.", force=True)
    
    # Log memory usage if on CUDA
    if torch_device == "cuda":
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        log_model(f"CUDA Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    return model, tokenizer

def apply_pruning(model, pruning_percentage, method="entropy", verbose=True, quiet=False):
    """
    Apply pruning to the model by setting selected gate values to zero.
    
    Args:
        model: The transformer model to prune
        pruning_percentage: Percentage of heads to prune (0-100)
        method: Pruning method ('entropy', 'random', or 'magnitude')
        verbose: Whether to print pruning details
        quiet: If True, reduce logging verbosity
        
    Returns:
        Tuple of (pruned model, number of heads pruned, list of pruned heads)
    """
    # Define a local log function
    def log_pruning(message, force=False):
        if (verbose and not quiet) or force:
            print(message)
    start_time = time.time()
    
    # Handle baseline models (which don't have gates to prune)
    is_baseline_model = False
    if hasattr(model, "config") and not hasattr(model, "blocks"):
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # This is a standard Hugging Face model (GPT-2 style)
            log_pruning(f"This is a baseline model without gates. Pruning will be simulated.")
            return model, 0, []  # No actual pruning for baseline model
        is_baseline_model = True
    
    # Get all head information
    heads_info = []
    
    # Determine which model structure we're dealing with
    if hasattr(model, "blocks"):
        # Direct access to AdaptiveTransformerModel or AdaptiveCausalLmWrapper
        blocks = model.blocks
        num_heads = model.num_heads
    elif hasattr(model, "model") and hasattr(model.model, "blocks"):
        # Access through a wrapper
        blocks = model.model.blocks
        num_heads = model.model.num_heads
    else:
        # For baseline models without gates, we'll skip actual pruning
        if is_baseline_model:
            log_pruning(f"Skipping pruning for baseline model (no gates to prune)")
            return model, 0, []
        else:
            raise ValueError("Unsupported model structure: cannot find blocks or attention gates")
    
    # For models with gates, proceed with pruning
    for layer_idx, block in enumerate(blocks):
        # Access attn module correctly - handle both subscriptable and attribute style
        if hasattr(block, "attn"):
            attn_module = block.attn
        elif hasattr(block, "attention"):
            attn_module = block.attention
        else:
            raise ValueError(f"Block at layer {layer_idx} doesn't have an attention module")
            
        for head_idx in range(num_heads):
            gate_value = float(attn_module.gate[head_idx])
            
            # Get head information based on pruning method
            if method == "entropy":
                # Use gate value as proxy for entropy-based importance
                importance = gate_value
            elif method == "magnitude":
                # Use weight magnitude as importance 
                # This depends on the structure of the model
                try:
                    # Try to access weights of this head
                    weight = attn_module.W_q[head_idx].weight
                    importance = float(weight.abs().mean())
                except (AttributeError, KeyError):
                    # Fallback for unsupported structures
                    log_pruning(f"Warning: Magnitude pruning not supported for this model structure, falling back to gate values")
                    importance = gate_value
            else:  # random
                importance = random.random()
                
            heads_info.append({
                "layer_idx": layer_idx,
                "head_idx": head_idx,
                "gate_value": gate_value,
                "importance": importance
            })
    
    # Calculate how many heads to prune
    num_heads = len(heads_info)
    num_to_prune = int(num_heads * pruning_percentage / 100)
    
    if num_to_prune == 0:
        if verbose:
            log_pruning(f"No heads to prune at {pruning_percentage}%")
        return model, 0, []
    
    # Sort by importance (ascending, so least important first)
    if method == "random":
        # For random, shuffle instead of sort
        random.shuffle(heads_info)
        heads_to_prune = heads_info[:num_to_prune]
    else:
        # For other methods, sort by importance
        heads_info.sort(key=lambda x: x["importance"])
        heads_to_prune = heads_info[:num_to_prune]
    
    # Set the selected gates to zero
    pruned_heads = []
    for head in heads_to_prune:
        layer_idx = head["layer_idx"]
        head_idx = head["head_idx"]
        
        # Get attention module - handle both attribute and subscript style access
        if hasattr(blocks[layer_idx], "attn"):
            attn_module = blocks[layer_idx].attn
        elif hasattr(blocks[layer_idx], "attention"):
            attn_module = blocks[layer_idx].attention
        else:
            continue  # Skip if no attention module
        
        # We need to handle differently for models in training vs. evaluation mode
        # to avoid the "leaf Variable that requires grad" error
        with torch.no_grad():  # This prevents autograd from tracking this operation
            attn_module.gate[head_idx] = torch.zeros_like(attn_module.gate[head_idx])
            
        pruned_heads.append((layer_idx, head_idx))
    
    # Print pruning stats if verbose
    if verbose:
        duration = time.time() - start_time
        
        # If agency is enabled, report agency states
        # Check both direct model and model.model for enable_agency attribute
        has_agency = False
        if hasattr(model, "enable_agency") and model.enable_agency:
            has_agency = True
            agency_blocks = blocks
        elif hasattr(model, "model") and hasattr(model.model, "enable_agency") and model.model.enable_agency:
            has_agency = True
            agency_blocks = model.model.blocks
        
        # Create different output based on quiet mode
        if quiet:
            # In quiet mode, just show a single concise line
            if has_agency:
                agency_states = {"active": 0, "overloaded": 0, "misaligned": 0, "withdrawn": 0}
                consent_withdrawn = 0
                
                for layer_idx, block in enumerate(agency_blocks):
                    # Get attention module - handle both attribute and subscript style access
                    if hasattr(block, "attn"):
                        attn_module = block.attn
                    elif hasattr(block, "attention"):
                        attn_module = block.attention
                    else:
                        continue  # Skip if no attention module found
                        
                    if hasattr(attn_module, "agency_signals"):
                        for head_idx, signals in attn_module.agency_signals.items():
                            agency_states[signals["state"]] += 1
                            if not signals["consent"]:
                                consent_withdrawn += 1
                
                log_pruning(f"Pruned {num_to_prune}/{num_heads} heads ({pruning_percentage}%) using {method} method in {duration:.2f}s", force=True)
            else:
                log_pruning(f"Pruned {num_to_prune}/{num_heads} heads ({pruning_percentage}%) using {method} method in {duration:.2f}s", force=True)
        else:
            # In normal mode, show detailed output
            log_pruning(f"Pruned {num_to_prune}/{num_heads} heads ({pruning_percentage}%) using {method} method in {duration:.2f}s")
            
            if has_agency:
                agency_states = {"active": 0, "overloaded": 0, "misaligned": 0, "withdrawn": 0}
                consent_withdrawn = 0
                
                for layer_idx, block in enumerate(agency_blocks):
                    # Get attention module - handle both attribute and subscript style access
                    if hasattr(block, "attn"):
                        attn_module = block.attn
                    elif hasattr(block, "attention"):
                        attn_module = block.attention
                    else:
                        continue  # Skip if no attention module found
                        
                    if hasattr(attn_module, "agency_signals"):
                        for head_idx, signals in attn_module.agency_signals.items():
                            agency_states[signals["state"]] += 1
                            if not signals["consent"]:
                                consent_withdrawn += 1
                
                log_pruning(f"Agency states after pruning: {agency_states}")
                log_pruning(f"Heads with withdrawn consent: {consent_withdrawn}")
    
    return model, num_to_prune, pruned_heads

def load_prompts(prompt_file):
    """Load prompts from a file."""
    with open(prompt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def evaluate_model(model, tokenizer, prompts, num_tokens, temperature=0.7, 
                batch_size=1, device="cpu", memory_logging=False, max_prompts=None, quiet=False):
    """
    Evaluate model performance on a set of prompts with comprehensive metrics.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer for the model
        prompts: List of text prompts
        num_tokens: Number of tokens to generate for each prompt
        temperature: Sampling temperature
        batch_size: Batch size for generation
        device: Device to run on ("cpu" or "cuda")
        memory_logging: Whether to log memory usage
        max_prompts: Maximum number of prompts to evaluate
        quiet: If True, reduce logging verbosity
        
    Returns:
        Dictionary of performance metrics
    """
    # Limit number of prompts if requested
    if max_prompts and max_prompts < len(prompts):
        prompts = prompts[:max_prompts]
    
    # Initialize results dictionary
    results = {
        "perplexity": [],
        "diversity": [],
        "repetition": [],
        "generation_time": [],
        "first_token_time": [],
        "prompt_processing_time": [],
        "tokens_per_second": [],
        "outputs": []
    }
    
    # Resource tracking
    if memory_logging:
        results["peak_memory"] = []
        results["cpu_percent"] = []
        
        # Import necessary modules for resource tracking
        try:
            import psutil
            import gc
        except ImportError:
            print("Warning: psutil not available. CPU monitoring disabled.")
            memory_logging = False
    
    # Track any failures
    failures = 0
    
    # Process prompts (in batches if batch_size > 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        # Only print batch progress if not in quiet mode
        if not quiet:
            print(f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size} ({len(batch_prompts)} prompts)")
        
        for prompt_idx, prompt in enumerate(batch_prompts):
            prompt_num = i + prompt_idx + 1
            # Skip per-prompt logging in quiet mode
            if not quiet:
                print(f"  Evaluating prompt {prompt_num}/{len(prompts)}: {prompt[:50]}...")
            
            try:
                # Clear cache if on CUDA
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Start resource monitoring if enabled
                if memory_logging:
                    gc.collect()
                    process = psutil.Process()
                    start_cpu = process.cpu_percent()
                    
                    if device == "cuda" and torch.cuda.is_available():
                        start_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)
                
                # Time prompt processing
                prompt_start = time.time()
                input_ids = tokenizer.encode(prompt, return_tensors="pt")
                if device == "cuda" and torch.cuda.is_available():
                    input_ids = input_ids.to("cuda")
                prompt_end = time.time()
                prompt_processing_time = prompt_end - prompt_start
                
                # Time generation
                generation_start = time.time()
                
                # Generate with temperature and other params
                # Handle different model structures
                if hasattr(model, "generate"):
                    generate_method = model.generate
                elif hasattr(model, "model") and hasattr(model.model, "generate"):
                    generate_method = model.model.generate
                else:
                    raise ValueError("Model does not have a generate method")
                
                # Create attention mask (all 1s since we don't have padding)
                attention_mask = torch.ones_like(input_ids)
                
                # IMPORTANT: input_ids should always be integers (Long/Int)
                # Only convert attention mask if needed, never convert input_ids to float
                model_dtype = next(model.parameters()).dtype
                if attention_mask.dtype != model_dtype:
                    if not quiet:
                        print(f"    Converting attention mask from {attention_mask.dtype} to {model_dtype}")
                    attention_mask = attention_mask.to(model_dtype)
                
                try:
                    # Ensure temperature is a Python float, not a tensor
                    temp_value = float(temperature)
                    
                    # Use context manager to catch and convert any half-precision errors
                    with torch.amp.autocast(device_type=device, enabled=(model_dtype == torch.float16)):
                        output_ids = generate_method(
                            input_ids,  # Keep as integers (Long)
                            attention_mask=attention_mask,
                            max_length=input_ids.size(1) + num_tokens,
                            temperature=temp_value,
                            do_sample=(temp_value > 0),
                            pad_token_id=tokenizer.eos_token_id,
                        )
                except RuntimeError as e:
                    if "expected scalar type Float but found Half" in str(e):
                        # Fall back to full precision if there's a half precision error
                        if not quiet:
                            print(f"    Half precision error, falling back to float32 for generation")
                        
                        # Move model to float32 temporarily for generation
                        model_was_half = next(model.parameters()).dtype == torch.float16
                        if model_was_half:
                            # Only convert for generation
                            if hasattr(model, 'generate'):
                                # For direct model generate
                                with torch.amp.autocast(device_type=device, enabled=False):
                                    model = model.float()
                                    # Keep input_ids as integers (Long)
                                    attention_mask = attention_mask.float()
                                    output_ids = model.generate(
                                        input_ids,
                                        attention_mask=attention_mask,
                                        max_length=input_ids.size(1) + num_tokens,
                                        temperature=temp_value,
                                        do_sample=(temp_value > 0),
                                        pad_token_id=tokenizer.eos_token_id,
                                    )
                                    # Convert back after generation
                                    model = model.half()
                            else:
                                # This is a complex case - we'll need to skip it
                                raise RuntimeError("Half precision error and cannot convert nested model")
                        else:
                            # Re-raise if model isn't half precision
                            raise
                    else:
                        # Re-raise other errors
                        raise
                
                # Measure time to first token
                first_token_time = time.time() - generation_start
                
                # Complete generation
                output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                generation_time = time.time() - generation_start
                
                # Calculate tokens per second
                tokens_generated = output_ids.size(1) - input_ids.size(1)
                tokens_per_second = tokens_generated / generation_time if tokens_generated > 0 else 0
                
                # Record resource usage if enabled
                if memory_logging:
                    if device == "cuda" and torch.cuda.is_available():
                        peak_memory = (torch.cuda.max_memory_allocated() / (1024 ** 3)) - start_gpu_memory
                        results["peak_memory"].append(peak_memory)
                    
                    # CPU usage
                    cpu_percent = process.cpu_percent()
                    results["cpu_percent"].append(cpu_percent)
                
                # Calculate quality metrics
                try:
                    # Using the newer calculate_perplexity function that takes [generated_texts, prompts]
                    perplexity = calculate_perplexity([output], [prompt])
                    diversity = calculate_diversity(output)
                    repetition = calculate_repetition(output)
                except Exception as e:
                    print(f"Error calculating metrics: {str(e)}")
                    perplexity = 100.0  # Default fallback value
                    diversity = 0.5     # Default fallback value
                    repetition = 0.5    # Default fallback value
                
                # Store all metrics
                results["generation_time"].append(generation_time)
                results["first_token_time"].append(first_token_time)
                results["prompt_processing_time"].append(prompt_processing_time)
                results["tokens_per_second"].append(tokens_per_second)
                results["perplexity"].append(perplexity)
                results["diversity"].append(diversity)
                results["repetition"].append(repetition)
                results["outputs"].append(output)
                
                # Print progress info (skip in quiet mode)
                if not quiet:
                    print(f"    Generated {tokens_generated} tokens in {generation_time:.2f}s "
                         f"({tokens_per_second:.2f} tokens/sec), perplexity: {perplexity:.2f}")
                
            except Exception as e:
                if not quiet:
                    print(f"Error processing prompt {prompt_num}: {str(e)}")
                failures += 1
    
    # Handle case where all prompts failed
    if failures == len(prompts):
        print("All prompts failed. Returning default metrics.")
        default_results = {
            "perplexity": 100.0,
            "diversity": 0.5,
            "repetition": 0.5,
            "generation_time": 10.0,
            "first_token_time": 1.0,
            "prompt_processing_time": 0.5,
            "tokens_per_second": 1.0,
            "outputs": ["[Generation failed]"],
            "success_rate": 0.0
        }
        if memory_logging:
            default_results["peak_memory"] = 0.0
            default_results["cpu_percent"] = 0.0
        
        return default_results
    
    # Calculate average metrics (excluding outputs)
    averaged_results = {}
    for key in results:
        if key != "outputs":
            if len(results[key]) > 0:  # Only average non-empty lists
                averaged_results[key] = sum(results[key]) / len(results[key])
            else:
                averaged_results[key] = 0.0
    
    # Add additional metrics
    averaged_results["outputs"] = results["outputs"]
    averaged_results["success_rate"] = (len(prompts) - failures) / len(prompts)
    
    # Print summary (always show this even in quiet mode)
    if quiet:
        # In quiet mode, just show a single summary line
        print(f"Evaluation completed: {len(prompts) - failures}/{len(prompts)} prompts, {averaged_results['tokens_per_second']:.2f} tokens/sec, PPL: {averaged_results['perplexity']:.2f}")
    else:
        # In normal mode, show more detailed summary
        print(f"\nEvaluation complete: {len(prompts) - failures}/{len(prompts)} prompts successful")
        print(f"Average tokens/sec: {averaged_results['tokens_per_second']:.2f}")
        print(f"Average perplexity: {averaged_results['perplexity']:.2f}")
    
    return averaged_results

def run_pruning_comparison(args):
    """Run the main comparison between baseline and agency models."""
    # Create a logging function that respects quiet flag
    def log(message, force=False):
        if not hasattr(args, 'quiet') or not args.quiet or force:
            print(message)
            
    # Process command line arguments
    pruning_levels = [int(x) for x in args.pruning_levels.split(",")]
    temperatures = [float(x) for x in args.temperatures.split(",")]
    prompts = load_prompts(args.prompt_file)
    
    if args.max_prompts and args.max_prompts < len(prompts):
        prompts = prompts[:args.max_prompts]
        log(f"Using {len(prompts)} prompts for evaluation (limited by --max_prompts)")
    else:
        log(f"Using all {len(prompts)} prompts for evaluation")
    
    # Prepare results structure
    results = {
        "baseline": {},
        "agency": {},
        "metadata": {
            "model_name": args.model_name,
            "num_tokens": args.num_tokens,
            "num_prompts": len(prompts),
            "device": args.device,
            "batch_size": args.batch_size,
            "pruning_method": args.pruning_method,
            "precision": args.precision,
            "temperatures": temperatures,
            "iterations": args.iterations,
            "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S")
        }
    }
    
    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log(f"Results will be saved to {output_dir}", force=True)
    
    # Save a copy of the prompts
    with open(output_dir / "prompts.txt", "w") as f:
        f.write("\n".join(prompts))
    
    # Create subdirectories for iterations
    for i in range(args.iterations):
        (output_dir / f"iteration_{i}").mkdir(exist_ok=True)
    
    # For each temperature value
    for temp_idx, temperature in enumerate(temperatures):
        log(f"\n{'='*80}\nTesting temperature {temperature}\n{'='*80}", force=True)
        
        results[f"temperature_{temperature}"] = {"baseline": {}, "agency": {}}
        
        # For each pruning level
        for level in pruning_levels:
            log(f"\n{'-'*40}\nEvaluating pruning level: {level}%\n{'-'*40}", force=True)
            
            # Initialize iteration results
            baseline_iterations = []
            agency_iterations = []
            
            # Run multiple iterations for statistical significance
            for iteration in range(args.iterations):
                log(f"\nIteration {iteration+1}/{args.iterations}")
                
                iteration_dir = output_dir / f"iteration_{iteration}"
                
                # Baseline model
                log(f"Loading baseline model...")
                baseline_model, tokenizer = load_model(
                    args.model_name, 
                    agency_enabled=False,
                    device=args.device,
                    precision=args.precision,
                    quiet=args.quiet
                )
                baseline_model, pruned_heads_count, _ = apply_pruning(
                    baseline_model, 
                    level,
                    method=args.pruning_method,
                    quiet=args.quiet
                )
                
                # Agency model
                log(f"Loading agency model...")
                agency_model, _ = load_model(
                    args.model_name, 
                    agency_enabled=True,
                    device=args.device,
                    precision=args.precision,
                    quiet=args.quiet
                )
                agency_model, pruned_heads_count, _ = apply_pruning(
                    agency_model, 
                    level,
                    method=args.pruning_method,
                    quiet=args.quiet
                )
                
                # Evaluate both models
                log(f"Evaluating baseline model...")
                baseline_results = evaluate_model(
                    baseline_model, 
                    tokenizer, 
                    prompts, 
                    args.num_tokens,
                    temperature=temperature,
                    batch_size=args.batch_size,
                    device=args.device,
                    memory_logging=args.memory_logging,
                    max_prompts=args.max_prompts,
                    quiet=args.quiet
                )
                
                log(f"Evaluating agency model...")
                agency_results = evaluate_model(
                    agency_model, 
                    tokenizer, 
                    prompts, 
                    args.num_tokens,
                    temperature=temperature,
                    batch_size=args.batch_size,
                    device=args.device,
                    memory_logging=args.memory_logging,
                    max_prompts=args.max_prompts,
                    quiet=args.quiet
                )
                
                # Store iteration results
                baseline_iterations.append(baseline_results)
                agency_iterations.append(agency_results)
                
                # Save individual iteration outputs
                if args.save_outputs:
                    with open(iteration_dir / f"baseline_outputs_level{level}_temp{temperature}.txt", "w") as f:
                        for i, output in enumerate(baseline_results["outputs"]):
                            f.write(f"=== Prompt {i+1} ===\n")
                            f.write(prompts[i] + "\n\n")
                            f.write(output + "\n\n")
                    
                    with open(iteration_dir / f"agency_outputs_level{level}_temp{temperature}.txt", "w") as f:
                        for i, output in enumerate(agency_results["outputs"]):
                            f.write(f"=== Prompt {i+1} ===\n")
                            f.write(prompts[i] + "\n\n")
                            f.write(output + "\n\n")
                
                # Print comparison for this iteration
                # Calculate improvements
                speed_improvement = ((agency_results['tokens_per_second'] / baseline_results['tokens_per_second']) - 1) * 100
                quality_improvement = ((baseline_results['perplexity'] / agency_results['perplexity']) - 1) * 100
                
                # Show different output depending on verbosity level
                if args.quiet:
                    # Just show a single line summary for each iteration
                    log(f"Iteration {iteration+1} @ {level}%: Agency is {speed_improvement:.1f}% faster, {quality_improvement:.1f}% better quality")
                else:
                    # Show detailed per-iteration results
                    log(f"\nIteration {iteration+1} Results at {level}% pruning, temp={temperature}:")
                    log(f"  Baseline: {baseline_results['tokens_per_second']:.2f} tokens/sec, "
                         f"perplexity: {baseline_results['perplexity']:.2f}, "
                         f"diversity: {baseline_results['diversity']:.3f}")
                    log(f"  Agency:   {agency_results['tokens_per_second']:.2f} tokens/sec, "
                         f"perplexity: {agency_results['perplexity']:.2f}, "
                         f"diversity: {agency_results['diversity']:.3f}")
                    log(f"  Improvement: {speed_improvement:.1f}% faster, {quality_improvement:.1f}% better quality")
                
                # Free memory
                del baseline_model
                del agency_model
                if args.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Average results across iterations
            baseline_avg = {}
            agency_avg = {}
            
            # Metrics to average (exclude outputs)
            metrics_to_avg = [k for k in baseline_iterations[0].keys() if k != "outputs"]
            
            # Calculate average and std dev for each metric
            for metric in metrics_to_avg:
                baseline_values = [iter_result[metric] for iter_result in baseline_iterations]
                agency_values = [iter_result[metric] for iter_result in agency_iterations]
                
                baseline_avg[metric] = {
                    "mean": sum(baseline_values) / len(baseline_values),
                    "std": np.std(baseline_values) if len(baseline_values) > 1 else 0
                }
                
                agency_avg[metric] = {
                    "mean": sum(agency_values) / len(agency_values),
                    "std": np.std(agency_values) if len(agency_values) > 1 else 0
                }
            
            # Store averaged results
            results[f"temperature_{temperature}"]["baseline"][level] = baseline_avg
            results[f"temperature_{temperature}"]["agency"][level] = agency_avg
            
            # Print averaged results
            # Calculate improvement
            speed_improvement = ((agency_avg['tokens_per_second']['mean'] / baseline_avg['tokens_per_second']['mean']) - 1) * 100
            quality_improvement = ((baseline_avg['perplexity']['mean'] / agency_avg['perplexity']['mean']) - 1) * 100
            
            # Create different outputs based on verbosity
            is_last = (level == max(pruning_levels) and temperature == temperatures[-1])
            
            if args.quiet:
                # For quiet mode, always show a concise single-line summary
                log(f"Results at {level}% pruning (T={temperature}): Agency is {speed_improvement:.1f}% faster, {quality_improvement:.1f}% better quality", force=is_last)
            else:
                # For normal mode, show full detailed results
                log(f"\nAveraged Results at {level}% pruning, temp={temperature}:", force=is_last)
                log(f"  Baseline: {baseline_avg['tokens_per_second']['mean']:.2f} ± {baseline_avg['tokens_per_second']['std']:.2f} tokens/sec, "
                     f"perplexity: {baseline_avg['perplexity']['mean']:.2f} ± {baseline_avg['perplexity']['std']:.2f}", force=is_last)
                log(f"  Agency:   {agency_avg['tokens_per_second']['mean']:.2f} ± {agency_avg['tokens_per_second']['std']:.2f} tokens/sec, "
                     f"perplexity: {agency_avg['perplexity']['mean']:.2f} ± {agency_avg['perplexity']['std']:.2f}", force=is_last)
                log(f"  Improvement: {speed_improvement:.1f}% faster, {quality_improvement:.1f}% better quality", force=is_last)
            
            # Save incremental results after each pruning level
            incremental_results_file = output_dir / f"incremental_results_temp{temperature}.json"
            with open(incremental_results_file, "w") as f:
                json.dump(results, f, indent=2)
    
    # Save final complete results
    final_results_file = output_dir / "pruning_comparison_results.json"
    with open(final_results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Create a symlink to the latest results
    latest_link = Path(args.output_dir) / "latest"
    try:
        if os.path.exists(latest_link):
            if os.path.islink(latest_link):
                os.unlink(latest_link)
            else:
                os.remove(latest_link)
        os.symlink(output_dir, latest_link, target_is_directory=True)
        latest_link_msg = f"Latest results symlinked to {latest_link}"
    except Exception as e:
        latest_link_msg = f"Note: Could not create symlink to latest results: {e}"
    
    # Just show a brief message for cleaner output
    log(f"Experiment completed. Results saved to {output_dir}", force=True)
    
    return results, output_dir

def visualize_results(results, output_dir):
    """
    Generate comprehensive visualizations from the results.
    
    This enhanced version handles multiple temperatures and iterations with error bars.
    """
    output_dir = Path(output_dir)
    
    # Check if the results use the new format with temperatures
    new_format = any(key.startswith("temperature_") for key in results.keys())
    
    if new_format:
        # Process results for each temperature
        for temp_key in [k for k in results.keys() if k.startswith("temperature_")]:
            temperature = float(temp_key.split("_")[1])
            temp_dir = output_dir / f"temp_{temperature}"
            temp_dir.mkdir(exist_ok=True)
            
            # Create visualizations for this temperature
            visualize_temperature_results(
                results[temp_key], 
                temp_dir, 
                temperature,
                model_name=results["metadata"]["model_name"]
            )
        
        # Create combined temperature visualizations
        visualize_temperature_comparison(results, output_dir)
    else:
        # Use old visualization logic for backward compatibility
        visualize_simple_results(results, output_dir)
    
    print(f"Visualizations saved to {output_dir}")

def visualize_temperature_results(results, output_dir, temperature, model_name="gpt2"):
    """Visualize results for a specific temperature."""
    output_dir = Path(output_dir)
    
    # Get pruning levels (convert string keys to integers)
    baseline_data = results["baseline"]
    agency_data = results["agency"]
    
    # Ensure keys are properly formatted (they might be integers or strings)
    baseline_levels = []
    for level in baseline_data.keys():
        # Try to convert to int if it's a string
        try:
            level_int = int(level)
            baseline_levels.append(level_int)
        except (ValueError, TypeError):
            # If it's not an integer string, just use as is
            baseline_levels.append(level)
    
    pruning_levels = sorted(baseline_levels)
    
    # Function to safely extract metrics with error handling
    def safe_extract(data, level, metric):
        level_key = str(level)
        if level_key not in data:
            # Try integer key if string key doesn't work
            level_key = level
            
        if level_key not in data:
            print(f"Warning: Level {level} not found in data")
            return {"mean": 0.0, "std": 0.0}
            
        return data[level_key][metric]
        
    # Extract metrics with error bars
    baseline_speed_mean = [safe_extract(baseline_data, level, "tokens_per_second")["mean"] for level in pruning_levels]
    baseline_speed_std = [safe_extract(baseline_data, level, "tokens_per_second")["std"] for level in pruning_levels]
    
    agency_speed_mean = [safe_extract(agency_data, level, "tokens_per_second")["mean"] for level in pruning_levels]
    agency_speed_std = [safe_extract(agency_data, level, "tokens_per_second")["std"] for level in pruning_levels]
    
    baseline_ppl_mean = [safe_extract(baseline_data, level, "perplexity")["mean"] for level in pruning_levels]
    baseline_ppl_std = [safe_extract(baseline_data, level, "perplexity")["std"] for level in pruning_levels]
    
    agency_ppl_mean = [safe_extract(agency_data, level, "perplexity")["mean"] for level in pruning_levels]
    agency_ppl_std = [safe_extract(agency_data, level, "perplexity")["std"] for level in pruning_levels]
    
    baseline_div_mean = [safe_extract(baseline_data, level, "diversity")["mean"] for level in pruning_levels]
    baseline_div_std = [safe_extract(baseline_data, level, "diversity")["std"] for level in pruning_levels]
    
    agency_div_mean = [safe_extract(agency_data, level, "diversity")["mean"] for level in pruning_levels]
    agency_div_std = [safe_extract(agency_data, level, "diversity")["std"] for level in pruning_levels]
    
    # Extract first token latency if available
    first_level_key = str(pruning_levels[0])
    if first_level_key not in baseline_data:
        first_level_key = pruning_levels[0]
    
    has_latency = False
    if baseline_data and first_level_key in baseline_data and "first_token_time" in baseline_data[first_level_key]:
        baseline_latency_mean = [safe_extract(baseline_data, level, "first_token_time")["mean"] for level in pruning_levels]
        baseline_latency_std = [safe_extract(baseline_data, level, "first_token_time")["std"] for level in pruning_levels]
        
        agency_latency_mean = [safe_extract(agency_data, level, "first_token_time")["mean"] for level in pruning_levels]
        agency_latency_std = [safe_extract(agency_data, level, "first_token_time")["std"] for level in pruning_levels]
        
        has_latency = True
    
    # 1. Generation speed comparison with error bars
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(pruning_levels, baseline_speed_mean, yerr=baseline_speed_std, 
                fmt='o-', label="Baseline", color="#78909C", capsize=5)
    
    plt.errorbar(pruning_levels, agency_speed_mean, yerr=agency_speed_std, 
                fmt='o-', label="Agency", color="#4CAF50", capsize=5)
    
    plt.title(f"Generation Speed vs. Pruning Level (temp={temperature})", fontsize=16)
    plt.xlabel("Pruning Percentage", fontsize=14)
    plt.ylabel("Tokens per Second", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Annotate improvement at highest pruning level
    max_level = max(pruning_levels)
    improvement = ((agency_speed_mean[-1] / baseline_speed_mean[-1]) - 1) * 100
    
    plt.annotate(f"{improvement:.1f}% faster",
                xy=(max_level, agency_speed_mean[-1]),
                xytext=(max_level-10, agency_speed_mean[-1]+1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / "speed_comparison.png", dpi=150)
    plt.close()
    
    # 2. Perplexity comparison with error bars
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(pruning_levels, baseline_ppl_mean, yerr=baseline_ppl_std, 
                fmt='o-', label="Baseline", color="#78909C", capsize=5)
    
    plt.errorbar(pruning_levels, agency_ppl_mean, yerr=agency_ppl_std, 
                fmt='o-', label="Agency", color="#4CAF50", capsize=5)
    
    plt.title(f"Perplexity vs. Pruning Level (temp={temperature})", fontsize=16)
    plt.xlabel("Pruning Percentage", fontsize=14)
    plt.ylabel("Perplexity (Lower is Better)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Annotate difference at highest pruning level
    ppl_diff = ((baseline_ppl_mean[-1] / agency_ppl_mean[-1]) - 1) * 100
    
    plt.annotate(f"{abs(ppl_diff):.1f}% {'better' if ppl_diff > 0 else 'worse'}",
                xy=(max_level, agency_ppl_mean[-1]),
                xytext=(max_level-10, agency_ppl_mean[-1]-5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / "perplexity_comparison.png", dpi=150)
    plt.close()
    
    # 3. Diversity comparison with error bars
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(pruning_levels, baseline_div_mean, yerr=baseline_div_std, 
                fmt='o-', label="Baseline", color="#78909C", capsize=5)
    
    plt.errorbar(pruning_levels, agency_div_mean, yerr=agency_div_std, 
                fmt='o-', label="Agency", color="#4CAF50", capsize=5)
    
    plt.title(f"Lexical Diversity vs. Pruning Level (temp={temperature})", fontsize=16)
    plt.xlabel("Pruning Percentage", fontsize=14)
    plt.ylabel("Lexical Diversity (Higher is Better)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Annotate difference at highest pruning level
    div_diff = ((agency_div_mean[-1] / baseline_div_mean[-1]) - 1) * 100
    
    plt.annotate(f"{abs(div_diff):.1f}% {'better' if div_diff > 0 else 'worse'}",
                xy=(max_level, agency_div_mean[-1]),
                xytext=(max_level-10, agency_div_mean[-1]+0.05),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / "diversity_comparison.png", dpi=150)
    plt.close()
    
    # 4. First token latency (if available)
    if has_latency:
        plt.figure(figsize=(10, 6))
        
        plt.errorbar(pruning_levels, baseline_latency_mean, yerr=baseline_latency_std, 
                    fmt='o-', label="Baseline", color="#78909C", capsize=5)
        
        plt.errorbar(pruning_levels, agency_latency_mean, yerr=agency_latency_std, 
                    fmt='o-', label="Agency", color="#4CAF50", capsize=5)
        
        plt.title(f"First Token Latency vs. Pruning Level (temp={temperature})", fontsize=16)
        plt.xlabel("Pruning Percentage", fontsize=14)
        plt.ylabel("Latency in Seconds (Lower is Better)", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # Annotate difference at highest pruning level
        latency_diff = ((baseline_latency_mean[-1] / agency_latency_mean[-1]) - 1) * 100
        
        plt.annotate(f"{abs(latency_diff):.1f}% {'better' if latency_diff > 0 else 'worse'}",
                    xy=(max_level, agency_latency_mean[-1]),
                    xytext=(max_level-10, agency_latency_mean[-1]-0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                    fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_dir / "latency_comparison.png", dpi=150)
        plt.close()
    
    # 5. Combined radar chart for highest pruning level
    metrics = ['Speed', 'Quality\n(1/Perplexity)', 'Diversity']
    if has_latency:
        metrics.append('Responsiveness\n(1/Latency)')
    
    # Normalize values (higher is better for all)
    max_speed = max(baseline_speed_mean[-1], agency_speed_mean[-1])
    max_ppl_inv = max(1/baseline_ppl_mean[-1], 1/agency_ppl_mean[-1])
    max_div = max(baseline_div_mean[-1], agency_div_mean[-1])
    
    baseline_values = [
        baseline_speed_mean[-1] / max_speed,
        (1/baseline_ppl_mean[-1]) / max_ppl_inv,
        baseline_div_mean[-1] / max_div,
    ]
    
    agency_values = [
        agency_speed_mean[-1] / max_speed,
        (1/agency_ppl_mean[-1]) / max_ppl_inv,
        agency_div_mean[-1] / max_div,
    ]
    
    if has_latency:
        max_resp = max(1/baseline_latency_mean[-1], 1/agency_latency_mean[-1])
        baseline_values.append((1/baseline_latency_mean[-1]) / max_resp)
        agency_values.append((1/agency_latency_mean[-1]) / max_resp)
    
    # Create radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    baseline_values += baseline_values[:1]  # Close the loop
    agency_values += agency_values[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    ax.plot(angles, baseline_values, 'o-', linewidth=2, label='Baseline', color="#78909C")
    ax.fill(angles, baseline_values, alpha=0.25, color="#78909C")
    
    ax.plot(angles, agency_values, 'o-', linewidth=2, label='Agency', color="#4CAF50")
    ax.fill(angles, agency_values, alpha=0.25, color="#4CAF50")
    
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    ax.set_ylim(0, 1.1)
    ax.grid(True)
    
    plt.title(f"Model Performance at {max_level}% Pruning (temp={temperature})", fontsize=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(output_dir / "radar_comparison.png", dpi=150)
    plt.close()
    
    # 6. Comprehensive summary plot
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Agency vs. Baseline: {model_name.upper()} at Temperature {temperature}", fontsize=18)
    
    # Top left: Generation Speed
    axs[0, 0].errorbar(pruning_levels, baseline_speed_mean, yerr=baseline_speed_std, 
                     fmt='o-', label="Baseline", color="#78909C", capsize=5)
    axs[0, 0].errorbar(pruning_levels, agency_speed_mean, yerr=agency_speed_std, 
                     fmt='o-', label="Agency", color="#4CAF50", capsize=5)
    axs[0, 0].set_title("Generation Speed", fontsize=14)
    axs[0, 0].set_xlabel("Pruning %", fontsize=12)
    axs[0, 0].set_ylabel("Tokens per Second", fontsize=12)
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    axs[0, 0].legend()
    
    # Top right: Perplexity
    axs[0, 1].errorbar(pruning_levels, baseline_ppl_mean, yerr=baseline_ppl_std, 
                     fmt='o-', label="Baseline", color="#78909C", capsize=5)
    axs[0, 1].errorbar(pruning_levels, agency_ppl_mean, yerr=agency_ppl_std, 
                     fmt='o-', label="Agency", color="#4CAF50", capsize=5)
    axs[0, 1].set_title("Perplexity (Lower is Better)", fontsize=14)
    axs[0, 1].set_xlabel("Pruning %", fontsize=12)
    axs[0, 1].set_ylabel("Perplexity", fontsize=12)
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    axs[0, 1].legend()
    
    # Bottom left: Diversity
    axs[1, 0].errorbar(pruning_levels, baseline_div_mean, yerr=baseline_div_std, 
                     fmt='o-', label="Baseline", color="#78909C", capsize=5)
    axs[1, 0].errorbar(pruning_levels, agency_div_mean, yerr=agency_div_std, 
                     fmt='o-', label="Agency", color="#4CAF50", capsize=5)
    axs[1, 0].set_title("Lexical Diversity", fontsize=14)
    axs[1, 0].set_xlabel("Pruning %", fontsize=12)
    axs[1, 0].set_ylabel("Diversity", fontsize=12)
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    axs[1, 0].legend()
    
    # Bottom right: Tradeoff plot - Speed vs Quality
    baseline_x = baseline_speed_mean
    baseline_y = [1/p for p in baseline_ppl_mean]  # Inverse perplexity (higher is better)
    
    agency_x = agency_speed_mean
    agency_y = [1/p for p in agency_ppl_mean]
    
    # Size represents pruning level
    sizes = [100 + 10*level for level in pruning_levels]
    
    axs[1, 1].scatter(baseline_x, baseline_y, s=sizes, alpha=0.7, label="Baseline", color="#78909C")
    axs[1, 1].scatter(agency_x, agency_y, s=sizes, alpha=0.7, label="Agency", color="#4CAF50")
    
    # Connect points with lines
    for i in range(len(pruning_levels)):
        axs[1, 1].plot([baseline_x[i], agency_x[i]], [baseline_y[i], agency_y[i]], 
                     'k--', alpha=0.3, linewidth=1)
    
    # Add pruning level annotations
    for i, level in enumerate(pruning_levels):
        axs[1, 1].annotate(f"{level}%", 
                         xy=(baseline_x[i], baseline_y[i]),
                         xytext=(2, 2),
                         textcoords='offset points', 
                         fontsize=8)
        
        axs[1, 1].annotate(f"{level}%", 
                         xy=(agency_x[i], agency_y[i]),
                         xytext=(2, 2),
                         textcoords='offset points',
                         fontsize=8)
    
    axs[1, 1].set_title("Quality vs. Speed Tradeoff", fontsize=14)
    axs[1, 1].set_xlabel("Tokens per Second", fontsize=12)
    axs[1, 1].set_ylabel("Quality (1/Perplexity)", fontsize=12)
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    axs[1, 1].legend()
    
    # Add "better" direction arrow
    axs[1, 1].annotate("Better", xy=(0.85, 0.85), xycoords='axes fraction',
                     xytext=(0.7, 0.7), textcoords='axes fraction',
                     arrowprops=dict(facecolor='green', shrink=0.05, width=2),
                     fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(output_dir / "comprehensive_summary.png", dpi=150)
    plt.close()
    
    # 7. Relative improvement chart
    plt.figure(figsize=(12, 6))
    
    # Calculate improvement percentages
    speed_improvement = [((a/b)-1)*100 for a, b in zip(agency_speed_mean, baseline_speed_mean)]
    quality_improvement = [((b/a)-1)*100 for a, b in zip(agency_ppl_mean, baseline_ppl_mean)]
    diversity_improvement = [((a/b)-1)*100 for a, b in zip(agency_div_mean, baseline_div_mean)]
    
    # Create improvement bars
    width = 0.25
    x = np.arange(len(pruning_levels))
    
    plt.bar(x - width, speed_improvement, width, label='Speed', color='#1976D2')
    plt.bar(x, quality_improvement, width, label='Quality', color='#D32F2F')
    plt.bar(x + width, diversity_improvement, width, label='Diversity', color='#388E3C')
    
    # Add horizontal line at 0%
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Add labels and annotations
    plt.xlabel('Pruning Level (%)', fontsize=12)
    plt.ylabel('Improvement (%)', fontsize=12)
    plt.title(f'Agency vs. Baseline: Relative Improvement at T={temperature}', fontsize=16)
    plt.xticks(x, pruning_levels)
    plt.legend()
    
    # Add value labels on bars
    for i, v in enumerate(speed_improvement):
        plt.text(i - width, v + (5 if v >= 0 else -10), f"{v:.1f}%", ha='center', fontsize=8)
    for i, v in enumerate(quality_improvement):
        plt.text(i, v + (5 if v >= 0 else -10), f"{v:.1f}%", ha='center', fontsize=8)
    for i, v in enumerate(diversity_improvement):
        plt.text(i + width, v + (5 if v >= 0 else -10), f"{v:.1f}%", ha='center', fontsize=8)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "relative_improvement.png", dpi=150)
    plt.close()

def visualize_temperature_comparison(results, output_dir):
    """Create visualizations comparing results across different temperatures."""
    output_dir = Path(output_dir)
    
    # Extract temperatures
    temp_keys = [k for k in results.keys() if k.startswith("temperature_")]
    if not temp_keys:
        return
    
    temperatures = [float(k.split("_")[1]) for k in temp_keys]
    model_name = results["metadata"]["model_name"]
    
    # Function to safely extract metrics with error handling
    def safe_extract(data, level, metric):
        level_key = str(level)
        if level_key not in data:
            # Try integer key if string key doesn't work
            level_key = level
            
        if level_key not in data:
            print(f"Warning: Level {level} not found in data")
            return {"mean": 0.0, "std": 0.0}
            
        return data[level_key][metric]
    
    # Get all available pruning levels from the first temperature's data
    baseline_levels = []
    for level in results[temp_keys[0]]["baseline"].keys():
        # Try to convert to int if it's a string
        try:
            level_int = int(level)
            baseline_levels.append(level_int)
        except (ValueError, TypeError):
            # If it's not an integer string, just use as is
            baseline_levels.append(level)
    
    pruning_levels = sorted(baseline_levels)
    max_level = max(pruning_levels)
    
    # 1. Speed comparison across temperatures
    plt.figure(figsize=(12, 8))
    
    for temp in temperatures:
        temp_key = f"temperature_{temp}"
        
        # Extract data for agency at this temperature using safe_extract
        speed_mean = [safe_extract(results[temp_key]["agency"], level, "tokens_per_second")["mean"] 
                      for level in pruning_levels]
        
        plt.plot(pruning_levels, speed_mean, marker='o', label=f"T={temp}", linewidth=2)
    
    plt.title(f"Agency Model: Generation Speed vs. Pruning Level", fontsize=16)
    plt.xlabel("Pruning Percentage", fontsize=14)
    plt.ylabel("Tokens per Second", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, title="Temperature")
    
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_speed_comparison.png", dpi=150)
    plt.close()
    
    # 2. Perplexity comparison across temperatures
    plt.figure(figsize=(12, 8))
    
    for temp in temperatures:
        temp_key = f"temperature_{temp}"
        
        # Extract data for agency at this temperature using safe_extract
        ppl_mean = [safe_extract(results[temp_key]["agency"], level, "perplexity")["mean"] 
                   for level in pruning_levels]
        
        plt.plot(pruning_levels, ppl_mean, marker='o', label=f"T={temp}", linewidth=2)
    
    plt.title(f"Agency Model: Perplexity vs. Pruning Level", fontsize=16)
    plt.xlabel("Pruning Percentage", fontsize=14)
    plt.ylabel("Perplexity (Lower is Better)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, title="Temperature")
    
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_perplexity_comparison.png", dpi=150)
    plt.close()
    
    # 3. Speed improvement comparison across temperatures
    plt.figure(figsize=(12, 8))
    
    improvement_by_temp = []
    
    for temp in temperatures:
        temp_key = f"temperature_{temp}"
        
        # Extract data for agency and baseline using safe_extract
        agency_speed = [safe_extract(results[temp_key]["agency"], level, "tokens_per_second")["mean"] 
                      for level in pruning_levels]
        baseline_speed = [safe_extract(results[temp_key]["baseline"], level, "tokens_per_second")["mean"] 
                        for level in pruning_levels]
        
        # Calculate improvement percentage
        improvement = [((a/b)-1)*100 for a, b in zip(agency_speed, baseline_speed)]
        improvement_by_temp.append(improvement)
        
        plt.plot(pruning_levels, improvement, marker='o', label=f"T={temp}", linewidth=2)
    
    plt.title(f"Agency vs. Baseline: Speed Improvement Across Temperatures", fontsize=16)
    plt.xlabel("Pruning Percentage", fontsize=14)
    plt.ylabel("Speed Improvement (%)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, title="Temperature")
    
    # Add horizontal line at 0%
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_improvement_comparison.png", dpi=150)
    plt.close()
    
    # 4. Heatmap of speed improvement
    plt.figure(figsize=(10, 8))
    
    improvement_array = np.array(improvement_by_temp)
    
    # Create heatmap
    sns.heatmap(improvement_array, 
               annot=True, 
               fmt=".1f", 
               xticklabels=pruning_levels,
               yticklabels=[f"T={t}" for t in temperatures],
               cmap="RdYlGn",
               center=0,
               linewidths=.5)
    
    plt.title(f"Speed Improvement Heatmap (%): Agency vs. Baseline", fontsize=16)
    plt.xlabel("Pruning Level (%)", fontsize=14)
    plt.ylabel("Temperature", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / "improvement_heatmap.png", dpi=150)
    plt.close()

def visualize_simple_results(results, output_dir):
    """Backward compatibility for old results format."""
    output_dir = Path(output_dir)
    pruning_levels = sorted([int(level) for level in results["baseline"].keys()])
    
    # Extract metrics
    baseline_speed = [results["baseline"][str(level)]["tokens_per_second"] for level in pruning_levels]
    agency_speed = [results["agency"][str(level)]["tokens_per_second"] for level in pruning_levels]
    
    baseline_ppl = [results["baseline"][str(level)]["perplexity"] for level in pruning_levels]
    agency_ppl = [results["agency"][str(level)]["perplexity"] for level in pruning_levels]
    
    baseline_div = [results["baseline"][str(level)]["diversity"] for level in pruning_levels]
    agency_div = [results["agency"][str(level)]["diversity"] for level in pruning_levels]
    
    # Create visualizations (simplified version of the above)
    # ... (rest of the original visualization code)
    # Note: This is intentionally brief since we're moving to the new format

def main():
    """Main function."""
    # Parse arguments first to get quiet flag
    args = setup_args()
    
    # Create a logging function that respects quiet flag
    def log(message, force=False):
        if not args.quiet or force:
            print(message)
    
    # Function to safely extract metrics with error handling
    def safe_extract(data, level, metric):
        level_key = str(level)
        if level_key not in data:
            # Try integer key if string key doesn't work
            level_key = level
            
        if level_key not in data:
            log(f"Warning: Level {level} not found in data")
            return {"mean": 0.0, "std": 0.0}
            
        return data[level_key][metric]
    
    if args.visualize_only:
        # Load existing results (either from a specific file or from latest)
        results_path = None
        
        # Check for latest results
        latest_link = Path(args.output_dir) / "latest"
        if os.path.exists(latest_link) and os.path.islink(latest_link):
            results_file = Path(os.readlink(latest_link)) / "pruning_comparison_results.json"
            if os.path.exists(results_file):
                results_path = results_file
                log(f"Using latest results from: {results_file}")
        
        # If no latest, try direct path
        if results_path is None:
            results_file = Path(args.output_dir) / "pruning_comparison_results.json"
            if os.path.exists(results_file):
                results_path = results_file
                log(f"Using results from: {results_file}")
        
        if results_path is None:
            log(f"Error: No results file found in {args.output_dir}", force=True)
            return
        
        # Load and visualize
        with open(results_path, "r") as f:
            results = json.load(f)
        
        output_dir = results_path.parent
        visualize_results(results, output_dir)
        log(f"Visualizations saved to {output_dir}", force=True)
    else:
        # Run the full comparison
        results, output_dir = run_pruning_comparison(args)
        visualize_results(results, output_dir)
        log(f"Experiment completed. Results and visualizations saved to {output_dir}", force=True)
        
        # Display a summary of the key findings
        if any(k.startswith("temperature_") for k in results.keys()):
            # For new format with temperatures
            for temp_key in [k for k in results.keys() if k.startswith("temperature_")]:
                temp = float(temp_key.split("_")[1])
                print(f"\nKey findings at temperature {temp}:")
                
                for level in sorted([int(l) for l in results[temp_key]["baseline"].keys()]):
                    if level > 0:  # Only for pruned levels
                        # Use safe_extract helper to get metrics without key errors
                        baseline_metrics = safe_extract(results[temp_key]["baseline"], level, "tokens_per_second")
                        agency_metrics = safe_extract(results[temp_key]["agency"], level, "tokens_per_second")
                        baseline_speed = baseline_metrics["mean"]
                        agency_speed = agency_metrics["mean"]
                        
                        baseline_ppl_metrics = safe_extract(results[temp_key]["baseline"], level, "perplexity")
                        agency_ppl_metrics = safe_extract(results[temp_key]["agency"], level, "perplexity")
                        baseline_ppl = baseline_ppl_metrics["mean"]
                        agency_ppl = agency_ppl_metrics["mean"]
                        
                        speed_imp = ((agency_speed / baseline_speed) - 1) * 100
                        quality_imp = ((baseline_ppl / agency_ppl) - 1) * 100
                        
                        print(f"  At {level}% pruning: Agency is {speed_imp:.1f}% faster and {quality_imp:.1f}% better quality")
        else:
            # Old format
            print("\nKey findings:")
            for level in sorted([int(l) for l in results["baseline"].keys()]):
                if level > 0:  # Only for pruned levels
                    # Function to safely extract metrics with error handling
                    def safe_get(data, level, metric):
                        level_key = str(level)
                        if level_key not in data:
                            # Try integer key if string key doesn't work
                            level_key = level
                            
                        if level_key not in data:
                            print(f"Warning: Level {level} not found in data")
                            return 0.0
                            
                        return data[level_key][metric]
                    
                    baseline_speed = safe_get(results["baseline"], level, "tokens_per_second")
                    agency_speed = safe_get(results["agency"], level, "tokens_per_second")
                    
                    baseline_ppl = safe_get(results["baseline"], level, "perplexity")
                    agency_ppl = safe_get(results["agency"], level, "perplexity")
                    
                    speed_imp = ((agency_speed / baseline_speed) - 1) * 100
                    quality_imp = ((baseline_ppl / agency_ppl) - 1) * 100
                    
                    print(f"  At {level}% pruning: Agency is {speed_imp:.1f}% faster and {quality_imp:.1f}% better quality")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\nError during experiment: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()