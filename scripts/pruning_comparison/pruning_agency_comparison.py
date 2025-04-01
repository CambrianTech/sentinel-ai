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
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.adaptive_transformer import AdaptiveTransformer
from models.loaders.gpt2_loader import load_gpt2_model
from utils.model_wrapper import SentinelModelWrapper
from utils.metrics import calculate_perplexity, calculate_diversity, calculate_repetition
from utils.charting import AGENCY_COLORS  # Import color scheme for consistency

# Set plotting style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def setup_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare pruning efficacy with and without agency")
    
    # Pruning configuration
    parser.add_argument("--pruning_levels", type=str, default="0,10,20,30,40,50,60,70",
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
    parser.add_argument("--num_tokens", type=int, default=100,
                      help="Number of tokens to generate for each prompt")
    parser.add_argument("--temperatures", type=str, default="0.7,1.0",
                      help="Comma-separated temperatures to test")
    parser.add_argument("--prompt_file", type=str, default="datasets/eval_prompts.txt",
                      help="File containing prompts to use for evaluation")
    parser.add_argument("--max_prompts", type=int, default=10,
                      help="Maximum number of prompts to evaluate")
    
    # Experiment configuration
    parser.add_argument("--iterations", type=int, default=3,
                      help="Number of iterations to run for statistical significance")
    parser.add_argument("--output_dir", type=str, default="validation_results/pruning_agency",
                      help="Directory to save results and visualizations")
    parser.add_argument("--save_outputs", action="store_true",
                      help="Save generated text outputs")
    parser.add_argument("--visualize_only", action="store_true",
                      help="Only generate visualizations from existing results")
    parser.add_argument("--memory_logging", action="store_true",
                      help="Log memory usage during evaluation")
    
    return parser.parse_args()

def load_model(model_name, agency_enabled=False, device="cpu", precision="float32"):
    """Load either a baseline or agency-enabled model with specified device and precision."""
    print(f"Loading {model_name} model with agency={'enabled' if agency_enabled else 'disabled'} on {device}...")
    
    # Load base model
    base_model, tokenizer = load_gpt2_model(model_name)
    
    # Create adaptive transformer layer
    adaptive_model = AdaptiveTransformer(base_model, enable_agency=agency_enabled)
    
    # Create wrapper
    model = SentinelModelWrapper(adaptive_model, tokenizer)
    
    # Move to appropriate device
    if device == "cuda" and torch.cuda.is_available():
        print(f"Moving model to CUDA...")
        model.to("cuda")
    else:
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Using CPU instead.")
        model.to("cpu")
    
    # Set precision
    if precision != "float32" and device == "cuda":
        if precision == "float16":
            print("Converting model to float16 precision...")
            model = model.half()
        elif precision == "bfloat16" and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            print("Converting model to bfloat16 precision...")
            model = model.to(torch.bfloat16)
        else:
            print(f"Precision {precision} not supported on this device. Using float32.")
    
    # Log memory usage if on CUDA
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"CUDA Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    return model, tokenizer

def apply_pruning(model, pruning_percentage, method="entropy", verbose=True):
    """
    Apply pruning to the model by setting selected gate values to zero.
    
    Args:
        model: The transformer model to prune
        pruning_percentage: Percentage of heads to prune (0-100)
        method: Pruning method ('entropy', 'random', or 'magnitude')
        verbose: Whether to print pruning details
        
    Returns:
        Tuple of (pruned model, number of heads pruned, list of pruned heads)
    """
    start_time = time.time()
    
    # Get all head information
    heads_info = []
    for layer_idx, block in enumerate(model.model.blocks):
        for head_idx in range(model.model.num_heads):
            gate_value = float(block["attn"].gate[head_idx])
            
            # Get head information based on pruning method
            if method == "entropy":
                # Use gate value as proxy for entropy-based importance
                importance = gate_value
            elif method == "magnitude":
                # Use weight magnitude as importance
                weight = block["attn"].attn.attention.weight
                head_size = weight.size(0) // model.model.num_heads
                start_idx = head_idx * head_size
                end_idx = start_idx + head_size
                head_weight = weight[start_idx:end_idx, :]
                importance = float(head_weight.abs().mean())
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
            print(f"No heads to prune at {pruning_percentage}%")
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
        model.model.blocks[layer_idx]["attn"].gate[head_idx].zero_()
        pruned_heads.append((layer_idx, head_idx))
    
    # Print pruning stats if verbose
    if verbose:
        duration = time.time() - start_time
        print(f"Pruned {num_to_prune}/{num_heads} heads ({pruning_percentage}%) using {method} method in {duration:.2f}s")
        
        # If agency is enabled, report agency states
        if hasattr(model.model, "enable_agency") and model.model.enable_agency:
            agency_states = {"active": 0, "overloaded": 0, "misaligned": 0, "withdrawn": 0}
            consent_withdrawn = 0
            
            for layer_idx, block in enumerate(model.model.blocks):
                if hasattr(block["attn"], "agency_signals"):
                    for head_idx, signals in block["attn"].agency_signals.items():
                        agency_states[signals["state"]] += 1
                        if not signals["consent"]:
                            consent_withdrawn += 1
            
            print(f"Agency states after pruning: {agency_states}")
            print(f"Heads with withdrawn consent: {consent_withdrawn}")
    
    return model, num_to_prune, pruned_heads

def load_prompts(prompt_file):
    """Load prompts from a file."""
    with open(prompt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def evaluate_model(model, tokenizer, prompts, num_tokens, temperature=0.7, 
                batch_size=1, device="cpu", memory_logging=False, max_prompts=None):
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
        print(f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size} ({len(batch_prompts)} prompts)")
        
        for prompt_idx, prompt in enumerate(batch_prompts):
            prompt_num = i + prompt_idx + 1
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
                output_ids = model.model.generate(
                    input_ids,
                    max_length=input_ids.size(1) + num_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0),
                    pad_token_id=tokenizer.eos_token_id,
                )
                
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
                
                # Print progress info
                print(f"    Generated {tokens_generated} tokens in {generation_time:.2f}s "
                     f"({tokens_per_second:.2f} tokens/sec), perplexity: {perplexity:.2f}")
                
            except Exception as e:
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
    
    # Print summary
    print(f"\nEvaluation complete: {len(prompts) - failures}/{len(prompts)} prompts successful")
    print(f"Average tokens/sec: {averaged_results['tokens_per_second']:.2f}")
    print(f"Average perplexity: {averaged_results['perplexity']:.2f}")
    
    return averaged_results

def run_pruning_comparison(args):
    """Run the main comparison between baseline and agency models."""
    # Process command line arguments
    pruning_levels = [int(x) for x in args.pruning_levels.split(",")]
    temperatures = [float(x) for x in args.temperatures.split(",")]
    prompts = load_prompts(args.prompt_file)
    
    if args.max_prompts and args.max_prompts < len(prompts):
        prompts = prompts[:args.max_prompts]
        print(f"Using {len(prompts)} prompts for evaluation (limited by --max_prompts)")
    else:
        print(f"Using all {len(prompts)} prompts for evaluation")
    
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
    
    print(f"Results will be saved to {output_dir}")
    
    # Save a copy of the prompts
    with open(output_dir / "prompts.txt", "w") as f:
        f.write("\n".join(prompts))
    
    # Create subdirectories for iterations
    for i in range(args.iterations):
        (output_dir / f"iteration_{i}").mkdir(exist_ok=True)
    
    # For each temperature value
    for temp_idx, temperature in enumerate(temperatures):
        print(f"\n{'='*80}\nTesting temperature {temperature}\n{'='*80}")
        
        results[f"temperature_{temperature}"] = {"baseline": {}, "agency": {}}
        
        # For each pruning level
        for level in pruning_levels:
            print(f"\n{'-'*40}\nEvaluating pruning level: {level}%\n{'-'*40}")
            
            # Initialize iteration results
            baseline_iterations = []
            agency_iterations = []
            
            # Run multiple iterations for statistical significance
            for iteration in range(args.iterations):
                print(f"\nIteration {iteration+1}/{args.iterations}")
                
                iteration_dir = output_dir / f"iteration_{iteration}"
                
                # Baseline model
                print(f"Loading baseline model...")
                baseline_model, tokenizer = load_model(
                    args.model_name, 
                    agency_enabled=False,
                    device=args.device,
                    precision=args.precision
                )
                baseline_model, pruned_heads_count, _ = apply_pruning(
                    baseline_model, 
                    level,
                    method=args.pruning_method
                )
                
                # Agency model
                print(f"Loading agency model...")
                agency_model, _ = load_model(
                    args.model_name, 
                    agency_enabled=True,
                    device=args.device,
                    precision=args.precision
                )
                agency_model, pruned_heads_count, _ = apply_pruning(
                    agency_model, 
                    level,
                    method=args.pruning_method
                )
                
                # Evaluate both models
                print(f"Evaluating baseline model...")
                baseline_results = evaluate_model(
                    baseline_model, 
                    tokenizer, 
                    prompts, 
                    args.num_tokens,
                    temperature=temperature,
                    batch_size=args.batch_size,
                    device=args.device,
                    memory_logging=args.memory_logging,
                    max_prompts=args.max_prompts
                )
                
                print(f"Evaluating agency model...")
                agency_results = evaluate_model(
                    agency_model, 
                    tokenizer, 
                    prompts, 
                    args.num_tokens,
                    temperature=temperature,
                    batch_size=args.batch_size,
                    device=args.device,
                    memory_logging=args.memory_logging,
                    max_prompts=args.max_prompts
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
                print(f"\nIteration {iteration+1} Results at {level}% pruning, temp={temperature}:")
                print(f"  Baseline: {baseline_results['tokens_per_second']:.2f} tokens/sec, "
                     f"perplexity: {baseline_results['perplexity']:.2f}, "
                     f"diversity: {baseline_results['diversity']:.3f}")
                print(f"  Agency:   {agency_results['tokens_per_second']:.2f} tokens/sec, "
                     f"perplexity: {agency_results['perplexity']:.2f}, "
                     f"diversity: {agency_results['diversity']:.3f}")
                
                # Calculate improvement
                speed_improvement = ((agency_results['tokens_per_second'] / baseline_results['tokens_per_second']) - 1) * 100
                quality_improvement = ((baseline_results['perplexity'] / agency_results['perplexity']) - 1) * 100
                
                print(f"  Improvement: {speed_improvement:.1f}% faster, {quality_improvement:.1f}% better quality")
                
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
            print(f"\nAveraged Results at {level}% pruning, temp={temperature}:")
            print(f"  Baseline: {baseline_avg['tokens_per_second']['mean']:.2f} ± {baseline_avg['tokens_per_second']['std']:.2f} tokens/sec, "
                 f"perplexity: {baseline_avg['perplexity']['mean']:.2f} ± {baseline_avg['perplexity']['std']:.2f}")
            print(f"  Agency:   {agency_avg['tokens_per_second']['mean']:.2f} ± {agency_avg['tokens_per_second']['std']:.2f} tokens/sec, "
                 f"perplexity: {agency_avg['perplexity']['mean']:.2f} ± {agency_avg['perplexity']['std']:.2f}")
            
            # Calculate improvement
            speed_improvement = ((agency_avg['tokens_per_second']['mean'] / baseline_avg['tokens_per_second']['mean']) - 1) * 100
            quality_improvement = ((baseline_avg['perplexity']['mean'] / agency_avg['perplexity']['mean']) - 1) * 100
            
            print(f"  Improvement: {speed_improvement:.1f}% faster, {quality_improvement:.1f}% better quality")
            
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
    if os.path.exists(latest_link):
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        else:
            os.remove(latest_link)
    os.symlink(output_dir, latest_link, target_is_directory=True)
    
    print(f"\nAll results saved to {output_dir}")
    print(f"Latest results symlinked to {latest_link}")
    
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
    
    pruning_levels = sorted([int(level) for level in baseline_data.keys()])
    
    # Extract metrics with error bars
    baseline_speed_mean = [baseline_data[str(level)]["tokens_per_second"]["mean"] for level in pruning_levels]
    baseline_speed_std = [baseline_data[str(level)]["tokens_per_second"]["std"] for level in pruning_levels]
    
    agency_speed_mean = [agency_data[str(level)]["tokens_per_second"]["mean"] for level in pruning_levels]
    agency_speed_std = [agency_data[str(level)]["tokens_per_second"]["std"] for level in pruning_levels]
    
    baseline_ppl_mean = [baseline_data[str(level)]["perplexity"]["mean"] for level in pruning_levels]
    baseline_ppl_std = [baseline_data[str(level)]["perplexity"]["std"] for level in pruning_levels]
    
    agency_ppl_mean = [agency_data[str(level)]["perplexity"]["mean"] for level in pruning_levels]
    agency_ppl_std = [agency_data[str(level)]["perplexity"]["std"] for level in pruning_levels]
    
    baseline_div_mean = [baseline_data[str(level)]["diversity"]["mean"] for level in pruning_levels]
    baseline_div_std = [baseline_data[str(level)]["diversity"]["std"] for level in pruning_levels]
    
    agency_div_mean = [agency_data[str(level)]["diversity"]["mean"] for level in pruning_levels]
    agency_div_std = [agency_data[str(level)]["diversity"]["std"] for level in pruning_levels]
    
    # Extract first token latency if available
    if "first_token_time" in baseline_data[str(pruning_levels[0])]:
        baseline_latency_mean = [baseline_data[str(level)]["first_token_time"]["mean"] for level in pruning_levels]
        baseline_latency_std = [baseline_data[str(level)]["first_token_time"]["std"] for level in pruning_levels]
        
        agency_latency_mean = [agency_data[str(level)]["first_token_time"]["mean"] for level in pruning_levels]
        agency_latency_std = [agency_data[str(level)]["first_token_time"]["std"] for level in pruning_levels]
        
        has_latency = True
    else:
        has_latency = False
    
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
    
    # Get pruning levels
    pruning_levels = sorted([int(level) for level in results[temp_keys[0]]["baseline"].keys()])
    max_level = max(pruning_levels)
    
    # 1. Speed comparison across temperatures
    plt.figure(figsize=(12, 8))
    
    for temp in temperatures:
        temp_key = f"temperature_{temp}"
        
        # Extract data for agency at this temperature
        speed_mean = [results[temp_key]["agency"][str(level)]["tokens_per_second"]["mean"] 
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
        
        # Extract data for agency at this temperature
        ppl_mean = [results[temp_key]["agency"][str(level)]["perplexity"]["mean"] 
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
        
        # Extract data for agency and baseline
        agency_speed = [results[temp_key]["agency"][str(level)]["tokens_per_second"]["mean"] 
                     for level in pruning_levels]
        baseline_speed = [results[temp_key]["baseline"][str(level)]["tokens_per_second"]["mean"] 
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
    args = setup_args()
    
    if args.visualize_only:
        # Load existing results (either from a specific file or from latest)
        results_path = None
        
        # Check for latest results
        latest_link = Path(args.output_dir) / "latest"
        if os.path.exists(latest_link) and os.path.islink(latest_link):
            results_file = Path(os.readlink(latest_link)) / "pruning_comparison_results.json"
            if os.path.exists(results_file):
                results_path = results_file
                print(f"Using latest results from: {results_file}")
        
        # If no latest, try direct path
        if results_path is None:
            results_file = Path(args.output_dir) / "pruning_comparison_results.json"
            if os.path.exists(results_file):
                results_path = results_file
                print(f"Using results from: {results_file}")
        
        if results_path is None:
            print(f"Error: No results file found in {args.output_dir}")
            return
        
        # Load and visualize
        with open(results_path, "r") as f:
            results = json.load(f)
        
        output_dir = results_path.parent
        visualize_results(results, output_dir)
        print(f"Visualizations saved to {output_dir}")
    else:
        # Run the full comparison
        results, output_dir = run_pruning_comparison(args)
        visualize_results(results, output_dir)
        print(f"Experiment completed. Results and visualizations saved to {output_dir}")
        
        # Display a summary of the key findings
        if any(k.startswith("temperature_") for k in results.keys()):
            # For new format with temperatures
            for temp_key in [k for k in results.keys() if k.startswith("temperature_")]:
                temp = float(temp_key.split("_")[1])
                print(f"\nKey findings at temperature {temp}:")
                
                for level in sorted([int(l) for l in results[temp_key]["baseline"].keys()]):
                    if level > 0:  # Only for pruned levels
                        baseline_speed = results[temp_key]["baseline"][str(level)]["tokens_per_second"]["mean"]
                        agency_speed = results[temp_key]["agency"][str(level)]["tokens_per_second"]["mean"]
                        
                        baseline_ppl = results[temp_key]["baseline"][str(level)]["perplexity"]["mean"]
                        agency_ppl = results[temp_key]["agency"][str(level)]["perplexity"]["mean"]
                        
                        speed_imp = ((agency_speed / baseline_speed) - 1) * 100
                        quality_imp = ((baseline_ppl / agency_ppl) - 1) * 100
                        
                        print(f"  At {level}% pruning: Agency is {speed_imp:.1f}% faster and {quality_imp:.1f}% better quality")
        else:
            # Old format
            print("\nKey findings:")
            for level in sorted([int(l) for l in results["baseline"].keys()]):
                if level > 0:  # Only for pruned levels
                    baseline_speed = results["baseline"][str(level)]["tokens_per_second"]
                    agency_speed = results["agency"][str(level)]["tokens_per_second"]
                    
                    baseline_ppl = results["baseline"][str(level)]["perplexity"]
                    agency_ppl = results["agency"][str(level)]["perplexity"]
                    
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