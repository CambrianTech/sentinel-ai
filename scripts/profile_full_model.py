#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Full Model Profiling Tool

This script profiles the complete model execution to identify bottlenecks
that may be preventing the optimized attention mechanism from showing its
performance benefits in the full model context.

Features:
- Detailed profiling of full model inference
- Component-level timing and analysis
- Memory utilization tracking
- Bottleneck identification
- Progressive feature isolation
"""

import os
import sys
import time
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from models.loaders.loader import load_baseline_model, load_adaptive_model


def setup_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Profile full model execution and identify bottlenecks")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="gpt2", 
                        help="Base model to use (default: gpt2)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (default: cuda if available, else cpu)")
    
    # Test parameters
    parser.add_argument("--sequence_length", type=int, default=64,
                        help="Input sequence length for profiling")
    parser.add_argument("--generated_tokens", type=int, default=20,
                        help="Number of tokens to generate during profiling")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Batch size for profiling")
    parser.add_argument("--pruning_levels", type=str, default="0,30,50,70",
                        help="Comma-separated pruning percentages to test")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of iterations for timing tests")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Number of warmup iterations before measurement")
    
    # Profiling options
    parser.add_argument("--profile_mode", type=str, default="basic", 
                        choices=["basic", "detailed", "component"],
                        help="Profiling detail level")
    parser.add_argument("--disable_baseline", action="store_true",
                        help="Disable baseline model integration to isolate its impact")
    parser.add_argument("--disable_unet", action="store_true",
                        help="Disable UNet connections to isolate their impact")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="profiling_results/full_model",
                        help="Directory to save profiling results")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations of profiling results")
    parser.add_argument("--trace_export", action="store_true",
                        help="Export Chrome trace files for detailed analysis")
    
    return parser.parse_args()


def prepare_input_data(args):
    """Prepare input data for profiling."""
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create sample prompt
    prompt = "The transformer model architecture revolutionized natural language processing by"
    
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Adjust batch size if needed
    if args.batch_size > 1:
        input_ids = input_ids.repeat(args.batch_size, 1)
    
    # Move to device
    input_ids = input_ids.to(args.device)
    
    # Create attention mask (all 1s since we don't have padding)
    attention_mask = torch.ones_like(input_ids)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "tokenizer": tokenizer,
        "prompt": prompt
    }


def profile_model_loading(args):
    """Profile model loading time and memory usage."""
    print("\n==== Profiling Model Loading ====")
    
    results = {
        "operation": "model_loading",
        "baseline_model": {},
        "original_model": {},
        "optimized_model": {}
    }
    
    def load_and_measure(model_type):
        # Clear cache if on CUDA
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            start_memory = torch.cuda.memory_allocated()
        else:
            start_memory = 0
        
        # Measure loading time
        start_time = time.time()
        
        if model_type == "baseline":
            model = load_baseline_model(args.model_name, args.device)
        elif model_type == "original":
            # Force original model
            os.environ["USE_OPTIMIZED_MODEL"] = "0"
            baseline_model = load_baseline_model(args.model_name, args.device)
            model = load_adaptive_model(args.model_name, baseline_model, args.device)
            del baseline_model
        elif model_type == "optimized":
            # Force optimized model
            os.environ["USE_OPTIMIZED_MODEL"] = "1"
            baseline_model = load_baseline_model(args.model_name, args.device)
            model = load_adaptive_model(args.model_name, baseline_model, args.device)
            del baseline_model
        
        load_time = time.time() - start_time
        
        # Measure memory usage
        if args.device == "cuda" and torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() - start_memory
        else:
            # Count parameters as a proxy for memory usage
            memory_usage = sum(p.numel() for p in model.parameters()) * 4  # Approximate bytes (float32)
        
        # Count parameters
        parameter_count = sum(p.numel() for p in model.parameters())
        
        result = {
            "load_time": load_time,
            "memory_usage": memory_usage,
            "parameter_count": parameter_count
        }
        
        print(f"{model_type.capitalize()} model loaded in {load_time:.2f}s")
        print(f"  Parameters: {parameter_count:,}")
        print(f"  Memory: {memory_usage/(1024**2):.2f} MB")
        
        del model
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result
    
    # Measure baseline model
    results["baseline_model"] = load_and_measure("baseline")
    
    # Measure original model
    results["original_model"] = load_and_measure("original")
    
    # Measure optimized model
    results["optimized_model"] = load_and_measure("optimized")
    
    return results


def apply_pruning(model, pruning_percentage):
    """Apply pruning to the model by setting selected gate values to zero."""
    from scripts.pruning_comparison.pruning_agency_comparison import apply_pruning as pruning_func
    
    # Apply pruning
    pruned_model, pruned_count, pruned_heads = pruning_func(
        model, 
        pruning_percentage, 
        verbose=False,
        quiet=True
    )
    
    return pruned_model, pruned_count, pruned_heads


def profile_inference(args, data, model_type="original", pruning_level=0):
    """Profile inference for a specific model type and pruning level."""
    print(f"\n==== Profiling {model_type.capitalize()} Model (Pruning: {pruning_level}%) ====")
    
    # Load appropriate model
    if model_type == "original":
        os.environ["USE_OPTIMIZED_MODEL"] = "0"
    else:  # optimized
        os.environ["USE_OPTIMIZED_MODEL"] = "1"
    
    # First load baseline model
    baseline_model = load_baseline_model(args.model_name, args.device)
    
    # Then load specified model
    model = load_adaptive_model(args.model_name, baseline_model, args.device)
    
    # Free baseline model memory if it's not needed anymore
    if model_type == "optimized" and args.disable_baseline:
        # Detach baseline model integration
        if hasattr(model, "baseline_model"):
            model.baseline_model = None
        elif hasattr(model, "model") and hasattr(model.model, "baseline_model"):
            model.model.baseline_model = None
    
    # Apply UNet settings if needed
    if args.disable_unet and model_type == "optimized":
        # Disable UNet connections
        for idx, block in enumerate(model.blocks if hasattr(model, "blocks") else model.model.blocks):
            if hasattr(block, "use_skip_connection"):
                block.use_skip_connection = False
    
    # Apply pruning if needed
    if pruning_level > 0:
        model, pruned_count, _ = apply_pruning(model, pruning_level)
    
    # Extract inputs
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    
    # Set up profiling based on mode
    if args.profile_mode == "detailed" and args.device == "cuda" and torch.cuda.is_available():
        profiler_ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            profile_memory=True
        )
    else:
        profiler_ctx = nullcontext()
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.size(1) + args.generated_tokens,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )
    
    # Measure generation performance
    generation_times = []
    first_token_times = []
    
    print(f"Running {args.iterations} iterations...")
    
    for i in range(args.iterations):
        # Clear CUDA cache if available
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        # Start timing
        start_time = time.time()
        
        with torch.no_grad(), profiler_ctx as prof:
            with record_function(f"{model_type}_inference"):
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=input_ids.size(1) + args.generated_tokens,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7
                )
            
            # Record first token time (for first iteration only)
            if i == 0:
                first_token_time = time.time() - start_time
                first_token_times.append(first_token_time)
        
        # Ensure CUDA operations are completed
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Calculate total generation time
        generation_time = time.time() - start_time
        generation_times.append(generation_time)
        
        print(f"  Iteration {i+1}: {generation_time:.4f}s ({output_ids.size(1) - input_ids.size(1)} tokens)")
    
    # Calculate average generation time
    avg_generation_time = sum(generation_times) / len(generation_times)
    tokens_per_second = args.generated_tokens / avg_generation_time
    
    print(f"Average generation time: {avg_generation_time:.4f}s")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    
    # Check if detailed profile is available
    if args.profile_mode == "detailed" and args.device == "cuda" and torch.cuda.is_available():
        # Save trace if requested
        if args.trace_export:
            os.makedirs(args.output_dir, exist_ok=True)
            trace_path = os.path.join(args.output_dir, f"{model_type}_pruning{pruning_level}_trace.json")
            prof.export_chrome_trace(trace_path)
            print(f"Trace exported to {trace_path}")
        
        # Get profile data
        profile_data = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
    else:
        profile_data = None
    
    # Extract generated text
    if attention_mask is not None:
        tokenizer = data["tokenizer"]
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    else:
        generated_text = None
    
    # Free memory
    del model
    del baseline_model
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Return results
    return {
        "model_type": model_type,
        "pruning_level": pruning_level,
        "generation_times": generation_times,
        "avg_generation_time": avg_generation_time,
        "tokens_per_second": tokens_per_second,
        "first_token_time": first_token_times[0] if first_token_times else None,
        "profile_data": profile_data,
        "generated_text": generated_text
    }


def profile_component_breakdown(args, data):
    """Profile the time spent in each component of the model."""
    print("\n==== Profiling Component Breakdown ====")
    
    if args.device != "cuda" or not torch.cuda.is_available():
        print("Component breakdown requires CUDA. Skipping...")
        return None
    
    # Extract inputs
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    
    # Test both original and optimized models
    results = {}
    
    for model_type in ["original", "optimized"]:
        print(f"\nProfiling {model_type} model components...")
        
        # Set model type
        if model_type == "original":
            os.environ["USE_OPTIMIZED_MODEL"] = "0"
        else:  # optimized
            os.environ["USE_OPTIMIZED_MODEL"] = "1"
        
        # Load models
        baseline_model = load_baseline_model(args.model_name, args.device)
        model = load_adaptive_model(args.model_name, baseline_model, args.device)
        
        # Extract model components based on model type
        if hasattr(model, "blocks"):
            blocks = model.blocks
        elif hasattr(model, "model") and hasattr(model.model, "blocks"):
            blocks = model.model.blocks
        else:
            print(f"  Unable to extract blocks from {model_type} model")
            continue
        
        # Enable CUDA profiler
        torch.cuda.synchronize()
        
        # Profile each component
        component_times = {
            "embeddings": [],
            "attention": [],
            "ffn": [],
            "ln": [],
            "other": []
        }
        
        with torch.no_grad():
            # Warmup
            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.size(1) + 5,  # Just a few tokens for warmup
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )
            
            # Profile embeddings
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                # Get input embeddings
                with record_function("embeddings"):
                    position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
                    embeddings = model.wte(input_ids) + model.wpe(position_ids)
            
            # Extract times
            for event in prof.key_averages():
                if event.key == "embeddings":
                    component_times["embeddings"].append(event.cuda_time_total / 1000)  # Convert to ms
            
            # Initialize hidden states
            hidden_states = embeddings
            
            # Profile each transformer block
            for i, block in enumerate(blocks):
                block_times = {}
                
                # Profile attention
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    with record_function("attention"):
                        if hasattr(block, "attn"):
                            # Direct attention access
                            _ = block.attn(block.ln1(hidden_states))
                        elif hasattr(block, "attention"):
                            # Different attribute name
                            _ = block.attention(block.ln1(hidden_states))
                
                # Extract times
                for event in prof.key_averages():
                    if event.key == "attention":
                        block_times["attention"] = event.cuda_time_total / 1000  # Convert to ms
                
                # Profile FFN
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    with record_function("ffn"):
                        if hasattr(block, "ffn"):
                            # Direct FFN access
                            _ = block.ffn(block.ln2(hidden_states))
                        elif hasattr(block, "mlp"):
                            # Different attribute name
                            _ = block.mlp(block.ln2(hidden_states))
                
                # Extract times
                for event in prof.key_averages():
                    if event.key == "ffn":
                        block_times["ffn"] = event.cuda_time_total / 1000  # Convert to ms
                
                # Store block times
                if "attention" in block_times:
                    component_times["attention"].append(block_times["attention"])
                if "ffn" in block_times:
                    component_times["ffn"].append(block_times["ffn"])
            
            # Update hidden states and continue
            # Note: This is just for timing components, not actual inference
        
        # Calculate average times
        avg_times = {}
        for component, times in component_times.items():
            if times:
                avg_times[component] = sum(times) / len(times)
            else:
                avg_times[component] = 0
        
        # Store results
        results[model_type] = {
            "component_times": component_times,
            "avg_times": avg_times
        }
        
        # Free memory
        del model
        del baseline_model
        torch.cuda.empty_cache()
    
    return results


def compare_pruning_levels(args, data):
    """Compare model performance across different pruning levels."""
    print("\n==== Comparing Pruning Levels ====")
    
    # Parse pruning levels
    pruning_levels = [int(x) for x in args.pruning_levels.split(",")]
    
    results = {
        "original": {},
        "optimized": {}
    }
    
    # Test each model type with each pruning level
    for model_type in ["original", "optimized"]:
        for level in pruning_levels:
            print(f"\nTesting {model_type} model with {level}% pruning")
            
            # Profile inference for this configuration
            inference_results = profile_inference(args, data, model_type, level)
            
            # Store results
            results[model_type][level] = inference_results
    
    return results


def visualize_results(results, args):
    """Create visualizations from profiling results."""
    if not args.visualize:
        return
    
    print("\n==== Creating Visualizations ====")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Model Loading Comparison
    if "model_loading" in results:
        data = results["model_loading"]
        
        # Extract data
        models = ["baseline_model", "original_model", "optimized_model"]
        load_times = [data[model]["load_time"] for model in models]
        param_counts = [data[model]["parameter_count"] for model in models]
        memory_usage = [data[model]["memory_usage"] / (1024**2) for model in models]  # Convert to MB
        
        # Create figure
        plt.figure(figsize=(15, 5))
        
        # Plot load times
        plt.subplot(1, 3, 1)
        bars = plt.bar([m.split("_")[0].capitalize() for m in models], load_times, color=['gray', 'dodgerblue', 'green'])
        plt.title('Model Loading Time')
        plt.ylabel('Time (seconds)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f"{height:.2f}s", ha='center', va='bottom')
        
        # Plot parameter counts
        plt.subplot(1, 3, 2)
        bars = plt.bar([m.split("_")[0].capitalize() for m in models], param_counts, color=['gray', 'dodgerblue', 'green'])
        plt.title('Model Parameters')
        plt.ylabel('Parameter Count')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels with formatting
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f"{height/1000000:.1f}M", ha='center', va='bottom')
        
        # Plot memory usage
        plt.subplot(1, 3, 3)
        bars = plt.bar([m.split("_")[0].capitalize() for m in models], memory_usage, color=['gray', 'dodgerblue', 'green'])
        plt.title('Memory Usage')
        plt.ylabel('Memory (MB)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f"{height:.1f}MB", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_loading_comparison.png"), dpi=150)
        plt.close()
    
    # 2. Pruning Performance Comparison
    if "pruning_comparison" in results:
        data = results["pruning_comparison"]
        
        # Extract pruning levels
        pruning_levels = sorted([level for level in data["original"].keys()])
        
        # Extract performance metrics
        original_speed = [data["original"][level]["tokens_per_second"] for level in pruning_levels]
        optimized_speed = [data["optimized"][level]["tokens_per_second"] for level in pruning_levels]
        speedup = [optimized_speed[i] / original_speed[i] for i in range(len(pruning_levels))]
        
        original_time = [data["original"][level]["avg_generation_time"] for level in pruning_levels]
        optimized_time = [data["optimized"][level]["avg_generation_time"] for level in pruning_levels]
        
        original_first = [data["original"][level]["first_token_time"] for level in pruning_levels]
        optimized_first = [data["optimized"][level]["first_token_time"] for level in pruning_levels]
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot tokens per second
        plt.subplot(2, 2, 1)
        plt.plot(pruning_levels, original_speed, 'o-', label="Original", color="dodgerblue", linewidth=2)
        plt.plot(pruning_levels, optimized_speed, 'o-', label="Optimized", color="green", linewidth=2)
        plt.title('Generation Speed vs. Pruning Level')
        plt.xlabel('Pruning Level (%)')
        plt.ylabel('Tokens per Second')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add annotations
        for i, level in enumerate(pruning_levels):
            plt.annotate(f"{original_speed[i]:.1f}",
                        xy=(level, original_speed[i]),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
            plt.annotate(f"{optimized_speed[i]:.1f}",
                        xy=(level, optimized_speed[i]),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
        
        # Plot generation time
        plt.subplot(2, 2, 2)
        plt.plot(pruning_levels, original_time, 'o-', label="Original", color="dodgerblue", linewidth=2)
        plt.plot(pruning_levels, optimized_time, 'o-', label="Optimized", color="green", linewidth=2)
        plt.title('Generation Time vs. Pruning Level')
        plt.xlabel('Pruning Level (%)')
        plt.ylabel('Time (seconds)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Plot speedup
        plt.subplot(2, 2, 3)
        bars = plt.bar(pruning_levels, speedup, color="coral")
        plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
        plt.title('Optimization Speedup by Pruning Level')
        plt.xlabel('Pruning Level (%)')
        plt.ylabel('Speedup Factor (>1 is better)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f"{height:.2f}x", ha='center', va='bottom')
        
        # Plot first token latency
        plt.subplot(2, 2, 4)
        plt.plot(pruning_levels, original_first, 'o-', label="Original", color="dodgerblue", linewidth=2)
        plt.plot(pruning_levels, optimized_first, 'o-', label="Optimized", color="green", linewidth=2)
        plt.title('First Token Latency vs. Pruning Level')
        plt.xlabel('Pruning Level (%)')
        plt.ylabel('Time (seconds)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pruning_performance.png"), dpi=150)
        plt.close()
    
    # 3. Component Breakdown
    if "component_breakdown" in results:
        data = results["component_breakdown"]
        
        # Check if data is available
        if not data:
            return
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Extract component data
        components = ["embeddings", "attention", "ffn", "ln", "other"]
        orig_times = [data["original"]["avg_times"].get(comp, 0) for comp in components]
        opt_times = [data["optimized"]["avg_times"].get(comp, 0) for comp in components]
        
        # Filter out zero values
        valid_components = []
        valid_orig = []
        valid_opt = []
        for i, comp in enumerate(components):
            if orig_times[i] > 0 or opt_times[i] > 0:
                valid_components.append(comp)
                valid_orig.append(orig_times[i])
                valid_opt.append(opt_times[i])
        
        # Plot component comparison
        plt.subplot(2, 1, 1)
        x = np.arange(len(valid_components))
        width = 0.35
        
        plt.bar(x - width/2, valid_orig, width, label='Original', color="dodgerblue")
        plt.bar(x + width/2, valid_opt, width, label='Optimized', color="green")
        
        plt.ylabel('Time (ms)')
        plt.title('Component Execution Time')
        plt.xticks(x, valid_components)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        
        # Calculate speedup by component
        speedups = []
        for i in range(len(valid_components)):
            if valid_orig[i] > 0:
                speedups.append(valid_orig[i] / valid_opt[i] if valid_opt[i] > 0 else 0)
            else:
                speedups.append(0)
        
        # Plot speedup by component
        plt.subplot(2, 1, 2)
        bars = plt.bar(valid_components, speedups, color="coral")
        plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
        plt.ylabel('Speedup Factor')
        plt.title('Component Speedup (Optimized vs. Original)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f"{height:.2f}x", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "component_breakdown.png"), dpi=150)
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def main():
    """Main function."""
    # Parse arguments
    args = setup_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Store all results
    results = {}
    
    # Profile model loading
    results["model_loading"] = profile_model_loading(args)
    
    # Prepare input data
    data = prepare_input_data(args)
    
    # Compare pruning levels
    results["pruning_comparison"] = compare_pruning_levels(args, data)
    
    # Profile component breakdown
    if args.profile_mode == "component":
        results["component_breakdown"] = profile_component_breakdown(args, data)
    
    # Save results
    results_file = os.path.join(args.output_dir, "full_model_profiling.json")
    with open(results_file, "w") as f:
        # Filter out non-JSON serializable items
        filtered_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                filtered_results[key] = value
        json.dump(filtered_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Create visualizations
    visualize_results(results, args)
    
    # Print summary
    print("\n===== Profiling Summary =====")
    
    # Loading time summary
    if "model_loading" in results:
        loading_data = results["model_loading"]
        print("\nModel Loading:")
        for model_type in ["baseline_model", "original_model", "optimized_model"]:
            model_name = model_type.split("_")[0].capitalize()
            print(f"  {model_name}: {loading_data[model_type]['load_time']:.2f}s, "
                 f"{loading_data[model_type]['parameter_count']:,} parameters")
    
    # Pruning comparison summary
    if "pruning_comparison" in results:
        pruning_data = results["pruning_comparison"]
        print("\nPerformance by Pruning Level:")
        for level in sorted([int(x) for x in next(iter(pruning_data["original"].keys()))]):
            orig_speed = pruning_data["original"][level]["tokens_per_second"]
            opt_speed = pruning_data["optimized"][level]["tokens_per_second"]
            speedup = opt_speed / orig_speed
            print(f"  Level {level}%: {speedup:.2f}x speedup ({opt_speed:.2f} vs {orig_speed:.2f} tokens/sec)")
    
    # Component breakdown summary
    if "component_breakdown" in results and results["component_breakdown"]:
        component_data = results["component_breakdown"]
        print("\nComponent Breakdown:")
        components = ["attention", "ffn", "embeddings"]
        for comp in components:
            if comp in component_data["original"]["avg_times"] and comp in component_data["optimized"]["avg_times"]:
                orig_time = component_data["original"]["avg_times"][comp]
                opt_time = component_data["optimized"]["avg_times"][comp]
                if orig_time > 0:
                    speedup = orig_time / opt_time if opt_time > 0 else 0
                    print(f"  {comp.capitalize()}: {speedup:.2f}x speedup")


if __name__ == "__main__":
    main()