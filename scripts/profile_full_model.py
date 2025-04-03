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
- Integration optimization evaluation
- Multi-model architecture comparison
- Enhanced visualization and reporting
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
from contextlib import nullcontext, redirect_stdout, redirect_stderr
import datetime
import psutil
import io
import warnings
import logging

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
    parser.add_argument("--model_family", type=str, choices=["gpt2", "bloom", "opt", "pythia", "llama"],
                        help="Specify model family for specialized handling (auto-detected if not specified)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (default: cuda if available, else cpu)")
    parser.add_argument("--multi_model_comparison", action="store_true",
                        help="Test multiple model families for comparison (must specify models with --compare_models)")
    parser.add_argument("--compare_models", type=str, default="gpt2,gpt2-medium",
                        help="Comma-separated list of models to compare when using --multi_model_comparison")
    parser.add_argument("--specific_model_size", type=str, choices=["small", "medium", "large", "xl"],
                        help="Test a specific model size across different model families")
    
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
    parser.add_argument("--custom_prompts", type=str,
                        help="Path to file with custom prompts for more realistic testing")
    
    # Profiling options
    parser.add_argument("--profile_mode", type=str, default="basic", 
                        choices=["basic", "detailed", "component", "all", "memory", "throughput"],
                        help="Profiling detail level (all = run all profiling types)")
    parser.add_argument("--disable_baseline", action="store_true",
                        help="Disable baseline model integration to isolate its impact")
    parser.add_argument("--disable_unet", action="store_true",
                        help="Disable UNet connections to isolate their impact")
    parser.add_argument("--test_integration_points", action="store_true",
                        help="Test different integration optimizations to isolate their impact")
    parser.add_argument("--test_all_optimization_levels", action="store_true",
                        help="Run tests across all optimization levels (0-3) for comparison")
    parser.add_argument("--optimization_level", type=int, default=1, choices=[0, 1, 2, 3],
                        help="Optimization level (0=None, 1=Default, 2=Aggressive, 3=Extreme)")
    parser.add_argument("--cpu_specific_optimizations", action="store_true",
                        help="Test CPU-specific optimizations on CPU devices")
    parser.add_argument("--memory_profile", action="store_true",
                        help="Profile memory usage over time during generation")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="profiling_results/full_model",
                        help="Directory to save profiling results")
    parser.add_argument("--output_prefix", type=str, default="",
                        help="Prefix for output filenames (e.g., experiment name)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations of profiling results")
    parser.add_argument("--trace_export", action="store_true",
                        help="Export Chrome trace files for detailed analysis")
    parser.add_argument("--compare_with_previous", action="store_true",
                        help="Compare with previously saved results to track improvements")
    parser.add_argument("--save_generated_text", action="store_true",
                        help="Save generated text samples for quality comparison")
    parser.add_argument("--export_csv", action="store_true",
                        help="Export results in CSV format for easier analysis")
    
    return parser.parse_args()


def prepare_input_data(args):
    """Prepare input data for profiling."""
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Select prompts based on args
    prompts = []
    
    if args.custom_prompts and os.path.exists(args.custom_prompts):
        # Load custom prompts from file
        try:
            with open(args.custom_prompts, 'r') as f:
                file_prompts = [line.strip() for line in f.readlines() if line.strip()]
                if file_prompts:
                    prompts = file_prompts[:min(5, len(file_prompts))]  # Use up to 5 prompts
        except Exception as e:
            print(f"Error loading custom prompts: {e}")
    
    # Use default prompt if no custom prompts available
    if not prompts:
        prompts = [
            "The transformer model architecture revolutionized natural language processing by",
            "In the field of machine learning, recent developments have shown that",
            "When considering the challenges of artificial intelligence, researchers must",
            "The integration of neural networks with symbolic reasoning enables",
            "To effectively implement attention mechanisms in transformers, one should"
        ]
    
    # Just use the first prompt if not doing comprehensive tests
    if args.profile_mode not in ["throughput", "all"]:
        prompts = [prompts[0]]
    
    # Process all prompts
    all_inputs = []
    for prompt in prompts:
        # Tokenize with sequence length constraint
        if args.sequence_length > 0:
            # Ensure prompt fits within sequence length
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            if input_ids.size(1) > args.sequence_length:
                # Truncate if too long
                input_ids = input_ids[:, :args.sequence_length]
            elif input_ids.size(1) < args.sequence_length and args.sequence_length <= 1024:
                # Pad with random tokens to desired length for stress testing
                # But only do this for reasonable sequence lengths
                pad_length = args.sequence_length - input_ids.size(1)
                # Use random token IDs in the vocab range for padding
                vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50257
                random_tokens = torch.randint(100, min(vocab_size-1, 20000), (1, pad_length))
                input_ids = torch.cat([input_ids, random_tokens], dim=1)
        else:
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        # Adjust batch size if needed
        if args.batch_size > 1:
            input_ids = input_ids.repeat(args.batch_size, 1)
        
        # Move to device
        input_ids = input_ids.to(args.device)
        
        # Create attention mask (all 1s since we don't have padding)
        attention_mask = torch.ones_like(input_ids)
        
        all_inputs.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt": prompt
        })
    
    return {
        "inputs": all_inputs,
        "tokenizer": tokenizer,
        "primary_input": all_inputs[0]  # For backwards compatibility
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


def profile_inference(args, data, model_type="original", pruning_level=0, run_dir=None):
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
    
    # Extract inputs - handle both legacy and new format
    if "primary_input" in data:
        # New format
        primary_input = data["primary_input"]
        input_ids = primary_input["input_ids"]
        attention_mask = primary_input["attention_mask"]
        prompt = primary_input.get("prompt", "")
    else:
        # Legacy format
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        prompt = data.get("prompt", "")
    
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
    
    # Track memory usage
    memory_snapshots = []
    if args.memory_profile:
        memory_snapshots.append(get_memory_usage(args.device))
    
    # Warmup
    print("Warming up...")
    # Capture stdout/stderr to suppress transformers generation messages
    f = io.StringIO()
    with torch.no_grad(), redirect_stdout(f), redirect_stderr(f):
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
    tokens_generated = []
    
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
        
        # Calculate tokens generated
        num_tokens = output_ids.size(1) - input_ids.size(1)
        tokens_generated.append(num_tokens)
        
        # Capture memory usage after generation
        if args.memory_profile:
            memory_snapshots.append(get_memory_usage(args.device))
        
        print(f"  Iteration {i+1}: {generation_time:.4f}s ({num_tokens} tokens)")
    
    # Calculate average generation time
    avg_generation_time = sum(generation_times) / len(generation_times)
    avg_tokens = sum(tokens_generated) / len(tokens_generated)
    tokens_per_second = avg_tokens / avg_generation_time
    
    print(f"Average generation time: {avg_generation_time:.4f}s")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    
    # Check if detailed profile is available
    profile_data = None
    if args.profile_mode == "detailed" and args.device == "cuda" and torch.cuda.is_available():
        # Save trace if requested
        if args.trace_export:
            if run_dir:
                trace_dir = os.path.join(run_dir, "traces")
                os.makedirs(trace_dir, exist_ok=True)
                trace_path = os.path.join(trace_dir, f"{model_type}_pruning{pruning_level}_trace.json")
            else:
                # Fall back to output_dir if run_dir is not provided
                trace_dir = os.path.join(args.output_dir, "traces")
                os.makedirs(trace_dir, exist_ok=True)
                trace_path = os.path.join(trace_dir, f"{model_type}_pruning{pruning_level}_trace.json")
            prof.export_chrome_trace(trace_path)
            print(f"Trace exported to {trace_path}")
        
        # Get profile data
        profile_data = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
    
    # Extract generated text
    generated_text = None
    try:
        tokenizer = data["tokenizer"]
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except (KeyError, TypeError):
        pass
    
    # Get stats from model if available
    model_stats = {}
    if hasattr(model, "stats") and model.stats.get("total_time", 0) > 0:
        model_stats = model.stats.copy()
    
    # Get agency information if available
    agency_stats = None
    if hasattr(model, "get_agency_report"):
        agency_stats = model.get_agency_report()
    
    # Free memory
    del model
    del baseline_model
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Return results
    results = {
        "model_type": model_type,
        "pruning_level": pruning_level,
        "generation_times": generation_times,
        "avg_generation_time": avg_generation_time,
        "tokens_per_second": tokens_per_second,
        "first_token_time": first_token_times[0] if first_token_times else None,
        "profile_data": profile_data,
        "generated_text": generated_text,
        "model_stats": model_stats,
        "agency_stats": agency_stats,
        "tokens_generated": tokens_generated,
        "avg_tokens_generated": avg_tokens
    }
    
    # Add memory profiling data if collected
    if args.memory_profile and memory_snapshots:
        results["memory_profile"] = memory_snapshots
    
    return results


def profile_component_breakdown(args, data, run_dir=None):
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
        
        # Set model type and optimization level
        if model_type == "original":
            os.environ["USE_OPTIMIZED_MODEL"] = "0"
        else:  # optimized
            os.environ["USE_OPTIMIZED_MODEL"] = "1"
            os.environ["OPTIMIZATION_LEVEL"] = str(args.optimization_level)
        
        # Load models
        baseline_model = load_baseline_model(args.model_name, args.device)
        model = load_adaptive_model(args.model_name, baseline_model, args.device)
        
        # Apply pruning if needed for component tests
        # Testing with moderate pruning (30%) lets us see impacts more clearly
        if model_type == "optimized":
            pruned_model, _, _ = apply_pruning(model, 30)
            model = pruned_model
        
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
            "baseline_integration": [],
            "unet_connection": [],
            "layernorm": [],
            "other": []
        }
        
        # Enable debug mode to collect timing stats if model supports it
        if hasattr(model, "debug"):
            model.debug = True
            if hasattr(model, "reset_stats"):
                model.reset_stats()
        
        # Apply to blocks too if they have debug flag
        for block in blocks:
            if hasattr(block, "debug"):
                block.debug = True
            if hasattr(block, "attn") and hasattr(block.attn, "profile_time"):
                block.attn.profile_time = True
        
        # Capture stdout/stderr to suppress transformers generation messages
        f = io.StringIO()
        with torch.no_grad(), redirect_stdout(f), redirect_stderr(f):
            # Warmup with full generation to ensure caches are properly initialized
            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.size(1) + 10,  # Just a few tokens for warmup
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )
            
            # Clear any warmup stats
            if hasattr(model, "reset_stats"):
                model.reset_stats()
            
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            # Profile full forward pass with detailed breakdown
            batch_size, seq_len = input_ids.shape
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            
            # Create causal attention mask
            if seq_len <= 1024:
                # Use pre-computed mask if model has one
                if hasattr(model, "bias"):
                    attn_mask = model.bias[:, :, :seq_len, :seq_len]
                else:
                    # Create standard causal mask
                    attn_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
                attn_mask = (1.0 - attn_mask) * -10000.0
            else:
                # Create mask on-the-fly for longer sequences
                attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device) * -10000.0, diagonal=1)
            
            # Profile embeddings
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                # Get input embeddings
                with record_function("embeddings"):
                    embeddings = model.wte(input_ids) + model.wpe(position_ids)
            
            # Extract times
            for event in prof.key_averages():
                if event.key == "embeddings":
                    component_times["embeddings"].append(event.cuda_time_total / 1000)  # Convert to ms
            
            # Initialize hidden states
            hidden_states = embeddings
            
            # Encoder outputs for UNet connections
            encoder_outputs = {}
            
            # Baseline outputs if using baseline integration
            baseline_outputs = None
            if model_type == "optimized" and hasattr(model, "_get_baseline_states"):
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    with record_function("baseline_integration"):
                        baseline_outputs = model._get_baseline_states(input_ids, attention_mask)
                
                # Record time for baseline integration
                for event in prof.key_averages():
                    if event.key == "baseline_integration":
                        component_times["baseline_integration"].append(event.cuda_time_total / 1000)
            
            # Process through each block with profiling for each component
            midpoint = len(blocks) // 2
            
            for i, block in enumerate(blocks):
                # Track UNet encoder outputs
                if i < midpoint:
                    encoder_outputs[i] = hidden_states.clone()
                
                # Profile entire block forward to measure full block time
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    with record_function(f"block_{i}_full"):
                        # Get baseline hidden states for this layer if available
                        baseline_states = None
                        if baseline_outputs is not None and i in baseline_outputs:
                            baseline_states = baseline_outputs[i]
                        
                        # Get encoder states for UNet skip connection if this is a decoder layer
                        encoder_states = None
                        if (hasattr(block, "use_skip_connection") and 
                            hasattr(block, "skip_source") and 
                            block.use_skip_connection and 
                            block.skip_source in encoder_outputs):
                            encoder_states = encoder_outputs[block.skip_source]
                        
                        # Process through the block - trace with nested records
                        with record_function(f"block_{i}_all_components"):
                            # Layer norm 1
                            with record_function(f"block_{i}_ln1"):
                                norm_hidden = block.ln1(hidden_states)
                            
                            # Attention
                            with record_function(f"block_{i}_attention"):
                                if hasattr(block, "attn"):
                                    attn_output = block.attn(norm_hidden, attn_mask=attn_mask)
                                elif hasattr(block, "attention"):
                                    attn_output = block.attention(norm_hidden, attn_mask=attn_mask)
                            
                            # Add residual
                            residual_out = hidden_states + attn_output
                            
                            # Process baseline integration if it exists
                            if baseline_states is not None and hasattr(block, "use_baseline_integration") and block.use_baseline_integration:
                                with record_function(f"block_{i}_baseline_integration"):
                                    if hasattr(block, "ln_baseline") and hasattr(block, "baseline_adapter"):
                                        adapted_baseline = block.baseline_adapter(block.ln_baseline(baseline_states))
                                        gate_value = torch.sigmoid(block.baseline_gate)
                                        residual_out = residual_out * (1 - gate_value) + adapted_baseline * gate_value
                            
                            # Process UNet connection if it exists
                            if encoder_states is not None and hasattr(block, "use_skip_connection") and block.use_skip_connection:
                                with record_function(f"block_{i}_unet_connection"):
                                    if hasattr(block, "skip_fuse"):
                                        combined = torch.cat([residual_out, encoder_states], dim=-1)
                                        fusion_output = block.skip_fuse(combined)
                                        residual_out = residual_out + fusion_output * block.skip_scale
                            
                            # Layer norm 2
                            with record_function(f"block_{i}_ln2"):
                                norm_hidden = block.ln2(residual_out)
                            
                            # FFN
                            with record_function(f"block_{i}_ffn"):
                                if hasattr(block, "ffn"):
                                    ffn_output = block.ffn(norm_hidden)
                                elif hasattr(block, "mlp"):
                                    ffn_output = block.mlp(norm_hidden)
                            
                            # Final addition
                            hidden_states = residual_out + ffn_output
                
                # Extract component times from the profile
                for event in prof.key_averages():
                    if "attention" in event.key:
                        component_times["attention"].append(event.cuda_time_total / 1000)
                    elif "ffn" in event.key:
                        component_times["ffn"].append(event.cuda_time_total / 1000)
                    elif "ln" in event.key:
                        component_times["layernorm"].append(event.cuda_time_total / 1000)
                    elif "baseline_integration" in event.key:
                        component_times["baseline_integration"].append(event.cuda_time_total / 1000)
                    elif "unet_connection" in event.key:
                        component_times["unet_connection"].append(event.cuda_time_total / 1000)
            
            # Profile final layer norm and output projection
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                with record_function("final_ln"):
                    hidden_states = model.ln_f(hidden_states)
                
                with record_function("lm_head"):
                    logits = model.lm_head(hidden_states)
            
            # Extract times
            for event in prof.key_averages():
                if event.key == "final_ln":
                    component_times["layernorm"].append(event.cuda_time_total / 1000)
        
        # Get stats from model if available
        model_stats = {}
        if hasattr(model, "stats") and model.stats.get("total_time", 0) > 0:
            model_stats = model.stats.copy()
        
        # Get agency information if available
        agency_stats = None
        if hasattr(model, "get_agency_report"):
            agency_stats = model.get_agency_report()
        
        # Calculate average times
        avg_times = {}
        total_times = {}
        percentages = {}
        
        for component, times in component_times.items():
            if times:
                avg_times[component] = sum(times) / len(times)
                total_times[component] = sum(times)
            else:
                avg_times[component] = 0
                total_times[component] = 0
        
        # Calculate component percentages
        total_time = sum(total_times.values())
        if total_time > 0:
            for component, time in total_times.items():
                percentages[component] = (time / total_time) * 100
        
        # Store results
        results[model_type] = {
            "component_times": component_times,
            "avg_times": avg_times,
            "total_times": total_times,
            "percentages": percentages,
            "model_stats": model_stats,
            "agency_stats": agency_stats
        }
        
        # Free memory
        del model
        del baseline_model
        torch.cuda.empty_cache()
    
    return results


def test_integration_optimizations(args, data, run_dir=None):
    """Test each integration optimization to isolate its impact."""
    print("\n==== Testing Integration Optimizations ====")
    
    if not args.test_integration_points:
        print("Skipping integration optimization tests (use --test_integration_points to run)")
        return None
    
    # Optimization configurations to test
    configs = [
        {"name": "original", "optimized": False, "baseline": True, "unet": True, "level": 1},
        {"name": "optimized_all", "optimized": True, "baseline": True, "unet": True, "level": args.optimization_level},
        {"name": "opt_no_baseline", "optimized": True, "baseline": False, "unet": True, "level": args.optimization_level},
        {"name": "opt_no_unet", "optimized": True, "baseline": True, "unet": False, "level": args.optimization_level},
        {"name": "opt_minimal", "optimized": True, "baseline": False, "unet": False, "level": args.optimization_level}
    ]
    
    # Add CPU-specific configurations if requested
    if args.cpu_specific_optimizations and args.device == "cpu":
        configs.extend([
            {"name": "cpu_optimized_level3", "optimized": True, "baseline": False, "unet": False, "level": 3},
            {"name": "cpu_optimized_aggressive", "optimized": True, "baseline": False, "unet": False, "level": 3, 
             "cpu_aggressive": True}
        ])
    
    # Pruning levels to test - focus on a smaller set for efficiency
    pruning_levels = [0, 50]
    
    # Store results
    results = {}
    
    for config in configs:
        print(f"\nTesting {config['name']} configuration")
        
        # Set environment variables for this configuration
        if config["optimized"]:
            os.environ["USE_OPTIMIZED_MODEL"] = "1"
            os.environ["OPTIMIZATION_LEVEL"] = str(config["level"])
            
            # Set CPU-specific optimizations if requested
            if args.device == "cpu" and config.get("cpu_aggressive", False):
                os.environ["CPU_AGGRESSIVE_OPTIMIZATIONS"] = "1"
        else:
            os.environ["USE_OPTIMIZED_MODEL"] = "0"
            
        # Reset any special env vars if not applicable
        if not config.get("cpu_aggressive", False):
            os.environ.pop("CPU_AGGRESSIVE_OPTIMIZATIONS", None)
        
        config_results = {}
        
        for level in pruning_levels:
            print(f"  - With {level}% pruning")
            
            # Load models
            baseline_model = load_baseline_model(args.model_name, args.device)
            model = load_adaptive_model(args.model_name, baseline_model, args.device)
            
            # Handle baseline integration
            if not config["baseline"]:
                # Disable baseline integration
                if hasattr(model, "baseline_model"):
                    model.baseline_model = None
                    model.use_baseline_integration = False
                elif hasattr(model, "model") and hasattr(model.model, "baseline_model"):
                    model.model.baseline_model = None
                    model.model.use_baseline_integration = False
            
            # Handle UNet connections
            if not config["unet"]:
                # Disable UNet connections
                for block in (model.blocks if hasattr(model, "blocks") else 
                             model.model.blocks if hasattr(model, "model") and hasattr(model.model, "blocks") else []):
                    if hasattr(block, "use_skip_connection"):
                        block.use_skip_connection = False
            
            # Apply pruning if needed
            if level > 0:
                model, pruned_count, pruned_heads = apply_pruning(model, level)
            
            # Use primary input from data
            primary_input = data["primary_input"]
            input_ids = primary_input["input_ids"]
            attention_mask = primary_input["attention_mask"]
            
            # Track memory usage throughout
            memory_usage = []
            peak_memory = 0
            
            # Warmup
            print("    Warming up...")
            # Capture stdout/stderr to suppress transformers generation messages
            f = io.StringIO()
            with torch.no_grad(), redirect_stdout(f), redirect_stderr(f):
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
            tokens_generated = []
            
            print(f"    Running {args.iterations} iterations...")
            
            # Additional tracking for memory profiling
            memory_snapshots = []
            if args.memory_profile:
                memory_snapshots.append(get_memory_usage(args.device))
            
            for i in range(args.iterations):
                # Clear CUDA cache if available
                if args.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                
                # Start timing
                start_time = time.time()
                
                # Capture stdout/stderr to suppress transformers generation messages
                f = io.StringIO()
                with torch.no_grad(), redirect_stdout(f), redirect_stderr(f):
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
                tokens_generated.append(output_ids.size(1) - input_ids.size(1))
                
                # Capture memory usage after generation
                if args.memory_profile:
                    memory_snapshots.append(get_memory_usage(args.device))
                
                print(f"      Iteration {i+1}: {generation_time:.4f}s ({output_ids.size(1) - input_ids.size(1)} tokens)")
            
            # Calculate average generation time
            avg_generation_time = sum(generation_times) / len(generation_times)
            avg_tokens = sum(tokens_generated) / len(tokens_generated)
            tokens_per_second = avg_tokens / avg_generation_time
            
            print(f"    Average generation time: {avg_generation_time:.4f}s")
            print(f"    Tokens per second: {tokens_per_second:.2f}")
            
            # Get stats from model if available
            model_stats = {}
            if hasattr(model, "stats") and model.stats.get("total_time", 0) > 0:
                model_stats = model.stats.copy()
            
            # Get agency information if available
            agency_stats = None
            if hasattr(model, "get_agency_report"):
                agency_stats = model.get_agency_report()
            
            # Store results
            config_results[level] = {
                "generation_times": generation_times,
                "avg_generation_time": avg_generation_time,
                "tokens_per_second": tokens_per_second,
                "first_token_time": first_token_times[0] if first_token_times else None,
                "model_stats": model_stats,
                "agency_stats": agency_stats
            }
            
            # Add memory profiling data if collected
            if args.memory_profile and memory_snapshots:
                config_results[level]["memory_profile"] = memory_snapshots
            
            # Free memory
            del model
            del baseline_model
            if args.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Store results for this configuration
        results[config["name"]] = config_results
    
    return results


def test_optimization_levels(args, data):
    """Comprehensively test all optimization levels to identify optimal settings."""
    print("\n==== Testing All Optimization Levels ====")
    
    if not args.test_all_optimization_levels:
        print("Skipping optimization level tests (use --test_all_optimization_levels to run)")
        return None
    
    # Optimization levels to test
    levels = [0, 1, 2, 3]
    
    # Fixed pruning level for this test to isolate effects
    pruning_level = 30
    
    # Store results for each level
    results = {}
    
    for level in levels:
        print(f"\nTesting optimization level {level}")
        
        # Set environment variables for this optimization level
        os.environ["USE_OPTIMIZED_MODEL"] = "1"
        os.environ["OPTIMIZATION_LEVEL"] = str(level)
        
        # Load models
        baseline_model = load_baseline_model(args.model_name, args.device)
        model = load_adaptive_model(args.model_name, baseline_model, args.device)
        
        # Apply pruning
        if pruning_level > 0:
            model, pruned_count, pruned_heads = apply_pruning(model, pruning_level)
        
        # Use primary input from data
        primary_input = data["primary_input"]
        input_ids = primary_input["input_ids"]
        attention_mask = primary_input["attention_mask"]
        
        # Track memory usage
        model_mem_usage = get_memory_usage(args.device)
        
        # Warmup
        print("  Warming up...")
        # Capture stdout/stderr to suppress transformers generation messages
        f = io.StringIO()
        with torch.no_grad(), redirect_stdout(f), redirect_stderr(f):
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
        tokens_generated = []
        
        print(f"  Running {args.iterations} iterations...")
        
        for i in range(args.iterations):
            # Clear CUDA cache if available
            if args.device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            # Start timing
            start_time = time.time()
            
            # Capture stdout/stderr to suppress transformers generation messages
            f = io.StringIO()
            with torch.no_grad(), redirect_stdout(f), redirect_stderr(f):
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
            tokens_generated.append(output_ids.size(1) - input_ids.size(1))
            
            print(f"    Iteration {i+1}: {generation_time:.4f}s ({output_ids.size(1) - input_ids.size(1)} tokens)")
        
        # Calculate average generation time
        avg_generation_time = sum(generation_times) / len(generation_times)
        avg_tokens = sum(tokens_generated) / len(tokens_generated)
        tokens_per_second = avg_tokens / avg_generation_time
        
        # Get generated text
        generated_text = None
        if args.save_generated_text:
            tokenizer = data["tokenizer"]
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        print(f"  Average generation time: {avg_generation_time:.4f}s")
        print(f"  Tokens per second: {tokens_per_second:.2f}")
        
        # Get stats from model if available
        model_stats = {}
        if hasattr(model, "stats") and model.stats.get("total_time", 0) > 0:
            model_stats = model.stats.copy()
        
        # Get agency information if available
        agency_stats = None
        if hasattr(model, "get_agency_report"):
            agency_stats = model.get_agency_report()
        
        # Store results
        results[level] = {
            "generation_times": generation_times,
            "avg_generation_time": avg_generation_time,
            "tokens_per_second": tokens_per_second,
            "first_token_time": first_token_times[0] if first_token_times else None,
            "memory_usage": model_mem_usage,
            "model_stats": model_stats,
            "agency_stats": agency_stats,
            "generated_text": generated_text
        }
        
        # Free memory
        del model
        del baseline_model
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Add comparison of levels
    optimization_comparison = {}
    for level in levels:
        level_data = results[level]
        optimization_comparison[level] = {
            "tokens_per_second": level_data["tokens_per_second"],
            "first_token_latency": level_data["first_token_time"]
        }
    
    # Add comparison to results
    results["comparison"] = optimization_comparison
    
    return results


def get_memory_usage(device):
    """Get memory usage for current process and device."""
    memory_data = {
        "timestamp": time.time(),
        "cpu": {
            "percent": psutil.Process().cpu_percent(),
            "rss_mb": psutil.Process().memory_info().rss / (1024 * 1024)
        }
    }
    
    # Add GPU info if applicable
    if device == "cuda" and torch.cuda.is_available():
        memory_data["gpu"] = {
            "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
            "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
            "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024)
        }
    
    return memory_data


def compare_pruning_levels(args, data, run_dir=None):
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
        # Set optimization level for optimized model
        if model_type == "optimized":
            os.environ["OPTIMIZATION_LEVEL"] = str(args.optimization_level)
            
        for level in pruning_levels:
            print(f"\nTesting {model_type} model with {level}% pruning")
            
            # Profile inference for this configuration
            inference_results = profile_inference(args, data, model_type, level, run_dir)
            
            # Store results
            if isinstance(level, int):
                results[model_type][level] = inference_results
            else:
                # Use string representation for JSON compatibility
                results[model_type][str(level)] = inference_results
    
    return results


def visualize_results(results, args, charts_dir):
    """Create visualizations from profiling results."""
    if not args.visualize:
        return
    
    print("\n==== Creating Visualizations ====")
    os.makedirs(charts_dir, exist_ok=True)
    
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
        plt.savefig(os.path.join(charts_dir, "model_loading_comparison.png"), dpi=150)
        plt.close()
        
    # Multi-Model Comparison Visualizations
    if "multi_model_comparison" in results and results["multi_model_comparison"]:
        data = results["multi_model_comparison"]
        
        # Skip if no valid data
        if not data:
            return
            
        # Extract model names
        model_names = list(data.keys())
        
        # Create figure for multi-model comparison
        plt.figure(figsize=(15, 10))
        
        # 1. Parameter Count Comparison
        plt.subplot(2, 2, 1)
        
        baseline_params = [data[model]["parameter_counts"]["baseline"] / 1000000 for model in model_names]  # Convert to millions
        adaptive_params = [data[model]["parameter_counts"]["adaptive"] / 1000000 for model in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, baseline_params, width, label='Baseline', color="dodgerblue")
        plt.bar(x + width/2, adaptive_params, width, label='Adaptive', color="green")
        
        plt.title('Parameter Count by Model')
        plt.ylabel('Parameters (Millions)')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add parameter increase percentage annotations
        for i, model in enumerate(model_names):
            increase = data[model]["parameter_counts"]["increase_percentage"]
            plt.annotate(f"+{increase:.1f}%", 
                         xy=(i, adaptive_params[i]), 
                         xytext=(0, 5),
                         textcoords="offset points",
                         ha='center', va='bottom',
                         fontsize=8,
                         color='darkgreen')
        
        # 2. Loading Time Comparison
        plt.subplot(2, 2, 2)
        
        baseline_load = [data[model]["loading"]["baseline_model"]["load_time"] for model in model_names]
        adaptive_load = [data[model]["loading"]["optimized_model"]["load_time"] for model in model_names]
        
        plt.bar(x - width/2, baseline_load, width, label='Baseline', color="dodgerblue")
        plt.bar(x + width/2, adaptive_load, width, label='Adaptive', color="green")
        
        plt.title('Model Loading Time')
        plt.ylabel('Time (seconds)')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        
        # 3. Generation Speed Comparison
        plt.subplot(2, 2, 3)
        
        gen_speed = [data[model]["inference"]["tokens_per_second"] for model in model_names]
        
        bars = plt.bar(model_names, gen_speed, color="coral")
        plt.title('Generation Speed (Tokens/Second)')
        plt.ylabel('Tokens per Second')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f"{height:.2f}", ha='center', va='bottom')
        
        # 4. First Token Latency
        plt.subplot(2, 2, 4)
        
        latency = [data[model]["inference"]["first_token_time"] for model in model_names]
        
        bars = plt.bar(model_names, latency, color="purple")
        plt.title('First Token Latency')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f"{height:.4f}s", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "multi_model_comparison.png"), dpi=150)
        plt.close()
        
        # Create a parameter growth figure specifically
        plt.figure(figsize=(10, 6))
        
        # Sort models by parameter count for clearer visualization
        sorted_idx = np.argsort(baseline_params)
        sorted_models = [model_names[i] for i in sorted_idx]
        sorted_baseline = [baseline_params[i] for i in sorted_idx]
        sorted_adaptive = [adaptive_params[i] for i in sorted_idx]
        sorted_increase = [data[model_names[i]]["parameter_counts"]["increase_percentage"] for i in sorted_idx]
        
        # Plot parameter counts
        x = np.arange(len(sorted_models))
        plt.bar(x - width/2, sorted_baseline, width, label='Baseline', color="dodgerblue")
        plt.bar(x + width/2, sorted_adaptive, width, label='Adaptive', color="green")
        
        plt.title('Parameter Count by Model (Sorted by Size)')
        plt.ylabel('Parameters (Millions)')
        plt.xticks(x, sorted_models, rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add parameter values and increase percentage
        for i, (base, adaptive, increase) in enumerate(zip(sorted_baseline, sorted_adaptive, sorted_increase)):
            plt.annotate(f"{base:.1f}M", 
                         xy=(i - width/2, base), 
                         xytext=(0, 5),
                         textcoords="offset points",
                         ha='center', va='bottom',
                         fontsize=8)
            
            plt.annotate(f"{adaptive:.1f}M\n(+{increase:.1f}%)", 
                         xy=(i + width/2, adaptive), 
                         xytext=(0, 5),
                         textcoords="offset points",
                         ha='center', va='bottom',
                         fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "parameter_growth.png"), dpi=150)
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
        plt.savefig(os.path.join(charts_dir, "pruning_performance.png"), dpi=150)
        plt.close()
    
    # 3. Component Breakdown
    if "component_breakdown" in results:
        data = results["component_breakdown"]
        
        # Check if data is available
        if not data:
            return
        
        # Create figure
        plt.figure(figsize=(14, 12))
        
        # Extract component data for original model
        orig_data = data["original"]
        
        # Get more components from the optimized component breakdown
        components = ["embeddings", "attention", "ffn", "baseline_integration", "unet_connection", "layernorm", "other"]
        
        # Get times from percentages for better visualization
        if "percentages" in orig_data and "percentages" in data["optimized"]:
            orig_times = [orig_data["percentages"].get(comp, 0) for comp in components]
            opt_times = [data["optimized"]["percentages"].get(comp, 0) for comp in components]
        else:
            # Fall back to avg_times
            orig_times = [orig_data["avg_times"].get(comp, 0) for comp in components]
            opt_times = [data["optimized"]["avg_times"].get(comp, 0) for comp in components]
        
        # Filter out zero or very small values
        valid_components = []
        valid_orig = []
        valid_opt = []
        for i, comp in enumerate(components):
            if orig_times[i] > 0.1 or opt_times[i] > 0.1:
                valid_components.append(comp)
                valid_orig.append(orig_times[i])
                valid_opt.append(opt_times[i])
        
        # Plot component comparison by percentage
        plt.subplot(2, 2, 1)
        x = np.arange(len(valid_components))
        width = 0.35
        
        plt.bar(x - width/2, valid_orig, width, label='Original', color="dodgerblue")
        plt.bar(x + width/2, valid_opt, width, label='Optimized', color="green")
        
        if "percentages" in orig_data:
            plt.ylabel('Time (%)')
            plt.title('Component Time Distribution')
        else:
            plt.ylabel('Time (ms)')
            plt.title('Component Execution Time')
            
        plt.xticks(x, [c.replace('_', ' ').title() for c in valid_components], rotation=45, ha='right')
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
        plt.subplot(2, 2, 2)
        bars = plt.bar([c.replace('_', ' ').title() for c in valid_components], speedups, color="coral")
        plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
        plt.ylabel('Speedup Factor')
        plt.title('Component Speedup (Original ÷ Optimized)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f"{height:.2f}x", ha='center', va='bottom')
        
        # Plot component timing for each layer in optimized model
        if "component_times" in data["optimized"]:
            opt_data = data["optimized"]
            
            # Get attention and FFN times by layer
            attn_times = opt_data["component_times"].get("attention", [])
            ffn_times = opt_data["component_times"].get("ffn", [])
            baseline_times = opt_data["component_times"].get("baseline_integration", [])
            unet_times = opt_data["component_times"].get("unet_connection", [])
            
            # Only continue if we have layer-by-layer data
            if attn_times or ffn_times:
                plt.subplot(2, 2, 3)
                
                # Get max length for uniform arrays
                max_len = max(len(attn_times), len(ffn_times))
                
                # Pad arrays if needed for plotting
                attn_times = attn_times + [0] * (max_len - len(attn_times))
                ffn_times = ffn_times + [0] * (max_len - len(ffn_times))
                
                # Plot by layer
                layers = list(range(max_len))
                plt.plot(layers, attn_times, 'o-', label="Attention", color="blue")
                plt.plot(layers, ffn_times, 'o-', label="FFN", color="red")
                
                # Add baseline and unet if they exist
                if baseline_times:
                    baseline_times = baseline_times + [0] * (max_len - len(baseline_times))
                    plt.plot(layers, baseline_times, 'o-', label="Baseline", color="purple")
                
                if unet_times:
                    unet_times = unet_times + [0] * (max_len - len(unet_times))
                    plt.plot(layers, unet_times, 'o-', label="UNet", color="orange")
                
                plt.title('Component Times by Layer')
                plt.xlabel('Layer')
                plt.ylabel('Time (ms)')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
        
        # Plot total time comparison
        if "model_stats" in data["optimized"] and "model_stats" in data["original"]:
            # Extract stats
            opt_stats = data["optimized"]["model_stats"]
            orig_stats = data["original"]["model_stats"]
            
            if opt_stats and orig_stats:
                plt.subplot(2, 2, 4)
                
                # Get total time and component times
                opt_total = opt_stats.get("total_time", 0)
                orig_total = orig_stats.get("total_time", 0)
                
                # Get component percentages
                opt_attn = opt_stats.get("attention_time", 0) / opt_total if opt_total else 0
                opt_baseline = opt_stats.get("baseline_time", 0) / opt_total if opt_total else 0
                opt_ffn = opt_stats.get("ffn_time", 0) / opt_total if opt_total else 0
                
                orig_attn = orig_stats.get("attention_time", 0) / orig_total if orig_total else 0
                orig_baseline = orig_stats.get("baseline_time", 0) / orig_total if orig_total else 0
                orig_ffn = orig_stats.get("ffn_time", 0) / orig_total if orig_total else 0
                
                # Normalize to percentages
                opt_attn *= 100
                opt_baseline *= 100
                opt_ffn *= 100
                orig_attn *= 100
                orig_baseline *= 100
                orig_ffn *= 100
                
                # Set up data
                labels = ['Attention', 'Baseline', 'FFN', 'Other']
                orig_vals = [orig_attn, orig_baseline, orig_ffn, 100 - (orig_attn + orig_baseline + orig_ffn)]
                opt_vals = [opt_attn, opt_baseline, opt_ffn, 100 - (opt_attn + opt_baseline + opt_ffn)]
                
                x = np.arange(len(labels))
                width = 0.35
                
                plt.bar(x - width/2, orig_vals, width, label='Original', color="dodgerblue")
                plt.bar(x + width/2, opt_vals, width, label='Optimized', color="green")
                
                plt.ylabel('Time (%)')
                plt.title('Time Distribution by Component')
                plt.xticks(x, labels)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "component_breakdown.png"), dpi=150)
        plt.close()
    
    # 4. Integration Optimization Test Results
    if "integration_tests" in results and results["integration_tests"]:
        data = results["integration_tests"]
        
        # Create figure for comparing integration optimizations
        plt.figure(figsize=(14, 10))
        
        # Get all configurations and pruning levels
        configs = list(data.keys())
        pruning_levels = list(data[configs[0]].keys())
        
        # Setup colors
        colors = {
            "original": "dodgerblue",
            "optimized_all": "green",
            "opt_no_baseline": "orange",
            "opt_no_unet": "purple",
            "opt_minimal": "red"
        }
        
        # 1. Tokens per second by configuration
        plt.subplot(2, 2, 1)
        
        # Get the tokens per second for each configuration at each pruning level
        for config in configs:
            tps_values = [data[config][level]["tokens_per_second"] for level in pruning_levels]
            plt.plot(pruning_levels, tps_values, 'o-', 
                    label=config.replace('_', ' ').title(), 
                    color=colors.get(config, "gray"),
                    linewidth=2)
        
        plt.title('Generation Speed by Configuration')
        plt.xlabel('Pruning Level (%)')
        plt.ylabel('Tokens per Second')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 2. Speedup relative to original
        plt.subplot(2, 2, 2)
        
        # Calculate speedup relative to original for each configuration
        for config in configs:
            if config == "original":
                continue  # Skip original as it's the baseline
                
            speedups = []
            for level in pruning_levels:
                orig_tps = data["original"][level]["tokens_per_second"]
                config_tps = data[config][level]["tokens_per_second"]
                speedups.append(config_tps / orig_tps if orig_tps > 0 else 0)
            
            plt.plot(pruning_levels, speedups, 'o-', 
                    label=config.replace('_', ' ').title(), 
                    color=colors.get(config, "gray"),
                    linewidth=2)
        
        plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
        plt.title('Speedup Relative to Original')
        plt.xlabel('Pruning Level (%)')
        plt.ylabel('Speedup Factor')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 3. First token latency
        plt.subplot(2, 2, 3)
        
        # Get first token latency for each configuration at each pruning level
        for config in configs:
            latency_values = [data[config][level]["first_token_time"] for level in pruning_levels]
            plt.plot(pruning_levels, latency_values, 'o-', 
                    label=config.replace('_', ' ').title(), 
                    color=colors.get(config, "gray"),
                    linewidth=2)
        
        plt.title('First Token Latency by Configuration')
        plt.xlabel('Pruning Level (%)')
        plt.ylabel('Time (seconds)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 4. Bar chart comparison at specific pruning level
        plt.subplot(2, 2, 4)
        
        # Choose a representative pruning level (e.g., 50%)
        level = 50
        if str(level) in pruning_levels:
            level = str(level)
        
        # Get tokens per second at this level
        tps_values = [data[config][level]["tokens_per_second"] for config in configs]
        
        # Create bar chart
        x = np.arange(len(configs))
        bars = plt.bar(x, tps_values, color=[colors.get(config, "gray") for config in configs])
        
        plt.title(f'Speed Comparison at {level}% Pruning')
        plt.ylabel('Tokens per Second')
        plt.xticks(x, [c.replace('_', ' ').title() for c in configs], rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f"{height:.1f}", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "integration_optimizations.png"), dpi=150)
        plt.close()
    
    print(f"Visualizations saved to {charts_dir}")


def compare_multiple_models(args):
    """Compare performance across multiple model architectures."""
    print("\n==== Multi-Model Architecture Comparison ====")
    
    if not args.multi_model_comparison:
        print("Skipping multi-model comparison (use --multi_model_comparison to run)")
        return None
    
    # Parse model list
    model_names = [name.strip() for name in args.compare_models.split(",")]
    
    # Ensure we have at least two models to compare
    if len(model_names) < 2:
        print("Error: Multi-model comparison requires at least two models")
        print(f"Only found: {model_names}")
        return None
        
    print(f"Comparing {len(model_names)} models: {', '.join(model_names)}")
    
    # Store results for each model
    results = {}
    model_families = {}
    
    # Test each model
    for model_name in model_names:
        print(f"\n--- Testing model: {model_name} ---")
        
        # Detect model family if not specified
        model_family = args.model_family
        if not model_family:
            if "gpt2" in model_name.lower():
                model_family = "gpt2"
            elif "bloom" in model_name.lower():
                model_family = "bloom"
            elif "opt" in model_name.lower():
                model_family = "opt"
            elif "pythia" in model_name.lower() or "neox" in model_name.lower():
                model_family = "pythia"
            elif "llama" in model_name.lower():
                model_family = "llama"
            else:
                model_family = "gpt2"  # Default to gpt2 handling
                
        print(f"Using model family handling: {model_family}")
        model_families[model_name] = model_family
                
        # Record start time for overall profiling
        start_time = time.time()
        
        # Set up a temporary modified args object for testing this model
        temp_args = argparse.Namespace(**vars(args))
        temp_args.model_name = model_name
        temp_args.model_family = model_family
        
        # Prepare input data specific to this model
        data = prepare_input_data(temp_args)
        
        # Profile model loading
        loading_results = profile_model_loading(temp_args)
        
        # Only test one pruning level (0%) for comparative clarity
        inference_results = profile_inference(temp_args, data, "optimized", 0, run_dir)
        
        # Store results together
        results[model_name] = {
            "model_family": model_family,
            "loading": loading_results,
            "inference": inference_results,
            "total_time": time.time() - start_time
        }
        
        # Add parameter counts
        results[model_name]["parameter_counts"] = {
            "baseline": loading_results["baseline_model"]["parameter_count"],
            "adaptive": loading_results["optimized_model"]["parameter_count"],
            "increase_percentage": ((loading_results["optimized_model"]["parameter_count"] - 
                                    loading_results["baseline_model"]["parameter_count"]) / 
                                    loading_results["baseline_model"]["parameter_count"] * 100)
        }
        
    return results


def main():
    """Main function."""
    # Set up logging and filter warnings
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Filter transformers warnings
    warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`")
    
    # Configure logging for transformers
    try:
        from transformers import logging as transformers_logging
        transformers_logging.set_verbosity_error()
    except ImportError:
        pass
        
    # Set general warnings filter
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings
    
    # Parse arguments
    args = setup_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create run-specific directory for better organization
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories for different types of outputs
    charts_dir = os.path.join(run_dir, "charts")
    data_dir = os.path.join(run_dir, "data")
    csv_dir = os.path.join(run_dir, "csv_reports")
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    # Create a unique prefix for files
    file_prefix = ""
    if args.output_prefix:
        file_prefix = f"{args.output_prefix}_"
    
    # Store all results
    results = {}
    
    # Record environment information
    results["environment"] = {
        "timestamp": time.time(),
        "date": datetime.datetime.now().isoformat(),
        "device": args.device,
        "cpu_info": {
            "count": psutil.cpu_count(logical=False),
            "logical_count": psutil.cpu_count(logical=True)
        },
        "memory_total_gb": psutil.virtual_memory().total / (1024**3)
    }
    
    # Add GPU info if applicable
    if args.device == "cuda" and torch.cuda.is_available():
        results["environment"]["gpu_info"] = {
            "name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
        }
    
    # Set optimization level environment variable
    os.environ["OPTIMIZATION_LEVEL"] = str(args.optimization_level)
    
    # Load previous results for comparison if requested
    previous_results = None
    if args.compare_with_previous:
        results_file = os.path.join(args.output_dir, "full_model_profiling.json")
        if os.path.exists(results_file):
            try:
                with open(results_file, "r") as f:
                    previous_results = json.load(f)
                print(f"Loaded previous results from {results_file} for comparison")
            except Exception as e:
                print(f"Error loading previous results: {e}")
    
    # If multi-model comparison is enabled, run that
    if args.multi_model_comparison:
        results["multi_model_comparison"] = compare_multiple_models(args)
    
    # Regular profiling flow
    # Profile model loading
    results["model_loading"] = profile_model_loading(args)
    
    # Prepare input data
    data = prepare_input_data(args)
    
    # Run profiling tests based on mode
    if args.profile_mode in ["all", "basic"]:
        # Compare pruning levels with basic generation metrics
        results["pruning_comparison"] = compare_pruning_levels(args, data, run_dir)
    
    if args.profile_mode in ["all", "component"]:
        # Profile component breakdown to identify bottlenecks
        results["component_breakdown"] = profile_component_breakdown(args, data, run_dir)
    
    if args.profile_mode in ["all"] or args.test_integration_points:
        # Test different integration optimizations
        results["integration_tests"] = test_integration_optimizations(args, data, run_dir)
    
    if args.profile_mode in ["all", "memory"]:
        # Additional memory profiling
        args.memory_profile = True
    
    if args.test_all_optimization_levels:
        # Test all optimization levels
        results["optimization_levels"] = test_optimization_levels(args, data)
    
    # Add throughput testing with multiple prompts and longer sequences
    if args.profile_mode in ["all", "throughput"]:
        # Temporarily modify args for throughput testing
        original_tokens = args.generated_tokens
        original_iterations = args.iterations
        
        try:
            args.generated_tokens = 100  # Generate more tokens for throughput test
            args.iterations = max(args.iterations, 3)  # Ensure enough iterations
            
            # Run throughput test across all inputs
            throughput_results = {}
            
            # Process for all inputs in parallelized batch
            total_tokens = 0
            total_time = 0
            
            all_inputs = data["inputs"]
            print(f"\n==== Throughput Testing with {len(all_inputs)} Prompts ====")
            
            for i, input_data in enumerate(all_inputs):
                print(f"\nPrompt {i+1}: {input_data['prompt'][:50]}...")
                
                # Create temporary modified args
                temp_data = {"primary_input": input_data, "tokenizer": data["tokenizer"]}
                
                # Profile using the optimized model
                os.environ["USE_OPTIMIZED_MODEL"] = "1"
                os.environ["OPTIMIZATION_LEVEL"] = str(args.optimization_level)
                
                # Run inference with current prompt
                inference_result = profile_inference(
                    args, temp_data, "optimized", 
                    pruning_level=30,  # Use fixed moderate pruning for throughput test
                    run_dir=run_dir
                )
                
                # Accumulate tokens and time
                throughput_results[f"prompt_{i}"] = {
                    "prompt": input_data["prompt"][:50] + "...",
                    "tokens_per_second": inference_result["tokens_per_second"],
                    "avg_generation_time": inference_result["avg_generation_time"]
                }
                
                total_tokens += args.generated_tokens
                total_time += inference_result["avg_generation_time"]
            
            # Calculate aggregate throughput
            if total_time > 0:
                overall_tokens_per_second = total_tokens / total_time
                throughput_results["overall"] = {
                    "total_tokens": total_tokens,
                    "total_time": total_time,
                    "tokens_per_second": overall_tokens_per_second
                }
                
                print(f"\nOverall throughput: {overall_tokens_per_second:.2f} tokens/sec")
            
            # Add to results
            results["throughput_test"] = throughput_results
            
        finally:
            # Restore original args
            args.generated_tokens = original_tokens
            args.iterations = original_iterations
    
    # Save timestamp for the results
    results["timestamp"] = time.time()
    results["args"] = {
        "model_name": args.model_name,
        "device": args.device,
        "optimization_level": args.optimization_level,
        "pruning_levels": args.pruning_levels,
        "profile_mode": args.profile_mode,
        "multi_model_comparison": args.multi_model_comparison,
        "compare_models": args.compare_models if args.multi_model_comparison else None,
        "model_family": args.model_family,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "generated_tokens": args.generated_tokens,
        "test_all_optimization_levels": args.test_all_optimization_levels,
        "cpu_specific_optimizations": args.cpu_specific_optimizations,
        "memory_profile": args.memory_profile
    }
    
    # Create results filename with timestamp
    results_file = os.path.join(data_dir, f"{file_prefix}full_model_profiling.json")
    
    # Create a summary file for quick access
    summary_file = os.path.join(run_dir, "summary.md")
    
    # Save results
    with open(results_file, "w") as f:
        # Filter out non-JSON serializable items
        filtered_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                filtered_results[key] = value
            elif key in ["timestamp", "args", "environment"]:
                filtered_results[key] = value
        json.dump(filtered_results, f, indent=2)
    
    print(f"\nResults saved to {run_dir}")
    
    # Generate markdown summary
    generate_summary_md(results, args, summary_file, run_dir)
    
    # Export to CSV if requested
    if args.export_csv:
        try:
            export_to_csv(results, args, file_prefix, csv_dir)
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
    
    # Create visualizations
    if args.visualize:
        visualize_results(results, args, charts_dir)
    
    # Print summary
    print("\n===== Profiling Summary =====")
    
    # Multi-model comparison summary (if available)
    if "multi_model_comparison" in results and results["multi_model_comparison"]:
        data = results["multi_model_comparison"]
        print("\nMulti-Model Comparison Summary:")
        
        # Table header
        print("\n{:<15} | {:<10} | {:<12} | {:<12} | {:<15} | {:<15}".format(
            "Model", "Family", "Base Params", "Adapt Params", "Param Growth", "Tokens/sec"))
        print("-" * 90)
        
        # Sort models by parameter count for clearer presentation
        model_names = list(data.keys())
        model_names.sort(key=lambda x: data[x]["parameter_counts"]["baseline"])
        
        for model_name in model_names:
            model_data = data[model_name]
            family = model_data["model_family"]
            baseline_params = model_data["parameter_counts"]["baseline"] / 1000000  # Convert to millions
            adaptive_params = model_data["parameter_counts"]["adaptive"] / 1000000
            param_growth = model_data["parameter_counts"]["increase_percentage"]
            tokens_per_sec = model_data["inference"]["tokens_per_second"]
            
            print("{:<15} | {:<10} | {:<12.2f}M | {:<12.2f}M | {:<15.2f}% | {:<15.2f}".format(
                model_name, family, baseline_params, adaptive_params, param_growth, tokens_per_sec))
        
        # Overall observations
        print("\nKey observations:")
        
        # Calculate average parameter growth
        avg_growth = sum(data[m]["parameter_counts"]["increase_percentage"] for m in model_names) / len(model_names)
        print(f"  • Average parameter increase: {avg_growth:.2f}%")
        
        # Find fastest model
        fastest_model = max(model_names, key=lambda m: data[m]["inference"]["tokens_per_second"])
        print(f"  • Fastest generation: {fastest_model} ({data[fastest_model]['inference']['tokens_per_second']:.2f} tokens/sec)")
        
        # Find model with lowest latency
        lowest_latency = min(model_names, key=lambda m: data[m]["inference"]["first_token_time"])
        print(f"  • Lowest first token latency: {lowest_latency} ({data[lowest_latency]['inference']['first_token_time']:.4f}s)")
    
    # Regular loading time summary
    if "model_loading" in results:
        loading_data = results["model_loading"]
        print("\nModel Loading:")
        for model_type in ["baseline_model", "original_model", "optimized_model"]:
            model_name = model_type.split("_")[0].capitalize()
            print(f"  {model_name}: {loading_data[model_type]['load_time']:.2f}s, "
                 f"{loading_data[model_type]['parameter_count']:,} parameters")
            
            # Compare with previous if available
            if previous_results and "model_loading" in previous_results:
                prev_load_time = previous_results["model_loading"][model_type]["load_time"]
                change = (loading_data[model_type]["load_time"] - prev_load_time) / prev_load_time * 100
                print(f"    {'↓' if change < 0 else '↑'} {abs(change):.1f}% from previous")
    
    # Pruning comparison summary
    if "pruning_comparison" in results:
        pruning_data = results["pruning_comparison"]
        print("\nPerformance by Pruning Level:")
        # Safely extract pruning levels
        try:
            if pruning_data and "original" in pruning_data and pruning_data["original"]:
                # Check if keys are integers or strings
                if isinstance(next(iter(pruning_data["original"].keys())), int):
                    pruning_levels = sorted(pruning_data["original"].keys())
                else:
                    pruning_levels = sorted([int(x) for x in pruning_data["original"].keys()])
                
                for level in pruning_levels:
                    level_key = level if isinstance(next(iter(pruning_data["original"].keys())), int) else str(level)
                    orig_speed = pruning_data["original"][level_key]["tokens_per_second"]
                    opt_speed = pruning_data["optimized"][level_key]["tokens_per_second"]
                    speedup = opt_speed / orig_speed
                    print(f"  Level {level}%: {speedup:.2f}x speedup ({opt_speed:.2f} vs {orig_speed:.2f} tokens/sec)")
                    
                    # Compare with previous if available
                    if (previous_results and "pruning_comparison" in previous_results and 
                        "optimized" in previous_results["pruning_comparison"] and 
                        level_key in previous_results["pruning_comparison"]["optimized"]):
                        
                        prev_opt_speed = previous_results["pruning_comparison"]["optimized"][level_key]["tokens_per_second"]
                        change = (opt_speed - prev_opt_speed) / prev_opt_speed * 100
                        print(f"    Optimized: {'↑' if change > 0 else '↓'} {abs(change):.1f}% from previous")
                        
                        prev_orig_speed = previous_results["pruning_comparison"]["original"][level_key]["tokens_per_second"]
                        prev_speedup = prev_opt_speed / prev_orig_speed
                        speedup_change = (speedup - prev_speedup) / prev_speedup * 100
                        print(f"    Speedup: {'↑' if speedup_change > 0 else '↓'} {abs(speedup_change):.1f}% from previous")
        except (KeyError, StopIteration, TypeError) as e:
            print(f"  Unable to print pruning summary: {e}")
    
    # Component breakdown summary
    if "component_breakdown" in results and results["component_breakdown"]:
        component_data = results["component_breakdown"]
        print("\nComponent Breakdown:")
        
        # Print component percentages if available
        if ("percentages" in component_data["optimized"] and 
            "percentages" in component_data["original"]):
            
            print("  Time Distribution (%):")
            opt_pct = component_data["optimized"]["percentages"]
            orig_pct = component_data["original"]["percentages"]
            
            # List components by importance
            components = ["attention", "ffn", "baseline_integration", "unet_connection", "layernorm", "embeddings"]
            
            for comp in components:
                if comp in opt_pct or comp in orig_pct:
                    orig_val = orig_pct.get(comp, 0)
                    opt_val = opt_pct.get(comp, 0)
                    print(f"  {comp.replace('_', ' ').title()}: {opt_val:.1f}% (optimized) vs {orig_val:.1f}% (original)")
        
        # Print speedups
        print("\n  Component Speedups:")
        components = ["attention", "ffn", "baseline_integration", "embeddings"]
        for comp in components:
            if comp in component_data["original"]["avg_times"] and comp in component_data["optimized"]["avg_times"]:
                orig_time = component_data["original"]["avg_times"][comp]
                opt_time = component_data["optimized"]["avg_times"][comp]
                if orig_time > 0:
                    speedup = orig_time / opt_time if opt_time > 0 else 0
                    print(f"  {comp.capitalize()}: {speedup:.2f}x speedup")
    
    # Integration optimization tests summary
    if "integration_tests" in results and results["integration_tests"]:
        integration_data = results["integration_tests"]
        print("\nIntegration Optimization Tests:")
        
        # Pruning level to focus on for detailed results
        focus_level = "50"
        if focus_level not in next(iter(integration_data.values())):
            focus_level = list(next(iter(integration_data.values())).keys())[0]
        
        # Get configurations and sort by performance
        configs = list(integration_data.keys())
        configs_by_speed = sorted(
            configs, 
            key=lambda c: integration_data[c][focus_level]["tokens_per_second"],
            reverse=True
        )
        
        print(f"\n  Performance at {focus_level}% pruning (sorted by speed):")
        for config in configs_by_speed:
            speed = integration_data[config][focus_level]["tokens_per_second"]
            # Calculate speedup over original
            if config != "original":
                orig_speed = integration_data["original"][focus_level]["tokens_per_second"]
                speedup = speed / orig_speed
                print(f"  {config.replace('_', ' ').title()}: {speed:.2f} tokens/sec ({speedup:.2f}x original)")
            else:
                print(f"  {config.replace('_', ' ').title()}: {speed:.2f} tokens/sec (baseline)")
        
        # Print most significant findings
        if "optimized_all" in integration_data and "opt_minimal" in integration_data:
            full_opt_speed = integration_data["optimized_all"][focus_level]["tokens_per_second"]
            minimal_opt_speed = integration_data["opt_minimal"][focus_level]["tokens_per_second"]
            
            print("\n  Key Insights:")
            if minimal_opt_speed > full_opt_speed:
                diff_pct = (minimal_opt_speed - full_opt_speed) / full_opt_speed * 100
                print(f"  • Minimal optimizations are {diff_pct:.1f}% faster than full optimizations")
                print(f"  → Consider removing baseline integration and UNet for better performance")
            elif full_opt_speed > minimal_opt_speed:
                diff_pct = (full_opt_speed - minimal_opt_speed) / minimal_opt_speed * 100
                print(f"  • Full optimizations are {diff_pct:.1f}% faster than minimal optimizations")
                print(f"  → The integration features are providing performance benefits")
        
        if "opt_no_baseline" in integration_data and "opt_no_unet" in integration_data:
            no_baseline_speed = integration_data["opt_no_baseline"][focus_level]["tokens_per_second"]
            no_unet_speed = integration_data["opt_no_unet"][focus_level]["tokens_per_second"]
            
            if no_baseline_speed > no_unet_speed:
                print(f"  • Removing baseline integration has more impact than removing UNet connections")
            else:
                print(f"  • Removing UNet connections has more impact than removing baseline integration")
        
    # Optimization level comparison summary
    if "optimization_levels" in results:
        opt_data = results["optimization_levels"]
        comparison = opt_data["comparison"]
        
        print("\nOptimization Level Comparison:")
        print("\n  {:<15} | {:<15} | {:<20}".format("Level", "Tokens/sec", "First Token Latency"))
        print("  " + "-" * 56)
        
        # Sort by performance
        levels = sorted(comparison.keys(), key=lambda l: comparison[l]["tokens_per_second"], reverse=True)
        
        for level in levels:
            if level == "comparison":
                continue
                
            level_data = comparison[level]
            print("  {:<15} | {:<15.2f} | {:<20.4f}s".format(
                level, level_data["tokens_per_second"], level_data["first_token_latency"]))
        
        # Print recommended level
        best_level = max([l for l in levels if l != "comparison"], 
                         key=lambda l: comparison[l]["tokens_per_second"])
        print(f"\n  Recommended optimization level: {best_level}")
        
        # Get best level for latency
        best_latency = min([l for l in levels if l != "comparison"], 
                          key=lambda l: comparison[l]["first_token_latency"])
        if best_latency != best_level:
            print(f"  Best level for latency: {best_latency}")
    
    # Throughput test summary
    if "throughput_test" in results:
        throughput_data = results["throughput_test"]
        
        if "overall" in throughput_data:
            overall = throughput_data["overall"]
            print(f"\nThroughput Test Summary:")
            print(f"  Overall: {overall['tokens_per_second']:.2f} tokens/sec " +
                 f"({overall['total_tokens']} tokens in {overall['total_time']:.2f}s)")
            
            # Print per-prompt breakdown
            prompt_keys = [k for k in throughput_data.keys() if k != "overall"]
            if prompt_keys:
                print("\n  Per-prompt performance:")
                for key in prompt_keys:
                    prompt_data = throughput_data[key]
                    print(f"  - \"{prompt_data['prompt']}\": " +
                         f"{prompt_data['tokens_per_second']:.2f} tokens/sec")


def generate_summary_md(results, args, summary_file, run_dir):
    """Generate a markdown summary of the profiling results."""
    with open(summary_file, 'w') as f:
        # Write header
        f.write(f"# Sentinel AI Model Profiling Results\n\n")
        f.write(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write environment info
        f.write(f"## Environment\n\n")
        f.write(f"- **Device:** {args.device}\n")
        f.write(f"- **Model:** {args.model_name}\n")
        f.write(f"- **Optimization Level:** {args.optimization_level}\n")
        f.write(f"- **Pruning Levels Tested:** {args.pruning_levels}\n")
        
        # Write model loading summary
        if "model_loading" in results:
            loading_data = results["model_loading"]
            f.write(f"\n## Model Loading\n\n")
            f.write("| Model Type | Loading Time | Parameters | Memory |\n")
            f.write("|------------|--------------|------------|--------|\n")
            
            for model_type in ["baseline_model", "original_model", "optimized_model"]:
                model_name = model_type.split("_")[0].capitalize()
                f.write(f"| {model_name} | {loading_data[model_type]['load_time']:.2f}s | " +
                       f"{loading_data[model_type]['parameter_count']:,} | " +
                       f"{loading_data[model_type]['memory_usage']/(1024**2):.2f} MB |\n")
        
        # Write pruning comparison
        if "pruning_comparison" in results:
            pruning_data = results["pruning_comparison"]
            f.write(f"\n## Pruning Performance\n\n")
            f.write("| Pruning Level | Original (tokens/sec) | Optimized (tokens/sec) | Speedup |\n")
            f.write("|---------------|------------------------|------------------------|---------|\n")
            
            # Get pruning levels
            if "original" in pruning_data and pruning_data["original"]:
                pruning_levels = sorted(pruning_data["original"].keys())
                
                for level in pruning_levels:
                    orig_speed = pruning_data["original"][level]["tokens_per_second"]
                    opt_speed = pruning_data["optimized"][level]["tokens_per_second"]
                    speedup = opt_speed / orig_speed
                    f.write(f"| {level}% | {orig_speed:.2f} | {opt_speed:.2f} | {speedup:.2f}x |\n")
        
        # Links to visualizations
        if args.visualize:
            f.write(f"\n## Visualizations\n\n")
            # Get a list of all PNG files in the charts directory
            chart_files = [f for f in os.listdir(os.path.join(run_dir, "charts")) if f.endswith(".png")]
            
            for chart_file in sorted(chart_files):
                display_name = chart_file.replace(".png", "").replace("_", " ").title()
                # Create a relative path for the markdown file
                relative_path = f"charts/{chart_file}"
                f.write(f"- [{display_name}]({relative_path})\n")
        
        # Add summary recommendations
        f.write(f"\n## Recommendations\n\n")
        
        # Find best pruning level
        best_pruning = "N/A"
        max_tps = 0
        if "pruning_comparison" in results and "optimized" in results["pruning_comparison"]:
            pruning_data = results["pruning_comparison"]["optimized"]
            
            for level, data in pruning_data.items():
                if data["tokens_per_second"] > max_tps:
                    max_tps = data["tokens_per_second"]
                    best_pruning = level
            
            f.write(f"- **Best Pruning Level:** {best_pruning}% (achieving {max_tps:.2f} tokens/sec)\n")
        
        # Optimization level recommendation
        if "optimization_levels" in results and "comparison" in results["optimization_levels"]:
            opt_data = results["optimization_levels"]["comparison"]
            best_level = max(opt_data.keys(), key=lambda l: opt_data[l]["tokens_per_second"])
            f.write(f"- **Recommended Optimization Level:** {best_level}\n")
        
        # Print test command for reproducibility
        f.write(f"\n## Reproduce This Test\n\n")
        f.write("```bash\n")
        f.write(f"python scripts/profile_full_model.py --model_name {args.model_name} --device {args.device} ")
        f.write(f"--optimization_level {args.optimization_level} --pruning_levels \"{args.pruning_levels}\" ")
        f.write(f"--profile_mode {args.profile_mode} ")
        
        # Add optional flags
        if args.visualize:
            f.write("--visualize ")
        if args.export_csv:
            f.write("--export_csv ")
        if args.memory_profile:
            f.write("--memory_profile ")
            
        f.write("\n```\n")
        
    print(f"Summary saved to {summary_file}")

def export_to_csv(results, args, file_prefix, csv_dir):
    """Export results to CSV files for easier analysis."""
    import csv
    
    # Ensure CSV directory exists
    os.makedirs(csv_dir, exist_ok=True)
    
    # Export pruning comparison if available
    if "pruning_comparison" in results:
        csv_path = os.path.join(csv_dir, f"{file_prefix}pruning_comparison.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Pruning Level", "Model Type", "Tokens per Second", "Generation Time", "First Token Time"])
            
            pruning_data = results["pruning_comparison"]
            for model_type in ["original", "optimized"]:
                if model_type in pruning_data:
                    for level, data in pruning_data[model_type].items():
                        writer.writerow([
                            level,
                            model_type,
                            data["tokens_per_second"],
                            data["avg_generation_time"],
                            data["first_token_time"]
                        ])
        
        print(f"Exported pruning comparison to {csv_path}")
    
    # Export optimization level comparison if available
    if "optimization_levels" in results:
        csv_path = os.path.join(csv_dir, f"{file_prefix}optimization_levels.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Optimization Level", "Tokens per Second", "First Token Time", "Memory Usage (MB)"])
            
            opt_data = results["optimization_levels"]
            for level, data in opt_data.items():
                if level == "comparison":
                    continue
                    
                # Extract memory usage
                memory_mb = 0
                if "memory_usage" in data:
                    memory_data = data["memory_usage"]
                    if "gpu" in memory_data:
                        memory_mb = memory_data["gpu"]["allocated_mb"]
                    else:
                        memory_mb = memory_data["cpu"]["rss_mb"]
                
                writer.writerow([
                    level,
                    data["tokens_per_second"],
                    data["first_token_time"],
                    memory_mb
                ])
        
        print(f"Exported optimization level comparison to {csv_path}")
    
    # Export integration test results if available
    if "integration_tests" in results:
        csv_path = os.path.join(csv_dir, f"{file_prefix}integration_tests.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "Configuration", "Pruning Level", "Tokens per Second", 
                "Generation Time", "First Token Time"
            ])
            
            integration_data = results["integration_tests"]
            for config, config_data in integration_data.items():
                for level, data in config_data.items():
                    writer.writerow([
                        config,
                        level,
                        data["tokens_per_second"],
                        data["avg_generation_time"],
                        data["first_token_time"]
                    ])
        
        print(f"Exported integration test results to {csv_path}")
    
    # Export throughput test results if available
    if "throughput_test" in results:
        csv_path = os.path.join(csv_dir, f"{file_prefix}throughput_test.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Prompt", "Tokens per Second", "Generation Time"])
            
            throughput_data = results["throughput_test"]
            for key, data in throughput_data.items():
                if key == "overall":
                    continue
                    
                writer.writerow([
                    data["prompt"],
                    data["tokens_per_second"],
                    data["avg_generation_time"]
                ])
            
            # Add overall summary
            if "overall" in throughput_data:
                overall = throughput_data["overall"]
                writer.writerow([
                    "OVERALL",
                    overall["tokens_per_second"],
                    overall["total_time"] / overall["total_tokens"]
                ])
        
        print(f"Exported throughput test results to {csv_path}")
    
    # Export model comparison if available
    if "multi_model_comparison" in results:
        csv_path = os.path.join(csv_dir, f"{file_prefix}model_comparison.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "Model", "Family", "Base Params (M)", "Adaptive Params (M)", 
                "Param Growth (%)", "Tokens per Second", "First Token Time"
            ])
            
            model_data = results["multi_model_comparison"]
            for model_name, data in model_data.items():
                baseline_params = data["parameter_counts"]["baseline"] / 1000000
                adaptive_params = data["parameter_counts"]["adaptive"] / 1000000
                param_growth = data["parameter_counts"]["increase_percentage"]
                
                writer.writerow([
                    model_name,
                    data["model_family"],
                    baseline_params,
                    adaptive_params,
                    param_growth,
                    data["inference"]["tokens_per_second"],
                    data["inference"]["first_token_time"]
                ])
        
        print(f"Exported model comparison to {csv_path}")
    
    # Create a summary CSV with key metrics
    csv_path = os.path.join(csv_dir, f"{file_prefix}summary.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Model", "Device", "Optimization Level", "Best Pruning Level",
            "Max Tokens/Sec", "Speedup vs Original", "First Token Latency"
        ])
        
        # Extract summary data
        model_name = args.model_name
        device = args.device
        opt_level = args.optimization_level
        
        # Find best pruning level
        best_pruning = "N/A"
        max_tps = 0
        speedup = 0
        latency = 0
        
        if "pruning_comparison" in results and "optimized" in results["pruning_comparison"]:
            pruning_data = results["pruning_comparison"]["optimized"]
            orig_data = results["pruning_comparison"]["original"]
            
            for level, data in pruning_data.items():
                if data["tokens_per_second"] > max_tps:
                    max_tps = data["tokens_per_second"]
                    best_pruning = level
                    if level in orig_data:
                        speedup = max_tps / orig_data[level]["tokens_per_second"]
                    latency = data["first_token_time"]
        
        writer.writerow([
            model_name,
            device,
            opt_level,
            best_pruning,
            max_tps,
            speedup,
            latency
        ])
    
    print(f"Exported summary to {csv_path}")


if __name__ == "__main__":
    main()