#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized Performance Test

This script verifies that the optimized model with the recommended settings
performs as expected based on our profiling results.
"""

import os
import sys
import time
import torch
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from models.loaders.loader import load_baseline_model, load_adaptive_model
from scripts.pruning_comparison.pruning_agency_comparison import apply_pruning

def setup_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test optimized model performance")
    
    parser.add_argument("--model_name", type=str, default="gpt2", 
                        help="Base model to use (default: gpt2)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run on (default: cpu)")
    parser.add_argument("--seq_len", type=int, default=64,
                        help="Input sequence length (default: 64)")
    parser.add_argument("--gen_tokens", type=int, default=20,
                        help="Number of tokens to generate (default: 20)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warmup iterations (default: 1)")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of test iterations (default: 3)")
    
    return parser.parse_args()

def create_sample_input(args):
    """Create sample input for testing."""
    from transformers import AutoTokenizer
    
    prompt = "The transformer model architecture revolutionized natural language processing by"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    input_ids = input_ids.to(args.device)
    attention_mask = torch.ones_like(input_ids)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "tokenizer": tokenizer,
        "prompt": prompt
    }

def test_configuration(config_name, model_type, optimization_level, pruning_level, args, input_data):
    """Test a specific model configuration."""
    print(f"\n=== Testing {config_name} ===")
    print(f"Model Type: {model_type}, Optimization Level: {optimization_level}, Pruning: {pruning_level}%")
    
    # Set environment variables
    if model_type == "original":
        os.environ["USE_OPTIMIZED_MODEL"] = "0"
    else:
        os.environ["USE_OPTIMIZED_MODEL"] = "1"
        os.environ["OPTIMIZATION_LEVEL"] = str(optimization_level)
    
    # Load models
    baseline_model = load_baseline_model(args.model_name, args.device)
    model = load_adaptive_model(args.model_name, baseline_model, args.device)
    
    # Apply pruning if needed
    if pruning_level > 0:
        model, pruned_count, _ = apply_pruning(model, pruning_level)
        print(f"Applied {pruning_level}% pruning, {pruned_count} heads pruned")
    
    # Extract inputs
    input_ids = input_data["input_ids"]
    attention_mask = input_data["attention_mask"]
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.size(1) + args.gen_tokens,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )
    
    # Measure performance
    generation_times = []
    
    print(f"Running {args.iterations} iterations...")
    for i in range(args.iterations):
        # Clear cache if needed
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        # Start timing
        start_time = time.time()
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.size(1) + args.gen_tokens,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )
        
        # End timing
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            
        generation_time = time.time() - start_time
        generation_times.append(generation_time)
        
        tokens_generated = output_ids.size(1) - input_ids.size(1)
        print(f"Iteration {i+1}: {generation_time:.4f}s ({tokens_generated} tokens)")
    
    # Calculate metrics
    avg_time = sum(generation_times) / len(generation_times)
    tokens_per_second = args.gen_tokens / avg_time
    
    # Generate output
    tokenizer = input_data["tokenizer"]
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Free memory
    del model
    del baseline_model
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Results
    results = {
        "config_name": config_name,
        "model_type": model_type,
        "optimization_level": optimization_level,
        "pruning_level": pruning_level,
        "avg_generation_time": avg_time,
        "tokens_per_second": tokens_per_second,
        "generated_text": generated_text
    }
    
    print(f"Average generation time: {avg_time:.4f}s")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    
    return results

def main():
    """Main function."""
    args = setup_args()
    input_data = create_sample_input(args)
    
    # Test configurations based on our profiling results
    configurations = [
        # Name, Model Type, Opt Level, Pruning
        ("Default Original", "original", 0, 0),
        ("Pruned Original", "original", 0, 70),
        ("Default Optimized", "optimized", 1, 0),
        ("Recommended CPU", "optimized", 2, 30),
        ("Max Performance", "optimized", 3, 70)
    ]
    
    # Store results
    results = []
    
    for config in configurations:
        result = test_configuration(*config, args, input_data)
        results.append(result)
    
    # Print comparison
    print("\n=== Performance Comparison ===")
    print(f"{'Configuration':<20} | {'Model Type':<10} | {'Opt Level':<9} | {'Pruning':<7} | {'Tokens/sec':<10}")
    print("-" * 70)
    
    for result in sorted(results, key=lambda x: x["tokens_per_second"], reverse=True):
        print(f"{result['config_name']:<20} | {result['model_type']:<10} | {result['optimization_level']:<9} | "
              f"{result['pruning_level']:<7}% | {result['tokens_per_second']:<10.2f}")
    
    # Verify that our recommended configuration performs well
    recommended = next((r for r in results if r["config_name"] == "Recommended CPU"), None)
    if recommended:
        baseline = next((r for r in results if r["config_name"] == "Default Original"), None)
        
        if baseline:
            speedup = recommended["tokens_per_second"] / baseline["tokens_per_second"]
            print(f"\nRecommended configuration is {speedup:.2f}x faster than baseline")
            
            # Check if it matches our profiling predictions
            if speedup > 1.2:
                print("✅ Recommended settings perform well as expected")
            else:
                print("⚠️ Performance is lower than expected from profiling")
    
    # Verify max performance configuration
    max_perf = next((r for r in results if r["config_name"] == "Max Performance"), None)
    if max_perf:
        if max_perf["tokens_per_second"] == max(r["tokens_per_second"] for r in results):
            print("✅ Maximum performance configuration is indeed the fastest")
        else:
            # Find what was fastest
            fastest = max(results, key=lambda x: x["tokens_per_second"])
            print(f"⚠️ Expected max performance config to be fastest, but {fastest['config_name']} was faster")

if __name__ == "__main__":
    main()