#!/usr/bin/env python
"""
Runtime Specialization Script for Sentinel-AI Agency

This script enables automatic detection and application of agency specialization
patterns at runtime, optimizing performance based on task type.

Features:
1. Auto-detection of task type from input text
2. Loading of optimal specialization patterns
3. Performance monitoring and metric tracking
4. Interactive mode for pattern testing
"""

import argparse
import time
import torch
import json
import os
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.loaders.gpt2_loader import load_gpt2_with_sentinel_gates
from models.specialization_registry import SpecializationRegistry
from models.agency_specialization import AgencySpecialization

def parse_args():
    parser = argparse.ArgumentParser(description="Runtime specialization for Sentinel-AI")
    
    parser.add_argument("--model", type=str, default="gpt2", 
                      help="Model to load (default: gpt2)")
    
    parser.add_argument("--prompt", type=str, 
                      help="Prompt for text generation")
    
    parser.add_argument("--task", type=str, choices=[
                      "pattern_matching", "logical_reasoning", 
                      "long_context", "creative_generation", 
                      "constrained_resources", "auto"],
                      default="auto",
                      help="Task type for specialization (default: auto-detect)")
    
    parser.add_argument("--output_file", type=str,
                      help="File to save generation output")
    
    parser.add_argument("--benchmark", action="store_true",
                      help="Run benchmarking on different specialization patterns")
    
    parser.add_argument("--interactive", action="store_true",
                      help="Run in interactive mode")
    
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug output")
    
    parser.add_argument("--max_tokens", type=int, default=100,
                      help="Maximum tokens to generate")
    
    parser.add_argument("--save_metrics", action="store_true",
                      help="Save performance metrics for pattern refinement")
    
    parser.add_argument("--agency_report", action="store_true",
                      help="Display detailed agency report after generation")
    
    return parser.parse_args()

def print_specialization_info(registry, specialization, task=None):
    """Print information about available specialization patterns."""
    print("\n📊 Specialization Patterns:")
    print("=" * 60)
    
    patterns = registry.get_available_patterns()
    for pattern_name, description in patterns.items():
        if task and pattern_name == task:
            print(f"▶️ {pattern_name}: {description} [SELECTED]")
        else:
            print(f"  {pattern_name}: {description}")
    
    if specialization and specialization.initialized:
        print("\n👤 Current Head Specialization:")
        print("-" * 60)
        report = specialization.get_specialization_report()
        
        if "categories" in report:
            for category, data in report["categories"].items():
                print(f"  {category}: {data['count']} heads ({data['percentage']*100:.1f}%)")
        
        if "state_distribution" in report:
            print("\n🔄 Current State Distribution:")
            for state, count in report["state_distribution"].items():
                print(f"  {state}: {count} heads")
    
    print("=" * 60)

def measure_generation_performance(model, tokenizer, prompt, max_tokens=100):
    """Measure generation performance metrics."""
    # Tokenize input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Measure generation time
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + max_tokens,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.8,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    end_time = time.time()
    
    # Calculate metrics
    generation_time = end_time - start_time
    generated_tokens = output.shape[1] - input_ids.shape[1]
    tokens_per_second = generated_tokens / generation_time if generation_time > 0 else 0
    
    # Get memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
    else:
        memory_allocated = 0
        memory_reserved = 0
    
    # Calculate simple diversity metrics
    generated_text = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    words = generated_text.split()
    unique_words = len(set(words))
    diversity_score = unique_words / len(words) if words else 0
    
    # Check for repetition
    repetition_detected = False
    ngram_sizes = [3, 4, 5]
    for n in ngram_sizes:
        if len(words) > n*2:
            ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
            ngram_counts = {}
            for ngram in ngrams:
                ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
            if max(ngram_counts.values(), default=0) > 1:
                repetition_detected = True
                break
    
    metrics = {
        "generation_time": generation_time,
        "generated_tokens": generated_tokens,
        "tokens_per_second": tokens_per_second,
        "memory_allocated_gb": memory_allocated,
        "memory_reserved_gb": memory_reserved,
        "diversity_score": diversity_score,
        "repetition_detected": repetition_detected
    }
    
    return generated_text, metrics

def benchmark_specialization_patterns(registry, model, tokenizer, prompt, max_tokens=100):
    """Benchmark different specialization patterns on the same prompt."""
    results = {}
    patterns = registry.get_available_patterns()
    
    print("\n🔍 Benchmarking Specialization Patterns...")
    print(f"Prompt: \"{prompt[:50]}{'...' if len(prompt) > 50 else ''}\"")
    print("-" * 60)
    
    for pattern_name in patterns.keys():
        print(f"Testing pattern: {pattern_name}...")
        
        # Apply the specialization pattern
        registry.apply_specialization(task_type=pattern_name)
        
        # Measure performance
        _, metrics = measure_generation_performance(model, tokenizer, prompt, max_tokens)
        
        # Store results
        results[pattern_name] = metrics
        
        print(f"  Tokens/sec: {metrics['tokens_per_second']:.2f}")
        print(f"  Memory: {metrics['memory_allocated_gb']:.2f} GB")
        print(f"  Diversity: {metrics['diversity_score']:.3f}")
        print("-" * 40)
    
    # Print comparison
    print("\n📊 Pattern Comparison Results:")
    print("-" * 60)
    print(f"{'Pattern':<20} {'Tokens/sec':<12} {'Memory (GB)':<12} {'Diversity':<10}")
    print("-" * 60)
    
    for pattern, metrics in results.items():
        print(f"{pattern:<20} {metrics['tokens_per_second']:<12.2f} {metrics['memory_allocated_gb']:<12.2f} {metrics['diversity_score']:<10.3f}")
    
    # Find the best pattern for different metrics
    best_speed = max(results.items(), key=lambda x: x[1]['tokens_per_second'])[0]
    best_memory = min(results.items(), key=lambda x: x[1]['memory_allocated_gb'])[0]
    best_diversity = max(results.items(), key=lambda x: x[1]['diversity_score'])[0]
    
    print("\n🏆 Best Patterns:")
    print(f"Speed: {best_speed}")
    print(f"Memory Efficiency: {best_memory}")
    print(f"Output Diversity: {best_diversity}")
    
    return results

def interactive_mode(registry, model, tokenizer, args):
    """Run in interactive mode with dynamic specialization control."""
    print("\n🤖 Sentinel-AI Interactive Mode")
    print("=" * 60)
    print("Enter prompts to generate text with specialized agency patterns.")
    print("Type 'exit' to quit, 'patterns' to see available patterns, 'task TYPE' to set task type.")
    print("=" * 60)
    
    current_task = "auto"
    
    while True:
        prompt = input("\n> ")
        
        if prompt.lower() == 'exit':
            break
            
        elif prompt.lower() == 'patterns':
            print_specialization_info(registry, registry.specialization, 
                                     registry.get_current_task())
            continue
            
        elif prompt.lower().startswith('task '):
            task_type = prompt.lower().split(' ', 1)[1].strip()
            if task_type == "auto":
                current_task = "auto"
                print("Task type set to auto-detect")
            elif task_type in registry.get_available_patterns():
                current_task = task_type
                print(f"Task type set to {task_type}")
            else:
                print(f"Unknown task type: {task_type}")
                print("Available tasks:", ', '.join(registry.get_available_patterns().keys()))
            continue
        
        # Auto-detect or use specified task
        task = None if current_task == "auto" else current_task
        
        # Apply specialization
        if task:
            report = registry.apply_specialization(task_type=task)
        else:
            report = registry.apply_specialization(input_text=prompt)
            task = report.get("task_type", "unknown")
        
        print(f"\nDetected/Selected task: {task}")
        
        # Generate text
        generated_text, metrics = measure_generation_performance(
            model, tokenizer, prompt, args.max_tokens
        )
        
        print("\n=== Generated Text ===")
        print(generated_text)
        print("=====================")
        
        print(f"Generation time: {metrics['generation_time']:.2f}s")
        print(f"Tokens per second: {metrics['tokens_per_second']:.2f}")
        print(f"Diversity score: {metrics['diversity_score']:.3f}")
        
        if args.agency_report:
            agency_report = model.get_agency_report()
            print("\n=== Agency Report ===")
            print(f"Active heads: {sum(layer['active_heads'] for layer in agency_report['layer_reports'].values())}")
            print(f"Withdrawn heads: {sum(layer['withdrawn_heads'] for layer in agency_report['layer_reports'].values())}")
            print(f"Overloaded heads: {sum(layer['overloaded_heads'] for layer in agency_report['layer_reports'].values())}")
            print(f"Misaligned heads: {sum(layer['misaligned_heads'] for layer in agency_report['layer_reports'].values())}")
            print(f"Total consent violations: {agency_report['total_violations']}")
            print("=====================")

def main():
    args = parse_args()
    
    print("📚 Loading model...")
    model, tokenizer = load_gpt2_with_sentinel_gates(
        model_name=args.model,
        gate_init=1.0,
        connection_init=0.0,
        norm_attn_output=True,
        debug=args.debug
    )
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create specialization registry
    registry = SpecializationRegistry(model, debug=args.debug)
    
    if args.benchmark:
        # Run benchmarking on different patterns
        if not args.prompt:
            print("Error: Prompt is required for benchmarking")
            return
        benchmark_specialization_patterns(registry, model, tokenizer, args.prompt, args.max_tokens)
        return
    
    if args.interactive:
        # Run in interactive mode
        interactive_mode(registry, model, tokenizer, args)
        return
    
    # Regular mode with a single prompt
    if not args.prompt:
        print("Error: Prompt is required (or use --interactive for interactive mode)")
        return
    
    # Determine task type
    task = None if args.task == "auto" else args.task
    
    # Apply specialization
    if task:
        report = registry.apply_specialization(task_type=task)
    else:
        report = registry.apply_specialization(input_text=args.prompt)
        task = report.get("task_type", "unknown")
    
    print(f"Applied specialization for task: {task}")
    print_specialization_info(registry, registry.specialization, task)
    
    # Generate text with performance measurement
    print("\n🔄 Generating text...")
    generated_text, metrics = measure_generation_performance(
        model, tokenizer, args.prompt, args.max_tokens
    )
    
    # Print generation results
    print("\n=== Generated Text ===")
    print(generated_text)
    print("=====================")
    
    # Print performance metrics
    print("\n📊 Performance Metrics:")
    print(f"Generation time: {metrics['generation_time']:.2f}s")
    print(f"Tokens generated: {metrics['generated_tokens']}")
    print(f"Tokens per second: {metrics['tokens_per_second']:.2f}")
    print(f"Memory allocated: {metrics['memory_allocated_gb']:.2f} GB")
    print(f"Diversity score: {metrics['diversity_score']:.3f}")
    print(f"Repetition detected: {'Yes' if metrics['repetition_detected'] else 'No'}")
    
    # Display agency report if requested
    if args.agency_report:
        agency_report = model.get_agency_report()
        print("\n🔍 Agency Report:")
        print(f"Active heads: {sum(layer['active_heads'] for layer in agency_report['layer_reports'].values())}")
        print(f"Withdrawn heads: {sum(layer['withdrawn_heads'] for layer in agency_report['layer_reports'].values())}")
        print(f"Overloaded heads: {sum(layer['overloaded_heads'] for layer in agency_report['layer_reports'].values())}")
        print(f"Misaligned heads: {sum(layer['misaligned_heads'] for layer in agency_report['layer_reports'].values())}")
        print(f"Total consent violations: {agency_report['total_violations']}")
    
    # Save metrics if requested
    if args.save_metrics:
        # Update pattern with observed metrics
        performance_improvement = 0.0  # Would need baseline comparison
        resource_reduction = 0.0       # Would need baseline comparison
        quality_enhancement = metrics['diversity_score']  # Simple proxy
        
        registry.update_pattern_metrics(task, {
            "performance_improvement": performance_improvement,
            "resource_reduction": resource_reduction,
            "quality_enhancement": quality_enhancement
        })
        
        print("\n💾 Updated pattern metrics saved.")
    
    # Save output if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            output = {
                "prompt": args.prompt,
                "task_type": task,
                "generated_text": generated_text,
                "metrics": metrics,
                "specialization": report
            }
            json.dump(output, f, indent=2)
        print(f"\n📝 Output saved to {args.output_file}")

if __name__ == "__main__":
    main()