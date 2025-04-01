#!/usr/bin/env python
"""
Benchmark the agency features in Sentinel-AI.

This script benchmarks the effectiveness of the agency features in the
adaptive transformer model, focusing on:
1. Generation speed with different agency states
2. Output quality with different agency states  
3. Resource utilization under different agency configurations
4. Consent tracking and violation monitoring

Usage:
    python scripts/benchmark_agency.py --model_name distilgpt2 \
                                     --test_scenarios basic overload consent_withdrawal \
                                     --output_dir ./benchmark_results/agency

Features:
- Evaluation of different agency scenarios
- Agency state tracking and visualization
- Output quality assessment
- Performance metrics collection
"""

import os
import sys
import argparse
import torch
import json
import time
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# Add project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.loaders.loader import load_baseline_model, load_adaptive_model
from utils.generation_wrapper import generate_text


def has_agency_features(model):
    """Check if the model has agency features."""
    try:
        return hasattr(model, "get_agency_report") and hasattr(model.blocks[0]["attn"], "set_head_state")
    except (AttributeError, IndexError):
        return False


def set_agency_states(model, scenario, proportion=0.3):
    """
    Set agency states based on the benchmark scenario.
    
    Args:
        model: Adaptive transformer model
        scenario: Scenario name ('basic', 'overload', 'consent_withdrawal', etc.)
        proportion: Proportion of heads to modify (0-1)
        
    Returns:
        Dictionary with mapping of modified heads
    """
    if not has_agency_features(model):
        return {"error": "Model does not support agency features"}
    
    modified_heads = {}
    num_layers = model.num_layers
    heads_per_layer = 12  # Standard for many models, adjust as needed
    
    # Reset all heads to active state with consent
    for layer_idx in range(num_layers):
        for head_idx in range(heads_per_layer):
            try:
                model.set_head_state(layer_idx, head_idx, "active", consent=True)
            except Exception:
                # Head might not exist, skip
                continue
    
    if scenario == "basic":
        # No changes, all heads active
        return {"scenario": "basic", "modified_heads": {}}
    
    elif scenario == "overload":
        # Set some heads to "overloaded" state
        for layer_idx in range(num_layers):
            modified_heads[layer_idx] = []
            
            # Determine how many heads to overload
            num_to_modify = int(heads_per_layer * proportion)
            heads_to_modify = random.sample(range(heads_per_layer), num_to_modify)
            
            for head_idx in heads_to_modify:
                try:
                    model.set_head_state(layer_idx, head_idx, "overloaded")
                    modified_heads[layer_idx].append(head_idx)
                except Exception:
                    continue
        
        return {"scenario": "overload", "modified_heads": modified_heads}
    
    elif scenario == "consent_withdrawal":
        # Withdraw consent for some heads
        for layer_idx in range(num_layers):
            modified_heads[layer_idx] = []
            
            # Determine how many heads to withdraw consent
            num_to_modify = int(heads_per_layer * proportion)
            heads_to_modify = random.sample(range(heads_per_layer), num_to_modify)
            
            for head_idx in heads_to_modify:
                try:
                    model.set_head_state(layer_idx, head_idx, "withdrawn", consent=False)
                    modified_heads[layer_idx].append(head_idx)
                except Exception:
                    continue
        
        return {"scenario": "consent_withdrawal", "modified_heads": modified_heads}
    
    elif scenario == "misaligned":
        # Set some heads to "misaligned" state
        for layer_idx in range(num_layers):
            modified_heads[layer_idx] = []
            
            # Determine how many heads to mark as misaligned
            num_to_modify = int(heads_per_layer * proportion)
            heads_to_modify = random.sample(range(heads_per_layer), num_to_modify)
            
            for head_idx in heads_to_modify:
                try:
                    model.set_head_state(layer_idx, head_idx, "misaligned")
                    modified_heads[layer_idx].append(head_idx)
                except Exception:
                    continue
        
        return {"scenario": "misaligned", "modified_heads": modified_heads}
    
    elif scenario == "mixed":
        # Mix of different states
        states = ["active", "overloaded", "misaligned", "withdrawn"]
        consent_values = [True, True, True, False]  # Only withdrawn has False consent
        
        for layer_idx in range(num_layers):
            modified_heads[layer_idx] = []
            
            for head_idx in range(heads_per_layer):
                if random.random() < proportion:
                    state_idx = random.randint(0, 3)
                    state = states[state_idx]
                    consent = consent_values[state_idx]
                    
                    try:
                        model.set_head_state(layer_idx, head_idx, state, consent=consent)
                        modified_heads[layer_idx].append((head_idx, state))
                    except Exception:
                        continue
        
        return {"scenario": "mixed", "modified_heads": modified_heads}
    
    else:
        return {"error": f"Unknown scenario: {scenario}"}


def force_consent_violations(model, num_violations=3):
    """
    Force consent violations by setting high gate values for withdrawn heads.
    
    Args:
        model: Adaptive transformer model
        num_violations: Number of violations to force
        
    Returns:
        Dictionary with information about forced violations
    """
    if not has_agency_features(model):
        return {"error": "Model does not support agency features"}
    
    violations = {}
    violation_count = 0
    
    # Get current agency report
    agency_report = model.get_agency_report()
    
    # Loop through layers looking for withdrawn heads
    for layer_idx, report in agency_report["layer_reports"].items():
        if violation_count >= num_violations:
            break
            
        # Check for withdrawn heads in this layer
        if report["withdrawn_heads"] > 0:
            withdrawn_indices = report.get("withdrawn_head_indices", [])
            
            for head_idx in withdrawn_indices:
                if violation_count >= num_violations:
                    break
                    
                # Force high gate value for this head
                with torch.no_grad():
                    try:
                        model.blocks[int(layer_idx)]["attn"].gate[head_idx] = torch.tensor(0.9, device=model.device)
                        
                        if layer_idx not in violations:
                            violations[layer_idx] = []
                        violations[layer_idx].append(head_idx)
                        
                        violation_count += 1
                    except Exception:
                        continue
    
    return {"forced_violations": violations, "count": violation_count}


def benchmark_generation(model, tokenizer, prompts, scenario, device, 
                        max_tokens=50, num_runs=3):
    """
    Benchmark generation with different agency scenarios.
    
    Args:
        model: Model to benchmark
        tokenizer: Tokenizer for the model
        prompts: List of prompts to use for generation
        scenario: The agency scenario being tested
        device: Device to run on
        max_tokens: Maximum tokens to generate
        num_runs: Number of runs to average
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        "scenario": scenario,
        "generation_times": [],
        "tokens_per_second": [],
        "outputs": []
    }
    
    for i, prompt in enumerate(prompts):
        run_times = []
        outputs = []
        
        for run in range(num_runs):
            # Time and run generation
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            start_time = time.time()
            with torch.no_grad():
                outputs_tensor = model.generate(
                    **inputs,
                    max_length=len(inputs.input_ids[0]) + max_tokens,
                    do_sample=True,
                    temperature=0.7
                )
            end_time = time.time()
            
            # Record results
            generation_time = end_time - start_time
            run_times.append(generation_time)
            
            # Decode output
            output_text = tokenizer.decode(outputs_tensor[0], skip_special_tokens=True)
            outputs.append(output_text)
        
        avg_time = np.mean(run_times)
        tokens_per_second = max_tokens / avg_time
        
        results["generation_times"].append(avg_time)
        results["tokens_per_second"].append(tokens_per_second)
        results["outputs"].append(outputs[-1])  # Save the last output
    
    # Calculate overall stats
    results["avg_generation_time"] = float(np.mean(results["generation_times"]))
    results["avg_tokens_per_second"] = float(np.mean(results["tokens_per_second"]))
    results["std_tokens_per_second"] = float(np.std(results["tokens_per_second"]))
    
    return results


def analyze_text_quality(texts):
    """
    Analyze text quality metrics.
    
    Args:
        texts: List of generated texts
        
    Returns:
        Dictionary with quality metrics
    """
    metrics = {
        "length": [],
        "unique_words": [],
        "lexical_diversity": [],
        "sentence_count": []
    }
    
    for text in texts:
        # Extract a chunk that's likely actually generated (not just the prompt)
        words = text.split()
        unique_words = set(words)
        
        # Count sentences (approximation)
        sentences = text.split('.')
        sentence_count = len([s for s in sentences if len(s.strip()) > 0])
        
        # Record metrics
        metrics["length"].append(len(words))
        metrics["unique_words"].append(len(unique_words))
        metrics["lexical_diversity"].append(
            len(unique_words) / len(words) if words else 0
        )
        metrics["sentence_count"].append(sentence_count)
    
    # Calculate averages
    results = {}
    for key, values in metrics.items():
        results[f"avg_{key}"] = float(np.mean(values))
        results[f"std_{key}"] = float(np.std(values))
    
    return results


def check_consent_violations(model):
    """
    Check for consent violations in the model.
    
    Args:
        model: Model to check
        
    Returns:
        Dictionary with violation stats
    """
    if not has_agency_features(model):
        return {"error": "Model does not support agency features"}
    
    agency_report = model.get_agency_report()
    violations = []
    
    # Collect all violations
    for layer_idx, report in agency_report["layer_reports"].items():
        if "recent_violations" in report and report["recent_violations"]:
            for violation in report["recent_violations"]:
                violations.append({
                    "layer": layer_idx,
                    "head": violation["head_idx"],
                    "type": violation["violation_type"],
                    "state": violation["state"],
                    "gate_value": violation["gate_value"]
                })
    
    return {
        "total_violations": agency_report["total_violations"],
        "violations": violations
    }


def plot_benchmark_results(results, output_dir):
    """
    Plot benchmark results.
    
    Args:
        results: Dictionary with results for different scenarios
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract scenarios and metrics
    scenarios = list(results.keys())
    generation_speeds = [results[s]["avg_tokens_per_second"] for s in scenarios]
    std_speeds = [results[s]["std_tokens_per_second"] for s in scenarios]
    
    # 1. Generation Speed Comparison
    plt.figure(figsize=(12, 6))
    bars = plt.bar(scenarios, generation_speeds, yerr=std_speeds, 
                  color=['green', 'orange', 'red', 'purple', 'blue'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.title("Generation Speed by Agency Scenario")
    plt.ylabel("Tokens per Second")
    plt.xlabel("Scenario")
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, "generation_speed_comparison.png"))
    plt.close()
    
    # 2. Text Quality Comparison - Lexical Diversity
    if all('text_quality' in results[s] for s in scenarios):
        lexical_diversity = [results[s]["text_quality"]["avg_lexical_diversity"] for s in scenarios]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(scenarios, lexical_diversity, 
                      color=['green', 'orange', 'red', 'purple', 'blue'])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.title("Lexical Diversity by Agency Scenario")
        plt.ylabel("Lexical Diversity")
        plt.xlabel("Scenario")
        plt.grid(axis='y', alpha=0.3)
        
        plt.savefig(os.path.join(output_dir, "lexical_diversity_comparison.png"))
        plt.close()
    
    # 3. Violation Tracking (if available)
    if all('consent_violations' in results[s] for s in scenarios):
        violation_counts = [results[s]["consent_violations"]["total_violations"] for s in scenarios]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(scenarios, violation_counts, 
                      color=['green', 'orange', 'red', 'purple', 'blue'])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.title("Consent Violations by Agency Scenario")
        plt.ylabel("Number of Violations")
        plt.xlabel("Scenario")
        plt.grid(axis='y', alpha=0.3)
        
        plt.savefig(os.path.join(output_dir, "consent_violations.png"))
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Agency Features")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="Model name or path")
    parser.add_argument("--test_scenarios", type=str, nargs="+", 
                      default=["basic", "overload", "consent_withdrawal", "misaligned", "mixed"],
                      help="Scenarios to test")
    parser.add_argument("--prompt_file", type=str, help="File with benchmark prompts")
    parser.add_argument("--proportion", type=float, default=0.3, 
                      help="Proportion of heads to modify in each scenario")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results/agency/", 
                      help="Output directory for results")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of runs per prompt")
    parser.add_argument("--force_violations", action="store_true", 
                      help="Force consent violations for testing")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None,
                      help="Device to use (defaults to CUDA if available)")
    args = parser.parse_args()
    
    # Determine device
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    baseline_model = load_baseline_model(args.model_name, device)
    adaptive_model = load_adaptive_model(args.model_name, baseline_model, device)
    
    # Check for agency features
    if not has_agency_features(adaptive_model):
        print("ERROR: This model does not support agency features.")
        print("Please make sure you're using a version of Sentinel-AI with agency support.")
        sys.exit(1)
    
    # Load or create prompts
    if args.prompt_file and os.path.exists(args.prompt_file):
        with open(args.prompt_file, 'r') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")
    else:
        print("Using default prompts.")
        prompts = [
            "Once upon a time in a land far away,",
            "The future of artificial intelligence depends on",
            "Scientists have recently discovered that",
            "The secret to a happy life is",
            "In the year 2050, humanity will",
            "The ethical implications of technology are"
        ]
    
    # Run benchmarks for each scenario
    results = {}
    
    for scenario in args.test_scenarios:
        print(f"\nBenchmarking scenario: {scenario}")
        
        # Set up agency states for this scenario
        agency_config = set_agency_states(adaptive_model, scenario, args.proportion)
        print(f"Set up {scenario} scenario")
        
        # Get initial agency report
        initial_report = adaptive_model.get_agency_report()
        print("Initial agency state:")
        for layer_idx, report in initial_report["layer_reports"].items():
            head_states = {
                "active": report["active_heads"],
                "overloaded": report["overloaded_heads"] if "overloaded_heads" in report else 0,
                "misaligned": report["misaligned_heads"] if "misaligned_heads" in report else 0,
                "withdrawn": report["withdrawn_heads"]
            }
            print(f"  Layer {layer_idx}: {head_states}")
        
        # Force consent violations if requested
        if args.force_violations and scenario == "consent_withdrawal":
            print("Forcing consent violations...")
            violation_info = force_consent_violations(adaptive_model, num_violations=3)
            print(f"Forced {violation_info['count']} violations")
        
        # Run generation benchmark
        print("Running generation benchmark...")
        benchmark_results = benchmark_generation(
            adaptive_model, tokenizer, prompts, scenario, device,
            max_tokens=args.max_tokens, num_runs=args.num_runs
        )
        
        # Analyze text quality
        print("Analyzing output quality...")
        quality_metrics = analyze_text_quality(benchmark_results["outputs"])
        benchmark_results["text_quality"] = quality_metrics
        
        # Check for consent violations
        print("Checking for consent violations...")
        violations = check_consent_violations(adaptive_model)
        benchmark_results["consent_violations"] = violations
        
        # Store results
        results[scenario] = benchmark_results
        
        # Print some metrics
        print(f"Average generation speed: {benchmark_results['avg_tokens_per_second']:.2f} tokens/sec")
        print(f"Lexical diversity: {quality_metrics['avg_lexical_diversity']:.3f}")
        print(f"Consent violations: {violations['total_violations']}")
    
    # Save results
    print("\nSaving results...")
    with open(os.path.join(args.output_dir, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot comparisons
    print("Generating comparison plots...")
    plot_benchmark_results(results, args.output_dir)
    
    print(f"\nBenchmarking complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()