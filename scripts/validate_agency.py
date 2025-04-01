#!/usr/bin/env python
"""
Validate the effectiveness of agency features in Sentinel-AI.

This script runs a series of experiments to empirically validate that the
agency features in our adaptive transformer provide measurable benefits:

1. Performance comparison with and without agency features
2. Resource utilization under different load conditions
3. Output quality assessment with different agency configurations
4. Resilience testing under resource constraints

Usage:
    python scripts/validate_agency.py --model_name distilgpt2 \
                                    --test_scenarios basic overload mixed \
                                    --output_dir ./validation_results/agency

"""

import os
import sys
import time
import json
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.loaders.loader import load_baseline_model, load_adaptive_model
from utils.metrics import calculate_perplexity, diversity_metrics, repetition_metrics


def has_agency_features(model):
    """Check if the model has agency features."""
    try:
        return hasattr(model, "get_agency_report") and hasattr(model.blocks[0]["attn"], "set_head_state")
    except (AttributeError, IndexError):
        return False


def create_test_scenarios():
    """Define test scenarios for validation."""
    return {
        "baseline": {
            "description": "Baseline model with no agency features activated",
            "agency_enabled": False,
        },
        "agency_default": {
            "description": "Agency model with all heads in active state",
            "agency_enabled": True,
            "head_states": {
                "state": "active",
                "consent": True,
            }
        },
        "agency_specialized": {
            "description": "Agency model with specialized head states for the task",
            "agency_enabled": True,
            "head_states": [
                # For some heads, we'll customize states (others remain active)
                {"layer": 0, "head_indices": [0, 3, 6], "state": "overloaded"},
                {"layer": 2, "head_indices": [1, 4, 7], "state": "misaligned"},
            ]
        },
        "agency_mixed": {
            "description": "Agency model with mixed head states (30% overloaded, 20% misaligned)",
            "agency_enabled": True,
            "head_states": "mixed",
            "overloaded_ratio": 0.3,
            "misaligned_ratio": 0.2,
        },
        "agency_constrained": {
            "description": "Agency model with constrained resources (40% withdrawn)",
            "agency_enabled": True,
            "head_states": "constrained",
            "withdrawn_ratio": 0.4,
        }
    }


def apply_scenario(model, scenario, verbose=False):
    """Apply a test scenario to the model."""
    if not has_agency_features(model):
        print("Model does not have agency features")
        return False
    
    # Reset all heads to active state with consent
    reset_all_heads(model)
    
    if not scenario["agency_enabled"]:
        if verbose:
            print(f"Applied scenario: {scenario['description']}")
        return True
    
    # Apply the scenario's head states
    head_states = scenario.get("head_states", {})
    
    if isinstance(head_states, dict):
        # Apply the same state to all heads
        state = head_states.get("state", "active")
        consent = head_states.get("consent", True)
        
        for layer_idx in range(model.num_layers):
            heads_per_layer = model.blocks[layer_idx]["attn"].num_heads
            for head_idx in range(heads_per_layer):
                model.set_head_state(layer_idx, head_idx, state, consent)
                
        if verbose:
            print(f"Applied scenario: {scenario['description']} - Set all heads to {state}")
    
    elif isinstance(head_states, list):
        # Apply specific states to specific heads
        for config in head_states:
            layer = config["layer"]
            head_indices = config["head_indices"]
            state = config["state"]
            consent = config.get("consent", True)
            
            for head_idx in head_indices:
                model.set_head_state(layer, head_idx, state, consent)
                
        if verbose:
            print(f"Applied scenario: {scenario['description']} - Applied customized head states")
    
    elif head_states == "mixed":
        # Apply mixed states randomly
        overloaded_ratio = scenario.get("overloaded_ratio", 0.3)
        misaligned_ratio = scenario.get("misaligned_ratio", 0.2)
        
        for layer_idx in range(model.num_layers):
            heads_per_layer = model.blocks[layer_idx]["attn"].num_heads
            all_head_indices = list(range(heads_per_layer))
            random.shuffle(all_head_indices)
            
            # Determine how many heads of each type
            num_overloaded = int(heads_per_layer * overloaded_ratio)
            num_misaligned = int(heads_per_layer * misaligned_ratio)
            
            # Assign states
            for i, head_idx in enumerate(all_head_indices):
                if i < num_overloaded:
                    model.set_head_state(layer_idx, head_idx, "overloaded")
                elif i < num_overloaded + num_misaligned:
                    model.set_head_state(layer_idx, head_idx, "misaligned")
                    
        if verbose:
            print(f"Applied scenario: {scenario['description']} - Applied mixed states")
    
    elif head_states == "constrained":
        # Apply withdrawn state to simulate resource constraints
        withdrawn_ratio = scenario.get("withdrawn_ratio", 0.4)
        
        for layer_idx in range(model.num_layers):
            heads_per_layer = model.blocks[layer_idx]["attn"].num_heads
            all_head_indices = list(range(heads_per_layer))
            random.shuffle(all_head_indices)
            
            # Determine how many heads should be withdrawn
            num_withdrawn = int(heads_per_layer * withdrawn_ratio)
            
            # Withdraw some heads
            for i in range(num_withdrawn):
                head_idx = all_head_indices[i]
                model.set_head_state(layer_idx, head_idx, "withdrawn", consent=False)
                
        if verbose:
            print(f"Applied scenario: {scenario['description']} - Applied resource constraints")
    
    else:
        print(f"Unknown head state configuration: {head_states}")
        return False
    
    return True


def reset_all_heads(model):
    """Reset all heads to active state with consent."""
    if not has_agency_features(model):
        return
    
    for layer_idx in range(model.num_layers):
        heads_per_layer = model.blocks[layer_idx]["attn"].num_heads
        for head_idx in range(heads_per_layer):
            model.set_head_state(layer_idx, head_idx, "active", consent=True)


def load_or_create_prompts(prompt_file=None, num_prompts=10):
    """Load prompts from a file or create default prompts."""
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, 'r') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        return prompts[:num_prompts]  # Limit to num_prompts
    
    # Default prompts for different domains
    default_prompts = [
        # General text
        "Once upon a time in a land far away,",
        "The future of artificial intelligence depends on",
        "Scientists have recently discovered that",
        
        # Code completion (Python)
        "def calculate_average(numbers):\n    total = sum(numbers)\n    return",
        "for i in range(10):\n    if i % 2 == 0:",
        "class DataProcessor:\n    def __init__(self, data):\n        self.data = data\n\n    def clean_data(self):",
        
        # Code completion (SQL)
        "SELECT user_id, COUNT(*) as order_count\nFROM orders\nGROUP BY",
        "CREATE TABLE users (\n    id INT PRIMARY KEY,\n    name VARCHAR(100),",
        "SELECT u.username, o.order_date\nFROM users u\nLEFT JOIN orders o ON",
        
        # Mixed content
        "# Process data and store in database\ndef process_data(data):\n    # Clean the data\n    clean_data = [x for x in data if x is not None]\n    \n    # Connect to database\n    conn = get_db_connection()\n    cursor = conn.cursor()\n    \n    # Insert data\n    query = \"INSERT INTO processed_data (value) VALUES"
    ]
    
    return default_prompts[:num_prompts]


def benchmark_inference(model, tokenizer, prompts, max_tokens=50, num_runs=3, device="cpu"):
    """Benchmark inference performance."""
    results = {
        "per_prompt": [],
        "average_time": 0,
        "tokens_per_second": 0,
        "outputs": []
    }
    
    all_times = []
    
    for prompt in tqdm(prompts, desc="Benchmarking prompts"):
        prompt_times = []
        
        # Encode the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Do multiple runs for consistency
        for _ in range(num_runs):
            # Clear cache if using CUDA
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Time the generation
            start_time = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_length=len(inputs.input_ids[0]) + max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
            end_time = time.time()
            
            # Record time
            prompt_times.append(end_time - start_time)
        
        # Get the generated text from the last run
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Calculate average time for this prompt
        avg_prompt_time = sum(prompt_times) / len(prompt_times)
        tokens_per_second = max_tokens / avg_prompt_time
        
        # Record results for this prompt
        results["per_prompt"].append({
            "prompt": prompt,
            "time": avg_prompt_time,
            "tokens_per_second": tokens_per_second,
            "output": output_text
        })
        
        # Record the output
        results["outputs"].append(output_text)
        
        # Add to overall times
        all_times.append(avg_prompt_time)
    
    # Calculate overall averages
    results["average_time"] = sum(all_times) / len(all_times)
    results["tokens_per_second"] = max_tokens / results["average_time"]
    
    return results


def measure_output_quality(outputs, prompts):
    """Measure output quality metrics."""
    quality_metrics = {}
    
    # Text diversity metrics
    diversity = diversity_metrics(outputs)
    quality_metrics.update(diversity)
    
    # Repetition metrics
    repetition = repetition_metrics(outputs)
    quality_metrics.update(repetition)
    
    # Perplexity approximation (if model supports it)
    try:
        avg_perplexity = calculate_perplexity(outputs, prompts)
        quality_metrics["perplexity"] = avg_perplexity
    except Exception as e:
        print(f"Could not calculate perplexity: {e}")
    
    return quality_metrics


def monitor_resource_usage(model, tokenizer, prompt, 
                          max_tokens=50, device="cpu"):
    """Monitor resource usage during generation."""
    usage_metrics = {}
    
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Clear CUDA cache if applicable
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Generate with resource monitoring
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=len(inputs.input_ids[0]) + max_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    end_time = time.time()
    
    # Record basic metrics
    usage_metrics["generation_time"] = end_time - start_time
    usage_metrics["tokens_per_second"] = max_tokens / usage_metrics["generation_time"]
    
    # Record CUDA memory usage if applicable
    if device == "cuda" and torch.cuda.is_available():
        usage_metrics["peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        usage_metrics["memory_per_token_kb"] = (usage_metrics["peak_memory_mb"] * 1024) / max_tokens
    
    # Get agency report if available
    if has_agency_features(model):
        try:
            agency_report = model.get_agency_report()
            usage_metrics["agency_report"] = {
                "total_violations": agency_report["total_violations"],
                "active_heads": sum(
                    report.get("active_heads", 0) 
                    for report in agency_report["layer_reports"].values()
                ),
                "overloaded_heads": sum(
                    report.get("overloaded_heads", 0) 
                    for report in agency_report["layer_reports"].values()
                ),
                "misaligned_heads": sum(
                    report.get("misaligned_heads", 0) 
                    for report in agency_report["layer_reports"].values()
                ),
                "withdrawn_heads": sum(
                    report.get("withdrawn_heads", 0) 
                    for report in agency_report["layer_reports"].values()
                ),
            }
        except Exception as e:
            print(f"Could not get agency report: {e}")
    
    # Add system info
    import psutil
    usage_metrics["cpu_percent"] = psutil.cpu_percent()
    usage_metrics["ram_percent"] = psutil.virtual_memory().percent
    
    return usage_metrics


def create_comparative_visualizations(results, output_dir):
    """Create visualizations comparing scenarios."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract scenario names and descriptions for labels
    scenarios = list(results.keys())
    scenario_labels = [results[s]["description"].split(" - ")[0] for s in scenarios]
    
    # 1. Generation Speed Comparison
    plt.figure(figsize=(12, 6))
    tokens_per_second = [results[s]["inference"]["tokens_per_second"] for s in scenarios]
    
    bars = plt.bar(scenario_labels, tokens_per_second, color=['blue', 'green', 'orange', 'red', 'purple'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.title("Generation Speed Comparison")
    plt.ylabel("Tokens per Second")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "generation_speed.png"))
    plt.close()
    
    # 2. Output Quality Metrics
    if all("quality" in results[s] for s in scenarios):
        quality_metrics = ["lexical_diversity", "unique_token_ratio", "repetition_score"]
        
        for metric in quality_metrics:
            if all(metric in results[s]["quality"] for s in scenarios):
                plt.figure(figsize=(12, 6))
                
                metric_values = [results[s]["quality"][metric] for s in scenarios]
                bars = plt.bar(scenario_labels, metric_values, 
                              color=['blue', 'green', 'orange', 'red', 'purple'])
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom')
                
                plt.title(f"{metric.replace('_', ' ').title()} Comparison")
                plt.ylabel(metric.replace('_', ' ').title())
                plt.xticks(rotation=45, ha="right")
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                
                plt.savefig(os.path.join(output_dir, f"{metric}_comparison.png"))
                plt.close()
    
    # 3. Resource Usage Comparison
    if all("resource_usage" in results[s] for s in scenarios):
        # Generation time
        plt.figure(figsize=(12, 6))
        
        gen_times = [results[s]["resource_usage"]["generation_time"] for s in scenarios]
        bars = plt.bar(scenario_labels, gen_times, 
                      color=['blue', 'green', 'orange', 'red', 'purple'])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.title("Generation Time Comparison")
        plt.ylabel("Seconds")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "generation_time.png"))
        plt.close()
        
        # Memory usage for CUDA
        if "peak_memory_mb" in results[scenarios[0]]["resource_usage"]:
            plt.figure(figsize=(12, 6))
            
            memory_usage = [results[s]["resource_usage"]["peak_memory_mb"] for s in scenarios]
            bars = plt.bar(scenario_labels, memory_usage, 
                          color=['blue', 'green', 'orange', 'red', 'purple'])
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}', ha='center', va='bottom')
            
            plt.title("Peak Memory Usage Comparison")
            plt.ylabel("Memory (MB)")
            plt.xticks(rotation=45, ha="right")
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, "memory_usage.png"))
            plt.close()

    # 4. Head states distribution (for agency models)
    agency_scenarios = [s for s in scenarios if results[s].get("agency_enabled", False)]
    
    if agency_scenarios and all("resource_usage" in results[s] and 
                               "agency_report" in results[s]["resource_usage"] 
                               for s in agency_scenarios):
        plt.figure(figsize=(14, 8))
        
        # Collect head state counts
        active_heads = []
        overloaded_heads = []
        misaligned_heads = []
        withdrawn_heads = []
        
        for s in agency_scenarios:
            report = results[s]["resource_usage"]["agency_report"]
            active_heads.append(report.get("active_heads", 0))
            overloaded_heads.append(report.get("overloaded_heads", 0))
            misaligned_heads.append(report.get("misaligned_heads", 0))
            withdrawn_heads.append(report.get("withdrawn_heads", 0))
        
        # Only keep agency scenarios for labels
        agency_labels = [results[s]["description"].split(" - ")[0] for s in agency_scenarios]
        
        # Plotting the stacked bar chart
        width = 0.65
        
        plt.bar(agency_labels, active_heads, width, label='Active', color='green')
        plt.bar(agency_labels, overloaded_heads, width, bottom=active_heads, 
               label='Overloaded', color='orange')
        
        # Calculate the position for misaligned
        if overloaded_heads:
            misaligned_position = [a + o for a, o in zip(active_heads, overloaded_heads)]
        else:
            misaligned_position = active_heads
            
        plt.bar(agency_labels, misaligned_heads, width, bottom=misaligned_position, 
               label='Misaligned', color='blue')
        
        # Calculate position for withdrawn
        if misaligned_heads:
            withdrawn_position = [a + o + m for a, o, m in 
                                zip(active_heads, overloaded_heads, misaligned_heads)]
        else:
            withdrawn_position = [a + o for a, o in zip(active_heads, overloaded_heads)]
            
        plt.bar(agency_labels, withdrawn_heads, width, bottom=withdrawn_position, 
               label='Withdrawn', color='red')
        
        plt.xlabel('Scenario')
        plt.ylabel('Number of Heads')
        plt.title('Head State Distribution by Scenario')
        plt.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "head_state_distribution.png"))
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Validate Sentinel-AI agency features")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="Model name")
    parser.add_argument("--scenarios", type=str, nargs="+", 
                      default=["baseline", "agency_default", "agency_mixed", "agency_constrained"],
                      choices=["baseline", "agency_default", "agency_specialized", 
                              "agency_mixed", "agency_constrained", "all"],
                      help="Test scenarios to run")
    parser.add_argument("--prompt_file", type=str, help="File with prompts")
    parser.add_argument("--num_prompts", type=int, default=5, help="Number of prompts to use")
    parser.add_argument("--output_dir", type=str, default="./validation_results/agency", 
                      help="Output directory")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None,
                      help="Device to use (defaults to CUDA if available)")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of runs per prompt")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    # Set device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Set padding token to EOS token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    baseline_model = load_baseline_model(args.model_name, device)
    model = load_adaptive_model(args.model_name, baseline_model, device)
    
    # Check for agency features
    if not has_agency_features(model):
        print("ERROR: This model does not support agency features.")
        print("Please make sure you're using a version of Sentinel-AI with agency support.")
        sys.exit(1)
    
    # Load or create prompts
    prompts = load_or_create_prompts(args.prompt_file, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts")
    
    # Choose a single representative prompt for resource testing
    resource_prompt = prompts[0] if prompts else "The meaning of life is"
    
    # Get available test scenarios
    all_scenarios = create_test_scenarios()
    
    # Select scenarios to run
    if "all" in args.scenarios:
        selected_scenarios = list(all_scenarios.keys())
    else:
        selected_scenarios = args.scenarios
    
    # Validate scenarios
    for scenario_name in selected_scenarios:
        if scenario_name not in all_scenarios:
            print(f"Warning: Unknown scenario '{scenario_name}', skipping")
            selected_scenarios.remove(scenario_name)
    
    print(f"Running {len(selected_scenarios)} validation scenarios:")
    for scenario_name in selected_scenarios:
        print(f"  - {scenario_name}: {all_scenarios[scenario_name]['description']}")
    
    # Initialize results dictionary
    results = {}
    
    # Run validation for each scenario
    for scenario_name in selected_scenarios:
        scenario = all_scenarios[scenario_name]
        print(f"\n{'-'*80}\nRunning scenario: {scenario_name}")
        
        # Initialize scenario results
        results[scenario_name] = {
            "description": scenario["description"],
            "agency_enabled": scenario["agency_enabled"],
        }
        
        # Apply the scenario configuration
        success = apply_scenario(model, scenario, verbose=args.verbose)
        if not success:
            print(f"Failed to apply scenario {scenario_name}")
            continue
        
        # 1. Benchmark inference performance
        print("\nRunning inference benchmark...")
        inference_results = benchmark_inference(
            model, tokenizer, prompts, 
            max_tokens=args.max_tokens, 
            num_runs=args.num_runs,
            device=device
        )
        results[scenario_name]["inference"] = inference_results
        
        print(f"Average generation speed: {inference_results['tokens_per_second']:.2f} tokens/sec")
        
        # 2. Measure output quality
        print("\nMeasuring output quality...")
        quality_metrics = measure_output_quality(inference_results["outputs"], prompts)
        results[scenario_name]["quality"] = quality_metrics
        
        print(f"Lexical diversity: {quality_metrics.get('lexical_diversity', 'N/A'):.3f}")
        print(f"Repetition score: {quality_metrics.get('repetition_score', 'N/A'):.3f}")
        
        # 3. Monitor resource usage
        print("\nMonitoring resource usage...")
        resource_usage = monitor_resource_usage(
            model, tokenizer, resource_prompt, 
            max_tokens=args.max_tokens * 2,  # Longer text for better measurement
            device=device
        )
        results[scenario_name]["resource_usage"] = resource_usage
        
        print(f"Generation time: {resource_usage['generation_time']:.3f} seconds")
        if "peak_memory_mb" in resource_usage:
            print(f"Peak memory usage: {resource_usage['peak_memory_mb']:.2f} MB")
        
        # 4. Check agency report if available
        if has_agency_features(model) and "agency_report" in resource_usage:
            agency_report = resource_usage["agency_report"]
            print("\nAgency report:")
            print(f"Active heads: {agency_report.get('active_heads', 'N/A')}")
            print(f"Overloaded heads: {agency_report.get('overloaded_heads', 'N/A')}")
            print(f"Misaligned heads: {agency_report.get('misaligned_heads', 'N/A')}")
            print(f"Withdrawn heads: {agency_report.get('withdrawn_heads', 'N/A')}")
            print(f"Consent violations: {agency_report.get('total_violations', 'N/A')}")
    
    # Save results to file
    results_file = os.path.join(args.output_dir, "validation_results.json")
    print(f"\nSaving results to {results_file}")
    
    # Prepare results for JSON serialization (remove non-serializable objects)
    json_results = {}
    for scenario, data in results.items():
        json_results[scenario] = {}
        for key, value in data.items():
            if key == "inference":
                # Simplify the inference results to avoid circular references
                json_results[scenario][key] = {
                    "average_time": value["average_time"],
                    "tokens_per_second": value["tokens_per_second"],
                    "per_prompt_count": len(value["per_prompt"]),
                }
            else:
                json_results[scenario][key] = value
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_comparative_visualizations(results, args.output_dir)
    
    print(f"\nValidation complete. Results saved to {args.output_dir}")
    
    # Print summary of findings
    print("\n" + "="*80)
    print("SUMMARY OF FINDINGS")
    print("="*80)
    
    # Sort scenarios to ensure baseline is first
    sorted_scenarios = sorted(
        selected_scenarios, 
        key=lambda s: 0 if s == "baseline" else 1
    )
    
    # Speed comparison
    if "baseline" in results:
        baseline_speed = results["baseline"]["inference"]["tokens_per_second"]
        print(f"\nGeneration Speed (relative to baseline):")
        for scenario in sorted_scenarios:
            if scenario == "baseline":
                print(f"  {scenario}: {baseline_speed:.2f} tokens/sec (baseline)")
            else:
                rel_speed = results[scenario]["inference"]["tokens_per_second"] / baseline_speed
                print(f"  {scenario}: {results[scenario]['inference']['tokens_per_second']:.2f} tokens/sec ({rel_speed:.2%} of baseline)")
    
    # Quality comparison
    if all("quality" in results[s] and "lexical_diversity" in results[s]["quality"] for s in selected_scenarios):
        print(f"\nOutput Quality (lexical diversity):")
        for scenario in sorted_scenarios:
            diversity = results[scenario]["quality"]["lexical_diversity"]
            print(f"  {scenario}: {diversity:.3f}")
    
    # Resource usage
    if all("resource_usage" in results[s] and "generation_time" in results[s]["resource_usage"] for s in selected_scenarios):
        print(f"\nResource Usage (generation time):")
        for scenario in sorted_scenarios:
            gen_time = results[scenario]["resource_usage"]["generation_time"]
            print(f"  {scenario}: {gen_time:.3f} seconds")
    
    # Conclusion
    print("\nCONCLUSION:")
    
    # Compare agency scenarios to baseline
    if "baseline" in results and any(s != "baseline" for s in selected_scenarios):
        baseline_metrics = {
            "speed": results["baseline"]["inference"]["tokens_per_second"],
            "quality": results["baseline"]["quality"].get("lexical_diversity", 0),
        }
        
        best_agency_scenario = None
        best_agency_speed = 0
        
        for scenario in selected_scenarios:
            if scenario != "baseline" and results[scenario]["agency_enabled"]:
                speed = results[scenario]["inference"]["tokens_per_second"]
                if speed > best_agency_speed:
                    best_agency_speed = speed
                    best_agency_scenario = scenario
        
        if best_agency_scenario:
            speed_diff = best_agency_speed / baseline_metrics["speed"] - 1
            direction = "faster" if speed_diff > 0 else "slower"
            
            print(f"The best agency configuration ({best_agency_scenario}) was {abs(speed_diff):.1%} {direction} than baseline")
            
            quality_diff = results[best_agency_scenario]["quality"].get("lexical_diversity", 0) / baseline_metrics["quality"] - 1
            quality_dir = "higher" if quality_diff > 0 else "lower"
            
            print(f"Output quality was {abs(quality_diff):.1%} {quality_dir} than baseline")
            
            if speed_diff > 0 and quality_diff >= -0.05:  # Allow slight quality reduction
                print("\nOverall: Agency features provide SIGNIFICANT PERFORMANCE BENEFITS with comparable quality")
            elif speed_diff > 0 and quality_diff < -0.05:
                print("\nOverall: Agency features provide PERFORMANCE BENEFITS with some quality trade-offs")
            elif speed_diff <= 0 and quality_diff > 0:
                print("\nOverall: Agency features provide QUALITY BENEFITS at some performance cost")
            else:
                print("\nOverall: Agency benefits not clearly demonstrated in this configuration - further optimization needed")
    
    print("="*80)


if __name__ == "__main__":
    main()