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
    
    parser.add_argument("--pruning_levels", type=str, default="0,30,50,70",
                      help="Comma-separated pruning percentages to evaluate")
    parser.add_argument("--output_dir", type=str, default="validation_results/pruning_agency",
                      help="Directory to save results and visualizations")
    parser.add_argument("--model_name", type=str, default="gpt2",
                      help="Base model to use (gpt2, distilgpt2)")
    parser.add_argument("--prompt_file", type=str, default="datasets/eval_prompts.txt",
                      help="File containing prompts to use for evaluation")
    parser.add_argument("--num_tokens", type=int, default=100,
                      help="Number of tokens to generate for each prompt")
    parser.add_argument("--visualize_only", action="store_true",
                      help="Only generate visualizations from existing results")
    
    return parser.parse_args()

def load_model(model_name, agency_enabled=False):
    """Load either a baseline or agency-enabled model."""
    base_model, tokenizer = load_gpt2_model(model_name)
    adaptive_model = AdaptiveTransformer(base_model, enable_agency=agency_enabled)
    model = SentinelModelWrapper(adaptive_model, tokenizer)
    return model, tokenizer

def apply_pruning(model, pruning_percentage):
    """Apply pruning to the model by setting the lowest gate values to zero."""
    # Get all gate values
    gate_values = []
    for layer_idx, block in enumerate(model.model.blocks):
        for head_idx in range(model.model.num_heads):
            gate_values.append((layer_idx, head_idx, float(block["attn"].gate[head_idx])))
    
    # Sort by gate value
    gate_values.sort(key=lambda x: x[2])
    
    # Calculate how many heads to prune
    num_heads = len(gate_values)
    num_to_prune = int(num_heads * pruning_percentage / 100)
    
    # Set the lowest gates to zero
    for i in range(num_to_prune):
        layer_idx, head_idx, _ = gate_values[i]
        model.model.blocks[layer_idx]["attn"].gate[head_idx].zero_()
    
    return model, num_to_prune

def load_prompts(prompt_file):
    """Load prompts from a file."""
    with open(prompt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def evaluate_model(model, tokenizer, prompts, num_tokens):
    """Evaluate model performance on a set of prompts."""
    results = {
        "perplexity": [],
        "diversity": [],
        "repetition": [],
        "generation_time": [],
        "outputs": []
    }
    
    for prompt in prompts:
        # Time the generation
        start_time = time.time()
        output = model.generate(prompt, max_length=len(tokenizer.encode(prompt)) + num_tokens)
        generation_time = time.time() - start_time
        
        # Store metrics
        results["generation_time"].append(generation_time)
        results["perplexity"].append(calculate_perplexity(model, tokenizer, output))
        results["diversity"].append(calculate_diversity(output))
        results["repetition"].append(calculate_repetition(output))
        results["outputs"].append(output)
    
    # Average the results
    for key in results:
        if key != "outputs":
            results[key] = sum(results[key]) / len(results[key])
    
    # Add tokens per second
    results["tokens_per_second"] = num_tokens / results["generation_time"]
    
    return results

def run_pruning_comparison(args):
    """Run the main comparison between baseline and agency models."""
    pruning_levels = [int(x) for x in args.pruning_levels.split(",")]
    prompts = load_prompts(args.prompt_file)
    
    # Prepare results structure
    results = {
        "baseline": {},
        "agency": {},
        "metadata": {
            "model_name": args.model_name,
            "num_tokens": args.num_tokens,
            "num_prompts": len(prompts)
        }
    }
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # For each pruning level
    for level in pruning_levels:
        print(f"Evaluating pruning level: {level}%")
        
        # Baseline model
        print(f"  Loading baseline model...")
        baseline_model, tokenizer = load_model(args.model_name, agency_enabled=False)
        baseline_model, pruned_heads = apply_pruning(baseline_model, level)
        print(f"  Pruned {pruned_heads} heads in baseline model")
        
        # Agency model
        print(f"  Loading agency model...")
        agency_model, _ = load_model(args.model_name, agency_enabled=True)
        agency_model, pruned_heads = apply_pruning(agency_model, level)
        print(f"  Pruned {pruned_heads} heads in agency model")
        
        # Evaluate both models
        print(f"  Evaluating baseline model...")
        baseline_results = evaluate_model(baseline_model, tokenizer, prompts, args.num_tokens)
        
        print(f"  Evaluating agency model...")
        agency_results = evaluate_model(agency_model, tokenizer, prompts, args.num_tokens)
        
        # Store results
        results["baseline"][level] = baseline_results
        results["agency"][level] = agency_results
        
        print(f"  Results at {level}% pruning:")
        print(f"    Baseline perplexity: {baseline_results['perplexity']:.2f}, "
              f"tokens/sec: {baseline_results['tokens_per_second']:.2f}")
        print(f"    Agency perplexity: {agency_results['perplexity']:.2f}, "
              f"tokens/sec: {agency_results['tokens_per_second']:.2f}")
    
    # Save results to JSON
    results_file = output_dir / "pruning_comparison_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    return results

def visualize_results(results, output_dir):
    """Generate visualizations from the results."""
    output_dir = Path(output_dir)
    pruning_levels = sorted([int(level) for level in results["baseline"].keys()])
    
    # Extract metrics
    baseline_speed = [results["baseline"][str(level)]["tokens_per_second"] for level in pruning_levels]
    agency_speed = [results["agency"][str(level)]["tokens_per_second"] for level in pruning_levels]
    
    baseline_ppl = [results["baseline"][str(level)]["perplexity"] for level in pruning_levels]
    agency_ppl = [results["agency"][str(level)]["perplexity"] for level in pruning_levels]
    
    baseline_div = [results["baseline"][str(level)]["diversity"] for level in pruning_levels]
    agency_div = [results["agency"][str(level)]["diversity"] for level in pruning_levels]
    
    # 1. Generation speed comparison
    plt.figure(figsize=(10, 6))
    plt.plot(pruning_levels, baseline_speed, marker='o', label="Baseline", color="#78909C")
    plt.plot(pruning_levels, agency_speed, marker='o', label="Agency", color="#4CAF50")
    plt.title("Generation Speed vs. Pruning Level", fontsize=16)
    plt.xlabel("Pruning Percentage", fontsize=14)
    plt.ylabel("Tokens per Second", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Annotate improvement at highest pruning level
    max_level = max(pruning_levels)
    improvement = ((agency_speed[-1] / baseline_speed[-1]) - 1) * 100
    plt.annotate(f"{improvement:.1f}% faster",
                xy=(max_level, agency_speed[-1]),
                xytext=(max_level-10, agency_speed[-1]+1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / "speed_comparison.png", dpi=150)
    
    # 2. Perplexity comparison
    plt.figure(figsize=(10, 6))
    plt.plot(pruning_levels, baseline_ppl, marker='o', label="Baseline", color="#78909C")
    plt.plot(pruning_levels, agency_ppl, marker='o', label="Agency", color="#4CAF50")
    plt.title("Perplexity vs. Pruning Level (Lower is Better)", fontsize=16)
    plt.xlabel("Pruning Percentage", fontsize=14)
    plt.ylabel("Perplexity", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Annotate difference at highest pruning level
    ppl_diff = ((baseline_ppl[-1] / agency_ppl[-1]) - 1) * 100
    plt.annotate(f"{abs(ppl_diff):.1f}% {'better' if ppl_diff > 0 else 'worse'}",
                xy=(max_level, agency_ppl[-1]),
                xytext=(max_level-10, agency_ppl[-1]-5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / "perplexity_comparison.png", dpi=150)
    
    # 3. Diversity comparison
    plt.figure(figsize=(10, 6))
    plt.plot(pruning_levels, baseline_div, marker='o', label="Baseline", color="#78909C")
    plt.plot(pruning_levels, agency_div, marker='o', label="Agency", color="#4CAF50")
    plt.title("Lexical Diversity vs. Pruning Level (Higher is Better)", fontsize=16)
    plt.xlabel("Pruning Percentage", fontsize=14)
    plt.ylabel("Lexical Diversity", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Annotate difference at highest pruning level
    div_diff = ((agency_div[-1] / baseline_div[-1]) - 1) * 100
    plt.annotate(f"{abs(div_diff):.1f}% {'better' if div_diff > 0 else 'worse'}",
                xy=(max_level, agency_div[-1]),
                xytext=(max_level-10, agency_div[-1]+0.05),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / "diversity_comparison.png", dpi=150)
    
    # 4. Combined radar chart for highest pruning level
    metrics = ['Speed', 'Quality\n(1/Perplexity)', 'Diversity', 'Efficiency\n(1/Generation Time)']
    
    # Normalize values (higher is better for all)
    max_speed = max(baseline_speed[-1], agency_speed[-1])
    max_ppl_inv = max(1/baseline_ppl[-1], 1/agency_ppl[-1])
    max_div = max(baseline_div[-1], agency_div[-1])
    max_eff = max(1/results["baseline"][str(max_level)]["generation_time"], 
                 1/results["agency"][str(max_level)]["generation_time"])
    
    baseline_values = [
        baseline_speed[-1] / max_speed,
        (1/baseline_ppl[-1]) / max_ppl_inv,
        baseline_div[-1] / max_div,
        (1/results["baseline"][str(max_level)]["generation_time"]) / max_eff
    ]
    
    agency_values = [
        agency_speed[-1] / max_speed,
        (1/agency_ppl[-1]) / max_ppl_inv,
        agency_div[-1] / max_div,
        (1/results["agency"][str(max_level)]["generation_time"]) / max_eff
    ]
    
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
    
    plt.title(f"Model Performance at {max_level}% Pruning", fontsize=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(output_dir / "radar_comparison.png", dpi=150)
    
    # 5. Create a summary visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Left: Speed vs Pruning
    ax1.plot(pruning_levels, baseline_speed, marker='o', label="Baseline", color="#78909C")
    ax1.plot(pruning_levels, agency_speed, marker='o', label="Agency", color="#4CAF50")
    ax1.set_title("Generation Speed", fontsize=14)
    ax1.set_xlabel("Pruning %", fontsize=12)
    ax1.set_ylabel("Tokens per Second", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Right: Perplexity vs Speed scatter
    baseline_x = baseline_speed
    baseline_y = [1/p for p in baseline_ppl]  # Inverse perplexity (higher is better)
    
    agency_x = agency_speed
    agency_y = [1/p for p in agency_ppl]
    
    # Size represents pruning level
    sizes = [100 + 10*level for level in pruning_levels]
    
    ax2.scatter(baseline_x, baseline_y, s=sizes, alpha=0.7, label="Baseline", color="#78909C")
    ax2.scatter(agency_x, agency_y, s=sizes, alpha=0.7, label="Agency", color="#4CAF50")
    
    # Connect points with lines
    for i in range(len(pruning_levels)):
        ax2.plot([baseline_x[i], agency_x[i]], [baseline_y[i], agency_y[i]], 
                 'k--', alpha=0.3, linewidth=1)
    
    # Add pruning level annotations
    for i, level in enumerate(pruning_levels):
        ax2.annotate(f"{level}%", 
                    xy=(baseline_x[i], baseline_y[i]),
                    xytext=(2, 2),
                    textcoords='offset points', 
                    fontsize=8)
        
        ax2.annotate(f"{level}%", 
                    xy=(agency_x[i], agency_y[i]),
                    xytext=(2, 2),
                    textcoords='offset points',
                    fontsize=8)
    
    ax2.set_title("Quality vs. Speed Tradeoff", fontsize=14)
    ax2.set_xlabel("Tokens per Second", fontsize=12)
    ax2.set_ylabel("Quality (1/Perplexity)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Add "better" direction
    ax2.annotate("Better", xy=(0.85, 0.85), xycoords='axes fraction',
                xytext=(0.7, 0.7), textcoords='axes fraction',
                arrowprops=dict(facecolor='green', shrink=0.05, width=2),
                fontsize=12)
    
    plt.suptitle(f"Pruning Efficacy: Agency vs. Baseline ({args.model_name.upper()})", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_dir / "pruning_summary.png", dpi=150)
    
    print(f"Visualizations saved to {output_dir}")

def main():
    """Main function."""
    args = setup_args()
    
    if args.visualize_only:
        # Load existing results
        results_file = Path(args.output_dir) / "pruning_comparison_results.json"
        if not results_file.exists():
            print(f"Error: Results file not found at {results_file}")
            return
        
        with open(results_file, "r") as f:
            results = json.load(f)
        
        visualize_results(results, args.output_dir)
    else:
        # Run the full comparison
        results = run_pruning_comparison(args)
        visualize_results(results, args.output_dir)

if __name__ == "__main__":
    main()