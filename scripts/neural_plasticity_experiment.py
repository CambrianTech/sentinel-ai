#!/usr/bin/env python
"""
Neural Plasticity Experiment Runner

This script runs multiple neural plasticity experiments with different configurations
to compare the effectiveness of various pruning and growth strategies.

Example usage:
    python scripts/neural_plasticity_experiment.py --model_name distilgpt2 --run_all
    python scripts/neural_plasticity_experiment.py --model_name gpt2 --compare_growth_strategies
    python scripts/neural_plasticity_experiment.py --model_name distilgpt2 --compare_pruning_levels
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Neural Plasticity Experiment Runner")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                      help="Model name (default: distilgpt2)")
    parser.add_argument("--dataset", type=str, default="tiny_shakespeare",
                      help="Dataset name (default: tiny_shakespeare)")
    parser.add_argument("--experiment_dir", type=str, default="./output/plasticity_experiments",
                      help="Directory to save experiment results (default: ./output/plasticity_experiments)")
    
    # Experiment selection
    parser.add_argument("--run_all", action="store_true",
                      help="Run all experiments")
    parser.add_argument("--compare_pruning_strategies", action="store_true",
                      help="Compare different pruning strategies")
    parser.add_argument("--compare_growth_strategies", action="store_true",
                      help="Compare different growth strategies")
    parser.add_argument("--compare_pruning_levels", action="store_true",
                      help="Compare different pruning levels")
    parser.add_argument("--compare_cycles", action="store_true",
                      help="Compare different numbers of plasticity cycles")
    
    # Common experiment parameters
    parser.add_argument("--initial_training_steps", type=int, default=50,
                      help="Initial training steps before first pruning (default: 50)")
    parser.add_argument("--learning_steps", type=int, default=50,
                      help="Learning steps after each growth phase (default: 50)")
    parser.add_argument("--eval_every", type=int, default=10,
                      help="Evaluate every N steps (default: 10)")
    parser.add_argument("--save_visualizations", action="store_true",
                      help="Save visualizations for each experiment")
    
    return parser.parse_args()

def run_experiment(cmd_args, experiment_name):
    """Run a single experiment with the given arguments"""
    print(f"\n=== Running Experiment: {experiment_name} ===\n")
    print(f"Command: {' '.join(cmd_args)}\n")
    
    # Run the experiment
    result = subprocess.run(cmd_args, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running experiment: {result.stderr}")
        return None
    
    # Extract experiment directory from output
    output_lines = result.stdout.split('\n')
    experiment_dir = None
    for line in output_lines:
        if "Results saved to" in line:
            experiment_dir = line.split("Results saved to ")[-1].strip()
            break
    
    print(f"Experiment completed. Results saved to {experiment_dir}\n")
    return experiment_dir

def compare_pruning_strategies(args):
    """Compare different pruning strategies"""
    base_cmd = [
        "python", "scripts/neural_plasticity_cycle.py",
        "--model_name", args.model_name,
        "--dataset", args.dataset,
        "--initial_training_steps", str(args.initial_training_steps),
        "--learning_steps", str(args.learning_steps),
        "--eval_every", str(args.eval_every),
        "--cycles", "1",
        "--growth_strategy", "gradient_sensitivity",
        "--growth_ratio", "0.5",
        "--initial_pruning", "0.3",
    ]
    
    # Add visualization flag if requested
    if args.save_visualizations:
        base_cmd.append("--save_visualizations")
    
    strategies = ["entropy", "magnitude", "random"]
    experiment_dirs = {}
    
    for strategy in strategies:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"pruning_strategy_{strategy}_{timestamp}"
        
        # Create command for this experiment
        cmd = base_cmd + [
            "--pruning_strategy", strategy,
            "--experiment_name", experiment_name
        ]
        
        # Run experiment
        experiment_dir = run_experiment(cmd, f"Pruning Strategy: {strategy}")
        if experiment_dir:
            experiment_dirs[strategy] = experiment_dir
    
    # Analyze and visualize results
    if experiment_dirs:
        analyze_pruning_strategies(experiment_dirs, args)
    
    return experiment_dirs

def compare_growth_strategies(args):
    """Compare different growth strategies"""
    base_cmd = [
        "python", "scripts/neural_plasticity_cycle.py",
        "--model_name", args.model_name,
        "--dataset", args.dataset,
        "--initial_training_steps", str(args.initial_training_steps),
        "--learning_steps", str(args.learning_steps),
        "--eval_every", str(args.eval_every),
        "--cycles", "1",
        "--pruning_strategy", "entropy",
        "--growth_ratio", "0.5",
        "--initial_pruning", "0.3",
    ]
    
    # Add visualization flag if requested
    if args.save_visualizations:
        base_cmd.append("--save_visualizations")
    
    strategies = ["gradient_sensitivity", "entropy_gap", "balanced", "random"]
    experiment_dirs = {}
    
    for strategy in strategies:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"growth_strategy_{strategy}_{timestamp}"
        
        # Create command for this experiment
        cmd = base_cmd + [
            "--growth_strategy", strategy,
            "--experiment_name", experiment_name
        ]
        
        # Run experiment
        experiment_dir = run_experiment(cmd, f"Growth Strategy: {strategy}")
        if experiment_dir:
            experiment_dirs[strategy] = experiment_dir
    
    # Analyze and visualize results
    if experiment_dirs:
        analyze_growth_strategies(experiment_dirs, args)
    
    return experiment_dirs

def compare_pruning_levels(args):
    """Compare different pruning levels"""
    base_cmd = [
        "python", "scripts/neural_plasticity_cycle.py",
        "--model_name", args.model_name,
        "--dataset", args.dataset,
        "--initial_training_steps", str(args.initial_training_steps),
        "--learning_steps", str(args.learning_steps),
        "--eval_every", str(args.eval_every),
        "--cycles", "1",
        "--pruning_strategy", "entropy",
        "--growth_strategy", "gradient_sensitivity",
        "--growth_ratio", "0.5",
    ]
    
    # Add visualization flag if requested
    if args.save_visualizations:
        base_cmd.append("--save_visualizations")
    
    pruning_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    experiment_dirs = {}
    
    for level in pruning_levels:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"pruning_level_{int(level*100)}percent_{timestamp}"
        
        # Create command for this experiment
        cmd = base_cmd + [
            "--initial_pruning", str(level),
            "--experiment_name", experiment_name
        ]
        
        # Run experiment
        experiment_dir = run_experiment(cmd, f"Pruning Level: {level*100:.0f}%")
        if experiment_dir:
            experiment_dirs[str(level)] = experiment_dir
    
    # Analyze and visualize results
    if experiment_dirs:
        analyze_pruning_levels(experiment_dirs, args)
    
    return experiment_dirs

def compare_cycles(args):
    """Compare different numbers of plasticity cycles"""
    base_cmd = [
        "python", "scripts/neural_plasticity_cycle.py",
        "--model_name", args.model_name,
        "--dataset", args.dataset,
        "--initial_training_steps", str(args.initial_training_steps),
        "--learning_steps", str(args.learning_steps),
        "--eval_every", str(args.eval_every),
        "--pruning_strategy", "entropy",
        "--growth_strategy", "gradient_sensitivity",
        "--growth_ratio", "0.5",
        "--initial_pruning", "0.3",
    ]
    
    # Add visualization flag if requested
    if args.save_visualizations:
        base_cmd.append("--save_visualizations")
    
    cycle_counts = [1, 2, 3, 4]
    experiment_dirs = {}
    
    for cycles in cycle_counts:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"cycles_{cycles}_{timestamp}"
        
        # Create command for this experiment
        cmd = base_cmd + [
            "--cycles", str(cycles),
            "--experiment_name", experiment_name
        ]
        
        # Run experiment
        experiment_dir = run_experiment(cmd, f"Plasticity Cycles: {cycles}")
        if experiment_dir:
            experiment_dirs[str(cycles)] = experiment_dir
    
    # Analyze and visualize results
    if experiment_dirs:
        analyze_cycles(experiment_dirs, args)
    
    return experiment_dirs

def load_metrics(experiment_dir):
    """Load metrics from an experiment directory"""
    metrics_file = os.path.join(experiment_dir, "metrics/metrics.jsonl")
    summary_file = os.path.join(experiment_dir, "summary.json")
    
    metrics = []
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            for line in f:
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    
    summary = {}
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
    
    return metrics, summary

def analyze_pruning_strategies(experiment_dirs, args):
    """Analyze and compare results from different pruning strategies"""
    results = {}
    
    for strategy, exp_dir in experiment_dirs.items():
        metrics, summary = load_metrics(exp_dir)
        
        # Extract key metrics
        initial_perplexity = None
        pruned_perplexity = None
        final_perplexity = None
        perplexity_recovery = None
        
        for metric in metrics:
            if metric.get('phase') == 'initial':
                initial_perplexity = metric.get('perplexity')
            elif metric.get('phase') == 'measurement':
                pruned_perplexity = metric.get('pruned_perplexity')
            elif metric.get('phase') == 'final':
                final_perplexity = metric.get('final_perplexity')
                perplexity_recovery = metric.get('perplexity_recovery')
        
        results[strategy] = {
            'initial_perplexity': initial_perplexity,
            'pruned_perplexity': pruned_perplexity,
            'final_perplexity': final_perplexity,
            'perplexity_recovery': perplexity_recovery
        }
    
    # Create visualization
    if results:
        plt.figure(figsize=(12, 8))
        
        # Set up data for plotting
        strategies = list(results.keys())
        initial_perplexities = [results[s]['initial_perplexity'] for s in strategies]
        pruned_perplexities = [results[s]['pruned_perplexity'] for s in strategies]
        final_perplexities = [results[s]['final_perplexity'] for s in strategies]
        recovery_percentages = [results[s]['perplexity_recovery'] for s in strategies]
        
        # Create bar chart for perplexities
        x = np.arange(len(strategies))
        width = 0.25
        
        plt.subplot(2, 1, 1)
        plt.bar(x - width, initial_perplexities, width, label='Initial')
        plt.bar(x, pruned_perplexities, width, label='After Pruning')
        plt.bar(x + width, final_perplexities, width, label='After Growth & Learning')
        plt.xlabel('Pruning Strategy')
        plt.ylabel('Perplexity (lower is better)')
        plt.title(f'Perplexity Comparison Across Pruning Strategies - {args.model_name}')
        plt.xticks(x, strategies)
        plt.legend()
        
        # Create bar chart for recovery percentages
        plt.subplot(2, 1, 2)
        plt.bar(x, recovery_percentages, 0.5)
        plt.xlabel('Pruning Strategy')
        plt.ylabel('Perplexity Recovery %')
        plt.title('Perplexity Recovery After Growth & Learning')
        plt.xticks(x, strategies)
        
        for i, v in enumerate(recovery_percentages):
            if v is not None:
                plt.text(i, v + 1, f"{v:.1f}%", ha='center')
        
        plt.tight_layout()
        
        # Save the figure
        os.makedirs(args.experiment_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_file = os.path.join(args.experiment_dir, f"pruning_strategies_comparison_{timestamp}.png")
        plt.savefig(output_file)
        print(f"Saved comparison visualization to {output_file}")
        plt.close()

def analyze_growth_strategies(experiment_dirs, args):
    """Analyze and compare results from different growth strategies"""
    results = {}
    
    for strategy, exp_dir in experiment_dirs.items():
        metrics, summary = load_metrics(exp_dir)
        
        # Extract key metrics
        initial_perplexity = None
        pruned_perplexity = None
        final_perplexity = None
        perplexity_recovery = None
        
        for metric in metrics:
            if metric.get('phase') == 'initial':
                initial_perplexity = metric.get('perplexity')
            elif metric.get('phase') == 'measurement':
                pruned_perplexity = metric.get('pruned_perplexity')
            elif metric.get('phase') == 'final':
                final_perplexity = metric.get('final_perplexity')
                perplexity_recovery = metric.get('perplexity_recovery')
        
        results[strategy] = {
            'initial_perplexity': initial_perplexity,
            'pruned_perplexity': pruned_perplexity,
            'final_perplexity': final_perplexity,
            'perplexity_recovery': perplexity_recovery
        }
    
    # Create visualization
    if results:
        plt.figure(figsize=(12, 8))
        
        # Set up data for plotting
        strategies = list(results.keys())
        initial_perplexities = [results[s]['initial_perplexity'] for s in strategies]
        pruned_perplexities = [results[s]['pruned_perplexity'] for s in strategies]
        final_perplexities = [results[s]['final_perplexity'] for s in strategies]
        recovery_percentages = [results[s]['perplexity_recovery'] for s in strategies]
        
        # Create bar chart for perplexities
        x = np.arange(len(strategies))
        width = 0.25
        
        plt.subplot(2, 1, 1)
        plt.bar(x - width, initial_perplexities, width, label='Initial')
        plt.bar(x, pruned_perplexities, width, label='After Pruning')
        plt.bar(x + width, final_perplexities, width, label='After Growth & Learning')
        plt.xlabel('Growth Strategy')
        plt.ylabel('Perplexity (lower is better)')
        plt.title(f'Perplexity Comparison Across Growth Strategies - {args.model_name}')
        plt.xticks(x, strategies)
        plt.legend()
        
        # Create bar chart for recovery percentages
        plt.subplot(2, 1, 2)
        plt.bar(x, recovery_percentages, 0.5)
        plt.xlabel('Growth Strategy')
        plt.ylabel('Perplexity Recovery %')
        plt.title('Perplexity Recovery After Growth & Learning')
        plt.xticks(x, strategies)
        
        for i, v in enumerate(recovery_percentages):
            if v is not None:
                plt.text(i, v + 1, f"{v:.1f}%", ha='center')
        
        plt.tight_layout()
        
        # Save the figure
        os.makedirs(args.experiment_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_file = os.path.join(args.experiment_dir, f"growth_strategies_comparison_{timestamp}.png")
        plt.savefig(output_file)
        print(f"Saved comparison visualization to {output_file}")
        plt.close()

def analyze_pruning_levels(experiment_dirs, args):
    """Analyze and compare results from different pruning levels"""
    results = {}
    
    for level, exp_dir in experiment_dirs.items():
        metrics, summary = load_metrics(exp_dir)
        
        # Extract key metrics
        initial_perplexity = None
        pruned_perplexity = None
        final_perplexity = None
        perplexity_recovery = None
        
        for metric in metrics:
            if metric.get('phase') == 'initial':
                initial_perplexity = metric.get('perplexity')
            elif metric.get('phase') == 'measurement':
                pruned_perplexity = metric.get('pruned_perplexity')
            elif metric.get('phase') == 'final':
                final_perplexity = metric.get('final_perplexity')
                perplexity_recovery = metric.get('perplexity_recovery')
        
        results[level] = {
            'initial_perplexity': initial_perplexity,
            'pruned_perplexity': pruned_perplexity,
            'final_perplexity': final_perplexity,
            'perplexity_recovery': perplexity_recovery
        }
    
    # Create visualization
    if results:
        plt.figure(figsize=(12, 10))
        
        # Set up data for plotting
        levels = sorted(results.keys(), key=lambda x: float(x))
        level_labels = [f"{float(level)*100:.0f}%" for level in levels]
        initial_perplexities = [results[s]['initial_perplexity'] for s in levels]
        pruned_perplexities = [results[s]['pruned_perplexity'] for s in levels]
        final_perplexities = [results[s]['final_perplexity'] for s in levels]
        recovery_percentages = [results[s]['perplexity_recovery'] for s in levels]
        
        # Create bar chart for perplexities
        x = np.arange(len(levels))
        width = 0.25
        
        plt.subplot(3, 1, 1)
        plt.bar(x - width, initial_perplexities, width, label='Initial')
        plt.bar(x, pruned_perplexities, width, label='After Pruning')
        plt.bar(x + width, final_perplexities, width, label='After Growth & Learning')
        plt.xlabel('Pruning Level')
        plt.ylabel('Perplexity (lower is better)')
        plt.title(f'Perplexity Comparison Across Pruning Levels - {args.model_name}')
        plt.xticks(x, level_labels)
        plt.legend()
        
        # Create bar chart for recovery percentages
        plt.subplot(3, 1, 2)
        plt.bar(x, recovery_percentages, 0.5)
        plt.xlabel('Pruning Level')
        plt.ylabel('Perplexity Recovery %')
        plt.title('Perplexity Recovery After Growth & Learning')
        plt.xticks(x, level_labels)
        
        for i, v in enumerate(recovery_percentages):
            if v is not None:
                plt.text(i, v + 1, f"{v:.1f}%", ha='center')
        
        # Create line chart for trends
        plt.subplot(3, 1, 3)
        plt.plot(x, pruned_perplexities, 'o-', label='After Pruning')
        plt.plot(x, final_perplexities, 'o-', label='After Growth & Learning')
        plt.xlabel('Pruning Level')
        plt.ylabel('Perplexity (lower is better)')
        plt.title('Perplexity Trend Across Pruning Levels')
        plt.xticks(x, level_labels)
        plt.legend()
        
        plt.tight_layout()
        
        # Save the figure
        os.makedirs(args.experiment_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_file = os.path.join(args.experiment_dir, f"pruning_levels_comparison_{timestamp}.png")
        plt.savefig(output_file)
        print(f"Saved comparison visualization to {output_file}")
        plt.close()

def analyze_cycles(experiment_dirs, args):
    """Analyze and compare results from different numbers of plasticity cycles"""
    results = {}
    
    for cycles, exp_dir in experiment_dirs.items():
        metrics, summary = load_metrics(exp_dir)
        
        # Extract key metrics
        initial_perplexity = None
        final_perplexity = None
        cycle_metrics = {}
        
        for metric in metrics:
            if metric.get('phase') == 'initial':
                initial_perplexity = metric.get('perplexity')
            elif metric.get('phase') == 'final':
                cycle_num = metric.get('cycle')
                if cycle_num is not None:
                    cycle_metrics[cycle_num] = {
                        'perplexity': metric.get('final_perplexity'),
                        'recovery': metric.get('perplexity_recovery')
                    }
                    
                    # Update final perplexity to be the last cycle's final perplexity
                    final_perplexity = metric.get('final_perplexity')
        
        results[cycles] = {
            'initial_perplexity': initial_perplexity,
            'final_perplexity': final_perplexity,
            'cycle_metrics': cycle_metrics
        }
    
    # Create visualization
    if results:
        plt.figure(figsize=(12, 10))
        
        # Set up data for plotting
        cycle_counts = sorted(results.keys(), key=lambda x: int(x))
        initial_perplexities = [results[s]['initial_perplexity'] for s in cycle_counts]
        final_perplexities = [results[s]['final_perplexity'] for s in cycle_counts]
        
        # Create bar chart for initial vs final perplexities
        x = np.arange(len(cycle_counts))
        width = 0.35
        
        plt.subplot(2, 1, 1)
        plt.bar(x - width/2, initial_perplexities, width, label='Initial')
        plt.bar(x + width/2, final_perplexities, width, label='Final')
        plt.xlabel('Number of Plasticity Cycles')
        plt.ylabel('Perplexity (lower is better)')
        plt.title(f'Perplexity Comparison Across Cycle Counts - {args.model_name}')
        plt.xticks(x, cycle_counts)
        plt.legend()
        
        # Create line chart for improvement over cycles
        plt.subplot(2, 1, 2)
        
        for i, cycles in enumerate(cycle_counts):
            if int(cycles) > 1:  # Only for experiments with multiple cycles
                cycle_nums = sorted(results[cycles]['cycle_metrics'].keys())
                cycle_perplexities = [results[cycles]['cycle_metrics'][c]['perplexity'] for c in cycle_nums]
                
                plt.plot(cycle_nums, cycle_perplexities, 'o-', label=f'{cycles} cycles')
        
        plt.xlabel('Cycle Number')
        plt.ylabel('Perplexity (lower is better)')
        plt.title('Perplexity Evolution Over Cycles')
        plt.legend()
        
        plt.tight_layout()
        
        # Save the figure
        os.makedirs(args.experiment_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_file = os.path.join(args.experiment_dir, f"cycles_comparison_{timestamp}.png")
        plt.savefig(output_file)
        print(f"Saved comparison visualization to {output_file}")
        plt.close()

def main():
    """Main function"""
    args = parse_args()
    
    # Ensure experiment directory exists
    os.makedirs(args.experiment_dir, exist_ok=True)
    
    # Track completed experiments
    completed_experiments = {}
    
    # Determine which experiments to run
    if args.run_all or args.compare_pruning_strategies:
        completed_experiments['pruning_strategies'] = compare_pruning_strategies(args)
    
    if args.run_all or args.compare_growth_strategies:
        completed_experiments['growth_strategies'] = compare_growth_strategies(args)
    
    if args.run_all or args.compare_pruning_levels:
        completed_experiments['pruning_levels'] = compare_pruning_levels(args)
    
    if args.run_all or args.compare_cycles:
        completed_experiments['cycles'] = compare_cycles(args)
    
    # Summary
    print("\n=== Experiment Summary ===\n")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Experiments completed: {len(completed_experiments)}")
    
    for exp_type, dirs in completed_experiments.items():
        if dirs:
            print(f"\n{exp_type.replace('_', ' ').title()}:")
            for name, directory in dirs.items():
                print(f"  - {name}: {directory}")
    
    # If no experiments were selected, print help
    if not completed_experiments:
        print("No experiments were selected. Use --run_all or specify individual experiments to run.")
        print("For help, use --help")

if __name__ == "__main__":
    main()