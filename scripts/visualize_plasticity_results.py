#!/usr/bin/env python
"""
Visualization of Adaptive Plasticity Results

This script generates visualizations for adaptive plasticity optimization results.
It can be used to:
1. Create visualizations from saved optimization results
2. Generate a comprehensive dashboard
3. Compare different optimization runs

Usage:
    python scripts/visualize_plasticity_results.py --results_dir ./output/adaptive_plasticity/run_20250401-120000
    python scripts/visualize_plasticity_results.py --results_dir ./output/adaptive_plasticity --compare
    python scripts/visualize_plasticity_results.py --results_dir ./output/adaptive_plasticity --dashboard
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import visualization tools
from utils.adaptive.visualization import (
    plot_head_gradient_with_status,
    plot_cycles_comparison,
    plot_plasticity_history,
    plot_strategy_effectiveness,
    create_dashboard
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Adaptive Plasticity Results Visualization")
    
    # Input parameters
    parser.add_argument("--results_dir", type=str, required=True,
                      help="Directory containing optimization results")
    parser.add_argument("--output_dir", type=str, default="./output/visualizations",
                      help="Directory to save visualizations (default: ./output/visualizations)")
    
    # Visualization options
    parser.add_argument("--dashboard", action="store_true",
                      help="Generate a comprehensive dashboard of results")
    parser.add_argument("--compare", action="store_true",
                      help="Compare multiple optimization runs")
    parser.add_argument("--cycle_plots", action="store_true",
                      help="Generate plots for each optimization cycle")
    parser.add_argument("--strategy_plots", action="store_true",
                      help="Generate strategy effectiveness plots")
    parser.add_argument("--head_history", action="store_true",
                      help="Generate head importance history plots")
    
    # Format options
    parser.add_argument("--figsize", type=str, default="12,8",
                      help="Figure size (width,height) in inches (default: 12,8)")
    parser.add_argument("--dpi", type=int, default=300,
                      help="Figure resolution (dots per inch) (default: 300)")
    parser.add_argument("--style", type=str, default="default",
                      choices=["default", "dark", "report"],
                      help="Visualization style (default: default)")
    
    return parser.parse_args()

def load_optimization_results(results_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load optimization results from a given directory.
    
    Args:
        results_dir: Directory containing optimization results
        
    Returns:
        Dictionary of optimization results or None if not found
    """
    # Try to find optimization_results.json
    results_path = os.path.join(results_dir, "optimization_results.json")
    if not os.path.exists(results_path):
        # Try metrics directory
        metrics_dir = os.path.join(results_dir, "metrics")
        if os.path.exists(metrics_dir):
            results_path = os.path.join(metrics_dir, "optimization_results.json")
    
    if not os.path.exists(results_path):
        print(f"Error: optimization_results.json not found in {results_dir}")
        return None
    
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"Error loading optimization results: {e}")
        return None

def load_head_history(results_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load head importance history from a given directory.
    
    Args:
        results_dir: Directory containing optimization results
        
    Returns:
        Dictionary of head importance history or None if not found
    """
    # Try to find head_importance data in analysis directory
    analysis_dir = os.path.join(results_dir, "analysis")
    if not os.path.exists(analysis_dir):
        print(f"Error: analysis directory not found in {results_dir}")
        return None
    
    # Find the latest head importance file
    importance_files = []
    for file in os.listdir(analysis_dir):
        if file.startswith("head_importance_") and file.endswith(".json"):
            importance_files.append(os.path.join(analysis_dir, file))
    
    if not importance_files:
        print(f"No head importance files found in {analysis_dir}")
        return None
    
    # Use the latest file
    latest_file = max(importance_files, key=os.path.getmtime)
    
    try:
        with open(latest_file, 'r') as f:
            history = json.load(f)
        return history
    except Exception as e:
        print(f"Error loading head importance history: {e}")
        return None

def find_optimization_runs(base_dir: str) -> List[str]:
    """
    Find optimization run directories in a given base directory.
    
    Args:
        base_dir: Base directory to search for runs
        
    Returns:
        List of directory paths containing optimization results
    """
    runs = []
    
    # Check if the base_dir itself is a run directory
    if os.path.exists(os.path.join(base_dir, "optimization_results.json")):
        runs.append(base_dir)
        return runs
    
    # Look for subdirectories containing optimization_results.json
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            if os.path.exists(os.path.join(item_path, "optimization_results.json")):
                runs.append(item_path)
            elif os.path.exists(os.path.join(item_path, "metrics", "optimization_results.json")):
                runs.append(item_path)
    
    return runs

def compare_optimization_runs(run_dirs: List[str], output_dir: str, figsize: tuple):
    """
    Compare multiple optimization runs.
    
    Args:
        run_dirs: List of directory paths containing optimization results
        output_dir: Directory to save visualizations
        figsize: Figure size (width, height) in inches
    """
    if len(run_dirs) < 2:
        print("Need at least 2 runs to compare")
        return
    
    # Load results from each run
    runs_data = []
    for run_dir in run_dirs:
        results = load_optimization_results(run_dir)
        if results:
            # Extract key metrics
            run_name = os.path.basename(run_dir)
            perplexity_improvement = results.get("perplexity_improvement", 0) * 100
            head_reduction = results.get("head_reduction", 0) * 100
            efficiency_improvement = results.get("efficiency_improvement", 0) * 100
            cycles = results.get("cycles_completed", 0)
            successful_cycles = results.get("successful_cycles", 0)
            success_rate = (successful_cycles / cycles * 100) if cycles > 0 else 0
            
            # Add to dataset
            runs_data.append({
                "run_name": run_name,
                "perplexity_improvement": perplexity_improvement,
                "head_reduction": head_reduction,
                "efficiency_improvement": efficiency_improvement,
                "cycles": cycles,
                "successful_cycles": successful_cycles,
                "success_rate": success_rate
            })
    
    if not runs_data:
        print("No valid optimization runs found")
        return
    
    # Create figure for comparison
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
    
    # Extract data for plotting
    run_names = [d["run_name"] for d in runs_data]
    perplexity_improvements = [d["perplexity_improvement"] for d in runs_data]
    head_reductions = [d["head_reduction"] for d in runs_data]
    efficiency_improvements = [d["efficiency_improvement"] for d in runs_data]
    
    x = np.arange(len(run_names))
    width = 0.8
    
    # Plot 1: Perplexity improvement
    ax1.bar(x, perplexity_improvements, width, label='Perplexity Improvement', color='blue', alpha=0.7)
    ax1.set_title("Perplexity Improvement Comparison")
    ax1.set_ylabel("Improvement (%)")
    ax1.grid(True, alpha=0.3)
    for i, v in enumerate(perplexity_improvements):
        ax1.text(i, v + 0.5, f"{v:.1f}%", ha='center')
    
    # Plot 2: Head reduction
    ax2.bar(x, head_reductions, width, label='Head Reduction', color='red', alpha=0.7)
    ax2.set_title("Head Reduction Comparison")
    ax2.set_ylabel("Reduction (%)")
    ax2.grid(True, alpha=0.3)
    for i, v in enumerate(head_reductions):
        ax2.text(i, v + 0.5, f"{v:.1f}%", ha='center')
    
    # Plot 3: Efficiency improvement
    ax3.bar(x, efficiency_improvements, width, label='Efficiency Improvement', color='green', alpha=0.7)
    ax3.set_title("Efficiency Improvement Comparison")
    ax3.set_xlabel("Optimization Run")
    ax3.set_ylabel("Improvement (%)")
    ax3.grid(True, alpha=0.3)
    for i, v in enumerate(efficiency_improvements):
        ax3.text(i, v + 0.5, f"{v:.1f}%", ha='center')
    
    # Set x-tick labels
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(x)
        ax.set_xticklabels(run_names, rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save comparison figure
    compare_path = os.path.join(output_dir, "runs_comparison.png")
    plt.savefig(compare_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison visualization to {compare_path}")

def main():
    """Main function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse figure size
    try:
        width, height = map(float, args.figsize.split(','))
        figsize = (width, height)
    except:
        print(f"Invalid figsize format, using default 12,8")
        figsize = (12, 8)
    
    # Set style
    if args.style == "dark":
        plt.style.use('dark_background')
    elif args.style == "report":
        plt.style.use('seaborn-v0_8-whitegrid')
    
    # Check if we're comparing multiple runs
    if args.compare:
        # Find optimization runs
        run_dirs = find_optimization_runs(args.results_dir)
        
        if not run_dirs:
            print(f"No optimization runs found in {args.results_dir}")
            return
        
        print(f"Found {len(run_dirs)} optimization runs to compare")
        compare_optimization_runs(run_dirs, args.output_dir, figsize)
        return
    
    # Load optimization results
    results = load_optimization_results(args.results_dir)
    if not results:
        return
    
    # Extract model and dataset names
    model_name = results.get("model_name", "Unknown Model")
    dataset_name = results.get("dataset_name", "Unknown Dataset")
    if not model_name or not dataset_name:
        # Try to find them in configuration
        config_path = os.path.join(args.results_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    model_name = config.get("model_name", model_name)
                    dataset_name = config.get("dataset", dataset_name)
            except Exception as e:
                print(f"Error loading configuration: {e}")
    
    print(f"Generating visualizations for {model_name} on {dataset_name}")
    
    # Generate dashboard
    if args.dashboard:
        dashboard_dir = create_dashboard(
            optimization_results=results,
            output_dir=args.output_dir,
            model_name=model_name,
            dataset_name=dataset_name
        )
        print(f"Generated dashboard at {dashboard_dir}")
        return
    
    # Generate cycle plots
    if args.cycle_plots or not (args.strategy_plots or args.head_history):
        # Extract data for cycle plots
        cycle_metrics = results.get("cycle_results", [])
        baseline_perplexity = results.get("baseline", {}).get("perplexity", 0)
        baseline_head_count = results.get("baseline", {}).get("head_count", 0)
        total_heads = baseline_head_count  # Assuming we start with all heads
        
        # Generate cycle comparison plot
        cycles_path = os.path.join(args.output_dir, "cycles_comparison.png")
        plot_cycles_comparison(
            cycle_metrics=cycle_metrics,
            baseline_perplexity=baseline_perplexity,
            baseline_head_count=baseline_head_count,
            total_heads=total_heads,
            figsize=figsize,
            save_path=cycles_path
        )
        print(f"Saved cycle comparison visualization to {cycles_path}")
    
    # Generate strategy plots
    if args.strategy_plots:
        if "strategy_effectiveness" in results:
            strategy_path = os.path.join(args.output_dir, "strategy_effectiveness.png")
            plot_strategy_effectiveness(
                strategy_results=results["strategy_effectiveness"],
                figsize=figsize,
                save_path=strategy_path
            )
            print(f"Saved strategy effectiveness visualization to {strategy_path}")
        else:
            print("No strategy effectiveness data found in results")
    
    # Generate head history plots
    if args.head_history:
        history = load_head_history(args.results_dir)
        if history:
            history_path = os.path.join(args.output_dir, "head_history.png")
            plot_plasticity_history(
                history=history,
                figsize=figsize,
                save_path=history_path
            )
            print(f"Saved head history visualization to {history_path}")
        else:
            print("No head importance history data found")

if __name__ == "__main__":
    main()