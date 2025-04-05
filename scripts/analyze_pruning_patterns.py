#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze Pruning Patterns

This script analyzes pruning patterns from model runs to identify regularities,
correlations between gates and performance, and compare different pruning strategies.

Usage:
    python scripts/analyze_pruning_patterns.py --results_dir ./pruning_results --output_dir ./analysis
"""

import os
import sys
import argparse
import json
import glob
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from sentinel.utils.metric_collection import analyze_pruning_strategy, compare_pruning_strategies


def setup_args():
    """Set up command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze pruning patterns")
    
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing pruning experiment results")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save analysis results")
    parser.add_argument("--compare", action="store_true",
                       help="Compare multiple pruning strategies")
    parser.add_argument("--strategy_dirs", type=str, default=None,
                       help="Comma-separated list of 'name:directory' pairs for strategy comparison")
    
    return parser.parse_args()


def find_strategy_directories(base_dir):
    """Automatically detect strategy directories."""
    strategy_dirs = {}
    
    # Look for directories that contain pruning results
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Check if this directory contains metrics files
            metrics_files = glob.glob(os.path.join(item_path, "**/*metrics*.jsonl"), recursive=True)
            if metrics_files:
                strategy_dirs[item] = item_path
    
    return strategy_dirs


def main():
    """Main function to run pruning pattern analysis."""
    args = setup_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' does not exist")
        return 1
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "analysis")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.compare:
        # Compare multiple strategies
        if args.strategy_dirs:
            # Parse strategy directories from command line
            strategy_dirs = {}
            for pair in args.strategy_dirs.split(","):
                name, directory = pair.split(":")
                strategy_dirs[name] = directory
        else:
            # Auto-detect strategy directories
            strategy_dirs = find_strategy_directories(args.results_dir)
        
        if not strategy_dirs:
            print("Error: No strategy directories found for comparison")
            return 1
        
        print(f"Comparing {len(strategy_dirs)} pruning strategies:")
        for name, directory in strategy_dirs.items():
            print(f"  - {name}: {directory}")
        
        results = compare_pruning_strategies(strategy_dirs, args.output_dir)
        
        # Print summary
        if "overall_best_strategy" in results:
            best = results["overall_best_strategy"]
            print(f"\nOverall best strategy: {best['strategy']} ({best['wins']}/{best['total_metrics']} metrics)")
        
        if "win_counts" in results:
            print("\nWin counts by strategy:")
            for strategy, wins in sorted(results["win_counts"].items(), key=lambda x: x[1], reverse=True):
                print(f"  - {strategy}: {wins}")
        
        print(f"\nDetailed results saved to {args.output_dir}")
        
    else:
        # Analyze a single strategy
        print(f"Analyzing pruning results in {args.results_dir}")
        
        results = analyze_pruning_strategy(args.results_dir, args.output_dir)
        
        # Print summary
        if "summary" in results:
            print("\nPerformance summary:")
            for phase, metrics in results["summary"].items():
                print(f"\n  {phase.upper()} PHASE:")
                for metric, values in metrics.items():
                    if "final" in values and values["final"] is not None:
                        print(f"    {metric}: {values['final']:.4f}")
        
        if "improvement" in results:
            print("\nImprovements:")
            for phase, metrics in results["improvement"].items():
                print(f"\n  {phase.upper()} PHASE:")
                for metric, values in metrics.items():
                    if all(v is not None for v in [values.get("first"), values.get("last"), values.get("relative_change")]):
                        rel_change = values["relative_change"] * 100
                        print(f"    {metric}: {values['first']:.4f} → {values['last']:.4f} ({rel_change:+.2f}%)")
        
        print(f"\nDetailed results saved to {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())