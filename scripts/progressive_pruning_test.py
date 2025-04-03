#!/usr/bin/env python3
"""
Progressive Pruning Test

Tests how much we can prune before the model completely breaks down.
Uses the JAX/Flax pruning library for stable operation on M1/M2 Macs.
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Set up pretty plots
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Import our pruning library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pruning import (
    Environment, 
    ResultsManager, 
    PruningBenchmark
)

def main():
    parser = argparse.ArgumentParser(description="Run progressive pruning test")
    parser.add_argument("--models", nargs="+", help="Models to test")
    parser.add_argument("--strategies", nargs="+", choices=["random", "magnitude", "entropy"],
                        default=["random", "magnitude"], help="Pruning strategies to test")
    parser.add_argument("--max_runtime", type=int, default=3600,
                        help="Maximum runtime in seconds (default: 1 hour)")
    parser.add_argument("--prompt", type=str, 
                        default="Artificial intelligence will revolutionize",
                        help="Prompt for text generation")
    parser.add_argument("--results_dir", type=str, default="pruning_results",
                        help="Directory to save results")
    parser.add_argument("--fine_grained", action="store_true",
                        help="Use fine-grained pruning levels")
    
    args = parser.parse_args()
    
    # Initialize environment and detect capabilities
    env = Environment()
    env.print_info()
    
    # Get available models if not specified
    if args.models is None:
        models = env.get_suitable_models()[:2]  # Use first two available models
    else:
        models = args.models
    
    # Initialize results manager
    results_manager = ResultsManager(args.results_dir)
    results_manager.load_results()
    
    # Initialize benchmark runner
    benchmark = PruningBenchmark(results_manager)
    
    # Set pruning levels - finer-grained if requested
    if args.fine_grained:
        pruning_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 
                          0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    else:
        pruning_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    print(f"Running progressive pruning test with:")
    print(f"  Models: {models}")
    print(f"  Strategies: {args.strategies}")
    print(f"  Pruning levels: {pruning_levels}")
    print(f"  Prompt: '{args.prompt}'")
    print(f"  Maximum runtime: {args.max_runtime/3600:.1f} hours")
    
    # Run the benchmarks
    results = benchmark.run_multiple_benchmarks(
        models=models,
        strategies=args.strategies,
        pruning_levels=pruning_levels,
        prompt=args.prompt,
        max_runtime=args.max_runtime
    )
    
    # Final analysis
    results_manager.load_results()
    results_manager.print_summary()
    
    # Basic visualization
    results_manager.plot_results(figsize=(12, 8))
    
    # Advanced visualization
    if hasattr(results_manager, 'plot_advanced_analysis'):
        results_manager.plot_advanced_analysis(figsize=(14, 10))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())