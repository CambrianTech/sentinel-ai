#!/usr/bin/env python3
"""
Multi-Model Pruning Comparison

Compares how different model architectures respond to pruning.
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
    parser = argparse.ArgumentParser(description="Compare pruning across multiple models")
    parser.add_argument("--strategy", type=str, choices=["random", "magnitude", "entropy"],
                        default="magnitude", help="Pruning strategy to test")
    parser.add_argument("--max_runtime", type=int, default=3600,
                        help="Maximum runtime in seconds (default: 1 hour)")
    parser.add_argument("--prompt", type=str, 
                        default="Artificial intelligence will revolutionize",
                        help="Prompt for text generation")
    parser.add_argument("--results_dir", type=str, default="pruning_results",
                        help="Directory to save results")
    parser.add_argument("--max_models", type=int, default=5,
                        help="Maximum number of models to test")
    
    args = parser.parse_args()
    
    # Initialize environment and detect capabilities
    env = Environment()
    env.print_info()
    
    # Get available models
    all_models = env.get_suitable_models()
    # Select a diverse set of models, limiting to max_models
    models = all_models[:min(args.max_models, len(all_models))]
    
    # Initialize results manager
    results_manager = ResultsManager(args.results_dir)
    results_manager.load_results()
    
    # Initialize benchmark runner
    benchmark = PruningBenchmark(results_manager)
    
    # Use moderate pruning levels 
    pruning_levels = [0.1, 0.3, 0.5, 0.7]
    
    print(f"Running multi-model pruning comparison with:")
    print(f"  Models: {models}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Pruning levels: {pruning_levels}")
    print(f"  Prompt: '{args.prompt}'")
    print(f"  Maximum runtime: {args.max_runtime/3600:.1f} hours")
    
    # Run the benchmarks
    results = benchmark.run_multiple_benchmarks(
        models=models,
        strategies=[args.strategy],
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
    
    # Model-specific analysis
    if len(models) > 1:
        plt.figure(figsize=(12, 6))
        
        # Filter for just our strategy
        if results_manager.results_df is not None and not results_manager.results_df.empty:
            strategy_df = results_manager.results_df[
                results_manager.results_df["strategy"] == args.strategy
            ]
            
            # Plot comparison
            for model in models:
                model_data = strategy_df[strategy_df["model"] == model]
                if not model_data.empty:
                    # Sort by pruning level
                    model_data = model_data.sort_values("pruning_level")
                    plt.plot(
                        model_data["pruning_level"], 
                        model_data["perplexity_change"],
                        marker='o',
                        label=model
                    )
            
            plt.xlabel("Pruning Level")
            plt.ylabel("Perplexity Change")
            plt.title(f"Model Comparison with {args.strategy.capitalize()} Pruning")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())