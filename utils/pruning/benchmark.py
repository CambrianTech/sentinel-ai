"""
Benchmark runner for pruning experiments
"""

import random
import time
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from .pruning_module import PruningModule
from .strategies import get_strategy

class PruningBenchmark:
    """Main benchmark runner"""
    
    def __init__(self, results_manager):
        self.results_manager = results_manager
    
    def run_single_benchmark(self, model_name, strategy_name, pruning_level, prompt):
        """Run a single pruning benchmark"""
        print(f"\nRunning benchmark for {model_name} with {strategy_name} strategy at {pruning_level:.2f} pruning level")
        
        # Initialize pruning module
        pruning_module = PruningModule(model_name)
        if not pruning_module.load_model():
            print(f"Failed to load model {model_name}")
            return None
        
        # Get strategy
        strategy = get_strategy(strategy_name, pruning_module, prompt)
        
        # Get original parameters and create a copy for pruning
        original_params = pruning_module.original_params
        params = original_params  # No need to copy yet, will be copied by the strategy
        
        # Evaluate model before pruning
        print("Evaluating model before pruning...")
        perplexity_before = pruning_module.evaluate_perplexity(params, prompt)
        print(f"Perplexity before pruning: {perplexity_before:.4f}")
        
        generated_before = pruning_module.generate_text(params, prompt)
        print(f"Generated (before pruning): {generated_before}")
        
        # Calculate importance scores
        print("\nCalculating head importance...")
        all_head_importance = strategy.get_head_importance(params)
        
        # Sort by importance (ascending)
        all_head_importance.sort(key=lambda x: x[2])
        
        # Determine number of heads to prune
        total_heads = pruning_module.num_layers * pruning_module.num_heads
        heads_to_prune = int(total_heads * pruning_level)
        print(f"Pruning {heads_to_prune} out of {total_heads} heads")
        
        # Get head indices to prune (least important first)
        head_indices = [(l, h) for l, h, _ in all_head_importance[:heads_to_prune]]
        
        # Prune heads
        print("\nPruning heads...")
        pruned_params = strategy.prune_heads(params, head_indices)
        
        # Evaluate model after pruning
        print("\nEvaluating model after pruning...")
        perplexity_after = pruning_module.evaluate_perplexity(pruned_params, prompt)
        print(f"Perplexity after pruning: {perplexity_after:.4f}")
        print(f"Perplexity change: {perplexity_after - perplexity_before:.4f}")
        
        generated_after = pruning_module.generate_text(pruned_params, prompt)
        print(f"Generated (after pruning): {generated_after}")
        
        # Prepare result
        result = {
            "model": model_name,
            "strategy": strategy_name,
            "pruning_level": pruning_level,
            "pruned_heads": heads_to_prune,
            "total_heads": total_heads,
            "prompt": prompt,
            "perplexity_before": float(perplexity_before),
            "perplexity_after": float(perplexity_after),
            "perplexity_change": float(perplexity_after - perplexity_before),
            "generated_before": generated_before,
            "generated_after": generated_after,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save result
        self.results_manager.save_result(result)
        
        print("\nBenchmark completed successfully!")
        return result
    
    def run_multiple_benchmarks(self, models=None, strategies=None, pruning_levels=None, prompt=None, max_runtime=None):
        """Run multiple benchmarks with different parameters"""
        # Default values
        if models is None:
            # You would need to import Environment here or pass it as a parameter
            raise ValueError("Models must be specified when no Environment is available")
        if strategies is None:
            strategies = ["random", "magnitude"]
        if pruning_levels is None:
            pruning_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        if prompt is None:
            prompt = "Artificial intelligence will transform"
            
        # Start time for runtime tracking
        start_time = time.time()
        
        # Generate all benchmark combinations
        benchmarks = []
        for model in models:
            for strategy in strategies:
                for level in pruning_levels:
                    benchmarks.append((model, strategy, level, prompt))
        
        # Shuffle to get more diverse results early
        random.shuffle(benchmarks)
        
        # Create progress bar
        pbar = tqdm(total=len(benchmarks), desc="Running benchmarks")
        
        # Run benchmarks
        results = []
        for i, (model, strategy, level, bench_prompt) in enumerate(benchmarks):
            # Check if we've exceeded the runtime limit
            if max_runtime is not None and time.time() - start_time > max_runtime:
                print(f"\nReached maximum runtime of {max_runtime/3600:.1f} hours")
                break
                
            # Update progress bar
            pbar.set_description(f"Running {model}, {strategy}, {level:.2f}")
            
            # Run benchmark
            try:
                result = self.run_single_benchmark(model, strategy, level, bench_prompt)
                if result is not None:
                    results.append(result)
                
                # Update progress bar
                pbar.update(1)
                
                # Plot intermediate results every few benchmarks
                if (i + 1) % 3 == 0 or i == len(benchmarks) - 1:
                    self.results_manager.plot_results()
                    plt.close()
            except Exception as e:
                print(f"Error in benchmark {model}, {strategy}, {level:.2f}: {e}")
                # Still update progress bar
                pbar.update(1)
        
        # Close progress bar
        pbar.close()
        
        # Final results
        print(f"\nCompleted {len(results)} benchmarks out of {len(benchmarks)} attempted")
        runtime = time.time() - start_time
        print(f"Total runtime: {runtime/3600:.2f} hours ({runtime/60:.2f} minutes)")
        
        # Plot final results
        self.results_manager.plot_results()
        
        return results