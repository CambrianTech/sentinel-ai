#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pruning and Fine-Tuning Benchmark Script

This script runs the pruning + fine-tuning experiment that demonstrates
how models can recover or improve performance through fine-tuning after pruning.
It uses JAX/Flax for both pruning and fine-tuning, which works reliably on all
platforms including M1/M2 Macs and Colab TPUs/GPUs.

Example usage:
    python scripts/pruning_and_finetuning.py --strategies random magnitude entropy --pruning_levels 0.1 0.3 0.5 --max_runtime 6h
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

# Import custom modules
from utils.pruning import (
    Environment, 
    PruningModule,
    ResultsManager,
    FineTuner,
    ImprovedFineTuner
)


def parse_time(time_str):
    """Parse time string like '1h', '30m', '1h30m', '3600s' to seconds"""
    seconds = 0
    if 'h' in time_str:
        h_parts = time_str.split('h')
        seconds += int(h_parts[0]) * 3600
        time_str = h_parts[1] if len(h_parts) > 1 else ''
    if 'm' in time_str:
        m_parts = time_str.split('m')
        seconds += int(m_parts[0]) * 60
        time_str = m_parts[1] if len(m_parts) > 1 else ''
    if 's' in time_str:
        seconds += int(time_str.rstrip('s'))
    if not seconds and time_str.isdigit():
        # Assume seconds if no unit specified
        seconds = int(time_str)
    return seconds


def setup_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run pruning and fine-tuning experiments")
    
    # Model selection
    parser.add_argument("--models", type=str, nargs="+", 
                        help="List of models to test (if not provided, will use models suitable for the environment)")
    
    # Pruning parameters
    parser.add_argument("--strategies", type=str, nargs="+", default=["random", "magnitude"],
                       help="Pruning strategies to test")
    parser.add_argument("--pruning_levels", type=float, nargs="+", default=[0.1, 0.3, 0.5],
                       help="Pruning levels to test")
    
    # Fine-tuning parameters
    parser.add_argument("--fine_tuning_epochs", type=int, default=2,
                       help="Number of fine-tuning epochs per model")
    parser.add_argument("--dataset", type=str, default="wikitext",
                       help="Dataset to use for fine-tuning")
    
    # Experiment parameters
    parser.add_argument("--prompt", type=str, 
                        default="Artificial intelligence will transform society by",
                       help="Prompt to use for evaluation")
    parser.add_argument("--results_dir", type=str, default="pruning_finetuning_results",
                       help="Directory to save results")
    parser.add_argument("--max_runtime", type=str, default="6h",
                       help="Maximum runtime in format like '6h', '30m', '3600s', etc.")
    
    return parser.parse_args()


class CommandLineExperiment:
    """Run pruning and fine-tuning experiment from command line"""
    
    def __init__(self, args):
        self.args = args
        self.results_dir = Path(args.results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.results = []
        
        # Initialize environment
        self.env = Environment()
        self.env.print_info()
        
        # Get models
        if args.models:
            self.models = args.models
        else:
            self.models = self.env.get_suitable_models()
        
        print(f"Models selected: {', '.join(self.models)}")
        
        # Setup paths
        self.log_file = self.results_dir / f"experiment_log_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
        
        # Convert max_runtime to seconds
        self.max_runtime = parse_time(args.max_runtime)
        print(f"Maximum runtime: {self.max_runtime/3600:.1f} hours")
    
    def log(self, message):
        """Log message to file and print to console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, "a") as f:
            f.write(log_message + "\n")
    
    def run(self):
        """Run the experiment"""
        self.log(f"Starting pruning and fine-tuning experiment")
        self.log(f"Strategies: {self.args.strategies}")
        self.log(f"Pruning levels: {self.args.pruning_levels}")
        self.log(f"Fine-tuning epochs: {self.args.fine_tuning_epochs}")
        self.log(f"Dataset: {self.args.dataset}")
        self.log(f"Prompt: '{self.args.prompt}'")
        
        # Generate experiment configurations
        experiments = []
        for model in self.models:
            for strategy in self.args.strategies:
                for level in self.args.pruning_levels:
                    experiments.append({
                        "model": model,
                        "strategy": strategy,
                        "pruning_level": level
                    })
        
        self.log(f"Generated {len(experiments)} experiment configurations")
        
        # Track start time
        start_time = time.time()
        completed = 0
        
        # Run experiments
        for i, config in enumerate(experiments):
            # Check runtime limit
            current_runtime = time.time() - start_time
            if current_runtime > self.max_runtime:
                self.log(f"Reached maximum runtime of {self.max_runtime/3600:.1f} hours")
                break
            
            model = config["model"]
            strategy = config["strategy"]
            level = config["pruning_level"]
            
            self.log(f"Running experiment {i+1}/{len(experiments)}: {model}, {strategy}, {level:.2f}")
            
            try:
                # Run experiment
                result = self.run_single_experiment(
                    model=model,
                    strategy=strategy,
                    pruning_level=level,
                    prompt=self.args.prompt,
                    dataset=self.args.dataset,
                    fine_tuning_epochs=self.args.fine_tuning_epochs
                )
                
                if result:
                    self.results.append(result)
                    self.log(f"Experiment completed successfully")
                    completed += 1
                else:
                    self.log(f"Experiment failed")
            except Exception as e:
                self.log(f"Error in experiment: {e}")
                import traceback
                self.log(traceback.format_exc())
        
        # Final summary
        runtime = time.time() - start_time
        self.log(f"Experiment completed. Total runtime: {runtime/3600:.2f} hours")
        self.log(f"Completed {completed} out of {len(experiments)} planned experiments")
        
        # Save final results
        self.save_results()
        
        return self.results
    
    def run_single_experiment(self, model, strategy, pruning_level, prompt, dataset, fine_tuning_epochs):
        """Run a single experiment"""
        # Create results structure
        result = {
            "model": model,
            "strategy": strategy,
            "pruning_level": pruning_level,
            "prompt": prompt,
            "dataset": dataset,
            "fine_tuning_epochs": fine_tuning_epochs,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stages": {}
        }
        
        # Initialize pruning module
        self.log(f"Loading model {model}...")
        pruning_module = PruningModule(model)
        if not pruning_module.load_model():
            self.log(f"Failed to load model {model}")
            return None
        
        # 1. Evaluate baseline model
        self.log(f"Evaluating baseline model...")
        original_params = pruning_module.original_params
        
        # Evaluate perplexity and generation
        perplexity_baseline = pruning_module.evaluate_perplexity(original_params, prompt)
        self.log(f"Baseline perplexity: {perplexity_baseline:.4f}")
        
        generated_baseline = pruning_module.generate_text(original_params, prompt)
        self.log(f"Baseline generated: {generated_baseline}")
        
        # Record baseline results
        result["stages"]["baseline"] = {
            "perplexity": float(perplexity_baseline),
            "generated_text": generated_baseline
        }
        
        # 2. Apply pruning
        self.log(f"Applying {strategy} pruning at {pruning_level:.2f} level...")
        
        # Get strategy
        from utils.pruning.strategies import get_strategy
        pruning_strat = get_strategy(strategy, pruning_module, prompt)
        
        # Calculate importance scores
        self.log(f"Calculating head importance...")
        all_head_importance = pruning_strat.get_head_importance(original_params)
        
        # Sort by importance (ascending)
        all_head_importance.sort(key=lambda x: x[2])
        
        # Determine number of heads to prune
        total_heads = pruning_module.num_layers * pruning_module.num_heads
        heads_to_prune = int(total_heads * pruning_level)
        self.log(f"Pruning {heads_to_prune} out of {total_heads} heads")
        
        # Get head indices to prune (least important first)
        head_indices = [(l, h) for l, h, _ in all_head_importance[:heads_to_prune]]
        
        # Prune heads
        pruned_params = pruning_strat.prune_heads(original_params, head_indices)
        
        # Evaluate after pruning
        perplexity_pruned = pruning_module.evaluate_perplexity(pruned_params, prompt)
        self.log(f"Post-pruning perplexity: {perplexity_pruned:.4f}")
        
        generated_pruned = pruning_module.generate_text(pruned_params, prompt)
        self.log(f"Post-pruning generated: {generated_pruned}")
        
        # Record pruning results
        result["stages"]["pruned"] = {
            "perplexity": float(perplexity_pruned),
            "perplexity_change": float(perplexity_pruned - perplexity_baseline),
            "generated_text": generated_pruned,
            "pruned_heads": heads_to_prune,
            "total_heads": total_heads
        }
        
        # 3. Fine-tune the pruned model
        self.log(f"Fine-tuning pruned model with {dataset} dataset for {fine_tuning_epochs} epochs...")
        
        # Determine batch size based on environment
        if self.env.in_colab and self.env.has_tpu:
            batch_size = 16
        elif self.env.in_colab and self.env.has_gpu:
            batch_size = 8
        else:
            batch_size = 4
        
        # Check if model name indicates this might be a large model (OPT-1.3B, etc.)
        model_name = model.lower()
        use_improved_tuner = any(x in model_name for x in ['opt', 'large', '1.3b', 'bloom'])
        
        if use_improved_tuner:
            self.log(f"Using ImprovedFineTuner for model {model} to enhance stability")
            # Initialize improved fine-tuner with better stability for large models
            fine_tuner = ImprovedFineTuner(
                pruning_module, 
                dataset_name=dataset,
                batch_size=batch_size
            )
        else:
            # Use standard fine-tuner for smaller models
            fine_tuner = FineTuner(
                pruning_module, 
                dataset_name=dataset,
                batch_size=batch_size
            )
        
        # Adjust learning rate for large models
        learning_rate = 1e-5 if use_improved_tuner else 5e-5
        
        # Fine-tune model
        try:
            tuned_params, metrics = fine_tuner.fine_tune(
                pruned_params,
                num_epochs=fine_tuning_epochs,
                learning_rate=learning_rate,
                evaluate_interval=5
            )
        except Exception as e:
            self.log(f"Error during fine-tuning: {e}")
            # If standard tuner fails, fall back to improved tuner
            if not use_improved_tuner:
                self.log("Falling back to ImprovedFineTuner after error")
                fine_tuner = ImprovedFineTuner(
                    pruning_module, 
                    dataset_name=dataset,
                    batch_size=max(1, batch_size // 2)  # Reduce batch size
                )
                tuned_params, metrics = fine_tuner.fine_tune(
                    pruned_params,
                    num_epochs=fine_tuning_epochs,
                    learning_rate=1e-5,  # Lower learning rate for stability
                    evaluate_interval=5
                )
        
        # Evaluate fine-tuned model
        perplexity_tuned = pruning_module.evaluate_perplexity(tuned_params, prompt)
        self.log(f"Post-fine-tuning perplexity: {perplexity_tuned:.4f}")
        
        generated_tuned = pruning_module.generate_text(tuned_params, prompt)
        self.log(f"Post-fine-tuning generated: {generated_tuned}")
        
        # Record fine-tuning results
        result["stages"]["fine_tuned"] = {
            "perplexity": float(perplexity_tuned),
            "perplexity_change_from_baseline": float(perplexity_tuned - perplexity_baseline),
            "perplexity_change_from_pruned": float(perplexity_tuned - perplexity_pruned),
            "generated_text": generated_tuned,
            "training_metrics": metrics
        }
        
        # Calculate recovery or improvement
        if perplexity_pruned > perplexity_baseline:
            # Calculate recovery percentage
            perplexity_increase = perplexity_pruned - perplexity_baseline
            perplexity_recovery = perplexity_pruned - perplexity_tuned
            recovery_percentage = (perplexity_recovery / perplexity_increase) * 100 if perplexity_increase > 0 else 0
            
            result["stages"]["fine_tuned"]["recovery_percentage"] = float(recovery_percentage)
            self.log(f"Recovery percentage: {recovery_percentage:.2f}%")
        else:
            # Calculate improvement percentage
            improvement_percentage = ((perplexity_baseline - perplexity_tuned) / perplexity_baseline) * 100
            
            result["stages"]["fine_tuned"]["improvement_percentage"] = float(improvement_percentage)
            self.log(f"Improvement percentage: {improvement_percentage:.2f}%")
        
        # Save individual result
        result_filename = f"{model.replace('/', '_')}_{strategy}_{pruning_level:.2f}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        result_path = self.results_dir / result_filename
        
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        
        self.log(f"Saved result to {result_path}")
        
        return result
    
    def save_results(self):
        """Save all results to a single file"""
        if not self.results:
            self.log("No results to save")
            return
        
        summary_path = self.results_dir / f"summary_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        
        with open(summary_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "environment": {
                    "in_colab": self.env.in_colab,
                    "is_mac": self.env.is_mac,
                    "is_arm_mac": self.env.is_arm_mac,
                    "default_device": self.env.default_device
                },
                "config": {
                    "models": self.models,
                    "strategies": self.args.strategies,
                    "pruning_levels": self.args.pruning_levels,
                    "fine_tuning_epochs": self.args.fine_tuning_epochs,
                    "dataset": self.args.dataset,
                    "prompt": self.args.prompt,
                    "max_runtime": self.max_runtime
                },
                "results": self.results
            }, f, indent=2)
        
        self.log(f"Saved summary to {summary_path}")


def main():
    """Main entry point"""
    args = setup_args()
    experiment = CommandLineExperiment(args)
    experiment.run()


if __name__ == "__main__":
    main()