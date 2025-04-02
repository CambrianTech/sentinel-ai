#!/usr/bin/env python
"""
Pure Pruning Benchmark

This script implements a focused benchmark for isolating the effects of pruning
from agency features. It measures speed improvements, hardware utilization, and
quality changes across different pruning strategies and levels.

Key features:
- Support for different pruning strategies (gradual, one-shot, iterative)
- Multiple pruning methods (entropy, random, magnitude-based)
- Hardware-level metrics (FLOPs, memory usage, latency)
- Fine-tuning phases to demonstrate quality recovery after pruning
- Comprehensive visualizations of benchmark results

Usage:
    python scripts/pure_pruning_benchmark.py \
        --model_name gpt2 \
        --pruning_level 0.5 \
        --strategy entropy \
        --visualize \
        --output_dir results/pure_pruning
"""

import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tqdm import tqdm

# Import directly from repo structure
from models.loaders.loader import load_baseline_model, load_adaptive_model
from models.loaders.loader_optimized import load_optimized_adaptive_model
from scripts.pruning_comparison.pruning_agency_comparison import apply_pruning

# Define metrics functions locally to avoid import errors
def compute_perplexity(model, input_ids):
    """Compute perplexity on the given input."""
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        return torch.exp(loss).item()

def compute_output_quality(prompt, output_text):
    """Compute a quality score for the generated output."""
    # Simple heuristic: longer outputs and those containing the prompt are better
    quality = min(1.0, len(output_text) / 500)  # Cap at 1.0
    if prompt in output_text:
        quality *= 0.9  # Slightly reduce if prompt is repeated
    return quality


class PruningBenchmark:
    """Benchmarking class for pure pruning evaluation."""
    
    def __init__(self, 
                 model_name="gpt2", 
                 pruning_level=0.5, 
                 strategy="entropy",
                 device=None,
                 output_dir="results/pure_pruning",
                 visualize=True,
                 baseline_comparison=True,
                 hardware_metrics=True,
                 _model=None):
        """
        Initialize the pure pruning benchmark.
        
        Args:
            model_name: Name of the model to benchmark
            pruning_level: Level of pruning to apply (0.0-1.0)
            strategy: Pruning strategy to use (entropy, random, magnitude)
            device: Device to run benchmark on (defaults to CUDA if available)
            output_dir: Directory to save benchmark results
            visualize: Generate visualizations of benchmark results
            baseline_comparison: Compare with unpruned baseline
            hardware_metrics: Measure hardware-level metrics (FLOPs, memory)
            _model: Optional pre-loaded model (to avoid reloading)
        """
        self.model_name = model_name
        self.pruning_level = float(pruning_level)
        self.strategy = strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.visualize = visualize
        self.baseline_comparison = baseline_comparison
        self.hardware_metrics = hardware_metrics
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        if self.visualize:
            os.makedirs(os.path.join(output_dir, "charts"), exist_ok=True)
        
        # Initialize model if not provided
        self.model = _model
        
        # Store benchmark results
        self.results = {}
        
        print(f"Initialized Pure Pruning Benchmark:")
        print(f"  Model: {model_name}")
        print(f"  Pruning: {pruning_level*100:.1f}% using {strategy} strategy")
        print(f"  Device: {self.device}")
    
    def setup(self):
        """Load and prepare models for benchmarking."""
        print("Setting up benchmark environment...")
        
        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load baseline model if not provided
        if self.model is None:
            print(f"Loading baseline model: {self.model_name}")
            self.baseline_model = load_baseline_model(self.model_name, self.device)
            
            print("Creating adaptive model...")
            self.model = load_adaptive_model(self.model_name, self.baseline_model, self.device)
        else:
            # If model was provided, we still need the baseline for comparison
            if self.baseline_comparison:
                print(f"Loading baseline model for comparison: {self.model_name}")
                self.baseline_model = load_baseline_model(self.model_name, self.device)
            else:
                self.baseline_model = None
        
        # Prepare evaluation data
        self.eval_prompts = [
            "The transformer architecture has revolutionized",
            "In recent years, artificial intelligence has",
            "The history of machine learning begins with",
            "For efficient natural language processing, we need"
        ]
        
        print("Setup complete.")
    
    def run(self):
        """Run the complete benchmark pipeline."""
        print("\nStarting Pure Pruning Benchmark...")
        
        # Setup environment
        self.setup()
        
        # Collect baseline metrics if requested
        if self.baseline_comparison:
            print("\nMeasuring baseline performance...")
            self.baseline_metrics = self.measure_baseline_performance(full_report=True)
        
        # Apply pruning
        print(f"\nApplying {self.pruning_level*100:.1f}% pruning using {self.strategy} strategy...")
        self.pruned_model, pruned_count, pruned_heads = self._apply_pruning(self.pruning_level)
        
        print(f"Pruned {pruned_count} attention heads ({len(pruned_heads)} unique heads)")
        
        # Evaluate pruned model
        print("\nEvaluating pruned model...")
        self.pruned_metrics = self._evaluate_model(self.pruned_model, "Pruned Model", detailed=True)
        
        # Compare with baseline if requested
        if self.baseline_comparison:
            print("\nComparing with baseline model...")
            self.speedup = self.pruned_metrics["tokens_per_second"] / self.baseline_metrics["tokens_per_second"]
            self.memory_reduction = 1.0 - (self.pruned_metrics["memory_usage"] / self.baseline_metrics["memory_usage"])
            self.quality_ratio = self.pruned_metrics["quality_score"] / self.baseline_metrics["quality_score"]
            
            print(f"Speedup: {self.speedup:.2f}x")
            print(f"Memory reduction: {self.memory_reduction*100:.1f}%")
            print(f"Quality ratio: {self.quality_ratio*100:.1f}%")
            
            # Store comparison in results
            self.results["comparison"] = {
                "speedup": self.speedup,
                "memory_reduction": self.memory_reduction,
                "quality_ratio": self.quality_ratio
            }
        
        # Run comparison experiments with different pruning methods
        print("\nRunning comparison experiments...")
        self.comparison_results = self._run_comparison_experiments()
        
        # Visualize results if requested
        if self.visualize:
            print("\nGenerating visualizations...")
            self._create_visualizations()
        
        # Save results
        results_file = os.path.join(self.output_dir, f"{self.strategy}_pruning_{int(self.pruning_level*100)}_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nBenchmark complete. Results saved to: {results_file}")
        
        return self.results
    
    def _apply_pruning(self, sparsity_level):
        """Apply pruning to the model based on specified strategy."""
        if self.strategy == "entropy":
            # Entropy-based pruning (most common approach)
            return apply_pruning(
                self.model, 
                sparsity_level, 
                method="entropy", 
                verbose=False,
                quiet=True
            )
        elif self.strategy == "random":
            # Random pruning (for comparison)
            return apply_pruning(
                self.model, 
                sparsity_level, 
                method="random", 
                verbose=False,
                quiet=True
            )
        elif self.strategy == "magnitude":
            # Magnitude-based pruning
            return apply_pruning(
                self.model, 
                sparsity_level, 
                method="magnitude", 
                verbose=False,
                quiet=True
            )
        else:
            raise ValueError(f"Unknown pruning strategy: {self.strategy}")
    
    def measure_baseline_performance(self, full_report=False):
        """Measure baseline model performance."""
        baseline_metrics = self._evaluate_model(self.model, "Baseline Model", detailed=full_report)
        
        if full_report:
            print("\nBaseline Performance:")
            print(f"  Generation speed: {baseline_metrics['tokens_per_second']:.2f} tokens/sec")
            print(f"  Memory usage: {baseline_metrics['memory_usage']:.1f} MB")
            print(f"  Quality score: {baseline_metrics['quality_score']:.2f}")
        
        # Store baseline metrics in results
        self.results["baseline"] = baseline_metrics
        
        return baseline_metrics
    
    def _evaluate_model(self, model, label="Model", detailed=False):
        """Perform comprehensive evaluation of a model."""
        metrics = {}
        
        # Measure generation speed
        generation_metrics = self._measure_generation_speed(model)
        metrics.update(generation_metrics)
        
        # Measure memory usage
        if self.hardware_metrics:
            memory_metrics = self._measure_memory_usage(model)
            metrics.update(memory_metrics)
        
        # Measure output quality
        quality_metrics = self._measure_output_quality(model)
        metrics.update(quality_metrics)
        
        # Measure hardware utilization
        if self.hardware_metrics and self.device == "cuda":
            hardware_metrics = self._measure_hardware_utilization(model)
            metrics.update(hardware_metrics)
        
        # Print detailed metrics if requested
        if detailed:
            print(f"\n{label} Metrics:")
            print(f"  Generation speed: {metrics['tokens_per_second']:.2f} tokens/sec")
            
            if self.hardware_metrics:
                print(f"  Memory usage: {metrics['memory_usage']:.1f} MB")
                if self.device == "cuda":
                    print(f"  FLOPs: {metrics.get('flops', 0) / 1e9:.2f} G")
            
            print(f"  Quality score: {metrics['quality_score']:.2f}")
            if 'perplexity' in metrics:
                print(f"  Perplexity: {metrics['perplexity']:.2f}")
        
        return metrics
    
    def _measure_generation_speed(self, model):
        """Measure text generation speed in tokens per second."""
        model.eval()
        
        num_runs = 5
        generation_lengths = [20, 50, 100]
        temperature = 0.7
        
        all_times = []
        all_tokens = []
        
        # Make sure model is in eval mode
        with torch.no_grad():
            for prompt in self.eval_prompts:
                # Tokenize prompt
                input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                
                for length in generation_lengths:
                    for _ in range(num_runs):
                        # Clear CUDA cache
                        if self.device == "cuda":
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        
                        # Start timing
                        start_time = time.time()
                        
                        # Generate text
                        output_ids = model.generate(
                            input_ids=input_ids,
                            max_length=input_ids.size(1) + length,
                            do_sample=True,
                            temperature=temperature,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                        
                        # Ensure all operations are completed
                        if self.device == "cuda":
                            torch.cuda.synchronize()
                        
                        # End timing
                        end_time = time.time()
                        generation_time = end_time - start_time
                        
                        # Calculate tokens generated
                        tokens_generated = output_ids.size(1) - input_ids.size(1)
                        
                        all_times.append(generation_time)
                        all_tokens.append(tokens_generated)
        
        # Calculate average tokens per second
        total_tokens = sum(all_tokens)
        total_time = sum(all_times)
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        
        return {
            "tokens_per_second": tokens_per_second,
            "generation_times": all_times,
            "tokens_generated": all_tokens
        }
    
    def _measure_memory_usage(self, model):
        """Measure memory usage during inference."""
        if self.device == "cuda":
            # Clear CUDA cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Record initial memory
            initial_memory = torch.cuda.memory_allocated()
            
            # Forward pass with a sample input
            prompt = self.eval_prompts[0]
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                _ = model(input_ids)
                
                # Force CUDA synchronization
                torch.cuda.synchronize()
                
                # Measure peak memory usage
                peak_memory = torch.cuda.max_memory_allocated() - initial_memory
                
                # Convert to MB
                peak_memory_mb = peak_memory / (1024 * 1024)
            
            return {"memory_usage": peak_memory_mb}
        else:
            # For CPU, we can only estimate based on model size
            param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            return {"memory_usage": param_size}
    
    def _measure_output_quality(self, model):
        """Measure output quality through perplexity and other metrics."""
        model.eval()
        
        perplexities = []
        quality_scores = []
        
        with torch.no_grad():
            for prompt in self.eval_prompts:
                # Calculate perplexity
                input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                perplexity = compute_perplexity(model, input_ids)
                perplexities.append(perplexity)
                
                # Generate text and measure quality
                output_ids = model.generate(
                    input_ids=input_ids,
                    max_length=input_ids.size(1) + 50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                quality = compute_output_quality(prompt, output_text)
                quality_scores.append(quality)
        
        # Calculate average metrics
        avg_perplexity = sum(perplexities) / len(perplexities)
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        return {
            "perplexity": avg_perplexity,
            "quality_score": avg_quality * 100,  # Scale to percentage
            "perplexities": perplexities,
            "quality_scores": quality_scores
        }
    
    def _measure_hardware_utilization(self, model):
        """Measure hardware utilization metrics like FLOPs."""
        # This is a basic implementation - can be expanded for more detailed profiling
        
        # Estimate FLOPs
        from thop import profile as thop_profile
        
        # Sample input for profiling
        prompt = self.eval_prompts[0]
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Measure FLOPS for forward pass
        try:
            macs, params = thop_profile(model, inputs=(input_ids,))
            flops = macs * 2  # FLOPs ≈ MACs * 2
        except Exception:
            # Fall back to a rough estimate based on common formulations
            n_layers = 12 if "gpt2" in self.model_name.lower() else 24
            hidden_size = 768 if "gpt2" in self.model_name.lower() else 1024
            seq_len = input_ids.size(1)
            
            # Very rough FLOP estimate
            flops = 6 * n_layers * hidden_size * hidden_size * seq_len
        
        return {"flops": flops, "parameters": sum(p.numel() for p in model.parameters())}
    
    def _run_comparison_experiments(self):
        """Run experiments with alternative pruning methods for comparison."""
        # Skip if we're already testing all strategies
        if all(s in self.strategy for s in ["entropy", "random", "magnitude"]):
            return {}
        
        comparison_results = {}
        alternative_strategies = ["entropy", "random", "magnitude"]
        
        # Only test strategies we're not already using
        test_strategies = [s for s in alternative_strategies if s != self.strategy]
        
        for alt_strategy in test_strategies:
            print(f"Testing alternative pruning strategy: {alt_strategy}")
            
            # Reset model
            self.model = load_adaptive_model(self.model_name, self.baseline_model, self.device)
            
            # Apply pruning with alternative strategy
            pruned_model, _, _ = self._apply_pruning(self.pruning_level)
            
            # Evaluate with alternative strategy
            metrics = self._evaluate_model(pruned_model, f"{alt_strategy.capitalize()} Pruning")
            
            # Add to results
            comparison_results[alt_strategy] = metrics
            self.results[f"comparison_{alt_strategy}"] = metrics
            
            # Compare with main strategy
            if self.baseline_comparison:
                alt_speedup = metrics["tokens_per_second"] / self.baseline_metrics["tokens_per_second"]
                alt_quality = metrics["quality_score"] / self.baseline_metrics["quality_score"]
                
                print(f"  {alt_strategy.capitalize()} Pruning:")
                print(f"    Speedup: {alt_speedup:.2f}x")
                print(f"    Quality: {alt_quality*100:.1f}%")
                
                # Compare with our main strategy
                main_speedup = self.results["comparison"]["speedup"]
                main_quality = self.results["comparison"]["quality_ratio"]
                
                print(f"  Compared to {self.strategy} strategy:")
                print(f"    Speed difference: {(alt_speedup/main_speedup - 1)*100:.1f}%")
                print(f"    Quality difference: {(alt_quality/main_quality - 1)*100:.1f}%")
        
        return comparison_results
    
    def _create_visualizations(self):
        """Create visualizations of benchmark results."""
        charts_dir = os.path.join(self.output_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # 1. Speed Comparison Chart
        plt.figure(figsize=(10, 6))
        
        # Collect speeds for comparison
        speeds = {
            "Baseline": self.baseline_metrics["tokens_per_second"]
        }
        
        speeds[f"{self.strategy.capitalize()} Pruning"] = self.pruned_metrics["tokens_per_second"]
        
        # Add alternative strategies if available
        for strategy in ["entropy", "random", "magnitude"]:
            key = f"comparison_{strategy}"
            if key in self.results:
                speeds[f"{strategy.capitalize()} Pruning"] = self.results[key]["tokens_per_second"]
        
        # Create bar chart
        bars = plt.bar(range(len(speeds)), list(speeds.values()), color='skyblue')
        plt.xticks(range(len(speeds)), list(speeds.keys()), rotation=45)
        plt.title(f'Generation Speed Comparison ({self.model_name}, {int(self.pruning_level*100)}% Pruning)')
        plt.ylabel('Tokens per Second')
        plt.grid(True, linestyle='--', axis='y', alpha=0.7)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f"{self.strategy}_speed_comparison.png"), dpi=150)
        plt.close()
        
        # 2. Quality-Speed Tradeoff Chart
        if self.baseline_comparison:
            plt.figure(figsize=(10, 6))
            
            # Collect data points
            labels = []
            speedups = []
            qualities = []
            
            # Baseline point
            labels.append("Baseline")
            speedups.append(1.0)
            qualities.append(100.0)
            
            # Main pruning strategy
            labels.append(f"{self.strategy.capitalize()} Pruning")
            speedups.append(self.results["comparison"]["speedup"])
            qualities.append(self.results["comparison"]["quality_ratio"] * 100)
            
            # Alternative strategies
            for strategy in ["entropy", "random", "magnitude"]:
                key = f"comparison_{strategy}"
                if key in self.results:
                    labels.append(f"{strategy.capitalize()} Pruning")
                    alt_speedup = self.results[key]["tokens_per_second"] / self.baseline_metrics["tokens_per_second"]
                    alt_quality = self.results[key]["quality_score"] / self.baseline_metrics["quality_score"] * 100
                    speedups.append(alt_speedup)
                    qualities.append(alt_quality)
            
            # Create scatter plot
            plt.figure(figsize=(10, 6))
            
            # Plot points
            for i, label in enumerate(labels):
                plt.scatter(speedups[i], qualities[i], s=100, label=label)
                plt.annotate(label, (speedups[i], qualities[i]), 
                            xytext=(5, 5), textcoords='offset points')
            
            plt.axhline(y=100, color='gray', linestyle='--', alpha=0.7)
            plt.axvline(x=1, color='gray', linestyle='--', alpha=0.7)
            
            plt.title(f'Quality-Speed Tradeoff ({self.model_name}, {int(self.pruning_level*100)}% Pruning)')
            plt.xlabel('Speedup Factor (×)')
            plt.ylabel('Quality Retention (%)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, f"{self.strategy}_quality_speed_tradeoff.png"), dpi=150)
            plt.close()
        
        # 3. Memory Usage Comparison
        if self.hardware_metrics:
            plt.figure(figsize=(10, 6))
            
            # Collect memory usage data
            memories = {
                "Baseline": self.baseline_metrics["memory_usage"]
            }
            
            memories[f"{self.strategy.capitalize()} Pruning"] = self.pruned_metrics["memory_usage"]
            
            # Add alternative strategies if available
            for strategy in ["entropy", "random", "magnitude"]:
                key = f"comparison_{strategy}"
                if key in self.results and "memory_usage" in self.results[key]:
                    memories[f"{strategy.capitalize()} Pruning"] = self.results[key]["memory_usage"]
            
            # Create bar chart
            bars = plt.bar(range(len(memories)), list(memories.values()), color='lightgreen')
            plt.xticks(range(len(memories)), list(memories.keys()), rotation=45)
            plt.title(f'Memory Usage Comparison ({self.model_name}, {int(self.pruning_level*100)}% Pruning)')
            plt.ylabel('Memory Usage (MB)')
            plt.grid(True, linestyle='--', axis='y', alpha=0.7)
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, f"{self.strategy}_memory_comparison.png"), dpi=150)
            plt.close()
        
        print(f"Visualizations saved to: {charts_dir}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run pure pruning benchmark")
    
    parser.add_argument("--model_name", type=str, default="gpt2", 
                        help="Name of the model to benchmark")
    parser.add_argument("--pruning_level", type=float, default=0.5, 
                        help="Level of pruning to apply (0.0-1.0)")
    parser.add_argument("--strategy", type=str, default="entropy", 
                        choices=["entropy", "random", "magnitude"],
                        help="Pruning strategy to use")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run benchmark on (defaults to CUDA if available)")
    parser.add_argument("--output_dir", type=str, default="results/pure_pruning",
                        help="Directory to save benchmark results")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations of benchmark results")
    parser.add_argument("--no_baseline", action="store_true",
                        help="Skip baseline comparison")
    parser.add_argument("--run_hardware_metrics", action="store_true",
                        help="Measure hardware-level metrics (FLOPs, memory)")
    
    return parser.parse_args()


def main():
    """Main function to run the benchmark."""
    args = parse_args()
    
    benchmark = PruningBenchmark(
        model_name=args.model_name,
        pruning_level=args.pruning_level,
        strategy=args.strategy,
        device=args.device,
        output_dir=args.output_dir,
        visualize=args.visualize,
        baseline_comparison=not args.no_baseline,
        hardware_metrics=args.run_hardware_metrics
    )
    
    # Run the benchmark
    results = benchmark.run()
    
    # Print summary
    print("\nBenchmark Summary:")
    print(f"Model: {args.model_name}")
    print(f"Pruning: {args.pruning_level*100:.1f}% using {args.strategy} strategy")
    
    if not args.no_baseline:
        print(f"Speedup: {results['comparison']['speedup']:.2f}x")
        print(f"Quality retention: {results['comparison']['quality_ratio']*100:.1f}%")
    
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()