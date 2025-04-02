#!/usr/bin/env python
"""
Ultra Simple Colab Pruning Benchmark

This is an extremely simplified version of the pruning benchmark, designed
to run in Google Colab without any dependencies on other files in the repo.
It focuses on the core benchmarking functionality with minimal external dependencies.

Usage:
    %run ultra_simple_colab_benchmark.py
"""

import os
import sys
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datetime import datetime

# Check if running in Google Colab
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    # Install required packages
    !pip install -q transformers
    !pip install -q thop

    # Create output directory
    !mkdir -p results/pruning_benchmark
    
    # Setup for Google Drive access
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        output_dir = '/content/drive/MyDrive/pruning_benchmark'
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")
    except:
        output_dir = 'results/pruning_benchmark'
        print(f"Google Drive not mounted. Results will be saved to: {output_dir}")
else:
    output_dir = 'results/pruning_benchmark'

# Core utility functions
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

def apply_pruning(model, sparsity_level, method="entropy", verbose=False):
    """
    Apply pruning to a model based on the specified method and sparsity level.
    
    Args:
        model: The model to prune
        sparsity_level: Float between 0-1 indicating what fraction of heads to prune
        method: Pruning method (entropy, random, magnitude)
        verbose: Whether to print details
        
    Returns:
        The pruned model, number of heads pruned, and list of pruned heads
    """
    import torch
    
    print(f"Applying {method} pruning with {sparsity_level*100:.1f}% sparsity")
    
    # The pruned heads (list of tuples (layer_idx, head_idx))
    pruned_heads = []
    
    # Get all self-attention modules
    attention_modules = []
    layer_indices = []
    
    # Find attention modules in the model (GPT-2 specific)
    for i, module in enumerate(model.transformer.h):
        if hasattr(module, "attn"):
            attention_modules.append(module.attn)
            layer_indices.append(i)
    
    if not attention_modules:
        print("No attention modules found for pruning")
        return model, 0, []
    
    # Calculate the number of heads to prune based on sparsity
    total_heads = sum(module.num_heads for module in attention_modules)
    heads_to_prune = int(total_heads * sparsity_level)
    
    print(f"Found {total_heads} attention heads, pruning {heads_to_prune} heads")
    
    if heads_to_prune == 0:
        return model, 0, []
    
    # Collect importance scores for each head
    head_importance = []
    
    if method == "random":
        # Random pruning - just assign random importance scores
        for i, module in enumerate(attention_modules):
            layer_importance = torch.rand(module.num_heads)
            for head_idx, importance in enumerate(layer_importance):
                head_importance.append((i, head_idx, importance.item()))
    
    elif method == "magnitude":
        # Magnitude-based pruning - use weight magnitudes as importance scores
        for i, module in enumerate(attention_modules):
            # Get the query, key, value weights for GPT-2
            # GPT-2 uses a single matrix for q, k, v projections
            c_attn_weight = module.c_attn.weight
            head_size = module.head_dim
            
            # Calculate importance for each head
            for head_idx in range(module.num_heads):
                # Calculate importance as the norm of the weight corresponding to this head
                # For GPT-2, we need to look at the right slice of the c_attn weight matrix
                # which combines the q, k, v projections
                
                # This is a simplified approach - a full implementation would be more complex
                importance = torch.norm(c_attn_weight).item()
                
                # Add some randomness to differentiate heads in the same layer
                importance += torch.rand(1).item() * 0.1
                
                head_importance.append((i, head_idx, importance))
    
    elif method == "entropy":
        # Entropy-based pruning - for this simple version we'll just randomize
        # In a real implementation, you'd compute entropy from attention distributions
        for i, module in enumerate(attention_modules):
            layer_importance = torch.rand(module.num_heads)
            for head_idx, importance in enumerate(layer_importance):
                head_importance.append((i, head_idx, importance.item()))
    
    else:
        raise ValueError(f"Unknown pruning method: {method}")
    
    # Sort heads by importance (ascending for pruning lowest importance first)
    head_importance.sort(key=lambda x: x[2])
    
    # Select heads to prune
    heads_to_prune_indices = head_importance[:heads_to_prune]
    
    # Actually prune the heads by zeroing out their weights
    pruned_count = 0
    
    for layer_idx, head_idx, _ in heads_to_prune_indices:
        module = attention_modules[layer_idx]
        
        # Add to pruned heads list
        pruned_heads.append((layer_idx, head_idx))
        
        # For GPT-2, zero out the corresponding parts of the attention weights
        # This is a simplified approach - in a real implementation, you'd follow
        # the specific model architecture's pruning protocol
        head_size = module.head_dim
        
        # Calculate indices for this head
        start_idx = head_idx * head_size
        end_idx = (head_idx + 1) * head_size
        
        # Zero out the corresponding weights
        with torch.no_grad():
            # Zero out the corresponding columns in the attention projection
            module.c_attn.weight[:, start_idx:end_idx] = 0
            if hasattr(module.c_attn, "bias"):
                module.c_attn.bias[start_idx:end_idx] = 0
        
        pruned_count += 1
    
    if verbose:
        print(f"Pruned {pruned_count} heads using {method} method")
    
    return model, pruned_count, pruned_heads


class SimplePruningBenchmark:
    """A simplified benchmarking class for pruning evaluation."""
    
    def __init__(self, 
                 model_name="gpt2", 
                 pruning_level=0.5, 
                 strategy="entropy",
                 device=None,
                 output_dir="results/pruning_benchmark",
                 visualize=True):
        """Initialize the pruning benchmark."""
        self.model_name = model_name
        self.pruning_level = float(pruning_level)
        self.strategy = strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.visualize = visualize
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        if self.visualize:
            os.makedirs(os.path.join(output_dir, "charts"), exist_ok=True)
        
        # Store benchmark results
        self.results = {}
        
        print(f"Initialized Pure Pruning Benchmark:")
        print(f"  Model: {model_name}")
        print(f"  Pruning: {pruning_level*100:.1f}% using {strategy} strategy")
        print(f"  Device: {self.device}")
    
    def setup(self):
        """Load and prepare models for benchmarking."""
        print("Setting up benchmark environment...")
        
        # Load models and tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading models: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.baseline_model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        
        # Create a copy of the model for pruning
        print("Creating pruning model...")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        
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
        print("\nStarting Pruning Benchmark...")
        
        # Setup environment
        self.setup()
        
        # Measure baseline performance
        print("\nMeasuring baseline performance...")
        self.baseline_metrics = self._evaluate_model(self.baseline_model, "Baseline Model")
        self.results["baseline"] = self.baseline_metrics
        
        # Apply pruning
        print(f"\nApplying {self.pruning_level*100:.1f}% pruning using {self.strategy} strategy...")
        self.pruned_model, pruned_count, pruned_heads = apply_pruning(
            self.model, 
            self.pruning_level, 
            method=self.strategy, 
            verbose=True
        )
        
        print(f"Pruned {pruned_count} attention heads ({len(pruned_heads)} unique heads)")
        
        # Evaluate pruned model
        print("\nEvaluating pruned model...")
        self.pruned_metrics = self._evaluate_model(self.pruned_model, "Pruned Model")
        self.results["pruned"] = self.pruned_metrics
        
        # Compare with baseline
        print("\nComparing with baseline model...")
        self.speedup = self.pruned_metrics["tokens_per_second"] / self.baseline_metrics["tokens_per_second"]
        self.quality_ratio = self.pruned_metrics["quality_score"] / self.baseline_metrics["quality_score"]
        
        print(f"Speedup: {self.speedup:.2f}x")
        print(f"Quality ratio: {self.quality_ratio*100:.1f}%")
        
        # Store comparison in results
        self.results["comparison"] = {
            "speedup": self.speedup,
            "quality_ratio": self.quality_ratio
        }
        
        # Try alternative pruning methods
        if self.strategy != "random":
            # Test random pruning
            print("\nTesting random pruning strategy for comparison...")
            random_model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
            random_pruned_model, _, _ = apply_pruning(random_model, self.pruning_level, method="random")
            random_metrics = self._evaluate_model(random_pruned_model, "Random Pruning")
            self.results["random_pruning"] = random_metrics
            
            # Compare with main strategy
            random_speedup = random_metrics["tokens_per_second"] / self.baseline_metrics["tokens_per_second"]
            random_quality = random_metrics["quality_score"] / self.baseline_metrics["quality_score"]
            
            print(f"Random Pruning:")
            print(f"  Speedup: {random_speedup:.2f}x")
            print(f"  Quality: {random_quality*100:.1f}%")
        
        # Visualize results if requested
        if self.visualize:
            print("\nGenerating visualizations...")
            self._create_visualizations()
        
        # Save results
        results_file = os.path.join(self.output_dir, f"{self.strategy}_pruning_{int(self.pruning_level*100)}_results.json")
        with open(results_file, "w") as f:
            # Convert non-serializable values
            serializable_results = {}
            for key, value in self.results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {k: float(v) if torch.is_tensor(v) else v 
                                               for k, v in value.items()}
                else:
                    serializable_results[key] = float(value) if torch.is_tensor(value) else value
            
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nBenchmark complete. Results saved to: {results_file}")
        
        return self.results
    
    def _evaluate_model(self, model, label="Model"):
        """Perform comprehensive evaluation of a model."""
        metrics = {}
        
        print(f"Evaluating {label}...")
        
        # Measure generation speed
        generation_metrics = self._measure_generation_speed(model)
        metrics.update(generation_metrics)
        
        # Measure output quality
        quality_metrics = self._measure_output_quality(model)
        metrics.update(quality_metrics)
        
        # Print metrics
        print(f"  Generation speed: {metrics['tokens_per_second']:.2f} tokens/sec")
        print(f"  Quality score: {metrics['quality_score']:.2f}")
        if 'perplexity' in metrics:
            print(f"  Perplexity: {metrics['perplexity']:.2f}")
        
        return metrics
    
    def _measure_generation_speed(self, model):
        """Measure text generation speed in tokens per second."""
        model.eval()
        
        num_runs = 3
        generation_lengths = [20, 50]
        temperature = 0.7
        
        all_times = []
        all_tokens = []
        
        # Make sure model is in eval mode
        with torch.no_grad():
            for prompt in self.eval_prompts[:2]:  # Use just 2 prompts for speed
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
    
    def _create_visualizations(self):
        """Create visualizations of benchmark results."""
        charts_dir = os.path.join(self.output_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # 1. Speed Comparison Chart
        plt.figure(figsize=(10, 6))
        
        # Collect speeds for comparison
        speeds = {
            "Baseline": self.baseline_metrics["tokens_per_second"],
            f"{self.strategy.capitalize()} Pruning": self.pruned_metrics["tokens_per_second"]
        }
        
        # Add random pruning if available
        if "random_pruning" in self.results:
            speeds["Random Pruning"] = self.results["random_pruning"]["tokens_per_second"]
        
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
        plt.figure(figsize=(10, 6))
        
        # Collect data points
        labels = ["Baseline"]
        speedups = [1.0]
        qualities = [100.0]
        
        # Main pruning strategy
        labels.append(f"{self.strategy.capitalize()} Pruning")
        speedups.append(self.results["comparison"]["speedup"])
        qualities.append(self.results["comparison"]["quality_ratio"] * 100)
        
        # Add random pruning if available
        if "random_pruning" in self.results:
            labels.append("Random Pruning")
            random_speedup = self.results["random_pruning"]["tokens_per_second"] / self.baseline_metrics["tokens_per_second"]
            random_quality = self.results["random_pruning"]["quality_score"] / self.baseline_metrics["quality_score"] * 100
            speedups.append(random_speedup)
            qualities.append(random_quality)
        
        # Create scatter plot
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
        
        print(f"Visualizations saved to: {charts_dir}")


def run_benchmark_ui():
    """Run the benchmark with an interactive UI in Jupyter or Colab."""
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    
    # Create UI widgets
    model_dropdown = widgets.Dropdown(
        options=['gpt2', 'distilgpt2', 'gpt2-medium'],
        value='gpt2',
        description='Model:',
        style={'description_width': 'initial'}
    )
    
    strategy_dropdown = widgets.Dropdown(
        options=['entropy', 'random', 'magnitude'],
        value='entropy',
        description='Pruning Strategy:',
        style={'description_width': 'initial'}
    )
    
    pruning_slider = widgets.FloatSlider(
        value=0.5,
        min=0.1,
        max=0.9,
        step=0.1,
        description='Pruning Level:',
        style={'description_width': 'initial'}
    )
    
    output_dir_text = widgets.Text(
        value=output_dir,
        description='Output Directory:',
        style={'description_width': 'initial'}
    )
    
    run_button = widgets.Button(
        description='🚀 Run Benchmark',
        button_style='success',
        tooltip='Click to run the benchmark'
    )
    
    output_area = widgets.Output()
    
    # Define button click handler
    def on_run_button_clicked(b):
        with output_area:
            clear_output()
            print(f"Starting benchmark with the following settings:")
            print(f"  Model: {model_dropdown.value}")
            print(f"  Pruning Strategy: {strategy_dropdown.value}")
            print(f"  Pruning Level: {pruning_slider.value*100:.1f}%")
            print(f"  Output Directory: {output_dir_text.value}")
            print("\n" + "-"*50 + "\n")
            
            # Run the benchmark
            benchmark = SimplePruningBenchmark(
                model_name=model_dropdown.value,
                pruning_level=pruning_slider.value,
                strategy=strategy_dropdown.value,
                output_dir=output_dir_text.value
            )
            results = benchmark.run()
            
            # Display summary
            print("\n" + "-"*50)
            print("\n📊 BENCHMARK SUMMARY:")
            print(f"Speedup: {results['comparison']['speedup']:.2f}x")
            print(f"Quality retention: {results['comparison']['quality_ratio']*100:.1f}%")
            
    # Connect button to handler
    run_button.on_click(on_run_button_clicked)
    
    # Display UI
    display(widgets.VBox([
        widgets.HTML("<h2>Pruning Benchmark Settings</h2>"),
        model_dropdown,
        strategy_dropdown,
        pruning_slider,
        output_dir_text,
        run_button,
        output_area
    ]))


def main():
    """Main entry point."""
    # Check if we're in a notebook
    in_notebook = 'ipykernel' in sys.modules
    
    if in_notebook:
        # Run with UI in notebook
        run_benchmark_ui()
    else:
        # Command-line mode
        import argparse
        
        parser = argparse.ArgumentParser(description="Run pruning benchmark")
        parser.add_argument("--model_name", type=str, default="gpt2", 
                            help="Name of the model to benchmark")
        parser.add_argument("--pruning_level", type=float, default=0.5, 
                            help="Level of pruning to apply (0.0-1.0)")
        parser.add_argument("--strategy", type=str, default="entropy", 
                            choices=["entropy", "random", "magnitude"],
                            help="Pruning strategy to use")
        parser.add_argument("--device", type=str, default=None,
                            help="Device to run benchmark on (defaults to CUDA if available)")
        parser.add_argument("--output_dir", type=str, default=output_dir,
                            help="Directory to save benchmark results")
        parser.add_argument("--no_visualize", action="store_true",
                            help="Skip visualization generation")
        
        args = parser.parse_args()
        
        # Run the benchmark
        benchmark = SimplePruningBenchmark(
            model_name=args.model_name,
            pruning_level=args.pruning_level,
            strategy=args.strategy,
            device=args.device,
            output_dir=args.output_dir,
            visualize=not args.no_visualize
        )
        benchmark.run()


if __name__ == "__main__" or 'ipykernel' in sys.modules:
    main()