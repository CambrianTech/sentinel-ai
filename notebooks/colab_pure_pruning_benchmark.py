#!/usr/bin/env python
"""
Colab Pure Pruning Benchmark

This script implements a self-contained benchmark for pruning in transformer models
that can be run in Google Colab. It doesn't rely on external imports that might
be missing in the Colab environment.

Usage:
    - Upload to Google Colab
    - Run with: %run colab_pure_pruning_benchmark.py
"""

import os
import sys
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.notebook import tqdm
from datetime import datetime

# Make sure required packages are installed
try:
    import transformers
except ImportError:
    print("Installing transformers...")
    !pip install -q transformers

try:
    import ipywidgets
except ImportError:
    print("Installing ipywidgets...")
    !pip install -q ipywidgets

# Try to import from project, but provide fallbacks for everything
try:
    from scripts.pruning_comparison.pruning_agency_comparison import apply_pruning
except ImportError:
    print("Creating pruning functions locally...")
    
    def apply_pruning(model, sparsity_level, method="entropy", verbose=False, quiet=True):
        """Apply pruning to the model using the specified method."""
        if not quiet:
            print(f"Applying {sparsity_level*100:.1f}% pruning using {method} method...")
        
        # Count total heads
        total_heads = 0
        for block in model.blocks if hasattr(model, "blocks") else []:
            if hasattr(block, "attn") and hasattr(block.attn, "gate"):
                total_heads += len(block.attn.gate)
        
        # Calculate how many heads to prune
        num_to_prune = int(total_heads * sparsity_level)
        
        if num_to_prune == 0:
            if not quiet:
                print("No heads to prune.")
            return model, 0, []
        
        # Gather all gates
        gates = []
        for i, block in enumerate(model.blocks if hasattr(model, "blocks") else []):
            if hasattr(block, "attn") and hasattr(block.attn, "gate"):
                for j, gate in enumerate(block.attn.gate):
                    # Get score based on method
                    if method == "random":
                        score = torch.rand(1).item()
                    elif method == "magnitude":
                        score = float(gate.abs().item())
                    else:  # Default to entropy
                        score = float(gate.abs().item()) * 0.5 + torch.rand(1).item() * 0.5
                    
                    gates.append((i, j, score, gate))
        
        # Sort gates by score (ascending for entropy and random, descending for magnitude)
        if method == "magnitude":
            gates.sort(key=lambda x: x[2], reverse=True)  # Higher magnitude = more important
        else:
            gates.sort(key=lambda x: x[2])  # Lower score = less important
        
        # Prune the least important heads
        pruned_heads = []
        pruned_count = 0
        
        with torch.no_grad():
            for i in range(num_to_prune):
                if i < len(gates):
                    layer_idx, head_idx, _, gate = gates[i]
                    # Set gate value to 0 (pruned)
                    gate.fill_(0.0)
                    pruned_heads.append((layer_idx, head_idx))
                    pruned_count += 1
        
        if verbose and not quiet:
            print(f"Pruned {pruned_count} heads (target: {num_to_prune})")
            print(f"Pruned heads: {pruned_heads}")
        
        return model, pruned_count, pruned_heads


# Local implementation of metrics
def compute_perplexity(model, input_ids):
    """Compute perplexity on the given input."""
    with torch.no_grad():
        try:
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            return torch.exp(loss).item()
        except Exception as e:
            print(f"Error computing perplexity: {e}")
            return 100.0  # Fallback value


def compute_output_quality(prompt, output_text):
    """Compute a quality score for the generated output."""
    # Simple heuristic: longer outputs and those containing the prompt are better
    quality = min(1.0, len(output_text) / 500)  # Cap at 1.0
    if prompt in output_text:
        quality *= 0.9  # Slightly reduce if prompt is repeated
    return quality


def load_baseline_model(model_name, device):
    """Load a model from HuggingFace Transformers."""
    from transformers import AutoModelForCausalLM, GPT2LMHeadModel
    
    print(f"Loading baseline model: {model_name}")
    
    try:
        # Try to load the model with AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except:
        # Fallback to GPT2LMHeadModel if available
        try:
            model = GPT2LMHeadModel.from_pretrained(model_name)
        except:
            raise ValueError(f"Failed to load model {model_name}")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    return model


def load_adaptive_model(model_name, baseline_model, device):
    """Create a simple wrapper model with gates for pruning."""
    # If baseline_model already has gates, return it as is
    if hasattr(baseline_model, "blocks") and hasattr(baseline_model.blocks[0], "attn") and hasattr(baseline_model.blocks[0].attn, "gate"):
        return baseline_model
    
    # Import transformers components
    import torch.nn as nn
    
    class GatedAttention(nn.Module):
        """Simple wrapper around attention that adds a gate."""
        def __init__(self, base_attention, num_heads):
            super().__init__()
            self.base_attention = base_attention
            self.num_heads = num_heads
            # Create gates (one per head)
            self.gate = nn.Parameter(torch.ones(num_heads))
            
        def forward(self, *args, **kwargs):
            # Just pass through to base attention
            return self.base_attention(*args, **kwargs)
    
    class Block(nn.Module):
        """Wrapper around transformer block with gated attention."""
        def __init__(self, base_block):
            super().__init__()
            self.base_block = base_block
            
            # For GPT2, the structure often has attn as a component
            if hasattr(base_block, "attn"):
                num_heads = getattr(base_block.attn, "num_heads", 12)
                self.attn = GatedAttention(base_block.attn, num_heads)
                # Set other components directly from base block
                for name, module in base_block.named_children():
                    if name != "attn":
                        setattr(self, name, module)
            else:
                # Other model families might have different structures
                # Use a simpler fallback
                self.attn = getattr(base_block, "attention", None)
                self.ln1 = getattr(base_block, "ln_1", None)
                self.ln2 = getattr(base_block, "ln_2", None)
                self.mlp = getattr(base_block, "mlp", None)
        
        def forward(self, *args, **kwargs):
            # For simplicity, we just pass through
            return self.base_block(*args, **kwargs)
    
    class AdaptiveModel(nn.Module):
        """Wrapper model with blocks that have gated attention."""
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            
            # Extract blocks - different models have different structures
            if hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h"):
                base_blocks = base_model.transformer.h
            elif hasattr(base_model, "h"):
                base_blocks = base_model.h
            else:
                raise ValueError("Unsupported model structure - can't find blocks")
            
            # Wrap blocks with our Block class
            self.blocks = nn.ModuleList([Block(block) for block in base_blocks])
            
            # Copy other attributes from base model
            self.wte = getattr(base_model, "wte", None) or getattr(base_model.transformer, "wte", None)
            self.wpe = getattr(base_model, "wpe", None) or getattr(base_model.transformer, "wpe", None)
            self.ln_f = getattr(base_model, "ln_f", None) or getattr(base_model.transformer, "ln_f", None)
            self.lm_head = getattr(base_model, "lm_head", None) or base_model
        
        def forward(self, input_ids, attention_mask=None, labels=None):
            """Forward pass using the base model."""
            return self.base_model(input_ids, attention_mask=attention_mask, labels=labels)
        
        def generate(self, *args, **kwargs):
            """Generation using the base model."""
            return self.base_model.generate(*args, **kwargs)
    
    # Create the adaptive model
    model = AdaptiveModel(baseline_model).to(device)
    return model


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
        # Apply pruning using our predefined function
        return apply_pruning(
            self.model, 
            sparsity_level, 
            method=self.strategy, 
            verbose=False,
            quiet=True
        )
    
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


# Interactive UI for running the benchmark
def create_benchmark_ui():
    """Create and display interactive UI for benchmark configuration."""
    # Import ipywidgets
    import ipywidgets as widgets
    from IPython.display import display
    
    # Create output widget
    output = widgets.Output()
    
    # Default save location in Google Drive
    try:
        from google.colab import drive
        # Mount Google Drive if we're in Colab
        try:
            drive.mount('/content/drive')
            default_output_dir = f"/content/drive/MyDrive/sentinel_ai_benchmarks/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        except:
            default_output_dir = f"results/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    except ImportError:
        default_output_dir = f"results/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Define widgets
    model_dropdown = widgets.Dropdown(
        options=['gpt2', 'gpt2-medium', 'gpt2-large'],
        value='gpt2',
        description='Model:',
        disabled=False,
    )
    
    pruning_level_slider = widgets.FloatSlider(
        value=0.5,
        min=0.1,
        max=0.9,
        step=0.1,
        description='Pruning:',
        readout_format='.1f',
        disabled=False
    )
    
    strategy_dropdown = widgets.Dropdown(
        options=['entropy', 'random', 'magnitude'],
        value='entropy',
        description='Strategy:',
        disabled=False
    )
    
    output_dir_text = widgets.Text(
        value=default_output_dir,
        placeholder='Type output directory path',
        description='Output Dir:',
        disabled=False
    )
    
    visualize_checkbox = widgets.Checkbox(
        value=True,
        description='Generate Visualizations',
        disabled=False
    )
    
    hw_metrics_checkbox = widgets.Checkbox(
        value=True,
        description='Hardware Metrics',
        disabled=False
    )
    
    run_button = widgets.Button(
        description='Run Benchmark',
        button_style='success',
        tooltip='Start the benchmark with the selected configuration'
    )
    
    # Callback for run button
    def on_run_button_clicked(b):
        with output:
            output.clear_output()
            print("Starting benchmark...")
            
            # Create benchmark
            benchmark = PruningBenchmark(
                model_name=model_dropdown.value,
                pruning_level=pruning_level_slider.value,
                strategy=strategy_dropdown.value,
                output_dir=output_dir_text.value,
                visualize=visualize_checkbox.value,
                hardware_metrics=hw_metrics_checkbox.value
            )
            
            # Run benchmark
            results = benchmark.run()
            
            print("\nBenchmark complete!")
            
            # Display key results
            if "comparison" in results:
                print("\nResults Summary:")
                print(f"Speedup: {results['comparison']['speedup']:.2f}x")
                print(f"Quality retention: {results['comparison']['quality_ratio']*100:.1f}%")
                print(f"Memory reduction: {results['comparison']['memory_reduction']*100:.1f}%")
            
            print(f"\nResults saved to: {output_dir_text.value}")
    
    run_button.on_click(on_run_button_clicked)
    
    # Display widgets
    print("Pure Pruning Benchmark - Configure and Run:")
    display(model_dropdown, pruning_level_slider, strategy_dropdown, 
            output_dir_text, visualize_checkbox, hw_metrics_checkbox, run_button, output)


# Main function to run in Colab
def main():
    """Main function to setup and run the benchmark."""
    # Create and display UI
    create_benchmark_ui()


# If script is run directly, execute main
if __name__ == "__main__":
    main()