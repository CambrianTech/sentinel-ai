#!/usr/bin/env python
"""
Benchmark the effect of different pruning strategies and levels.

This script specializes in measuring the impact of different pruning levels
and strategies on model performance. It creates a detailed analysis of:
1. Perplexity vs. pruning level
2. Inference speed vs. pruning level
3. Memory savings vs. pruning level
4. Generation quality vs. pruning level
5. Head importance metrics and correlations

Usage:
    python scripts/benchmark_pruning.py --model gpt2 \
                                       --pruning_levels 0.0 0.1 0.3 0.5 0.7 \
                                       --strategies entropy gradient random \
                                       --dataset wikitext

Features:
- Detailed pruning analysis across multiple metrics
- Support for different pruning strategies
- Analysis of model behavior at various pruning levels
- Visualization of results with detailed plots
"""

import os
import sys
import argparse
import torch
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from torch.cuda.amp import autocast
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.loaders.loader import load_baseline_model, load_adaptive_model
from datasets.dataset_loader import load_dataset
from utils.metrics import compute_perplexity
from utils.generation_wrapper import generate_text
from controller.controller_manager import ControllerManager
from controller.metrics.head_metrics import collect_head_metrics


class PruningBenchmark:
    """
    Comprehensive benchmark for different pruning strategies and levels.
    """
    
    def __init__(self, args):
        """Initialize the pruning benchmark with command line arguments."""
        self.args = args
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seed for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(
            args.output_dir, f"pruning_benchmark_{args.model.split('/')[-1]}_{timestamp}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Dictionary to store results
        self.results = {
            "model": args.model,
            "pruning_levels": args.pruning_levels,
            "strategies": args.strategies,
            "metrics": {}
        }
        
        print(f"🚀 Initializing pruning benchmark on device: {self.device}")
        print(f"📊 Testing pruning levels: {args.pruning_levels}")
        print(f"🔄 Using strategies: {args.strategies}")
    
    def run(self):
        """Run the complete benchmark across all configurations."""
        # Load tokenizer and model
        print(f"\n⚙️ Loading tokenizer and model: {self.args.model}")
        tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load dataset
        print(f"\n📚 Loading dataset: {self.args.dataset}")
        train_dataset, eval_dataset = load_dataset(
            self.args.dataset, tokenizer, self.args.max_length
        )
        
        # Create batch for metrics collection
        print("\n🔍 Creating sample batch for metrics collection")
        dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True
        )
        sample_batch = next(iter(dataloader))
        sample_batch = {k: v.to(self.device) for k, v in sample_batch.items()}
        
        # Create evaluation dataloader
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=self.args.batch_size
        )
        
        # Create prompts for generation tests
        print("\n✏️ Preparing prompts for generation tests")
        prompts = [
            "Once upon a time in a land far away,",
            "The future of artificial intelligence depends on",
            "Scientists have recently discovered that",
            "The most important lesson I've learned is",
            "When considering the environmental impact,",
            "In the history of technological innovation,"
        ]
        
        # Benchmark each strategy
        for strategy in self.args.strategies:
            print(f"\n\n🔬 Testing pruning strategy: {strategy}")
            self.results["metrics"][strategy] = {}
            
            # Load fresh model for each strategy
            baseline_model = load_baseline_model(self.args.model, self.device)
            model = load_adaptive_model(self.args.model, baseline_model, self.device)
            
            # Calculate head importance metrics once
            print(f"\n📊 Calculating head importance metrics...")
            metrics = self._calculate_head_importance(model, sample_batch)
            
            # Benchmark each pruning level
            for pruning_level in self.args.pruning_levels:
                print(f"\n📏 Testing pruning level: {pruning_level}")
                
                # Create a fresh copy of the model for this pruning level
                model_copy = load_adaptive_model(self.args.model, baseline_model, self.device)
                
                # Apply pruning
                pruned_model = self._apply_pruning(model_copy, pruning_level, strategy, metrics)
                
                # Run benchmarks
                results = self._benchmark_model(
                    pruned_model, tokenizer, eval_loader, prompts, pruning_level
                )
                
                # Store results
                self.results["metrics"][strategy][str(pruning_level)] = results
                
                # Clear memory
                del model_copy
                del pruned_model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Clean up
            del baseline_model
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Save results and generate plots
        self._save_results()
        self._generate_plots()
        
        print(f"\n✅ Benchmark complete. Results saved to: {self.output_dir}")
    
    def _calculate_head_importance(self, model, sample_batch):
        """Calculate importance metrics for each attention head."""
        model.eval()
        num_layers = len(model.blocks)
        num_heads = model.blocks[0]["attn"].num_heads
        
        print("   Collecting head metrics...")
        metrics = collect_head_metrics(model, batch=sample_batch)
        
        # Store head metrics for reference
        metrics_file = os.path.join(self.output_dir, "head_metrics.json")
        
        # Convert tensor values to Python floats for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                serializable_metrics[key] = value.cpu().numpy().tolist()
            else:
                serializable_metrics[key] = value
                
        with open(metrics_file, "w") as f:
            json.dump(serializable_metrics, f, indent=2)
        
        # For entropy and gradient metrics, ensure proper shape
        entropy = metrics.get("entropy", torch.zeros((num_layers, num_heads)))
        if not isinstance(entropy, torch.Tensor):
            entropy = torch.tensor(entropy)
        if len(entropy.shape) < 2:
            entropy = entropy.reshape(num_layers, num_heads)
            
        grad_norm = metrics.get("grad_norm", torch.zeros((num_layers, num_heads)))
        if not isinstance(grad_norm, torch.Tensor):
            grad_norm = torch.tensor(grad_norm)
        if len(grad_norm.shape) < 2:
            grad_norm = grad_norm.reshape(num_layers, num_heads)
        
        metrics["entropy"] = entropy.to(self.device)
        metrics["grad_norm"] = grad_norm.to(self.device)
        
        return metrics
    
    def _apply_pruning(self, model, pruning_level, strategy, metrics):
        """Apply pruning to the model based on the specified strategy and level."""
        print(f"   Applying {strategy} pruning at level {pruning_level}...")
        
        if pruning_level <= 0.0:
            return model  # No pruning needed
        
        num_layers = len(model.blocks)
        num_heads = model.blocks[0]["attn"].num_heads
        total_heads = num_layers * num_heads
        heads_to_prune = int(total_heads * pruning_level)
        
        if heads_to_prune <= 0:
            return model
        
        # Determine which heads to prune based on strategy
        head_scores = torch.zeros((num_layers, num_heads)).to(self.device)
        
        if strategy == "random":
            # Random strategy: assign random scores
            head_scores = torch.rand((num_layers, num_heads)).to(self.device)
            
        elif strategy == "entropy":
            # Entropy strategy: higher entropy = less useful head
            if "entropy" in metrics:
                head_scores = metrics["entropy"]
            else:
                print("   ⚠️ Warning: Entropy metrics not available. Using random strategy.")
                head_scores = torch.rand((num_layers, num_heads)).to(self.device)
                
        elif strategy == "gradient":
            # Gradient strategy: lower gradient norm = less important head
            if "grad_norm" in metrics:
                # Invert so smaller gradients get higher scores (more likely to be pruned)
                head_scores = -metrics["grad_norm"]
            else:
                print("   ⚠️ Warning: Gradient norm metrics not available. Using random strategy.")
                head_scores = torch.rand((num_layers, num_heads)).to(self.device)
                
        elif strategy == "attention_mass":
            # Attention mass strategy: less focused attention pattern = less useful head
            if "attention_mass" in metrics:
                head_scores = metrics["attention_mass"]
            else:
                print("   ⚠️ Warning: Attention mass metrics not available. Using random strategy.")
                head_scores = torch.rand((num_layers, num_heads)).to(self.device)
                
        elif strategy == "combined":
            # Combined strategy: weighted combination of metrics
            if "entropy" in metrics and "grad_norm" in metrics:
                # Normalize each metric to [0, 1] range
                entropy = metrics["entropy"]
                entropy_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)
                
                grad_norm = metrics["grad_norm"]
                grad_norm_norm = (grad_norm - grad_norm.min()) / (grad_norm.max() - grad_norm.min() + 1e-8)
                
                # Higher entropy and lower gradient norm means less important
                head_scores = 0.7 * entropy_norm - 0.3 * grad_norm_norm
            else:
                print("   ⚠️ Warning: Required metrics not available for combined strategy. Using random.")
                head_scores = torch.rand((num_layers, num_heads)).to(self.device)
        
        # Flatten scores and find threshold
        flat_scores = head_scores.view(-1)
        _, indices = torch.sort(flat_scores, descending=True)
        threshold_idx = min(heads_to_prune, len(indices) - 1)
        threshold_value = flat_scores[indices[threshold_idx]]
        
        # Apply pruning
        with torch.no_grad():
            heads_pruned = 0
            pruned_mask = torch.zeros((num_layers, num_heads)).to(self.device)
            
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    if head_scores[layer_idx, head_idx] >= threshold_value and heads_pruned < heads_to_prune:
                        # Prune this head (set gate to near-zero)
                        model.blocks[layer_idx]["attn"].gate[head_idx] = torch.tensor(0.001, device=self.device)
                        pruned_mask[layer_idx, head_idx] = 1
                        heads_pruned += 1
                    else:
                        # Keep this head active
                        model.blocks[layer_idx]["attn"].gate[head_idx] = torch.tensor(0.999, device=self.device)
            
            # Save pruning mask
            pruning_dir = os.path.join(self.output_dir, f"{strategy}_pruning")
            os.makedirs(pruning_dir, exist_ok=True)
            np.save(
                os.path.join(pruning_dir, f"pruned_heads_level{pruning_level:.1f}.npy"),
                pruned_mask.cpu().numpy()
            )
            
            print(f"   ✂️ Pruned {heads_pruned} of {total_heads} attention heads ({pruning_level*100:.1f}%)")
        
        return model
    
    def _benchmark_model(self, model, tokenizer, eval_loader, prompts, pruning_level):
        """Run comprehensive benchmarks on the pruned model."""
        results = {
            "pruning_level": pruning_level,
            "active_heads": 0,
            "perplexity": 0,
            "inference_speed": {},
            "memory_usage": {},
            "generation_quality": {}
        }
        
        # Count active heads
        total_heads = 0
        active_heads = 0
        for layer_idx in range(len(model.blocks)):
            for head_idx in range(model.blocks[0]["attn"].num_heads):
                total_heads += 1
                if model.blocks[layer_idx]["attn"].gate[head_idx].item() > 0.1:
                    active_heads += 1
        
        results["active_heads"] = active_heads
        results["total_heads"] = total_heads
        print(f"   📊 Active heads: {active_heads}/{total_heads}")
        
        # Measure perplexity
        if self.args.test_perplexity:
            print("   📏 Measuring perplexity...")
            try:
                perplexity = self._measure_perplexity(model, eval_loader)
                results["perplexity"] = perplexity
                print(f"   📊 Perplexity: {perplexity:.2f}")
            except Exception as e:
                print(f"   ❌ Error measuring perplexity: {e}")
                results["perplexity"] = float('nan')
        
        # Measure inference speed
        if self.args.test_speed:
            print("   ⏱️ Measuring inference speed...")
            try:
                speed_results = self._measure_inference_speed(model, tokenizer, prompts)
                results["inference_speed"] = speed_results
                # Print a sample result
                for batch_size in speed_results:
                    print(f"   🚀 Speed (batch {batch_size}): {speed_results[batch_size]:.2f} tokens/sec")
                    break
            except Exception as e:
                print(f"   ❌ Error measuring inference speed: {e}")
                results["inference_speed"] = {}
        
        # Measure memory usage
        if self.args.test_memory and torch.cuda.is_available() and self.device.type == "cuda":
            print("   🧠 Measuring memory usage...")
            try:
                memory_results = self._measure_memory_usage(model, tokenizer, prompts)
                results["memory_usage"] = memory_results
                # Print a sample result
                for batch_size in memory_results:
                    print(f"   📊 Peak memory (batch {batch_size}): {memory_results[batch_size]:.2f} MB")
                    break
            except Exception as e:
                print(f"   ❌ Error measuring memory usage: {e}")
                results["memory_usage"] = {}
        
        # Measure text generation quality
        if self.args.test_generation:
            print("   📝 Measuring text generation quality...")
            try:
                quality_results = self._measure_text_quality(model, tokenizer, prompts)
                results["generation_quality"] = quality_results
                print(f"   📊 Repetition penalty: {quality_results.get('repetition_score', 'N/A'):.3f}")
                print(f"   📊 Lexical diversity: {quality_results.get('lexical_diversity', 'N/A'):.3f}")
            except Exception as e:
                print(f"   ❌ Error measuring text quality: {e}")
                results["generation_quality"] = {}
        
        return results
    
    def _measure_perplexity(self, model, eval_loader):
        """Measure perplexity of the model on the evaluation dataset."""
        return compute_perplexity(model, eval_loader, self.device)
    
    def _measure_inference_speed(self, model, tokenizer, prompts):
        """Measure inference speed across different batch sizes."""
        model.eval()
        results = {}
        
        batch_sizes = self.args.batch_sizes
        max_length = self.args.generation_length
        num_runs = 3  # Number of runs to average over
        
        for batch_size in batch_sizes:
            # Skip if batch size is larger than available prompts
            if batch_size > len(prompts):
                continue
                
            # Prepare input batch
            batch_prompts = prompts[:batch_size]
            
            # Measure time
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            batch_times = []
            
            for _ in range(num_runs):
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.device)
                
                # Warmup run
                with torch.no_grad():
                    _ = model.generate(
                        **inputs,
                        max_length=max_length,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7
                    )
                
                # Timed run
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                with torch.no_grad():
                    _ = model.generate(
                        **inputs,
                        max_length=max_length,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7
                    )
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                batch_times.append(end_time - start_time)
            
            # Calculate tokens per second
            avg_time = np.mean(batch_times)
            tokens_generated = batch_size * max_length  # Approximation
            tokens_per_second = tokens_generated / avg_time
            
            results[batch_size] = tokens_per_second
        
        return results
    
    def _measure_memory_usage(self, model, tokenizer, prompts):
        """Measure memory usage during inference."""
        model.eval()
        results = {}
        
        batch_sizes = self.args.batch_sizes
        max_length = self.args.generation_length
        
        for batch_size in batch_sizes:
            # Skip if batch size is larger than available prompts
            if batch_size > len(prompts):
                continue
                
            # Prepare input batch
            batch_prompts = prompts[:batch_size]
            
            # Clear cache and reset stats
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Run generation
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7
                )
            
            # Record memory
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            results[batch_size] = peak_memory_mb
        
        return results
    
    def _measure_text_quality(self, model, tokenizer, prompts):
        """Measure text generation quality metrics."""
        model.eval()
        results = {}
        
        generated_texts = []
        for prompt in prompts:
            try:
                output = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_length=self.args.generation_length,
                    temperature=0.7,
                    top_p=0.9,
                    device=self.device
                )
                generated_texts.append(output)
            except Exception as e:
                print(f"      ⚠️ Error generating text: {e}")
        
        # Skip quality metrics if no text was generated
        if not generated_texts:
            return results
        
        # Calculate quality metrics
        # 1. Repetition score (lower is better)
        repetition_scores = []
        for text in generated_texts:
            words = text.split()
            if len(words) <= 1:
                continue
                
            # Count word repetitions within a window
            window_size = min(50, len(words))
            repeats = 0
            for i in range(len(words) - 1):
                end_idx = min(i + window_size, len(words))
                if words[i] in words[i+1:end_idx]:
                    repeats += 1
            
            repetition_score = repeats / (len(words) - 1) if len(words) > 1 else 0
            repetition_scores.append(repetition_score)
        
        if repetition_scores:
            results["repetition_score"] = np.mean(repetition_scores)
        
        # 2. Lexical diversity (higher is better)
        diversity_scores = []
        for text in generated_texts:
            words = text.split()
            if not words:
                continue
                
            unique_words = len(set(words))
            total_words = len(words)
            
            diversity = unique_words / total_words if total_words > 0 else 0
            diversity_scores.append(diversity)
        
        if diversity_scores:
            results["lexical_diversity"] = np.mean(diversity_scores)
        
        # Save sample generations
        generations_dir = os.path.join(self.output_dir, "sample_generations")
        os.makedirs(generations_dir, exist_ok=True)
        
        with open(os.path.join(generations_dir, f"level_{self.results['model']}_{self.args.pruning_levels}.txt"), "w") as f:
            for i, text in enumerate(generated_texts):
                f.write(f"Prompt {i+1}: {prompts[i]}\n")
                f.write(f"Generated: {text}\n\n")
        
        return results
    
    def _save_results(self):
        """Save benchmark results to disk."""
        results_file = os.path.join(self.output_dir, "benchmark_results.json")
        
        # Convert any non-serializable values (like NumPy types) to native Python types
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        serializable_results = json.loads(
            json.dumps(self.results, default=convert_to_serializable)
        )
        
        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Results saved to {results_file}")
    
    def _generate_plots(self):
        """Generate visualization plots from benchmark results."""
        print("\n📊 Generating visualization plots...")
        
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Extract data for plotting
        strategies = list(self.results["metrics"].keys())
        pruning_levels = self.args.pruning_levels
        
        # 1. Plot perplexity vs pruning level
        if self.args.test_perplexity:
            self._plot_metric_vs_pruning("perplexity", "Perplexity", plots_dir)
        
        # 2. Plot inference speed vs pruning level
        if self.args.test_speed:
            self._plot_speed_vs_pruning(plots_dir)
        
        # 3. Plot memory usage vs pruning level
        if self.args.test_memory and any("memory_usage" in self.results["metrics"][s].get(str(pl), {}) 
                                       for s in strategies for pl in pruning_levels):
            self._plot_memory_vs_pruning(plots_dir)
        
        # 4. Plot text quality metrics vs pruning level
        if self.args.test_generation:
            quality_metrics = ["repetition_score", "lexical_diversity"]
            for metric in quality_metrics:
                pretty_name = " ".join(word.capitalize() for word in metric.split("_"))
                self._plot_quality_metric_vs_pruning(metric, pretty_name, plots_dir)
        
        # 5. Plot active heads vs pruning level
        self._plot_active_heads_vs_pruning(plots_dir)
        
        print(f"📊 Plots saved to {plots_dir}")
    
    def _plot_metric_vs_pruning(self, metric_name, pretty_name, plots_dir):
        """Plot a specific metric vs pruning level for all strategies."""
        plt.figure(figsize=(10, 6))
        
        for strategy in self.results["metrics"]:
            x_values = []
            y_values = []
            
            for level in self.args.pruning_levels:
                level_str = str(level)
                if level_str in self.results["metrics"][strategy]:
                    if metric_name in self.results["metrics"][strategy][level_str]:
                        x_values.append(level)
                        y_values.append(self.results["metrics"][strategy][level_str][metric_name])
            
            if x_values and y_values:
                plt.plot(x_values, y_values, 'o-', label=strategy.capitalize())
        
        plt.xlabel("Pruning Level")
        plt.ylabel(pretty_name)
        plt.title(f"{pretty_name} vs Pruning Level")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(os.path.join(plots_dir, f"{metric_name}_vs_pruning.png"))
        plt.close()
    
    def _plot_speed_vs_pruning(self, plots_dir):
        """Plot inference speed vs pruning level for all strategies."""
        # Plot for each batch size
        for batch_size in self.args.batch_sizes:
            batch_size = str(batch_size)  # Convert to string for lookup
            
            plt.figure(figsize=(10, 6))
            
            for strategy in self.results["metrics"]:
                x_values = []
                y_values = []
                
                for level in self.args.pruning_levels:
                    level_str = str(level)
                    if level_str in self.results["metrics"][strategy]:
                        if "inference_speed" in self.results["metrics"][strategy][level_str]:
                            speed_results = self.results["metrics"][strategy][level_str]["inference_speed"]
                            if batch_size in speed_results:
                                x_values.append(level)
                                y_values.append(speed_results[batch_size])
                
                if x_values and y_values:
                    plt.plot(x_values, y_values, 'o-', label=strategy.capitalize())
            
            plt.xlabel("Pruning Level")
            plt.ylabel("Tokens per Second")
            plt.title(f"Inference Speed vs Pruning Level (Batch Size {batch_size})")
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.savefig(os.path.join(plots_dir, f"speed_batch{batch_size}_vs_pruning.png"))
            plt.close()
    
    def _plot_memory_vs_pruning(self, plots_dir):
        """Plot memory usage vs pruning level for all strategies."""
        # Plot for each batch size
        for batch_size in self.args.batch_sizes:
            batch_size = str(batch_size)  # Convert to string for lookup
            
            plt.figure(figsize=(10, 6))
            
            for strategy in self.results["metrics"]:
                x_values = []
                y_values = []
                
                for level in self.args.pruning_levels:
                    level_str = str(level)
                    if level_str in self.results["metrics"][strategy]:
                        if "memory_usage" in self.results["metrics"][strategy][level_str]:
                            memory_results = self.results["metrics"][strategy][level_str]["memory_usage"]
                            if batch_size in memory_results:
                                x_values.append(level)
                                y_values.append(memory_results[batch_size])
                
                if x_values and y_values:
                    plt.plot(x_values, y_values, 'o-', label=strategy.capitalize())
            
            plt.xlabel("Pruning Level")
            plt.ylabel("Peak Memory (MB)")
            plt.title(f"Memory Usage vs Pruning Level (Batch Size {batch_size})")
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.savefig(os.path.join(plots_dir, f"memory_batch{batch_size}_vs_pruning.png"))
            plt.close()
    
    def _plot_quality_metric_vs_pruning(self, metric_name, pretty_name, plots_dir):
        """Plot text quality metrics vs pruning level for all strategies."""
        plt.figure(figsize=(10, 6))
        
        for strategy in self.results["metrics"]:
            x_values = []
            y_values = []
            
            for level in self.args.pruning_levels:
                level_str = str(level)
                if level_str in self.results["metrics"][strategy]:
                    if "generation_quality" in self.results["metrics"][strategy][level_str]:
                        quality_results = self.results["metrics"][strategy][level_str]["generation_quality"]
                        if metric_name in quality_results:
                            x_values.append(level)
                            y_values.append(quality_results[metric_name])
            
            if x_values and y_values:
                plt.plot(x_values, y_values, 'o-', label=strategy.capitalize())
        
        plt.xlabel("Pruning Level")
        plt.ylabel(pretty_name)
        plt.title(f"{pretty_name} vs Pruning Level")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(os.path.join(plots_dir, f"{metric_name}_vs_pruning.png"))
        plt.close()
    
    def _plot_active_heads_vs_pruning(self, plots_dir):
        """Plot number of active heads vs pruning level for all strategies."""
        plt.figure(figsize=(10, 6))
        
        for strategy in self.results["metrics"]:
            x_values = []
            y_values = []
            
            for level in self.args.pruning_levels:
                level_str = str(level)
                if level_str in self.results["metrics"][strategy]:
                    if "active_heads" in self.results["metrics"][strategy][level_str]:
                        active_heads = self.results["metrics"][strategy][level_str]["active_heads"]
                        total_heads = self.results["metrics"][strategy][level_str]["total_heads"]
                        x_values.append(level)
                        y_values.append(active_heads / total_heads * 100)  # As percentage
            
            if x_values and y_values:
                plt.plot(x_values, y_values, 'o-', label=strategy.capitalize())
        
        plt.xlabel("Target Pruning Level")
        plt.ylabel("Actual % of Active Heads")
        plt.title("Active Attention Heads vs Pruning Level")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(os.path.join(plots_dir, "active_heads_vs_pruning.png"))
        plt.close()


def parse_args():
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Benchmark different pruning strategies and levels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument("--model", type=str, default="gpt2",
                      help="Model name (e.g., 'gpt2', 'distilgpt2')")
    
    # Pruning configuration
    parser.add_argument("--pruning_levels", type=float, nargs="+", 
                      default=[0.0, 0.1, 0.3, 0.5, 0.7],
                      help="Pruning levels to test (fraction of heads to prune)")
    parser.add_argument("--strategies", type=str, nargs="+",
                      default=["entropy", "gradient", "random"],
                      help="Pruning strategies to test")
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, default="wikitext",
                      help="Dataset to use for evaluation")
    parser.add_argument("--max_length", type=int, default=128,
                      help="Maximum sequence length for dataset")
    
    # Benchmark configuration
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Batch size for metrics calculation")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8],
                      help="Batch sizes to test for inference speed and memory")
    parser.add_argument("--generation_length", type=int, default=100,
                      help="Length of generated text")
    
    # Test selection
    parser.add_argument("--test_perplexity", action="store_true", default=True,
                      help="Test perplexity")
    parser.add_argument("--test_speed", action="store_true", default=True,
                      help="Test inference speed")
    parser.add_argument("--test_memory", action="store_true", default=True,
                      help="Test memory usage")
    parser.add_argument("--test_generation", action="store_true", default=True,
                      help="Test text generation quality")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                      help="Directory to save benchmark results")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None,
                      help="Device to use (default: auto-detect)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark = PruningBenchmark(args)
    benchmark.run()