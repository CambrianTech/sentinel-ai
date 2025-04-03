#!/usr/bin/env python
"""
Analyze and visualize the impact of pruning on model performance.

This script provides a focused analysis on how different pruning levels
affect model performance metrics. It's designed to be more lightweight
and visual than the comprehensive benchmark_pruning.py script.

Usage:
    python scripts/pruning_impact_analyzer.py --model gpt2 \
                                             --pruning_levels 0.0 0.3 0.5 0.7 \
                                             --metric perplexity
"""

import os
import sys
import argparse
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.loaders.loader import load_baseline_model, load_adaptive_model
from sentinel_data.dataset_loader import load_dataset
from utils.metrics import compute_perplexity, compute_text_statistics
from utils.generation_wrapper import generate_text
from controller.metrics.head_metrics import collect_head_metrics


class PruningImpactAnalyzer:
    """
    Analyzer for visualizing the impact of pruning on model performance.
    """
    
    def __init__(self, args):
        """Initialize the analyzer with command line arguments."""
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
            args.output_dir, f"pruning_analysis_{args.model.split('/')[-1]}_{timestamp}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"🔍 Pruning impact analyzer initialized for model: {args.model}")
        print(f"💻 Using device: {self.device}")
        print(f"📊 Analyzing pruning levels: {args.pruning_levels}")
        print(f"📏 Metric: {args.metric}")
    
    def run(self):
        """Run the analysis across all pruning levels."""
        # Load tokenizer
        print(f"\n⚙️ Loading tokenizer for model: {self.args.model}")
        tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load dataset
        print(f"\n📚 Loading dataset: {self.args.dataset}")
        train_dataset, eval_dataset = load_dataset(
            self.args.dataset, tokenizer, self.args.max_length
        )
        
        # Create evaluation dataloader
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=self.args.batch_size
        )
        
        # Create a batch for head importance metrics
        dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True
        )
        sample_batch = next(iter(dataloader))
        sample_batch = {k: v.to(self.device) for k, v in sample_batch.items()}
        
        # Test baseline model first
        print(f"\n📊 Testing baseline model (no pruning)")
        baseline_model = load_baseline_model(self.args.model, self.device)
        baseline_metric = self._evaluate_model(baseline_model, eval_loader)
        
        # Load adaptive model
        model = load_adaptive_model(self.args.model, baseline_model, self.device)
        
        # Calculate head importance metrics
        print(f"\n📏 Calculating head importance metrics...")
        importance_metrics = collect_head_metrics(model, batch=sample_batch)
        
        # Track results for each pruning level
        results = {
            "model": self.args.model,
            "metric": self.args.metric,
            "baseline": baseline_metric,
            "pruning_levels": {}
        }
        
        # Test each pruning level
        for pruning_level in self.args.pruning_levels:
            if pruning_level <= 0.0:
                # Skip pruning for level 0.0
                results["pruning_levels"][str(pruning_level)] = baseline_metric
                continue
                
            print(f"\n📏 Testing pruning level: {pruning_level}")
            
            # Create a fresh model copy for each pruning level
            pruned_model = load_adaptive_model(self.args.model, baseline_model, self.device)
            
            # Apply pruning based on importance metrics
            self._apply_pruning(pruned_model, pruning_level, importance_metrics)
            
            # Evaluate the pruned model
            metric_value = self._evaluate_model(pruned_model, eval_loader)
            results["pruning_levels"][str(pruning_level)] = metric_value
            
            # Free memory
            del pruned_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Clean up
        del baseline_model
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Save and visualize results
        self._save_results(results)
        self._visualize_results(results)
        
        print(f"\n✅ Analysis complete. Results saved to: {self.output_dir}")
    
    def _apply_pruning(self, model, pruning_level, metrics):
        """Apply pruning to the model based on head importance metrics."""
        num_layers = len(model.blocks)
        num_heads = model.blocks[0]["attn"].num_heads
        total_heads = num_layers * num_heads
        heads_to_prune = int(total_heads * pruning_level)
        
        if heads_to_prune <= 0:
            return model
        
        # Determine which heads to prune based on importance metrics
        importance_scores = torch.zeros((num_layers, num_heads)).to(self.device)
        
        # Decide which metric to use for importance
        if "entropy" in metrics:
            # Higher entropy = less focused attention = less important
            importance_scores = metrics["entropy"]
        elif "grad_norm" in metrics:
            # Lower gradient norm = less important
            importance_scores = -metrics["grad_norm"]  # Negate so highest scores are least important
        else:
            # Fallback to random pruning
            importance_scores = torch.rand((num_layers, num_heads)).to(self.device)
        
        # Flatten scores and find threshold
        flat_scores = importance_scores.view(-1)
        _, indices = torch.sort(flat_scores, descending=True)
        threshold_idx = min(heads_to_prune, len(indices) - 1)
        threshold_value = flat_scores[indices[threshold_idx]]
        
        # Apply pruning
        with torch.no_grad():
            heads_pruned = 0
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    if importance_scores[layer_idx, head_idx] >= threshold_value and heads_pruned < heads_to_prune:
                        # Prune this head (set gate to near-zero)
                        model.blocks[layer_idx]["attn"].gate[head_idx] = torch.tensor(0.001, device=self.device)
                        heads_pruned += 1
                    else:
                        # Keep this head active
                        model.blocks[layer_idx]["attn"].gate[head_idx] = torch.tensor(0.999, device=self.device)
        
        print(f"   ✂️ Pruned {heads_pruned} of {total_heads} attention heads ({pruning_level*100:.1f}%)")
        return model
    
    def _evaluate_model(self, model, eval_loader):
        """Evaluate the model using the selected metric."""
        model.eval()
        
        if self.args.metric == "perplexity":
            try:
                value = compute_perplexity(model, eval_loader, self.device)
                print(f"   📊 Perplexity: {value:.2f}")
                return value
            except Exception as e:
                print(f"   ❌ Error computing perplexity: {e}")
                return float("nan")
        
        else:
            print(f"   ⚠️ Unsupported metric: {self.args.metric}")
            return None
    
    def _save_results(self, results):
        """Save analysis results to disk."""
        results_file = os.path.join(self.output_dir, "pruning_impact_results.json")
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to {results_file}")
    
    def _visualize_results(self, results):
        """Create visualizations of the pruning impact."""
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Extract data for plotting
        pruning_levels = [float(level) for level in results["pruning_levels"].keys()]
        metric_values = list(results["pruning_levels"].values())
        baseline_value = results["baseline"]
        metric_name = results["metric"]
        
        # Create line plot
        plt.figure(figsize=(10, 6))
        plt.plot(pruning_levels, metric_values, 'o-', linewidth=2, color='#1f77b4')
        plt.axhline(y=baseline_value, color='#d62728', linestyle='--', label=f'Baseline ({baseline_value:.2f})')
        
        plt.xlabel("Pruning Level (fraction of heads pruned)")
        plt.ylabel(metric_name.capitalize())
        plt.title(f"Impact of Pruning on {metric_name.capitalize()}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{metric_name}_vs_pruning.png"))
        
        # Create bar plot with percentage change
        plt.figure(figsize=(10, 6))
        
        # Calculate percentage change from baseline
        if baseline_value != 0:
            pct_changes = [(value - baseline_value) / baseline_value * 100 for value in metric_values]
            
            # For perplexity, negative change is better
            colors = ['#2ca02c' if change < 0 else '#d62728' for change in pct_changes]
            
            # Create bar plot
            sns.barplot(x=pruning_levels, y=pct_changes, palette=colors)
            
            plt.xlabel("Pruning Level")
            plt.ylabel(f"% Change in {metric_name.capitalize()}")
            plt.title(f"Percentage Change in {metric_name.capitalize()} from Baseline")
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.grid(True, axis='y', alpha=0.3)
            
            for i, (level, change) in enumerate(zip(pruning_levels, pct_changes)):
                plt.text(i, change + (5 if change > 0 else -5), f"{change:.1f}%", 
                        ha='center', va='center', fontweight='bold', color='white')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{metric_name}_pct_change.png"))
        
        plt.close('all')


def parse_args():
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Analyze the impact of pruning on model performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--model", type=str, default="gpt2",
                      help="Model name (e.g., 'gpt2', 'distilgpt2')")
    parser.add_argument("--pruning_levels", type=float, nargs="+", 
                      default=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
                      help="Pruning levels to test (fraction of heads to prune)")
    parser.add_argument("--metric", type=str, default="perplexity",
                      choices=["perplexity"],
                      help="Metric to evaluate")
    parser.add_argument("--dataset", type=str, default="wikitext",
                      help="Dataset to use for evaluation")
    parser.add_argument("--max_length", type=int, default=128,
                      help="Maximum sequence length for dataset")
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default="./analysis_results",
                      help="Directory to save analysis results")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None,
                      help="Device to use (default: auto-detect)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyzer = PruningImpactAnalyzer(args)
    analyzer.run()