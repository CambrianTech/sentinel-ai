#!/usr/bin/env python
"""
Run Neural Plasticity Benchmark

This script benchmarks multiple configurations of the neural plasticity system:
1. Tests both entropy and magnitude-based pruning at different levels
2. Compares recovery rates for different models and pruning levels
3. Analyzes which heads regrow after pruning
4. Measures the impact of different learning rates on recovery
5. Generates comprehensive visualizations and reports

The benchmark helps identify optimal pruning strategies and configurations
for different model architectures.
"""

import os
import sys
import torch
import argparse
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from datetime import datetime
from itertools import product

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Sentinel-AI modules
from sentinel.plasticity.plasticity_loop import PlasticityExperiment, run_plasticity_experiment
from sentinel.utils.viz.heatmaps import (
    plot_entropy_heatmap,
    plot_entropy_deltas_heatmap,
    plot_gate_activity,
    plot_regrowth_heatmap
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run Neural Plasticity Benchmark")
    
    # Model parameters
    parser.add_argument("--models", type=str, default="distilgpt2",
                        help="Comma-separated list of models to benchmark (default: distilgpt2)")
    parser.add_argument("--output_dir", type=str, default="./output/plasticity_benchmark",
                        help="Directory to save results (default: ./output/plasticity_benchmark)")
    
    # Experiment parameters
    parser.add_argument("--pruning_strategies", type=str, default="entropy,magnitude",
                        help="Comma-separated list of pruning strategies (default: entropy,magnitude)")
    parser.add_argument("--pruning_levels", type=str, default="0.1,0.3,0.5",
                        help="Comma-separated list of pruning levels (default: 0.1,0.3,0.5)")
    parser.add_argument("--training_steps", type=int, default=300,
                        help="Number of fine-tuning steps (default: 300)")
    parser.add_argument("--learning_rates", type=str, default="5e-5",
                        help="Comma-separated list of learning rates (default: 5e-5)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (default: 4)")
    
    # Benchmark parameters
    parser.add_argument("--quick", action="store_true",
                        help="Run a quicker benchmark with fewer configurations")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: auto-detect)")
    parser.add_argument("--log_level", type=str, default="info",
                        choices=["debug", "info", "warning", "error"],
                        help="Logging level (default: info)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    return parser.parse_args()

def get_dataloader_builder(batch_size=4):
    """
    Create a function that returns train and evaluation dataloaders.
    Uses a simple dataset for testing purposes.
    """
    from transformers import AutoTokenizer
    import torch
    
    # Create synthetic data
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a world where technology dominates, humans seek connection.",
        "Once upon a time, there lived a wise king who ruled with compassion.",
        "The history of artificial intelligence dates back to ancient myths.",
        "Climate change is affecting ecosystems worldwide, leading to rising sea levels.",
        "The transformer architecture revolutionized natural language processing tasks.",
        "Neural plasticity allows models to adapt their structure during training.",
        "Deep learning models can recognize patterns in complex data.",
        "The attention mechanism focuses on different parts of the input sequence.",
        "Language models predict the next token based on previous context.",
        "The adaptive transformer uses a controller network to modulate attention.",
        "Scientists observed similar mechanisms in biological neural networks.",
        "Transfer learning enables models to apply knowledge from different domains.",
        "Self-supervised learning reduces the need for labeled training data.",
        "Entropy-based pruning identifies heads with diffuse attention patterns.",
        "Magnitude-based pruning removes weights with the smallest absolute values.",
        "Attention heads specialize in different aspects of language understanding.",
        "The model adapts to structural changes through neural plasticity.",
        "Fine-tuning with differential learning rates accelerates adaptation.",
        "Performance can recover even after pruning significant parts of the network."
    ] * 10  # Repeat to create more samples
    
    def build_dataloaders(model_name="distilgpt2", batch_size=4):
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenize
        from torch.utils.data import TensorDataset, DataLoader
        
        encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        
        dataset = TensorDataset(input_ids, attention_mask)
        
        # Split into train and eval
        train_size = int(0.8 * len(dataset))
        eval_size = len(dataset) - train_size
        
        train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
        
        return train_dataloader, eval_dataloader
    
    # Return a function that will create dataloaders with the specified batch size
    return lambda batch_size=batch_size: build_dataloaders(batch_size=batch_size)

def generate_comparison_plots(benchmark_results, output_dir):
    """Generate comparison plots across different configurations"""
    plots_dir = os.path.join(output_dir, "comparison_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract data for plotting
    models = []
    strategies = []
    pruning_levels = []
    learning_rates = []
    recovery_rates = []
    perplexity_improvements = []
    efficiency_improvements = []
    
    for result in benchmark_results:
        models.append(result["model"])
        strategies.append(result["pruning_strategy"])
        pruning_levels.append(result["pruning_level"])
        learning_rates.append(result["learning_rate"])
        recovery_rates.append(result["recovery_rate"])
        
        # Calculate perplexity improvement
        baseline_perplexity = result["metrics"]["baseline"]["perplexity"]
        final_perplexity = result["metrics"]["final"]["perplexity"]
        improvement = (baseline_perplexity - final_perplexity) / baseline_perplexity
        perplexity_improvements.append(improvement)
        
        # Calculate efficiency improvement (perplexity per head)
        baseline_head_count = result["head_counts"]["original"]
        final_head_count = result["head_counts"]["final"]
        baseline_efficiency = baseline_perplexity / baseline_head_count
        final_efficiency = final_perplexity / final_head_count
        efficiency = (baseline_efficiency - final_efficiency) / baseline_efficiency
        efficiency_improvements.append(efficiency)
    
    # 1. Recovery rate by pruning level and strategy
    unique_strategies = sorted(set(strategies))
    unique_levels = sorted(set(pruning_levels))
    
    # Create a grid of subplots (one per model)
    unique_models = sorted(set(models))
    fig, axes = plt.subplots(1, len(unique_models), figsize=(15, 6), sharey=True)
    if len(unique_models) == 1:
        axes = [axes]
    
    # Set up bar width and positions
    width = 0.8 / len(unique_strategies)
    x = list(range(len(unique_levels)))
    
    for model_idx, model in enumerate(unique_models):
        ax = axes[model_idx]
        
        for strategy_idx, strategy in enumerate(unique_strategies):
            # Filter data for this model and strategy
            indices = [i for i, (m, s) in enumerate(zip(models, strategies))
                       if m == model and s == strategy]
            
            # Get recovery rates and levels for this combination
            level_values = [pruning_levels[i] for i in indices]
            recovery_values = [recovery_rates[i] for i in indices]
            
            # Sort by pruning level
            sorted_data = sorted(zip(level_values, recovery_values))
            sorted_levels = [d[0] for d in sorted_data]
            sorted_recovery = [d[1] for d in sorted_data]
            
            # Plot bars
            x_pos = [i + strategy_idx * width - (len(unique_strategies) - 1) * width / 2 for i in x]
            ax.bar(x_pos, sorted_recovery, width=width, label=f"{strategy}")
        
        ax.set_title(f"Model: {model}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{level:.1f}" for level in unique_levels])
        ax.set_xlabel("Pruning Level")
        ax.grid(True, linestyle='--', alpha=0.3)
    
    # Set common labels
    axes[0].set_ylabel("Recovery Rate")
    fig.suptitle("Recovery Rate by Pruning Level and Strategy")
    fig.legend(unique_strategies, loc="lower center", ncol=len(unique_strategies), bbox_to_anchor=(0.5, 0))
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    # Save the plot
    plt.savefig(os.path.join(plots_dir, "recovery_rate_by_pruning_level.png"))
    
    # 2. Efficiency improvement by pruning level
    plt.figure(figsize=(10, 6))
    
    for strategy in unique_strategies:
        # Filter data for this strategy
        indices = [i for i, s in enumerate(strategies) if s == strategy]
        level_values = [pruning_levels[i] for i in indices]
        efficiency_values = [efficiency_improvements[i] for i in indices]
        
        # Sort by pruning level
        sorted_data = sorted(zip(level_values, efficiency_values))
        sorted_levels = [d[0] for d in sorted_data]
        sorted_efficiency = [d[1] for d in sorted_data]
        
        # Plot line
        plt.plot(sorted_levels, sorted_efficiency, 'o-', label=strategy)
    
    plt.title("Efficiency Improvement by Pruning Level")
    plt.xlabel("Pruning Level")
    plt.ylabel("Efficiency Improvement")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(plots_dir, "efficiency_by_pruning_level.png"))
    
    # 3. Recovery vs. pruning level for different learning rates (if multiple rates)
    if len(set(learning_rates)) > 1:
        unique_lrs = sorted(set(learning_rates))
        plt.figure(figsize=(10, 6))
        
        for lr in unique_lrs:
            # Filter data for this learning rate
            indices = [i for i, r in enumerate(learning_rates) if r == lr]
            level_values = [pruning_levels[i] for i in indices]
            recovery_values = [recovery_rates[i] for i in indices]
            
            # Sort by pruning level
            sorted_data = sorted(zip(level_values, recovery_values))
            sorted_levels = [d[0] for d in sorted_data]
            sorted_recovery = [d[1] for d in sorted_data]
            
            # Plot line
            plt.plot(sorted_levels, sorted_recovery, 'o-', label=f"LR: {lr}")
        
        plt.title("Recovery Rate by Pruning Level for Different Learning Rates")
        plt.xlabel("Pruning Level")
        plt.ylabel("Recovery Rate")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(plots_dir, "recovery_by_learning_rate.png"))
    
    return plots_dir

def generate_summary_report(benchmark_results, output_dir):
    """Generate a summary report with benchmark results"""
    # Create report directory
    report_dir = os.path.join(output_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate HTML report
    report_path = os.path.join(report_dir, "benchmark_summary.html")
    
    # Start building HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Neural Plasticity Benchmark Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2, h3 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .highlight { font-weight: bold; background-color: #e6f7ff; }
            .section { margin-top: 30px; margin-bottom: 20px; }
            .images { display: flex; flex-wrap: wrap; justify-content: center; }
            .image-container { margin: 10px; text-align: center; }
            .image-container img { max-width: 100%; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <h1>Neural Plasticity Benchmark Report</h1>
        <div class="section">
            <h2>Summary</h2>
            <p>Generated on: {date}</p>
            <p>Total configurations tested: {config_count}</p>
        </div>
        
        <div class="section">
            <h2>Recovery Rate Comparison Table</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Pruning Strategy</th>
                    <th>Pruning Level</th>
                    <th>Learning Rate</th>
                    <th>Recovery Rate</th>
                    <th>Perplexity Change</th>
                    <th>Head Reduction</th>
                </tr>
    """.format(
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        config_count=len(benchmark_results)
    )
    
    # Sort results by recovery rate (descending)
    sorted_results = sorted(benchmark_results, key=lambda x: x["recovery_rate"], reverse=True)
    
    # Add rows for each configuration
    for result in sorted_results:
        model = result["model"]
        strategy = result["pruning_strategy"]
        level = result["pruning_level"]
        lr = result["learning_rate"]
        recovery = result["recovery_rate"]
        
        baseline_perplexity = result["metrics"]["baseline"]["perplexity"]
        final_perplexity = result["metrics"]["final"]["perplexity"]
        perplexity_change = (final_perplexity - baseline_perplexity) / baseline_perplexity * 100
        
        head_count_before = result["head_counts"]["original"]
        head_count_after = result["head_counts"]["final"]
        head_reduction = (head_count_before - head_count_after) / head_count_before * 100
        
        # Determine if this is a top performer
        is_top = recovery > 0.8  # 80% recovery rate or better
        
        row_class = "highlight" if is_top else ""
        
        html += f"""
        <tr class="{row_class}">
            <td>{model}</td>
            <td>{strategy}</td>
            <td>{level:.2f}</td>
            <td>{lr}</td>
            <td>{recovery:.2%}</td>
            <td>{perplexity_change:+.2f}%</td>
            <td>{head_reduction:.2f}%</td>
        </tr>
        """
    
    # Close the table
    html += """
            </table>
        </div>
        
        <div class="section">
            <h2>Comparison Plots</h2>
            <div class="images">
                <div class="image-container">
                    <img src="../comparison_plots/recovery_rate_by_pruning_level.png" alt="Recovery Rate by Pruning Level">
                    <p>Recovery Rate by Pruning Level</p>
                </div>
                <div class="image-container">
                    <img src="../comparison_plots/efficiency_by_pruning_level.png" alt="Efficiency by Pruning Level">
                    <p>Efficiency Improvement by Pruning Level</p>
                </div>
    """
    
    # Add learning rate comparison if available
    if os.path.exists(os.path.join(output_dir, "comparison_plots", "recovery_by_learning_rate.png")):
        html += """
                <div class="image-container">
                    <img src="../comparison_plots/recovery_by_learning_rate.png" alt="Recovery by Learning Rate">
                    <p>Recovery Rate by Learning Rate</p>
                </div>
        """
    
    # Close images and section
    html += """
            </div>
        </div>
        
        <div class="section">
            <h2>Conclusions</h2>
            <ul>
    """
    
    # Best pruning strategy
    strategy_results = {}
    for result in benchmark_results:
        strategy = result["pruning_strategy"]
        if strategy not in strategy_results:
            strategy_results[strategy] = []
        strategy_results[strategy].append(result["recovery_rate"])
    
    # Calculate average recovery rate per strategy
    avg_recovery = {strat: sum(rates)/len(rates) for strat, rates in strategy_results.items()}
    best_strategy = max(avg_recovery.items(), key=lambda x: x[1])
    
    html += f"""
                <li>Best pruning strategy: <strong>{best_strategy[0]}</strong> with average recovery rate of {best_strategy[1]:.2%}</li>
    """
    
    # Best pruning level
    level_results = {}
    for result in benchmark_results:
        level = result["pruning_level"]
        if level not in level_results:
            level_results[level] = []
        level_results[level].append(result["recovery_rate"])
    
    # Calculate average recovery rate per level
    avg_recovery_by_level = {level: sum(rates)/len(rates) for level, rates in level_results.items()}
    best_level = max(avg_recovery_by_level.items(), key=lambda x: x[1])
    
    html += f"""
                <li>Best pruning level: <strong>{best_level[0]:.2f}</strong> with average recovery rate of {best_level[1]:.2%}</li>
    """
    
    # Best model
    model_results = {}
    for result in benchmark_results:
        model = result["model"]
        if model not in model_results:
            model_results[model] = []
        model_results[model].append(result["recovery_rate"])
    
    # Calculate average recovery rate per model
    avg_recovery_by_model = {model: sum(rates)/len(rates) for model, rates in model_results.items()}
    best_model = max(avg_recovery_by_model.items(), key=lambda x: x[1])
    
    html += f"""
                <li>Best performing model: <strong>{best_model[0]}</strong> with average recovery rate of {best_model[1]:.2%}</li>
    """
    
    # Best overall configuration
    best_config = max(benchmark_results, key=lambda x: x["recovery_rate"])
    html += f"""
                <li>Best overall configuration: <strong>{best_config["model"]}</strong> with {best_config["pruning_strategy"]} pruning at {best_config["pruning_level"]:.2f} level and learning rate {best_config["learning_rate"]}, achieving {best_config["recovery_rate"]:.2%} recovery rate</li>
    """
    
    # Close list and section
    html += """
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(report_path, "w") as f:
        f.write(html)
    
    # Save JSON summary
    json_path = os.path.join(report_dir, "benchmark_summary.json")
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "configurations_tested": len(benchmark_results),
        "best_strategy": best_strategy[0],
        "best_strategy_recovery": best_strategy[1],
        "best_level": best_level[0],
        "best_level_recovery": best_level[1],
        "best_model": best_model[0],
        "best_model_recovery": best_model[1],
        "best_configuration": {
            "model": best_config["model"],
            "pruning_strategy": best_config["pruning_strategy"],
            "pruning_level": best_config["pruning_level"],
            "learning_rate": best_config["learning_rate"],
            "recovery_rate": best_config["recovery_rate"]
        },
        "strategy_recovery_rates": avg_recovery,
        "level_recovery_rates": avg_recovery_by_level,
        "model_recovery_rates": avg_recovery_by_model
    }
    
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    return report_path

def main():
    """Main function"""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Parse parameters
    models = args.models.split(",")
    pruning_strategies = args.pruning_strategies.split(",")
    pruning_levels = [float(level) for level in args.pruning_levels.split(",")]
    learning_rates = [float(lr) for lr in args.learning_rates.split(",")]
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, f"benchmark_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Print benchmark configuration
    print(f"=== Neural Plasticity Benchmark ===")
    print(f"Models: {models}")
    print(f"Pruning strategies: {pruning_strategies}")
    print(f"Pruning levels: {pruning_levels}")
    print(f"Learning rates: {learning_rates}")
    print(f"Training steps: {args.training_steps}")
    print(f"Output directory: {output_dir}")
    
    # Create dataloader builder
    dataloader_builder = get_dataloader_builder(batch_size=args.batch_size)
    
    # If in quick mode, reduce the number of configurations
    if args.quick:
        print("\nRunning in quick mode with reduced configurations")
        models = models[:1]  # Just the first model
        pruning_strategies = pruning_strategies[:1]  # Just the first strategy
        pruning_levels = pruning_levels[:2]  # Just the first two levels
        learning_rates = learning_rates[:1]  # Just the first learning rate
    
    # Generate all combinations of parameters
    combinations = list(product(models, pruning_strategies, pruning_levels, learning_rates))
    
    print(f"\nRunning {len(combinations)} configurations...")
    
    # Store all results
    benchmark_results = []
    
    # Run each configuration
    for i, (model, strategy, level, lr) in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Testing: model={model}, strategy={strategy}, level={level:.2f}, lr={lr}")
        
        # Create experiment-specific output directory
        exp_dir = os.path.join(output_dir, f"{model}_{strategy}_{level:.2f}_{lr}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # Run plasticity experiment
        start_time = time.time()
        results = run_plasticity_experiment(
            model_name=model,
            pruning_strategy=strategy,
            prune_ratio=level,
            learning_rate=lr,
            adaptive_lr=True,  # Use differential learning rates
            learning_steps=args.training_steps,
            batch_size=args.batch_size,
            dataloader_builder_fn=dataloader_builder,
            device=args.device,
            output_dir=exp_dir
        )
        elapsed_time = time.time() - start_time
        
        # Add metadata to results
        results["model"] = model
        results["pruning_strategy"] = strategy
        results["pruning_level"] = level
        results["learning_rate"] = lr
        results["execution_time"] = elapsed_time
        
        benchmark_results.append(results)
        
        print(f"Completed in {elapsed_time:.1f} seconds")
        print(f"Recovery rate: {results.get('recovery_rate', 0.0):.2%}")
    
    # Save all benchmark results
    results_path = os.path.join(output_dir, "benchmark_results.json")
    
    # Convert results to a more serialization-friendly format
    serializable_results = []
    for result in benchmark_results:
        # Make a copy of the result object
        clean_result = {}
        for key, value in result.items():
            # Skip complex objects like tensors
            if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                clean_result[key] = value
        serializable_results.append(clean_result)
    
    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plots_dir = generate_comparison_plots(serializable_results, output_dir)
    
    # Generate summary report
    print("Generating summary report...")
    report_path = generate_summary_report(serializable_results, output_dir)
    
    # Print benchmark summary
    print(f"\n=== Benchmark Complete ===")
    print(f"Tested {len(combinations)} configurations")
    print(f"Results saved to: {output_dir}")
    print(f"Summary report: {report_path}")
    
    # Find best configuration
    best_idx = max(range(len(benchmark_results)), key=lambda i: benchmark_results[i].get("recovery_rate", 0))
    best = benchmark_results[best_idx]
    
    print(f"\nBest configuration:")
    print(f"- Model: {best['model']}")
    print(f"- Pruning strategy: {best['pruning_strategy']}")
    print(f"- Pruning level: {best['pruning_level']:.2f}")
    print(f"- Learning rate: {best['learning_rate']}")
    print(f"- Recovery rate: {best.get('recovery_rate', 0.0):.2%}")

if __name__ == "__main__":
    main()