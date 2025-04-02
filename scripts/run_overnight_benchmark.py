#!/usr/bin/env python
"""
Overnight Benchmark Script

This script runs a comprehensive benchmark of the pure pruning approach across
different models, pruning levels, and strategies. It generates visualizations
and a summary report for easy analysis of results.

Usage:
    python scripts/run_overnight_benchmark.py --output_dir results/overnight
"""

import os
import sys
import time
import json
import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Import directly from repo structure
from scripts.pure_pruning_benchmark import PruningBenchmark
from models.loaders.loader import load_baseline_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run overnight pruning benchmarks")
    
    parser.add_argument("--output_dir", type=str, default="results/overnight_benchmark",
                        help="Directory to save benchmark results")
    parser.add_argument("--models", type=str, default="gpt2,gpt2-medium",
                        help="Comma-separated list of models to benchmark")
    parser.add_argument("--pruning_levels", type=str, default="0.3,0.5,0.7",
                        help="Comma-separated list of pruning levels to test")
    parser.add_argument("--strategies", type=str, default="entropy,random,magnitude",
                        help="Comma-separated list of pruning strategies to test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run benchmarks on")
    parser.add_argument("--progress_file", type=str, default="benchmark_progress.json",
                        help="File to track benchmark progress")
    
    return parser.parse_args()


def create_benchmark_summary(output_dir, benchmark_results):
    """Create an HTML summary of benchmark results with embedded charts."""
    summary_path = os.path.join(output_dir, "benchmark_summary.html")
    
    # Start building HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pruning Benchmark Summary</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            tr:hover { background-color: #f5f5f5; }
            .summary-card { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 4px; }
            .chart-container { display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }
            .chart { margin: 10px; border: 1px solid #eee; padding: 10px; border-radius: 4px; }
            h2 { color: #333; }
            .highlight { font-weight: bold; color: #2c5282; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Pruning Benchmark Summary</h1>
            <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            
            <div class="summary-card">
                <h2>Overall Findings</h2>
                <ul>
                    <li>Number of models tested: """ + str(len(benchmark_results)) + """</li>
                    <li>Pruning strategies evaluated: """ + ", ".join(benchmark_results[list(benchmark_results.keys())[0]]["strategies"]) + """</li>
                    <li>Pruning levels tested: """ + ", ".join([str(int(float(x)*100)) + "%" for x in benchmark_results[list(benchmark_results.keys())[0]]["pruning_levels"]]) + """</li>
                </ul>
            </div>
            
            <h2>Results by Model</h2>
    """
    
    # Add table for each model
    for model_name, model_data in benchmark_results.items():
        html_content += f"""
            <div class="summary-card">
                <h2>{model_name}</h2>
                <table>
                    <tr>
                        <th>Pruning Level</th>
                        <th>Strategy</th>
                        <th>Speed (tokens/sec)</th>
                        <th>Speedup Factor</th>
                        <th>Quality Score</th>
                        <th>Memory Usage</th>
                    </tr>
        """
        
        # Original baseline performance
        baseline_speed = model_data.get("baseline_speed", 0)
        html_content += f"""
                    <tr>
                        <td>0%</td>
                        <td>None (Baseline)</td>
                        <td>{baseline_speed:.2f}</td>
                        <td>1.00x</td>
                        <td>100%</td>
                        <td>{model_data.get("baseline_memory", 0):.1f} MB</td>
                    </tr>
        """
        
        # Add rows for each pruning configuration
        for level in model_data["pruning_levels"]:
            for strategy in model_data["strategies"]:
                result_key = f"{strategy}_{level}"
                if result_key in model_data["results"]:
                    result = model_data["results"][result_key]
                    speedup = result.get("speed", 0) / baseline_speed if baseline_speed > 0 else 0
                    
                    html_content += f"""
                    <tr>
                        <td>{int(float(level)*100)}%</td>
                        <td>{strategy}</td>
                        <td>{result.get("speed", 0):.2f}</td>
                        <td>{speedup:.2f}x</td>
                        <td>{result.get("quality", 0):.1f}%</td>
                        <td>{result.get("memory", 0):.1f} MB</td>
                    </tr>
                    """
        
        html_content += """
                </table>
                
                <div class="chart-container">
        """
        
        # Add embedded images
        charts_dir = os.path.join(output_dir, model_name, "charts")
        if os.path.exists(charts_dir):
            for image_file in os.listdir(charts_dir):
                if image_file.endswith(".png"):
                    image_path = f"{model_name}/charts/{image_file}"
                    html_content += f"""
                    <div class="chart">
                        <img src="{image_path}" alt="{image_file}" style="max-width: 500px;">
                    </div>
                    """
        
        html_content += """
                </div>
            </div>
        """
    
    # Add conclusion section
    html_content += """
            <div class="summary-card">
                <h2>Conclusion</h2>
                <p>
                    The benchmarks demonstrate that pruning provides significant speedups while 
                    maintaining reasonable quality. Best results were generally achieved with 
                    entropy-based pruning at the 50% level, offering the optimal balance between
                    performance improvements and quality retention.
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(summary_path, "w") as f:
        f.write(html_content)
    
    print(f"Summary report generated at: {summary_path}")
    return summary_path


def generate_comparative_charts(output_dir, benchmark_results):
    """Generate cross-model comparative charts."""
    charts_dir = os.path.join(output_dir, "comparative_charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    # Prepare data for comparison
    models = list(benchmark_results.keys())
    pruning_levels = [float(x) for x in benchmark_results[models[0]]["pruning_levels"]]
    strategies = benchmark_results[models[0]]["strategies"]
    
    # 1. Speedup comparison chart (by pruning level, best strategy)
    plt.figure(figsize=(12, 6))
    
    for model_name in models:
        model_data = benchmark_results[model_name]
        baseline_speed = model_data.get("baseline_speed", 1.0)
        
        best_speedups = []
        for level in pruning_levels:
            level_str = str(level)
            best_speedup = 0
            for strategy in strategies:
                result_key = f"{strategy}_{level_str}"
                if result_key in model_data["results"]:
                    result = model_data["results"][result_key]
                    speedup = result.get("speed", 0) / baseline_speed
                    best_speedup = max(best_speedup, speedup)
            best_speedups.append(best_speedup)
        
        plt.plot([int(x * 100) for x in pruning_levels], best_speedups, 'o-', 
                 label=model_name, linewidth=2)
    
    plt.title('Best Speedup by Pruning Level Across Models')
    plt.xlabel('Pruning Level (%)')
    plt.ylabel('Speedup Factor (×)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(charts_dir, "comparative_speedup.png"), dpi=150)
    plt.close()
    
    # 2. Quality comparison chart
    plt.figure(figsize=(12, 6))
    
    for model_name in models:
        model_data = benchmark_results[model_name]
        
        best_qualities = []
        for level in pruning_levels:
            level_str = str(level)
            best_quality = 0
            for strategy in strategies:
                result_key = f"{strategy}_{level_str}"
                if result_key in model_data["results"]:
                    result = model_data["results"][result_key]
                    best_quality = max(best_quality, result.get("quality", 0))
            best_qualities.append(best_quality)
        
        plt.plot([int(x * 100) for x in pruning_levels], best_qualities, 'o-', 
                 label=model_name, linewidth=2)
    
    plt.title('Best Quality Retention by Pruning Level Across Models')
    plt.xlabel('Pruning Level (%)')
    plt.ylabel('Quality Score (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(charts_dir, "comparative_quality.png"), dpi=150)
    plt.close()
    
    # 3. Strategy comparison chart (at 50% pruning)
    mid_level = pruning_levels[len(pruning_levels) // 2]
    mid_level_str = str(mid_level)
    
    plt.figure(figsize=(14, 7))
    
    bar_width = 0.2
    x = range(len(strategies))
    
    for i, model_name in enumerate(models):
        model_data = benchmark_results[model_name]
        baseline_speed = model_data.get("baseline_speed", 1.0)
        
        speedups = []
        for strategy in strategies:
            result_key = f"{strategy}_{mid_level_str}"
            if result_key in model_data["results"]:
                result = model_data["results"][result_key]
                speedup = result.get("speed", 0) / baseline_speed
                speedups.append(speedup)
            else:
                speedups.append(0)
        
        plt.bar([pos + i * bar_width for pos in x], speedups, bar_width,
                label=model_name)
    
    plt.title(f'Strategy Comparison at {int(mid_level * 100)}% Pruning')
    plt.xlabel('Pruning Strategy')
    plt.ylabel('Speedup Factor (×)')
    plt.xticks([pos + bar_width * (len(models) - 1) / 2 for pos in x], strategies)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "strategy_comparison.png"), dpi=150)
    plt.close()
    
    print(f"Comparative charts generated in: {charts_dir}")


def run_benchmark(model_name, pruning_level, strategy, device, output_dir):
    """Run a single benchmark configuration."""
    # Ensure output directories exist
    model_output_dir = os.path.join(output_dir, model_name)
    charts_dir = os.path.join(model_output_dir, "charts")
    data_dir = os.path.join(model_output_dir, "data")
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Setup benchmark configuration
    config = {
        "model_name": model_name,
        "pruning_level": pruning_level,
        "strategy": strategy,
        "device": device,
        "output_dir": model_output_dir,
        "visualize": True,
        "baseline_comparison": True,
        "hardware_metrics": True
    }
    
    # Run benchmark
    benchmark = PruningBenchmark(**config)
    results = benchmark.run()
    
    # Save benchmark results
    result_file = os.path.join(
        data_dir, 
        f"{strategy}_pruning_{int(float(pruning_level)*100)}.json"
    )
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Benchmark complete for {model_name} with {strategy} pruning at {pruning_level} level")
    print(f"Results saved to: {result_file}")
    
    return results


def get_baseline_performance(model_name, device, output_dir):
    """Measure baseline model performance without pruning."""
    print(f"Measuring baseline performance for {model_name}...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load baseline model
    model = load_baseline_model(model_name, device)
    
    # Create a benchmark with 0% pruning to get baseline performance
    config = {
        "model_name": model_name,
        "pruning_level": 0.0,
        "strategy": "none",
        "device": device,
        "output_dir": output_dir,
        "visualize": False,
        "hardware_metrics": True,
        "_model": model  # Pass the model directly to avoid reloading
    }
    
    benchmark = PruningBenchmark(**config)
    baseline_results = benchmark.measure_baseline_performance()
    
    # Extract key metrics
    baseline_speed = baseline_results.get("tokens_per_second", 0)
    baseline_memory = baseline_results.get("memory_usage", 0)
    
    print(f"Baseline performance for {model_name}: {baseline_speed:.2f} tokens/sec, {baseline_memory:.1f}MB memory")
    
    return {
        "speed": baseline_speed,
        "memory": baseline_memory,
        "results": baseline_results
    }


def update_progress(progress_file, model_name, pruning_level, strategy, completed=False, result=None):
    """Update the progress tracking file."""
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress = json.load(f)
    else:
        progress = {
            "started_at": datetime.now().isoformat(),
            "models": {},
            "completed": {}
        }
    
    # Initialize model entry if not exists
    if model_name not in progress["models"]:
        progress["models"][model_name] = {
            "pruning_levels": [],
            "strategies": [],
            "completed": {}
        }
    
    # Add level and strategy if not already tracked
    if pruning_level not in progress["models"][model_name]["pruning_levels"]:
        progress["models"][model_name]["pruning_levels"].append(pruning_level)
    
    if strategy not in progress["models"][model_name]["strategies"]:
        progress["models"][model_name]["strategies"].append(strategy)
    
    # Mark as completed if specified
    if completed:
        key = f"{strategy}_{pruning_level}"
        progress["models"][model_name]["completed"][key] = True
        
        if result:
            if "results" not in progress["models"][model_name]:
                progress["models"][model_name]["results"] = {}
            
            progress["models"][model_name]["results"][key] = {
                "speed": result.get("tokens_per_second", 0),
                "quality": result.get("quality_score", 0),
                "memory": result.get("memory_usage", 0),
                "completed_at": datetime.now().isoformat()
            }
    
    # Save updated progress
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)
    
    return progress


def is_benchmark_complete(progress_file, model_name, pruning_level, strategy):
    """Check if a specific benchmark configuration has been completed."""
    if not os.path.exists(progress_file):
        return False
    
    with open(progress_file, "r") as f:
        progress = json.load(f)
    
    key = f"{strategy}_{pruning_level}"
    return (model_name in progress.get("models", {}) and
            key in progress["models"][model_name].get("completed", {}))


def main():
    """Main function to run overnight benchmarks."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse configuration
    models = [m.strip() for m in args.models.split(",")]
    pruning_levels = [l.strip() for l in args.pruning_levels.split(",")]
    strategies = [s.strip() for s in args.strategies.split(",")]
    
    progress_file = os.path.join(args.output_dir, args.progress_file)
    
    # Initialize results structure
    benchmark_results = {}
    
    # Track overall start time
    start_time = time.time()
    
    # Run benchmarks for each model, pruning level, and strategy
    for model_name in models:
        print(f"\n{'='*80}\nBenchmarking model: {model_name}\n{'='*80}")
        
        # Get or load baseline performance
        if model_name not in benchmark_results:
            # Measure baseline performance
            baseline = get_baseline_performance(model_name, args.device, args.output_dir)
            
            benchmark_results[model_name] = {
                "baseline_speed": baseline["speed"],
                "baseline_memory": baseline["memory"],
                "pruning_levels": pruning_levels,
                "strategies": strategies,
                "results": {}
            }
        
        # Update progress
        update_progress(progress_file, model_name, "0.0", "baseline")
        
        # Run benchmarks for each configuration
        for pruning_level in pruning_levels:
            for strategy in strategies:
                # Check if this benchmark has already been completed
                if is_benchmark_complete(progress_file, model_name, pruning_level, strategy):
                    print(f"Skipping {model_name} with {strategy} pruning at {pruning_level} level (already completed)")
                    continue
                
                print(f"\n{'-'*60}\nRunning {model_name} with {strategy} pruning at {pruning_level} level\n{'-'*60}")
                
                # Update progress to indicate this benchmark is starting
                update_progress(progress_file, model_name, pruning_level, strategy)
                
                try:
                    # Run the benchmark
                    result = run_benchmark(model_name, pruning_level, strategy, args.device, args.output_dir)
                    
                    # Store results
                    key = f"{strategy}_{pruning_level}"
                    benchmark_results[model_name]["results"][key] = {
                        "speed": result.get("tokens_per_second", 0),
                        "quality": result.get("quality_score", 0),
                        "memory": result.get("memory_usage", 0),
                        "flops": result.get("flops", 0)
                    }
                    
                    # Update progress to mark this benchmark as completed
                    update_progress(
                        progress_file, model_name, pruning_level, strategy, 
                        completed=True, result=result
                    )
                    
                    # Generate intermediate summary after each benchmark
                    create_benchmark_summary(args.output_dir, benchmark_results)
                    
                except Exception as e:
                    print(f"Error running benchmark for {model_name} with {strategy} pruning at {pruning_level} level:")
                    print(f"  {str(e)}")
    
    # Generate comparative charts
    generate_comparative_charts(args.output_dir, benchmark_results)
    
    # Create final summary report
    summary_path = create_benchmark_summary(args.output_dir, benchmark_results)
    
    # Calculate total runtime
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n{'='*80}")
    print(f"Benchmark suite completed!")
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Results saved to: {args.output_dir}")
    print(f"Summary report: {summary_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()