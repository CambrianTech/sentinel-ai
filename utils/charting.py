"""
Charting utilities for Sentinel-AI.

This module provides functions for generating high-quality visualizations
of model performance, agency states, and validation results.

It uses a consistent visual language to tell the story behind the metrics,
with proper annotations and styling.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path

# Set the default style
sns.set(style="whitegrid")

# Define the agency color scheme
AGENCY_COLORS = {
    "active": "#4CAF50",       # Green
    "misaligned": "#2196F3",   # Blue
    "overloaded": "#FF9800",   # Orange
    "withdrawn": "#F44336",    # Red
    "baseline": "#78909C"      # Gray
}

# Define scenario color palettes
SCENARIO_PALETTE = "crest"     # Blue-green gradient
METRIC_PALETTE = "rocket_r"    # Purple gradient

def ensure_output_dir(output_dir=None):
    """
    Ensure the output directory exists.
    
    Args:
        output_dir: Directory path where charts will be saved.
                   If None, defaults to validation_results/agency/
                   
    Returns:
        Path object for the output directory
    """
    if output_dir is None:
        # Default to validation_results/agency/
        output_dir = Path("validation_results/agency/")
    else:
        output_dir = Path(output_dir)
        
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_validation_results(results_path=None):
    """
    Load validation results from the JSON file.
    
    Args:
        results_path: Path to the results JSON file.
                     If None, searches in common locations.
                     
    Returns:
        Dictionary containing the validation results
    """
    if results_path is None:
        # Try common locations
        possible_paths = [
            "validation_results/agency/validation_results.json",
            "validation_results.json",
            "../validation_results/agency/validation_results.json"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                results_path = path
                break
        
        if results_path is None:
            raise FileNotFoundError("Could not find validation results JSON file.")
    
    with open(results_path, "r") as f:
        results = json.load(f)
    
    return results

def plot_generation_speed(results=None, output_dir=None, show=True):
    """
    Plot generation speed comparison across different scenarios.
    
    Args:
        results: Validation results dictionary. If None, loads from file.
        output_dir: Directory to save the plot. If None, uses default location.
        show: Whether to display the plot interactively.
        
    Returns:
        Path to the saved plot
    """
    # Load results if not provided
    if results is None:
        results = load_validation_results()
    
    # Ensure output directory exists
    output_dir = ensure_output_dir(output_dir)
    
    # Prepare data
    scenarios = []
    tokens_per_sec = []
    
    for scenario, data in results.items():
        inference = data.get("inference", {})
        speed = inference.get("tokens_per_second")
        if speed is not None:
            # Beautify scenario names for display
            label = scenario.replace("_", " ").title()
            scenarios.append(label)
            tokens_per_sec.append(speed)
    
    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=scenarios, y=tokens_per_sec, palette=SCENARIO_PALETTE)
    
    # Add values on top of bars
    for i, v in enumerate(tokens_per_sec):
        ax.text(i, v + 0.5, f"{v:.1f}", ha='center', fontsize=10)
    
    plt.title("🎯 Generation Speed Across Agency Configurations", fontsize=16)
    plt.ylabel("Tokens per Second", fontsize=12)
    plt.xlabel("Scenario", fontsize=12)
    plt.xticks(rotation=15)
    
    # Add insight annotation
    if len(tokens_per_sec) > 1:
        best_idx = np.argmax(tokens_per_sec)
        worst_idx = np.argmin(tokens_per_sec)
        improvement = ((tokens_per_sec[best_idx] / tokens_per_sec[worst_idx]) - 1) * 100
        
        plt.figtext(0.5, 0.01, 
                   f"Insight: {scenarios[best_idx]} is {improvement:.1f}% faster than {scenarios[worst_idx]}",
                   ha="center", fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = output_dir / "generation_speed_comparison.png"
    plt.savefig(output_path, dpi=150)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_path

def plot_resource_usage(results=None, output_dir=None, show=True):
    """
    Plot resource usage (memory and compute) across different scenarios.
    
    Args:
        results: Validation results dictionary. If None, loads from file.
        output_dir: Directory to save the plot. If None, uses default location.
        show: Whether to display the plot interactively.
        
    Returns:
        Path to the saved plot
    """
    # Load results if not provided
    if results is None:
        results = load_validation_results()
    
    # Ensure output directory exists
    output_dir = ensure_output_dir(output_dir)
    
    # Prepare data
    scenarios = []
    memory_usage = []
    compute = []
    
    for scenario, data in results.items():
        resources = data.get("resource_usage", {})
        cpu = resources.get("cpu_percent")
        ram = resources.get("ram_percent")
        
        if cpu is not None and ram is not None:
            # Beautify scenario names for display
            label = scenario.replace("_", " ").title()
            scenarios.append(label)
            memory_usage.append(ram)
            compute.append(cpu)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # RAM usage
    sns.barplot(x=scenarios, y=memory_usage, ax=ax1, palette=SCENARIO_PALETTE)
    ax1.set_title("RAM Usage", fontsize=14)
    ax1.set_ylabel("RAM (%)", fontsize=12)
    ax1.set_xlabel("Scenario", fontsize=12)
    ax1.tick_params(axis='x', rotation=15)
    
    # Add values on top of bars
    for i, v in enumerate(memory_usage):
        ax1.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=10)
    
    # CPU usage
    sns.barplot(x=scenarios, y=compute, ax=ax2, palette=SCENARIO_PALETTE)
    ax2.set_title("CPU Usage", fontsize=14)
    ax2.set_ylabel("CPU (%)", fontsize=12)
    ax2.set_xlabel("Scenario", fontsize=12)
    ax2.tick_params(axis='x', rotation=15)
    
    # Add values on top of bars
    for i, v in enumerate(compute):
        ax2.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=10)
    
    # Main title
    fig.suptitle("💻 Resource Utilization: RAM & CPU", fontsize=16)
    
    # Add insight annotation
    if len(memory_usage) > 1:
        best_idx = np.argmin(memory_usage)
        worst_idx = np.argmax(memory_usage)
        improvement = ((memory_usage[worst_idx] / memory_usage[best_idx]) - 1) * 100
        
        plt.figtext(0.5, 0.01, 
                   f"Insight: {scenarios[best_idx]} uses {improvement:.1f}% less RAM than {scenarios[worst_idx]}",
                   ha="center", fontsize=10, style='italic')
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.15)
    
    # Save the plot
    output_path = output_dir / "resource_utilization.png"
    plt.savefig(output_path, dpi=150)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_path

def plot_quality_metrics(results=None, output_dir=None, show=True):
    """
    Plot quality metrics (perplexity, diversity, repetition) across different scenarios.
    
    Args:
        results: Validation results dictionary. If None, loads from file.
        output_dir: Directory to save the plot. If None, uses default location.
        show: Whether to display the plot interactively.
        
    Returns:
        Path to the saved plot
    """
    # Load results if not provided
    if results is None:
        results = load_validation_results()
    
    # Ensure output directory exists
    output_dir = ensure_output_dir(output_dir)
    
    # Prepare data
    scenarios = []
    perplexity = []
    diversity = []
    repetition = []
    
    for scenario, data in results.items():
        quality = data.get("quality", {})
        ppl = quality.get("perplexity")
        div = quality.get("lexical_diversity")
        rep = quality.get("repetition_score")
        
        if ppl is not None and div is not None and rep is not None:
            # Beautify scenario names for display
            label = scenario.replace("_", " ").title()
            scenarios.append(label)
            perplexity.append(ppl)
            diversity.append(div)
            repetition.append(rep)
    
    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
    
    # Perplexity (lower is better)
    sns.barplot(x=scenarios, y=perplexity, ax=ax1, palette=SCENARIO_PALETTE)
    ax1.set_title("Perplexity (Lower is Better)", fontsize=14)
    ax1.set_ylabel("Perplexity", fontsize=12)
    ax1.set_xlabel("Scenario", fontsize=12)
    ax1.tick_params(axis='x', rotation=15)
    
    # Add values on top of bars
    for i, v in enumerate(perplexity):
        ax1.text(i, v + 0.5, f"{v:.1f}", ha='center', fontsize=10)
    
    # Diversity (higher is better)
    sns.barplot(x=scenarios, y=diversity, ax=ax2, palette=SCENARIO_PALETTE)
    ax2.set_title("Diversity Score (Higher is Better)", fontsize=14)
    ax2.set_ylabel("Diversity", fontsize=12)
    ax2.set_xlabel("Scenario", fontsize=12)
    ax2.tick_params(axis='x', rotation=15)
    
    # Add values on top of bars
    for i, v in enumerate(diversity):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)
    
    # Repetition rate (lower is better)
    sns.barplot(x=scenarios, y=repetition, ax=ax3, palette=SCENARIO_PALETTE)
    ax3.set_title("Repetition Rate (Lower is Better)", fontsize=14)
    ax3.set_ylabel("Repetition Rate", fontsize=12)
    ax3.set_xlabel("Scenario", fontsize=12)
    ax3.tick_params(axis='x', rotation=15)
    
    # Add values on top of bars
    for i, v in enumerate(repetition):
        ax3.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=10)
    
    # Main title
    fig.suptitle("📝 Output Quality Metrics", fontsize=16)
    
    # Add insight annotation
    if len(diversity) > 1:
        best_div_idx = np.argmax(diversity)
        worst_div_idx = np.argmin(diversity)
        improvement = ((diversity[best_div_idx] / diversity[worst_div_idx]) - 1) * 100
        
        plt.figtext(0.5, 0.01, 
                   f"Insight: {scenarios[best_div_idx]} produces {improvement:.1f}% more diverse output than {scenarios[worst_div_idx]}",
                   ha="center", fontsize=10, style='italic')
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.15)
    
    # Save the plot
    output_path = output_dir / "quality_metrics.png"
    plt.savefig(output_path, dpi=150)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_path

def plot_agency_violations(results=None, output_dir=None, show=True):
    """
    Plot agency violations across different scenarios.
    
    Args:
        results: Validation results dictionary. If None, loads from file.
        output_dir: Directory to save the plot. If None, uses default location.
        show: Whether to display the plot interactively.
        
    Returns:
        Path to the saved plot
    """
    # Load results if not provided
    if results is None:
        results = load_validation_results()
    
    # Ensure output directory exists
    output_dir = ensure_output_dir(output_dir)
    
    # Prepare data
    scenarios = []
    violations = []
    
    for scenario, data in results.items():
        # Skip baseline which doesn't have agency data
        if scenario.lower() == "baseline":
            continue
            
        resource_usage = data.get("resource_usage", {})
        agency_report = resource_usage.get("agency_report", {})
        
        if agency_report:
            # Beautify scenario names for display
            label = scenario.replace("_", " ").title()
            scenarios.append(label)
            violations.append(agency_report.get("total_violations", 0))
    
    # Skip if no agency data
    if not scenarios:
        print("No agency violation data found.")
        return None
    
    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=scenarios, y=violations, palette=SCENARIO_PALETTE)
    
    # Use log scale if there's a large range of values
    if max(violations) > 100:
        plt.yscale('log')
    
    # Add values on top of bars
    for i, v in enumerate(violations):
        plt.text(i, v * 1.05, f"{v}", ha='center', fontsize=10)
    
    plt.title("🚨 Agency Violations by Scenario", fontsize=16)
    plt.ylabel("Number of Violations", fontsize=12)
    plt.xlabel("Scenario", fontsize=12)
    plt.xticks(rotation=15)
    
    # Add insight annotation
    if len(violations) > 1:
        min_idx = np.argmin(violations)
        max_idx = np.argmax(violations)
        
        plt.figtext(0.5, 0.01, 
                   f"Insight: {scenarios[min_idx]} has {violations[min_idx]} violations vs {violations[max_idx]} in {scenarios[max_idx]}",
                   ha="center", fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save the plot
    output_path = output_dir / "agency_violations.png"
    plt.savefig(output_path, dpi=150)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_path

def plot_head_state_distribution(results=None, output_dir=None, show=True):
    """
    Plot head state distribution across different scenarios.
    
    Args:
        results: Validation results dictionary. If None, loads from file.
        output_dir: Directory to save the plot. If None, uses default location.
        show: Whether to display the plot interactively.
        
    Returns:
        Path to the saved plot
    """
    # Load results if not provided
    if results is None:
        results = load_validation_results()
    
    # Ensure output directory exists
    output_dir = ensure_output_dir(output_dir)
    
    # Prepare data
    scenarios = []
    active = []
    overloaded = []
    misaligned = []
    withdrawn = []
    
    for scenario, data in results.items():
        # Skip baseline which doesn't have agency states
        if scenario.lower() == "baseline":
            continue
            
        # Check for resource_usage.agency_report
        resource_usage = data.get("resource_usage", {})
        agency_report = resource_usage.get("agency_report", {})
            
        if agency_report:
            # Beautify scenario names for display
            label = scenario.replace("_", " ").title()
            scenarios.append(label)
            active.append(agency_report.get("active_heads", 0))
            overloaded.append(agency_report.get("overloaded_heads", 0))
            misaligned.append(agency_report.get("misaligned_heads", 0))
            withdrawn.append(agency_report.get("withdrawn_heads", 0))
    
    # Skip if no agency data
    if not scenarios:
        print("No head state data found.")
        return None
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Create stacked bar chart
    labels = scenarios
    width = 0.7
    
    # Create bottom positions for stacking
    p1 = plt.bar(labels, active, width, label='Active', color=AGENCY_COLORS["active"])
    p2 = plt.bar(labels, misaligned, width, bottom=active, label='Misaligned', color=AGENCY_COLORS["misaligned"])
    
    # Calculate positions for overloaded bars
    overloaded_bottom = [a + m for a, m in zip(active, misaligned)]
    p3 = plt.bar(labels, overloaded, width, bottom=overloaded_bottom, label='Overloaded', color=AGENCY_COLORS["overloaded"])
    
    # Calculate positions for withdrawn bars
    withdrawn_bottom = [a + m + o for a, m, o in zip(active, misaligned, overloaded)]
    p4 = plt.bar(labels, withdrawn, width, bottom=withdrawn_bottom, label='Withdrawn', color=AGENCY_COLORS["withdrawn"])
    
    plt.ylabel('Head Count', fontsize=12)
    plt.title('🧠 Attention Head State Distribution', fontsize=16)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    
    # Add counts as labels
    for i, scenario in enumerate(scenarios):
        total = active[i] + misaligned[i] + overloaded[i] + withdrawn[i]
        
        # Only add percentages for non-zero values
        if active[i] > 0:
            active_pct = active[i] / total * 100
            plt.text(i, active[i]/2, f"{active_pct:.0f}%", ha='center', va='center', color='white', fontweight='bold')
        
        if misaligned[i] > 0:
            misaligned_pct = misaligned[i] / total * 100
            plt.text(i, active[i] + misaligned[i]/2, f"{misaligned_pct:.0f}%", ha='center', va='center', color='white', fontweight='bold')
        
        if overloaded[i] > 0:
            overloaded_pct = overloaded[i] / total * 100
            plt.text(i, active[i] + misaligned[i] + overloaded[i]/2, f"{overloaded_pct:.0f}%", ha='center', va='center', color='white', fontweight='bold')
        
        if withdrawn[i] > 0:
            withdrawn_pct = withdrawn[i] / total * 100
            plt.text(i, active[i] + misaligned[i] + overloaded[i] + withdrawn[i]/2, f"{withdrawn_pct:.0f}%", ha='center', va='center', color='white', fontweight='bold')
    
    # Add insight annotation
    if scenarios:
        max_withdrawal_idx = np.argmax(withdrawn)
        min_withdrawal_idx = np.argmin(withdrawn)
        scenario_with_most_withdrawn = scenarios[max_withdrawal_idx]
        withdrawn_pct = withdrawn[max_withdrawal_idx] / (active[max_withdrawal_idx] + misaligned[max_withdrawal_idx] + overloaded[max_withdrawal_idx] + withdrawn[max_withdrawal_idx]) * 100
        
        plt.figtext(0.5, 0.01, 
                   f"Insight: In the {scenario_with_most_withdrawn} scenario, {withdrawn_pct:.1f}% of heads enter withdrawn state",
                   ha="center", fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    # Save the plot
    output_path = output_dir / "head_state_distribution.png"
    plt.savefig(output_path, dpi=150)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_path

def plot_generation_time(results=None, output_dir=None, show=True):
    """
    Plot generation time comparison across different scenarios.
    
    Args:
        results: Validation results dictionary. If None, loads from file.
        output_dir: Directory to save the plot. If None, uses default location.
        show: Whether to display the plot interactively.
        
    Returns:
        Path to the saved plot
    """
    # Load results if not provided
    if results is None:
        results = load_validation_results()
    
    # Ensure output directory exists
    output_dir = ensure_output_dir(output_dir)
    
    # Prepare data
    scenarios = []
    generation_times = []
    
    for scenario, data in results.items():
        resource_usage = data.get("resource_usage", {})
        gen_time = resource_usage.get("generation_time")
        if gen_time is not None:
            # Beautify scenario names for display
            label = scenario.replace("_", " ").title()
            scenarios.append(label)
            generation_times.append(gen_time)
    
    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=scenarios, y=generation_times, palette=SCENARIO_PALETTE)
    
    # Add values on top of bars
    for i, v in enumerate(generation_times):
        ax.text(i, v + 0.1, f"{v:.2f}s", ha='center', fontsize=10)
    
    plt.title("⏱️ Generation Time Across Agency Configurations", fontsize=16)
    plt.ylabel("Generation Time (seconds)", fontsize=12)
    plt.xlabel("Scenario", fontsize=12)
    plt.xticks(rotation=15)
    
    # Add insight annotation
    if len(generation_times) > 1:
        best_idx = np.argmin(generation_times)
        worst_idx = np.argmax(generation_times)
        improvement = ((generation_times[worst_idx] / generation_times[best_idx]) - 1) * 100
        
        plt.figtext(0.5, 0.01, 
                   f"Insight: {scenarios[best_idx]} is {improvement:.1f}% faster than {scenarios[worst_idx]}",
                   ha="center", fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = output_dir / "generation_time_comparison.png"
    plt.savefig(output_path, dpi=150)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_path

def plot_efficiency_vs_quality(results=None, output_dir=None, show=True):
    """
    Plot the tradeoff between efficiency and quality.
    
    Args:
        results: Validation results dictionary. If None, loads from file.
        output_dir: Directory to save the plot. If None, uses default location.
        show: Whether to display the plot interactively.
        
    Returns:
        Path to the saved plot
    """
    # Load results if not provided
    if results is None:
        results = load_validation_results()
    
    # Ensure output directory exists
    output_dir = ensure_output_dir(output_dir)
    
    # Prepare data
    scenarios = []
    performance = []
    quality = []
    resources = []
    
    for scenario, data in results.items():
        inference = data.get("inference", {})
        quality_data = data.get("quality", {})
        resource_data = data.get("resources", {})
        
        speed = inference.get("tokens_per_second")
        ppl = quality_data.get("perplexity")
        memory = resource_data.get("memory_gb")
        
        if speed is not None and ppl is not None and memory is not None:
            # Beautify scenario names for display
            label = scenario.replace("_", " ").title()
            scenarios.append(label)
            performance.append(speed)
            
            # Quality score = 1/perplexity (higher is better)
            quality_score = 1 / ppl * 20  # Scale for visibility
            quality.append(quality_score)
            
            # Resource efficiency = 1/memory (higher is better)
            resource_efficiency = 1 / memory * 10  # Scale for visibility
            resources.append(resource_efficiency)
    
    # Skip if not enough data
    if len(scenarios) < 2:
        print("Not enough data for efficiency vs quality plot.")
        return None
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot with size representing resource efficiency
    for i, scenario in enumerate(scenarios):
        plt.scatter(quality[i], performance[i], s=resources[i]*100, 
                   label=scenario, alpha=0.7)
    
    plt.xlabel('Quality Score (Higher is Better)', fontsize=12)
    plt.ylabel('Generation Speed (Tokens/sec)', fontsize=12)
    plt.title('⚖️ Efficiency vs Quality Tradeoff', fontsize=16)
    
    # Add scenario labels to points
    for i, scenario in enumerate(scenarios):
        plt.annotate(scenario, (quality[i], performance[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    # Add "better" direction indicator
    plt.annotate('Better', xy=(0.85, 0.9), xycoords='axes fraction',
                xytext=(0.7, 0.8), textcoords='axes fraction',
                arrowprops=dict(facecolor='green', shrink=0.05, width=2),
                fontsize=12)
    
    # Find the Pareto frontier (best quality/performance tradeoff)
    pareto_points = []
    for i in range(len(scenarios)):
        is_pareto = True
        for j in range(len(scenarios)):
            if i != j:
                if quality[j] >= quality[i] and performance[j] >= performance[i]:
                    is_pareto = False
                    break
        if is_pareto:
            pareto_points.append(i)
    
    # Highlight Pareto frontier
    pareto_quality = [quality[i] for i in pareto_points]
    pareto_perf = [performance[i] for i in pareto_points]
    pareto_scenarios = [scenarios[i] for i in pareto_points]
    
    # Sort pareto points by quality
    pareto_sorted = sorted(zip(pareto_quality, pareto_perf, pareto_scenarios))
    pareto_quality = [x[0] for x in pareto_sorted]
    pareto_perf = [x[1] for x in pareto_sorted]
    
    plt.plot(pareto_quality, pareto_perf, 'r--', label='Pareto Frontier')
    
    # Add legend for resource efficiency
    plt.subplots_adjust(bottom=0.15)
    
    # Insight annotation
    if pareto_points:
        best_idx = pareto_points[np.argmax([performance[i] for i in pareto_points])]
        plt.figtext(0.5, 0.01, 
                   f"Insight: {scenarios[best_idx]} offers the optimal balance of quality and speed",
                   ha="center", fontsize=10, style='italic')
    
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save the plot
    output_path = output_dir / "efficiency_vs_quality.png"
    plt.savefig(output_path, dpi=150)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_path

def generate_all_charts(results_path=None, output_dir=None, show=False):
    """
    Generate all charts for the validation results.
    
    Args:
        results_path: Path to the results JSON file. If None, searches in common locations.
        output_dir: Directory to save the plots. If None, uses default location.
        show: Whether to display the plots interactively.
        
    Returns:
        Dictionary mapping chart names to output paths
    """
    # Load results
    try:
        results = load_validation_results(results_path)
    except FileNotFoundError:
        print("Could not find validation results file. Please provide a valid path.")
        return {}
    
    # Ensure output directory exists
    output_dir = ensure_output_dir(output_dir)
    
    # Generate all charts
    charts = {}
    
    print("Generating charts...")
    
    charts["generation_speed"] = plot_generation_speed(results, output_dir, show)
    print("✓ Generation speed chart")
    
    charts["generation_time"] = plot_generation_time(results, output_dir, show)
    print("✓ Generation time chart")
    
    charts["resource_usage"] = plot_resource_usage(results, output_dir, show)
    print("✓ Resource usage chart")
    
    charts["quality_metrics"] = plot_quality_metrics(results, output_dir, show)
    print("✓ Quality metrics chart")
    
    charts["agency_violations"] = plot_agency_violations(results, output_dir, show)
    if charts["agency_violations"]:
        print("✓ Agency violations chart")
    else:
        print("✗ Agency violations chart (insufficient data)")
    
    charts["head_state_distribution"] = plot_head_state_distribution(results, output_dir, show)
    if charts["head_state_distribution"]:
        print("✓ Head state distribution chart")
    else:
        print("✗ Head state distribution chart (insufficient data)")
    
    charts["efficiency_vs_quality"] = plot_efficiency_vs_quality(results, output_dir, show)
    if charts["efficiency_vs_quality"]:
        print("✓ Efficiency vs quality chart")
    else:
        print("✗ Efficiency vs quality chart (insufficient data)")
    
    print(f"\nAll charts saved to {output_dir}/")
    
    return {k: v for k, v in charts.items() if v is not None}

if __name__ == "__main__":
    # Generate all charts when run as a script
    generate_all_charts(show=True)