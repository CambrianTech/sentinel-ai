"""
Charting utilities for Sentinel-AI.

This module provides a consistent color scheme and plotting utilities
for visualizing experimental results and agency states.
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

def set_chart_style():
    """Set the default chart style for consistency."""
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.family'] = 'sans-serif'
    
def create_comparison_barplot(baseline_values, agency_values, labels, 
                             title="Comparison", ylabel="Value", 
                             baseline_label="Baseline", agency_label="Agency",
                             show_improvement=True, higher_is_better=True):
    """
    Create a bar plot comparing baseline and agency values.
    
    Args:
        baseline_values: List of values for baseline model
        agency_values: List of values for agency model
        labels: Labels for each pair of bars
        title: Chart title
        ylabel: Y-axis label
        baseline_label: Label for baseline model
        agency_label: Label for agency model
        show_improvement: Whether to show improvement percentages
        higher_is_better: Whether higher values are better
        
    Returns:
        The matplotlib figure
    """
    set_chart_style()
    fig, ax = plt.subplots()
    
    x = np.arange(len(labels))
    width = 0.35
    
    baseline_bars = ax.bar(x - width/2, baseline_values, width, 
                         label=baseline_label, color=AGENCY_COLORS["baseline"])
    agency_bars = ax.bar(x + width/2, agency_values, width, 
                        label=agency_label, color=AGENCY_COLORS["active"])
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add improvement annotations if requested
    if show_improvement:
        for i in range(len(labels)):
            baseline_val = baseline_values[i]
            agency_val = agency_values[i]
            
            if baseline_val == 0:
                continue  # Avoid division by zero
                
            if higher_is_better:
                improvement = ((agency_val / baseline_val) - 1) * 100
            else:
                improvement = ((baseline_val / agency_val) - 1) * 100
                
            ax.annotate(f"{improvement:.1f}%",
                       xy=(x[i] + width/2, agency_val),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_line_plot(x_values, y_values_dict, title="Line Plot", 
                    xlabel="X", ylabel="Y", add_legend=True,
                    colors=None):
    """
    Create a line plot with multiple series.
    
    Args:
        x_values: X-axis values
        y_values_dict: Dictionary mapping series names to y values
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        add_legend: Whether to add a legend
        colors: Dictionary mapping series names to colors
        
    Returns:
        The matplotlib figure
    """
    set_chart_style()
    fig, ax = plt.subplots()
    
    for i, (name, values) in enumerate(y_values_dict.items()):
        color = colors.get(name) if colors else None
        ax.plot(x_values, values, 'o-', label=name, linewidth=2, color=color)
        
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if add_legend:
        ax.legend()
        
    plt.tight_layout()
    return fig

def create_heatmap(data, row_labels, col_labels, title="Heatmap",
                 cmap="RdYlGn", center=None, annot=True):
    """
    Create a heatmap visualization.
    
    Args:
        data: 2D array or list of lists with values
        row_labels: Labels for rows
        col_labels: Labels for columns
        title: Chart title
        cmap: Colormap name
        center: Center value for diverging colormaps
        annot: Whether to annotate cells with values
        
    Returns:
        The matplotlib figure
    """
    set_chart_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(data, annot=annot, fmt=".1f", cmap=cmap,
              xticklabels=col_labels, yticklabels=row_labels,
              center=center, ax=ax)
    
    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    return fig

def create_radar_chart(categories, values_dict, title="Radar Chart", colors=None):
    """
    Create a radar chart comparing multiple series.
    
    Args:
        categories: List of category names
        values_dict: Dictionary mapping series names to values
        title: Chart title
        colors: Dictionary mapping series names to colors
        
    Returns:
        The matplotlib figure
    """
    set_chart_style()
    
    # Number of variables
    N = len(categories)
    
    # What will be the angle of each axis in the plot (divide the plot / number of variables)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Initialize the figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw the series
    for i, (name, values) in enumerate(values_dict.items()):
        values = list(values)
        values += values[:1]  # Close the loop
        color = colors.get(name) if colors else None
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    # Add title and legend
    plt.title(title, size=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    return fig

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
    
    charts["quality_metrics"] = plot_quality_metrics(results, output_dir, show)
    print("✓ Quality metrics chart")
    
    print(f"\nAll charts saved to {output_dir}/")
    
    return {k: v for k, v in charts.items() if v is not None}

if __name__ == "__main__":
    # Generate all charts when run as a script
    generate_all_charts(show=True)