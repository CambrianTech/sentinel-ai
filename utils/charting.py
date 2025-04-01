"""
Charting utilities for Sentinel-AI.

This module provides a consistent color scheme and plotting utilities
for visualizing experimental results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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