"""
Adaptive Plasticity Visualization

This module provides visualization tools for the adaptive plasticity system,
showing gradient norms, pruning/revival decisions, and optimization progress
across multiple cycles.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime

def plot_head_gradient_with_status(
    grad_norms: Union[np.ndarray, "torch.Tensor"],
    pruned_heads: List[Tuple[int, int]] = None,
    revived_heads: List[Tuple[int, int]] = None,
    vulnerable_heads: List[Tuple[int, int]] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Head Gradient Norms with Plasticity Status",
    save_path: Optional[str] = None,
    cmap: str = "plasma",
    colorbar_label: str = "Gradient Norm",
    vulnerable_threshold: float = 0.01
) -> plt.Figure:
    """
    Generate a visualization of head gradient norms with pruning/revival markers.
    
    Args:
        grad_norms: Gradient norm array or tensor with shape [layer, head]
        pruned_heads: List of (layer_idx, head_idx) tuples of pruned heads
        revived_heads: List of (layer_idx, head_idx) tuples of revived heads
        vulnerable_heads: List of (layer_idx, head_idx) tuples of vulnerable heads
        figsize: Figure size (width, height) in inches
        title: Plot title
        save_path: Optional path to save the figure
        cmap: Colormap name for the heatmap
        colorbar_label: Label for the colorbar
        vulnerable_threshold: Threshold below which heads are considered vulnerable
        
    Returns:
        Matplotlib figure object
    """
    # Convert to numpy if it's a tensor
    if hasattr(grad_norms, 'detach') and hasattr(grad_norms, 'cpu') and hasattr(grad_norms, 'numpy'):
        grad_norms = grad_norms.detach().cpu().numpy()
    
    # Create figure and plot gradient norms
    fig, ax = plt.subplots(figsize=figsize)
    heatmap = ax.imshow(grad_norms, cmap=cmap)
    
    # Add colorbar for gradient norms
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label(colorbar_label)
    
    # Create empty lists if not provided
    pruned_heads = pruned_heads or []
    revived_heads = revived_heads or []
    vulnerable_heads = vulnerable_heads or []
    
    # Auto-detect vulnerable heads if not provided
    if not vulnerable_heads and vulnerable_threshold is not None:
        for layer_idx in range(grad_norms.shape[0]):
            for head_idx in range(grad_norms.shape[1]):
                # Skip heads that are already pruned or revived
                if (layer_idx, head_idx) in pruned_heads or (layer_idx, head_idx) in revived_heads:
                    continue
                    
                # Check if gradient norm is below threshold
                if grad_norms[layer_idx, head_idx] < vulnerable_threshold:
                    vulnerable_heads.append((layer_idx, head_idx))
    
    # Add markers for pruned heads (red X)
    for layer_idx, head_idx in pruned_heads:
        ax.text(head_idx, layer_idx, "❌", ha="center", va="center", color="white", fontsize=12)
    
    # Add markers for revived heads (green plus)
    for layer_idx, head_idx in revived_heads:
        ax.text(head_idx, layer_idx, "➕", ha="center", va="center", color="white", fontsize=12)
    
    # Add markers for vulnerable heads (yellow warning)
    for layer_idx, head_idx in vulnerable_heads:
        ax.text(head_idx, layer_idx, "⚠️", ha="center", va="center", color="white", fontsize=10)
    
    # Set labels and title
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")
    ax.set_title(title)
    
    # Add grid lines
    ax.set_xticks(np.arange(-.5, grad_norms.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, grad_norms.shape[0], 1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Set major ticks
    ax.set_xticks(np.arange(0, grad_norms.shape[1], 1))
    ax.set_yticks(np.arange(0, grad_norms.shape[0], 1))
    
    # Add Legend
    import matplotlib.patches as mpatches
    
    legend_elements = []
    if pruned_heads:
        legend_elements.append(mpatches.Patch(color='red', alpha=0.7, label='Pruned Head (❌)'))
    if revived_heads:
        legend_elements.append(mpatches.Patch(color='green', alpha=0.7, label='Revived Head (➕)'))
    if vulnerable_heads:
        legend_elements.append(mpatches.Patch(color='yellow', alpha=0.7, label='Vulnerable Head (⚠️)'))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.7)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_cycles_comparison(
    cycle_metrics: List[Dict[str, Any]],
    baseline_perplexity: float,
    baseline_head_count: int,
    total_heads: int,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison of metrics across multiple plasticity cycles.
    
    Args:
        cycle_metrics: List of cycle metrics dictionaries
        baseline_perplexity: Baseline perplexity before optimization
        baseline_head_count: Baseline active head count
        total_heads: Total number of heads in the model
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Extract data for plotting
    cycles = [m.get("cycle", i+1) for i, m in enumerate(cycle_metrics)]
    pruned_perplexities = [m.get("pruned", {}).get("perplexity", None) for m in cycle_metrics]
    grown_perplexities = [m.get("grown", {}).get("perplexity", None) for m in cycle_metrics]
    final_perplexities = [m.get("final", {}).get("perplexity", None) for m in cycle_metrics]
    
    head_counts = [m.get("final", {}).get("head_count", None) for m in cycle_metrics]
    head_percentages = [count/total_heads*100 if count is not None else None for count in head_counts]
    
    # Calculate efficiency
    baseline_efficiency = baseline_perplexity / baseline_head_count
    efficiencies = [perp/count if perp is not None and count is not None else None 
                   for perp, count in zip(final_perplexities, head_counts)]
    efficiency_improvements = [(baseline_efficiency - eff)/baseline_efficiency*100 
                             if eff is not None else None for eff in efficiencies]
    
    # Plot 1: Perplexity through cycles
    ax1.plot(cycles, pruned_perplexities, 'r--o', label="After Pruning", alpha=0.7)
    ax1.plot(cycles, grown_perplexities, 'b--^', label="After Growth", alpha=0.7)
    ax1.plot(cycles, final_perplexities, 'g-s', label="After Learning", linewidth=2)
    ax1.axhline(y=baseline_perplexity, color='k', linestyle='-', alpha=0.5, label="Baseline")
    
    ax1.set_title("Perplexity Through Optimization Cycles")
    ax1.set_ylabel("Perplexity")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Active heads percentage
    ax2.plot(cycles, head_percentages, 'b-o')
    ax2.axhline(y=baseline_head_count/total_heads*100, color='k', linestyle='-', alpha=0.5, label="Baseline")
    
    ax2.set_title("Active Heads Through Optimization Cycles")
    ax2.set_ylabel("Active Heads (%)")
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Efficiency improvement
    ax3.plot(cycles, efficiency_improvements, 'g-o')
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    
    ax3.set_title("Efficiency Improvement Through Optimization Cycles")
    ax3.set_xlabel("Cycle")
    ax3.set_ylabel("Efficiency Improvement (%)")
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_plasticity_history(
    history: Dict[str, List[Dict[str, Any]]],
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot historical evolution of head importance and plasticity decisions.
    
    Args:
        history: Dictionary of head importance history
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Determine number of layers and heads
    layers = set()
    heads = set()
    
    for key in history.keys():
        if isinstance(key, str) and "_" in key:
            parts = key.split("_")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                layers.add(int(parts[0]))
                heads.add(int(parts[1]))
    
    if not layers or not heads:
        raise ValueError("No valid layer/head data found in history")
    
    num_layers = max(layers) + 1
    num_heads = max(heads) + 1
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Create matrices for plotting
    pruned_counts = np.zeros((num_layers, num_heads))
    grown_counts = np.zeros((num_layers, num_heads))
    
    for key, data in history.items():
        if isinstance(key, str) and "_" in key:
            parts = key.split("_")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                layer_idx = int(parts[0])
                head_idx = int(parts[1])
                pruned_counts[layer_idx, head_idx] = data.get("pruned_count", 0)
                grown_counts[layer_idx, head_idx] = data.get("grown_count", 0)
    
    # Plot 1: Pruning frequency heatmap
    im1 = ax1.imshow(pruned_counts, cmap="Reds")
    ax1.set_title("Head Pruning Frequency")
    ax1.set_xlabel("Head Index")
    ax1.set_ylabel("Layer Index")
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("Times Pruned")
    
    # Add text annotations for pruning counts
    for i in range(num_layers):
        for j in range(num_heads):
            count = pruned_counts[i, j]
            if count > 0:
                ax1.text(j, i, int(count), ha="center", va="center", 
                         color="white" if count > 2 else "black", fontsize=8)
    
    # Plot 2: Growth frequency heatmap
    im2 = ax2.imshow(grown_counts, cmap="Greens")
    ax2.set_title("Head Revival Frequency")
    ax2.set_xlabel("Head Index")
    ax2.set_ylabel("Layer Index")
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("Times Revived")
    
    # Add text annotations for growth counts
    for i in range(num_layers):
        for j in range(num_heads):
            count = grown_counts[i, j]
            if count > 0:
                ax2.text(j, i, int(count), ha="center", va="center", 
                         color="white" if count > 2 else "black", fontsize=8)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_strategy_effectiveness(
    strategy_results: Dict[str, Dict[str, Any]],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the effectiveness of different pruning and growth strategies.
    
    Args:
        strategy_results: Dictionary of strategy effectiveness metrics
        figsize: Figure size (width, height) in inches
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Separate pruning and growth strategies
    pruning_strategies = {k.replace("pruning_", ""): v for k, v in strategy_results.items() 
                         if k.startswith("pruning_")}
    growth_strategies = {k.replace("growth_", ""): v for k, v in strategy_results.items() 
                        if k.startswith("growth_")}
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot pruning strategies
    if pruning_strategies:
        names = list(pruning_strategies.keys())
        success_rates = [v.get("success_rate", 0) * 100 for v in pruning_strategies.values()]
        perplexity_changes = [v.get("avg_perplexity_change", 0) * 100 for v in pruning_strategies.values()]
        trials = [v.get("trials", 0) for v in pruning_strategies.values()]
        
        # Create bar width based on number of strategies
        width = 0.35
        x = np.arange(len(names))
        
        # Create bars
        bars1 = ax1.bar(x - width/2, success_rates, width, label='Success Rate (%)', color='green', alpha=0.7)
        bars2 = ax1.bar(x + width/2, perplexity_changes, width, label='Perplexity Change (%)', color='red', alpha=0.7)
        
        # Add trial count as text
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 3,
                    f"{trials[i]} trials", ha='center', va='bottom', fontsize=8)
        
        ax1.set_title("Pruning Strategy Effectiveness")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names)
        ax1.set_ylabel("Percentage")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)
    
    # Plot growth strategies
    if growth_strategies:
        names = list(growth_strategies.keys())
        success_rates = [v.get("success_rate", 0) * 100 for v in growth_strategies.values()]
        perplexity_changes = [v.get("avg_perplexity_change", 0) * 100 for v in growth_strategies.values()]
        trials = [v.get("trials", 0) for v in growth_strategies.values()]
        
        # Create bar width based on number of strategies
        width = 0.35
        x = np.arange(len(names))
        
        # Create bars
        bars1 = ax2.bar(x - width/2, success_rates, width, label='Success Rate (%)', color='green', alpha=0.7)
        bars2 = ax2.bar(x + width/2, perplexity_changes, width, label='Perplexity Change (%)', color='red', alpha=0.7)
        
        # Add trial count as text
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            ax2.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 3,
                    f"{trials[i]} trials", ha='center', va='bottom', fontsize=8)
        
        ax2.set_title("Growth Strategy Effectiveness")
        ax2.set_xticks(x)
        ax2.set_xticklabels(names)
        ax2.set_ylabel("Percentage")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_dashboard(
    optimization_results: Dict[str, Any],
    output_dir: str,
    baseline_params: Any = None,
    current_params: Any = None,
    model_name: str = "Unknown Model",
    dataset_name: str = "Unknown Dataset"
) -> str:
    """
    Create a comprehensive dashboard of optimization results.
    
    Args:
        optimization_results: Dictionary of optimization results
        output_dir: Directory to save the dashboard
        baseline_params: Optional baseline model parameters
        current_params: Optional current model parameters
        model_name: Name of the model
        dataset_name: Name of the dataset
        
    Returns:
        Path to the dashboard directory
    """
    # Create dashboard directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dashboard_dir = os.path.join(output_dir, f"dashboard_{timestamp}")
    os.makedirs(dashboard_dir, exist_ok=True)
    
    # Create cycle comparison visualization
    if "cycle_results" in optimization_results:
        cycle_metrics = optimization_results["cycle_results"]
        baseline_perplexity = optimization_results.get("baseline", {}).get("perplexity", 0)
        baseline_head_count = optimization_results.get("baseline", {}).get("head_count", 0)
        total_heads = baseline_head_count  # Assuming we start with all heads
        
        cycles_path = os.path.join(dashboard_dir, "cycles_comparison.png")
        plot_cycles_comparison(
            cycle_metrics=cycle_metrics,
            baseline_perplexity=baseline_perplexity,
            baseline_head_count=baseline_head_count,
            total_heads=total_heads,
            save_path=cycles_path
        )
    
    # Create strategy effectiveness visualization
    if "strategy_effectiveness" in optimization_results:
        strategy_path = os.path.join(dashboard_dir, "strategy_effectiveness.png")
        plot_strategy_effectiveness(
            strategy_results=optimization_results["strategy_effectiveness"],
            save_path=strategy_path
        )
    
    # Create HTML report
    html_path = os.path.join(dashboard_dir, "dashboard.html")
    with open(html_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Adaptive Plasticity Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f5f5f5; padding: 20px; margin-bottom: 20px; }}
        .section {{ margin-bottom: 30px; }}
        .metrics {{ display: flex; flex-wrap: wrap; }}
        .metric {{ background-color: #f9f9f9; margin: 10px; padding: 15px; border-radius: 5px; width: 200px; }}
        .metric h3 {{ margin-top: 0; }}
        .chart {{ margin: 20px 0; }}
        .chart img {{ max-width: 100%; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Adaptive Plasticity Optimization Dashboard</h1>
        <p>Model: {model_name}</p>
        <p>Dataset: {dataset_name}</p>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <div class="section">
        <h2>Summary</h2>
        <div class="metrics">
            <div class="metric">
                <h3>Cycles</h3>
                <p>Completed: {optimization_results.get('cycles_completed', 'N/A')}</p>
                <p>Successful: {optimization_results.get('successful_cycles', 'N/A')}</p>
                <p>Early stopped: {'Yes' if optimization_results.get('early_stopped', False) else 'No'}</p>
            </div>
            <div class="metric">
                <h3>Perplexity</h3>
                <p>Baseline: {optimization_results.get('baseline', {}).get('perplexity', 'N/A'):.2f}</p>
                <p>Final: {optimization_results.get('final', {}).get('perplexity', 'N/A'):.2f}</p>
                <p>Improvement: {optimization_results.get('perplexity_improvement', 0)*100:+.1f}%</p>
            </div>
            <div class="metric">
                <h3>Active Heads</h3>
                <p>Baseline: {optimization_results.get('baseline', {}).get('head_count', 'N/A')}</p>
                <p>Final: {optimization_results.get('final', {}).get('head_count', 'N/A')}</p>
                <p>Reduction: {optimization_results.get('head_reduction', 0)*100:.1f}%</p>
            </div>
            <div class="metric">
                <h3>Efficiency</h3>
                <p>Baseline: {optimization_results.get('baseline', {}).get('efficiency', 'N/A'):.6f}</p>
                <p>Final: {optimization_results.get('final', {}).get('efficiency', 'N/A'):.6f}</p>
                <p>Improvement: {optimization_results.get('efficiency_improvement', 0)*100:+.1f}%</p>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Optimization Progress</h2>
        <div class="chart">
            <img src="cycles_comparison.png" alt="Cycles Comparison">
        </div>
    </div>
    
    <div class="section">
        <h2>Strategy Effectiveness</h2>
        <div class="chart">
            <img src="strategy_effectiveness.png" alt="Strategy Effectiveness">
        </div>
    </div>
    
    <div class="section">
        <h2>Cycle Details</h2>
        <table>
            <tr>
                <th>Cycle</th>
                <th>Pruning Strategy</th>
                <th>Growth Strategy</th>
                <th>Initial Perplexity</th>
                <th>Pruned Perplexity</th>
                <th>Final Perplexity</th>
                <th>Head Count</th>
                <th>Success</th>
            </tr>
""")
        
        # Add cycle details if available
        for i, cycle in enumerate(optimization_results.get("cycle_results", [])):
            success = cycle.get("success", False)
            bg_color = "#e6ffe6" if success else "#ffe6e6"  # Green if success, red if failure
            
            f.write(f"""
            <tr style="background-color: {bg_color}">
                <td>{i+1}</td>
                <td>{cycle.get("pruning_strategy", "N/A")}</td>
                <td>{cycle.get("growth_strategy", "N/A")}</td>
                <td>{cycle.get("initial", {}).get("perplexity", "N/A"):.2f}</td>
                <td>{cycle.get("pruned", {}).get("perplexity", "N/A"):.2f}</td>
                <td>{cycle.get("final", {}).get("perplexity", "N/A"):.2f}</td>
                <td>{cycle.get("final", {}).get("head_count", "N/A")}</td>
                <td>{"✓" if success else "✗"}</td>
            </tr>""")
        
        f.write("""
        </table>
    </div>
    
    <div class="footer">
        <p>Generated by Sentinel AI Adaptive Plasticity System</p>
    </div>
</body>
</html>""")
    
    return dashboard_dir