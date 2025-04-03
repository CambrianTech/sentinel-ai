"""
Visualization utilities for pruning experiments.

This module provides reusable visualization functions for analyzing and
comparing pruning experiment results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path


def plot_experiment_summary(results_df: pd.DataFrame, figsize: tuple = (12, 10)) -> plt.Figure:
    """
    Create a comprehensive visualization of experiment results.
    
    Args:
        results_df: DataFrame containing experiment results
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib figure object
    """
    if results_df.empty:
        logging.warning("No data available for plotting")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data available for plotting", 
                ha="center", va="center", fontsize=14)
        return fig
    
    # Reset plot parameters to avoid contamination from previous plots
    plt.rcParams.update(plt.rcParamsDefault)
    
    # Set better plot styling - Use standard layout, not constrained layout
    plt.rcParams.update({
        'figure.figsize': figsize,
        'figure.titlesize': 14,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'font.family': 'sans-serif'
    })
    
    # Create figure with subplots explicitly
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Get unique models and strategies
    models = results_df["model"].unique()
    strategies = results_df["strategy"].unique()
    
    # For display, shorten model names
    model_display = {m: m.split('/')[-1] if '/' in m else m for m in models}
    
    # 1. Perplexity across stages (top-left)
    ax1 = axs[0, 0]
    
    # Filter to main stages
    stages_df = results_df[results_df["stage"].isin(["baseline", "pruned", "fine_tuned"])]
    
    # Plot lines connecting stages for each experiment
    for model in models:
        model_df = stages_df[stages_df["model"] == model]
        
        for strategy in strategies:
            strategy_df = model_df[model_df["strategy"] == strategy]
            
            for pruning_level in strategy_df["pruning_level"].unique():
                experiment_df = strategy_df[strategy_df["pruning_level"] == pruning_level]
                
                # Sort by stage to ensure correct order
                stage_order = {"baseline": 0, "pruned": 1, "fine_tuned": 2}
                experiment_df = experiment_df.sort_values(by="stage", key=lambda x: x.map(stage_order))
                
                # Plot if we have at least two stages
                if len(experiment_df) >= 2:
                    label = f"{model_display[model][:6]}-{strategy[:3]}-{pruning_level:.1f}"
                    ax1.plot(experiment_df["stage"], experiment_df["perplexity"], "o-", label=label)
    
    ax1.set_title("Perplexity Across Stages")
    ax1.set_xlabel("Stage")
    ax1.set_ylabel("Perplexity")
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(fontsize=7, loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # 2. Recovery percentage vs pruning level (top-right)
    ax2 = axs[0, 1]
    
    # Get data with recovery information
    recovery_df = results_df[results_df["stage"] == "fine_tuned"].copy()
    
    if not recovery_df.empty:
        # Create recovery column (combining both metrics)
        recovery_df["recovery"] = recovery_df["recovery_percentage"]
        # If improvement percentage exists and recovery is NaN, use negative of improvement
        mask = recovery_df["recovery"].isna() & recovery_df["improvement_percentage"].notna()
        recovery_df.loc[mask, "recovery"] = -recovery_df.loc[mask, "improvement_percentage"]
        
        # Plot by strategy
        for strategy in strategies:
            strategy_df = recovery_df[recovery_df["strategy"] == strategy]
            if not strategy_df.empty:
                for model in models:
                    model_strategy_df = strategy_df[strategy_df["model"] == model]
                    if not model_strategy_df.empty:
                        # Sort by pruning level
                        model_strategy_df = model_strategy_df.sort_values("pruning_level")
                        ax2.plot(model_strategy_df["pruning_level"], model_strategy_df["recovery"], 
                                "o-", label=f"{model_display[model][:6]}-{strategy[:3]}")
        
        ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax2.axhline(y=100, color="g", linestyle="--", alpha=0.3)
        
        # Add labels at positions that don't interfere with data
        # Move "Full Recovery" label closer to y-axis
        ax2.text(0.05, 95, "Full Recovery", color="green", ha="left", va="top", fontsize=8)
        ax2.text(0.05, 5, "Improvement", color="blue", ha="left", va="bottom", fontsize=8)
        
        ax2.set_title("Recovery/Improvement by Pruning Level")
        ax2.set_xlabel("Pruning Level")
        ax2.set_ylabel("% (negative = improvement)")
        ax2.legend(fontsize=7, loc='best')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No recovery data available yet", 
                ha="center", va="center", fontsize=12)
    
    # 3. Perplexity change: pruning vs fine-tuning effect (bottom-left)
    ax3 = axs[1, 0]
    
    if "perplexity_change" in results_df.columns and "perplexity_change_from_pruned" in results_df.columns:
        # Get pruning change
        pruned_df = results_df[results_df["stage"] == "pruned"].copy()
        pruned_df = pruned_df[["model", "strategy", "pruning_level", "perplexity_change"]]
        
        # Get fine-tuning change
        finetuned_df = results_df[results_df["stage"] == "fine_tuned"].copy()
        finetuned_df = finetuned_df[["model", "strategy", "pruning_level", "perplexity_change_from_pruned"]]
        
        # Merge
        effects_df = pd.merge(
            pruned_df, finetuned_df,
            on=["model", "strategy", "pruning_level"],
            suffixes=("_pruning", "_finetuning")
        )
        
        if not effects_df.empty:
            # Plot scatter with size based on pruning level
            for strategy in strategies:
                strategy_df = effects_df[effects_df["strategy"] == strategy]
                if not strategy_df.empty:
                    for model in models:
                        model_df = strategy_df[strategy_df["model"] == model]
                        if not model_df.empty:
                            ax3.scatter(
                                model_df["perplexity_change"], 
                                model_df["perplexity_change_from_pruned"],
                                s=model_df["pruning_level"] * 300,  # Size based on pruning level
                                label=f"{model_display[model][:6]}-{strategy[:3]}",
                                alpha=0.7
                            )
            
            ax3.axhline(y=0, color="k", linestyle="--", alpha=0.3)
            ax3.axvline(x=0, color="k", linestyle="--", alpha=0.3)
            
            # Add quadrant labels (smaller font, closer to axes)
            ax3.text(-2, -2, "Both improved", fontsize=8, ha="center", va="center",
                    bbox=dict(facecolor="lightgreen", alpha=0.5))
            ax3.text(2, -2, "Pruning hurt,\nFine-tuning fixed", fontsize=8, ha="center", va="center",
                    bbox=dict(facecolor="lightblue", alpha=0.5))
            ax3.text(-2, 2, "Pruning helped,\nFine-tuning hurt", fontsize=8, ha="center", va="center",
                    bbox=dict(facecolor="lightyellow", alpha=0.5))
            ax3.text(2, 2, "Both hurt", fontsize=8, ha="center", va="center",
                    bbox=dict(facecolor="lightcoral", alpha=0.5))
            
            ax3.set_title("Effect of Pruning vs. Fine-tuning")
            ax3.set_xlabel("Perplexity Change from Pruning")
            ax3.set_ylabel("Perplexity Change from Fine-tuning")
            ax3.legend(fontsize=7, loc='best')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "No effect data available yet", 
                    ha="center", va="center", fontsize=12)
    else:
        ax3.text(0.5, 0.5, "No effect data available yet", 
                ha="center", va="center", fontsize=12)
    
    # 4. Final results: perplexity reduction by pruning level and strategy (bottom-right)
    ax4 = axs[1, 1]
    
    if "perplexity_change_from_baseline" in results_df.columns:
        # Get baseline and final results
        baseline_df = results_df[results_df["stage"] == "baseline"].copy()
        baseline_df = baseline_df[["model", "strategy", "pruning_level", "perplexity"]]
        baseline_df = baseline_df.rename(columns={"perplexity": "baseline_perplexity"})
        
        final_df = results_df[results_df["stage"] == "fine_tuned"].copy()
        final_df = final_df[["model", "strategy", "pruning_level", "perplexity", "perplexity_change_from_baseline"]]
        final_df = final_df.rename(columns={"perplexity": "final_perplexity"})
        
        # Merge
        final_results = pd.merge(
            baseline_df, final_df,
            on=["model", "strategy", "pruning_level"],
            how="inner"
        )
        
        if not final_results.empty:
            # Plot as bar chart
            # Group by pruning level and strategy
            grouped = final_results.groupby(["pruning_level", "strategy"])["perplexity_change_from_baseline"].mean().reset_index()
            
            # Pivot for grouped bar chart
            pivot_df = grouped.pivot(index="pruning_level", columns="strategy", values="perplexity_change_from_baseline")
            
            # Plot
            pivot_df.plot(kind="bar", ax=ax4)
            
            ax4.axhline(y=0, color="k", linestyle="--", alpha=0.3)
            ax4.set_title("Final Perplexity Change from Baseline")
            ax4.set_xlabel("Pruning Level")
            ax4.set_ylabel("Perplexity Change")
            ax4.legend(title="Strategy", fontsize=7)
            ax4.grid(True, alpha=0.3, axis="y")
        else:
            ax4.text(0.5, 0.5, "No final results available yet", 
                    ha="center", va="center", fontsize=12)
    else:
        ax4.text(0.5, 0.5, "No final results available yet", 
                ha="center", va="center", fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    return fig


def plot_strategy_comparison(comparison_df: pd.DataFrame, 
                            strategies: List[str], 
                            model_name: str,
                            pruning_level: float,
                            figsize: tuple = (12, 6)) -> plt.Figure:
    """
    Create a bar chart comparing different pruning strategies.
    
    Args:
        comparison_df: DataFrame containing comparison data
        strategies: List of strategy names
        model_name: Name of the model used
        pruning_level: Pruning level used
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib figure object
    """
    # Reset style parameters
    plt.rcParams.update(plt.rcParamsDefault)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot baseline, pruned, and fine-tuned perplexity for each strategy
    x = np.arange(len(strategies))
    width = 0.25
    
    if "Fine-tuned Perplexity" in comparison_df.columns:
        # If we have fine-tuning results
        plt.bar(x - width, comparison_df["Baseline Perplexity"], width, label="Baseline")
        plt.bar(x, comparison_df["Pruned Perplexity"], width, label="Pruned")
        plt.bar(x + width, comparison_df["Fine-tuned Perplexity"], width, label="Fine-tuned")
    else:
        # If we only have pruning results
        plt.bar(x - width/2, comparison_df["Baseline Perplexity"], width, label="Baseline")
        plt.bar(x + width/2, comparison_df["Pruned Perplexity"], width, label="Pruned")
    
    plt.xlabel("Pruning Strategy")
    plt.ylabel("Perplexity")
    plt.title(f"Comparison of Pruning Strategies ({model_name}, {pruning_level*100:.0f}% pruning)")
    plt.xticks(x, strategies)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, strategy in enumerate(strategies):
        row = comparison_df[comparison_df["Strategy"] == strategy].iloc[0]
        
        # Baseline
        plt.text(i - width, row["Baseline Perplexity"] + 5, 
                f"{row['Baseline Perplexity']:.1f}", 
                ha="center", va="bottom", fontsize=9)
        
        # Pruned
        plt.text(i, row["Pruned Perplexity"] + 5, 
                f"{row['Pruned Perplexity']:.1f}", 
                ha="center", va="bottom", fontsize=9)
        
        # Fine-tuned (if available)
        if "Fine-tuned Perplexity" in comparison_df.columns:
            plt.text(i + width, row["Fine-tuned Perplexity"] + 5, 
                    f"{row['Fine-tuned Perplexity']:.1f}", 
                    ha="center", va="bottom", fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    return fig


def plot_recovery_comparison(comparison_df: pd.DataFrame, 
                            strategies: List[str],
                            model_name: str,
                            pruning_level: float,
                            figsize: tuple = (10, 5)) -> plt.Figure:
    """
    Create a bar chart comparing recovery or improvement percentages.
    
    Args:
        comparison_df: DataFrame containing comparison data
        strategies: List of strategy names
        model_name: Name of the model used
        pruning_level: Pruning level used
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib figure object or None if no recovery data available
    """
    if "Recovery %" not in comparison_df.columns and "Improvement %" not in comparison_df.columns:
        return None
    
    # Reset style parameters
    plt.rcParams.update(plt.rcParamsDefault)
        
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a combined metric for display (recovery is positive, improvement is negative)
    recovery_values = []
    for strategy in strategies:
        row = comparison_df[comparison_df["Strategy"] == strategy].iloc[0]
        
        if pd.notna(row["Recovery %"]):
            # This is a recovery scenario (pruning hurt, fine-tuning helped)
            recovery_values.append(row["Recovery %"])
        elif pd.notna(row["Improvement %"]):
            # This is an improvement scenario (pruning helped, fine-tuning helped more)
            recovery_values.append(-row["Improvement %"])
        else:
            recovery_values.append(0)
    
    # Create bars with different colors based on whether it's recovery or improvement
    colors = ["red" if val >= 0 else "green" for val in recovery_values]
    plt.bar(strategies, recovery_values, color=colors)
    
    # Add horizontal line at 0
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    
    # Add labels
    plt.text(strategies[0], 50, "Recovery %", color="red", ha="center", va="center", fontsize=10)
    plt.text(strategies[0], -50, "Improvement %", color="green", ha="center", va="center", fontsize=10)
    
    # Add value labels to bars
    for i, val in enumerate(recovery_values):
        if val >= 0:
            plt.text(i, val + 5, f"{val:.1f}%", ha="center", va="bottom", color="red", fontsize=9)
        else:
            plt.text(i, val - 5, f"{-val:.1f}%", ha="center", va="top", color="green", fontsize=9)
    
    plt.xlabel("Pruning Strategy")
    plt.ylabel("Percentage")
    plt.title(f"Recovery or Improvement by Strategy ({model_name}, {pruning_level*100:.0f}% pruning)")
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    return fig


def visualize_head_importance(model_name: str, 
                             importance_scores: Dict[str, List],
                             num_layers: int,
                             num_heads: int,
                             figsize: Optional[tuple] = None) -> plt.Figure:
    """
    Visualize attention head importance scores for different strategies.
    
    Args:
        model_name: Name of the model
        importance_scores: Dictionary of strategy name -> list of (layer, head, score) tuples
        num_layers: Number of layers in the model
        num_heads: Number of attention heads per layer
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib figure object
    """
    # Reset style parameters
    plt.rcParams.update(plt.rcParamsDefault)
    
    if figsize is None:
        figsize = (12, 1.5 * num_layers)
        
    strategies = list(importance_scores.keys())
    
    # Create figure
    fig, axes = plt.subplots(num_layers, len(strategies), figsize=figsize)
    
    # Create title
    fig.suptitle(f"Attention Head Importance by Strategy for {model_name}", fontsize=16)
    
    # Set column titles
    for i, strategy_name in enumerate(strategies):
        axes[0, i].set_title(f"{strategy_name.capitalize()} Strategy")
    
    # Create a heatmap for each strategy showing head importance
    for layer in range(num_layers):
        for i, strategy_name in enumerate(strategies):
            # Extract importance scores for this layer
            layer_scores = [score for l, h, score in importance_scores[strategy_name] if l == layer]
            
            # Create array for visualization
            scores_array = np.array(layer_scores).reshape(1, -1)
            
            # Create heatmap
            cax = axes[layer, i].imshow(scores_array, cmap="viridis", aspect="auto")
            
            # Add labels
            axes[layer, i].set_yticks([0])
            axes[layer, i].set_yticklabels([f"Layer {layer}"])
            axes[layer, i].set_xticks(range(num_heads))
            axes[layer, i].set_xticklabels([f"H{h}" for h in range(num_heads)], 
                                       rotation=90 if num_heads > 8 else 0)
            
            # Add importance values as text
            for h in range(num_heads):
                score = scores_array[0, h]
                if np.isnan(score):
                    text_color = "black"
                else:
                    text_color = "white" if score > 0.5 else "black"
                axes[layer, i].text(h, 0, f"{score:.2f}", ha="center", va="center", 
                               color=text_color, fontsize=8)
    
    # Add a colorbar
    fig.colorbar(cax, ax=axes.ravel().tolist(), shrink=0.6)
    
    # Make sure there's enough space between subplots
    plt.subplots_adjust(hspace=0.3, wspace=0.3, top=0.9, bottom=0.05, left=0.05, right=0.95)
    return fig