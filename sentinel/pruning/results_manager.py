"""
Manages pruning benchmark results and visualization
"""

import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

class ResultsManager:
    """Manages pruning benchmark results"""
    
    def __init__(self, results_dir="pruning_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.all_results = []
        self.results_df = None
    
    def save_result(self, result):
        """Save a single result to disk and update dataframe"""
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        strategy = result["strategy"]
        model = result["model"].replace("/", "_")
        pruning_level = result["pruning_level"]
        
        filename = f"{model}_{strategy}_{pruning_level}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Save as JSON
        with open(filepath, "w") as f:
            json.dump(result, f, indent=2)
            
        # Add to results list and update dataframe
        self.all_results.append(result)
        self._update_dataframe()
        
        return filepath
    
    def load_results(self):
        """Load all results from disk"""
        self.all_results = []
        
        # Find all result files
        result_files = list(self.results_dir.glob("*.json"))
        
        if not result_files:
            print(f"No result files found in {self.results_dir}")
            return []
        
        # Load results
        for filepath in result_files:
            try:
                with open(filepath, "r") as f:
                    result = json.load(f)
                self.all_results.append(result)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        # Update dataframe
        self._update_dataframe()
        
        return self.all_results
    
    def _update_dataframe(self):
        """Convert results to a pandas DataFrame for easier analysis"""
        if not self.all_results:
            self.results_df = pd.DataFrame()
            return
        
        # Extract the fields we care about for analysis
        data = []
        for result in self.all_results:
            data.append({
                "model": result.get("model", "unknown"),
                "strategy": result.get("strategy", "unknown"),
                "pruning_level": result.get("pruning_level", 0),
                "perplexity_before": result.get("perplexity_before", 0),
                "perplexity_after": result.get("perplexity_after", 0),
                "perplexity_change": result.get("perplexity_change", 0),
                "timestamp": result.get("timestamp", "")
            })
        
        self.results_df = pd.DataFrame(data)
        
    def print_summary(self):
        """Print a summary of all results"""
        if self.results_df is None or self.results_df.empty:
            print("No results available")
            return
        
        # Group by model and strategy
        groups = self.results_df.groupby(["model", "strategy"])
        
        print(f"Found {len(self.all_results)} result files:\n")
        
        for (model, strategy), group in groups:
            print(f"Model: {model}, Strategy: {strategy}")
            # Sort by pruning level
            sorted_group = group.sort_values("pruning_level")
            for _, row in sorted_group.iterrows():
                print(f"  Pruning level: {row['pruning_level']:.2f}, " + 
                      f"Perplexity change: {row['perplexity_change']:.4f}")
            print()
    
    def plot_results(self, figsize=(12, 8)):
        """Plot results as an interactive visualization"""
        if self.results_df is None or self.results_df.empty:
            print("No results to plot")
            return
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=False)
        
        # Colors for different strategies
        strategies = self.results_df["strategy"].unique()
        strategy_colors = dict(zip(strategies, sns.color_palette("colorblind", len(strategies))))
        
        # Plot 1: Perplexity change vs pruning level, grouped by model and strategy
        for model in sorted(self.results_df["model"].unique()):
            model_data = self.results_df[self.results_df["model"] == model]
            for strategy in strategies:
                strategy_data = model_data[model_data["strategy"] == strategy]
                if not strategy_data.empty:
                    # Sort by pruning level
                    strategy_data = strategy_data.sort_values("pruning_level")
                    ax1.plot(strategy_data["pruning_level"], strategy_data["perplexity_change"],
                            marker="o", linestyle="-", label=f"{model} - {strategy}",
                            color=strategy_colors[strategy])
        
        # Add horizontal line at y=0
        ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        ax1.set_xlabel("Pruning Level")
        ax1.set_ylabel("Perplexity Change")
        ax1.set_title("Effect of Pruning on Model Perplexity")
        ax1.grid(True, linestyle="--", alpha=0.7)
        
        # Plot 2: Before vs After perplexity, grouped by model and strategy
        for model in sorted(self.results_df["model"].unique()):
            model_data = self.results_df[self.results_df["model"] == model]
            for strategy in strategies:
                strategy_data = model_data[model_data["strategy"] == strategy]
                if not strategy_data.empty:
                    # Point size proportional to pruning level
                    sizes = 100 * strategy_data["pruning_level"] + 20
                    ax2.scatter(strategy_data["perplexity_before"], strategy_data["perplexity_after"],
                               s=sizes, alpha=0.7, label=f"{model} - {strategy}",
                               color=strategy_colors[strategy])
        
        # Add diagonal line (y=x)
        lims = [0, max(self.results_df["perplexity_before"].max(), 
                      self.results_df["perplexity_after"].max()) * 1.1]
        ax2.plot(lims, lims, 'k--', alpha=0.5)
        ax2.set_xlabel("Perplexity Before Pruning")
        ax2.set_ylabel("Perplexity After Pruning")
        ax2.set_title("Perplexity Comparison")
        ax2.grid(True, linestyle="--", alpha=0.7)
        
        # Add legend
        handles, labels = [], []
        for ax in [ax1, ax2]:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        
        # Remove duplicates
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc='lower center', 
                   ncol=min(5, len(by_label)), bbox_to_anchor=(0.5, -0.05))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.show()
        
        return fig
    
    def plot_advanced_analysis(self, figsize=(14, 10)):
        """Create advanced visualizations for more detailed analysis"""
        if self.results_df is None or self.results_df.empty:
            print("No results to plot")
            return
        
        # Set figure size for all plots
        plt.figure(figsize=figsize)
        
        # 1. Box plot of perplexity change by strategy
        plt.subplot(2, 2, 1)
        sns.boxplot(x="strategy", y="perplexity_change", data=self.results_df)
        plt.title("Perplexity Change by Strategy")
        plt.grid(True, linestyle="--", alpha=0.7)
        
        # 2. Box plot of perplexity change by model
        plt.subplot(2, 2, 2)
        sns.boxplot(x="model", y="perplexity_change", data=self.results_df)
        plt.title("Perplexity Change by Model")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, linestyle="--", alpha=0.7)
        
        # 3. Heatmap of average perplexity change (strategy vs pruning level)
        plt.subplot(2, 2, 3)
        try:
            pivot_df = self.results_df.pivot_table(
                index="strategy", 
                columns="pruning_level", 
                values="perplexity_change", 
                aggfunc="mean"
            )
            sns.heatmap(pivot_df, annot=True, cmap="RdYlGn_r", center=0)
            plt.title("Average Perplexity Change by Strategy and Pruning Level")
        except Exception as e:
            print(f"Could not create heatmap: {e}")
            plt.text(0.5, 0.5, "Not enough data points\nfor heatmap visualization",
                    ha='center', va='center', fontsize=12)
            plt.title("Heatmap (Needs More Data)")
        
        # 4. Relationship between perplexity before and change
        plt.subplot(2, 2, 4)
        sns.scatterplot(
            x="perplexity_before", 
            y="perplexity_change", 
            hue="strategy", 
            size="pruning_level",
            sizes=(50, 200),
            data=self.results_df
        )
        plt.title("Perplexity Change vs Initial Perplexity")
        plt.grid(True, linestyle="--", alpha=0.7)
        
        plt.tight_layout()
        plt.show()