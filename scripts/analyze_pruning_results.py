#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze Pruning and Fine-Tuning Results

This script analyzes results from pruning and fine-tuning experiments,
generating summary statistics and visualizations for inclusion in papers
or the project README.

Example usage:
    python scripts/analyze_pruning_results.py --results_dir pruning_finetuning_results
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))


def setup_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze pruning and fine-tuning results")
    
    parser.add_argument("--results_dir", type=str, default="pruning_finetuning_results",
                       help="Directory containing result files")
    parser.add_argument("--output_dir", type=str, default="pruning_analysis",
                       help="Directory to save analysis outputs")
    parser.add_argument("--summary_file", type=str, default=None,
                       help="Specific summary file to analyze (if provided, will look only at this file)")
    parser.add_argument("--save_figures", action="store_true",
                       help="Save figures as PNG files")
    parser.add_argument("--save_tables", action="store_true",
                       help="Save tables as CSV files")
    
    return parser.parse_args()


class PruningAnalyzer:
    """Analyze pruning and fine-tuning results"""
    
    def __init__(self, args):
        self.args = args
        
        # Setup directories
        self.results_dir = Path(args.results_dir)
        if not self.results_dir.exists():
            raise ValueError(f"Results directory {self.results_dir} does not exist")
            
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Result loading flags
        self.results_loaded = False
        self.results_df = None
        self.raw_results = []
    
    def load_results(self):
        """Load results from files"""
        print(f"Loading results from {self.results_dir}")
        
        # If summary file is specified, load only that file
        if self.args.summary_file:
            summary_path = self.results_dir / self.args.summary_file
            if not summary_path.exists():
                raise ValueError(f"Summary file {summary_path} does not exist")
            
            print(f"Loading summary file: {summary_path}")
            with open(summary_path, "r") as f:
                summary = json.load(f)
                
            # Extract results
            self.raw_results = summary.get("results", [])
            print(f"Loaded {len(self.raw_results)} results from summary file")
        else:
            # Load all individual result files
            result_files = list(self.results_dir.glob("*.json"))
            
            # Filter out summary files
            result_files = [f for f in result_files if not f.name.startswith("summary_")]
            
            print(f"Found {len(result_files)} result files")
            
            # Load each file
            for file_path in result_files:
                try:
                    with open(file_path, "r") as f:
                        result = json.load(f)
                    
                    # Only add if it has all required stages
                    if "stages" in result and all(stage in result["stages"] for stage in ["baseline", "pruned", "fine_tuned"]):
                        self.raw_results.append(result)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            
            print(f"Successfully loaded {len(self.raw_results)} valid results")
        
        # Convert to DataFrame
        self._create_dataframe()
        
        self.results_loaded = True
        return len(self.raw_results) > 0
    
    def _create_dataframe(self):
        """Convert raw results to a DataFrame for analysis"""
        if not self.raw_results:
            print("No results to convert to DataFrame")
            return
        
        # Extract data for DataFrame
        data = []
        
        for result in self.raw_results:
            # Extract experiment info
            model = result["model"]
            strategy = result["strategy"]
            pruning_level = result["pruning_level"]
            timestamp = result.get("timestamp", "unknown")
            
            # Add baseline stage
            if "baseline" in result["stages"]:
                baseline = result["stages"]["baseline"]
                data.append({
                    "model": model,
                    "strategy": strategy,
                    "pruning_level": pruning_level,
                    "stage": "baseline",
                    "perplexity": baseline["perplexity"],
                    "timestamp": timestamp
                })
            
            # Add pruned stage
            if "pruned" in result["stages"]:
                pruned = result["stages"]["pruned"]
                data.append({
                    "model": model,
                    "strategy": strategy,
                    "pruning_level": pruning_level,
                    "stage": "pruned",
                    "perplexity": pruned["perplexity"],
                    "perplexity_change": pruned.get("perplexity_change", 0),
                    "timestamp": timestamp
                })
            
            # Add fine-tuned stage
            if "fine_tuned" in result["stages"]:
                fine_tuned = result["stages"]["fine_tuned"]
                data.append({
                    "model": model,
                    "strategy": strategy,
                    "pruning_level": pruning_level,
                    "stage": "fine_tuned",
                    "perplexity": fine_tuned["perplexity"],
                    "perplexity_change_from_baseline": fine_tuned.get("perplexity_change_from_baseline", 0),
                    "perplexity_change_from_pruned": fine_tuned.get("perplexity_change_from_pruned", 0),
                    "recovery_percentage": fine_tuned.get("recovery_percentage", None),
                    "improvement_percentage": fine_tuned.get("improvement_percentage", None),
                    "timestamp": timestamp
                })
        
        self.results_df = pd.DataFrame(data)
        
        # Save full DataFrame to CSV
        if self.args.save_tables:
            csv_path = self.output_dir / "full_results.csv"
            self.results_df.to_csv(csv_path, index=False)
            print(f"Saved full results to {csv_path}")
    
    def create_summary_tables(self):
        """Create summary tables from results"""
        if not self.results_loaded:
            if not self.load_results():
                print("No results available for summary tables")
                return
        
        print("Creating summary tables...")
        
        # 1. Strategy comparison table
        strategy_table = self._create_strategy_comparison()
        if strategy_table is not None and self.args.save_tables:
            csv_path = self.output_dir / "strategy_comparison.csv"
            strategy_table.to_csv(csv_path, index=False)
            print(f"Saved strategy comparison to {csv_path}")
        
        # 2. Pruning level comparison table
        pruning_level_table = self._create_pruning_level_comparison()
        if pruning_level_table is not None and self.args.save_tables:
            csv_path = self.output_dir / "pruning_level_comparison.csv"
            pruning_level_table.to_csv(csv_path, index=False)
            print(f"Saved pruning level comparison to {csv_path}")
        
        # 3. Model comparison table
        model_table = self._create_model_comparison()
        if model_table is not None and self.args.save_tables:
            csv_path = self.output_dir / "model_comparison.csv"
            model_table.to_csv(csv_path, index=False)
            print(f"Saved model comparison to {csv_path}")
        
        # 4. Best results table
        best_results = self._create_best_results_table()
        if best_results is not None and self.args.save_tables:
            csv_path = self.output_dir / "best_results.csv"
            best_results.to_csv(csv_path, index=False)
            print(f"Saved best results to {csv_path}")
        
        return {
            "strategy_comparison": strategy_table,
            "pruning_level_comparison": pruning_level_table,
            "model_comparison": model_table,
            "best_results": best_results
        }
    
    def _create_strategy_comparison(self):
        """Create a comparison table for different strategies"""
        if self.results_df is None or self.results_df.empty:
            return None
        
        # Get only fine-tuned results
        fine_tuned = self.results_df[self.results_df["stage"] == "fine_tuned"].copy()
        
        if fine_tuned.empty:
            return None
        
        # Group by strategy
        grouped = fine_tuned.groupby("strategy").agg({
            "perplexity_change_from_baseline": ["mean", "std", "min", "max"],
            "recovery_percentage": ["mean", "std", "min", "max", "count"]
        }).reset_index()
        
        # Rename columns
        grouped.columns = ["strategy", 
                          "perplexity_change_mean", "perplexity_change_std", "perplexity_change_min", "perplexity_change_max",
                          "recovery_pct_mean", "recovery_pct_std", "recovery_pct_min", "recovery_pct_max", "count"]
        
        return grouped
    
    def _create_pruning_level_comparison(self):
        """Create a comparison table for different pruning levels"""
        if self.results_df is None or self.results_df.empty:
            return None
        
        # Get pruned and fine-tuned results
        stages = self.results_df[self.results_df["stage"].isin(["pruned", "fine_tuned"])].copy()
        
        if stages.empty:
            return None
        
        # Create a pivoted table with pruning level as index
        pivot_table = stages.pivot_table(
            index="pruning_level",
            columns="stage",
            values="perplexity",
            aggfunc=["mean", "std"]
        ).reset_index()
        
        # Flatten column names
        pivot_table.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in pivot_table.columns]
        
        # Calculate improvement
        pivot_table["improvement"] = pivot_table["mean_pruned"] - pivot_table["mean_fine_tuned"]
        pivot_table["improvement_pct"] = (pivot_table["improvement"] / pivot_table["mean_pruned"]) * 100
        
        # Sort by pruning level
        pivot_table = pivot_table.sort_values("pruning_level")
        
        return pivot_table
    
    def _create_model_comparison(self):
        """Create a comparison table for different models"""
        if self.results_df is None or self.results_df.empty:
            return None
        
        # Get fine-tuned results
        fine_tuned = self.results_df[self.results_df["stage"] == "fine_tuned"].copy()
        
        if fine_tuned.empty:
            return None
        
        # Group by model
        grouped = fine_tuned.groupby("model").agg({
            "perplexity_change_from_baseline": ["mean", "std", "min", "max"],
            "recovery_percentage": ["mean", "std", "min", "max", "count"]
        }).reset_index()
        
        # Rename columns
        grouped.columns = ["model", 
                          "perplexity_change_mean", "perplexity_change_std", "perplexity_change_min", "perplexity_change_max",
                          "recovery_pct_mean", "recovery_pct_std", "recovery_pct_min", "recovery_pct_max", "count"]
        
        return grouped
    
    def _create_best_results_table(self):
        """Create a table of best results"""
        if self.results_df is None or self.results_df.empty:
            return None
        
        # Get all stages
        all_stages = self.results_df.copy()
        
        if all_stages.empty:
            return None
        
        # Create a pivot table with model, strategy, pruning_level as index
        pivot_table = all_stages.pivot_table(
            index=["model", "strategy", "pruning_level"],
            columns="stage",
            values="perplexity",
            aggfunc="mean"
        ).reset_index()
        
        # Calculate relative changes
        pivot_table["pruning_change"] = (pivot_table["pruned"] - pivot_table["baseline"]) / pivot_table["baseline"] * 100
        pivot_table["finetuning_change"] = (pivot_table["fine_tuned"] - pivot_table["baseline"]) / pivot_table["baseline"] * 100
        
        # Sort by total change (improvement is negative)
        pivot_table = pivot_table.sort_values("finetuning_change")
        
        # Get top 10 results
        top_results = pivot_table.head(10)
        
        return top_results
    
    def create_visualizations(self):
        """Create visualizations from results"""
        if not self.results_loaded:
            if not self.load_results():
                print("No results available for visualizations")
                return
        
        print("Creating visualizations...")
        
        # Create all plots
        plots = {}
        
        # 1. Perplexity across stages
        plots["stages_plot"] = self._plot_perplexity_across_stages()
        
        # 2. Recovery percentage by pruning level
        plots["recovery_plot"] = self._plot_recovery_by_pruning_level()
        
        # 3. Strategy comparison
        plots["strategy_plot"] = self._plot_strategy_comparison()
        
        # 4. Best experiments
        plots["best_plot"] = self._plot_best_experiments()
        
        return plots
    
    def _plot_perplexity_across_stages(self):
        """Plot perplexity across different stages"""
        if self.results_df is None or self.results_df.empty:
            return None
        
        # Filter to main stages and group
        stages_df = self.results_df[self.results_df["stage"].isin(["baseline", "pruned", "fine_tuned"])]
        group_avg = stages_df.groupby(["strategy", "stage"])["perplexity"].mean().reset_index()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot lines for each strategy
        for strategy in group_avg["strategy"].unique():
            strategy_data = group_avg[group_avg["strategy"] == strategy]
            
            # Sort by stage
            stage_order = {"baseline": 0, "pruned": 1, "fine_tuned": 2}
            strategy_data = strategy_data.sort_values(by="stage", key=lambda x: x.map(stage_order))
            
            ax.plot(strategy_data["stage"], strategy_data["perplexity"], "o-", 
                   label=strategy, linewidth=2)
        
        ax.set_title("Perplexity Across Stages", fontsize=14)
        ax.set_xlabel("Stage", fontsize=12)
        ax.set_ylabel("Perplexity", fontsize=12)
        ax.legend(title="Strategy", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.7)
        
        # Prettify
        plt.tight_layout()
        
        # Save figure
        if self.args.save_figures:
            fig_path = self.output_dir / "perplexity_across_stages.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to {fig_path}")
        
        return fig
    
    def _plot_recovery_by_pruning_level(self):
        """Plot recovery percentage by pruning level"""
        if self.results_df is None or self.results_df.empty:
            return None
        
        # Get fine-tuned results
        fine_tuned = self.results_df[self.results_df["stage"] == "fine_tuned"].copy()
        
        if fine_tuned.empty:
            return None
        
        # Create recovery column combining recovery and improvement percentages
        fine_tuned["recovery"] = fine_tuned["recovery_percentage"]
        # If improvement percentage exists and recovery is NaN, use negative of improvement
        mask = fine_tuned["recovery"].isna() & fine_tuned["improvement_percentage"].notna()
        fine_tuned.loc[mask, "recovery"] = -fine_tuned.loc[mask, "improvement_percentage"]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot by strategy
        for strategy in fine_tuned["strategy"].unique():
            strategy_df = fine_tuned[fine_tuned["strategy"] == strategy]
            
            # Group by pruning level
            grouped = strategy_df.groupby("pruning_level")["recovery"].mean().reset_index()
            
            # Sort by pruning level
            grouped = grouped.sort_values("pruning_level")
            
            ax.plot(grouped["pruning_level"], grouped["recovery"], "o-", 
                   label=strategy, linewidth=2)
        
        # Add reference lines
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.axhline(y=100, color="g", linestyle="--", alpha=0.3)
        plt.text(0.01, 100, "Full Recovery", color="green", ha="left", va="bottom")
        plt.text(0.01, -5, "Improvement", color="blue", ha="left", va="top")
        
        ax.set_title("Recovery After Fine-tuning by Pruning Level", fontsize=14)
        ax.set_xlabel("Pruning Level", fontsize=12)
        ax.set_ylabel("Recovery % (negative means improvement)", fontsize=12)
        ax.legend(title="Strategy", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.7)
        
        # Prettify
        plt.tight_layout()
        
        # Save figure
        if self.args.save_figures:
            fig_path = self.output_dir / "recovery_by_pruning_level.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to {fig_path}")
        
        return fig
    
    def _plot_strategy_comparison(self):
        """Plot comparison of different strategies"""
        if self.results_df is None or self.results_df.empty:
            return None
        
        # Get data for all stages
        stages_df = self.results_df.copy()
        
        if stages_df.empty:
            return None
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Bar chart comparing final perplexity change by strategy
        fine_tuned = stages_df[stages_df["stage"] == "fine_tuned"].copy()
        
        if not fine_tuned.empty:
            # Group by strategy
            strategy_perf = fine_tuned.groupby("strategy")["perplexity_change_from_baseline"].mean().reset_index()
            
            # Sort by performance (lower is better)
            strategy_perf = strategy_perf.sort_values("perplexity_change_from_baseline")
            
            # Plot
            bars = axes[0].bar(strategy_perf["strategy"], strategy_perf["perplexity_change_from_baseline"])
            
            # Color bars based on value (red for increase, green for decrease)
            for i, bar in enumerate(bars):
                color = "green" if strategy_perf["perplexity_change_from_baseline"].iloc[i] < 0 else "red"
                bar.set_color(color)
            
            axes[0].axhline(y=0, color="k", linestyle="--", alpha=0.3)
            axes[0].set_title("Final Perplexity Change by Strategy", fontsize=14)
            axes[0].set_xlabel("Strategy", fontsize=12)
            axes[0].set_ylabel("Perplexity Change from Baseline", fontsize=12)
            axes[0].grid(True, axis="y", linestyle="--", alpha=0.7)
        
        # 2. Box plot of recovery percentage by strategy
        if "recovery" in fine_tuned.columns:
            sns.boxplot(x="strategy", y="recovery", data=fine_tuned, ax=axes[1])
            
            axes[1].axhline(y=0, color="k", linestyle="--", alpha=0.3)
            axes[1].axhline(y=100, color="g", linestyle="--", alpha=0.3)
            axes[1].set_title("Recovery Percentage by Strategy", fontsize=14)
            axes[1].set_xlabel("Strategy", fontsize=12)
            axes[1].set_ylabel("Recovery % (negative means improvement)", fontsize=12)
            axes[1].grid(True, axis="y", linestyle="--", alpha=0.7)
        
        # Prettify
        plt.tight_layout()
        
        # Save figure
        if self.args.save_figures:
            fig_path = self.output_dir / "strategy_comparison.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to {fig_path}")
        
        return fig
    
    def _plot_best_experiments(self):
        """Plot the best performing experiments"""
        if self.results_df is None or self.results_df.empty:
            return None
        
        # Create a pivot table with model, strategy, pruning_level as index
        pivot_table = self.results_df.pivot_table(
            index=["model", "strategy", "pruning_level"],
            columns="stage",
            values="perplexity",
            aggfunc="mean"
        ).reset_index()
        
        if pivot_table.empty or "fine_tuned" not in pivot_table.columns:
            return None
        
        # Calculate improvement
        pivot_table["improvement"] = ((pivot_table["baseline"] - pivot_table["fine_tuned"]) / 
                                    pivot_table["baseline"] * 100)
        
        # Sort by improvement (higher is better)
        pivot_table = pivot_table.sort_values("improvement", ascending=False)
        
        # Get top 5 results
        top_results = pivot_table.head(5)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create index labels
        labels = [f"{row['model']}\n{row['strategy']}, {row['pruning_level']:.2f}" 
                 for _, row in top_results.iterrows()]
        
        # Create grouped bar chart
        x = np.arange(len(labels))
        width = 0.25
        
        ax.bar(x - width, top_results["baseline"], width, label="Baseline", color="blue", alpha=0.7)
        ax.bar(x, top_results["pruned"], width, label="After Pruning", color="orange", alpha=0.7)
        ax.bar(x + width, top_results["fine_tuned"], width, label="After Fine-tuning", color="green", alpha=0.7)
        
        # Add improvement labels
        for i, improvement in enumerate(top_results["improvement"]):
            ax.text(i, top_results["fine_tuned"].iloc[i], f"{improvement:.1f}%", 
                   ha="center", va="bottom", fontsize=9, color="darkgreen")
        
        ax.set_title("Top 5 Experiments by Perplexity Improvement", fontsize=14)
        ax.set_xlabel("Experiment", fontsize=12)
        ax.set_ylabel("Perplexity", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)
        
        # Prettify
        plt.tight_layout()
        
        # Save figure
        if self.args.save_figures:
            fig_path = self.output_dir / "best_experiments.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to {fig_path}")
        
        return fig
    
    def generate_markdown_report(self):
        """Generate a Markdown report with results and visualizations"""
        if not self.results_loaded:
            if not self.load_results():
                print("No results available for report")
                return None
        
        print("Generating Markdown report...")
        
        # Create visualizations for report
        self.create_visualizations()
        
        # Create summary tables
        tables = self.create_summary_tables()
        
        # Build report
        report = []
        
        # Header
        report.append("# Pruning and Fine-Tuning Experiments Report")
        report.append("")
        report.append(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overview
        report.append("## Overview")
        report.append("")
        model_count = len(self.results_df["model"].unique()) if self.results_df is not None else 0
        strategy_count = len(self.results_df["strategy"].unique()) if self.results_df is not None else 0
        pruning_level_count = len(self.results_df["pruning_level"].unique()) if self.results_df is not None else 0
        experiment_count = len(set(zip(
            self.results_df["model"], 
            self.results_df["strategy"], 
            self.results_df["pruning_level"]
        ))) if self.results_df is not None else 0
        
        report.append(f"This report summarizes results from pruning and fine-tuning experiments:")
        report.append("")
        report.append(f"- **Models tested:** {model_count}")
        report.append(f"- **Pruning strategies:** {strategy_count}")
        report.append(f"- **Pruning levels:** {pruning_level_count}")
        report.append(f"- **Total experiments:** {experiment_count}")
        report.append("")
        
        # Key Findings
        report.append("## Key Findings")
        report.append("")
        
        if tables and "best_results" in tables and tables["best_results"] is not None:
            best = tables["best_results"].iloc[0]
            
            report.append("1. **Best overall result:**")
            report.append(f"   - Model: {best['model']}")
            report.append(f"   - Strategy: {best['strategy']}")
            report.append(f"   - Pruning level: {best['pruning_level']:.2f}")
            report.append(f"   - Perplexity change after pruning: {best['pruning_change']:.2f}%")
            report.append(f"   - Perplexity change after fine-tuning: {best['finetuning_change']:.2f}%")
            report.append("")
        
        if tables and "strategy_comparison" in tables and tables["strategy_comparison"] is not None:
            best_strategy = tables["strategy_comparison"].iloc[
                tables["strategy_comparison"]["perplexity_change_mean"].argmin()
            ]
            
            report.append("2. **Best pruning strategy:**")
            report.append(f"   - Strategy: {best_strategy['strategy']}")
            report.append(f"   - Average perplexity change: {best_strategy['perplexity_change_mean']:.2f}")
            report.append(f"   - Average recovery percentage: {best_strategy['recovery_pct_mean']:.2f}%")
            report.append("")
        
        if tables and "pruning_level_comparison" in tables and tables["pruning_level_comparison"] is not None:
            # Find optimal pruning level (best improvement ratio)
            pruning_comparison = tables["pruning_level_comparison"]
            pruning_comparison["efficiency"] = pruning_comparison["improvement_pct"] / pruning_comparison["pruning_level"]
            best_level = pruning_comparison.iloc[pruning_comparison["efficiency"].argmax()]
            
            report.append("3. **Optimal pruning level:**")
            report.append(f"   - Pruning level: {best_level['pruning_level']:.2f}")
            report.append(f"   - Average perplexity after pruning: {best_level['mean_pruned']:.2f}")
            report.append(f"   - Average perplexity after fine-tuning: {best_level['mean_fine_tuned']:.2f}")
            report.append(f"   - Improvement from pruned: {best_level['improvement_pct']:.2f}%")
            report.append("")
        
        # Include visualizations
        report.append("## Visualizations")
        report.append("")
        
        # List the figures
        if self.args.save_figures:
            report.append("### Perplexity Across Stages")
            report.append("")
            report.append("![Perplexity Across Stages](perplexity_across_stages.png)")
            report.append("")
            
            report.append("### Recovery by Pruning Level")
            report.append("")
            report.append("![Recovery by Pruning Level](recovery_by_pruning_level.png)")
            report.append("")
            
            report.append("### Strategy Comparison")
            report.append("")
            report.append("![Strategy Comparison](strategy_comparison.png)")
            report.append("")
            
            report.append("### Best Experiments")
            report.append("")
            report.append("![Best Experiments](best_experiments.png)")
            report.append("")
        
        # Include tables
        report.append("## Detailed Results")
        report.append("")
        
        if tables and "best_results" in tables and tables["best_results"] is not None:
            report.append("### Top Performing Experiments")
            report.append("")
            report.append(tables["best_results"].to_markdown(index=False))
            report.append("")
        
        if tables and "strategy_comparison" in tables and tables["strategy_comparison"] is not None:
            report.append("### Strategy Comparison")
            report.append("")
            report.append(tables["strategy_comparison"].to_markdown(index=False))
            report.append("")
        
        if tables and "pruning_level_comparison" in tables and tables["pruning_level_comparison"] is not None:
            report.append("### Pruning Level Comparison")
            report.append("")
            report.append(tables["pruning_level_comparison"].to_markdown(index=False))
            report.append("")
        
        # Join report
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.output_dir / "pruning_report.md"
        with open(report_path, "w") as f:
            f.write(report_text)
        
        print(f"Report saved to {report_path}")
        
        return report_text


def main():
    """Main entry point"""
    args = setup_args()
    analyzer = PruningAnalyzer(args)
    
    # Load results
    if not analyzer.load_results():
        print("No valid results found. Exiting.")
        return
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Create summary tables
    analyzer.create_summary_tables()
    
    # Generate markdown report
    analyzer.generate_markdown_report()
    
    print("Analysis completed successfully!")


if __name__ == "__main__":
    main()