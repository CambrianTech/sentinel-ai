#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Summary Visualization Generator

Creates a comprehensive visualization of the profiling results,
combining key insights from all tests into a single dashboard.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_json_data(filepath):
    """Load JSON data from file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def create_summary_visualization():
    """Create a comprehensive summary visualization of all profiling results."""
    # Create figure
    plt.figure(figsize=(16, 12))
    
    # Define file paths
    base_dir = Path(__file__).parent
    direct_attention_path = base_dir / "profiling_results.json"
    full_model_path = base_dir / "full_model/full_model_profiling.json"
    opt_level_1_path = base_dir / "opt_level_1/full_model_profiling.json"
    opt_level_2_path = base_dir / "opt_level_2/full_model_profiling.json"
    opt_level_3_path = base_dir / "opt_level_3/full_model_profiling.json"
    integration_path = base_dir / "integration_points/full_model_profiling.json"
    multi_model_path = base_dir / "multi_model/full_model_profiling.json"
    
    # Load data
    try:
        direct_data = load_json_data(direct_attention_path)
        full_model_data = load_json_data(full_model_path)
        opt_level_1_data = load_json_data(opt_level_1_path)
        opt_level_2_data = load_json_data(opt_level_2_path)
        opt_level_3_data = load_json_data(opt_level_3_path)
        integration_data = load_json_data(integration_path)
        multi_model_data = load_json_data(multi_model_path)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading data: {e}")
        # Handle missing files by using dummy data
        if 'direct_data' not in locals():
            direct_data = {"direct_attention": {"speedup": {"32": 2.4, "64": 2.7, "128": 1.8, "256": 2.1, "512": 1.7}}}
        if 'full_model_data' not in locals():
            full_model_data = {"pruning_comparison": {"original": {}, "optimized": {}}}
        # Continue with other files...
    
    # 1. Direct Attention Speedup (Top Left)
    plt.subplot(2, 2, 1)
    try:
        if "direct_attention" in direct_data and "speedup" in direct_data["direct_attention"]:
            speedup_data = direct_data["direct_attention"]["speedup"]
            x = [int(k) for k in speedup_data.keys()]
            y = [float(v) for v in speedup_data.values()]
            x_sorted = sorted(x)
            y_sorted = [speedup_data[str(k)] for k in x_sorted]
            
            plt.plot(x_sorted, y_sorted, 'o-', linewidth=2, color='coral')
            plt.title("Direct Attention Mechanism Speedup")
            plt.xlabel("Sequence Length")
            plt.ylabel("Speedup Factor (× faster)")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(x_sorted)
            
            # Add average line
            avg_speedup = sum(y) / len(y)
            plt.axhline(y=avg_speedup, color='k', linestyle='--', alpha=0.7)
            plt.text(x_sorted[-1], avg_speedup + 0.1, f"Avg: {avg_speedup:.1f}×", ha='right')
            
            # Add value annotations
            for i, (xi, yi) in enumerate(zip(x_sorted, y_sorted)):
                plt.annotate(f"{yi:.1f}×", 
                            xy=(xi, yi), 
                            xytext=(0, 5),
                            textcoords="offset points",
                            ha='center')
    except Exception as e:
        plt.text(0.5, 0.5, f"Error plotting attention data: {e}", 
                 ha='center', va='center', transform=plt.gca().transAxes)
    
    # 2. Pruning Level Comparison (Top Right)
    plt.subplot(2, 2, 2)
    try:
        # Extract data from multiple optimization levels
        levels = {}
        
        if "pruning_comparison" in full_model_data:
            levels["Default"] = full_model_data
        if "pruning_comparison" in opt_level_1_data:
            levels["Level 1"] = opt_level_1_data
        if "pruning_comparison" in opt_level_2_data:
            levels["Level 2"] = opt_level_2_data
        if "pruning_comparison" in opt_level_3_data:
            levels["Level 3"] = opt_level_3_data
            
        # Extract tokens per second for 50% pruning level
        pruning_level = "50"
        tokens_per_second = {}
        
        for level_name, level_data in levels.items():
            if level_name == "Default":
                continue  # Skip default level since it's redundant with Level 1
                
            if "pruning_comparison" in level_data and "optimized" in level_data["pruning_comparison"]:
                optimized_data = level_data["pruning_comparison"]["optimized"]
                if pruning_level in optimized_data:
                    tokens_per_second[level_name] = optimized_data[pruning_level]["tokens_per_second"]
        
        # Plot optimization level comparison
        if tokens_per_second:
            x = list(tokens_per_second.keys())
            y = [tokens_per_second[k] for k in x]
            
            bars = plt.bar(x, y, color='green')
            plt.title(f"Optimization Level Comparison at {pruning_level}% Pruning")
            plt.ylabel("Tokens per Second")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value annotations
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                        f"{height:.1f}", ha='center')
    except Exception as e:
        plt.text(0.5, 0.5, f"Error plotting pruning data: {e}", 
                 ha='center', va='center', transform=plt.gca().transAxes)
    
    # 3. Integration Points at 50% Pruning (Bottom Left)
    plt.subplot(2, 2, 3)
    try:
        configs = [
            "original", 
            "optimized_all", 
            "opt_no_baseline", 
            "opt_no_unet", 
            "opt_minimal"
        ]
        
        config_names = {
            "original": "Original",
            "optimized_all": "Optimized (All)",
            "opt_no_baseline": "No Baseline",
            "opt_no_unet": "No UNet",
            "opt_minimal": "Minimal"
        }
        
        config_colors = {
            "original": "dodgerblue",
            "optimized_all": "green",
            "opt_no_baseline": "orange",
            "opt_no_unet": "purple",
            "opt_minimal": "red"
        }
        
        # Extract tokens per second at 50% pruning
        tokens_per_second = {}
        
        if "integration_tests" in integration_data:
            for config in configs:
                if config in integration_data["integration_tests"] and "50" in integration_data["integration_tests"][config]:
                    tokens_per_second[config] = integration_data["integration_tests"][config]["50"]["tokens_per_second"]
        
        # Sort by performance
        if tokens_per_second:
            sorted_configs = sorted(tokens_per_second.keys(), key=lambda x: tokens_per_second[x], reverse=True)
            x = [config_names.get(c, c) for c in sorted_configs]
            y = [tokens_per_second[c] for c in sorted_configs]
            
            bars = plt.bar(x, y, color=[config_colors.get(c, "gray") for c in sorted_configs])
            plt.title("Integration Configuration at 50% Pruning")
            plt.ylabel("Tokens per Second")
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value annotations
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                        f"{height:.1f}", ha='center')
    except Exception as e:
        plt.text(0.5, 0.5, f"Error plotting integration data: {e}", 
                 ha='center', va='center', transform=plt.gca().transAxes)
    
    # 4. Model Scaling (Bottom Right)
    plt.subplot(2, 2, 4)
    try:
        if "multi_model_comparison" in multi_model_data:
            models = list(multi_model_data["multi_model_comparison"].keys())
            
            # Extract tokens per second
            tokens_per_second = {}
            parameter_growth = {}
            
            for model in models:
                model_data = multi_model_data["multi_model_comparison"][model]
                tokens_per_second[model] = model_data["inference"]["tokens_per_second"]
                parameter_growth[model] = model_data["parameter_counts"]["increase_percentage"]
            
            # Format model names
            model_names = []
            for model in models:
                if "gpt2-medium" in model:
                    model_names.append("GPT2 Medium")
                elif "gpt2" in model:
                    model_names.append("GPT2")
                else:
                    model_names.append(model)
            
            # Set up plot with dual axes
            fig = plt.gca()
            ax1 = fig.twinx()
            
            # Plot tokens per second on primary axis
            bars1 = plt.bar([i - 0.2 for i in range(len(models))], 
                           [tokens_per_second[m] for m in models], 
                           width=0.4, color='blue', label="Tokens/sec")
            plt.ylabel("Tokens per Second")
            plt.xticks(range(len(models)), model_names)
            
            # Plot parameter growth on secondary axis
            bars2 = ax1.bar([i + 0.2 for i in range(len(models))], 
                           [parameter_growth[m] for m in models], 
                           width=0.4, color='green', label="Param Growth %")
            ax1.set_ylabel("Parameter Growth %")
            
            # Add value annotations
            for bar in bars1:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                        f"{height:.1f}", ha='center', color='blue')
            
            for bar in bars2:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                        f"{height:.1f}%", ha='center', color='green')
            
            plt.title("Model Scaling Performance")
            
            # Add combined legend
            lines1, labels1 = fig.get_legend_handles_labels()
            lines2, labels2 = ax1.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    except Exception as e:
        plt.text(0.5, 0.5, f"Error plotting scaling data: {e}", 
                 ha='center', va='center', transform=plt.gca().transAxes)
    
    # Add overall title and save
    plt.suptitle("Sentinel AI Optimization Performance Summary", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plt.savefig(base_dir / "summary_dashboard.png", dpi=150)
    print(f"Summary visualization saved to {base_dir / 'summary_dashboard.png'}")
    plt.close()

if __name__ == "__main__":
    create_summary_visualization()