#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Export Profiling Results to Different Formats

This script converts profiling results from JSON to different
formats for analysis and sharing, including CSV, Markdown tables,
and HTML reports.
"""

import os
import sys
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))


def setup_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export profiling results to different formats")
    
    # Input options
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input JSON file with profiling results")
    
    # Export options
    parser.add_argument("--format", type=str, default="all",
                        choices=["csv", "markdown", "html", "all"],
                        help="Export format (default: all)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for exported files (default: same as input file)")
    
    # Visualization options
    parser.add_argument("--create_charts", action="store_true",
                        help="Create charts from the data")
    parser.add_argument("--chart_format", type=str, default="png",
                        choices=["png", "svg", "pdf"],
                        help="Format for chart export (default: png)")
    
    return parser.parse_args()


def load_profiling_data(file_path):
    """Load profiling data from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading profiling data: {e}")
        return None


def extract_pruning_comparison(data):
    """Extract pruning comparison data into a pandas DataFrame."""
    if "pruning_comparison" not in data:
        return None
        
    pruning_data = data["pruning_comparison"]
    
    # Create rows for the DataFrame
    rows = []
    
    for model_type in ["original", "optimized"]:
        if model_type not in pruning_data:
            continue
            
        for level, level_data in pruning_data[model_type].items():
            row = {
                "Model Type": model_type.capitalize(),
                "Pruning Level": level,
                "Tokens per Second": level_data["tokens_per_second"],
                "Generation Time (s)": level_data["avg_generation_time"],
                "First Token Time (s)": level_data["first_token_time"]
            }
            rows.append(row)
    
    if not rows:
        return None
        
    df = pd.DataFrame(rows)
    
    # Add speedup calculation
    if "original" in pruning_data and "optimized" in pruning_data:
        # Create a mapping of speedups by pruning level
        speedups = {}
        for level in pruning_data["optimized"].keys():
            if level in pruning_data["original"]:
                orig = pruning_data["original"][level]["tokens_per_second"]
                opt = pruning_data["optimized"][level]["tokens_per_second"]
                speedups[level] = opt / orig if orig > 0 else 0
        
        # Add speedup column to optimized rows
        for i, row in df.iterrows():
            if row["Model Type"] == "Optimized" and row["Pruning Level"] in speedups:
                df.at[i, "Speedup vs Original"] = speedups[row["Pruning Level"]]
    
    return df


def extract_optimization_levels(data):
    """Extract optimization level comparison data into a pandas DataFrame."""
    if "optimization_levels" not in data:
        return None
        
    opt_data = data["optimization_levels"]
    
    # Skip the comparison key if present
    levels = [k for k in opt_data.keys() if k != "comparison"]
    
    # Create rows for the DataFrame
    rows = []
    
    for level in levels:
        level_data = opt_data[level]
        row = {
            "Optimization Level": level,
            "Tokens per Second": level_data["tokens_per_second"],
            "First Token Time (s)": level_data["first_token_time"]
        }
        
        # Add memory usage if available
        if "memory_usage" in level_data:
            memory = level_data["memory_usage"]
            if "gpu" in memory:
                row["GPU Memory (MB)"] = memory["gpu"]["allocated_mb"]
            row["CPU Memory (MB)"] = memory["cpu"]["rss_mb"]
        
        rows.append(row)
    
    if not rows:
        return None
        
    return pd.DataFrame(rows)


def extract_model_loading(data):
    """Extract model loading data into a pandas DataFrame."""
    if "model_loading" not in data:
        return None
        
    loading_data = data["model_loading"]
    
    # Create rows for the DataFrame
    rows = []
    
    for model_type, type_data in loading_data.items():
        if model_type == "operation":
            continue
            
        row = {
            "Model Type": model_type.replace("_model", "").capitalize(),
            "Load Time (s)": type_data["load_time"],
            "Parameters": type_data["parameter_count"],
            "Memory (MB)": type_data["memory_usage"] / (1024 * 1024)
        }
        rows.append(row)
    
    if not rows:
        return None
        
    return pd.DataFrame(rows)


def export_to_csv(dataframes, output_dir, input_file_name):
    """Export DataFrames to CSV files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_name = os.path.splitext(input_file_name)[0]
    
    for name, df in dataframes.items():
        if df is None:
            continue
            
        file_path = os.path.join(output_dir, f"{base_name}_{name}.csv")
        df.to_csv(file_path, index=False)
        print(f"Exported {name} to {file_path}")


def export_to_markdown(dataframes, output_dir, input_file_name):
    """Export DataFrames to Markdown files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_name = os.path.splitext(input_file_name)[0]
    file_path = os.path.join(output_dir, f"{base_name}_report.md")
    
    try:
        # Try to use tabulate if available
        from tabulate import tabulate
        has_tabulate = True
    except ImportError:
        # Fall back to a simple markdown table formatter
        has_tabulate = False
        print("Warning: tabulate package not found. Using simple markdown table format instead.")
        
        def simple_markdown_table(df):
            """Create a simple markdown table."""
            result = []
            
            # Header
            header = "| " + " | ".join(str(col) for col in df.columns) + " |"
            result.append(header)
            
            # Separator
            separator = "| " + " | ".join("---" for _ in df.columns) + " |"
            result.append(separator)
            
            # Rows
            for _, row in df.iterrows():
                row_str = "| " + " | ".join(str(val) for val in row) + " |"
                result.append(row_str)
            
            return "\n".join(result)
    
    with open(file_path, 'w') as f:
        f.write(f"# Profiling Results: {base_name}\n\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for name, df in dataframes.items():
            if df is None:
                continue
                
            f.write(f"## {name.replace('_', ' ').title()}\n\n")
            
            if has_tabulate:
                f.write(df.to_markdown(index=False))
            else:
                f.write(simple_markdown_table(df))
                
            f.write("\n\n")
    
    print(f"Exported Markdown report to {file_path}")


def export_to_html(dataframes, output_dir, input_file_name, data):
    """Export DataFrames to an HTML report."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_name = os.path.splitext(input_file_name)[0]
    file_path = os.path.join(output_dir, f"{base_name}_report.html")
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Profiling Report: {base_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333366; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metadata {{ color: #666; margin-bottom: 20px; }}
            .chart {{ margin: 20px 0; width: 100%; max-width: 800px; }}
        </style>
    </head>
    <body>
        <h1>Profiling Report: {base_name}</h1>
        <div class="metadata">
            <p>Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """
    
    # Add environment info if available
    if "environment" in data:
        env = data["environment"]
        html_content += f"""
            <h2>Environment</h2>
            <table>
                <tr><th>Device</th><td>{env.get('device', 'Unknown')}</td></tr>
                <tr><th>CPU Cores</th><td>{env.get('cpu_info', {}).get('count', 'Unknown')}</td></tr>
                <tr><th>Memory (GB)</th><td>{env.get('memory_total_gb', 'Unknown')}</td></tr>
        """
        
        if "gpu_info" in env:
            html_content += f"""
                <tr><th>GPU</th><td>{env['gpu_info'].get('name', 'Unknown')}</td></tr>
                <tr><th>GPU Memory (GB)</th><td>{env['gpu_info'].get('memory_total_gb', 'Unknown')}</td></tr>
            """
            
        html_content += "</table>\n"
    
    # Add optimization level recommendations if available
    if "optimization_levels" in data and "comparison" in data["optimization_levels"]:
        comparison = data["optimization_levels"]["comparison"]
        best_level = max(comparison.keys(), key=lambda x: comparison[x]["tokens_per_second"] if x != "comparison" else 0)
        
        html_content += f"""
            <h2>Optimization Recommendations</h2>
            <p>Best optimization level: <strong>{best_level}</strong> ({comparison[best_level]["tokens_per_second"]:.2f} tokens/sec)</p>
        """
    
    # Add pruning level recommendations if available
    if "pruning_comparison" in data and "optimized" in data["pruning_comparison"]:
        pruning_data = data["pruning_comparison"]["optimized"]
        best_level = max(pruning_data.keys(), key=lambda x: pruning_data[x]["tokens_per_second"])
        
        html_content += f"""
            <p>Best pruning level: <strong>{best_level}%</strong> ({pruning_data[best_level]["tokens_per_second"]:.2f} tokens/sec)</p>
        """
    
    html_content += "</div>\n"
    
    # Add tables for each DataFrame
    for name, df in dataframes.items():
        if df is None:
            continue
            
        html_content += f"<h2>{name.replace('_', ' ').title()}</h2>\n"
        html_content += df.to_html(index=False)
        html_content += "\n"
    
    # Close HTML content
    html_content += """
    </body>
    </html>
    """
    
    # Write to file
    with open(file_path, 'w') as f:
        f.write(html_content)
    
    print(f"Exported HTML report to {file_path}")


def create_charts(dataframes, output_dir, input_file_name, chart_format):
    """Create charts from the data."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_name = os.path.splitext(input_file_name)[0]
    charts_dir = os.path.join(output_dir, f"{base_name}_charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    # Create pruning comparison chart
    if "pruning_comparison" in dataframes and dataframes["pruning_comparison"] is not None:
        df = dataframes["pruning_comparison"]
        
        # Use a simpler direct plotting approach
        plt.figure(figsize=(10, 6))
        
        # Get unique pruning levels
        pruning_levels = sorted(df["Pruning Level"].unique())
        
        # Get data for each model type
        original_data = df[df["Model Type"] == "Original"]
        optimized_data = df[df["Model Type"] == "Optimized"]
        
        # Create x-positions
        x_positions = range(len(pruning_levels))
        
        # Plot each series
        if not original_data.empty:
            original_values = []
            for level in pruning_levels:
                row = original_data[original_data["Pruning Level"] == level]
                if not row.empty:
                    original_values.append(row["Tokens per Second"].values[0])
                else:
                    original_values.append(None)
            
            plt.plot(x_positions, original_values, 'o-', label="Original", color="dodgerblue", linewidth=2)
        
        if not optimized_data.empty:
            optimized_values = []
            for level in pruning_levels:
                row = optimized_data[optimized_data["Pruning Level"] == level]
                if not row.empty:
                    optimized_values.append(row["Tokens per Second"].values[0])
                else:
                    optimized_values.append(None)
            
            plt.plot(x_positions, optimized_values, 'o-', label="Optimized", color="green", linewidth=2)
        
        # Set up the plot
        plt.title("Performance by Pruning Level")
        plt.xlabel("Pruning Level (%)")
        plt.ylabel("Tokens per Second")
        plt.xticks(x_positions, pruning_levels)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add speedup annotations if available
        if "Speedup vs Original" in df.columns:
            speedup_df = df[df["Model Type"] == "Optimized"]
            for i, level in enumerate(pruning_levels):
                row = speedup_df[speedup_df["Pruning Level"] == level]
                if not row.empty and pd.notnull(row["Speedup vs Original"].values[0]):
                    speedup = row["Speedup vs Original"].values[0]
                    tokens_per_sec = row["Tokens per Second"].values[0]
                    plt.annotate(f"{speedup:.2f}x", 
                                xy=(i, tokens_per_sec),
                                xytext=(0, 10),
                                textcoords="offset points",
                                ha='center')
        
        plt.savefig(os.path.join(charts_dir, f"pruning_comparison.{chart_format}"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Created pruning comparison chart in {charts_dir}")
    
    # Create optimization levels chart
    if "optimization_levels" in dataframes and dataframes["optimization_levels"] is not None:
        df = dataframes["optimization_levels"]
        
        plt.figure(figsize=(10, 6))
        
        # Sort by performance
        df = df.sort_values("Tokens per Second", ascending=False)
        
        # Get data
        levels = df["Optimization Level"].tolist()
        tokens_per_second = df["Tokens per Second"].tolist()
        
        # Bar chart with numeric positions
        positions = range(len(levels))
        bars = plt.bar(positions, tokens_per_second, color="dodgerblue")
        
        # Configure plot
        plt.title("Performance by Optimization Level")
        plt.xlabel("Optimization Level")
        plt.ylabel("Tokens per Second")
        plt.xticks(positions, levels)
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, tokens_per_second)):
            plt.text(i, val + 0.2, f"{val:.2f}", ha='center', va='bottom')
        
        plt.savefig(os.path.join(charts_dir, f"optimization_levels.{chart_format}"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create latency comparison
        plt.figure(figsize=(10, 6))
        
        # Sort by latency (ascending)
        df = df.sort_values("First Token Time (s)")
        
        # Get data
        levels = df["Optimization Level"].tolist()
        latency_times = df["First Token Time (s)"].tolist()
        
        # Bar chart with numeric positions
        positions = range(len(levels))
        bars = plt.bar(positions, latency_times, color="coral")
        
        # Configure plot
        plt.title("First Token Latency by Optimization Level")
        plt.xlabel("Optimization Level")
        plt.ylabel("Latency (seconds)")
        plt.xticks(positions, levels)
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, latency_times)):
            plt.text(i, val + 0.02, f"{val:.2f}s", ha='center', va='bottom')
        
        plt.savefig(os.path.join(charts_dir, f"optimization_latency.{chart_format}"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Created optimization level charts in {charts_dir}")
    
    # Create model loading comparison chart
    if "model_loading" in dataframes and dataframes["model_loading"] is not None:
        df = dataframes["model_loading"]
        
        # Get model types
        model_types = df["Model Type"].tolist()
        positions = range(len(model_types))
        
        # Split up the plots
        plt.figure(figsize=(12, 5))
        
        # Plot load times
        plt.subplot(1, 3, 1)
        load_times = df["Load Time (s)"].tolist()
        colors = ['gray', 'dodgerblue', 'green'][:len(model_types)]  # Ensure we have the right number of colors
        
        # Create bars with numeric positions
        bars = plt.bar(positions, load_times, color=colors)
        plt.title('Model Loading Time')
        plt.ylabel('Time (seconds)')
        plt.xticks(positions, model_types)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, val in enumerate(load_times):
            plt.text(i, val + 0.05, f"{val:.2f}s", ha='center', va='bottom')
        
        # Plot parameter counts
        plt.subplot(1, 3, 2)
        param_counts = (df["Parameters"] / 1000000).tolist()  # Convert to millions
        
        # Create bars with numeric positions
        bars = plt.bar(positions, param_counts, color=colors)
        plt.title('Model Parameters')
        plt.ylabel('Parameters (millions)')
        plt.xticks(positions, model_types)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, val in enumerate(param_counts):
            plt.text(i, val + 0.5, f"{val:.1f}M", ha='center', va='bottom')
        
        # Plot memory usage
        plt.subplot(1, 3, 3)
        memory_usage = df["Memory (MB)"].tolist()
        
        # Create bars with numeric positions
        bars = plt.bar(positions, memory_usage, color=colors)
        plt.title('Memory Usage')
        plt.ylabel('Memory (MB)')
        plt.xticks(positions, model_types)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, val in enumerate(memory_usage):
            plt.text(i, val + 10, f"{val:.0f}MB", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f"model_loading.{chart_format}"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Created model loading chart in {charts_dir}")


def main():
    """Main function."""
    # Parse arguments
    args = setup_args()
    
    # Load profiling data
    data = load_profiling_data(args.input_file)
    if data is None:
        return
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.dirname(args.input_file)
    else:
        output_dir = args.output_dir
    
    # Extract input file name
    input_file_name = os.path.basename(args.input_file)
    
    # Extract data into DataFrames
    dataframes = {
        "pruning_comparison": extract_pruning_comparison(data),
        "optimization_levels": extract_optimization_levels(data),
        "model_loading": extract_model_loading(data)
    }
    
    # Export based on format
    if args.format in ["csv", "all"]:
        export_to_csv(dataframes, output_dir, input_file_name)
    
    if args.format in ["markdown", "all"]:
        export_to_markdown(dataframes, output_dir, input_file_name)
    
    if args.format in ["html", "all"]:
        export_to_html(dataframes, output_dir, input_file_name, data)
    
    # Create charts if requested
    if args.create_charts:
        create_charts(dataframes, output_dir, input_file_name, args.chart_format)


if __name__ == "__main__":
    main()