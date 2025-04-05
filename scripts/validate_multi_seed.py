#!/usr/bin/env python
"""
Multi-Seed Validation of Neural Plasticity System

This script automates full validation of the neural plasticity system across:
- Multiple random seeds
- Various pruning strategies
- Different pruning levels
- Model architectures

It produces comprehensive statistical results including:
- Mean and standard deviation for all metrics
- Statistical significance tests (p-values)
- Formatted tables for publication
- Summary visualizations

Output is saved in markdown and LaTeX formats suitable for direct
inclusion in research papers.
"""

import os
import sys
import logging
import argparse
import json
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from scipy import stats
from tqdm import tqdm
import re
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)

# Import validation experiment runner
from scripts.run_validation_experiment import run_validation_experiment, ValidationConfig

# Default configuration
DEFAULT_CONFIG = {
    "experiments": ["single_cycle", "multi_cycle"],
    "models": ["distilgpt2"],
    "seeds": 5,
    "pruning_ratios": [0.1, 0.3, 0.5],
    "cycles": 5,
    "steps_per_cycle": 50,
    "methods": ["random", "magnitude", "entropy", "adaptive"],
    "output_format": ["md", "tex", "csv"]
}


class ValidationSummary:
    """
    Aggregates and analyzes validation results across experiments, methods, and seeds.
    
    This class provides functionality to:
    1. Collect results from multiple validation runs
    2. Perform statistical analysis
    3. Generate publication-ready tables and figures
    4. Export results in various formats
    """
    
    def __init__(self, output_dir: str, config: Dict[str, Any]):
        """
        Initialize validation summary.
        
        Args:
            output_dir: Directory containing validation results
            config: Configuration dictionary
        """
        self.output_dir = output_dir
        self.config = config
        self.results_dir = os.path.join(output_dir, "results")
        self.summary_dir = os.path.join(output_dir, "summary")
        
        # Create output directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)
        
        # Initialize results storage
        self.experiment_results = {}
        self.statistical_tests = {}
        self.aggregated_metrics = {}
        
        logger.info(f"Initialized ValidationSummary (output: {output_dir})")
    
    def add_experiment_result(self, experiment_type: str, result: Dict[str, Any]):
        """
        Add result from a validation experiment.
        
        Args:
            experiment_type: Type of experiment (e.g., "single_cycle")
            result: Result dictionary from experiment
        """
        if experiment_type not in self.experiment_results:
            self.experiment_results[experiment_type] = []
        
        self.experiment_results[experiment_type].append(result)
        
        # Save individual result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.results_dir, f"{experiment_type}_{timestamp}.json")
        
        with open(result_file, 'w') as f:
            # Convert result to JSON-serializable format
            serializable_result = self._make_serializable(result)
            json.dump(serializable_result, f, indent=2)
    
    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format"""
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(item) for item in obj)
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj
    
    def aggregate_results(self):
        """
        Aggregate results across experiments, methods, and seeds.
        
        This builds combined metrics and performs statistical analysis.
        """
        logger.info("Aggregating validation results")
        
        # Process each experiment type
        for experiment_type, results in self.experiment_results.items():
            logger.info(f"Processing {experiment_type} results")
            
            # Skip if no results
            if not results:
                continue
            
            # Initialize experiment metrics
            if experiment_type not in self.aggregated_metrics:
                self.aggregated_metrics[experiment_type] = {
                    "models": {},
                    "methods": {},
                    "overall": {}
                }
            
            # Collect metrics by model and method
            for result in results:
                # Process model results
                for model_name, model_data in result.get("models", {}).items():
                    if model_name not in self.aggregated_metrics[experiment_type]["models"]:
                        self.aggregated_metrics[experiment_type]["models"][model_name] = {
                            "methods": {}
                        }
                    
                    # Process method results for this model
                    for method_name, method_data in model_data.get("methods", {}).items():
                        if method_name not in self.aggregated_metrics[experiment_type]["models"][model_name]["methods"]:
                            self.aggregated_metrics[experiment_type]["models"][model_name]["methods"][method_name] = {
                                "recovery_rates": [],
                                "function_preservation": [],
                                "perplexity": []
                            }
                        
                        # Collect metrics
                        if experiment_type == "single_cycle":
                            # For single cycle, find best ratio and collect those metrics
                            best_ratio = method_data.get("best_ratio")
                            if best_ratio and best_ratio in method_data:
                                ratio_data = method_data[best_ratio]
                                
                                # Recovery rate
                                if "avg_recovery_rate" in ratio_data:
                                    self.aggregated_metrics[experiment_type]["models"][model_name]["methods"][method_name]["recovery_rates"].append(
                                        ratio_data["avg_recovery_rate"]
                                    )
                                
                                # Function preservation
                                if "avg_function_preservation" in ratio_data:
                                    self.aggregated_metrics[experiment_type]["models"][model_name]["methods"][method_name]["function_preservation"].append(
                                        ratio_data["avg_function_preservation"]
                                    )
                                
                                # Perplexity (from seed results)
                                for seed_result in ratio_data.get("seeds", []):
                                    if "post_recovery_perplexity" in seed_result:
                                        self.aggregated_metrics[experiment_type]["models"][model_name]["methods"][method_name]["perplexity"].append(
                                            seed_result["post_recovery_perplexity"]
                                        )
                                        
                        elif experiment_type == "multi_cycle":
                            # For multi cycle, collect metrics from cycle results
                            if "avg_recovery_rate" in method_data:
                                self.aggregated_metrics[experiment_type]["models"][model_name]["methods"][method_name]["recovery_rates"].append(
                                    method_data["avg_recovery_rate"]
                                )
                            
                            # Function preservation is trickier in multi-cycle, get from cycle results
                            cycle_results = method_data.get("cycle_results", [])
                            if cycle_results:
                                # Get last cycle's function tracking score
                                last_cycle = cycle_results[-1]
                                func_score = last_cycle.get("function_tracking", {}).get("overall_score")
                                if func_score is not None:
                                    self.aggregated_metrics[experiment_type]["models"][model_name]["methods"][method_name]["function_preservation"].append(
                                        func_score
                                    )
                            
                            # Final perplexity
                            if "final_perplexity" in method_data:
                                self.aggregated_metrics[experiment_type]["models"][model_name]["methods"][method_name]["perplexity"].append(
                                    method_data["final_perplexity"]
                                )
                
                # Process method results
                for method_name, method_data in result.get("methods", {}).items():
                    if method_name not in self.aggregated_metrics[experiment_type]["methods"]:
                        self.aggregated_metrics[experiment_type]["methods"][method_name] = {
                            "recovery_rates": [],
                            "function_preservation": [],
                            "perplexity": []
                        }
                    
                    # Collect overall method metrics
                    if "avg_recovery_rate" in method_data:
                        self.aggregated_metrics[experiment_type]["methods"][method_name]["recovery_rates"].append(
                            method_data["avg_recovery_rate"]
                        )
                    
                    if "avg_function_preservation" in method_data:
                        self.aggregated_metrics[experiment_type]["methods"][method_name]["function_preservation"].append(
                            method_data["avg_function_preservation"]
                        )
            
            # Calculate aggregate statistics for each method and model
            self._calculate_aggregate_statistics(experiment_type)
            
            # Perform statistical significance tests
            self._perform_statistical_tests(experiment_type)
        
        # Consolidate results across experiment types
        self._consolidate_results()
        
        logger.info("Completed result aggregation")
    
    def _calculate_aggregate_statistics(self, experiment_type):
        """
        Calculate aggregate statistics for an experiment type.
        
        Args:
            experiment_type: Type of experiment
        """
        # Process model-method metrics
        for model_name, model_data in self.aggregated_metrics[experiment_type]["models"].items():
            for method_name, method_metrics in model_data["methods"].items():
                # Recovery rate
                recovery_rates = method_metrics["recovery_rates"]
                if recovery_rates:
                    method_metrics["avg_recovery_rate"] = np.mean(recovery_rates)
                    method_metrics["std_recovery_rate"] = np.std(recovery_rates)
                    method_metrics["min_recovery_rate"] = np.min(recovery_rates)
                    method_metrics["max_recovery_rate"] = np.max(recovery_rates)
                
                # Function preservation
                func_preservation = method_metrics["function_preservation"]
                if func_preservation:
                    method_metrics["avg_function_preservation"] = np.mean(func_preservation)
                    method_metrics["std_function_preservation"] = np.std(func_preservation)
                    method_metrics["min_function_preservation"] = np.min(func_preservation)
                    method_metrics["max_function_preservation"] = np.max(func_preservation)
                
                # Perplexity
                perplexity = method_metrics["perplexity"]
                if perplexity:
                    method_metrics["avg_perplexity"] = np.mean(perplexity)
                    method_metrics["std_perplexity"] = np.std(perplexity)
                    method_metrics["min_perplexity"] = np.min(perplexity)
                    method_metrics["max_perplexity"] = np.max(perplexity)
        
        # Process overall method metrics
        for method_name, method_metrics in self.aggregated_metrics[experiment_type]["methods"].items():
            # Recovery rate
            recovery_rates = method_metrics["recovery_rates"]
            if recovery_rates:
                method_metrics["avg_recovery_rate"] = np.mean(recovery_rates)
                method_metrics["std_recovery_rate"] = np.std(recovery_rates)
                method_metrics["min_recovery_rate"] = np.min(recovery_rates)
                method_metrics["max_recovery_rate"] = np.max(recovery_rates)
            
            # Function preservation
            func_preservation = method_metrics["function_preservation"]
            if func_preservation:
                method_metrics["avg_function_preservation"] = np.mean(func_preservation)
                method_metrics["std_function_preservation"] = np.std(func_preservation)
                method_metrics["min_function_preservation"] = np.min(func_preservation)
                method_metrics["max_function_preservation"] = np.max(func_preservation)
            
            # Perplexity
            perplexity = method_metrics["perplexity"]
            if perplexity:
                method_metrics["avg_perplexity"] = np.mean(perplexity)
                method_metrics["std_perplexity"] = np.std(perplexity)
                method_metrics["min_perplexity"] = np.min(perplexity)
                method_metrics["max_perplexity"] = np.max(perplexity)
    
    def _perform_statistical_tests(self, experiment_type):
        """
        Perform statistical significance tests.
        
        Args:
            experiment_type: Type of experiment
        """
        if experiment_type not in self.statistical_tests:
            self.statistical_tests[experiment_type] = {
                "recovery_rate": {},
                "function_preservation": {},
                "perplexity": {}
            }
        
        # Find adaptive method results (baseline for comparison)
        adaptive_results = {}
        
        # Collect adaptive results for each model
        for model_name, model_data in self.aggregated_metrics[experiment_type]["models"].items():
            if "adaptive" in model_data["methods"]:
                adaptive_method = model_data["methods"]["adaptive"]
                
                adaptive_results[model_name] = {
                    "recovery_rates": adaptive_method.get("recovery_rates", []),
                    "function_preservation": adaptive_method.get("function_preservation", []),
                    "perplexity": adaptive_method.get("perplexity", [])
                }
        
        # If no adaptive results, can't do statistical tests
        if not adaptive_results:
            logger.warning(f"No adaptive method results found for {experiment_type}, skipping statistical tests")
            return
        
        # Perform tests for each metric and method
        metrics = ["recovery_rate", "function_preservation", "perplexity"]
        
        for metric in metrics:
            metric_key = f"{metric}s"  # e.g., "recovery_rates"
            
            for method in self.aggregated_metrics[experiment_type]["methods"]:
                if method == "adaptive":
                    continue  # Skip comparing adaptive to itself
                
                # Collect all values for this method across models
                method_values = []
                for model_name, model_data in self.aggregated_metrics[experiment_type]["models"].items():
                    if method in model_data["methods"]:
                        method_values.extend(model_data["methods"][method].get(metric_key, []))
                
                # Collect all adaptive values
                adaptive_values = []
                for model_values in adaptive_results.values():
                    adaptive_values.extend(model_values.get(metric_key, []))
                
                # Perform t-test if enough data
                if len(method_values) >= 3 and len(adaptive_values) >= 3:
                    stat, p_value = stats.ttest_ind(adaptive_values, method_values)
                    
                    # Store result
                    self.statistical_tests[experiment_type][metric][method] = {
                        "t_statistic": stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                        "sample_sizes": {
                            "adaptive": len(adaptive_values),
                            method: len(method_values)
                        }
                    }
                else:
                    logger.warning(f"Not enough data for statistical test of {metric} between adaptive and {method}")
    
    def _consolidate_results(self):
        """Consolidate results across experiment types"""
        # Initialize overall metrics
        self.aggregated_metrics["overall"] = {
            "methods": {},
            "models": {}
        }
        
        # Collect method metrics across experiments
        all_methods = set()
        for exp_type in self.aggregated_metrics:
            if exp_type == "overall":
                continue
                
            for method in self.aggregated_metrics[exp_type].get("methods", {}):
                all_methods.add(method)
        
        # Initialize method metrics
        for method in all_methods:
            self.aggregated_metrics["overall"]["methods"][method] = {
                "recovery_rates": [],
                "function_preservation": [],
                "perplexity": []
            }
        
        # Collect metrics across experiments
        for exp_type in self.aggregated_metrics:
            if exp_type == "overall":
                continue
                
            for method, method_data in self.aggregated_metrics[exp_type].get("methods", {}).items():
                # Only use methods that have computed averages
                if "avg_recovery_rate" in method_data:
                    self.aggregated_metrics["overall"]["methods"][method]["recovery_rates"].append(
                        method_data["avg_recovery_rate"]
                    )
                
                if "avg_function_preservation" in method_data:
                    self.aggregated_metrics["overall"]["methods"][method]["function_preservation"].append(
                        method_data["avg_function_preservation"]
                    )
                
                if "avg_perplexity" in method_data:
                    self.aggregated_metrics["overall"]["methods"][method]["perplexity"].append(
                        method_data["avg_perplexity"]
                    )
        
        # Calculate overall statistics
        for method, method_metrics in self.aggregated_metrics["overall"]["methods"].items():
            # Recovery rate
            recovery_rates = method_metrics["recovery_rates"]
            if recovery_rates:
                method_metrics["avg_recovery_rate"] = np.mean(recovery_rates)
                method_metrics["std_recovery_rate"] = np.std(recovery_rates)
            
            # Function preservation
            func_preservation = method_metrics["function_preservation"]
            if func_preservation:
                method_metrics["avg_function_preservation"] = np.mean(func_preservation)
                method_metrics["std_function_preservation"] = np.std(func_preservation)
            
            # Perplexity
            perplexity = method_metrics["perplexity"]
            if perplexity:
                method_metrics["avg_perplexity"] = np.mean(perplexity)
                method_metrics["std_perplexity"] = np.std(perplexity)
    
    def generate_reports(self):
        """Generate summary reports in different formats"""
        logger.info("Generating summary reports")
        
        # Ensure results are aggregated
        if not self.aggregated_metrics:
            logger.warning("No aggregated metrics available, aggregating results first")
            self.aggregate_results()
        
        # Generate reports in each requested format
        for output_format in self.config.get("output_format", ["md"]):
            if output_format.lower() == "md":
                self._generate_markdown_report()
            elif output_format.lower() == "tex":
                self._generate_latex_report()
            elif output_format.lower() == "csv":
                self._generate_csv_report()
            else:
                logger.warning(f"Unknown output format: {output_format}")
        
        # Generate visualizations
        self._generate_visualizations()
        
        logger.info("Completed report generation")
    
    def _generate_markdown_report(self):
        """Generate markdown summary report"""
        report_file = os.path.join(self.summary_dir, "validation_summary.md")
        
        with open(report_file, 'w') as f:
            f.write("# Neural Plasticity Validation Results\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write overall summary
            f.write("## Overall Summary\n\n")
            
            # Recovery rate table
            f.write("### Recovery Rate\n\n")
            f.write("| Method | Recovery Rate (mean ± std) | Significance vs Adaptive |\n")
            f.write("|--------|---------------------------|-------------------------|\n")
            
            adaptive_recovery = self.aggregated_metrics.get("overall", {}).get("methods", {}).get("adaptive", {}).get("avg_recovery_rate")
            
            for method, method_data in self.aggregated_metrics.get("overall", {}).get("methods", {}).items():
                if "avg_recovery_rate" in method_data and "std_recovery_rate" in method_data:
                    avg = method_data["avg_recovery_rate"]
                    std = method_data["std_recovery_rate"]
                    
                    # Check for significance
                    significance = ""
                    for exp_type in self.statistical_tests:
                        if method in self.statistical_tests[exp_type].get("recovery_rate", {}):
                            test_result = self.statistical_tests[exp_type]["recovery_rate"][method]
                            if test_result["significant"]:
                                p_value = test_result["p_value"]
                                if avg > adaptive_recovery:
                                    significance = f"Better (p={p_value:.3f})"
                                else:
                                    significance = f"Worse (p={p_value:.3f})"
                                break
                    
                    if not significance and method != "adaptive":
                        significance = "Not significant"
                    
                    f.write(f"| {method} | {avg:.4f} ± {std:.4f} | {significance} |\n")
            
            # Function preservation table
            f.write("\n### Function Preservation\n\n")
            f.write("| Method | Function Preservation (mean ± std) | Significance vs Adaptive |\n")
            f.write("|--------|-----------------------------------|-------------------------|\n")
            
            adaptive_fp = self.aggregated_metrics.get("overall", {}).get("methods", {}).get("adaptive", {}).get("avg_function_preservation")
            
            for method, method_data in self.aggregated_metrics.get("overall", {}).get("methods", {}).items():
                if "avg_function_preservation" in method_data and "std_function_preservation" in method_data:
                    avg = method_data["avg_function_preservation"]
                    std = method_data["std_function_preservation"]
                    
                    # Check for significance
                    significance = ""
                    for exp_type in self.statistical_tests:
                        if method in self.statistical_tests[exp_type].get("function_preservation", {}):
                            test_result = self.statistical_tests[exp_type]["function_preservation"][method]
                            if test_result["significant"]:
                                p_value = test_result["p_value"]
                                if avg > adaptive_fp:
                                    significance = f"Better (p={p_value:.3f})"
                                else:
                                    significance = f"Worse (p={p_value:.3f})"
                                break
                    
                    if not significance and method != "adaptive":
                        significance = "Not significant"
                    
                    f.write(f"| {method} | {avg:.4f} ± {std:.4f} | {significance} |\n")
            
            # Write per-experiment summaries
            for exp_type in self.aggregated_metrics:
                if exp_type == "overall":
                    continue
                
                f.write(f"\n## {exp_type.replace('_', ' ').title()} Experiment\n\n")
                
                # Per-model results
                for model_name, model_data in self.aggregated_metrics[exp_type]["models"].items():
                    f.write(f"\n### Model: {model_name}\n\n")
                    
                    # Recovery rate table
                    f.write("#### Recovery Rate\n\n")
                    f.write("| Method | Recovery Rate (mean ± std) |\n")
                    f.write("|--------|---------------------------|\n")
                    
                    for method, method_data in model_data["methods"].items():
                        if "avg_recovery_rate" in method_data and "std_recovery_rate" in method_data:
                            avg = method_data["avg_recovery_rate"]
                            std = method_data["std_recovery_rate"]
                            f.write(f"| {method} | {avg:.4f} ± {std:.4f} |\n")
                    
                    # Function preservation table
                    f.write("\n#### Function Preservation\n\n")
                    f.write("| Method | Function Preservation (mean ± std) |\n")
                    f.write("|--------|-----------------------------------|\n")
                    
                    for method, method_data in model_data["methods"].items():
                        if "avg_function_preservation" in method_data and "std_function_preservation" in method_data:
                            avg = method_data["avg_function_preservation"]
                            std = method_data["std_function_preservation"]
                            f.write(f"| {method} | {avg:.4f} ± {std:.4f} |\n")
                    
                    # Perplexity table
                    f.write("\n#### Perplexity\n\n")
                    f.write("| Method | Perplexity (mean ± std) |\n")
                    f.write("|--------|-----------------------|\n")
                    
                    for method, method_data in model_data["methods"].items():
                        if "avg_perplexity" in method_data and "std_perplexity" in method_data:
                            avg = method_data["avg_perplexity"]
                            std = method_data["std_perplexity"]
                            f.write(f"| {method} | {avg:.2f} ± {std:.2f} |\n")
                
                # Statistical significance
                if exp_type in self.statistical_tests:
                    f.write("\n### Statistical Tests\n\n")
                    f.write("Comparing adaptive method to each baseline:\n\n")
                    
                    for metric, tests in self.statistical_tests[exp_type].items():
                        f.write(f"\n#### {metric.replace('_', ' ').title()}\n\n")
                        f.write("| Method | t-statistic | p-value | Significant? |\n")
                        f.write("|--------|------------|---------|-------------|\n")
                        
                        for method, test_result in tests.items():
                            t_stat = test_result["t_statistic"]
                            p_value = test_result["p_value"]
                            significant = "Yes" if test_result["significant"] else "No"
                            
                            f.write(f"| {method} | {t_stat:.4f} | {p_value:.4f} | {significant} |\n")
            
            # Write conclusions
            f.write("\n## Key Findings\n\n")
            
            # Best method overall
            best_method = None
            best_recovery = -1
            
            for method, method_data in self.aggregated_metrics.get("overall", {}).get("methods", {}).items():
                if "avg_recovery_rate" in method_data and method_data["avg_recovery_rate"] > best_recovery:
                    best_recovery = method_data["avg_recovery_rate"]
                    best_method = method
            
            if best_method:
                f.write(f"- **Best overall method**: {best_method} (recovery rate: {best_recovery:.4f})\n")
            
            # Adaptive method comparison
            if "adaptive" in self.aggregated_metrics.get("overall", {}).get("methods", {}):
                adaptive_data = self.aggregated_metrics["overall"]["methods"]["adaptive"]
                
                if "avg_recovery_rate" in adaptive_data:
                    better_methods = []
                    worse_methods = []
                    
                    for method, method_data in self.aggregated_metrics["overall"]["methods"].items():
                        if method == "adaptive":
                            continue
                            
                        if "avg_recovery_rate" in method_data:
                            if method_data["avg_recovery_rate"] > adaptive_data["avg_recovery_rate"]:
                                better_methods.append(method)
                            else:
                                worse_methods.append(method)
                    
                    if better_methods:
                        f.write(f"- Methods outperforming adaptive: {', '.join(better_methods)}\n")
                    else:
                        f.write("- **Adaptive method outperforms all baselines** in recovery rate\n")
                    
                if "avg_function_preservation" in adaptive_data:
                    f.write(f"- Adaptive method function preservation: {adaptive_data['avg_function_preservation']:.4f} ± {adaptive_data.get('std_function_preservation', 0):.4f}\n")
            
            # Statistical significance summary
            significant_findings = []
            
            for exp_type in self.statistical_tests:
                for metric, tests in self.statistical_tests[exp_type].items():
                    for method, test_result in tests.items():
                        if test_result["significant"]:
                            direction = "better than" if test_result["t_statistic"] < 0 else "worse than"
                            finding = f"For {metric}, {method} is significantly {direction} adaptive (p={test_result['p_value']:.4f})"
                            significant_findings.append(finding)
            
            if significant_findings:
                f.write("\n### Statistically Significant Findings\n\n")
                for finding in significant_findings:
                    f.write(f"- {finding}\n")
            else:
                f.write("\n- No statistically significant differences found between methods\n")
        
        logger.info(f"Generated markdown report: {report_file}")
    
    def _generate_latex_report(self):
        """Generate LaTeX summary report"""
        report_file = os.path.join(self.summary_dir, "validation_summary.tex")
        
        with open(report_file, 'w') as f:
            # LaTeX preamble
            f.write("% Neural Plasticity Validation Results\n")
            f.write("% Generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
            f.write("\\documentclass{article}\n")
            f.write("\\usepackage{booktabs}\n")
            f.write("\\usepackage{multirow}\n")
            f.write("\\usepackage{graphicx}\n")
            f.write("\\usepackage{caption}\n")
            f.write("\\usepackage{subcaption}\n")
            f.write("\\usepackage{float}\n")
            f.write("\\begin{document}\n\n")
            
            # Title
            f.write("\\section{Neural Plasticity Validation Results}\n\n")
            
            # Overall summary
            f.write("\\subsection{Overall Summary}\n\n")
            
            # Overall recovery rate table
            f.write("\\begin{table}[h]\n")
            f.write("\\caption{Recovery Rate by Method}\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\toprule\n")
            f.write("Method & Recovery Rate (mean $\\pm$ std) & Significance vs Adaptive \\\\\n")
            f.write("\\midrule\n")
            
            adaptive_recovery = self.aggregated_metrics.get("overall", {}).get("methods", {}).get("adaptive", {}).get("avg_recovery_rate")
            
            for method, method_data in self.aggregated_metrics.get("overall", {}).get("methods", {}).items():
                if "avg_recovery_rate" in method_data and "std_recovery_rate" in method_data:
                    avg = method_data["avg_recovery_rate"]
                    std = method_data["std_recovery_rate"]
                    
                    # Check for significance
                    significance = ""
                    for exp_type in self.statistical_tests:
                        if method in self.statistical_tests[exp_type].get("recovery_rate", {}):
                            test_result = self.statistical_tests[exp_type]["recovery_rate"][method]
                            if test_result["significant"]:
                                p_value = test_result["p_value"]
                                if avg > adaptive_recovery:
                                    significance = f"Better ($p={p_value:.3f}$)"
                                else:
                                    significance = f"Worse ($p={p_value:.3f}$)"
                                break
                    
                    if not significance and method != "adaptive":
                        significance = "Not significant"
                    
                    f.write(f"{method} & {avg:.4f} $\\pm$ {std:.4f} & {significance} \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
            
            # Overall function preservation table
            f.write("\\begin{table}[h]\n")
            f.write("\\caption{Function Preservation by Method}\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\toprule\n")
            f.write("Method & Function Preservation (mean $\\pm$ std) & Significance vs Adaptive \\\\\n")
            f.write("\\midrule\n")
            
            adaptive_fp = self.aggregated_metrics.get("overall", {}).get("methods", {}).get("adaptive", {}).get("avg_function_preservation")
            
            for method, method_data in self.aggregated_metrics.get("overall", {}).get("methods", {}).items():
                if "avg_function_preservation" in method_data and "std_function_preservation" in method_data:
                    avg = method_data["avg_function_preservation"]
                    std = method_data["std_function_preservation"]
                    
                    # Check for significance
                    significance = ""
                    for exp_type in self.statistical_tests:
                        if method in self.statistical_tests[exp_type].get("function_preservation", {}):
                            test_result = self.statistical_tests[exp_type]["function_preservation"][method]
                            if test_result["significant"]:
                                p_value = test_result["p_value"]
                                if avg > adaptive_fp:
                                    significance = f"Better ($p={p_value:.3f}$)"
                                else:
                                    significance = f"Worse ($p={p_value:.3f}$)"
                                break
                    
                    if not significance and method != "adaptive":
                        significance = "Not significant"
                    
                    f.write(f"{method} & {avg:.4f} $\\pm$ {std:.4f} & {significance} \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
            
            # Per-experiment tables
            for exp_type in self.aggregated_metrics:
                if exp_type == "overall":
                    continue
                
                exp_title = exp_type.replace('_', ' ').title()
                f.write(f"\\subsection{{{exp_title} Experiment}}\n\n")
                
                # Recovery rate table by model
                f.write("\\begin{table}[h]\n")
                f.write(f"\\caption{{{exp_title} Recovery Rate by Model and Method}}\n")
                f.write("\\centering\n")
                f.write("\\begin{tabular}{llc}\n")
                f.write("\\toprule\n")
                f.write("Model & Method & Recovery Rate (mean $\\pm$ std) \\\\\n")
                f.write("\\midrule\n")
                
                for model_name, model_data in self.aggregated_metrics[exp_type]["models"].items():
                    first_row = True
                    num_methods = len([m for m in model_data["methods"] if "avg_recovery_rate" in model_data["methods"][m]])
                    
                    for method, method_data in model_data["methods"].items():
                        if "avg_recovery_rate" in method_data and "std_recovery_rate" in method_data:
                            avg = method_data["avg_recovery_rate"]
                            std = method_data["std_recovery_rate"]
                            
                            if first_row:
                                f.write(f"\\multirow{{{num_methods}}}{{*}}{{{model_name}}} & {method} & {avg:.4f} $\\pm$ {std:.4f} \\\\\n")
                                first_row = False
                            else:
                                f.write(f"& {method} & {avg:.4f} $\\pm$ {std:.4f} \\\\\n")
                    
                    # Add a midrule between models
                    if num_methods > 0:
                        f.write("\\midrule\n")
                
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n\n")
                
                # Function preservation table by model
                f.write("\\begin{table}[h]\n")
                f.write(f"\\caption{{{exp_title} Function Preservation by Model and Method}}\n")
                f.write("\\centering\n")
                f.write("\\begin{tabular}{llc}\n")
                f.write("\\toprule\n")
                f.write("Model & Method & Function Preservation (mean $\\pm$ std) \\\\\n")
                f.write("\\midrule\n")
                
                for model_name, model_data in self.aggregated_metrics[exp_type]["models"].items():
                    first_row = True
                    num_methods = len([m for m in model_data["methods"] if "avg_function_preservation" in model_data["methods"][m]])
                    
                    for method, method_data in model_data["methods"].items():
                        if "avg_function_preservation" in method_data and "std_function_preservation" in method_data:
                            avg = method_data["avg_function_preservation"]
                            std = method_data["std_function_preservation"]
                            
                            if first_row:
                                f.write(f"\\multirow{{{num_methods}}}{{*}}{{{model_name}}} & {method} & {avg:.4f} $\\pm$ {std:.4f} \\\\\n")
                                first_row = False
                            else:
                                f.write(f"& {method} & {avg:.4f} $\\pm$ {std:.4f} \\\\\n")
                    
                    # Add a midrule between models
                    if num_methods > 0:
                        f.write("\\midrule\n")
                
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n\n")
                
                # Statistical significance table
                if exp_type in self.statistical_tests:
                    f.write("\\subsubsection{Statistical Tests}\n\n")
                    
                    for metric, tests in self.statistical_tests[exp_type].items():
                        metric_title = metric.replace('_', ' ').title()
                        
                        f.write("\\begin{table}[h]\n")
                        f.write(f"\\caption{{{exp_title} Statistical Tests for {metric_title}}}\n")
                        f.write("\\centering\n")
                        f.write("\\begin{tabular}{lccc}\n")
                        f.write("\\toprule\n")
                        f.write("Method & t-statistic & p-value & Significant? \\\\\n")
                        f.write("\\midrule\n")
                        
                        for method, test_result in tests.items():
                            t_stat = test_result["t_statistic"]
                            p_value = test_result["p_value"]
                            significant = "Yes" if test_result["significant"] else "No"
                            
                            f.write(f"{method} & {t_stat:.4f} & {p_value:.4f} & {significant} \\\\\n")
                        
                        f.write("\\bottomrule\n")
                        f.write("\\end{tabular}\n")
                        f.write("\\end{table}\n\n")
            
            # End document
            f.write("\\end{document}\n")
        
        logger.info(f"Generated LaTeX report: {report_file}")
    
    def _generate_csv_report(self):
        """Generate CSV summary report"""
        # Create overall summary CSV
        overall_file = os.path.join(self.summary_dir, "overall_summary.csv")
        
        with open(overall_file, 'w') as f:
            # Write header
            f.write("Method,Recovery Rate (mean),Recovery Rate (std),Function Preservation (mean),Function Preservation (std),Perplexity (mean),Perplexity (std)\n")
            
            # Write data
            for method, method_data in self.aggregated_metrics.get("overall", {}).get("methods", {}).items():
                recovery_mean = method_data.get("avg_recovery_rate", "")
                recovery_std = method_data.get("std_recovery_rate", "")
                fp_mean = method_data.get("avg_function_preservation", "")
                fp_std = method_data.get("std_function_preservation", "")
                ppl_mean = method_data.get("avg_perplexity", "")
                ppl_std = method_data.get("std_perplexity", "")
                
                f.write(f"{method},{recovery_mean},{recovery_std},{fp_mean},{fp_std},{ppl_mean},{ppl_std}\n")
        
        # Create experiment-specific CSVs
        for exp_type in self.aggregated_metrics:
            if exp_type == "overall":
                continue
                
            # Model-method metrics
            model_file = os.path.join(self.summary_dir, f"{exp_type}_model_metrics.csv")
            
            with open(model_file, 'w') as f:
                # Write header
                f.write("Model,Method,Recovery Rate (mean),Recovery Rate (std),Function Preservation (mean),Function Preservation (std),Perplexity (mean),Perplexity (std)\n")
                
                # Write data
                for model_name, model_data in self.aggregated_metrics[exp_type]["models"].items():
                    for method, method_data in model_data["methods"].items():
                        recovery_mean = method_data.get("avg_recovery_rate", "")
                        recovery_std = method_data.get("std_recovery_rate", "")
                        fp_mean = method_data.get("avg_function_preservation", "")
                        fp_std = method_data.get("std_function_preservation", "")
                        ppl_mean = method_data.get("avg_perplexity", "")
                        ppl_std = method_data.get("std_perplexity", "")
                        
                        f.write(f"{model_name},{method},{recovery_mean},{recovery_std},{fp_mean},{fp_std},{ppl_mean},{ppl_std}\n")
            
            # Statistical tests
            if exp_type in self.statistical_tests:
                stats_file = os.path.join(self.summary_dir, f"{exp_type}_statistical_tests.csv")
                
                with open(stats_file, 'w') as f:
                    # Write header
                    f.write("Metric,Method,t-statistic,p-value,Significant\n")
                    
                    # Write data
                    for metric, tests in self.statistical_tests[exp_type].items():
                        for method, test_result in tests.items():
                            t_stat = test_result["t_statistic"]
                            p_value = test_result["p_value"]
                            significant = "Yes" if test_result["significant"] else "No"
                            
                            f.write(f"{metric},{method},{t_stat},{p_value},{significant}\n")
        
        logger.info(f"Generated CSV reports in {self.summary_dir}")
    
    def _generate_visualizations(self):
        """Generate summary visualizations"""
        viz_dir = os.path.join(self.summary_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Only proceed if there are metrics to visualize
        if not self.aggregated_metrics:
            logger.warning("No metrics to visualize")
            return
        
        # Plot overall recovery rate comparison
        if "overall" in self.aggregated_metrics and self.aggregated_metrics["overall"]["methods"]:
            # Recovery rate by method
            methods = []
            means = []
            errors = []
            
            for method, method_data in self.aggregated_metrics["overall"]["methods"].items():
                if "avg_recovery_rate" in method_data and "std_recovery_rate" in method_data:
                    methods.append(method)
                    means.append(method_data["avg_recovery_rate"])
                    errors.append(method_data["std_recovery_rate"])
            
            if methods:
                plt.figure(figsize=(10, 6))
                plt.bar(methods, means, yerr=errors)
                plt.ylim(0, 1.1)
                plt.xlabel('Method')
                plt.ylabel('Recovery Rate')
                plt.title('Recovery Rate by Method (Overall)')
                
                # Add value labels
                for i, v in enumerate(means):
                    plt.text(i, v + 0.05, f"{v:.2f}", ha='center')
                
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, "overall_recovery_rate.png"))
                plt.close()
                
                # Also save as PDF for publication
                plt.figure(figsize=(10, 6))
                plt.bar(methods, means, yerr=errors)
                plt.ylim(0, 1.1)
                plt.xlabel('Method')
                plt.ylabel('Recovery Rate')
                plt.title('Recovery Rate by Method (Overall)')
                
                # Add value labels
                for i, v in enumerate(means):
                    plt.text(i, v + 0.05, f"{v:.2f}", ha='center')
                
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, "overall_recovery_rate.pdf"))
                plt.close()
            
            # Function preservation by method
            methods = []
            means = []
            errors = []
            
            for method, method_data in self.aggregated_metrics["overall"]["methods"].items():
                if "avg_function_preservation" in method_data and "std_function_preservation" in method_data:
                    methods.append(method)
                    means.append(method_data["avg_function_preservation"])
                    errors.append(method_data["std_function_preservation"])
            
            if methods:
                plt.figure(figsize=(10, 6))
                plt.bar(methods, means, yerr=errors)
                plt.ylim(0, 1.1)
                plt.xlabel('Method')
                plt.ylabel('Function Preservation')
                plt.title('Function Preservation by Method (Overall)')
                
                # Add value labels
                for i, v in enumerate(means):
                    plt.text(i, v + 0.05, f"{v:.2f}", ha='center')
                
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, "overall_function_preservation.png"))
                plt.close()
                
                # Also save as PDF for publication
                plt.figure(figsize=(10, 6))
                plt.bar(methods, means, yerr=errors)
                plt.ylim(0, 1.1)
                plt.xlabel('Method')
                plt.ylabel('Function Preservation')
                plt.title('Function Preservation by Method (Overall)')
                
                # Add value labels
                for i, v in enumerate(means):
                    plt.text(i, v + 0.05, f"{v:.2f}", ha='center')
                
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, "overall_function_preservation.pdf"))
                plt.close()
        
        # Create experiment-specific visualizations
        for exp_type in self.aggregated_metrics:
            if exp_type == "overall":
                continue
                
            exp_title = exp_type.replace('_', ' ').title()
            
            # Recovery rate by model and method
            plt.figure(figsize=(12, 6))
            
            models = list(self.aggregated_metrics[exp_type]["models"].keys())
            methods = set()
            
            for model_data in self.aggregated_metrics[exp_type]["models"].values():
                for method in model_data["methods"]:
                    if "avg_recovery_rate" in model_data["methods"][method]:
                        methods.add(method)
            
            methods = sorted(methods)
            
            # Create grouped bar chart
            x = np.arange(len(models))
            width = 0.8 / len(methods)
            
            for i, method in enumerate(methods):
                means = []
                errors = []
                
                for model in models:
                    method_data = self.aggregated_metrics[exp_type]["models"][model]["methods"].get(method, {})
                    if "avg_recovery_rate" in method_data and "std_recovery_rate" in method_data:
                        means.append(method_data["avg_recovery_rate"])
                        errors.append(method_data["std_recovery_rate"])
                    else:
                        means.append(0)
                        errors.append(0)
                
                offset = width * i - width * (len(methods) - 1) / 2
                bars = plt.bar(x + offset, means, width, yerr=errors, label=method)
            
            plt.xlabel('Model')
            plt.ylabel('Recovery Rate')
            plt.title(f'Recovery Rate by Model and Method ({exp_title})')
            plt.xticks(x, models)
            plt.legend()
            plt.ylim(0, 1.1)
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"{exp_type}_recovery_rate.png"))
            plt.close()
            
            # Function preservation by model and method
            plt.figure(figsize=(12, 6))
            
            methods = set()
            
            for model_data in self.aggregated_metrics[exp_type]["models"].values():
                for method in model_data["methods"]:
                    if "avg_function_preservation" in model_data["methods"][method]:
                        methods.add(method)
            
            methods = sorted(methods)
            
            # Create grouped bar chart
            x = np.arange(len(models))
            width = 0.8 / len(methods)
            
            for i, method in enumerate(methods):
                means = []
                errors = []
                
                for model in models:
                    method_data = self.aggregated_metrics[exp_type]["models"][model]["methods"].get(method, {})
                    if "avg_function_preservation" in method_data and "std_function_preservation" in method_data:
                        means.append(method_data["avg_function_preservation"])
                        errors.append(method_data["std_function_preservation"])
                    else:
                        means.append(0)
                        errors.append(0)
                
                offset = width * i - width * (len(methods) - 1) / 2
                plt.bar(x + offset, means, width, yerr=errors, label=method)
            
            plt.xlabel('Model')
            plt.ylabel('Function Preservation')
            plt.title(f'Function Preservation by Model and Method ({exp_title})')
            plt.xticks(x, models)
            plt.legend()
            plt.ylim(0, 1.1)
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"{exp_type}_function_preservation.png"))
            plt.close()
        
        logger.info(f"Generated visualizations in {viz_dir}")


def run_validation(args):
    """
    Run the multi-seed validation process.
    
    Args:
        args: Command line arguments
    
    Returns:
        Summary of validation results
    """
    # Set up output directory
    output_dir = args.output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(output_dir, f"multi_seed_validation_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save configuration
    config = {
        "experiments": args.experiments.split(','),
        "models": args.models.split(','),
        "seeds": args.seeds,
        "pruning_ratios": [float(r) for r in args.pruning_ratios.split(',')],
        "cycles": args.cycles,
        "steps_per_cycle": args.steps,
        "methods": args.methods.split(','),
        "output_format": args.output_format.split(','),
        "disable_rl": args.disable_rl
    }
    
    config_file = os.path.join(experiment_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize validation summary
    summary = ValidationSummary(experiment_dir, config)
    
    # Run experiments
    for experiment_type in config["experiments"]:
        logger.info(f"Running {experiment_type} experiment")
        
        # Create experiment directory
        exp_dir = os.path.join(experiment_dir, experiment_type)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Run the experiment
        args_dict = vars(args).copy()
        args_dict["experiment"] = experiment_type
        args_dict["output_dir"] = exp_dir
        
        # Create experiment configuration
        exp_config = argparse.Namespace(**args_dict)
        
        # Run the experiment
        result = run_validation_experiment(exp_config)
        
        # Add result to summary
        summary.add_experiment_result(experiment_type, result)
    
    # Aggregate results and generate reports
    summary.aggregate_results()
    summary.generate_reports()
    
    logger.info(f"Validation complete. Results saved to {experiment_dir}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Seed Validation of Neural Plasticity System")
    parser.add_argument("--experiments", type=str, default="single_cycle,multi_cycle",
                      help="Experiments to run (comma-separated)")
    parser.add_argument("--models", type=str, default="distilgpt2",
                      help="Models to test (comma-separated)")
    parser.add_argument("--output_dir", type=str, default="./output/validation",
                      help="Output directory")
    parser.add_argument("--seeds", type=int, default=5,
                      help="Number of random seeds")
    parser.add_argument("--pruning_ratios", type=str, default="0.1,0.3,0.5",
                      help="Pruning ratios to test (comma-separated)")
    parser.add_argument("--cycles", type=int, default=5,
                      help="Number of cycles for multi-cycle experiment")
    parser.add_argument("--steps", type=int, default=50,
                      help="Training steps per cycle")
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Batch size for training")
    parser.add_argument("--device", type=str, default=None,
                      help="Device (cuda or cpu)")
    parser.add_argument("--dataset", type=str, default=None,
                      help="Dataset to use (leave empty for synthetic)")
    parser.add_argument("--disable_rl", action="store_true",
                      help="Disable RL controller")
    parser.add_argument("--methods", type=str, default="random,magnitude,entropy,adaptive",
                      help="Methods to test (comma-separated)")
    parser.add_argument("--output_format", type=str, default="md,tex,csv",
                      help="Output formats (comma-separated)")
    
    args = parser.parse_args()
    
    run_validation(args)