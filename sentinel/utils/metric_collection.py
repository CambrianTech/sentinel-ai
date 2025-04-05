"""
Metric Collection and Analysis Pipeline for Sentinel-AI

This module provides a comprehensive system for collecting, analyzing, and visualizing 
metrics for adaptive transformer models. It brings together performance metrics, 
head-level metrics, pruning pattern analysis, and controller gate values to provide 
insights into model behavior during training and inference.
"""

import os
import time
import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from collections import defaultdict
import logging
from datetime import datetime

# Import metrics functions from existing modules
from sentinel.utils.metrics import (
    calculate_metrics, calculate_perplexity, calculate_diversity, calculate_repetition
)
from sentinel.utils.metrics_logger import MetricsLogger
from sentinel.controller.metrics.head_metrics import (
    compute_attention_entropy, compute_gradient_norm, compute_head_importance, collect_head_metrics
)


class MetricCollector:
    """
    A comprehensive system for collecting and analyzing metrics for adaptive transformer models.
    
    This class acts as the central hub for all metrics collection, recording both model-level
    performance metrics and head-level metrics. It provides automated analysis to identify 
    patterns and correlations between metrics.
    """
    
    def __init__(self, 
                 output_dir: str,
                 model_name: str,
                 track_gate_values: bool = True,
                 track_head_metrics: bool = True,
                 track_performance: bool = True,
                 track_pruning_patterns: bool = True,
                 compare_with_static: bool = False,
                 log_level: str = "INFO"):
        """
        Initialize the metric collector.
        
        Args:
            output_dir: Directory to store metrics and analysis results
            model_name: Name of the model being tracked
            track_gate_values: Whether to track attention gate values
            track_head_metrics: Whether to track head-level metrics
            track_performance: Whether to track model performance metrics
            track_pruning_patterns: Whether to track pruning patterns
            compare_with_static: Whether to compare with static pruning
            log_level: Logging level
        """
        self.output_dir = output_dir
        self.model_name = model_name
        self.track_gate_values = track_gate_values
        self.track_head_metrics = track_head_metrics
        self.track_performance = track_performance
        self.track_pruning_patterns = track_pruning_patterns
        self.compare_with_static = compare_with_static
        
        # Set up logging
        self.logger = logging.getLogger("MetricCollector")
        self.logger.setLevel(getattr(logging, log_level))
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metric storages
        self.head_metrics_history = defaultdict(list)
        self.gate_values_history = defaultdict(list)
        self.performance_metrics_history = defaultdict(list)
        self.pruning_patterns_history = defaultdict(list)
        self.static_pruning_metrics = defaultdict(list)
        
        # Create metrics logger
        self.metrics_log_path = os.path.join(output_dir, f"{model_name}_metrics.jsonl")
        self.metrics_logger = MetricsLogger(self.metrics_log_path)
        
        self.logger.info(f"Initialized MetricCollector for {model_name} (output: {output_dir})")
        
        # Record start time and configuration
        self.start_time = time.time()
        self.config = {
            "model_name": model_name,
            "track_gate_values": track_gate_values,
            "track_head_metrics": track_head_metrics,
            "track_performance": track_performance,
            "track_pruning_patterns": track_pruning_patterns,
            "compare_with_static": compare_with_static,
            "start_time": datetime.now().isoformat()
        }
        
        # Save configuration
        with open(os.path.join(output_dir, "metric_collector_config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
    
    def collect_step_metrics(self, 
                            model: torch.nn.Module, 
                            step: int, 
                            phase: str = "train",
                            additional_metrics: Optional[Dict[str, float]] = None,
                            inputs: Optional[Dict[str, torch.Tensor]] = None,
                            labels: Optional[torch.Tensor] = None,
                            logits: Optional[torch.Tensor] = None,
                            dataloader: Optional[Any] = None,
                            loss_fn: Optional[Callable] = None):
        """
        Collect all metrics for a single training/evaluation step.
        
        Args:
            model: The model to collect metrics for
            step: Current step number
            phase: Phase identifier (e.g., "train", "val", "test")
            additional_metrics: Any additional metrics to log
            inputs: Input tensors from current batch
            labels: Label tensors from current batch
            logits: Model output logits
            dataloader: DataLoader for collecting head importance metrics
            loss_fn: Loss function for head importance metrics
        """
        metrics = {}
        
        # Add step, phase and timestamp
        metrics["step"] = step
        metrics["phase"] = phase
        metrics["timestamp"] = datetime.now().isoformat()
        
        # Collect performance metrics if requested
        if self.track_performance and logits is not None and labels is not None:
            perf_metrics = calculate_metrics(logits, labels, prefix=f"{phase}/")
            metrics.update(perf_metrics)
            
            for key, value in perf_metrics.items():
                self.performance_metrics_history[key].append((step, value))
        
        # Collect head metrics if requested
        if self.track_head_metrics:
            head_metrics = {}
            
            try:
                # Check if dataloader and loss_fn are provided for comprehensive metrics
                if dataloader is not None and loss_fn is not None:
                    head_metrics = collect_head_metrics(
                        model, dataloader=dataloader, loss_fn=loss_fn, 
                        num_batches=1, batch=inputs
                    )
                else:
                    # Collect basic metrics without dataloader
                    head_metrics = collect_head_metrics(
                        model, batch=inputs
                    )
                
                # Save raw head metrics for detailed analysis
                head_metrics_path = os.path.join(
                    self.output_dir, f"{phase}_head_metrics_step_{step}.pt"
                )
                torch.save(head_metrics, head_metrics_path)
                
                # Store basic metrics for logging
                metrics[f"{phase}/mean_entropy"] = head_metrics["entropy"].mean().item()
                if "grad_norm" in head_metrics:
                    metrics[f"{phase}/mean_grad_norm"] = head_metrics["grad_norm"].mean().item()
                if "importance" in head_metrics:
                    metrics[f"{phase}/mean_importance"] = head_metrics["importance"].mean().item()
                
                # Store in history
                for metric_name, tensor in head_metrics.items():
                    if isinstance(tensor, torch.Tensor):
                        self.head_metrics_history[metric_name].append((step, tensor.detach().cpu()))
                
            except Exception as e:
                self.logger.warning(f"Error collecting head metrics: {e}")
        
        # Collect gate values if requested
        if self.track_gate_values and hasattr(model, "blocks"):
            try:
                gate_values = {}
                
                # Extract gate values from each layer
                for layer_idx, block in enumerate(model.blocks):
                    if hasattr(block, "attn") and hasattr(block["attn"], "gate"):
                        gate = block["attn"].gate.detach().cpu()
                        gate_values[f"layer_{layer_idx}"] = gate
                        
                        # Compute active heads ratio
                        active_heads = (gate > 0.5).float().mean().item()
                        metrics[f"{phase}/active_heads_layer_{layer_idx}"] = active_heads
                
                # Store gate values for later analysis
                gate_values_path = os.path.join(
                    self.output_dir, f"{phase}_gate_values_step_{step}.pt"
                )
                torch.save(gate_values, gate_values_path)
                
                # Store in history
                for layer_name, gate in gate_values.items():
                    self.gate_values_history[layer_name].append((step, gate))
                
                # Calculate overall active heads ratio
                if gate_values:
                    all_gates = torch.cat([g.flatten() for g in gate_values.values()])
                    active_ratio = (all_gates > 0.5).float().mean().item()
                    metrics[f"{phase}/active_heads_ratio"] = active_ratio
                    
            except Exception as e:
                self.logger.warning(f"Error collecting gate values: {e}")
        
        # Collect pruning patterns if requested
        if self.track_pruning_patterns and hasattr(model, "blocks"):
            try:
                pruning_pattern = {}
                
                # Extract pruning patterns from each layer
                for layer_idx, block in enumerate(model.blocks):
                    if hasattr(block, "attn") and hasattr(block["attn"], "gate"):
                        gate = block["attn"].gate.detach().cpu()
                        # Get binary pruning pattern (1 = active, 0 = pruned)
                        pattern = (gate > 0.5).float()
                        pruning_pattern[f"layer_{layer_idx}"] = pattern
                
                # Calculate pattern metrics
                if pruning_pattern:
                    # Count clusters of pruned heads
                    clusters = 0
                    for layer_idx, pattern in pruning_pattern.items():
                        in_cluster = False
                        for i in range(len(pattern)):
                            if pattern[i] < 0.5:  # Pruned head
                                if not in_cluster:
                                    clusters += 1
                                    in_cluster = True
                            else:
                                in_cluster = False
                    
                    metrics[f"{phase}/pruned_clusters"] = clusters
                    
                    # Store in history
                    self.pruning_patterns_history[step] = pruning_pattern
                    
            except Exception as e:
                self.logger.warning(f"Error analyzing pruning patterns: {e}")
        
        # Add any additional metrics
        if additional_metrics:
            metrics.update(additional_metrics)
        
        # Log metrics
        self.metrics_logger.log(metrics)
        
        return metrics
    
    def analyze_pruning_patterns(self):
        """
        Analyze pruning patterns to identify regularities and patterns.
        
        Returns:
            Dictionary with analysis results
        """
        if not self.pruning_patterns_history:
            return {"error": "No pruning pattern data available for analysis"}
        
        results = {}
        
        # Convert history to sorted list of (step, pattern) tuples
        sorted_history = sorted(self.pruning_patterns_history.items())
        steps = [item[0] for item in sorted_history]
        patterns = [item[1] for item in sorted_history]
        
        # Count number of pattern changes over time
        changes = 0
        for i in range(1, len(patterns)):
            # Compare current pattern to previous
            current = patterns[i]
            previous = patterns[i-1]
            
            # Check if any layer changed its pattern
            for layer in current.keys():
                if layer in previous:
                    if not torch.equal(current[layer], previous[layer]):
                        changes += 1
        
        results["pattern_changes"] = changes
        
        # Analyze final pruning pattern
        if patterns:
            final_pattern = patterns[-1]
            
            # Calculate layer-wise pruning ratios
            layer_pruning = {}
            for layer, pattern in final_pattern.items():
                pruned_ratio = 1.0 - (pattern > 0.5).float().mean().item()
                layer_pruning[layer] = pruned_ratio
            
            results["layer_pruning_ratios"] = layer_pruning
            
            # Calculate overall pruning ratio
            all_gates = torch.cat([p.flatten() for p in final_pattern.values()])
            overall_ratio = 1.0 - (all_gates > 0.5).float().mean().item()
            results["overall_pruning_ratio"] = overall_ratio
            
            # Identify layers with most/least pruning
            if layer_pruning:
                most_pruned_layer = max(layer_pruning.items(), key=lambda x: x[1])
                least_pruned_layer = min(layer_pruning.items(), key=lambda x: x[1])
                
                results["most_pruned_layer"] = {
                    "layer": most_pruned_layer[0],
                    "ratio": most_pruned_layer[1]
                }
                results["least_pruned_layer"] = {
                    "layer": least_pruned_layer[0],
                    "ratio": least_pruned_layer[1]
                }
        
        # Save analysis results
        analysis_path = os.path.join(self.output_dir, "pruning_pattern_analysis.json")
        with open(analysis_path, "w") as f:
            # Convert any torch tensors or numpy arrays to lists
            clean_results = {}
            for k, v in results.items():
                if isinstance(v, dict):
                    clean_results[k] = {
                        sub_k: sub_v.tolist() if isinstance(sub_v, (torch.Tensor, np.ndarray)) 
                        else sub_v for sub_k, sub_v in v.items()
                    }
                elif isinstance(v, (torch.Tensor, np.ndarray)):
                    clean_results[k] = v.tolist()
                else:
                    clean_results[k] = v
            
            json.dump(clean_results, f, indent=2)
        
        return results
    
    def analyze_gate_performance_correlation(self):
        """
        Analyze correlation between gate values and model performance.
        
        Returns:
            Dictionary with correlation analysis results
        """
        if not self.gate_values_history or not self.performance_metrics_history:
            return {"error": "Insufficient data for correlation analysis"}
        
        results = {}
        
        # Extract performance metrics (e.g., loss, perplexity)
        performance_keys = [k for k in self.performance_metrics_history.keys() 
                           if any(m in k for m in ["loss", "perplexity", "accuracy"])]
        
        if not performance_keys:
            return {"error": "No relevant performance metrics found"}
        
        # For each performance metric, calculate correlation with gate values
        for perf_key in performance_keys:
            perf_data = self.performance_metrics_history[perf_key]
            perf_steps = [item[0] for item in perf_data]
            perf_values = [item[1] for item in perf_data]
            
            # Create a mapping of step to performance value
            step_to_perf = dict(zip(perf_steps, perf_values))
            
            # Calculate correlations for each layer's gate values
            correlations = {}
            
            for layer_name, gate_data in self.gate_values_history.items():
                # Align steps with performance data
                aligned_data = []
                
                for step, gate in gate_data:
                    if step in step_to_perf:
                        perf_value = step_to_perf[step]
                        
                        # For each head in this layer, record (gate_value, performance)
                        for head_idx in range(len(gate)):
                            aligned_data.append((gate[head_idx].item(), perf_value))
                
                if aligned_data:
                    # Convert to arrays for correlation calculation
                    gate_values = np.array([item[0] for item in aligned_data])
                    perf_values = np.array([item[1] for item in aligned_data])
                    
                    # Calculate correlation coefficient
                    correlation = np.corrcoef(gate_values, perf_values)[0, 1]
                    correlations[layer_name] = correlation
            
            results[perf_key] = correlations
        
        # Identify strongest correlations
        strongest_pos_correlation = (None, -1.0)
        strongest_neg_correlation = (None, 1.0)
        
        for metric, layer_correlations in results.items():
            for layer, corr in layer_correlations.items():
                if corr > strongest_pos_correlation[1]:
                    strongest_pos_correlation = ((metric, layer), corr)
                if corr < strongest_neg_correlation[1]:
                    strongest_neg_correlation = ((metric, layer), corr)
        
        if strongest_pos_correlation[0]:
            results["strongest_positive_correlation"] = {
                "metric": strongest_pos_correlation[0][0],
                "layer": strongest_pos_correlation[0][1],
                "correlation": strongest_pos_correlation[1]
            }
        
        if strongest_neg_correlation[0]:
            results["strongest_negative_correlation"] = {
                "metric": strongest_neg_correlation[0][0],
                "layer": strongest_neg_correlation[0][1],
                "correlation": strongest_neg_correlation[1]
            }
        
        # Save correlation analysis
        analysis_path = os.path.join(self.output_dir, "gate_performance_correlation.json")
        with open(analysis_path, "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def compare_with_static_pruning(self, static_pruning_metrics):
        """
        Compare adaptive pruning with static pruning strategies.
        
        Args:
            static_pruning_metrics: Dictionary of metrics from static pruning runs
            
        Returns:
            Dictionary with comparison results
        """
        if not self.performance_metrics_history:
            return {"error": "No adaptive pruning data available for comparison"}
        
        if not static_pruning_metrics:
            return {"error": "No static pruning data provided for comparison"}
        
        results = {}
        
        # Extract final performance metrics for adaptive pruning
        adaptive_final_metrics = {}
        for key, values in self.performance_metrics_history.items():
            if values:
                # Get the latest value
                adaptive_final_metrics[key] = values[-1][1]
        
        # Compare with each static pruning strategy
        comparisons = {}
        
        for strategy, metrics in static_pruning_metrics.items():
            strategy_comparison = {}
            
            # Compare common metrics
            common_metrics = set(adaptive_final_metrics.keys()) & set(metrics.keys())
            
            for metric in common_metrics:
                adaptive_value = adaptive_final_metrics[metric]
                static_value = metrics[metric]
                
                # Calculate improvement (note: for loss/perplexity, lower is better)
                if "loss" in metric or "perplexity" in metric:
                    improvement = (static_value - adaptive_value) / static_value
                else:
                    improvement = (adaptive_value - static_value) / static_value
                
                strategy_comparison[metric] = {
                    "adaptive": adaptive_value,
                    "static": static_value,
                    "improvement": improvement
                }
            
            comparisons[strategy] = strategy_comparison
        
        results["comparisons"] = comparisons
        
        # Calculate overall winner
        if comparisons:
            # Count wins for each approach
            wins = {"adaptive": 0, "static": 0, "tie": 0}
            
            for strategy, metrics in comparisons.items():
                for metric, values in metrics.items():
                    if "loss" in metric or "perplexity" in metric:
                        # Lower is better
                        if values["adaptive"] < values["static"]:
                            wins["adaptive"] += 1
                        elif values["adaptive"] > values["static"]:
                            wins["static"] += 1
                        else:
                            wins["tie"] += 1
                    else:
                        # Higher is better
                        if values["adaptive"] > values["static"]:
                            wins["adaptive"] += 1
                        elif values["adaptive"] < values["static"]:
                            wins["static"] += 1
                        else:
                            wins["tie"] += 1
            
            results["wins"] = wins
            
            # Determine overall winner
            if wins["adaptive"] > wins["static"]:
                results["overall_winner"] = "adaptive"
            elif wins["adaptive"] < wins["static"]:
                results["overall_winner"] = "static"
            else:
                results["overall_winner"] = "tie"
        
        # Save comparison results
        comparison_path = os.path.join(self.output_dir, "pruning_strategy_comparison.json")
        with open(comparison_path, "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def generate_report(self):
        """
        Generate a comprehensive report with all analysis results.
        
        Returns:
            Dictionary with all analysis results
        """
        report = {
            "model_name": self.model_name,
            "collection_time": datetime.now().isoformat(),
            "duration_hours": (time.time() - self.start_time) / 3600
        }
        
        # Analyze pruning patterns
        if self.track_pruning_patterns:
            pruning_analysis = self.analyze_pruning_patterns()
            report["pruning_patterns"] = pruning_analysis
        
        # Analyze correlation between gate values and performance
        if self.track_gate_values and self.track_performance:
            correlation_analysis = self.analyze_gate_performance_correlation()
            report["gate_performance_correlation"] = correlation_analysis
        
        # Compare with static pruning if data is available
        if self.compare_with_static and self.static_pruning_metrics:
            comparison = self.compare_with_static_pruning(self.static_pruning_metrics)
            report["static_comparison"] = comparison
        
        # Extract performance trends
        if self.track_performance:
            performance_trends = {}
            
            for key, values in self.performance_metrics_history.items():
                if len(values) > 1:
                    steps = [item[0] for item in values]
                    metric_values = [item[1] for item in values]
                    
                    # Calculate trend (positive slope = increasing, negative = decreasing)
                    if len(steps) > 1:
                        # Simple linear regression to get slope
                        steps_arr = np.array(steps)
                        values_arr = np.array(metric_values)
                        
                        # Normalize steps to [0, 1] range for numeric stability
                        steps_norm = (steps_arr - steps_arr.min()) / (steps_arr.max() - steps_arr.min())
                        
                        # Calculate slope using least squares
                        slope = np.polyfit(steps_norm, values_arr, 1)[0]
                        
                        performance_trends[key] = {
                            "initial": metric_values[0],
                            "final": metric_values[-1],
                            "change": metric_values[-1] - metric_values[0],
                            "slope": slope,
                            "trend": "increasing" if slope > 0 else "decreasing"
                        }
            
            report["performance_trends"] = performance_trends
        
        # Save comprehensive report
        report_path = os.path.join(self.output_dir, "comprehensive_analysis_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Generated comprehensive analysis report: {report_path}")
        
        return report
    
    def visualize_metrics(self, save_dir=None):
        """
        Generate visualizations for collected metrics.
        
        Args:
            save_dir: Directory to save visualizations (defaults to output_dir)
        """
        if save_dir is None:
            save_dir = self.output_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Visualize performance metrics
        if self.track_performance and self.performance_metrics_history:
            self._visualize_performance_metrics(save_dir)
        
        # Visualize gate values over time
        if self.track_gate_values and self.gate_values_history:
            self._visualize_gate_values(save_dir)
        
        # Visualize head metrics
        if self.track_head_metrics and self.head_metrics_history:
            self._visualize_head_metrics(save_dir)
        
        # Visualize pruning patterns
        if self.track_pruning_patterns and self.pruning_patterns_history:
            self._visualize_pruning_patterns(save_dir)
        
        # Visualize correlation between gates and performance
        if self.track_gate_values and self.track_performance:
            self._visualize_gate_performance_correlation(save_dir)
        
        self.logger.info(f"Generated visualizations in {save_dir}")
    
    def _visualize_performance_metrics(self, save_dir):
        """Helper method to visualize performance metrics."""
        # Group related metrics
        metric_groups = defaultdict(list)
        
        for key in self.performance_metrics_history.keys():
            if "loss" in key:
                metric_groups["loss"].append(key)
            elif "perplexity" in key:
                metric_groups["perplexity"].append(key)
            elif "accuracy" in key:
                metric_groups["accuracy"].append(key)
            else:
                metric_groups["other"].append(key)
        
        # Create a plot for each metric group
        for group_name, metrics in metric_groups.items():
            if not metrics:
                continue
                
            plt.figure(figsize=(10, 6))
            
            for metric in metrics:
                data = self.performance_metrics_history[metric]
                steps = [item[0] for item in data]
                values = [item[1] for item in data]
                
                plt.plot(steps, values, label=metric)
            
            plt.title(f"{group_name.capitalize()} Metrics")
            plt.xlabel("Training Step")
            plt.ylabel(group_name.capitalize())
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            save_path = os.path.join(save_dir, f"{group_name}_metrics.png")
            plt.savefig(save_path)
            plt.close()
    
    def _visualize_gate_values(self, save_dir):
        """Helper method to visualize gate values over time."""
        # Visualize gate values for each layer
        for layer_name, gate_data in self.gate_values_history.items():
            if not gate_data:
                continue
                
            steps = [item[0] for item in gate_data]
            gates = [item[1] for item in gate_data]
            
            # Create a heatmap-style visualization
            num_heads = gates[0].shape[0]
            gate_matrix = np.zeros((len(steps), num_heads))
            
            for i, gate in enumerate(gates):
                gate_matrix[i] = gate.numpy()
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(gate_matrix, cmap="viridis", vmin=0, vmax=1)
            plt.title(f"Gate Values Over Time - {layer_name}")
            plt.xlabel("Head Index")
            plt.ylabel("Training Step")
            plt.yticks(np.arange(len(steps))[::max(1, len(steps)//10)], 
                      [steps[i] for i in range(0, len(steps), max(1, len(steps)//10))])
            
            save_path = os.path.join(save_dir, f"gate_values_{layer_name}.png")
            plt.savefig(save_path)
            plt.close()
            
        # Create an overall visualization of gate activity
        if self.gate_values_history:
            plt.figure(figsize=(12, 8))
            
            # Calculate active head ratio per layer over time
            active_ratios = defaultdict(list)
            
            for layer_name, gate_data in self.gate_values_history.items():
                steps = [item[0] for item in gate_data]
                gates = [item[1] for item in gate_data]
                
                for step, gate in zip(steps, gates):
                    active_ratio = (gate > 0.5).float().mean().item()
                    active_ratios[layer_name].append((step, active_ratio))
            
            # Plot active head ratio for each layer
            for layer_name, ratio_data in active_ratios.items():
                steps = [item[0] for item in ratio_data]
                ratios = [item[1] for item in ratio_data]
                
                plt.plot(steps, ratios, label=layer_name)
            
            plt.title("Active Head Ratio Over Time")
            plt.xlabel("Training Step")
            plt.ylabel("Active Head Ratio")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            save_path = os.path.join(save_dir, "active_head_ratio.png")
            plt.savefig(save_path)
            plt.close()
    
    def _visualize_head_metrics(self, save_dir):
        """Helper method to visualize head metrics."""
        # Visualize average head metrics over time
        for metric_name, metric_data in self.head_metrics_history.items():
            if not metric_data:
                continue
                
            steps = [item[0] for item in metric_data]
            metrics = [item[1] for item in metric_data]
            
            # Calculate average per layer over time
            avg_per_layer = defaultdict(list)
            
            for step, metric in zip(steps, metrics):
                for layer_idx in range(metric.shape[0]):
                    layer_avg = metric[layer_idx].mean().item()
                    avg_per_layer[f"layer_{layer_idx}"].append((step, layer_avg))
            
            plt.figure(figsize=(12, 8))
            
            for layer_name, layer_data in avg_per_layer.items():
                layer_steps = [item[0] for item in layer_data]
                layer_values = [item[1] for item in layer_data]
                
                plt.plot(layer_steps, layer_values, label=layer_name)
            
            plt.title(f"Average {metric_name.capitalize()} Per Layer")
            plt.xlabel("Training Step")
            plt.ylabel(f"Average {metric_name.capitalize()}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            save_path = os.path.join(save_dir, f"avg_{metric_name}_per_layer.png")
            plt.savefig(save_path)
            plt.close()
            
            # Create a heatmap of the final metric state
            if metrics:
                final_metric = metrics[-1]
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(final_metric.numpy(), cmap="viridis", annot=True, fmt=".2f")
                plt.title(f"Final {metric_name.capitalize()} Heatmap")
                plt.xlabel("Head Index")
                plt.ylabel("Layer Index")
                
                save_path = os.path.join(save_dir, f"final_{metric_name}_heatmap.png")
                plt.savefig(save_path)
                plt.close()
    
    def _visualize_pruning_patterns(self, save_dir):
        """Helper method to visualize pruning patterns."""
        # Visualize pruning pattern evolution
        if not self.pruning_patterns_history:
            return
            
        # Get steps and patterns
        sorted_history = sorted(self.pruning_patterns_history.items())
        steps = [item[0] for item in sorted_history]
        patterns = [item[1] for item in sorted_history]
        
        # Select a subset of steps for visualization
        if len(steps) > 10:
            # Choose steps evenly spaced
            indices = np.linspace(0, len(steps)-1, 10).astype(int)
            vis_steps = [steps[i] for i in indices]
            vis_patterns = [patterns[i] for i in indices]
        else:
            vis_steps = steps
            vis_patterns = patterns
        
        # Create a grid of pruning pattern heatmaps
        num_layers = len(vis_patterns[0])
        
        fig, axes = plt.subplots(len(vis_steps), 1, figsize=(10, 2*len(vis_steps)))
        if len(vis_steps) == 1:
            axes = [axes]
        
        for i, (step, pattern) in enumerate(zip(vis_steps, vis_patterns)):
            # Create a binary matrix for all layers
            pattern_matrix = np.zeros((num_layers, max(len(p) for p in pattern.values())))
            
            for j, (layer_name, layer_pattern) in enumerate(pattern.items()):
                pattern_array = layer_pattern.numpy()
                pattern_matrix[j, :len(pattern_array)] = pattern_array
            
            # Plot heatmap
            sns.heatmap(pattern_matrix, cmap="Blues", vmin=0, vmax=1, 
                       ax=axes[i], cbar=False, annot=True, fmt=".0f")
            axes[i].set_title(f"Step {step}")
            axes[i].set_ylabel("Layer")
            
            if i == len(vis_steps) - 1:
                axes[i].set_xlabel("Head Index")
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, "pruning_pattern_evolution.png")
        plt.savefig(save_path)
        plt.close()
        
        # Visualize pruning ratio per layer over time
        pruning_ratios = defaultdict(list)
        
        for step, pattern in sorted_history:
            for layer_name, layer_pattern in pattern.items():
                pruned_ratio = 1.0 - (layer_pattern > 0.5).float().mean().item()
                pruning_ratios[layer_name].append((step, pruned_ratio))
        
        plt.figure(figsize=(12, 8))
        
        for layer_name, ratio_data in pruning_ratios.items():
            layer_steps = [item[0] for item in ratio_data]
            layer_ratios = [item[1] for item in ratio_data]
            
            plt.plot(layer_steps, layer_ratios, label=layer_name)
        
        plt.title("Pruning Ratio Per Layer Over Time")
        plt.xlabel("Training Step")
        plt.ylabel("Pruning Ratio")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(save_dir, "pruning_ratio_per_layer.png")
        plt.savefig(save_path)
        plt.close()
    
    def _visualize_gate_performance_correlation(self, save_dir):
        """Helper method to visualize correlation between gates and performance."""
        if not self.gate_values_history or not self.performance_metrics_history:
            return
            
        # Extract performance metrics
        perf_keys = [k for k in self.performance_metrics_history.keys() 
                   if any(m in k for m in ["loss", "perplexity", "accuracy"])]
        
        if not perf_keys:
            return
            
        # For each performance metric, visualize correlation
        for perf_key in perf_keys[:3]:  # Limit to first 3 metrics for brevity
            perf_data = self.performance_metrics_history[perf_key]
            perf_steps = [item[0] for item in perf_data]
            perf_values = [item[1] for item in perf_data]
            
            # Create a mapping of step to performance value
            step_to_perf = dict(zip(perf_steps, perf_values))
            
            # Collect data for scatter plots
            scatterplot_data = defaultdict(list)
            
            for layer_name, gate_data in self.gate_values_history.items():
                for step, gate in gate_data:
                    if step in step_to_perf:
                        perf_value = step_to_perf[step]
                        
                        # Store average gate value against performance
                        avg_gate = gate.mean().item()
                        scatterplot_data[layer_name].append((avg_gate, perf_value))
            
            # Create scatter plots
            if scatterplot_data:
                plt.figure(figsize=(12, 8))
                
                for layer_name, data in scatterplot_data.items():
                    gate_values = [item[0] for item in data]
                    perf_values = [item[1] for item in data]
                    
                    plt.scatter(gate_values, perf_values, label=layer_name, alpha=0.7)
                    
                    # Add regression line if enough points
                    if len(gate_values) > 2:
                        z = np.polyfit(gate_values, perf_values, 1)
                        p = np.poly1d(z)
                        x_range = np.linspace(min(gate_values), max(gate_values), 100)
                        plt.plot(x_range, p(x_range), linestyle='--')
                
                plt.title(f"Correlation: Average Gate Value vs {perf_key}")
                plt.xlabel("Average Gate Value")
                plt.ylabel(perf_key)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                save_path = os.path.join(save_dir, f"gate_vs_{perf_key.replace('/', '_')}.png")
                plt.savefig(save_path)
                plt.close()
    
    def register_static_pruning_metrics(self, strategy_name, metrics):
        """
        Register metrics from a static pruning strategy for comparison.
        
        Args:
            strategy_name: Name of the static pruning strategy
            metrics: Dictionary of metrics from the static pruning run
        """
        self.static_pruning_metrics[strategy_name] = metrics
        self.logger.info(f"Registered metrics for static pruning strategy: {strategy_name}")
    
    def save_metrics_csv(self, save_path=None):
        """
        Save all tracked metrics to a CSV file for easy analysis in other tools.
        
        Args:
            save_path: Path to save CSV file (defaults to output_dir/metrics.csv)
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "metrics.csv")
        
        # Get all metrics from logger
        all_metrics = self.metrics_logger.get_metrics()
        
        if not all_metrics:
            self.logger.warning("No metrics available to save to CSV")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(all_metrics)
        
        # Ensure step and timestamp columns are first
        columns = df.columns.tolist()
        for col in ["timestamp", "step"]:
            if col in columns:
                columns.remove(col)
                columns = [col] + columns
        
        df = df[columns]
        
        # Save to CSV
        df.to_csv(save_path, index=False)
        self.logger.info(f"Saved metrics to CSV: {save_path}")
    
    def __del__(self):
        """Ensure metrics are flushed when the collector is destroyed."""
        try:
            if hasattr(self, 'metrics_logger'):
                self.metrics_logger.flush()
        except:
            # Ignore errors during cleanup
            pass


# Analysis functions that can be used without instantiating the collector
def analyze_pruning_strategy(pruning_results_dir, output_dir=None):
    """
    Analyze results from a pruning experiment directory.
    
    Args:
        pruning_results_dir: Directory containing pruning experiment results
        output_dir: Directory to save analysis results (defaults to pruning_results_dir/analysis)
        
    Returns:
        Dictionary with analysis results
    """
    if output_dir is None:
        output_dir = os.path.join(pruning_results_dir, "analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Look for metrics files
    metrics_files = []
    for root, dirs, files in os.walk(pruning_results_dir):
        for file in files:
            if file.endswith(".jsonl") and "metrics" in file:
                metrics_files.append(os.path.join(root, file))
    
    if not metrics_files:
        results["error"] = "No metrics files found in the provided directory"
        return results
    
    # Process each metrics file
    metrics_data = []
    
    for metrics_file in metrics_files:
        try:
            with open(metrics_file, "r") as f:
                file_data = [json.loads(line) for line in f if line.strip()]
                metrics_data.extend(file_data)
        except Exception as e:
            print(f"Error reading {metrics_file}: {e}")
    
    if not metrics_data:
        results["error"] = "No valid metrics data found in the provided files"
        return results
    
    # Extract common metrics
    metric_keys = set()
    for entry in metrics_data:
        metric_keys.update(entry.keys())
    
    # Remove metadata fields
    metadata_keys = {"step", "phase", "timestamp", "description"}
    metric_keys = metric_keys - metadata_keys
    
    # Group metrics by phase
    phase_metrics = defaultdict(list)
    for entry in metrics_data:
        phase = entry.get("phase", "unknown")
        phase_metrics[phase].append(entry)
    
    # Calculate summary statistics for each phase
    summary = {}
    
    for phase, entries in phase_metrics.items():
        phase_summary = {}
        
        for key in metric_keys:
            values = [entry.get(key) for entry in entries if key in entry]
            values = [v for v in values if v is not None and not np.isnan(v)]
            
            if values:
                phase_summary[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "final": values[-1] if len(values) > 0 else None
                }
        
        summary[phase] = phase_summary
    
    results["summary"] = summary
    
    # Calculate improvement from first to last step
    improvement = {}
    
    for phase, entries in phase_metrics.items():
        if len(entries) < 2:
            continue
            
        sorted_entries = sorted(entries, key=lambda x: x.get("step", 0))
        first_entry = sorted_entries[0]
        last_entry = sorted_entries[-1]
        
        phase_improvement = {}
        
        for key in metric_keys:
            if key in first_entry and key in last_entry:
                first_value = first_entry[key]
                last_value = last_entry[key]
                
                if first_value is not None and last_value is not None:
                    # Calculate relative improvement
                    if "loss" in key or "perplexity" in key:
                        # Lower is better
                        change = (first_value - last_value) / first_value if first_value != 0 else 0
                    else:
                        # Higher is better
                        change = (last_value - first_value) / first_value if first_value != 0 else 0
                    
                    phase_improvement[key] = {
                        "first": first_value,
                        "last": last_value,
                        "absolute_change": last_value - first_value,
                        "relative_change": change
                    }
        
        improvement[phase] = phase_improvement
    
    results["improvement"] = improvement
    
    # Extract pruning-specific metrics
    pruning_metrics = {}
    
    # Look for metrics related to pruning (active heads ratio, pruned clusters, etc.)
    pruning_keys = [k for k in metric_keys if any(p in k.lower() for p in 
                                               ["active", "head", "prune", "cluster"])]
    
    if pruning_keys:
        for phase, entries in phase_metrics.items():
            phase_pruning = {}
            
            for key in pruning_keys:
                values = [entry.get(key) for entry in entries if key in entry]
                values = [v for v in values if v is not None and not np.isnan(v)]
                
                if values:
                    # Track evolution over steps
                    steps = [entry.get("step", i) for i, entry in enumerate(entries) if key in entry]
                    phase_pruning[key] = {
                        "steps": steps,
                        "values": values,
                        "final": values[-1] if values else None
                    }
            
            pruning_metrics[phase] = phase_pruning
    
    results["pruning_metrics"] = pruning_metrics
    
    # Generate visualizations
    try:
        # Performance metrics over time
        perf_keys = [k for k in metric_keys if any(m in k.lower() for m in 
                                               ["loss", "perplexity", "accuracy"])]
        
        if perf_keys:
            plt.figure(figsize=(12, 8))
            
            for phase, entries in phase_metrics.items():
                sorted_entries = sorted(entries, key=lambda x: x.get("step", 0))
                steps = [entry.get("step", i) for i, entry in enumerate(sorted_entries)]
                
                for key in perf_keys[:3]:  # Limit to first 3 for clarity
                    values = [entry.get(key) for entry in sorted_entries]
                    # Filter out None and NaN
                    valid_indices = [i for i, v in enumerate(values) if v is not None and not np.isnan(v)]
                    valid_steps = [steps[i] for i in valid_indices]
                    valid_values = [values[i] for i in valid_indices]
                    
                    if valid_values:
                        plt.plot(valid_steps, valid_values, label=f"{phase}-{key}")
            
            plt.title("Performance Metrics Over Time")
            plt.xlabel("Step")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            save_path = os.path.join(output_dir, "performance_metrics.png")
            plt.savefig(save_path)
            plt.close()
        
        # Pruning metrics over time
        if pruning_keys:
            plt.figure(figsize=(12, 8))
            
            for phase, phase_data in pruning_metrics.items():
                for key, data in phase_data.items():
                    if "values" in data and "steps" in data:
                        plt.plot(data["steps"], data["values"], label=f"{phase}-{key}")
            
            plt.title("Pruning Metrics Over Time")
            plt.xlabel("Step")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            save_path = os.path.join(output_dir, "pruning_metrics.png")
            plt.savefig(save_path)
            plt.close()
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    # Save analysis results
    results_path = os.path.join(output_dir, "pruning_analysis.json")
    with open(results_path, "w") as f:
        # Convert numpy values to native Python types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            else:
                return obj
        
        json.dump(convert_numpy(results), f, indent=2)
    
    print(f"Analysis results saved to {results_path}")
    
    return results


def compare_pruning_strategies(strategy_dirs, output_dir=None):
    """
    Compare results from multiple pruning strategies.
    
    Args:
        strategy_dirs: Dictionary mapping strategy names to result directories
        output_dir: Directory to save comparison results
        
    Returns:
        Dictionary with comparison results
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(next(iter(strategy_dirs.values()))), "comparison")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Analyze each strategy
    strategy_results = {}
    
    for strategy, directory in strategy_dirs.items():
        analysis = analyze_pruning_strategy(directory)
        strategy_results[strategy] = analysis
    
    results["individual_analyses"] = strategy_results
    
    # Extract common metrics for comparison
    common_metrics = defaultdict(set)
    
    for strategy, analysis in strategy_results.items():
        if "summary" in analysis:
            for phase, metrics in analysis["summary"].items():
                for metric in metrics.keys():
                    common_metrics[phase].add(metric)
    
    # Perform direct comparison
    comparison = {}
    
    for phase, metrics in common_metrics.items():
        phase_comparison = {}
        
        for metric in metrics:
            metric_comparison = {}
            
            for strategy, analysis in strategy_results.items():
                if ("summary" in analysis and phase in analysis["summary"] and 
                    metric in analysis["summary"][phase]):
                    metric_data = analysis["summary"][phase][metric]
                    if "final" in metric_data and metric_data["final"] is not None:
                        metric_comparison[strategy] = metric_data["final"]
            
            if metric_comparison:
                # Determine best strategy
                if "loss" in metric or "perplexity" in metric:
                    # Lower is better
                    best_strategy = min(metric_comparison.items(), key=lambda x: x[1])
                else:
                    # Higher is better
                    best_strategy = max(metric_comparison.items(), key=lambda x: x[1])
                
                metric_comparison["best_strategy"] = best_strategy[0]
                metric_comparison["best_value"] = best_strategy[1]
                
                phase_comparison[metric] = metric_comparison
        
        comparison[phase] = phase_comparison
    
    results["metric_comparison"] = comparison
    
    # Count wins by strategy
    wins = defaultdict(int)
    
    for phase, metrics in comparison.items():
        for metric, data in metrics.items():
            if "best_strategy" in data:
                wins[data["best_strategy"]] += 1
    
    results["win_counts"] = dict(wins)
    
    # Determine overall best strategy
    if wins:
        best_strategy = max(wins.items(), key=lambda x: x[1])
        results["overall_best_strategy"] = {
            "strategy": best_strategy[0],
            "wins": best_strategy[1],
            "total_metrics": sum(wins.values())
        }
    
    # Generate visualizations
    try:
        # Compare final values for key metrics
        key_metrics = ["loss", "perplexity", "accuracy"]
        selected_metrics = []
        
        for metric_name in key_metrics:
            for phase, metrics in comparison.items():
                matching_metrics = [m for m in metrics.keys() if metric_name in m.lower()]
                selected_metrics.extend([(phase, m) for m in matching_metrics])
        
        # Limit to first 5 metrics for clarity
        selected_metrics = selected_metrics[:5]
        
        if selected_metrics:
            # Create a separate plot for each metric
            for phase, metric in selected_metrics:
                if phase in comparison and metric in comparison[phase]:
                    metric_data = comparison[phase][metric]
                    
                    # Remove metadata entries
                    plot_data = {k: v for k, v in metric_data.items() 
                               if k not in ["best_strategy", "best_value"]}
                    
                    if not plot_data:
                        continue
                    
                    plt.figure(figsize=(10, 6))
                    
                    strategies = list(plot_data.keys())
                    values = list(plot_data.values())
                    
                    # Sort by value
                    if "loss" in metric or "perplexity" in metric:
                        # Lower is better
                        sorted_indices = np.argsort(values)
                    else:
                        # Higher is better
                        sorted_indices = np.argsort(values)[::-1]
                    
                    sorted_strategies = [strategies[i] for i in sorted_indices]
                    sorted_values = [values[i] for i in sorted_indices]
                    
                    bars = plt.bar(sorted_strategies, sorted_values)
                    
                    # Highlight best strategy
                    if "best_strategy" in metric_data:
                        best_idx = sorted_strategies.index(metric_data["best_strategy"])
                        bars[best_idx].set_color("green")
                    
                    plt.title(f"{phase.capitalize()}: {metric}")
                    plt.xlabel("Strategy")
                    plt.ylabel("Value")
                    plt.xticks(rotation=45, ha="right")
                    
                    for i, v in enumerate(sorted_values):
                        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
                    
                    plt.tight_layout()
                    
                    save_path = os.path.join(output_dir, f"{phase}_{metric}_comparison.png")
                    plt.savefig(save_path)
                    plt.close()
        
        # Create win count visualization
        if wins:
            plt.figure(figsize=(10, 6))
            
            strategies = list(wins.keys())
            win_counts = list(wins.values())
            
            # Sort by win count
            sorted_indices = np.argsort(win_counts)[::-1]
            sorted_strategies = [strategies[i] for i in sorted_indices]
            sorted_win_counts = [win_counts[i] for i in sorted_indices]
            
            plt.bar(sorted_strategies, sorted_win_counts)
            
            plt.title("Strategy Win Counts")
            plt.xlabel("Strategy")
            plt.ylabel("Number of Wins")
            plt.xticks(rotation=45, ha="right")
            
            for i, v in enumerate(sorted_win_counts):
                plt.text(i, v, str(v), ha="center", va="bottom")
            
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, "strategy_win_counts.png")
            plt.savefig(save_path)
            plt.close()
        
    except Exception as e:
        print(f"Error generating comparison visualizations: {e}")
    
    # Save comparison results
    results_path = os.path.join(output_dir, "strategy_comparison.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Comparison results saved to {results_path}")
    
    return results