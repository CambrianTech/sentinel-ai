"""
Neural Plasticity Dashboard Generator

This module provides utilities for creating interactive dashboards to visualize
neural plasticity training progress, including sample predictions, metrics,
and pruning visualizations.

Version: v0.0.1 (2025-04-20 10:15:00)
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import base64
from io import BytesIO

# Check if we're running in Colab
IS_COLAB = False
try:
    import google.colab
    IS_COLAB = True
    print("🌐 Running dashboard in Google Colab environment")
except (ImportError, ModuleNotFoundError):
    pass

# Detect Apple Silicon environment
IS_APPLE_SILICON = False
try:
    import platform
    if platform.system() == "Darwin" and platform.processor() == "arm":
        IS_APPLE_SILICON = True
        print("🍎 Apple Silicon detected - applying dashboard optimizations")
except Exception:
    pass

# Import visualization utilities
from .visualization import (
    visualize_head_entropy,
    visualize_head_gradients,
    visualize_pruning_decisions,
    visualize_training_metrics,
    create_pruning_state_heatmap,
    visualize_attention_patterns,
    VisualizationReporter
)

# Define HTML template for dashboard
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Plasticity Dashboard</title>
    <style>
        :root {
            --primary-color: #3498db;
            --secondary-color: #2ecc71;
            --accent-color: #e74c3c;
            --bg-color: #f8f9fa;
            --card-bg: #ffffff;
            --text-color: #2c3e50;
            --border-color: #e0e0e0;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--bg-color);
            margin: 0;
            padding: 20px;
        }
        
        .dashboard {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .dashboard-header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background-color: var(--card-bg);
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .dashboard-title {
            font-size: 24px;
            font-weight: bold;
            margin: 0;
            color: var(--primary-color);
        }
        
        .dashboard-subtitle {
            font-size: 16px;
            color: #666;
            margin-top: 5px;
        }
        
        .metrics-summary {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 30px;
            justify-content: center;
        }
        
        .metric-card {
            background-color: var(--card-bg);
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            min-width: 150px;
            flex: 1;
        }
        
        .metric-title {
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: var(--primary-color);
        }
        
        .metric-change {
            font-size: 12px;
            margin-top: 5px;
        }
        
        .positive-change {
            color: var(--secondary-color);
        }
        
        .negative-change {
            color: var(--accent-color);
        }
        
        .section {
            background-color: var(--card-bg);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .section-title {
            font-size: 18px;
            font-weight: bold;
            margin-top: 0;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border-color);
            color: var(--primary-color);
        }
        
        .viz-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }
        
        .viz-item {
            flex: 1;
            min-width: 300px;
            margin-bottom: 20px;
        }
        
        .viz-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 10px;
            text-align: center;
        }
        
        .viz-image {
            width: 100%;
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            border: 1px solid var(--border-color);
        }
        
        .sample-container {
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f5f5f5;
            border-radius: 5px;
            border-left: 4px solid var(--primary-color);
        }
        
        .sample-step {
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .sample-text {
            font-family: monospace;
            white-space: pre-wrap;
            margin-bottom: 10px;
            padding: 10px;
            background-color: #fff;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
        }
        
        .token-container {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-bottom: 5px;
        }
        
        .token {
            background-color: #e8f4fc;
            border-radius: 4px;
            padding: 3px 6px;
            font-family: monospace;
            font-size: 14px;
            display: inline-flex;
            align-items: center;
        }
        
        .token-value {
            font-weight: bold;
        }
        
        .token-prob {
            font-size: 12px;
            color: #666;
            margin-left: 5px;
        }
        
        .token-correct {
            background-color: #d4edda;
        }
        
        .token-incorrect {
            background-color: #f8d7da;
        }
        
        .perplexity-value {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        
        .pruning-info {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .pruning-stat {
            background-color: #f5f5f5;
            border-radius: 5px;
            padding: 10px 15px;
            font-size: 14px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        th {
            background-color: #f5f5f5;
            font-weight: bold;
        }
        
        tr:hover {
            background-color: #f9f9f9;
        }
        
        .tab-container {
            margin-bottom: 20px;
        }
        
        .tab-buttons {
            display: flex;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 15px;
        }
        
        .tab-button {
            padding: 10px 15px;
            cursor: pointer;
            background: none;
            border: none;
            font-size: 16px;
            outline: none;
            position: relative;
            transition: all 0.3s ease;
        }
        
        .tab-button.active {
            color: var(--primary-color);
            font-weight: bold;
        }
        
        .tab-button.active::after {
            content: '';
            position: absolute;
            bottom: -1px;
            left: 0;
            width: 100%;
            height: 3px;
            background-color: var(--primary-color);
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        @media (max-width: 768px) {
            .metrics-summary {
                flex-direction: column;
            }
            
            .viz-item {
                min-width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="dashboard-header">
            <h1 class="dashboard-title">Neural Plasticity Dashboard</h1>
            <p class="dashboard-subtitle">{timestamp}</p>
        </div>
        
        <div class="metrics-summary">
            {metrics_summary}
        </div>
        
        <div class="tab-container">
            <div class="tab-buttons">
                <button class="tab-button active" onclick="showTab('training-tab')">Training Metrics</button>
                <button class="tab-button" onclick="showTab('pruning-tab')">Pruning Visualization</button>
                <button class="tab-button" onclick="showTab('samples-tab')">Sample Predictions</button>
                <button class="tab-button" onclick="showTab('attention-tab')">Attention Patterns</button>
            </div>
            
            <div id="training-tab" class="tab-content active">
                <div class="section">
                    <h2 class="section-title">Training Progress</h2>
                    <div class="viz-container">
                        {training_metrics_viz}
                    </div>
                </div>
            </div>
            
            <div id="pruning-tab" class="tab-content">
                <div class="section">
                    <h2 class="section-title">Pruning Analysis</h2>
                    <div class="pruning-info">
                        {pruning_info}
                    </div>
                    <div class="viz-container">
                        {pruning_viz}
                    </div>
                </div>
            </div>
            
            <div id="samples-tab" class="tab-content">
                <div class="section">
                    <h2 class="section-title">Sample Predictions</h2>
                    {sample_predictions}
                </div>
            </div>
            
            <div id="attention-tab" class="tab-content">
                <div class="section">
                    <h2 class="section-title">Attention Patterns</h2>
                    <div class="viz-container">
                        {attention_viz}
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function showTab(tabId) {
            // Hide all tab contents
            const tabContents = document.getElementsByClassName('tab-content');
            for (let i = 0; i < tabContents.length; i++) {
                tabContents[i].classList.remove('active');
            }
            
            // Remove active class from all tab buttons
            const tabButtons = document.getElementsByClassName('tab-button');
            for (let i = 0; i < tabButtons.length; i++) {
                tabButtons[i].classList.remove('active');
            }
            
            // Show the selected tab content
            document.getElementById(tabId).classList.add('active');
            
            // Add active class to the clicked button
            const activeButton = document.querySelector(`.tab-button[onclick="showTab('${tabId}')"]`);
            activeButton.classList.add('active');
        }
    </script>
</body>
</html>
"""

def fig_to_base64(fig):
    """Convert a matplotlib figure to base64 encoded PNG data."""
    # Adjust for Colab environment
    if IS_COLAB:
        # Use a lower DPI in Colab for better performance
        dpi = 80
        # Force Agg backend for reliability in Colab
        import matplotlib
        matplotlib.use('Agg')
    elif IS_APPLE_SILICON:
        # Lower DPI for Apple Silicon to avoid issues
        dpi = 80
        # Use Agg backend for stability
        import matplotlib
        matplotlib.use('Agg')
    else:
        # Standard DPI for desktop environments
        dpi = 100
    
    # Create buffer and save figure
    buf = BytesIO()
    try:
        # Handle potential rendering issues in Colab
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    except Exception as e:
        print(f"Warning: Error saving figure: {e}. Using fallback method.")
        # Fallback to simpler rendering approach
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    
    buf.seek(0)
    img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_data

def create_metric_card(title, value, previous=None, format_str="{:.2f}", is_lower_better=True):
    """Create an HTML metric card with optional change indicator."""
    # Format the main value
    formatted_value = format_str.format(value)
    
    # Calculate and format change if previous value provided
    change_html = ""
    if previous is not None:
        change = value - previous
        pct_change = (change / previous) * 100 if previous != 0 else 0
        
        # Determine if change is positive (improvement) or negative
        is_improvement = (change < 0 if is_lower_better else change > 0)
        change_class = "positive-change" if is_improvement else "negative-change"
        change_sign = "↓" if change < 0 else "↑"
        
        change_html = f"""
        <div class="metric-change {change_class}">
            {change_sign} {abs(pct_change):.1f}%
        </div>
        """
    
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{formatted_value}</div>
        {change_html}
    </div>
    """

def create_sample_html(sample_data, step, max_tokens=15):
    """Create HTML for a sample prediction display."""
    sample_text = sample_data.get("input_text", "")
    predicted_tokens = sample_data.get("predicted_tokens", [])
    predicted_probs = sample_data.get("predicted_probs", [])
    actual_tokens = sample_data.get("actual_tokens", [])
    actual_probs = sample_data.get("actual_probs", [])
    perplexities = sample_data.get("perplexities", [])
    
    # Show more of the input text for better context
    input_display = sample_text
    if len(input_display) > 500:
        input_display = input_display[:497] + "..."
    
    # Format the predicted tokens with probabilities
    predicted_html = '<div class="token-container">'
    for i, (token, prob) in enumerate(zip(predicted_tokens[:max_tokens], predicted_probs[:max_tokens])):
        is_correct = i < len(actual_tokens) and token == actual_tokens[i]
        token_class = "token-correct" if is_correct else "token-incorrect"
        predicted_html += f"""
        <div class="token {token_class}">
            <span class="token-value">{token}</span>
            <span class="token-prob">{prob:.2f}</span>
        </div>
        """
    predicted_html += '</div>'
    
    # Format the actual tokens with probabilities
    actual_html = '<div class="token-container">'
    for i, (token, prob) in enumerate(zip(actual_tokens[:max_tokens], actual_probs[:max_tokens])):
        actual_html += f"""
        <div class="token">
            <span class="token-value">{token}</span>
            <span class="token-prob">{prob:.2f}</span>
        </div>
        """
    actual_html += '</div>'
    
    # Format perplexity if available
    perplexity_html = ""
    if perplexities and len(perplexities) > 0:
        avg_perplexity = sum(perplexities) / len(perplexities)
        perplexity_html = f'<div class="perplexity-value">Avg. Perplexity: {avg_perplexity:.2f}</div>'
    
    return f"""
    <div class="sample-container">
        <div class="sample-step">Step {step}</div>
        <div class="sample-text">{input_display}</div>
        <div class="predictions">
            <div><strong>Predicted Next Tokens:</strong></div>
            {predicted_html}
            <div><strong>Actual Next Tokens:</strong></div>
            {actual_html}
            {perplexity_html}
        </div>
    </div>
    """

def create_visualization_item(fig, title):
    """Create an HTML visualization item from a matplotlib figure."""
    img_data = fig_to_base64(fig)
    return f"""
    <div class="viz-item">
        <div class="viz-title">{title}</div>
        <img class="viz-image" src="data:image/png;base64,{img_data}" alt="{title}">
    </div>
    """

def create_dashboard(
    metrics_history: Dict[str, List[float]],
    entropy_values: Optional[torch.Tensor] = None,
    grad_norm_values: Optional[torch.Tensor] = None,
    pruning_mask: Optional[torch.Tensor] = None,
    pruned_heads: Optional[List[Tuple[int, int]]] = None,
    attention_maps: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    sample_data: Optional[List[Dict[str, Any]]] = None,
    model_info: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None
) -> str:
    """
    Generate an interactive HTML dashboard for neural plasticity training.
    
    Args:
        metrics_history: Dictionary with training metrics history
        entropy_values: Optional tensor of entropy values for attention heads
        grad_norm_values: Optional tensor of gradient norm values for attention heads
        pruning_mask: Optional boolean tensor where True indicates a pruned head
        pruned_heads: Optional list of (layer, head) tuples for pruned heads
        attention_maps: Optional tensor of attention patterns
        sample_data: Optional list of sample prediction data from training
        model_info: Optional dictionary with model information
        save_path: Optional path to save the dashboard HTML file
        
    Returns:
        HTML string containing the dashboard
    """
    # Generate current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create metrics summary cards
    metrics_summary_html = ""
    
    # Extract final metrics
    final_metrics = {}
    initial_metrics = {}
    
    for key, values in metrics_history.items():
        if len(values) > 0:
            final_metrics[key] = values[-1]
            initial_metrics[key] = values[0]
    
    # Create metric cards for key metrics
    if "train_loss" in final_metrics:
        metrics_summary_html += create_metric_card(
            "Training Loss", 
            final_metrics["train_loss"], 
            initial_metrics.get("train_loss")
        )
    
    if "eval_loss" in final_metrics:
        metrics_summary_html += create_metric_card(
            "Evaluation Loss", 
            final_metrics["eval_loss"], 
            initial_metrics.get("eval_loss")
        )
    
    if "perplexity" in final_metrics:
        metrics_summary_html += create_metric_card(
            "Perplexity", 
            final_metrics["perplexity"], 
            initial_metrics.get("perplexity")
        )
    
    if "sparsity" in final_metrics:
        metrics_summary_html += create_metric_card(
            "Sparsity (%)", 
            final_metrics["sparsity"] * 100, 
            initial_metrics.get("sparsity", 0) * 100,
            format_str="{:.1f}%",
            is_lower_better=False
        )
    
    # Add additional model info if provided
    if model_info:
        if "total_params" in model_info:
            metrics_summary_html += create_metric_card(
                "Parameters", 
                model_info["total_params"], 
                None,
                format_str="{:,}"
            )
        
        if "model_size_mb" in model_info:
            metrics_summary_html += create_metric_card(
                "Model Size", 
                model_info["model_size_mb"], 
                None,
                format_str="{:.2f} MB"
            )
    
    # Generate training metrics visualization
    training_metrics_viz = ""
    if metrics_history:
        # Create training metrics plot
        metrics_fig = visualize_training_metrics(
            metrics_history=metrics_history,
            title="Training Progress"
        )
        training_metrics_viz += create_visualization_item(metrics_fig, "Training Metrics")
    
    # Generate pruning visualizations
    pruning_viz = ""
    if entropy_values is not None and grad_norm_values is not None:
        entropy_fig = visualize_head_entropy(
            entropy_values=entropy_values,
            title="Attention Entropy Heatmap"
        )
        pruning_viz += create_visualization_item(entropy_fig, "Attention Entropy")
        
        gradient_fig = visualize_head_gradients(
            grad_norm_values=grad_norm_values,
            pruned_heads=pruned_heads,
            title="Gradient Norms"
        )
        pruning_viz += create_visualization_item(gradient_fig, "Gradient Norms")
        
        if pruning_mask is not None:
            pruning_fig = visualize_pruning_decisions(
                grad_norm_values=grad_norm_values,
                pruning_mask=pruning_mask,
                title="Pruning Decisions"
            )
            pruning_viz += create_visualization_item(pruning_fig, "Pruning Decisions")
    
    # Generate pruning info section
    pruning_info = ""
    if pruned_heads:
        total_layers = len(set(layer for layer, _ in pruned_heads))
        pruning_info += f"""
        <div class="pruning-stat">
            Total pruned heads: <strong>{len(pruned_heads)}</strong>
        </div>
        <div class="pruning-stat">
            Affected layers: <strong>{total_layers}</strong>
        </div>
        """
        
        # Add sparsity if available
        if "sparsity" in final_metrics:
            pruning_info += f"""
            <div class="pruning-stat">
                Model sparsity: <strong>{final_metrics['sparsity']*100:.1f}%</strong>
            </div>
            """
    
    # Generate attention visualizations
    attention_viz = ""
    if attention_maps is not None:
        # If it's a list of attention maps per layer
        if isinstance(attention_maps, list):
            # Visualize a few representative layers
            if len(attention_maps) > 0:
                first_layer = visualize_attention_patterns(
                    attention_maps=attention_maps,
                    layer_idx=0,
                    title="First Layer Attention"
                )
                attention_viz += create_visualization_item(first_layer, "First Layer Attention")
                
                if len(attention_maps) > 1:
                    middle_idx = len(attention_maps) // 2
                    middle_layer = visualize_attention_patterns(
                        attention_maps=attention_maps,
                        layer_idx=middle_idx,
                        title=f"Middle Layer Attention (Layer {middle_idx})"
                    )
                    attention_viz += create_visualization_item(middle_layer, f"Middle Layer (Layer {middle_idx})")
                
                if len(attention_maps) > 2:
                    last_idx = len(attention_maps) - 1
                    last_layer = visualize_attention_patterns(
                        attention_maps=attention_maps,
                        layer_idx=last_idx,
                        title=f"Last Layer Attention (Layer {last_idx})"
                    )
                    attention_viz += create_visualization_item(last_layer, f"Last Layer (Layer {last_idx})")
        else:
            # Visualize a single attention tensor
            attn_fig = visualize_attention_patterns(
                attention_maps=attention_maps,
                layer_idx=0,  # Layer index doesn't matter for single tensor
                title="Attention Patterns"
            )
            attention_viz += create_visualization_item(attn_fig, "Attention Patterns")
    
    # Generate sample predictions section
    sample_predictions = ""
    if sample_data:
        # Sort samples by step for chronological display
        sorted_samples = sorted(sample_data, key=lambda x: x.get("step", 0))
        
        # Display a subset of samples if there are many
        max_samples = 5
        selected_samples = []
        
        if len(sorted_samples) <= max_samples:
            selected_samples = sorted_samples
        else:
            # Select samples distributed across training
            step_size = len(sorted_samples) // max_samples
            for i in range(0, len(sorted_samples), step_size):
                if len(selected_samples) < max_samples:
                    selected_samples.append(sorted_samples[i])
        
        # Create HTML for each sample
        for sample in selected_samples:
            step = sample.get("step", 0)
            sample_predictions += create_sample_html(sample, step)
    
    # Assemble the dashboard HTML
    dashboard_html = HTML_TEMPLATE.format(
        timestamp=timestamp,
        metrics_summary=metrics_summary_html,
        training_metrics_viz=training_metrics_viz,
        pruning_viz=pruning_viz,
        pruning_info=pruning_info,
        sample_predictions=sample_predictions,
        attention_viz=attention_viz
    )
    
    # Save dashboard to file if path provided
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(dashboard_html)
        print(f"Dashboard saved to: {save_path}")
    
    return dashboard_html


class DashboardReporter:
    """
    Reporter for creating and updating neural plasticity dashboards.
    
    This class collects metrics and visualizations during training
    and generates interactive HTML dashboards.
    """
    
    def __init__(
        self,
        output_dir: str = "dashboard",
        dashboard_name: str = "neural_plasticity_dashboard.html",
        auto_update: bool = True,
        update_interval: int = 20
    ):
        """
        Initialize the dashboard reporter.
        
        Args:
            output_dir: Directory to save dashboard files
            dashboard_name: Name of the main dashboard HTML file
            auto_update: Whether to automatically update the dashboard during training
            update_interval: How often to update the dashboard (in steps)
        """
        self.output_dir = output_dir
        self.dashboard_name = dashboard_name
        self.auto_update = auto_update
        self.update_interval = update_interval
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data storage
        self.metrics_history = {}
        self.sample_data = []
        self.model_info = {}
        self.pruning_info = {
            "entropy_values": None,
            "grad_norm_values": None,
            "pruning_mask": None,
            "pruned_heads": [],
        }
        self.attention_maps = []
        
        # Last update step
        self.last_update_step = -1
        
    def add_metrics(self, metrics: Dict[str, float], step: int):
        """
        Add metrics for the current step.
        
        Args:
            metrics: Dictionary of metric values
            step: Current training step
        """
        # Initialize history for new metrics
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            
            # Append the value
            self.metrics_history[key].append(value)
        
        # Add step if not present
        if "step" not in self.metrics_history:
            self.metrics_history["step"] = []
        
        # Ensure steps is the same length as other metrics
        while len(self.metrics_history["step"]) < len(next(iter(self.metrics_history.values()))):
            self.metrics_history["step"].append(step)
        
        # Auto-update dashboard if enabled
        if self.auto_update and (step - self.last_update_step >= self.update_interval):
            self.update_dashboard()
            self.last_update_step = step
    
    def add_sample(
        self,
        step: int,
        input_text: str,
        predicted_tokens: List[str],
        predicted_probs: List[float],
        actual_tokens: List[str],
        actual_probs: List[float],
        perplexities: List[float] = None
    ):
        """
        Add a sample prediction to the dashboard.
        
        Args:
            step: Current training step
            input_text: Input text context
            predicted_tokens: List of predicted next tokens
            predicted_probs: List of probabilities for predicted tokens
            actual_tokens: List of actual next tokens
            actual_probs: List of probabilities for actual tokens
            perplexities: Optional list of per-token perplexities
        """
        sample_entry = {
            "step": step,
            "input_text": input_text,
            "predicted_tokens": predicted_tokens,
            "predicted_probs": predicted_probs,
            "actual_tokens": actual_tokens,
            "actual_probs": actual_probs
        }
        
        if perplexities:
            sample_entry["perplexities"] = perplexities
        
        self.sample_data.append(sample_entry)
        
        # Auto-update dashboard if enabled
        if self.auto_update and (step - self.last_update_step >= self.update_interval):
            self.update_dashboard()
            self.last_update_step = step
    
    def update_pruning_info(
        self,
        entropy_values: Optional[torch.Tensor] = None,
        grad_norm_values: Optional[torch.Tensor] = None,
        pruning_mask: Optional[torch.Tensor] = None,
        pruned_heads: Optional[List[Tuple[int, int]]] = None
    ):
        """
        Update pruning visualization data.
        
        Args:
            entropy_values: Tensor of entropy values for attention heads
            grad_norm_values: Tensor of gradient norm values for attention heads
            pruning_mask: Boolean tensor where True indicates a pruned head
            pruned_heads: List of (layer, head) tuples for pruned heads
        """
        if entropy_values is not None:
            if isinstance(entropy_values, torch.Tensor):
                self.pruning_info["entropy_values"] = entropy_values.detach().cpu()
            else:
                self.pruning_info["entropy_values"] = entropy_values
        
        if grad_norm_values is not None:
            if isinstance(grad_norm_values, torch.Tensor):
                self.pruning_info["grad_norm_values"] = grad_norm_values.detach().cpu()
            else:
                self.pruning_info["grad_norm_values"] = grad_norm_values
        
        if pruning_mask is not None:
            if isinstance(pruning_mask, torch.Tensor):
                self.pruning_info["pruning_mask"] = pruning_mask.detach().cpu()
            else:
                self.pruning_info["pruning_mask"] = pruning_mask
        
        if pruned_heads is not None:
            self.pruning_info["pruned_heads"] = pruned_heads
    
    def add_attention_map(self, attention_map: torch.Tensor, layer_idx: int = 0):
        """
        Add an attention map for visualization.
        
        Args:
            attention_map: Attention tensor with shape [batch, heads, seq_len, seq_len]
            layer_idx: Layer index for this attention map
        """
        # Convert tensor to CPU numpy array for storage
        if isinstance(attention_map, torch.Tensor):
            attention_map = attention_map.detach().cpu()
        
        # Store in per-layer format
        while layer_idx >= len(self.attention_maps):
            self.attention_maps.append(None)
        
        self.attention_maps[layer_idx] = attention_map
    
    def set_model_info(self, model_info: Dict[str, Any]):
        """
        Set model information for the dashboard.
        
        Args:
            model_info: Dictionary with model information
        """
        self.model_info = model_info
    
    def update_dashboard(self):
        """
        Update the dashboard with current data.
        """
        dashboard_path = os.path.join(self.output_dir, self.dashboard_name)
        
        # Generate dashboard HTML
        create_dashboard(
            metrics_history=self.metrics_history,
            entropy_values=self.pruning_info["entropy_values"],
            grad_norm_values=self.pruning_info["grad_norm_values"],
            pruning_mask=self.pruning_info["pruning_mask"],
            pruned_heads=self.pruning_info["pruned_heads"],
            attention_maps=self.attention_maps if len(self.attention_maps) > 0 else None,
            sample_data=self.sample_data,
            model_info=self.model_info,
            save_path=dashboard_path
        )
        
        return dashboard_path
    
    def get_sample_callback(self):
        """
        Get a callback function for adding samples during training.
        
        Returns:
            Function that can be used as a sample_callback in PlasticityTrainer
        """
        def sample_callback(step, sample_data):
            self.add_sample(
                step=step,
                input_text=sample_data.get("input_text", ""),
                predicted_tokens=sample_data.get("predicted_tokens", []),
                predicted_probs=sample_data.get("predicted_probs", []),
                actual_tokens=sample_data.get("actual_tokens", []),
                actual_probs=sample_data.get("actual_probs", []),
                perplexities=sample_data.get("perplexities", [])
            )
        
        return sample_callback
    
    def get_metrics_callback(self):
        """
        Get a callback function for adding metrics during training.
        
        Returns:
            Function that can be used as a callback in PlasticityTrainer
        """
        def metrics_callback(step, metrics):
            self.add_metrics(metrics, step)
        
        return metrics_callback