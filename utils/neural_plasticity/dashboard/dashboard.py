"""
Neural Plasticity Interactive Dashboard

This module provides a real-time interactive dashboard for neural plasticity
experiments, allowing users to monitor the experiment progress, visualize
pruning decisions, and analyze model performance.

Version: v0.0.2 (2025-04-20 17:45:00)
"""

import os
import json
import base64
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

import numpy as np
import matplotlib
# Use Agg backend for non-interactive plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO

# Try to import web server components
try:
    from http.server import HTTPServer, BaseHTTPRequestHandler
    WEB_SERVER_AVAILABLE = True
except ImportError:
    WEB_SERVER_AVAILABLE = False
    print("Warning: HTTP server components not available. Live dashboard will be static only.")

# Constants
DEFAULT_PORT = 8080
DEFAULT_HOST = 'localhost'
REFRESH_INTERVAL = 5  # seconds
MAX_HISTORY_LENGTH = 1000

logger = logging.getLogger(__name__)

# HTML template for the dashboard
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dynamic Neural Plasticity Experiment Report</title>
    <meta http-equiv="refresh" content="{refresh_interval}" />
    <style>
        :root {
            --primary-color: #3f51b5;
            --secondary-color: #8bc34a;
            --accent-color: #ff5722;
            --background-color: #f5f5f5;
            --card-color: #ffffff;
            --text-color: #333333;
            --border-color: #dddddd;
            --success-color: #4caf50;
            --warning-color: #ff9800;
            --danger-color: #f44336;
            --info-color: #2196f3;
            --pruned-color: #e53935;
            --active-color: #43a047;
            --neutral-color: #607d8b;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: var(--background-color);
            color: var(--text-color);
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background-color: var(--primary-color);
            color: white;
            padding: 15px 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        header h1 {
            margin: 0;
            font-size: 1.8rem;
        }
        
        .timestamp {
            font-size: 0.9rem;
            opacity: 0.9;
        }
        
        .card {
            background-color: var(--card-color);
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            overflow: hidden;
        }
        
        .card-header {
            background-color: #f0f0f0;
            padding: 10px 15px;
            border-bottom: 1px solid var(--border-color);
            font-weight: bold;
        }
        
        .card-body {
            padding: 15px;
        }
        
        .tabs {
            display: flex;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 20px;
        }
        
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }
        
        .tab:hover {
            background-color: rgba(0,0,0,0.05);
        }
        
        .tab.active {
            border-bottom: 3px solid var(--primary-color);
            font-weight: bold;
            color: var(--primary-color);
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background-color: var(--card-color);
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 15px;
            text-align: center;
        }
        
        .metric-title {
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: var(--primary-color);
        }
        
        .metric-change {
            font-size: 0.8rem;
            margin-top: 5px;
        }
        
        .positive-change {
            color: var(--success-color);
        }
        
        .negative-change {
            color: var(--danger-color);
        }
        
        .charts-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .chart-card {
            background-color: var(--card-color);
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .chart-header {
            padding: 10px 15px;
            border-bottom: 1px solid var(--border-color);
            font-weight: bold;
        }
        
        .chart-body {
            padding: 15px;
            text-align: center;
        }
        
        .chart-img {
            max-width: 100%;
            height: auto;
        }
        
        .timeline {
            position: relative;
            margin: 20px 0;
            padding-left: 30px;
        }
        
        .timeline-item {
            position: relative;
            margin-bottom: 20px;
            padding-left: 20px;
        }
        
        .timeline-item:before {
            content: '';
            position: absolute;
            left: -12px;
            top: 0;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: var(--primary-color);
            border: 2px solid white;
        }
        
        .timeline-item:after {
            content: '';
            position: absolute;
            left: -6px;
            top: 14px;
            width: 2px;
            height: calc(100% + 10px);
            background-color: var(--border-color);
        }
        
        .timeline-item:last-child:after {
            display: none;
        }
        
        .timeline-title {
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .timeline-time {
            font-size: 0.8rem;
            color: #666;
            margin-bottom: 10px;
        }
        
        .timeline-content {
            background-color: var(--card-color);
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 15px;
        }
        
        .pruning-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .pruning-card {
            background-color: var(--card-color);
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .pruning-header {
            padding: 10px 15px;
            border-bottom: 1px solid var(--border-color);
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .pruning-cycle {
            font-size: 0.8rem;
            padding: 3px 8px;
            background-color: var(--primary-color);
            color: white;
            border-radius: 10px;
        }
        
        .pruning-body {
            padding: 15px;
        }
        
        .pruning-details {
            margin-bottom: 15px;
        }
        
        .details-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        
        .detail-item {
            font-size: 0.9rem;
        }
        
        .detail-label {
            font-weight: bold;
            margin-right: 5px;
        }
        
        .head-status {
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 3px;
            margin-right: 5px;
            vertical-align: middle;
        }
        
        .head-pruned {
            background-color: var(--pruned-color);
        }
        
        .head-active {
            background-color: var(--active-color);
        }
        
        .head-neutral {
            background-color: var(--neutral-color);
        }
        
        .progress-section {
            margin-bottom: 20px;
        }
        
        .progress-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .progress-title {
            font-weight: bold;
        }
        
        .progress-value {
            font-size: 0.9rem;
            color: #666;
        }
        
        .progress-bar-container {
            width: 100%;
            height: 8px;
            background-color: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-bar {
            height: 100%;
            background-color: var(--primary-color);
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        
        .text-samples {
            margin-top: 20px;
        }
        
        .sample-card {
            background-color: var(--card-color);
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 15px;
            margin-bottom: 20px;
        }
        
        .sample-prompt {
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .sample-text {
            font-family: monospace;
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            white-space: pre-wrap;
            overflow-x: auto;
        }
        
        .status-box {
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
        }
        
        .status-icon {
            font-size: 1.5rem;
            margin-right: 10px;
        }
        
        .status-running {
            background-color: rgba(33, 150, 243, 0.1);
            border: 1px solid rgba(33, 150, 243, 0.5);
            color: var(--info-color);
        }
        
        .status-success {
            background-color: rgba(76, 175, 80, 0.1);
            border: 1px solid rgba(76, 175, 80, 0.5);
            color: var(--success-color);
        }
        
        .status-warning {
            background-color: rgba(255, 152, 0, 0.1);
            border: 1px solid rgba(255, 152, 0, 0.5);
            color: var(--warning-color);
        }
        
        .status-error {
            background-color: rgba(244, 67, 54, 0.1);
            border: 1px solid rgba(244, 67, 54, 0.5);
            color: var(--danger-color);
        }
        
        .phase-indicator {
            display: flex;
            margin-bottom: 20px;
        }
        
        .phase {
            flex: 1;
            text-align: center;
            padding: 10px;
            background-color: #e0e0e0;
            margin: 0 5px;
            border-radius: 5px;
            position: relative;
        }
        
        .phase:not(:last-child):after {
            content: '';
            position: absolute;
            top: 50%;
            right: -10px;
            width: 10px;
            height: 2px;
            background-color: #e0e0e0;
        }
        
        .phase.completed {
            background-color: var(--success-color);
            color: white;
        }
        
        .phase.active {
            background-color: var(--primary-color);
            color: white;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        
        table th,
        table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        table th {
            background-color: #f0f0f0;
            font-weight: bold;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .charts-container {
                grid-template-columns: 1fr;
            }
            
            .metric-card {
                grid-column: span 2;
            }
            
            .container {
                padding: 10px;
            }
            
            header {
                flex-direction: column;
                align-items: flex-start;
            }
            
            .timestamp {
                margin-top: 5px;
            }
        }
        
        footer {
            text-align: center;
            margin-top: 40px;
            font-size: 0.8rem;
            color: #666;
            border-top: 1px solid var(--border-color);
            padding-top: 20px;
        }
        
        .footer-quote {
            font-style: italic;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Dynamic Neural Plasticity Experiment Report</h1>
            <div class="timestamp">Generated: {timestamp}</div>
        </header>
        
        <div class="status-box {status_class}">
            <div class="status-icon">●</div>
            <div class="status-message">
                <strong>Status:</strong> {status_message}
                {status_details}
            </div>
        </div>
        
        <div class="phase-indicator">
            <div class="phase {setup_phase_class}">Setup</div>
            <div class="phase {warmup_phase_class}">Warmup</div>
            <div class="phase {analysis_phase_class}">Analysis</div>
            <div class="phase {pruning_phase_class}">Pruning</div>
            <div class="phase {evaluation_phase_class}">Evaluation</div>
            <div class="phase {completion_phase_class}">Complete</div>
        </div>
        
        <div class="metrics-grid">
            {metrics_grid}
        </div>
        
        <div class="tabs">
            <div class="tab active" data-tab="overview-tab">Overview</div>
            <div class="tab" data-tab="pruning-tab">Pruning Analysis</div>
            <div class="tab" data-tab="head-timeline-tab">Head Timeline</div>
            <div class="tab" data-tab="event-log-tab">Event Log</div>
            <div class="tab" data-tab="visualization-tab">Decision Visualizations</div>
            <div class="tab" data-tab="text-generation-tab">Text Generation</div>
        </div>
        
        <!-- Overview Tab -->
        <div id="overview-tab" class="tab-content active">
            <div class="charts-container">
                {overview_charts}
            </div>
        </div>
        
        <!-- Pruning Analysis Tab -->
        <div id="pruning-tab" class="tab-content">
            <div class="card">
                <div class="card-header">Pruning Decision Process</div>
                <div class="card-body">
                    <p>Detailed visualizations showing exactly why the system made specific pruning and growing decisions. These visualizations provide transparency into the mathematical decision process for each operation.</p>
                    
                    <h3>Decision Criteria</h3>
                    <ul>
                        <li><strong>Entropy:</strong> Higher values indicate more dispersed attention (less focused)</li>
                        <li><strong>Gradient Intensity:</strong> Higher values indicate less contribution to model learning</li>
                        <li><strong>Combined Score:</strong> f(x) = Entropy * 0.6 + Gradient * 0.4 are pruning candidates</li>
                    </ul>
                    
                    <div class="pruning-grid">
                        {pruning_decision_cards}
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Head Timeline Tab -->
        <div id="head-timeline-tab" class="tab-content">
            <div class="card">
                <div class="card-header">Attention Head Activity Timeline</div>
                <div class="card-body">
                    {head_timeline_chart}
                    <p>Dynamic visualization of the neural plasticity process, showing training loss, pruning events (red), and growing events (green).</p>
                </div>
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Layer</th>
                        <th>Head</th>
                        <th>Status</th>
                        <th>Entropy</th>
                        <th>Gradient Norm</th>
                        <th>Pruned At Step</th>
                    </tr>
                </thead>
                <tbody>
                    {head_status_rows}
                </tbody>
            </table>
        </div>
        
        <!-- Event Log Tab -->
        <div id="event-log-tab" class="tab-content">
            <div class="card">
                <div class="card-header">Experiment Events</div>
                <div class="card-body">
                    <div class="timeline">
                        {event_timeline}
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Decision Visualizations Tab -->
        <div id="visualization-tab" class="tab-content">
            <div class="card">
                <div class="card-header">Decision Process Visualizations</div>
                <div class="card-body">
                    <p>Detailed visualizations showing exactly why the system made specific pruning and growing decisions. These visualizations provide transparency into the mathematical decision process for each operation.</p>
                    
                    <h3>This experiment includes {pruning_decision_count} pruning decisions with detailed visualizations:</h3>
                    
                    <div class="pruning-grid">
                        {detailed_visualizations}
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Text Generation Tab -->
        <div id="text-generation-tab" class="tab-content">
            <div class="card">
                <div class="card-header">Generated Text Samples</div>
                <div class="card-body">
                    <p>Text samples generated by the model at different stages of the experiment.</p>
                    
                    <div class="text-samples">
                        {text_samples}
                    </div>
                </div>
            </div>
        </div>
        
        <footer>
            <p>Neural Plasticity Experiment Report | Generated on {timestamp}</p>
            <p class="footer-quote">"A carefully pruned network is like a well-written sentence - nothing extra, nothing missing."</p>
        </footer>
    </div>
    
    <script>
        // Tab switching functionality
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                // Remove active class from all tabs
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                
                // Add active class to clicked tab
                tab.classList.add('active');
                document.getElementById(tab.dataset.tab).classList.add('active');
            });
        });
    </script>
</body>
</html>
"""

class LiveDashboard:
    """
    LiveDashboard manages the data and state for the neural plasticity dashboard.
    """
    
    def __init__(self, output_dir: str, dashboard_name: str = "dashboard.html", 
                 auto_refresh: bool = True, refresh_interval: int = REFRESH_INTERVAL):
        """
        Initialize the dashboard with the output directory and settings.
        
        Args:
            output_dir: Directory to save the dashboard files
            dashboard_name: Filename for the main dashboard HTML file
            auto_refresh: Whether to auto-refresh the dashboard page
            refresh_interval: How often to refresh the page (in seconds)
        """
        self.output_dir = Path(output_dir)
        self.dashboard_name = dashboard_name
        self.auto_refresh = auto_refresh
        self.refresh_interval = refresh_interval
        
        # Create dashboard directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize data structures
        self.metrics_history = {}
        self.training_loss = []
        self.eval_loss = []
        self.perplexity = []
        self.entropy_values = None
        self.gradient_values = None
        self.pruning_decisions = []
        self.pruned_heads = []
        self.event_log = []
        self.text_samples = []
        self.status = "initializing"
        self.phase = "setup"
        self.experiment_config = {}
        
        # Add initialization event
        self._add_event("Experiment initialized", "Dashboard created and ready for data updates.")
        
        # Create a minimal dashboard file to start with
        self._create_minimal_dashboard()
        
        logger.info(f"Dashboard initialized at {self.output_dir / self.dashboard_name}")
        
    def _create_minimal_dashboard(self):
        """Create a minimal initial dashboard to avoid template parsing issues."""
        minimal_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Neural Plasticity Dashboard</title>
    <meta http-equiv="refresh" content="{self.refresh_interval}" />
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ font-size: 24px; margin-bottom: 20px; }}
        .card {{ border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 4px; }}
        .status {{ font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">Neural Plasticity Experiment Dashboard</div>
    <div class="card">
        <div class="status">Status: {self.status}</div>
        <p>Phase: {self.phase}</p>
        <p>Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
</body>
</html>
"""
        # Write the file
        with open(self.output_dir / self.dashboard_name, "w", encoding="utf-8") as f:
            f.write(minimal_html)
    
    def update_metrics(self, metrics: Dict[str, Union[float, int, List[float]]]):
        """
        Update the metrics data for the dashboard.
        
        Args:
            metrics: Dictionary of metrics to update
        """
        # Add metrics to history
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            
            # Handle different data types
            if isinstance(value, (int, float)):
                self.metrics_history[key].append(value)
            elif isinstance(value, list):
                # If we receive a list, extend the existing list
                self.metrics_history[key].extend(value)
            
            # Trim to max history length
            if len(self.metrics_history[key]) > MAX_HISTORY_LENGTH:
                self.metrics_history[key] = self.metrics_history[key][-MAX_HISTORY_LENGTH:]
        
        # Extract key metrics for easier access
        if 'train_loss' in metrics:
            self.training_loss = self.metrics_history.get('train_loss', [])
        if 'eval_loss' in metrics:
            self.eval_loss = self.metrics_history.get('eval_loss', [])
        if 'perplexity' in metrics:
            self.perplexity = self.metrics_history.get('perplexity', [])
        
        # Update dashboard
        self._update_dashboard()
    
    def update_entropy(self, entropy_values: np.ndarray):
        """
        Update entropy values for attention heads.
        
        Args:
            entropy_values: Tensor or array of entropy values [layers, heads]
        """
        # Convert to numpy if needed
        if not isinstance(entropy_values, np.ndarray):
            try:
                import torch
                if isinstance(entropy_values, torch.Tensor):
                    entropy_values = entropy_values.detach().cpu().numpy()
            except (ImportError, AttributeError):
                pass
        
        self.entropy_values = entropy_values
        self._add_event("Entropy analysis completed", 
                       f"Analyzed entropy for {entropy_values.shape[0]} layers with {entropy_values.shape[1]} heads each.")
        self._update_dashboard()
    
    def update_gradients(self, gradient_values: np.ndarray):
        """
        Update gradient values for attention heads.
        
        Args:
            gradient_values: Tensor or array of gradient values [layers, heads]
        """
        # Convert to numpy if needed
        if not isinstance(gradient_values, np.ndarray):
            try:
                import torch
                if isinstance(gradient_values, torch.Tensor):
                    gradient_values = gradient_values.detach().cpu().numpy()
            except (ImportError, AttributeError):
                pass
        
        self.gradient_values = gradient_values
        self._add_event("Gradient analysis completed", 
                       f"Analyzed gradients for {gradient_values.shape[0]} layers with {gradient_values.shape[1]} heads each.")
        self._update_dashboard()
    
    def add_pruning_decision(self, decision: Dict[str, Any]):
        """
        Add a new pruning decision event.
        
        Args:
            decision: Dictionary with pruning decision details
        """
        self.pruning_decisions.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'cycle': len(self.pruning_decisions) + 1,
            'details': decision
        })
        
        # Update pruned heads list
        if 'pruned_heads' in decision:
            self.pruned_heads.extend(decision['pruned_heads'])
            # Remove duplicates
            self.pruned_heads = list(set([tuple(h) for h in self.pruned_heads]))
            self.pruned_heads = [list(h) for h in self.pruned_heads]
        
        self._add_event(f"Pruning cycle {len(self.pruning_decisions)} completed", 
                       f"Pruned {len(decision.get('pruned_heads', []))} heads based on {decision.get('strategy', 'unknown')} strategy.")
        self._update_dashboard()
    
    def add_text_sample(self, prompt: str, generated_text: str):
        """
        Add a text generation sample.
        
        Args:
            prompt: Text prompt used for generation
            generated_text: Text generated by the model
        """
        self.text_samples.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'prompt': prompt,
            'text': generated_text
        })
        self._update_dashboard()
    
    def set_status(self, status: str, details: str = ""):
        """
        Update the experiment status.
        
        Args:
            status: Status string (initializing, running, completed, error)
            details: Additional status details
        """
        self.status = status
        self._add_event(f"Status changed to: {status}", details)
        self._update_dashboard()
    
    def set_phase(self, phase: str):
        """
        Update the current experiment phase.
        
        Args:
            phase: Current phase (setup, warmup, analysis, pruning, evaluation, complete)
        """
        self.phase = phase
        self._add_event(f"Phase changed to: {phase}", f"Experiment entered {phase} phase.")
        self._update_dashboard()
    
    def update_config(self, config: Dict[str, Any]):
        """
        Update experiment configuration information.
        
        Args:
            config: Dictionary of configuration parameters
        """
        self.experiment_config = config
        self._add_event("Experiment configuration updated", 
                       f"Updated configuration with {len(config)} parameters.")
        self._update_dashboard()
    
    def _add_event(self, title: str, description: str):
        """
        Add an event to the event log.
        
        Args:
            title: Event title
            description: Event description
        """
        self.event_log.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'title': title,
            'description': description
        })
        
        # Trim event log if too long
        if len(self.event_log) > MAX_HISTORY_LENGTH:
            self.event_log = self.event_log[-MAX_HISTORY_LENGTH:]
    
    def _update_dashboard(self):
        """Generate and save the dashboard HTML file."""
        try:
            # Generate dashboard components
            metrics_grid = self._generate_metrics_grid()
            overview_charts = self._generate_overview_charts()
            pruning_decision_cards = self._generate_pruning_cards()
            head_timeline_chart = self._generate_head_timeline()
            head_status_rows = self._generate_head_status_table()
            event_timeline = self._generate_event_timeline()
            detailed_visualizations = self._generate_detailed_visualizations()
            text_samples = self._generate_text_samples()
            
            # Determine status class
            status_class = "status-running"
            status_message = "Experiment in progress"
            status_details = ""
            
            if self.status == "initializing":
                status_class = "status-info"
                status_message = "Initializing experiment"
            elif self.status == "running":
                status_class = "status-running"
                status_message = "Experiment in progress"
                if self.phase:
                    status_details = f"<div>Current phase: {self.phase}</div>"
            elif self.status == "completed":
                status_class = "status-success"
                status_message = "Experiment completed successfully"
            elif self.status == "error":
                status_class = "status-error"
                status_message = "Experiment encountered an error"
            elif self.status == "warning":
                status_class = "status-warning"
                status_message = "Experiment running with warnings"
            
            # Determine phase classes
            phase_classes = {
                "setup": "",
                "warmup": "",
                "analysis": "",
                "pruning": "",
                "evaluation": "",
                "completion": ""
            }
            
            # Mark completed phases
            completed_phases = []
            if self.phase == "warmup":
                completed_phases = ["setup"]
            elif self.phase == "analysis":
                completed_phases = ["setup", "warmup"]
            elif self.phase == "pruning":
                completed_phases = ["setup", "warmup", "analysis"]
            elif self.phase == "evaluation":
                completed_phases = ["setup", "warmup", "analysis", "pruning"]
            elif self.phase == "complete":
                completed_phases = ["setup", "warmup", "analysis", "pruning", "evaluation"]
            
            for p in completed_phases:
                phase_classes[p] = "completed"
            
            # Mark active phase
            if self.phase in phase_classes:
                phase_classes[self.phase] = "active"
            
            try:
                # Generate HTML
                html = HTML_TEMPLATE.format(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    refresh_interval=self.refresh_interval if self.auto_refresh else 0,
                    metrics_grid=metrics_grid,
                    overview_charts=overview_charts,
                    pruning_decision_cards=pruning_decision_cards,
                    head_timeline_chart=head_timeline_chart,
                    head_status_rows=head_status_rows,
                    event_timeline=event_timeline,
                    detailed_visualizations=detailed_visualizations,
                    text_samples=text_samples,
                    status_class=status_class,
                    status_message=status_message,
                    status_details=status_details,
                    setup_phase_class=phase_classes["setup"],
                    warmup_phase_class=phase_classes["warmup"],
                    analysis_phase_class=phase_classes["analysis"],
                    pruning_phase_class=phase_classes["pruning"],
                    evaluation_phase_class=phase_classes["evaluation"],
                    completion_phase_class=phase_classes["completion"],
                    pruning_decision_count=len(self.pruning_decisions)
                )
                
                # Save HTML file
                with open(self.output_dir / self.dashboard_name, 'w', encoding='utf-8') as f:
                    f.write(html)
            except Exception as e:
                logger.error(f"Error generating complex dashboard: {e}")
                # Fall back to minimal dashboard
                self._create_minimal_dashboard()
            
            # Save metrics to JSON for client-side updates
            with open(self.output_dir / 'dashboard_data.json', 'w', encoding='utf-8') as f:
                json.dump({
                    'metrics': self.metrics_history,
                    'pruning_decisions': self.pruning_decisions,
                    'pruned_heads': self.pruned_heads,
                    'event_log': self.event_log,
                    'text_samples': self.text_samples,
                    'status': self.status,
                    'phase': self.phase,
                    'config': self.experiment_config,
                    'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")
            # Ensure at least a minimal dashboard exists
            try:
                self._create_minimal_dashboard()
            except Exception as e2:
                logger.error(f"Failed to create even minimal dashboard: {e2}")
    
    def _generate_metrics_grid(self) -> str:
        """Generate the metrics summary grid HTML."""
        if not self.metrics_history:
            return "<div class='metric-card'>No metrics data available yet</div>"
        
        html = ""
        
        # Get the latest metrics
        if self.training_loss:
            latest_loss = self.training_loss[-1]
            change = ""
            if len(self.training_loss) > 1:
                loss_change = latest_loss - self.training_loss[-2]
                change_class = "positive-change" if loss_change < 0 else "negative-change"
                change = f"<div class='metric-change {change_class}'>{abs(loss_change):.4f} {'↓' if loss_change < 0 else '↑'}</div>"
            
            html += f"""
            <div class='metric-card'>
                <div class='metric-title'>Training Loss</div>
                <div class='metric-value'>{latest_loss:.4f}</div>
                {change}
            </div>
            """
        
        if self.eval_loss:
            latest_loss = self.eval_loss[-1]
            change = ""
            if len(self.eval_loss) > 1:
                loss_change = latest_loss - self.eval_loss[-2]
                change_class = "positive-change" if loss_change < 0 else "negative-change"
                change = f"<div class='metric-change {change_class}'>{abs(loss_change):.4f} {'↓' if loss_change < 0 else '↑'}</div>"
            
            html += f"""
            <div class='metric-card'>
                <div class='metric-title'>Evaluation Loss</div>
                <div class='metric-value'>{latest_loss:.4f}</div>
                {change}
            </div>
            """
        
        if self.perplexity:
            latest_ppl = self.perplexity[-1]
            change = ""
            if len(self.perplexity) > 1:
                ppl_change = latest_ppl - self.perplexity[-2]
                change_class = "positive-change" if ppl_change < 0 else "negative-change"
                change = f"<div class='metric-change {change_class}'>{abs(ppl_change):.2f} {'↓' if ppl_change < 0 else '↑'}</div>"
            
            html += f"""
            <div class='metric-card'>
                <div class='metric-title'>Perplexity</div>
                <div class='metric-value'>{latest_ppl:.2f}</div>
                {change}
            </div>
            """
        
        # Add pruned head count
        if self.pruned_heads:
            html += f"""
            <div class='metric-card'>
                <div class='metric-title'>Pruned Heads</div>
                <div class='metric-value'>{len(self.pruned_heads)}</div>
            </div>
            """
        
        # Add model sparsity if available
        if 'sparsity' in self.metrics_history and self.metrics_history['sparsity']:
            latest_sparsity = self.metrics_history['sparsity'][-1] * 100  # Convert to percentage
            html += f"""
            <div class='metric-card'>
                <div class='metric-title'>Model Sparsity</div>
                <div class='metric-value'>{latest_sparsity:.1f}%</div>
            </div>
            """
        
        # Add total steps
        if 'step' in self.metrics_history and self.metrics_history['step']:
            html += f"""
            <div class='metric-card'>
                <div class='metric-title'>Training Steps</div>
                <div class='metric-value'>{self.metrics_history['step'][-1]}</div>
            </div>
            """
        
        return html
    
    def _generate_overview_charts(self) -> str:
        """Generate overview charts HTML with embedded images."""
        html = ""
        
        # Generate training loss chart if data available
        if self.training_loss:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            steps = list(range(len(self.training_loss)))
            
            # Plot training loss
            ax.plot(steps, self.training_loss, label='Training Loss')
            
            # Plot eval loss if available
            if self.eval_loss and len(self.eval_loss) == len(self.training_loss):
                ax.plot(steps, self.eval_loss, label='Eval Loss')
            
            # Mark pruning events
            for decision in self.pruning_decisions:
                cycle = decision['cycle']
                # Find the step where this cycle happened (approximate)
                step_idx = min(len(steps) - 1, (cycle * len(steps)) // max(1, len(self.pruning_decisions)))
                ax.axvline(x=step_idx, color='r', linestyle='--', alpha=0.5)
            
            ax.set_title('Dynamic Neural Plasticity: Training Process')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Save to buffer
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            
            # Convert to base64
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            
            # Add to HTML
            html += f"""
            <div class="chart-card">
                <div class="chart-header">Neural Plasticity: Training Process</div>
                <div class="chart-body">
                    <img class="chart-img" src="data:image/png;base64,{img_str}" alt="Training Loss Chart">
                </div>
            </div>
            """
        
        # Generate perplexity chart if data available
        if self.perplexity:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            steps = list(range(len(self.perplexity)))
            
            # Plot perplexity
            ax.plot(steps, self.perplexity, label='Perplexity', color='purple')
            
            # Mark pruning events
            for decision in self.pruning_decisions:
                cycle = decision['cycle']
                # Find the step where this cycle happened (approximate)
                step_idx = min(len(steps) - 1, (cycle * len(steps)) // max(1, len(self.pruning_decisions)))
                ax.axvline(x=step_idx, color='r', linestyle='--', alpha=0.5)
            
            ax.set_title('Model Perplexity')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Perplexity')
            ax.grid(True, alpha=0.3)
            
            # Save to buffer
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            
            # Convert to base64
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            
            # Add to HTML
            html += f"""
            <div class="chart-card">
                <div class="chart-header">Model Perplexity</div>
                <div class="chart-body">
                    <img class="chart-img" src="data:image/png;base64,{img_str}" alt="Perplexity Chart">
                </div>
            </div>
            """
        
        # Generate entropy heatmap if data available
        if self.entropy_values is not None:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot entropy heatmap
            im = ax.imshow(self.entropy_values, cmap='viridis')
            fig.colorbar(im, ax=ax, label='Entropy')
            
            # Mark pruned heads with X
            if self.pruned_heads:
                for layer, head in self.pruned_heads:
                    if 0 <= layer < self.entropy_values.shape[0] and 0 <= head < self.entropy_values.shape[1]:
                        ax.plot(head, layer, 'rx', markersize=10)
            
            ax.set_title('Attention Head Entropy')
            ax.set_xlabel('Head Index')
            ax.set_ylabel('Layer Index')
            
            # Save to buffer
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            
            # Convert to base64
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            
            # Add to HTML
            html += f"""
            <div class="chart-card">
                <div class="chart-header">Attention Head Entropy</div>
                <div class="chart-body">
                    <img class="chart-img" src="data:image/png;base64,{img_str}" alt="Entropy Heatmap">
                </div>
            </div>
            """
        
        # Generate gradient heatmap if data available
        if self.gradient_values is not None:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot gradient heatmap
            im = ax.imshow(self.gradient_values, cmap='plasma')
            fig.colorbar(im, ax=ax, label='Gradient Norm')
            
            # Mark pruned heads with X
            if self.pruned_heads:
                for layer, head in self.pruned_heads:
                    if 0 <= layer < self.gradient_values.shape[0] and 0 <= head < self.gradient_values.shape[1]:
                        ax.plot(head, layer, 'rx', markersize=10)
            
            ax.set_title('Attention Head Gradient Norms')
            ax.set_xlabel('Head Index')
            ax.set_ylabel('Layer Index')
            
            # Save to buffer
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            
            # Convert to base64
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            
            # Add to HTML
            html += f"""
            <div class="chart-card">
                <div class="chart-header">Attention Head Gradient Norms</div>
                <div class="chart-body">
                    <img class="chart-img" src="data:image/png;base64,{img_str}" alt="Gradient Heatmap">
                </div>
            </div>
            """
        
        # If no charts yet, show placeholder
        if not html:
            html = """
            <div class="chart-card">
                <div class="chart-header">No Data Available Yet</div>
                <div class="chart-body">
                    <p>Waiting for experiment data to become available...</p>
                </div>
            </div>
            """
        
        return html
    
    def _generate_pruning_cards(self) -> str:
        """Generate pruning decision cards HTML."""
        if not self.pruning_decisions:
            return "<p>No pruning decisions made yet.</p>"
        
        html = ""
        for decision in self.pruning_decisions:
            cycle = decision['cycle']
            timestamp = decision['timestamp']
            details = decision['details']
            
            # Extract key metrics
            strategy = details.get('strategy', 'Unknown')
            pruned_count = len(details.get('pruned_heads', []))
            
            html += f"""
            <div class="pruning-card">
                <div class="pruning-header">
                    Pruning Decision <span class="pruning-cycle">Cycle {cycle}</span>
                </div>
                <div class="pruning-body">
                    <div class="pruning-details">
                        <div class="details-grid">
                            <div class="detail-item">
                                <span class="detail-label">Strategy:</span> {strategy}
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">Heads Pruned:</span> {pruned_count}
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">Timestamp:</span> {timestamp}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """
        
        return html
    
    def _generate_head_timeline(self) -> str:
        """Generate the head timeline visualization HTML."""
        # If no pruning decisions yet, show placeholder
        if not self.pruned_heads:
            return "<p>No pruning events recorded yet.</p>"
        
        # Calculate total number of heads (assume model structure from entropy values)
        if self.entropy_values is not None:
            num_layers, num_heads = self.entropy_values.shape
            total_heads = num_layers * num_heads
        else:
            # Estimate from pruned heads
            max_layer = max([h[0] for h in self.pruned_heads]) if self.pruned_heads else 0
            max_head = max([h[1] for h in self.pruned_heads]) if self.pruned_heads else 0
            num_layers = max_layer + 1
            num_heads = max_head + 1
            total_heads = num_layers * num_heads
        
        # Create a timeline figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Y-axis will represent heads (layer * num_heads + head)
        y_positions = []
        head_states = np.ones((num_layers, num_heads))  # 1 = active, 0 = pruned
        
        # Mark pruned heads
        for layer, head in self.pruned_heads:
            head_states[layer, head] = 0
        
        # Plot pruned heads as red bars
        for layer in range(num_layers):
            for head in range(num_heads):
                head_idx = layer * num_heads + head
                y_positions.append(head_idx)
                
                # Plot pruned head
                if head_states[layer, head] == 0:
                    ax.barh(head_idx, width=150, left=40, height=0.8, color='darkred', alpha=0.8)
        
        # Add labels for clarity
        ax.set_title('Attention Head Activity Timeline')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Head Index (layer × num_heads + head)')
        ax.set_ylim(-1, total_heads)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkred', edgecolor='darkred', label='Pruned Head'),
            Patch(facecolor='yellow', edgecolor='yellow', label='Active Head', alpha=0.1)
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Highlight steps where pruning occurred
        for i, decision in enumerate(self.pruning_decisions):
            step = i * 40  # Approximate step based on cycle
            ax.axvline(x=step, color='r', linestyle='--', alpha=0.5)
        
        # Save to buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        
        # Convert to base64
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        # Return HTML img tag
        return f'<img class="chart-img" src="data:image/png;base64,{img_str}" alt="Head Timeline Chart">'
    
    def _generate_head_status_table(self) -> str:
        """Generate the head status table HTML."""
        if not self.pruned_heads and not self.entropy_values:
            return "<tr><td colspan='6'>No head data available yet.</td></tr>"
        
        html = ""
        
        # If we have entropy and gradient values, show all heads
        if self.entropy_values is not None and self.gradient_values is not None:
            num_layers, num_heads = self.entropy_values.shape
            
            for layer in range(num_layers):
                for head in range(num_heads):
                    # Determine if head is pruned
                    is_pruned = [layer, head] in self.pruned_heads or (layer, head) in self.pruned_heads
                    status = "Pruned" if is_pruned else "Active"
                    status_class = "head-pruned" if is_pruned else "head-active"
                    
                    # Get entropy and gradient values
                    entropy = self.entropy_values[layer, head]
                    gradient = self.gradient_values[layer, head]
                    
                    # Get pruning step if available
                    pruned_at = "N/A"
                    for i, decision in enumerate(self.pruning_decisions):
                        if [layer, head] in decision.get('details', {}).get('pruned_heads', []) or (layer, head) in decision.get('details', {}).get('pruned_heads', []):
                            pruned_at = i + 1
                            break
                    
                    html += f"""
                    <tr>
                        <td>{layer}</td>
                        <td>{head}</td>
                        <td><span class="head-status {status_class}"></span> {status}</td>
                        <td>{entropy:.4f}</td>
                        <td>{gradient:.4f}</td>
                        <td>{pruned_at}</td>
                    </tr>
                    """
        # Otherwise just show pruned heads
        elif self.pruned_heads:
            for layer, head in self.pruned_heads:
                html += f"""
                <tr>
                    <td>{layer}</td>
                    <td>{head}</td>
                    <td><span class="head-status head-pruned"></span> Pruned</td>
                    <td>N/A</td>
                    <td>N/A</td>
                    <td>N/A</td>
                </tr>
                """
        
        return html
    
    def _generate_event_timeline(self) -> str:
        """Generate the event timeline HTML."""
        if not self.event_log:
            return "<div class='timeline-item'>No events recorded yet.</div>"
        
        html = ""
        for event in reversed(self.event_log):
            html += f"""
            <div class="timeline-item">
                <div class="timeline-title">{event['title']}</div>
                <div class="timeline-time">{event['timestamp']}</div>
                <div class="timeline-content">
                    <p>{event['description']}</p>
                </div>
            </div>
            """
        
        return html
    
    def _generate_detailed_visualizations(self) -> str:
        """Generate detailed pruning visualization cards."""
        if not self.pruning_decisions or not self.entropy_values or not self.gradient_values:
            return "<p>No detailed visualization data available yet.</p>"
        
        html = ""
        # For each pruning decision
        for decision in self.pruning_decisions:
            cycle = decision['cycle']
            details = decision['details']
            
            # Generate combined visualization
            if 'pruned_heads' in details and self.entropy_values is not None and self.gradient_values is not None:
                # Create figure with 2x2 subplots
                fig, axs = plt.subplots(2, 2, figsize=(12, 10))
                
                # Plot entropy heatmap (top left)
                im1 = axs[0, 0].imshow(self.entropy_values, cmap='viridis')
                fig.colorbar(im1, ax=axs[0, 0], label='Entropy')
                axs[0, 0].set_title('Attention Head Entropy')
                axs[0, 0].set_xlabel('Head Index')
                axs[0, 0].set_ylabel('Layer Index')
                
                # Mark pruned heads with X
                for layer, head in details.get('pruned_heads', []):
                    if 0 <= layer < self.entropy_values.shape[0] and 0 <= head < self.entropy_values.shape[1]:
                        axs[0, 0].plot(head, layer, 'rx', markersize=10)
                
                # Plot gradient heatmap (top right)
                im2 = axs[0, 1].imshow(self.gradient_values, cmap='plasma')
                fig.colorbar(im2, ax=axs[0, 1], label='Gradient Norm')
                axs[0, 1].set_title('Attention Head Gradient Norms')
                axs[0, 1].set_xlabel('Head Index')
                axs[0, 1].set_ylabel('Layer Index')
                
                # Mark pruned heads with X
                for layer, head in details.get('pruned_heads', []):
                    if 0 <= layer < self.gradient_values.shape[0] and 0 <= head < self.gradient_values.shape[1]:
                        axs[0, 1].plot(head, layer, 'rx', markersize=10)
                
                # Plot combined score (bottom left)
                # Calculate combined score as weighted sum
                combined_score = self.entropy_values * 0.6 + (1.0 - self.gradient_values) * 0.4
                im3 = axs[1, 0].imshow(combined_score, cmap='YlOrRd')
                fig.colorbar(im3, ax=axs[1, 0], label='Combined Score')
                axs[1, 0].set_title('Head Importance (Gradient × Entropy)')
                axs[1, 0].set_xlabel('Head Index')
                axs[1, 0].set_ylabel('Layer Index')
                
                # Mark pruned heads with X
                for layer, head in details.get('pruned_heads', []):
                    if 0 <= layer < combined_score.shape[0] and 0 <= head < combined_score.shape[1]:
                        axs[1, 0].plot(head, layer, 'rx', markersize=10)
                
                # Plot pruning mask (bottom right)
                pruning_mask = np.zeros_like(self.entropy_values)
                for layer, head in details.get('pruned_heads', []):
                    if 0 <= layer < pruning_mask.shape[0] and 0 <= head < pruning_mask.shape[1]:
                        pruning_mask[layer, head] = 1
                
                im4 = axs[1, 1].imshow(pruning_mask, cmap='Reds')
                axs[1, 1].set_title('Pruning Decisions')
                axs[1, 1].set_xlabel('Head Index')
                axs[1, 1].set_ylabel('Layer Index')
                
                plt.tight_layout()
                
                # Save to buffer
                buf = BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                
                # Convert to base64
                img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close(fig)
                
                # Add to HTML
                html += f"""
                <div class="pruning-card">
                    <div class="pruning-header">
                        Pruning Decision <span class="pruning-cycle">Cycle {cycle}</span>
                    </div>
                    <div class="pruning-body">
                        <img class="chart-img" src="data:image/png;base64,{img_str}" alt="Pruning Decision Cycle {cycle}">
                        <div class="pruning-details">
                            <p>Detailed analysis of pruning criteria and head selection</p>
                        </div>
                    </div>
                </div>
                """
        
        return html
    
    def _generate_text_samples(self) -> str:
        """Generate text sample cards HTML."""
        if not self.text_samples:
            return "<p>No text samples generated yet.</p>"
        
        html = ""
        for sample in self.text_samples:
            prompt = sample['prompt']
            text = sample['text']
            timestamp = sample['timestamp']
            
            html += f"""
            <div class="sample-card">
                <div class="sample-prompt">{prompt}</div>
                <div class="sample-text">{text}</div>
                <div class="timestamp">Generated at: {timestamp}</div>
            </div>
            """
        
        return html

class DashboardServer:
    """
    Server to host the neural plasticity dashboard for real-time access.
    """
    
    def __init__(self, dashboard: LiveDashboard, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        """
        Initialize the dashboard server.
        
        Args:
            dashboard: LiveDashboard instance
            host: Host address to bind the server
            port: Port number to use
        """
        if not WEB_SERVER_AVAILABLE:
            raise ImportError("HTTP server components not available. Cannot start dashboard server.")
        
        self.dashboard = dashboard
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        self.running = False
    
    def start(self):
        """Start the dashboard server in a background thread."""
        if self.running:
            logger.warning("Dashboard server is already running")
            return
        
        # Create request handler
        dashboard_dir = self.dashboard.output_dir
        
        class DashboardRequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                # Set default path to index.html
                if self.path == '/':
                    self.path = '/' + self.dashboard.dashboard_name
                
                # Handle specific file paths
                try:
                    # Clean the path to prevent directory traversal
                    clean_path = os.path.normpath(self.path).lstrip('/')
                    file_path = os.path.join(dashboard_dir, clean_path)
                    
                    # Check if file exists
                    if not os.path.exists(file_path):
                        self.send_error(404, "File not found")
                        return
                    
                    # Determine content type
                    if file_path.endswith('.html'):
                        content_type = 'text/html'
                    elif file_path.endswith('.js'):
                        content_type = 'application/javascript'
                    elif file_path.endswith('.css'):
                        content_type = 'text/css'
                    elif file_path.endswith('.json'):
                        content_type = 'application/json'
                    elif file_path.endswith('.png'):
                        content_type = 'image/png'
                    elif file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
                        content_type = 'image/jpeg'
                    else:
                        content_type = 'application/octet-stream'
                    
                    # Send response
                    self.send_response(200)
                    self.send_header('Content-type', content_type)
                    self.end_headers()
                    
                    # Send file content
                    with open(file_path, 'rb') as f:
                        self.wfile.write(f.read())
                
                except Exception as e:
                    logger.error(f"Error serving file: {e}")
                    self.send_error(500, str(e))
        
        # Create and start server
        try:
            self.server = HTTPServer((self.host, self.port), DashboardRequestHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            self.running = True
            
            logger.info(f"Dashboard server started at http://{self.host}:{self.port}/")
            print(f"🔍 Dashboard available at: http://{self.host}:{self.port}/")
        except Exception as e:
            logger.error(f"Failed to start dashboard server: {e}")
            raise
    
    def stop(self):
        """Stop the dashboard server."""
        if not self.running:
            return
        
        try:
            self.server.shutdown()
            self.server.server_close()
            self.running = False
            logger.info("Dashboard server stopped")
        except Exception as e:
            logger.error(f"Error stopping dashboard server: {e}")