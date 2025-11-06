"""
Neural Plasticity Dashboard Reporter

This module provides a reporter class that connects to the neural plasticity
experiment and updates the live dashboard with experiment progress.

Version: v0.0.2 (2025-04-20 17:40:00)
"""

import os
import time
import threading
from typing import Dict, List, Optional, Any, Union, Callable
import logging

import numpy as np

from .dashboard import LiveDashboard, DashboardServer

logger = logging.getLogger(__name__)

class DashboardReporter:
    """
    Reporter that connects to the neural plasticity experiment and updates
    the dashboard with real-time experiment progress.
    """
    
    def __init__(
        self,
        output_dir: str,
        dashboard_name: str = "dashboard.html",
        auto_update: bool = True,
        update_interval: int = 5,
        start_server: bool = True,
        host: str = "localhost",
        port: int = 8080
    ):
        """
        Initialize the dashboard reporter.
        
        Args:
            output_dir: Directory to save dashboard files
            dashboard_name: Name of the dashboard HTML file
            auto_update: Whether to automatically update the dashboard
            update_interval: How often to update the dashboard (in seconds)
            start_server: Whether to start a web server for the dashboard
            host: Host address for the web server
            port: Port number for the web server
        """
        self.output_dir = output_dir
        self.auto_update = auto_update
        self.update_interval = update_interval
        
        # Create dashboard directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create dashboard
        try:
            self.dashboard = LiveDashboard(
                output_dir=output_dir,
                dashboard_name=dashboard_name,
                auto_refresh=auto_update,
                refresh_interval=update_interval
            )
            logger.info(f"Dashboard initialized at {os.path.join(output_dir, dashboard_name)}")
        except Exception as e:
            logger.error(f"Error initializing dashboard: {e}")
            # Create a simple fallback dashboard file
            self._create_simple_dashboard(output_dir, dashboard_name)
    
    def _create_simple_dashboard(self, output_dir, dashboard_name):
        """Create a simple fallback dashboard if the full one fails."""
        os.makedirs(output_dir, exist_ok=True)
        simple_html = """<!DOCTYPE html>
<html>
<head>
    <title>Neural Plasticity Dashboard</title>
    <meta http-equiv="refresh" content="5" />
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { font-size: 24px; margin-bottom: 20px; }
        .card { border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="header">Neural Plasticity Experiment Dashboard</div>
    <div class="card">
        <p>Experiment is running. Data will appear here as it becomes available.</p>
        <p>Last update: {timestamp}</p>
    </div>
</body>
</html>
""".format(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
        
        try:
            with open(os.path.join(output_dir, dashboard_name), 'w') as f:
                f.write(simple_html)
            logger.info(f"Created simple fallback dashboard at {os.path.join(output_dir, dashboard_name)}")
        except Exception as e:
            logger.error(f"Failed to create fallback dashboard: {e}")
        
        # Start server if requested
        self.server = None
        if start_server:
            try:
                try:
                    self.server = DashboardServer(
                        dashboard=self.dashboard,
                        host=host,
                        port=port
                    )
                    self.server.start()
                    logger.info(f"Dashboard server started at http://{host}:{port}/")
                except Exception as e:
                    logger.error(f"Failed to start dashboard server: {e}")
                    logger.info("Dashboard will be saved to file only.")
            except Exception as e:
                logger.error(f"Failed to start dashboard server: {e}")
                logger.info("Dashboard will be saved to file only.")
        
        # Setup auto-update thread
        self.update_thread = None
        self.running = False
        
        if auto_update:
            self.start_auto_update()
    
    def start_auto_update(self):
        """Start the auto-update thread."""
        if self.running:
            return
        
        self.running = True
        self.update_thread = threading.Thread(target=self._auto_update_thread)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def stop_auto_update(self):
        """Stop the auto-update thread."""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=self.update_interval * 2)
    
    def _auto_update_thread(self):
        """Background thread for auto-updating the dashboard."""
        while self.running:
            self.update_dashboard()
            time.sleep(self.update_interval)
    
    def update_dashboard(self):
        """Force an update of the dashboard."""
        # The dashboard will be updated automatically by the LiveDashboard
        # when any of its methods are called with new data.
        # This method is mainly a placeholder for any future logic.
        pass
    
    def set_status(self, status: str, details: str = ""):
        """
        Update the experiment status.
        
        Args:
            status: Status string (initializing, running, completed, error)
            details: Additional status details
        """
        self.dashboard.set_status(status, details)
    
    def set_phase(self, phase: str):
        """
        Update the current experiment phase.
        
        Args:
            phase: Current phase (setup, warmup, analysis, pruning, evaluation, complete)
        """
        self.dashboard.set_phase(phase)
    
    def update_config(self, config: Dict[str, Any]):
        """
        Update experiment configuration information.
        
        Args:
            config: Dictionary of configuration parameters
        """
        self.dashboard.update_config(config)
    
    def update_metrics(self, metrics: Dict[str, Union[float, int, List[float]]], step: Optional[int] = None):
        """
        Update the metrics data for the dashboard.
        
        Args:
            metrics: Dictionary of metrics to update
            step: Current step number (optional)
        """
        # Add step if provided
        if step is not None and 'step' not in metrics:
            metrics['step'] = step
        
        self.dashboard.update_metrics(metrics)
    
    def update_entropy(self, entropy_values: np.ndarray):
        """
        Update entropy values for attention heads.
        
        Args:
            entropy_values: Tensor or array of entropy values [layers, heads]
        """
        self.dashboard.update_entropy(entropy_values)
    
    def update_gradients(self, gradient_values: np.ndarray):
        """
        Update gradient values for attention heads.
        
        Args:
            gradient_values: Tensor or array of gradient values [layers, heads]
        """
        self.dashboard.update_gradients(gradient_values)
    
    def add_pruning_decision(self, decision: Dict[str, Any]):
        """
        Add a new pruning decision event.
        
        Args:
            decision: Dictionary with pruning decision details
        """
        self.dashboard.add_pruning_decision(decision)
    
    def add_text_sample(self, prompt: str, generated_text: str):
        """
        Add a text generation sample.
        
        Args:
            prompt: Text prompt used for generation
            generated_text: Text generated by the model
        """
        self.dashboard.add_text_sample(prompt, generated_text)
    
    def get_metrics_callback(self) -> Callable:
        """
        Get a callback function for adding metrics during training.
        
        Returns:
            Function that can be used as a callback in training
        """
        def metrics_callback(step, metrics):
            self.update_metrics(metrics, step)
        
        return metrics_callback
    
    def get_sample_callback(self) -> Callable:
        """
        Get a callback function for adding text samples during training.
        
        Returns:
            Function that can be used as a sample_callback in training
        """
        def sample_callback(step, sample_data):
            if 'input_text' in sample_data and 'predicted_tokens' in sample_data:
                # Format the sample for display
                prompt = sample_data.get('input_text', '')
                predicted_tokens = sample_data.get('predicted_tokens', [])
                predicted_text = ' '.join(predicted_tokens)
                
                self.add_text_sample(prompt, predicted_text)
        
        return sample_callback
    
    def close(self):
        """Clean up resources and stop server."""
        self.stop_auto_update()
        
        if self.server:
            try:
                self.server.stop()
            except Exception as e:
                logger.error(f"Error stopping dashboard server: {e}")