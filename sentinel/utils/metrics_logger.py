"""
Metrics logger for tracking and saving various metrics during training and evaluation.

This module provides a simple metrics logging facility that supports
appending metrics to JSONL files and summarizing metric statistics.
"""

import json
import os
import time
from typing import Dict, Any, List, Optional
from datetime import datetime


class MetricsLogger:
    """
    Logger for recording metrics during model training and evaluation.
    Supports appending to JSONL files and providing metric summaries.
    """
    
    def __init__(self, log_file: str, buffer_size: int = 10):
        """
        Initialize the metrics logger.
        
        Args:
            log_file: Path to the log file (JSONL format)
            buffer_size: Number of entries to buffer before writing to disk
        """
        self.log_file = log_file
        self.buffer_size = buffer_size
        self.buffer = []
        
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    def log(self, metrics: Dict[str, Any]) -> None:
        """
        Log metrics to the buffer and flush if necessary.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        # Add timestamp if not present
        if "timestamp" not in metrics:
            metrics["timestamp"] = datetime.now().isoformat()
        
        # Add to buffer
        self.buffer.append(metrics)
        
        # Flush if buffer is full
        if len(self.buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self) -> None:
        """Write buffered metrics to disk."""
        if not self.buffer:
            return
        
        # Open file in append mode
        with open(self.log_file, "a") as f:
            for metrics in self.buffer:
                f.write(json.dumps(metrics) + "\n")
        
        # Clear buffer
        self.buffer = []
    
    def get_metrics(self, phase: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Read all metrics from the log file.
        
        Args:
            phase: Optional filter for a specific phase
            
        Returns:
            List of metric dictionaries
        """
        # Flush any pending metrics
        self.flush()
        
        # Read all metrics from file
        metrics = []
        
        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    if line.strip():
                        metric_dict = json.loads(line)
                        if phase is None or metric_dict.get("phase") == phase:
                            metrics.append(metric_dict)
        except FileNotFoundError:
            # File doesn't exist yet
            pass
        
        return metrics
    
    def get_latest(self, phase: Optional[str] = None, count: int = 1) -> List[Dict[str, Any]]:
        """
        Get the most recent metrics.
        
        Args:
            phase: Optional filter for a specific phase
            count: Number of recent entries to return
            
        Returns:
            List of most recent metric dictionaries
        """
        metrics = self.get_metrics(phase)
        
        # Sort by timestamp if available
        metrics.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return metrics[:count]
    
    def summarize(self, phase: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute summary statistics for numerical metrics.
        
        Args:
            phase: Optional filter for a specific phase
            
        Returns:
            Dictionary with mean, min, max for each numerical metric
        """
        metrics = self.get_metrics(phase)
        
        if not metrics:
            return {}
        
        # Collect all numerical metrics
        numerical_metrics = {}
        
        for metric_dict in metrics:
            for key, value in metric_dict.items():
                # Skip non-numerical values and metadata fields
                if key in ["phase", "timestamp", "description", "samples", "results"]:
                    continue
                
                try:
                    # Try to convert to float
                    float_value = float(value)
                    
                    if key not in numerical_metrics:
                        numerical_metrics[key] = []
                    
                    numerical_metrics[key].append(float_value)
                except (ValueError, TypeError):
                    # Not a numerical value
                    pass
        
        # Compute summary statistics
        summary = {}
        
        for key, values in numerical_metrics.items():
            summary[key] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }
        
        return summary
    
    def __del__(self):
        """Ensure all metrics are flushed when the logger is destroyed."""
        try:
            self.flush()
        except:
            # Ignore errors during cleanup
            pass