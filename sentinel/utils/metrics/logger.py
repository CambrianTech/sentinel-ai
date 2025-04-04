"""
Metrics logger for neural plasticity experiments.

This module provides metrics logging functionality with two implementations:
1. Traditional logging using Python's logging module
2. JSON-based structured logging for experiment tracking
"""

import os
import json
import logging
from datetime import datetime


class MetricsLogger:
    """
    Logger that supports both traditional logging and structured JSON logging.
    
    The original implementation using Python's logging module is maintained for
    backward compatibility, while new methods for structured JSON logging are added
    for neural plasticity experiments.
    """
    
    def __init__(self, log_file="training.log"):
        # Traditional logging setup
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create file handler and set level to info
        self.log_file = log_file
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Create console handler with a higher level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # JSON logging setup
        self.json_log_file = None
        if log_file.endswith(".log"):
            self.json_log_file = log_file.replace(".log", ".jsonl")
        elif log_file.endswith(".jsonl"):
            self.json_log_file = log_file
        else:
            self.json_log_file = log_file + ".jsonl"
        
        # Create directory for JSON log file if it doesn't exist
        if self.json_log_file:
            os.makedirs(os.path.dirname(os.path.abspath(self.json_log_file)), exist_ok=True)

    # Original logging methods
    
    def log_metrics(self, epoch, step, train_loss, val_loss, perplexity, active_head_count, param_count):
        log_message = (
            f"Epoch: {epoch}, Step: {step}, "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Perplexity: {perplexity:.4f}, Active Heads: {active_head_count}, "
            f"Parameter Count: {param_count}"
        )
        self.logger.info(log_message)
        
        # Also log as structured data
        self.log({
            "phase": "training",
            "epoch": epoch,
            "step": step,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "perplexity": float(perplexity),
            "active_head_count": active_head_count,
            "param_count": param_count
        })

    def log_eval(self, val_loss, perplexity, baseline_ppl):
        log_message = (
            f"Validation Loss: {val_loss:.4f}, Perplexity: {perplexity:.4f}, "
            f"Baseline Perplexity: {baseline_ppl:.4f}"
        )
        self.logger.info(log_message)
        
        # Also log as structured data
        self.log({
            "phase": "evaluation",
            "val_loss": float(val_loss),
            "perplexity": float(perplexity),
            "baseline_perplexity": float(baseline_ppl)
        })

    def log_train(self, train_loss):
        log_message = f"Training Loss: {train_loss:.4f}"
        self.logger.info(log_message)
        
        # Also log as structured data
        self.log({
            "phase": "training_step",
            "train_loss": float(train_loss)
        })

    def log_active_heads(self, active_head_count):
        log_message = f"Active Heads: {active_head_count}"
        self.logger.info(log_message)
        
        # Also log as structured data
        self.log({
            "phase": "architecture",
            "active_head_count": active_head_count
        })
    
    # New structured logging methods
    
    def log(self, metrics_dict):
        """
        Log metrics to the JSON log file.
        
        Args:
            metrics_dict: Dictionary of metrics to log
        """
        # Add timestamp if not present
        if "timestamp" not in metrics_dict:
            metrics_dict["timestamp"] = datetime.now().isoformat()
        
        # Write to JSON log file if specified
        if self.json_log_file is not None:
            try:
                with open(self.json_log_file, 'a') as f:
                    f.write(json.dumps(metrics_dict) + '\n')
            except Exception as e:
                self.logger.error(f"Error writing to JSON log file: {e}")
    
    def load_metrics(self):
        """
        Load metrics from the JSON log file.
        
        Returns:
            List of metrics dictionaries
        """
        if self.json_log_file is None or not os.path.exists(self.json_log_file):
            return []
        
        metrics = []
        with open(self.json_log_file, 'r') as f:
            for line in f:
                try:
                    metrics.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    self.logger.warning(f"Could not parse line as JSON: {line}")
        
        return metrics
    
    def get_metrics_by_phase(self, phase):
        """
        Get metrics for a specific phase.
        
        Args:
            phase: Phase to filter by
            
        Returns:
            List of metrics dictionaries for the specified phase
        """
        metrics = self.load_metrics()
        return [m for m in metrics if m.get("phase") == phase]
    
    def get_latest_metrics(self, n=1):
        """
        Get the latest n metrics entries.
        
        Args:
            n: Number of entries to return
            
        Returns:
            List of the latest n metrics dictionaries
        """
        metrics = self.load_metrics()
        return metrics[-n:] if metrics else []
    
    def summarize_metrics(self):
        """
        Generate a summary of the logged metrics.
        
        Returns:
            Dictionary with metrics summary
        """
        metrics = self.load_metrics()
        if not metrics:
            return {}
        
        # Group metrics by phase
        phases = {}
        for m in metrics:
            phase = m.get("phase", "unknown")
            if phase not in phases:
                phases[phase] = []
            phases[phase].append(m)
        
        # Create summary
        summary = {
            "total_entries": len(metrics),
            "phases": {phase: len(entries) for phase, entries in phases.items()},
            "first_timestamp": metrics[0].get("timestamp"),
            "last_timestamp": metrics[-1].get("timestamp")
        }
        
        return summary
