"""Metrics logger - DEPRECATED MODULE

This module is a backwards compatibility layer for the old utils.metrics_logger module.
The functionality has been moved to sentinel.utils.metrics_logger.

Please update your imports to use the new module path.
"""

import warnings
import json
import os
from datetime import datetime

# Emit deprecation warning
warnings.warn(
    "The module utils.metrics_logger is deprecated. "
    "Please use sentinel.utils.metrics_logger instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location
from sentinel.utils.metrics_logger import MetricsLogger as _NewMetricsLogger


# Create a subclass that maintains backward compatibility
class MetricsLogger(_NewMetricsLogger):
    """Legacy MetricsLogger with backward compatibility"""
    
    def __init__(self, log_file="training.log"):
        super().__init__(log_file)
        
        # For backward compatibility with original implementation
        import logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create file handler and set level to info
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
    
    # Legacy methods
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


# Add all imported symbols to __all__
__all__ = ["MetricsLogger"]