"""
Metrics utilities for calculating and logging model metrics.

This module provides functions for calculating and logging metrics during
model training and evaluation, to track progress and identify issues.
"""

import torch
import numpy as np
import re
import math
from collections import Counter
from typing import Dict, Any, List, Optional

def calculate_metrics(logits: torch.Tensor, labels: torch.Tensor, prefix: str = "") -> Dict[str, float]:
    """
    Calculate metrics from model outputs and labels.
    
    Args:
        logits: Model logits of shape [batch_size, sequence_length, vocab_size]
                or [batch_size, vocab_size]
        labels: Ground truth labels of shape [batch_size, sequence_length]
                or [batch_size]
        prefix: Prefix to add to metric names (e.g., "train/" or "val/")
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # For sequence tasks, reshape if needed
    if len(logits.shape) == 3:
        # [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
        flat_logits = logits.reshape(-1, logits.size(-1))
        # [batch_size, seq_len] -> [batch_size * seq_len]
        flat_labels = labels.reshape(-1)
    else:
        # Already [batch_size, vocab_size] and [batch_size]
        flat_logits = logits
        flat_labels = labels
    
    # Calculate cross-entropy loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fct(flat_logits, flat_labels)
    metrics[f"{prefix}loss"] = loss.item()
    
    # Calculate perplexity
    perplexity = torch.exp(loss)
    metrics[f"{prefix}perplexity"] = perplexity.item()
    
    # Calculate accuracy
    predictions = torch.argmax(flat_logits, dim=-1)
    correct = (predictions == flat_labels).float().sum()
    accuracy = correct / flat_labels.numel()
    metrics[f"{prefix}accuracy"] = accuracy.item()
    
    return metrics


def log_metrics(metrics: Dict[str, float], step: int, logger=None) -> None:
    """
    Log metrics to console and optionally to a logger.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Current training step
        logger: Optional logger to use
    """
    # Log to console
    metrics_str = f"Step {step}: "
    metrics_str += ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
    print(metrics_str)
    
    # Log to logger if provided
    if logger is not None:
        # Support different logger interfaces
        if hasattr(logger, "log_metrics"):
            # MetricsLogger
            logger.log_metrics(step=step, **metrics)
        elif hasattr(logger, "log"):
            # Generic logger with log method
            logger.log({"step": step, **metrics})
        elif hasattr(logger, "add_scalars"):
            # TensorBoard
            for key, value in metrics.items():
                logger.add_scalar(key, value, step)


def calculate_perplexity(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate perplexity for language modeling.
    
    Args:
        logits: Logits from model of shape [batch_size, sequence_length, vocab_size]
        labels: Labels of shape [batch_size, sequence_length]
        
    Returns:
        Perplexity value
    """
    # Reshape if needed
    if len(logits.shape) == 3:
        # [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
        flat_logits = logits.reshape(-1, logits.size(-1))
        # [batch_size, seq_len] -> [batch_size * seq_len]
        flat_labels = labels.reshape(-1)
    else:
        flat_logits = logits
        flat_labels = labels
    
    # Calculate cross-entropy loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fct(flat_logits, flat_labels)
    
    # Calculate perplexity
    perplexity = torch.exp(loss).item()
    
    return perplexity


def calculate_sequence_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Calculate metrics for sequence tasks with optional attention mask.
    
    Args:
        logits: Logits from model of shape [batch_size, sequence_length, vocab_size]
        labels: Labels of shape [batch_size, sequence_length]
        attention_mask: Optional mask of shape [batch_size, sequence_length]
        
    Returns:
        Dictionary with metrics
    """
    # Get device for computation
    device = logits.device
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Reshape to [batch_size * seq_len, vocab_size] and [batch_size * seq_len]
    batch_size, seq_len, vocab_size = logits.shape
    flat_logits = logits.reshape(-1, vocab_size)
    flat_labels = labels.reshape(-1)
    
    # Apply attention mask if provided
    if attention_mask is not None:
        flat_mask = attention_mask.reshape(-1)
        # Only consider positions where mask is 1
        mask_indices = flat_mask.nonzero().squeeze()
        flat_logits = flat_logits[mask_indices]
        flat_labels = flat_labels[mask_indices]
    
    # Calculate loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fct(flat_logits, flat_labels)
    metrics["loss"] = loss.item()
    
    # Calculate perplexity
    perplexity = torch.exp(loss)
    metrics["perplexity"] = perplexity.item()
    
    # Calculate accuracy
    predictions = torch.argmax(flat_logits, dim=-1)
    correct = (predictions == flat_labels).float().sum()
    accuracy = correct / flat_labels.numel()
    metrics["accuracy"] = accuracy.item()
    
    return metrics


def calculate_perplexity(model, tokenizer, text):
    """
    Calculate an approximate perplexity measure for a generated text.
    
    This is a simplified version for validation that doesn't require running the model.
    
    Args:
        model: The model (used for reference only in this simplified version)
        tokenizer: The tokenizer
        text: The text to evaluate
        
    Returns:
        Approximate perplexity score
    """
    if not text:
        return 100.0  # Default high perplexity for empty text
    
    # Simple tokenization (split by whitespace and punctuation)
    tokens = re.findall(r'\w+|[^\w\s]', text.lower())
    
    if not tokens:
        return 100.0
    
    # Calculate token frequencies
    token_counts = Counter(tokens)
    total_tokens = len(tokens)
    
    # Calculate entropy
    entropy = 0
    for token, count in token_counts.items():
        prob = count / total_tokens
        entropy -= prob * math.log(prob)
    
    # Convert entropy to perplexity
    perplexity = math.exp(entropy)
    
    return perplexity


def calculate_diversity(text):
    """
    Calculate lexical diversity of a text.
    
    Args:
        text: The text to evaluate
        
    Returns:
        A diversity score (higher is better)
    """
    if not text:
        return 0.0
    
    # Simple word tokenization
    words = re.findall(r'\w+', text.lower())
    
    if not words:
        return 0.0
    
    # Calculate lexical diversity as unique words / total words
    unique_words = len(set(words))
    total_words = len(words)
    
    return unique_words / total_words


def calculate_repetition(text):
    """
    Calculate repetition score for a text.
    
    Args:
        text: The text to evaluate
        
    Returns:
        A repetition score (lower is better)
    """
    if not text:
        return 0.0
    
    words = re.findall(r'\w+', text.lower())
    
    if len(words) <= 1:
        return 0.0
    
    # Find repeated words in a window
    window_size = min(50, len(words))
    repeats = 0
    
    for i in range(len(words) - 1):
        end_idx = min(i + window_size, len(words))
        if words[i] in words[i+1:end_idx]:
            repeats += 1
            
    return repeats / (len(words) - 1)