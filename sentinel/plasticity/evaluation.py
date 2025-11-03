"""
Model Evaluation Module for Neural Plasticity

This module provides evaluation utilities for transformer models,
calculating metrics like perplexity and loss.

Version: v0.0.34 (2025-04-20 17:00:00)
"""

import torch
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Evaluates transformer model performance on language modeling tasks.
    
    This class provides:
    1. Perplexity calculation
    2. Loss evaluation
    3. Batch-wise evaluation with proper handling of different model types
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the model evaluator.
        
        Args:
            device: Device to run evaluation on (autodetected if None)
        """
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
    def evaluate(
        self,
        model,
        eval_dataloader,
        max_eval_batches: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on the provided dataloader.
        
        Args:
            model: Model to evaluate
            eval_dataloader: DataLoader for evaluation data
            max_eval_batches: Maximum number of batches to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Temporarily set model to evaluation mode
        model.eval()
        was_training = model.training
        
        # Evaluation metrics
        total_loss = 0.0
        total_tokens = 0
        
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
                    # Check if we've reached max_eval_batches
                    if max_eval_batches is not None and batch_idx >= max_eval_batches:
                        break
                        
                    # Move batch to device
                    batch = {k: v.to(model.device) if hasattr(v, 'to') else v 
                             for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = model(**batch)
                    
                    # Get loss and accumulate (handle case where loss might be None)
                    loss = outputs.loss
                    
                    # Skip if no loss
                    if loss is None:
                        logger.warning("Model returned None for loss, skipping batch")
                        continue
                    
                    # Get batch size and sequence length
                    if 'input_ids' in batch:
                        batch_tokens = batch['input_ids'].numel()
                    else:
                        # Default to assuming a standard batch shape [batch_size, seq_len]
                        batch_tokens = 1
                        for dim in batch['labels'].shape:
                            batch_tokens *= dim
                            
                    # Accumulate loss and token count
                    total_loss += loss.item() * batch_tokens
                    total_tokens += batch_tokens
                    
            # Calculate averages
            avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
            
            # Calculate perplexity
            perplexity = float(np.exp(avg_loss))
            
            # Return metrics
            metrics = {
                'loss': avg_loss,
                'perplexity': perplexity,
                'total_tokens': total_tokens
            }
            
            logger.debug(f"Evaluation results: Loss = {avg_loss:.4f}, Perplexity = {perplexity:.2f}")
            
            return metrics
            
        finally:
            # Restore model's previous training state
            if was_training:
                model.train()
                
    def calculate_perplexity(self, loss: float) -> float:
        """
        Calculate perplexity from loss.
        
        Args:
            loss: Loss value
            
        Returns:
            Perplexity
        """
        return float(np.exp(loss))