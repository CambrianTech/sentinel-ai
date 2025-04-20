"""
Model evaluation module for Neural Plasticity.

This module provides tools for evaluating model performance
during neural plasticity experiments.

Version: v0.0.34 (2025-04-20 14:30:00)
"""

import torch
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Evaluates models for neural plasticity experiments.
    
    This class handles:
    1. Consistent evaluation across experiment phases
    2. Proper device placement and resource management
    3. Safe calculation of metrics like perplexity
    """
    
    def __init__(
        self, 
        device: Optional[str] = None,
        max_eval_steps: int = 10
    ):
        """
        Initialize the model evaluator.
        
        Args:
            device: Device to run evaluation on
            max_eval_steps: Maximum number of steps to evaluate
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_eval_steps = max_eval_steps
        
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model on provided dataloader.
        
        Args:
            model: The model to evaluate
            dataloader: DataLoader for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        
        # Try to move model to the selected device
        try:
            model = model.to(self.device)
        except RuntimeError as e:
            logger.warning(f"Failed to move model to {self.device}: {e}")
            # Try to continue with the model's current device
            self.device = next(model.parameters()).device
        
        device = self.device
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Break if we've evaluated enough steps
                if batch_idx >= self.max_eval_steps:
                    break
                    
                # Move batch to device
                if isinstance(batch, dict):
                    inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                else:
                    # Assume batch is a tuple/list with input_ids first
                    input_ids = batch[0].to(device)
                    inputs = {"input_ids": input_ids}
                    if len(batch) > 1:
                        inputs["attention_mask"] = batch[1].to(device)
                
                # Forward pass
                outputs = model(**inputs)
                
                # Calculate loss
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs
                    
                # For causal language modeling, shift labels
                labels = inputs["input_ids"].clone()
                labels = labels[:, 1:].contiguous()
                shift_logits = logits[:, :-1, :].contiguous()
                
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
                losses.append(loss.item())
        
        # Calculate metrics
        avg_loss = sum(losses) / len(losses) if losses else float("inf")
        
        # Calculate perplexity directly from loss
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # Add a simple sanity check warning if perplexity seems unreasonable
        if perplexity > 10000:
            logger.warning(f"Unusually high perplexity: {perplexity:.1f}. This may indicate a data quality issue.")
            
        return {
            "loss": avg_loss,
            "perplexity": perplexity
        }