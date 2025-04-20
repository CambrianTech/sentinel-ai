"""
Entropy-based Pruning Strategy

This module provides an implementation of entropy-based pruning for
attention heads in transformer models.

Version: v0.0.34 (2025-04-20 17:00:00)
"""

import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm

from .base import PruningStrategy

logger = logging.getLogger(__name__)

class EntropyPruner(PruningStrategy):
    """
    Prunes attention heads based on attention entropy.
    
    This class implements entropy-based pruning by:
    1. Calculating the entropy of attention distributions
    2. Ranking heads by their entropy values
    3. Pruning heads with lowest entropy (most focused attention)
    """
    
    def __init__(
        self,
        device: Optional[str] = None, 
        batch_size: int = 4,
        epsilon: float = 1e-10
    ):
        """
        Initialize the entropy pruner.
        
        Args:
            device: Device to run on
            batch_size: Batch size for processing
            epsilon: Small value for numerical stability
        """
        super().__init__(device)
        self.batch_size = batch_size
        self.epsilon = epsilon
        
    def collect_distributions(
        self,
        model,
        dataloader
    ) -> Dict[int, torch.Tensor]:
        """
        Collect attention distributions and calculate entropy.
        
        Args:
            model: The model to analyze
            dataloader: DataLoader for input data
            
        Returns:
            Dictionary mapping layer indices to entropy values
        """
        logger.info("Collecting attention distributions for entropy calculation")
        
        # Detect model structure
        num_layers, num_heads = self.detect_model_structure(model)
        
        # Store entropy values for each layer and head
        # Entropy tensor shape: [num_layers, num_heads]
        entropy = torch.zeros((num_layers, num_heads), device=self.device)
        
        # Counter for normalization
        sample_count = 0
        
        # Set model to eval mode
        model.eval()
        was_training = model.training
        
        try:
            # Register hooks to capture attention patterns
            attention_patterns = {}
            handles = []
            
            # Define hook to collect attention patterns
            def attention_hook(module, input, output, layer_idx):
                nonlocal attention_patterns
                
                # Get attention weights
                if isinstance(output, tuple):
                    # Most models return attention weights as part of output tuple
                    for item in output:
                        if isinstance(item, torch.Tensor) and len(item.shape) == 4:
                            # Found attention weights [batch, heads, seq_len, seq_len]
                            attention_patterns[layer_idx] = item
                            break
                else:
                    # Some models directly return attention weights
                    attention_patterns[layer_idx] = output
            
            # Register hooks for each layer
            for i in range(num_layers):
                # Try to find attention modules
                for name, module in model.named_modules():
                    if ('attention' in name.lower() or 'attn' in name.lower()) and f"layer.{i}" in name:
                        # Create a closure to capture the correct layer index
                        hook = lambda module, input, output, i=i: attention_hook(module, input, output, i)
                        handle = module.register_forward_hook(hook)
                        handles.append(handle)
                        break
            
            # Process batches
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(dataloader, desc="Calculating entropy")):
                    # Move batch to device
                    batch = {k: v.to(model.device) if hasattr(v, 'to') else v 
                             for k, v in batch.items()}
                    
                    # Clear patterns for new batch
                    attention_patterns.clear()
                    
                    # Forward pass to collect attention patterns
                    _ = model(**batch)
                    
                    # Process collected patterns
                    for layer_idx, attention in attention_patterns.items():
                        # Skip if layer index is out of bounds
                        if layer_idx >= num_layers:
                            continue
                            
                        # Get attention tensor [batch, heads, seq_len, seq_len]
                        if attention is None or not isinstance(attention, torch.Tensor):
                            continue
                            
                        # Extract batched head attentions [batch, heads, seq_len, seq_len]
                        batch_attention = attention
                        
                        # Calculate entropy for each head in the batch
                        for head_idx in range(min(num_heads, batch_attention.shape[1])):
                            # Get head attention [batch, seq_len, seq_len]
                            head_attention = batch_attention[:, head_idx]
                            
                            # Calculate entropy for this head
                            head_entropy = self._calculate_entropy(head_attention)
                            
                            # Accumulate entropy
                            entropy[layer_idx, head_idx] += head_entropy.sum()
                    
                    # Increment sample count
                    sample_count += batch_attention.shape[0]
                    
                    # Process limited batches for efficiency
                    if batch_idx >= 10:  # Limit to 10 batches for efficiency
                        break
            
            # Remove hooks
            for handle in handles:
                handle.remove()
                
            # Normalize entropy by sample count
            if sample_count > 0:
                entropy /= sample_count
                
            # Convert to dictionary
            entropy_dict = {i: entropy[i] for i in range(num_layers)}
            
            logger.info("Completed attention entropy calculation")
            
            return entropy_dict
            
        finally:
            # Restore model's previous training state
            if was_training:
                model.train()
                
    def _calculate_entropy(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Calculate entropy of attention distributions.
        
        Args:
            attention: Attention weights tensor [batch, seq_len, seq_len]
            
        Returns:
            Entropy values tensor [batch]
        """
        # Ensure valid probability distribution
        # Add small epsilon to avoid log(0)
        probs = attention + self.epsilon
        
        # Normalize to ensure sum is 1
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs), dim=(-2, -1))
        
        return entropy
        
    def prune(
        self,
        model,
        prune_percent: float = 0.2,
        distributions: Optional[Dict[int, torch.Tensor]] = None,
        **kwargs
    ) -> List[Tuple[int, int, float]]:
        """
        Prune heads with lowest entropy values.
        
        Args:
            model: The model to prune
            prune_percent: Percentage of heads to prune (0.0 to 1.0)
            distributions: Pre-computed entropy values (optional)
            
        Returns:
            List of (layer_idx, head_idx, entropy) for pruned heads
        """
        logger.info(f"Starting entropy-based pruning with prune_percent={prune_percent}")
        
        # Collect entropy if not provided
        if distributions is None:
            raise ValueError("Entropy distributions must be provided for entropy pruning")
            
        # Detect model structure
        num_layers, num_heads = self.detect_model_structure(model)
        
        # Calculate total number of heads
        total_heads = num_layers * num_heads
        
        # Calculate number of heads to prune
        num_to_prune = int(total_heads * prune_percent)
        
        if num_to_prune <= 0:
            logger.warning(f"No heads to prune with prune_percent={prune_percent}")
            return []
            
        # Create flat list of (layer, head, entropy) tuples
        head_entropies = []
        for layer_idx, layer_entropy in distributions.items():
            for head_idx, entropy in enumerate(layer_entropy):
                if isinstance(entropy, torch.Tensor):
                    entropy_value = entropy.item()
                else:
                    entropy_value = float(entropy)
                    
                head_entropies.append((int(layer_idx), int(head_idx), entropy_value))
                
        # Skip if no valid entropies
        if not head_entropies:
            logger.warning("No valid entropy values found")
            return []
            
        # Sort by entropy (lowest first)
        head_entropies.sort(key=lambda x: x[2])
        
        # Select the heads to prune (lowest entropy)
        pruned_heads = head_entropies[:num_to_prune]
        
        logger.info(f"Selected {len(pruned_heads)} heads for pruning")
        
        # Apply pruning mask to the model
        self.apply_pruning_mask(model, pruned_heads)
        
        return pruned_heads