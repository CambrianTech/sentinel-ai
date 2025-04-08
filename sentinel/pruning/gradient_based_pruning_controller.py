"""
Gradient-Based Pruning Controller for Dynamic Pruning

This module implements a controller for real-time neural plasticity that dynamically
prunes attention heads based on gradient-based utility metrics during training.
This is a simplified version of PlasticityController that only uses gradient information
when entropy calculation doesn't work reliably.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional

from sentinel.pruning.dual_mode_pruning import prune_head_in_model, apply_pruning_hooks, PruningMode, get_model_info


class GradientPruningController:
    """
    Controller for dynamic head pruning based purely on gradient information.
    
    This controller tracks head gradient metrics over time and applies
    pruning decisions based on configurable percentiles. Heads with the lowest
    gradient norms are pruned periodically.
    """
    
    def __init__(self, 
                 model: torch.nn.Module, 
                 total_layers: int, 
                 heads_per_layer: int,
                 history_length: int = 10,
                 prune_percent: float = 0.1,
                 gradient_percentile: float = 30,
                 min_zero_epochs: int = 1,
                 mode: PruningMode = PruningMode.ADAPTIVE,
                 max_prune_percent: float = 0.5):
        """
        Initialize the gradient-based pruning controller.
        
        Args:
            model: The transformer model
            total_layers: Total number of layers in the model
            heads_per_layer: Number of attention heads per layer
            history_length: Maximum number of history entries to keep
            prune_percent: Percentage of heads to prune in each step (0-1)
            gradient_percentile: Percentile threshold for gradients (only used if percentile_based=True)
            min_zero_epochs: Minimum number of epochs a head should remain pruned
            mode: Pruning mode (ADAPTIVE or COMPRESSED)
            max_prune_percent: Maximum percentage of heads that can be pruned in total
        """
        self.model = model
        self.total_layers = total_layers
        self.heads_per_layer = heads_per_layer
        self.history_length = history_length
        self.prune_percent = prune_percent
        self.gradient_percentile = gradient_percentile
        self.min_zero_epochs = min_zero_epochs
        self.mode = mode
        self.max_prune_percent = max_prune_percent
        
        # Initialize statistics tracking for each head
        self.stats = defaultdict(lambda: defaultdict(lambda: {
            'grad_norm': [],
            'zeroed_epochs': 0,
            'is_zeroed': False,
            'pruned_at_step': -1
        }))
        
        # Store hooks for pruning
        self.hooks = []
        
        # Epoch counter
        self.epoch = 0
        self.step = 0
    
    def update_stats(self, layer_idx: int, head_idx: int, grad_norm: float):
        """
        Update statistics for a specific attention head.
        
        Args:
            layer_idx: Layer index
            head_idx: Head index
            grad_norm: Current gradient norm
        """
        # Get stats for this head
        s = self.stats[layer_idx][head_idx]
        
        # Add new values, maintaining history limit
        s['grad_norm'].append(grad_norm)
        if len(s['grad_norm']) > self.history_length:
            s['grad_norm'] = s['grad_norm'][-self.history_length:]
    
    def collect_grad_metrics(self, dataloader, num_batches=1):
        """
        Collect gradient metrics for all heads.
        
        Args:
            dataloader: DataLoader for input data
            num_batches: Number of batches to process
            
        Returns:
            grad_norm_values as 2D tensor [layer, head]
        """
        # Initialize tensor to store metrics
        grad_norm_values = torch.zeros((self.total_layers, self.heads_per_layer), 
                                      device=next(self.model.parameters()).device)
        
        # Capture gradients during backward pass
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0)  # Dummy optimizer, lr=0 to avoid updates
        
        # Run forward-backward pass
        batch_iterator = iter(dataloader)
        for _ in range(min(num_batches, len(dataloader))):
            try:
                batch = next(batch_iterator)
                
                # Prepare input
                if isinstance(batch, dict):
                    inputs = {k: v.to(next(self.model.parameters()).device) 
                             for k, v in batch.items() if isinstance(v, torch.Tensor)}
                else:
                    # Handle tuple/list batch types
                    inputs = {"input_ids": batch[0].to(next(self.model.parameters()).device)}
                    if len(batch) > 1:
                        inputs["attention_mask"] = batch[1].to(next(self.model.parameters()).device)
                
                # Forward pass
                outputs = self.model(**inputs)
                
                # If loss is not already in outputs, compute it (for LM models)
                if "labels" in inputs and not hasattr(outputs, "loss"):
                    if hasattr(outputs, "logits"):
                        loss = torch.nn.functional.cross_entropy(
                            outputs.logits.view(-1, outputs.logits.size(-1)),
                            inputs["labels"].view(-1)
                        )
                    else:
                        # Generic fallback
                        loss = outputs[0].mean()
                else:
                    # Use the model's loss
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                
                # Backward pass to get gradients
                optimizer.zero_grad()
                loss.backward()
                
                # Update gradient norms
                for layer_idx in range(self.total_layers):
                    for head_idx in range(self.heads_per_layer):
                        grad_norm = self._get_head_grad_norm(layer_idx, head_idx)
                        grad_norm_values[layer_idx, head_idx] += grad_norm
                
            except StopIteration:
                break
            
        # Average gradient norms over batches if more than one
        if num_batches > 1:
            grad_norm_values = grad_norm_values / num_batches
            
        # No longer need gradients
        optimizer.zero_grad()
        
        return grad_norm_values
    
    def make_pruning_decisions(self, grad_norm_values, target_prune_count=None):
        """
        Make pruning decisions based on gradient metrics.
        
        Args:
            grad_norm_values: Tensor of gradient norms [layer, head]
            target_prune_count: Target number of heads to prune (overrides self.prune_percent)
            
        Returns:
            Tuple of (pruning_mask, target_count)
        """
        # Calculate how many heads we want to prune
        total_heads = self.total_layers * self.heads_per_layer
        max_pruned = int(total_heads * self.max_prune_percent)
        current_pruned = self._count_pruned_heads()
        
        if target_prune_count is None:
            target_prune_count = int(total_heads * self.prune_percent)
        
        # Limit the target by max_prune_percent
        available_to_prune = max_pruned - current_pruned
        if available_to_prune <= 0:
            print(f"Already at maximum pruning level: {current_pruned}/{total_heads} " 
                  f"({current_pruned/total_heads:.1%}) heads pruned")
            return torch.zeros_like(grad_norm_values, dtype=torch.bool), 0
        
        target_prune_count = min(target_prune_count, available_to_prune)
        
        # Create a mask for already pruned heads
        already_pruned_mask = torch.zeros_like(grad_norm_values, dtype=torch.bool)
        for layer_idx in range(self.total_layers):
            for head_idx in range(self.heads_per_layer):
                if self.stats[layer_idx][head_idx]['is_zeroed']:
                    already_pruned_mask[layer_idx, head_idx] = True
        
        # Set gradient values for already pruned heads to a high value
        # so they don't get selected again
        grad_values_for_selection = grad_norm_values.clone()
        grad_values_for_selection[already_pruned_mask] = 1e10
        
        # Flatten for selection
        flat_grad_values = grad_values_for_selection.reshape(-1)
        
        # Get the indices of the heads with the lowest gradient norms
        _, indices = torch.topk(flat_grad_values, k=target_prune_count, largest=False)
        
        # Create pruning mask
        pruning_mask = torch.zeros_like(grad_norm_values, dtype=torch.bool)
        pruning_mask.reshape(-1)[indices] = True
        
        print(f"Gradient-based pruning - target: {target_prune_count} heads")
        print(f"Final pruning decision: pruning {pruning_mask.sum().item()} heads")
        
        return pruning_mask, target_prune_count
    
    def apply_pruning(self, pruning_mask, verbose=False):
        """
        Apply pruning decisions based on the pruning mask.
        
        Args:
            pruning_mask: Boolean tensor of heads to prune [layer, head]
            verbose: Whether to print verbose output
            
        Returns:
            List of (layer_idx, head_idx) tuples that were pruned
        """
        pruned_heads = []
        
        # Convert to list of (layer, head) tuples for pruning
        for layer_idx in range(self.total_layers):
            for head_idx in range(self.heads_per_layer):
                if pruning_mask[layer_idx, head_idx]:
                    # Check if head is already pruned
                    if not self.stats[layer_idx][head_idx]['is_zeroed']:
                        pruned_heads.append((layer_idx, head_idx))
        
        # Apply pruning
        for layer_idx, head_idx in pruned_heads:
            result = prune_head_in_model(
                self.model, 
                layer_idx, 
                head_idx, 
                mode=self.mode, 
                verbose=verbose
            )
            if result:
                # Update controller stats
                self.stats[layer_idx][head_idx]['is_zeroed'] = True
                self.stats[layer_idx][head_idx]['zeroed_epochs'] = 1
                self.stats[layer_idx][head_idx]['pruned_at_step'] = self.step
                if verbose:
                    print(f"Pruned head [{layer_idx},{head_idx}]")
        
        # Update controller hooks
        self._update_pruning_hooks()
        
        return pruned_heads
    
    def step(self, dataloader, num_batches=1, verbose=False):
        """
        Perform a complete pruning step: collect metrics and apply pruning decisions.
        
        Args:
            dataloader: DataLoader for evaluation
            num_batches: Number of batches to process
            verbose: Whether to print verbose output
            
        Returns:
            Tuple of (pruned_heads, metrics)
        """
        # Increment step counter
        self.step += 1
        
        # Collect metrics
        grad_norm_values = self.collect_grad_metrics(dataloader, num_batches)
        
        # Make pruning decisions
        pruning_mask, target_count = self.make_pruning_decisions(grad_norm_values)
        
        # Apply pruning
        pruned_heads = self.apply_pruning(pruning_mask, verbose)
        
        # Get current model info
        model_info = get_model_info(self.model)
        
        # Return comprehensive metrics
        metrics = {
            "epoch": self.epoch,
            "step": self.step,
            "grad_norm": grad_norm_values,
            "pruned_heads": pruned_heads,
            "target_prune_count": target_count,
            "total_pruned": self._count_pruned_heads(),
            "sparsity": model_info["sparsity"],
            "nonzero_params": model_info["nonzero_params"]
        }
        
        # Update head stats
        for layer_idx in range(self.total_layers):
            for head_idx in range(self.heads_per_layer):
                grad_norm = grad_norm_values[layer_idx, head_idx].item()
                self.update_stats(layer_idx, head_idx, grad_norm)
                
                # Update zeroed_epochs counter
                if self.stats[layer_idx][head_idx]['is_zeroed']:
                    self.stats[layer_idx][head_idx]['zeroed_epochs'] += 1
        
        return pruned_heads, metrics
    
    def _get_head_grad_norm(self, layer_idx, head_idx):
        """
        Compute gradient norm for a specific attention head.
        
        Args:
            layer_idx: Layer index
            head_idx: Head index
            
        Returns:
            Gradient norm for this head
        """
        try:
            # Find the transformer blocks
            blocks = None
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                # GPT-2 style
                blocks = self.model.transformer.h
            elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
                # BERT style
                blocks = self.model.encoder.layer
            elif hasattr(self.model, 'layers'):
                # Some models
                blocks = self.model.layers
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                # Wrapped models
                blocks = self.model.model.layers
            
            if blocks is None or layer_idx >= len(blocks):
                return 0.0
            
            # Get the block
            block = blocks[layer_idx]
            
            # Find attention module
            attn_module = None
            if hasattr(block, 'attention'):
                attn_module = block.attention
            elif hasattr(block, 'attn'):
                attn_module = block.attn
            elif hasattr(block, 'self_attention'):
                attn_module = block.self_attention
            elif hasattr(block, 'self_attn'):
                attn_module = block.self_attn
            
            if attn_module is None:
                return 0.0
            
            # Get gradient norm for this head
            norm = 0.0
            
            # For GPT-2 style models with a combined QKV matrix
            if hasattr(attn_module, 'c_attn'):
                if attn_module.c_attn.weight.grad is None:
                    return 0.0
                
                # Get dimensions
                n_heads = getattr(attn_module, 'n_head', None)
                if n_heads is None:
                    n_heads = getattr(attn_module, 'num_heads', 
                                     getattr(attn_module, 'num_attention_heads', 12))
                
                hidden_size = attn_module.c_attn.weight.size(0)
                head_size = hidden_size // n_heads
                
                # Get the starting indices for the head
                q_idx = head_idx * head_size
                k_idx = hidden_size + head_idx * head_size
                v_idx = 2 * hidden_size + head_idx * head_size
                
                # Add up gradient norms for query, key, value
                norm += attn_module.c_attn.weight.grad[q_idx:q_idx+head_size, :].norm().item()
                norm += attn_module.c_attn.weight.grad[k_idx:k_idx+head_size, :].norm().item()
                norm += attn_module.c_attn.weight.grad[v_idx:v_idx+head_size, :].norm().item()
                
                # Add bias gradient if present
                if hasattr(attn_module.c_attn, 'bias') and attn_module.c_attn.bias is not None and attn_module.c_attn.bias.grad is not None:
                    norm += attn_module.c_attn.bias.grad[q_idx:q_idx+head_size].norm().item()
                    norm += attn_module.c_attn.bias.grad[k_idx:k_idx+head_size].norm().item()
                    norm += attn_module.c_attn.bias.grad[v_idx:v_idx+head_size].norm().item()
            
            # For models with separate QKV matrices
            elif (hasattr(attn_module, 'q_proj') and hasattr(attn_module, 'k_proj') and 
                  hasattr(attn_module, 'v_proj')):
                
                if (attn_module.q_proj.weight.grad is None or 
                    attn_module.k_proj.weight.grad is None or 
                    attn_module.v_proj.weight.grad is None):
                    return 0.0
                
                # Get dimensions
                num_heads = getattr(attn_module, 'num_heads', 
                                   getattr(attn_module, 'num_attention_heads',
                                          getattr(attn_module, 'n_head', 12)))
                
                head_size = attn_module.q_proj.weight.size(0) // num_heads
                start_idx = head_idx * head_size
                end_idx = start_idx + head_size
                
                # Add up gradient norms
                norm += attn_module.q_proj.weight.grad[start_idx:end_idx, :].norm().item()
                norm += attn_module.k_proj.weight.grad[start_idx:end_idx, :].norm().item()
                norm += attn_module.v_proj.weight.grad[start_idx:end_idx, :].norm().item()
                
                # Add bias gradient if present
                if (hasattr(attn_module.q_proj, 'bias') and attn_module.q_proj.bias is not None and 
                    attn_module.q_proj.bias.grad is not None):
                    norm += attn_module.q_proj.bias.grad[start_idx:end_idx].norm().item()
                    norm += attn_module.k_proj.bias.grad[start_idx:end_idx].norm().item()
                    norm += attn_module.v_proj.bias.grad[start_idx:end_idx].norm().item()
            
            return norm
        except Exception as e:
            print(f"Error computing gradient norm for head [{layer_idx},{head_idx}]: {e}")
            return 0.0
    
    def _update_pruning_hooks(self):
        """
        Update pruning hooks to maintain pruned state during training.
        """
        # Remove old hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Collect currently pruned heads
        pruned_heads = []
        for layer_idx in range(self.total_layers):
            for head_idx in range(self.heads_per_layer):
                if self.stats[layer_idx][head_idx]['is_zeroed']:
                    pruned_heads.append((layer_idx, head_idx))
        
        # Apply new hooks for all pruned heads
        if pruned_heads:
            self.hooks = apply_pruning_hooks(self.model, pruned_heads, mode=self.mode)
    
    def _count_pruned_heads(self):
        """Count how many heads are currently pruned."""
        count = 0
        for layer_idx in range(self.total_layers):
            for head_idx in range(self.heads_per_layer):
                if self.stats[layer_idx][head_idx]['is_zeroed']:
                    count += 1
        return count
    
    def visualize_gradient_patterns(self, figsize=(12, 8), save_path=None):
        """
        Visualize gradient patterns across layers and heads.
        
        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
        
        Returns:
            Matplotlib figure
        """
        # Create a matrix of gradient norms
        grad_matrix = np.zeros((self.total_layers, self.heads_per_layer))
        
        for layer_idx in range(self.total_layers):
            for head_idx in range(self.heads_per_layer):
                stats = self.stats[layer_idx][head_idx]
                if stats['grad_norm']:
                    # Use average of recent gradient norms
                    grad_matrix[layer_idx, head_idx] = np.mean(stats['grad_norm'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(grad_matrix, cmap="viridis", ax=ax)
        
        # Mark pruned heads with an X
        for layer_idx in range(self.total_layers):
            for head_idx in range(self.heads_per_layer):
                if self.stats[layer_idx][head_idx]['is_zeroed']:
                    ax.text(head_idx + 0.5, layer_idx + 0.5, 'X', 
                           ha='center', va='center', color='white', fontsize=10, fontweight='bold')
        
        # Set labels and title
        ax.set_title("Gradient Norms by Layer and Head")
        ax.set_xlabel("Head Index")
        ax.set_ylabel("Layer Index")
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_summary(self):
        """
        Get a summary of current controller state.
        
        Returns:
            Dictionary with summary statistics
        """
        total_heads = self.total_layers * self.heads_per_layer
        pruned_count = self._count_pruned_heads()
        
        # Get model info
        model_info = get_model_info(self.model)
        
        return {
            "epoch": self.epoch,
            "step": self.step,
            "total_heads": total_heads,
            "pruned_heads": pruned_count,
            "pruning_rate": pruned_count / total_heads,
            "sparsity": model_info["sparsity"],
            "model_size_mb": model_info["size_mb"]
        }


def create_gradient_pruning_controller(model, 
                                      mode=PruningMode.ADAPTIVE, 
                                      prune_percent=0.1,
                                      gradient_percentile=30,
                                      min_zero_epochs=1,
                                      max_prune_percent=0.5):
    """
    Factory function to create a GradientPruningController for a model.
    
    This automatically detects the model architecture and creates an appropriate controller.
    
    Args:
        model: The transformer model
        mode: Pruning mode (ADAPTIVE allows recovery, COMPRESSED prevents recovery)
        prune_percent: Percentage of heads to prune in each step (0-1)
        gradient_percentile: Percentile threshold for gradients (heads with gradients below this are pruned)
        min_zero_epochs: Minimum number of epochs a head should remain pruned
        max_prune_percent: Maximum percentage of heads that can be pruned in total
        
    Returns:
        GradientPruningController instance
    """
    # Find the transformer blocks to count layers and heads
    blocks = None
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-2 style
        blocks = model.transformer.h
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        # BERT style
        blocks = model.encoder.layer
    elif hasattr(model, 'layers'):
        # Some models
        blocks = model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Wrapped models
        blocks = model.model.layers
    
    if blocks is None:
        raise ValueError("Could not detect transformer blocks in model")
    
    # Get number of layers
    total_layers = len(blocks)
    
    # Get first block to determine number of heads
    first_block = blocks[0]
    
    # Find attention module
    attn_module = None
    if hasattr(first_block, 'attention'):
        attn_module = first_block.attention
    elif hasattr(first_block, 'attn'):
        attn_module = first_block.attn
    elif hasattr(first_block, 'self_attention'):
        attn_module = first_block.self_attention
    elif hasattr(first_block, 'self_attn'):
        attn_module = first_block.self_attn
    
    if attn_module is None:
        raise ValueError("Could not detect attention module in model")
    
    # Get number of heads
    heads_per_layer = 12  # Default fallback
    if hasattr(attn_module, 'num_heads'):
        heads_per_layer = attn_module.num_heads
    elif hasattr(attn_module, 'n_head'):
        heads_per_layer = attn_module.n_head
    elif hasattr(attn_module, 'num_attention_heads'):
        heads_per_layer = attn_module.num_attention_heads
    
    # Create and return controller
    return GradientPruningController(
        model=model,
        total_layers=total_layers,
        heads_per_layer=heads_per_layer,
        prune_percent=prune_percent,
        gradient_percentile=gradient_percentile,
        min_zero_epochs=min_zero_epochs,
        mode=mode,
        max_prune_percent=max_prune_percent
    )