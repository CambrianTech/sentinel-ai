"""
Neural Plasticity Controller for Dynamic Pruning and Regrowth

This module implements a controller for real-time neural plasticity that dynamically
prunes and revives attention heads based on their utility during training.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional
from sentinel.pruning.dual_mode_pruning import prune_head_in_model, apply_pruning_hooks, PruningMode, get_model_info


class PlasticityDecision(Enum):
    """Possible decisions for each attention head."""
    KEEP = 0    # Keep head as is
    PRUNE = 1   # Prune the head (zero out)
    REVIVE = 2  # Revive a previously pruned head


class PlasticityController:
    """
    Controller for dynamic head pruning and growth during training.
    
    This controller tracks head utility metrics over time and applies
    pruning/revival decisions based on configurable thresholds.
    
    Attributes:
        model: The transformer model
        total_layers: Total number of layers in the model
        heads_per_layer: Number of attention heads per layer
        history_length: Maximum number of history entries to keep
        high_entropy_threshold: Entropy threshold above which heads are considered for pruning
        low_entropy_threshold: Entropy threshold below which pruned heads are considered for revival
        grad_threshold: Gradient norm threshold for pruning decisions
        min_zero_epochs: Minimum number of epochs a head should remain pruned
        mode: Pruning mode (ADAPTIVE allows recovery, COMPRESSED prevents recovery)
        prune_rate_limit: Maximum percentage of heads to prune in one step
        stats: Head statistics tracking
        hooks: Forward/backward hooks for maintaining pruned state
        epoch: Current epoch counter
    """
    
    def __init__(self, 
                 model: torch.nn.Module, 
                 total_layers: int, 
                 heads_per_layer: int,
                 history_length: int = 10,
                 high_entropy_threshold: float = 0.8,
                 low_entropy_threshold: float = 0.4,
                 grad_threshold: float = 1e-3,
                 min_zero_epochs: int = 3,
                 mode: PruningMode = PruningMode.ADAPTIVE,
                 prune_rate_limit: float = 0.1):
        """
        Initialize the plasticity controller.
        
        Args:
            model: The transformer model
            total_layers: Total number of layers in the model
            heads_per_layer: Number of attention heads per layer
            history_length: Maximum number of history entries to keep
            high_entropy_threshold: Entropy threshold above which heads are considered for pruning
            low_entropy_threshold: Entropy threshold below which pruned heads are considered for revival
            grad_threshold: Gradient norm threshold for pruning decisions
            min_zero_epochs: Minimum number of epochs a head should remain pruned
            mode: Pruning mode (ADAPTIVE or COMPRESSED)
            prune_rate_limit: Maximum percentage of heads to prune in one step
        """
        self.model = model
        self.total_layers = total_layers
        self.heads_per_layer = heads_per_layer
        self.history_length = history_length
        self.high_entropy_threshold = high_entropy_threshold
        self.low_entropy_threshold = low_entropy_threshold
        self.grad_threshold = grad_threshold
        self.min_zero_epochs = min_zero_epochs
        self.mode = mode
        self.prune_rate_limit = prune_rate_limit
        
        # Initialize statistics tracking for each head
        self.stats = defaultdict(lambda: defaultdict(lambda: {
            'entropy': [],
            'grad_norm': [],
            'zeroed_epochs': 0,
            'is_zeroed': False,
            'decision_history': []
        }))
        
        # Store hooks for pruning
        self.hooks = []
        
        # Epoch counter
        self.epoch = 0
    
    def update_stats(self, layer_idx: int, head_idx: int, entropy: float, grad_norm: float):
        """
        Update statistics for a specific attention head.
        
        Args:
            layer_idx: Layer index
            head_idx: Head index
            entropy: Current entropy value
            grad_norm: Current gradient norm
        """
        # Get stats for this head
        s = self.stats[layer_idx][head_idx]
        
        # Add new values, maintaining history limit
        s['entropy'].append(entropy)
        if len(s['entropy']) > self.history_length:
            s['entropy'] = s['entropy'][-self.history_length:]
            
        s['grad_norm'].append(grad_norm)
        if len(s['grad_norm']) > self.history_length:
            s['grad_norm'] = s['grad_norm'][-self.history_length:]
    
    def collect_head_metrics(self, dataloader, num_batches=1):
        """
        Collect entropy and gradient metrics for all heads.
        
        Args:
            dataloader: DataLoader for input data
            num_batches: Number of batches to process
            
        Returns:
            Tuple of (entropy_values, grad_norm_values) as 2D tensors [layer, head]
        """
        # Initialize tensors to store metrics
        entropy_values = torch.zeros((self.total_layers, self.heads_per_layer), 
                                    device=next(self.model.parameters()).device)
        grad_norm_values = torch.zeros((self.total_layers, self.heads_per_layer), 
                                      device=next(self.model.parameters()).device)
        
        # Collect attention distributions
        attention_distributions = self._collect_attention_from_model(dataloader, num_batches)
        
        # Calculate entropy for each head
        for layer_idx in range(self.total_layers):
            for head_idx in range(self.heads_per_layer):
                key = (layer_idx, head_idx)
                if key in attention_distributions:
                    entropy_values[layer_idx, head_idx] = self._compute_entropy(attention_distributions[key])
        
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
        
        return entropy_values, grad_norm_values
    
    def apply_plasticity(self, entropy_values, grad_norm_values, verbose=False):
        """
        Apply neural plasticity decisions based on current metrics.
        
        Args:
            entropy_values: Tensor of entropy values [layer, head]
            grad_norm_values: Tensor of gradient norms [layer, head]
            verbose: Whether to print verbose output
            
        Returns:
            Tuple of (pruned_heads, revived_heads) lists
        """
        pruned_heads = []
        revived_heads = []
        
        # Update stats for all heads
        for layer_idx in range(self.total_layers):
            for head_idx in range(self.heads_per_layer):
                entropy = entropy_values[layer_idx, head_idx].item()
                grad_norm = grad_norm_values[layer_idx, head_idx].item()
                
                # Update statistics
                self.update_stats(layer_idx, head_idx, entropy, grad_norm)
                
                # Get head stats
                s = self.stats[layer_idx][head_idx]
                
                # Make decision for this head
                decision = self._decide_head_fate(s, entropy, grad_norm)
                s['decision_history'].append(decision)
                
                if len(s['decision_history']) > self.history_length:
                    s['decision_history'] = s['decision_history'][-self.history_length:]
                
                # Apply decision
                if decision == PlasticityDecision.PRUNE and not s['is_zeroed']:
                    # Prune this head
                    result = prune_head_in_model(self.model, layer_idx, head_idx, 
                                                mode=self.mode, verbose=verbose)
                    if result:
                        s['is_zeroed'] = True
                        s['zeroed_epochs'] = 1
                        pruned_heads.append((layer_idx, head_idx))
                        if verbose:
                            print(f"Pruned head [{layer_idx},{head_idx}] - entropy: {entropy:.3f}, grad: {grad_norm:.5f}")
                
                elif decision == PlasticityDecision.REVIVE and s['is_zeroed']:
                    # Revival only happens in adaptive mode
                    if self.mode == PruningMode.ADAPTIVE:
                        self._revive_head(layer_idx, head_idx)
                        s['is_zeroed'] = False
                        s['zeroed_epochs'] = 0
                        revived_heads.append((layer_idx, head_idx))
                        if verbose:
                            print(f"Revived head [{layer_idx},{head_idx}] - entropy: {entropy:.3f}, grad: {grad_norm:.5f}")
                
                elif s['is_zeroed']:
                    # Update counter for zeroed heads
                    s['zeroed_epochs'] += 1
        
        # Update hooks if there were changes
        if pruned_heads or revived_heads:
            self._update_pruning_hooks()
            
        # Increment epoch counter
        self.epoch += 1
        
        return pruned_heads, revived_heads
    
    def step(self, dataloader, num_batches=1, verbose=False):
        """
        Perform a complete plasticity step: collect metrics and apply pruning decisions.
        
        Args:
            dataloader: DataLoader for evaluation
            num_batches: Number of batches to process
            verbose: Whether to print verbose output
            
        Returns:
            Tuple of (pruned_heads, revived_heads, metrics)
        """
        # Collect metrics
        entropy_values, grad_norm_values = self.collect_head_metrics(dataloader, num_batches)
        
        # Apply plasticity decisions
        pruned_heads, revived_heads = self.apply_plasticity(entropy_values, grad_norm_values, verbose)
        
        # Get current model info
        model_info = get_model_info(self.model)
        
        # Return comprehensive metrics
        metrics = {
            "epoch": self.epoch,
            "entropy": entropy_values,
            "grad_norm": grad_norm_values,
            "pruned_heads": pruned_heads,
            "revived_heads": revived_heads,
            "total_pruned": self._count_pruned_heads(),
            "sparsity": model_info["sparsity"],
            "nonzero_params": model_info["nonzero_params"]
        }
        
        return pruned_heads, revived_heads, metrics
    
    def _decide_head_fate(self, head_stats, entropy, grad_norm):
        """
        Decide whether to prune, keep, or revive a head based on metrics.
        
        Args:
            head_stats: Statistics dictionary for this head
            entropy: Current entropy value
            grad_norm: Current gradient norm
            
        Returns:
            PlasticityDecision (KEEP, PRUNE, or REVIVE)
        """
        if not head_stats['is_zeroed']:
            # For active heads, decide whether to prune
            if (entropy > self.high_entropy_threshold and 
                grad_norm < self.grad_threshold):
                # High entropy and low gradients suggest this head isn't very useful
                return PlasticityDecision.PRUNE
            else:
                return PlasticityDecision.KEEP
        else:
            # For zeroed heads, decide whether to revive (only if in adaptive mode)
            if (self.mode == PruningMode.ADAPTIVE and
                head_stats['zeroed_epochs'] >= self.min_zero_epochs and
                entropy < self.low_entropy_threshold and 
                grad_norm > self.grad_threshold * 2):
                # Low entropy and higher gradients suggest this head might be useful
                return PlasticityDecision.REVIVE
            else:
                return PlasticityDecision.KEEP
    
    def _revive_head(self, layer_idx, head_idx):
        """
        Revive a previously pruned head (only for adaptive mode).
        
        Args:
            layer_idx: Layer index
            head_idx: Head index
        """
        if self.mode != PruningMode.ADAPTIVE:
            return
        
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
                return
            
            # Get the transformer block
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
                return
            
            # Check for head mask
            if hasattr(attn_module, 'head_mask') and attn_module.head_mask is not None:
                with torch.no_grad():
                    attn_module.head_mask[head_idx] = 1.0
                    
            # For our dual-mode pruning system, check the pruning hooks to ensure this head is revived
            # This will be handled by modifying the hooks next time they're applied
        except Exception as e:
            print(f"Error reviving head [{layer_idx},{head_idx}]: {e}")
    
    def _collect_attention_from_model(self, dataloader, num_batches=1):
        """
        Collect attention distributions from the model.
        
        Args:
            dataloader: DataLoader with input data
            num_batches: Number of batches to process
            
        Returns:
            Dictionary mapping (layer_idx, head_idx) to attention distributions
        """
        model = self.model
        device = next(model.parameters()).device
        
        model.eval()
        distributions = {}
        
        # Hook to capture attention distributions
        hooks = []
        attention_outputs = {}
        
        def get_attention_hook(layer_idx):
            def hook(module, input, output):
                try:
                    # Try to extract attention weights
                    if isinstance(output, tuple) and len(output) > 1:
                        # Some modules return attentions as second element
                        attn_weights = output[1]
                    elif hasattr(output, "attentions") and output.attentions is not None:
                        # HuggingFace models with output_attentions=True
                        attn_weights = output.attentions
                    else:
                        # Direct attention output
                        attn_weights = output
                    
                    # Store attention weights
                    if isinstance(attn_weights, torch.Tensor):
                        attention_outputs[layer_idx] = attn_weights.detach()
                except Exception as e:
                    print(f"Error in attention hook: {e}")
            return hook
        
        # Register hooks for each layer's attention module
        for layer_idx in range(self.total_layers):
            # Find the transformer blocks
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
            
            if blocks is None or layer_idx >= len(blocks):
                continue
            
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
                continue
            
            # Register hook
            hook = attn_module.register_forward_hook(get_attention_hook(layer_idx))
            hooks.append(hook)
        
        # Process batches
        with torch.no_grad():
            batch_iterator = iter(dataloader)
            for _ in range(min(num_batches, len(dataloader))):
                try:
                    # Get next batch
                    batch = next(batch_iterator)
                    
                    # Prepare inputs
                    if isinstance(batch, dict):
                        # Dataloader returns dict of tensors
                        inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    else:
                        # Dataloader returns tuple of tensors
                        inputs = {
                            "input_ids": batch[0].to(device),
                            "attention_mask": batch[1].to(device) if len(batch) > 1 else None
                        }
                    
                    # Forward pass
                    _ = model(**inputs, output_attentions=True)
                    
                    # Process collected attention outputs
                    for layer_idx, attn_output in attention_outputs.items():
                        # Attention output should be [batch_size, num_heads, seq_len, seq_len]
                        if attn_output.dim() == 4:
                            # Extract each head's distribution
                            for head_idx in range(attn_output.size(1)):
                                head_attn = attn_output[:, head_idx]
                                key = (layer_idx, head_idx)
                                
                                if key not in distributions:
                                    distributions[key] = [head_attn]
                                else:
                                    distributions[key].append(head_attn)
                
                except StopIteration:
                    break
                except Exception as e:
                    print(f"Error processing batch: {e}")
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return distributions
    
    def _compute_entropy(self, attention_maps, eps=1e-10):
        """
        Compute entropy from attention maps.
        
        Args:
            attention_maps: List of attention maps [batch_size, seq_len, seq_len]
            eps: Small epsilon to avoid log(0)
            
        Returns:
            Average entropy
        """
        if not attention_maps:
            return 0.0
        
        # Concatenate all maps
        maps = torch.cat(attention_maps, dim=0)
        
        # Ensure it's a valid probability distribution
        maps = maps + eps
        maps = maps / maps.sum(dim=-1, keepdim=True)
        
        # Compute entropy: -sum(p * log(p))
        entropy = -torch.sum(maps * torch.log(maps), dim=-1)
        
        # Average over batch and sequence length
        avg_entropy = entropy.mean().item()
        
        # Normalize to [0,1] range
        max_entropy = torch.log(torch.tensor(maps.size(-1), dtype=torch.float))
        normalized_entropy = avg_entropy / max_entropy.item()
        
        return normalized_entropy
    
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
    
    def visualize_head_dynamics(self, metric='entropy', figsize=(12, 8), save_path=None):
        """
        Visualize head dynamics over time.
        
        Args:
            metric: Metric to visualize ('entropy', 'grad_norm', or 'decision')
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data matrix
        max_epochs = self.epoch + 1
        data = np.zeros((self.total_layers * self.heads_per_layer, max_epochs))
        
        # Fill data based on selected metric
        for layer_idx in range(self.total_layers):
            for head_idx in range(self.heads_per_layer):
                row = layer_idx * self.heads_per_layer + head_idx
                s = self.stats[layer_idx][head_idx]
                
                if metric == 'entropy':
                    # Get entropy history
                    for i, value in enumerate(s['entropy']):
                        if i < max_epochs:
                            data[row, i] = value
                
                elif metric == 'grad_norm':
                    # Get gradient norm history
                    for i, value in enumerate(s['grad_norm']):
                        if i < max_epochs:
                            data[row, i] = value
                
                elif metric == 'decision':
                    # Convert decisions to numeric values (0=keep, 1=prune, 2=revive)
                    for i, decision in enumerate(s['decision_history']):
                        if i < max_epochs:
                            data[row, i] = decision.value
                            
                # Mark pruned heads
                if s['is_zeroed']:
                    # Draw a red box around currently pruned heads
                    rect = plt.Rectangle((max_epochs-1-0.5, row-0.5), 1, 1, 
                                        fill=False, edgecolor='red', lw=2)
                    ax.add_patch(rect)
        
        # Plot heatmap
        if metric == 'decision':
            # For decisions, use a discrete colormap
            cmap = plt.cm.get_cmap('viridis', 3)
            im = ax.imshow(data, aspect='auto', cmap=cmap, interpolation='nearest')
            cbar = plt.colorbar(im, ticks=[0, 1, 2])
            cbar.set_ticklabels(['Keep', 'Prune', 'Revive'])
        else:
            # For continuous metrics, use a continuous colormap
            im = ax.imshow(data, aspect='auto', cmap='viridis')
            plt.colorbar(im)
        
        # Set labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Head Index')
        ax.set_title(f'Head {metric.capitalize()} Over Time')
        
        # Set y-tick labels to show layer and head indices
        y_ticks = np.arange(self.total_layers * self.heads_per_layer)
        y_ticklabels = [f"{layer},{head}" for layer in range(self.total_layers) 
                       for head in range(self.heads_per_layer)]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticklabels)
        
        # Set x-tick labels to show epochs
        x_ticks = np.arange(0, max_epochs, max(1, max_epochs // 10))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks)
        
        plt.tight_layout()
        
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
        
        # Count decisions made this epoch
        decisions = {"keep": 0, "prune": 0, "revive": 0}
        for layer_idx in range(self.total_layers):
            for head_idx in range(self.heads_per_layer):
                s = self.stats[layer_idx][head_idx]
                if s['decision_history'] and len(s['decision_history']) > 0:
                    latest = s['decision_history'][-1]
                    if latest == PlasticityDecision.KEEP:
                        decisions["keep"] += 1
                    elif latest == PlasticityDecision.PRUNE:
                        decisions["prune"] += 1
                    elif latest == PlasticityDecision.REVIVE:
                        decisions["revive"] += 1
        
        # Get model info
        model_info = get_model_info(self.model)
        
        return {
            "epoch": self.epoch,
            "total_heads": total_heads,
            "pruned_heads": pruned_count,
            "pruning_rate": pruned_count / total_heads,
            "decisions": decisions,
            "sparsity": model_info["sparsity"],
            "model_size_mb": model_info["size_mb"]
        }


def create_plasticity_controller(model, mode=PruningMode.ADAPTIVE, 
                                high_entropy_threshold=0.8, 
                                low_entropy_threshold=0.4, 
                                grad_threshold=1e-3,
                                min_zero_epochs=3,
                                prune_rate_limit=0.1):
    """
    Factory function to create a PlasticityController for a model.
    
    This automatically detects the model architecture and creates an appropriate controller.
    
    Args:
        model: The transformer model
        mode: Pruning mode (ADAPTIVE allows recovery, COMPRESSED prevents recovery)
        high_entropy_threshold: Entropy threshold above which heads are considered for pruning
        low_entropy_threshold: Entropy threshold below which pruned heads are considered for revival
        grad_threshold: Gradient norm threshold for pruning decisions
        min_zero_epochs: Minimum number of epochs a head should remain pruned
        prune_rate_limit: Maximum percentage of heads to prune in one step
        
    Returns:
        PlasticityController instance
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
    return PlasticityController(
        model=model,
        total_layers=total_layers,
        heads_per_layer=heads_per_layer,
        high_entropy_threshold=high_entropy_threshold,
        low_entropy_threshold=low_entropy_threshold,
        grad_threshold=grad_threshold,
        min_zero_epochs=min_zero_epochs,
        mode=mode,
        prune_rate_limit=prune_rate_limit
    )


# Usage example:
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    
    # Load model and tokenizer
    model_name = "distilgpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set up tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load a small dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:100]")
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.with_format("torch")
    
    # Create dataloader
    dataloader = DataLoader(
        tokenized_dataset, 
        batch_size=4, 
        collate_fn=default_data_collator
    )
    
    # Create plasticity controller
    controller = create_plasticity_controller(
        model=model,
        mode=PruningMode.ADAPTIVE,
        high_entropy_threshold=0.8,
        low_entropy_threshold=0.4,
        grad_threshold=1e-4
    )
    
    # Run a few plasticity steps
    for i in range(5):
        pruned, revived, metrics = controller.step(dataloader, num_batches=1, verbose=True)
        print(f"Epoch {i}:")
        print(f"  Pruned: {len(pruned)} heads, Revived: {len(revived)} heads")
        print(f"  Total pruned: {metrics['total_pruned']} heads")
        print(f"  Model sparsity: {metrics['sparsity']:.4f}")
    
    # Visualize head dynamics
    controller.visualize_head_dynamics(metric='entropy', save_path="entropy_dynamics.png")
    controller.visualize_head_dynamics(metric='decision', save_path="decision_dynamics.png")