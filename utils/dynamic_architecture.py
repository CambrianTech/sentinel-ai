"""
Dynamic Architecture Management for Adaptive Transformer

This module provides functionality for dynamically adjusting the model architecture
during training, including pruning inactive heads and growing new heads in layers
that would benefit from additional capacity.

The key features:
1. Safe pruning that preserves model performance
2. Safe head expansion with sensible initialization
3. Architecture change scheduling during training
4. Validation of changes to prevent catastrophic performance drops
5. Metrics-based decision making for when and where to make changes
"""

import torch
import torch.nn as nn
import copy
import math
from .head_metrics import (
    compute_attention_entropy,
    compute_head_importance,
    compute_gradient_norms,
    recommend_pruning_growth,
    analyze_head_clustering
)


class DynamicArchitectureManager:
    """
    Manager for dynamically adjusting model architecture during training.
    
    This class handles the decision-making process and implementation of
    architecture changes, including pruning and growing attention heads.
    """
    
    def __init__(self, 
                model,
                dataloader,
                loss_fn,
                device="cpu",
                prune_frequency=2000,
                grow_frequency=5000,
                max_prune_per_step=2,
                max_grow_per_step=2,
                min_heads_per_layer=4,
                max_heads_per_layer=16,
                importance_threshold=0.05,
                entropy_threshold=1.5,
                performance_margin=0.05,
                keep_baseline_structure=False):
        """
        Initialize the dynamic architecture manager.
        
        Args:
            model: The adaptive transformer model
            dataloader: DataLoader for evaluation during change validation
            loss_fn: Loss function for evaluating changes
            device: Device to run computations on
            prune_frequency: How often to consider pruning (steps)
            grow_frequency: How often to consider growing (steps)
            max_prune_per_step: Maximum number of heads to prune at once
            max_grow_per_step: Maximum number of heads to grow at once
            min_heads_per_layer: Minimum number of heads per layer
            max_heads_per_layer: Maximum number of heads per layer
            importance_threshold: Maximum importance for prunable heads
            entropy_threshold: Minimum entropy for prunable heads
            performance_margin: Maximum allowed relative increase in validation loss 
                               when making architecture changes (0.05 = 5%)
            keep_baseline_structure: Whether to maintain the same structure as the 
                                   baseline model (only pruning/growing the same
                                   number of heads)
        """
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.device = device
        
        # Configuration
        self.prune_frequency = prune_frequency
        self.grow_frequency = grow_frequency
        self.max_prune_per_step = max_prune_per_step
        self.max_grow_per_step = max_grow_per_step
        self.min_heads_per_layer = min_heads_per_layer
        self.max_heads_per_layer = max_heads_per_layer
        self.importance_threshold = importance_threshold
        self.entropy_threshold = entropy_threshold
        self.performance_margin = performance_margin
        self.keep_baseline_structure = keep_baseline_structure
        
        # Metrics tracking
        self.total_pruned = 0
        self.total_grown = 0
        self.last_prune_step = 0
        self.last_grow_step = 0
        
        # Candidates tracking
        self.prune_candidates = []
        self.grow_candidates = []
        
        # Performance tracking
        self.last_baseline_loss = None
        self.changes_history = []
        
    def should_update_architecture(self, step):
        """
        Determine if architecture should be updated at the current step.
        
        Args:
            step: Current training step
            
        Returns:
            Tuple of (should_prune, should_grow)
        """
        should_prune = (step - self.last_prune_step) >= self.prune_frequency
        should_grow = (step - self.last_grow_step) >= self.grow_frequency
        
        return should_prune, should_grow
    
    def compute_metrics(self):
        """
        Compute metrics for decision making.
        
        Returns:
            Tuple of (entropy_dict, importance_dict, grad_norms_dict)
        """
        # Set model to eval mode temporarily
        training_mode = self.model.training
        self.model.eval()
        
        # Compute metrics
        entropy_dict = compute_attention_entropy(self.model, device=self.device)
        
        # Create optimizer for gradient computation
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        
        importance_dict = compute_head_importance(
            self.model, self.dataloader, self.loss_fn, device=self.device)
        
        grad_norms_dict = compute_gradient_norms(
            self.model, self.dataloader, self.loss_fn, optimizer, device=self.device)
        
        # Restore model mode
        if training_mode:
            self.model.train()
        
        return entropy_dict, importance_dict, grad_norms_dict
    
    def evaluate_model(self, model=None):
        """
        Evaluate model on the dataloader.
        
        Args:
            model: Model to evaluate (defaults to self.model)
            
        Returns:
            Average loss on the dataloader
        """
        if model is None:
            model = self.model
            
        # Set model to eval mode temporarily
        training_mode = model.training
        model.eval()
        
        total_loss = 0.0
        sample_count = 0
        
        with torch.no_grad():
            for batch in self.dataloader:
                input_ids = batch[0].to(self.device)
                outputs = model(input_ids)
                loss = self.loss_fn(outputs[:, :-1, :], input_ids[:, 1:])
                total_loss += loss.item() * input_ids.size(0)
                sample_count += input_ids.size(0)
        
        # Restore model mode
        if training_mode:
            model.train()
        
        return total_loss / sample_count
    
    def update_architecture(self, step, optimizer=None, head_lr_multipliers=None):
        """
        Update the model architecture if appropriate.
        
        Args:
            step: Current training step
            optimizer: Optimizer instance (to update parameter groups)
            head_lr_multipliers: Learning rate multipliers for heads
            
        Returns:
            Tuple of (model, optimizer, head_lr_multipliers, updated)
        """
        should_prune, should_grow = self.should_update_architecture(step)
        
        if not (should_prune or should_grow):
            return self.model, optimizer, head_lr_multipliers, False
        
        # Make copy of model for validation
        updated = False
        model_copy = copy.deepcopy(self.model)
        
        # Compute baseline loss before changes
        baseline_loss = self.evaluate_model()
        self.last_baseline_loss = baseline_loss
        
        # Compute metrics for decision making
        entropy_dict, importance_dict, grad_norms_dict = self.compute_metrics()
        
        # Get candidates
        self.prune_candidates, self.grow_candidates = recommend_pruning_growth(
            entropy_dict, importance_dict, grad_norms_dict,
            entropy_threshold=self.entropy_threshold,
            importance_threshold=self.importance_threshold
        )
        
        # Handle pruning if it's time
        if should_prune:
            self.last_prune_step = step
            updated = updated or self._prune_heads(
                model_copy, baseline_loss, head_lr_multipliers)
        
        # Handle growing if it's time
        if should_grow:
            self.last_grow_step = step
            updated = updated or self._grow_heads(
                model_copy, baseline_loss, head_lr_multipliers)
        
        # If changes were made and validated, update the main model
        if updated:
            # Update the main model with the validated changes
            self.model.load_state_dict(model_copy.state_dict())
            
            # Update optimizer if provided
            if optimizer is not None:
                # Re-create optimizer with updated parameters
                param_groups = optimizer.param_groups
                optimizer = type(optimizer)(self.model.parameters())
                for i, group in enumerate(param_groups):
                    if i < len(optimizer.param_groups):
                        optimizer.param_groups[i]['lr'] = group['lr']
        
        return self.model, optimizer, head_lr_multipliers, updated

    def _prune_heads(self, model_copy, baseline_loss, head_lr_multipliers):
        """
        Attempt to prune heads in the model.
        
        Args:
            model_copy: Copy of the model to modify and validate
            baseline_loss: Current baseline loss before changes
            head_lr_multipliers: Learning rate multipliers dictionary
            
        Returns:
            Whether pruning was performed
        """
        if not self.prune_candidates:
            return False
        
        # Filter candidates to ensure min_heads_per_layer constraint
        filtered_candidates = []
        heads_by_layer = {}
        
        # Count current active heads per layer
        for layer_idx, block in enumerate(model_copy.blocks):
            attn_module = block["attn"]
            active_heads = sum(float(g) > 0.01 for g in attn_module.gate)
            heads_by_layer[layer_idx] = active_heads
        
        # Filter candidates that would violate the min_heads constraint
        for layer_idx, head_idx in self.prune_candidates:
            if heads_by_layer.get(layer_idx, 0) > self.min_heads_per_layer:
                filtered_candidates.append((layer_idx, head_idx))
                heads_by_layer[layer_idx] -= 1
        
        # Apply limit on number of heads to prune
        prune_candidates = filtered_candidates[:self.max_prune_per_step]
        
        if not prune_candidates:
            return False
            
        # Attempt pruning
        print(f"Step {self.last_prune_step}: Attempting to prune {len(prune_candidates)} heads")
        
        # Apply pruning to the model copy
        for layer_idx, head_idx in prune_candidates:
            attn_module = model_copy.blocks[layer_idx]["attn"]
            attn_module.gate.data[head_idx] = 0.0
            
            # Freeze parameters for this head
            for param in attn_module.W_q[head_idx].parameters():
                param.requires_grad = False
            for param in attn_module.W_k[head_idx].parameters():
                param.requires_grad = False
            for param in attn_module.W_v[head_idx].parameters():
                param.requires_grad = False
            for param in attn_module.W_o[head_idx].parameters():
                param.requires_grad = False
        
        # Validate changes
        pruned_loss = self.evaluate_model(model_copy)
        
        # Check if performance is still acceptable
        relative_increase = (pruned_loss - baseline_loss) / baseline_loss
        
        if relative_increase <= self.performance_margin:
            # Changes are acceptable, update tracking
            self.total_pruned += len(prune_candidates)
            
            print(f"Pruning accepted: relative loss increase = {relative_increase:.4f} "
                 f"(below threshold {self.performance_margin:.4f})")
            
            # Update head learning rate multipliers if provided
            if head_lr_multipliers is not None:
                for layer_idx, head_idx in prune_candidates:
                    key = (layer_idx, head_idx)
                    if key in head_lr_multipliers:
                        del head_lr_multipliers[key]
            
            # Record the change
            self.changes_history.append({
                'step': self.last_prune_step,
                'action': 'prune',
                'heads': prune_candidates,
                'baseline_loss': baseline_loss,
                'new_loss': pruned_loss,
                'relative_change': relative_increase
            })
            
            return True
        else:
            # Changes would hurt performance too much, revert
            print(f"Pruning rejected: relative loss increase = {relative_increase:.4f} "
                 f"(above threshold {self.performance_margin:.4f})")
            return False
    
    def _grow_heads(self, model_copy, baseline_loss, head_lr_multipliers):
        """
        Attempt to grow heads in the model.
        
        Args:
            model_copy: Copy of the model to modify and validate
            baseline_loss: Current baseline loss before changes
            head_lr_multipliers: Learning rate multipliers dictionary
            
        Returns:
            Whether head growth was performed
        """
        if not self.grow_candidates:
            return False
        
        # Filter candidates to ensure max_heads_per_layer constraint
        filtered_candidates = []
        
        # Check current head counts
        for layer_idx in self.grow_candidates:
            if layer_idx >= len(model_copy.blocks):
                continue
                
            attn_module = model_copy.blocks[layer_idx]["attn"]
            if attn_module.num_heads < self.max_heads_per_layer:
                filtered_candidates.append(layer_idx)
        
        # Apply limit on number of layers to grow
        grow_candidates = filtered_candidates[:self.max_grow_per_step]
        
        if not grow_candidates:
            return False
            
        # Attempt growing
        heads_per_layer = 2  # Default number of heads to add per layer
        print(f"Step {self.last_grow_step}: Attempting to grow {heads_per_layer} heads "
             f"in {len(grow_candidates)} layers")
        
        # Expand heads in the model copy
        self._expand_attention_heads(model_copy, grow_candidates, heads_per_layer)
        
        # Validate changes
        grown_loss = self.evaluate_model(model_copy)
        
        # Check if performance is still acceptable
        relative_increase = (grown_loss - baseline_loss) / baseline_loss
        
        if relative_increase <= self.performance_margin:
            # Changes are acceptable, update tracking
            self.total_grown += len(grow_candidates) * heads_per_layer
            
            print(f"Growth accepted: relative loss increase = {relative_increase:.4f} "
                 f"(below threshold {self.performance_margin:.4f})")
            
            # Update head learning rate multipliers for the new heads
            if head_lr_multipliers is not None:
                for layer_idx in grow_candidates:
                    block = model_copy.blocks[layer_idx]
                    attn_module = block["attn"]
                    
                    # Set multipliers for new heads
                    old_head_count = attn_module.num_heads - heads_per_layer
                    for head_idx in range(old_head_count, attn_module.num_heads):
                        key = (layer_idx, head_idx)
                        head_lr_multipliers[key] = 1.0
            
            # Record the change
            self.changes_history.append({
                'step': self.last_grow_step,
                'action': 'grow',
                'layers': grow_candidates,
                'heads_per_layer': heads_per_layer,
                'baseline_loss': baseline_loss,
                'new_loss': grown_loss,
                'relative_change': relative_increase
            })
            
            return True
        else:
            # Changes would hurt performance too much, revert
            print(f"Growth rejected: relative loss increase = {relative_increase:.4f} "
                 f"(above threshold {self.performance_margin:.4f})")
            return False
    
    def _expand_attention_heads(self, model, layer_indices, heads_per_layer):
        """
        Expand attention heads in specified layers.
        
        This is a simplified version of the expand_heads.py script, specifically
        designed to be used within the architecture manager.
        
        Args:
            model: The model to modify
            layer_indices: List of layer indices to expand
            heads_per_layer: Number of heads to add per layer
        """
        for layer_idx in layer_indices:
            if layer_idx >= len(model.blocks):
                continue
                
            # Get the block to modify
            block = model.blocks[layer_idx]
            attn_module = block["attn"]
            
            # Current number of heads and dimensions
            current_heads = attn_module.num_heads
            embed_dim = attn_module.embed_dim
            head_dim = attn_module.head_dim
            
            # Target number of heads
            new_heads = current_heads + heads_per_layer
            
            # Create new projection modules for each component
            new_W_q = nn.ModuleList([nn.Linear(embed_dim, head_dim, bias=True) for _ in range(new_heads)])
            new_W_k = nn.ModuleList([nn.Linear(embed_dim, head_dim, bias=True) for _ in range(new_heads)])
            new_W_v = nn.ModuleList([nn.Linear(embed_dim, head_dim, bias=True) for _ in range(new_heads)])
            new_W_o = nn.ModuleList([nn.Linear(head_dim, embed_dim, bias=True) for _ in range(new_heads)])
            
            # Copy existing weights
            for i in range(current_heads):
                new_W_q[i].weight.data.copy_(attn_module.W_q[i].weight.data)
                new_W_q[i].bias.data.copy_(attn_module.W_q[i].bias.data)
                
                new_W_k[i].weight.data.copy_(attn_module.W_k[i].weight.data)
                new_W_k[i].bias.data.copy_(attn_module.W_k[i].bias.data)
                
                new_W_v[i].weight.data.copy_(attn_module.W_v[i].weight.data)
                new_W_v[i].bias.data.copy_(attn_module.W_v[i].bias.data)
                
                new_W_o[i].weight.data.copy_(attn_module.W_o[i].weight.data)
                new_W_o[i].bias.data.copy_(attn_module.W_o[i].bias.data)
            
            # Initialize new heads
            # We'll use a combination of:
            # 1. Clone and slightly perturb existing heads (for stability)
            # 2. Initialize new heads with small random values (for diversity)
            for i in range(current_heads, new_heads):
                # Choose a source head to clone (pick heads with highest activity)
                gate_values = attn_module.gate.detach().cpu().numpy()
                source_idx = gate_values.argmax().item()
                
                # Q, K, V projections - clone with random perturbation
                new_W_q[i].weight.data.copy_(attn_module.W_q[source_idx].weight.data)
                new_W_q[i].weight.data += torch.randn_like(new_W_q[i].weight.data) * 0.01
                new_W_q[i].bias.data.copy_(attn_module.W_q[source_idx].bias.data)
                new_W_q[i].bias.data += torch.randn_like(new_W_q[i].bias.data) * 0.01
                
                new_W_k[i].weight.data.copy_(attn_module.W_k[source_idx].weight.data)
                new_W_k[i].weight.data += torch.randn_like(new_W_k[i].weight.data) * 0.01
                new_W_k[i].bias.data.copy_(attn_module.W_k[source_idx].bias.data)
                new_W_k[i].bias.data += torch.randn_like(new_W_k[i].bias.data) * 0.01
                
                new_W_v[i].weight.data.copy_(attn_module.W_v[source_idx].weight.data)
                new_W_v[i].weight.data += torch.randn_like(new_W_v[i].weight.data) * 0.01
                new_W_v[i].bias.data.copy_(attn_module.W_v[source_idx].bias.data)
                new_W_v[i].bias.data += torch.randn_like(new_W_v[i].bias.data) * 0.01
                
                # Output projection - initialize with small values for stability
                new_W_o[i].weight.data.normal_(mean=0.0, std=0.01)
                new_W_o[i].bias.data.zero_()
            
            # Create new gate values tensor
            new_gate = nn.Parameter(torch.zeros(new_heads))
            # Copy existing gates
            new_gate.data[:current_heads].copy_(attn_module.gate.data)
            # Initialize new gates with slightly conservative values
            new_gate.data[current_heads:].fill_(0.5)  # Start at 0.5 (moderate contribution)
            
            # Replace the module components
            attn_module.W_q = new_W_q
            attn_module.W_k = new_W_k
            attn_module.W_v = new_W_v
            attn_module.W_o = new_W_o
            attn_module.gate = new_gate
            attn_module.num_heads = new_heads
    
    def get_architecture_summary(self):
        """
        Get a summary of the current architecture.
        
        Returns:
            Dictionary with architecture information
        """
        summary = {
            'total_pruned': self.total_pruned,
            'total_grown': self.total_grown,
            'last_prune_step': self.last_prune_step,
            'last_grow_step': self.last_grow_step,
            'prune_candidates': len(self.prune_candidates),
            'grow_candidates': len(self.grow_candidates),
            'layers': []
        }
        
        for layer_idx, block in enumerate(self.model.blocks):
            attn_module = block["attn"]
            active_heads = sum(float(g) > 0.01 for g in attn_module.gate)
            
            summary['layers'].append({
                'layer_idx': layer_idx,
                'total_heads': attn_module.num_heads,
                'active_heads': active_heads,
                'gate_values': attn_module.gate.detach().cpu().tolist()
            })
        
        return summary


# Integration function to use in training.py
def integrate_dynamic_architecture(
    model, train_loader, val_loader, loss_fn, optimizer, head_lr_multipliers,
    global_step, device, **kwargs):
    """
    Integrate dynamic architecture management into the training loop.
    
    Args:
        model: The adaptive transformer model
        train_loader: Training data loader
        val_loader: Validation data loader
        loss_fn: Loss function for evaluation
        optimizer: Model optimizer
        head_lr_multipliers: Learning rate multipliers
        global_step: Current training step
        device: Device to run on
        **kwargs: Additional configuration arguments
        
    Returns:
        Tuple of (model, optimizer, head_lr_multipliers, updated)
    """
    # Initialize or retrieve the architecture manager
    if not hasattr(integrate_dynamic_architecture, 'manager'):
        integrate_dynamic_architecture.manager = DynamicArchitectureManager(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            **kwargs
        )
    
    # Update architecture if appropriate
    manager = integrate_dynamic_architecture.manager
    model, optimizer, head_lr_multipliers, updated = manager.update_architecture(
        global_step, optimizer, head_lr_multipliers)
    
    return model, optimizer, head_lr_multipliers, updated