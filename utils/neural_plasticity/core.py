"""
Core Neural Plasticity Functions

This module implements the fundamental algorithms for neural plasticity, including:
- Entropy and gradient-based head importance calculations
- Pruning mask generation and application
- Model evaluation functions
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any


def calculate_head_entropy(
    attention_maps: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Calculate entropy of attention patterns.
    
    Higher entropy indicates more dispersed/unfocused attention.
    Lower entropy indicates more focused attention.
    
    Args:
        attention_maps: Attention matrix tensor [batch, heads, seq_len, seq_len]
        eps: Small epsilon value to ensure numerical stability
        
    Returns:
        Tensor of shape [layers, heads] containing entropy values
    """
    # Ensure tensor is properly formatted and on the right device
    if not isinstance(attention_maps, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(attention_maps)}")
    
    # Calculate entropy with proper numerical stability
    # Add small epsilon to avoid log(0) issues
    attn_probs = attention_maps.clamp(min=eps)
    
    # Normalize to ensure it's a proper probability distribution
    attn_probs = attn_probs / attn_probs.sum(dim=-1, keepdim=True)
    
    # Calculate entropy: -sum(p * log(p))
    entropy = -torch.sum(attn_probs * torch.log(attn_probs), dim=-1)
    
    # Average over batch and sequence length dimensions if present
    dims_to_reduce = list(range(entropy.dim() - 2))
    if dims_to_reduce:
        entropy = entropy.mean(dim=dims_to_reduce)
    
    return entropy


def calculate_head_gradients(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_batches: int = 2,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Calculate gradient magnitudes for each attention head.
    
    Args:
        model: The transformer model
        dataloader: DataLoader for input data
        num_batches: Number of batches to use for gradient calculation
        device: Device to run on (defaults to model's device)
        
    Returns:
        Tensor of shape [layers, heads] containing gradient magnitudes
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Identify number of layers and heads
    num_layers, num_heads = detect_model_structure(model)
    
    # Initialize gradient accumulation
    grad_norms = torch.zeros((num_layers, num_heads), device=device)
    
    # Set up loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Ensure model is in train mode
    model.train()
    
    # Process batches
    batch_count = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
            
        # Move batch to device
        if isinstance(batch, dict):
            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        else:
            inputs = {"input_ids": batch[0].to(device)}
            if len(batch) > 1:
                inputs["attention_mask"] = batch[1].to(device)
            if len(batch) > 2:
                inputs["labels"] = batch[2].to(device)
        
        # Forward pass with gradient tracking
        outputs = model(**inputs)
        
        # Get loss
        if hasattr(outputs, "loss"):
            loss = outputs.loss
        else:
            if "labels" in inputs:
                labels = inputs["labels"]
            else:
                # For causal language modeling, shift labels
                labels = inputs["input_ids"].clone()[:, 1:].contiguous()
                logits = outputs.logits[:, :-1, :].contiguous()
                
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Collect gradients for each attention head
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                grad = extract_head_gradient(model, layer_idx, head_idx)
                if grad is not None:
                    grad_norms[layer_idx, head_idx] += grad.norm().item()
        
        # Zero gradients
        model.zero_grad()
        batch_count += 1
    
    # Average gradients over batches
    if batch_count > 0:
        grad_norms /= batch_count
    
    return grad_norms


def detect_model_structure(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Detect number of layers and heads in a transformer model.
    
    Args:
        model: The transformer model
        
    Returns:
        Tuple of (num_layers, num_heads)
    """
    # Try various model architectures
    num_layers, num_heads = 0, 0
    
    # Check for direct blocks access (Sentinel-AI adaptive model)
    if hasattr(model, 'blocks'):
        num_layers = len(model.blocks)
        if hasattr(model.blocks[0].attn, 'num_heads'):
            num_heads = model.blocks[0].attn.num_heads
    
    # Try HuggingFace architectures
    else:
        # Get transformer component
        transformer = None
        if hasattr(model, 'transformer'):
            transformer = model.transformer
        elif hasattr(model, 'model') and hasattr(model.model, 'transformer'):
            transformer = model.model.transformer
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'transformer'):
            transformer = model.base_model.transformer
        
        # Find layers and heads
        if transformer:
            if hasattr(transformer, 'h'):  # GPT-2 style
                num_layers = len(transformer.h)
                if hasattr(transformer.h[0].attn, 'num_heads'):
                    num_heads = transformer.h[0].attn.num_heads
            elif hasattr(transformer, 'layer'):  # BERT style
                num_layers = len(transformer.layer)
                if hasattr(transformer.layer[0].attention.self, 'num_attention_heads'):
                    num_heads = transformer.layer[0].attention.self.num_attention_heads
    
    # If still not found, check config
    if num_heads == 0 and hasattr(model, 'config'):
        if hasattr(model.config, 'num_heads'):
            num_heads = model.config.num_heads
        elif hasattr(model.config, 'num_attention_heads'):
            num_heads = model.config.num_attention_heads
        
        if hasattr(model.config, 'num_hidden_layers'):
            num_layers = model.config.num_hidden_layers
        elif hasattr(model.config, 'n_layer'):
            num_layers = model.config.n_layer
    
    if num_layers == 0 or num_heads == 0:
        raise ValueError("Could not detect model structure. Please specify num_layers and num_heads manually.")
    
    return num_layers, num_heads


def extract_head_gradient(
    model: torch.nn.Module,
    layer_idx: int,
    head_idx: int
) -> Optional[torch.Tensor]:
    """
    Extract gradient for a specific attention head.
    
    Args:
        model: The transformer model
        layer_idx: Layer index
        head_idx: Head index
        
    Returns:
        Gradient tensor for the specified head, or None if not found
    """
    # Try different model architectures
    
    # Check for direct blocks access (Sentinel-AI adaptive model)
    if hasattr(model, 'blocks') and layer_idx < len(model.blocks):
        if hasattr(model.blocks[layer_idx].attn, 'gate') and model.blocks[layer_idx].attn.gate.grad is not None:
            return model.blocks[layer_idx].attn.gate.grad[head_idx]
        elif hasattr(model.blocks[layer_idx].attn, 'c_proj.weight') and model.blocks[layer_idx].attn.c_proj.weight.grad is not None:
            # For GPT-2 style
            head_size = model.blocks[layer_idx].attn.c_proj.weight.size(0) // model.blocks[layer_idx].attn.num_heads
            start_idx = head_idx * head_size
            end_idx = (head_idx + 1) * head_size
            return model.blocks[layer_idx].attn.c_proj.weight.grad[start_idx:end_idx]
    
    # Try HuggingFace architectures
    transformer = None
    if hasattr(model, 'transformer'):
        transformer = model.transformer
    elif hasattr(model, 'model') and hasattr(model.model, 'transformer'):
        transformer = model.model.transformer
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'transformer'):
        transformer = model.base_model.transformer
    
    if transformer:
        if hasattr(transformer, 'h') and layer_idx < len(transformer.h):  # GPT-2 style
            # Method 1: Check for gate
            if hasattr(transformer.h[layer_idx].attn, 'gate') and transformer.h[layer_idx].attn.gate.grad is not None:
                return transformer.h[layer_idx].attn.gate.grad[head_idx]
            
            # Method 2: Check for attention output weights
            elif hasattr(transformer.h[layer_idx].attn, 'c_proj') and transformer.h[layer_idx].attn.c_proj.weight.grad is not None:
                head_size = transformer.h[layer_idx].attn.c_proj.weight.size(0) // transformer.h[layer_idx].attn.num_heads
                start_idx = head_idx * head_size
                end_idx = (head_idx + 1) * head_size
                return transformer.h[layer_idx].attn.c_proj.weight.grad[start_idx:end_idx]
        
        elif hasattr(transformer, 'layer') and layer_idx < len(transformer.layer):  # BERT style
            if hasattr(transformer.layer[layer_idx].attention.self, 'gate') and transformer.layer[layer_idx].attention.self.gate.grad is not None:
                return transformer.layer[layer_idx].attention.self.gate.grad[head_idx]
            
            # Fallback to attention output weights for BERT
            elif hasattr(transformer.layer[layer_idx].attention.output, 'dense') and transformer.layer[layer_idx].attention.output.dense.weight.grad is not None:
                attention_head_size = transformer.layer[layer_idx].attention.output.dense.weight.size(0) // transformer.layer[layer_idx].attention.self.num_attention_heads
                start_idx = head_idx * attention_head_size
                end_idx = (head_idx + 1) * attention_head_size
                return transformer.layer[layer_idx].attention.output.dense.weight.grad[start_idx:end_idx]
    
    # Fallback to None if no appropriate gradient found
    return None


def generate_pruning_mask(
    grad_norm_values: torch.Tensor,
    prune_percent: float = 0.1,
    strategy: str = "gradient",
    entropy_values: Optional[torch.Tensor] = None,
    random_seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate a pruning mask based on specified strategy.
    
    Args:
        grad_norm_values: Tensor of gradient norm values for all heads [layers, heads]
        prune_percent: Target percentage of heads to prune (0-1)
        strategy: Pruning strategy - "gradient", "entropy", "random", or "combined"
        entropy_values: Optional tensor of entropy values [layers, heads]
        random_seed: Optional random seed for reproducibility
        
    Returns:
        Boolean tensor where True indicates a head should be pruned
    """
    # Set random seed if provided
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # Flatten tensor for calculating percentiles
    flat_grad_norm = grad_norm_values.view(-1)
    
    # Calculate how many heads we want to prune
    total_heads = grad_norm_values.numel()
    target_prune_count = int(total_heads * prune_percent)
    
    if target_prune_count == 0:
        # Nothing to prune
        return torch.zeros_like(grad_norm_values, dtype=torch.bool)
    
    # Generate mask based on strategy
    if strategy == "gradient":
        # Prune heads with LOWEST gradient norms
        _, indices = torch.topk(flat_grad_norm, k=target_prune_count, largest=False)
        
    elif strategy == "entropy":
        if entropy_values is None:
            raise ValueError("Entropy values required for entropy-based pruning")
        
        # Flatten entropy values
        flat_entropy = entropy_values.view(-1)
        
        # Prune heads with HIGHEST entropy (most unfocused)
        _, indices = torch.topk(flat_entropy, k=target_prune_count, largest=True)
        
    elif strategy == "random":
        # Randomly select heads to prune
        indices = torch.randperm(total_heads)[:target_prune_count]
        
    elif strategy == "combined":
        if entropy_values is None:
            raise ValueError("Entropy values required for combined pruning")
        
        # Normalize gradient norms (higher is better)
        norm_grad = 1.0 - (flat_grad_norm - flat_grad_norm.min()) / (flat_grad_norm.max() - flat_grad_norm.min() + 1e-8)
        
        # Normalize entropy (higher is worse)
        flat_entropy = entropy_values.view(-1)
        norm_entropy = (flat_entropy - flat_entropy.min()) / (flat_entropy.max() - flat_entropy.min() + 1e-8)
        
        # Combine metrics (higher score = more likely to prune)
        combined_score = norm_entropy * 0.6 + norm_grad * 0.4
        
        # Select heads with highest combined score
        _, indices = torch.topk(combined_score, k=target_prune_count, largest=True)
    
    else:
        raise ValueError(f"Unknown pruning strategy: {strategy}")
    
    # Create pruning mask where True = head should be pruned
    pruning_mask = torch.zeros_like(grad_norm_values, dtype=torch.bool)
    pruning_mask.view(-1)[indices] = True
    
    return pruning_mask


def apply_pruning_mask(
    model: torch.nn.Module,
    pruning_mask: torch.Tensor,
    mode: str = "zero_weights"
) -> List[Tuple[int, int]]:
    """
    Apply pruning to a model based on the provided mask.
    
    Args:
        model: The transformer model
        pruning_mask: Boolean tensor where True indicates a head should be pruned
        mode: Pruning mode - "zero_weights", "mask_forward", or "gate"
        
    Returns:
        List of (layer_idx, head_idx) tuples of pruned heads
    """
    pruned_heads = []
    num_layers, num_heads = detect_model_structure(model)
    
    # Verify mask shape
    if pruning_mask.shape != (num_layers, num_heads):
        raise ValueError(f"Mask shape {pruning_mask.shape} doesn't match model structure {(num_layers, num_heads)}")
    
    # Apply pruning for each True value in the mask
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            if pruning_mask[layer_idx, head_idx]:
                # Prune this head
                success = prune_single_head(model, layer_idx, head_idx, mode)
                if success:
                    pruned_heads.append((layer_idx, head_idx))
    
    return pruned_heads


def prune_single_head(
    model: torch.nn.Module,
    layer_idx: int,
    head_idx: int,
    mode: str = "zero_weights"
) -> bool:
    """
    Prune a single attention head in the model.
    
    Args:
        model: The transformer model
        layer_idx: Layer index
        head_idx: Head index
        mode: Pruning mode - "zero_weights", "mask_forward", or "gate"
        
    Returns:
        Boolean indicating success
    """
    # Try different model architectures
    
    # Check for direct blocks access (Sentinel-AI adaptive model)
    if hasattr(model, 'blocks') and layer_idx < len(model.blocks):
        if mode == "gate" and hasattr(model.blocks[layer_idx].attn, 'gate'):
            # Set gate value to 0
            with torch.no_grad():
                model.blocks[layer_idx].attn.gate[head_idx] = 0.0
            return True
        elif mode == "zero_weights" and hasattr(model.blocks[layer_idx].attn, 'c_proj'):
            # Zero out output projection weights for this head
            head_size = model.blocks[layer_idx].attn.c_proj.weight.size(0) // model.blocks[layer_idx].attn.num_heads
            start_idx = head_idx * head_size
            end_idx = (head_idx + 1) * head_size
            with torch.no_grad():
                model.blocks[layer_idx].attn.c_proj.weight[start_idx:end_idx, :] = 0.0
                if hasattr(model.blocks[layer_idx].attn.c_proj, 'bias'):
                    model.blocks[layer_idx].attn.c_proj.bias[start_idx:end_idx] = 0.0
            return True
    
    # Try HuggingFace architectures
    transformer = None
    if hasattr(model, 'transformer'):
        transformer = model.transformer
    elif hasattr(model, 'model') and hasattr(model.model, 'transformer'):
        transformer = model.model.transformer
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'transformer'):
        transformer = model.base_model.transformer
    
    if transformer:
        if hasattr(transformer, 'h') and layer_idx < len(transformer.h):  # GPT-2 style
            if mode == "gate" and hasattr(transformer.h[layer_idx].attn, 'gate'):
                # Set gate value to 0
                with torch.no_grad():
                    transformer.h[layer_idx].attn.gate[head_idx] = 0.0
                return True
            elif mode == "zero_weights" and hasattr(transformer.h[layer_idx].attn, 'c_proj'):
                # Zero out output projection weights for this head
                head_size = transformer.h[layer_idx].attn.c_proj.weight.size(0) // transformer.h[layer_idx].attn.num_heads
                start_idx = head_idx * head_size
                end_idx = (head_idx + 1) * head_size
                with torch.no_grad():
                    transformer.h[layer_idx].attn.c_proj.weight[start_idx:end_idx, :] = 0.0
                    if hasattr(transformer.h[layer_idx].attn.c_proj, 'bias'):
                        transformer.h[layer_idx].attn.c_proj.bias[start_idx:end_idx] = 0.0
                return True
        
        elif hasattr(transformer, 'layer') and layer_idx < len(transformer.layer):  # BERT style
            if mode == "gate" and hasattr(transformer.layer[layer_idx].attention.self, 'gate'):
                # Set gate value to 0
                with torch.no_grad():
                    transformer.layer[layer_idx].attention.self.gate[head_idx] = 0.0
                return True
            elif mode == "zero_weights" and hasattr(transformer.layer[layer_idx].attention.output, 'dense'):
                # Zero out output projection weights for this head
                attention_head_size = transformer.layer[layer_idx].attention.output.dense.weight.size(0) // transformer.layer[layer_idx].attention.self.num_attention_heads
                start_idx = head_idx * attention_head_size
                end_idx = (head_idx + 1) * attention_head_size
                with torch.no_grad():
                    transformer.layer[layer_idx].attention.output.dense.weight[start_idx:end_idx, :] = 0.0
                    if hasattr(transformer.layer[layer_idx].attention.output.dense, 'bias'):
                        transformer.layer[layer_idx].attention.output.dense.bias[start_idx:end_idx] = 0.0
                return True
    
    # If we couldn't prune using the specified mode, try to implement mask_forward
    if mode == "mask_forward":
        # Implement a forward hook to mask attention
        def mask_head_forward_hook(module, input, output):
            # Find appropriate attention output dimensions
            if output.dim() == 4:  # [batch, heads, seq_len, seq_len] or [batch, heads, seq_len, head_dim]
                mask = torch.ones_like(output)
                mask[:, head_idx] = 0
                return output * mask
            elif output.dim() == 3:  # [batch, seq_len, hidden_size]
                # More complex - need to identify head dimension
                try:
                    head_dim = output.size(-1) // num_heads
                    reshaped = output.view(output.size(0), output.size(1), num_heads, head_dim)
                    mask = torch.ones_like(reshaped)
                    mask[:, :, head_idx, :] = 0
                    masked = reshaped * mask
                    return masked.view(output.size())
                except:
                    # Can't properly mask, return original
                    return output
            else:
                # Unexpected shape, return original
                return output
        
        # Try to find the right module to hook
        try:
            if hasattr(model, 'blocks') and layer_idx < len(model.blocks):
                model.blocks[layer_idx].attn.register_forward_hook(mask_head_forward_hook)
                return True
            elif transformer and hasattr(transformer, 'h') and layer_idx < len(transformer.h):
                transformer.h[layer_idx].attn.register_forward_hook(mask_head_forward_hook)
                return True
            elif transformer and hasattr(transformer, 'layer') and layer_idx < len(transformer.layer):
                transformer.layer[layer_idx].attention.self.register_forward_hook(mask_head_forward_hook)
                return True
        except:
            return False
    
    # Could not prune with any method
    return False


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
    max_eval_steps: int = 10
) -> Dict[str, float]:
    """
    Evaluate model on the provided dataloader.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on (defaults to model's device)
        max_eval_steps: Maximum number of steps to use for evaluation
        
    Returns:
        Dictionary with 'loss' and 'perplexity' metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    total_loss = 0.0
    total_steps = 0
    
    # Set up loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Break if we've evaluated enough steps
            if batch_idx >= max_eval_steps:
                break
                
            # Move batch to device
            if isinstance(batch, dict):
                inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            else:
                inputs = {"input_ids": batch[0].to(device)}
                if len(batch) > 1:
                    inputs["attention_mask"] = batch[1].to(device)
                if len(batch) > 2:
                    inputs["labels"] = batch[2].to(device)
            
            # Forward pass
            outputs = model(**inputs)
            
            # Get loss
            if hasattr(outputs, "loss"):
                loss = outputs.loss
            else:
                if "labels" in inputs:
                    labels = inputs["labels"]
                else:
                    # For causal language modeling, shift labels
                    labels = inputs["input_ids"].clone()[:, 1:].contiguous()
                    logits = outputs.logits[:, :-1, :].contiguous()
                    
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            total_loss += loss.item()
            total_steps += 1
    
    avg_loss = total_loss / total_steps if total_steps > 0 else float("inf")
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity
    }