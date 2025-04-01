import torch
import numpy as np
from torch import nn

def compute_attention_entropy(attention_weights, eps=1e-8):
    """
    Compute entropy of attention distributions for each head.
    
    High entropy indicates less focused attention patterns (more uniform),
    which may suggest the head is less specialized or useful.
    
    Args:
        attention_weights: Tensor of attention weights [batch, heads, seq_len, seq_len]
        eps: Small epsilon value for numerical stability
    
    Returns:
        Tensor of entropy values for each attention head [heads]
    """
    # Ensure we're working with probabilities that sum to 1
    # attention_weights should already be softmax-ed, but we normalize to be safe
    probs = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + eps)
    
    # Compute entropy: -sum(p * log(p))
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
    
    # Average over batch and sequence length
    head_entropy = entropy.mean(dim=[0, 2])
    
    return head_entropy

def compute_gradient_norm(model, head_mapping_fn=None):
    """
    Compute the gradient norm for each attention head's parameters.
    
    Low gradient norms suggest the head is either well-learned (saturated)
    or contributing little to the model's outputs.
    
    Args:
        model: The transformer model
        head_mapping_fn: Function that maps parameter names to layer/head indices
                       If None, attempt automatic mapping based on parameter names
    
    Returns:
        Tensor of gradient norms for each attention head [layers, heads]
    """
    if head_mapping_fn is None:
        # Default mapping for our AdaptiveTransformer architecture
        def default_mapping(name):
            # Return (layer_idx, head_idx) or None if not an attention head parameter
            if "blocks" in name and ("W_q" in name or "W_k" in name or "W_v" in name or "W_o" in name):
                parts = name.split(".")
                layer_idx = int(parts[1])  # Assuming format: blocks.{layer_idx}.attn.W_q.{head_idx}...
                if "W_q" in name or "W_k" in name or "W_v" in name or "W_o" in name:
                    head_idx = int(parts[4])
                    return layer_idx, head_idx
            return None
        
        head_mapping_fn = default_mapping
    
    num_layers = len(model.blocks) if hasattr(model, "blocks") else 0
    num_heads = model.blocks[0]["attn"].num_heads if num_layers > 0 else 0
    
    # Initialize gradient norm storage
    grad_norms = torch.zeros(num_layers, num_heads, device=next(model.parameters()).device)
    
    # Compute gradient norms for each parameter
    for name, param in model.named_parameters():
        if param.grad is not None:
            mapping = head_mapping_fn(name)
            if mapping is not None:
                layer_idx, head_idx = mapping
                if layer_idx < num_layers and head_idx < num_heads:
                    # Accumulate gradient norms for this head
                    grad_norms[layer_idx, head_idx] += param.grad.norm().item()
    
    return grad_norms

def compute_head_importance(model, dataloader, loss_fn, device, num_batches=10):
    """
    Estimate the importance of each attention head by measuring change in loss
    when the head is masked.
    
    This is a more rigorous but computationally expensive way to determine
    which heads are most critical.
    
    Args:
        model: The transformer model
        dataloader: DataLoader for evaluation
        loss_fn: Loss function
        device: Computation device
        num_batches: Number of batches to evaluate for each head
    
    Returns:
        Tensor of importance scores for each attention head [layers, heads]
    """
    model.eval()
    num_layers = len(model.blocks) if hasattr(model, "blocks") else 0
    num_heads = model.blocks[0]["attn"].num_heads if num_layers > 0 else 0
    
    # Get baseline loss
    baseline_loss = 0.0
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, inputs["labels"])
            baseline_loss += loss.item()
    
    baseline_loss /= min(num_batches, len(dataloader))
    
    # Compute importance for each head
    importance_scores = torch.zeros(num_layers, num_heads, device=device)
    
    # Store original gate values
    original_gates = {}
    for layer_idx, block in enumerate(model.blocks):
        original_gates[layer_idx] = block["attn"].gate.detach().clone()
    
    # Measure loss with each head masked (one at a time)
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            # Temporarily mask this head
            model.blocks[layer_idx]["attn"].gate[head_idx] = 0.0
            
            head_loss = 0.0
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                with torch.no_grad():
                    outputs = model(**inputs)
                    loss = loss_fn(outputs.logits, inputs["labels"])
                    head_loss += loss.item()
            
            head_loss /= min(num_batches, len(dataloader))
            
            # Importance is the increase in loss when head is masked
            # Using ReLU to ensure we only count heads that increase loss when removed
            importance_scores[layer_idx, head_idx] = torch.relu(torch.tensor(head_loss - baseline_loss))
            
            # Restore the original gate value
            model.blocks[layer_idx]["attn"].gate[head_idx] = original_gates[layer_idx][head_idx]
    
    # Normalize importance scores to [0,1] range
    if importance_scores.max() > 0:
        importance_scores = importance_scores / importance_scores.max()
    
    return importance_scores

def collect_head_metrics(model, dataloader=None, loss_fn=None, device=None, num_batches=5):
    """
    Collect all head metrics in a single pass.
    
    Args:
        model: The transformer model
        dataloader: DataLoader for evaluation (optional, needed for importance)
        loss_fn: Loss function (optional, needed for importance)
        device: Computation device
        num_batches: Number of batches to evaluate
    
    Returns:
        Dictionary containing metrics for each head:
            - entropy: Attention entropy
            - grad_norm: Gradient norms
            - importance: Head importance scores (if dataloader provided)
    """
    metrics = {}
    
    # Collect attention entropy
    attention_weights = {}
    
    def collect_weights_hook(layer_idx):
        def hook(module, input, output):
            if hasattr(module, "attention_weights"):
                attention_weights[layer_idx] = module.attention_weights
        return hook
    
    # Register forward hooks to collect attention weights
    hooks = []
    for layer_idx, block in enumerate(model.blocks):
        hook = block["attn"].register_forward_hook(collect_weights_hook(layer_idx))
        hooks.append(hook)
    
    # Run a forward pass to collect attention weights
    if dataloader is not None:
        batch = next(iter(dataloader))
        inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            model(**inputs)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Compute entropy from collected weights
    entropy = torch.zeros(len(model.blocks), model.blocks[0]["attn"].num_heads, device=device)
    for layer_idx, weights_dict in attention_weights.items():
        # Combine all head weights into a single tensor [batch, heads, seq_len, seq_len]
        head_weights = []
        for head_idx in range(model.blocks[layer_idx]["attn"].num_heads):
            if head_idx in weights_dict:
                head_weights.append(weights_dict[head_idx])
        
        if head_weights:
            # Stack head weights along the head dimension
            stacked_weights = torch.stack(head_weights, dim=1)
            head_entropy = compute_attention_entropy(stacked_weights)
            entropy[layer_idx, :len(head_entropy)] = head_entropy
    
    metrics["entropy"] = entropy
    
    # Compute gradient norms if model has gradients
    if any(p.grad is not None for p in model.parameters()):
        metrics["grad_norm"] = compute_gradient_norm(model)
    
    # Compute head importance if dataloader and loss_fn provided
    if dataloader is not None and loss_fn is not None:
        metrics["importance"] = compute_head_importance(model, dataloader, loss_fn, device, num_batches)
    
    return metrics