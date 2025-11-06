"""
Utility functions for calculating attention head metrics and evaluating importance.

These functions help with:
1. Calculating attention entropy
2. Measuring head utilization and importance
3. Detecting redundant heads
4. Supporting controller decisions about pruning/growth
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def compute_attention_entropy(model, batch_size=8, seq_len=128, device="cpu"):
    """
    Calculate the entropy of attention distributions for each head.
    
    High entropy indicates more uniform attention (potentially less specialized/useful),
    while low entropy suggests the head focuses on specific positions (potentially more useful).
    
    Args:
        model: The adaptive transformer model
        batch_size: Batch size for the random inputs
        seq_len: Sequence length for the random inputs
        device: Device to run the computation on
        
    Returns:
        Dictionary mapping layer indices to tensors of shape [num_heads] containing
        the entropy values for each head
    """
    # Create random input
    random_input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    
    # Set model to eval mode
    model.eval()
    
    # Create storage for attention weights (model must store these in forward pass)
    entropy_dict = {}
    
    # Forward pass to collect attention weights
    with torch.no_grad():
        _ = model(random_input_ids)
        
        # Compute entropy for each layer's attention weights
        for layer_idx, block in enumerate(model.blocks):
            attn_module = block["attn"]
            
            # Check if attention weights are available
            if not hasattr(attn_module, "attention_weights") or not attn_module.attention_weights:
                print(f"Warning: No attention weights stored for layer {layer_idx}")
                continue
            
            # Get attention weights for all heads
            layer_entropies = []
            for head_idx in range(attn_module.num_heads):
                if head_idx not in attn_module.attention_weights:
                    # If head was pruned, assign high entropy
                    layer_entropies.append(torch.tensor(10.0, device=device))
                    continue
                    
                # Get attention weights for this head
                # Shape: [batch_size, seq_len, seq_len]
                attn_weights = attn_module.attention_weights[head_idx]
                
                # Compute entropy across the attention dimension
                # -sum(p * log(p)) where p is the attention probability
                # Small epsilon for numerical stability
                entropy = -torch.sum(
                    attn_weights * torch.log(attn_weights + 1e-10),
                    dim=-1  # Sum over attended positions
                )
                
                # Average across sequence positions and batch
                avg_entropy = entropy.mean()
                layer_entropies.append(avg_entropy)
            
            # Store entropies for this layer
            entropy_dict[layer_idx] = torch.stack(layer_entropies)
    
    return entropy_dict


def compute_head_importance(model, dataloader, loss_fn, device="cpu"):
    """
    Estimate head importance by measuring increase in loss when each head is masked.
    
    Args:
        model: The adaptive transformer model
        dataloader: DataLoader yielding input_ids
        loss_fn: Loss function to evaluate performance
        device: Device to run the computation on
        
    Returns:
        Dictionary mapping layer indices to tensors of shape [num_heads] containing
        importance scores for each head
    """
    # Set model to eval mode
    model.eval()
    
    # Get baseline loss
    baseline_loss = 0.0
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            outputs = model(input_ids)
            loss = loss_fn(outputs[:, :-1, :], input_ids[:, 1:])
            baseline_loss += loss.item() * input_ids.size(0)
            sample_count += input_ids.size(0)
    
    baseline_loss /= sample_count
    
    # Measure importance by masking heads one at a time
    importance_dict = {}
    
    for layer_idx, block in enumerate(model.blocks):
        attn_module = block["attn"]
        head_importance = []
        
        for head_idx in range(attn_module.num_heads):
            # Store original gate value
            original_gate = attn_module.gate[head_idx].item()
            
            # Temporarily mask this head
            attn_module.gate[head_idx] = 0.0
            
            # Compute loss with the head masked
            masked_loss = 0.0
            sample_count = 0
            
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch[0].to(device)
                    outputs = model(input_ids)
                    loss = loss_fn(outputs[:, :-1, :], input_ids[:, 1:])
                    masked_loss += loss.item() * input_ids.size(0)
                    sample_count += input_ids.size(0)
            
            masked_loss /= sample_count
            
            # Importance is the increase in loss when head is masked
            importance = masked_loss - baseline_loss
            head_importance.append(importance)
            
            # Restore original gate value
            attn_module.gate[head_idx] = original_gate
        
        importance_dict[layer_idx] = torch.tensor(head_importance, device=device)
    
    return importance_dict


def compute_gradient_norms(model, dataloader, loss_fn, optimizer, device="cpu"):
    """
    Compute gradient norms for all attention head parameters.
    
    Args:
        model: The adaptive transformer model
        dataloader: DataLoader yielding input_ids
        loss_fn: Loss function to compute gradients
        optimizer: Optimizer to zero_grad/step
        device: Device to run the computation on
        
    Returns:
        Dictionary mapping layer indices to tensors of shape [num_heads] containing
        gradient norms for each head
    """
    # Set model to train mode
    model.train()
    
    # Get a batch of data
    batch = next(iter(dataloader))
    input_ids = batch[0].to(device)
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(input_ids)
    loss = loss_fn(outputs[:, :-1, :], input_ids[:, 1:])
    
    # Backward pass
    loss.backward()
    
    # Compute gradient norms for each head
    grad_norms_dict = {}
    
    for layer_idx, block in enumerate(model.blocks):
        attn_module = block["attn"]
        head_grad_norms = []
        
        for head_idx in range(attn_module.num_heads):
            # Collect all parameters for this head
            head_params = [
                attn_module.W_q[head_idx].weight, attn_module.W_q[head_idx].bias,
                attn_module.W_k[head_idx].weight, attn_module.W_k[head_idx].bias,
                attn_module.W_v[head_idx].weight, attn_module.W_v[head_idx].bias,
                attn_module.W_o[head_idx].weight, attn_module.W_o[head_idx].bias
            ]
            
            # Compute squared sum of gradients
            grad_squared_sum = 0.0
            param_count = 0
            
            for param in head_params:
                if param.grad is not None:
                    grad_squared_sum += torch.sum(param.grad ** 2).item()
                    param_count += param.numel()
            
            # Compute root mean square of gradients
            grad_rms = (grad_squared_sum / param_count) ** 0.5 if param_count > 0 else 0.0
            head_grad_norms.append(grad_rms)
        
        grad_norms_dict[layer_idx] = torch.tensor(head_grad_norms, device=device)
    
    return grad_norms_dict


def visualize_head_metrics(entropy_dict, importance_dict=None, grad_norms_dict=None, save_path=None):
    """
    Visualize attention head metrics as heatmaps.
    
    Args:
        entropy_dict: Dictionary from compute_attention_entropy
        importance_dict: Optional dictionary from compute_head_importance
        grad_norms_dict: Optional dictionary from compute_gradient_norms
        save_path: Optional path to save the figure
    """
    num_metrics = 1 + (importance_dict is not None) + (grad_norms_dict is not None)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 3 * num_metrics))
    
    if num_metrics == 1:
        axes = [axes]
    
    # Plot entropy
    ax = axes[0]
    plot_heatmap(entropy_dict, ax, "Attention Entropy", cmap="viridis")
    
    metric_idx = 1
    
    # Plot importance if provided
    if importance_dict is not None:
        ax = axes[metric_idx]
        plot_heatmap(importance_dict, ax, "Head Importance", cmap="coolwarm")
        metric_idx += 1
    
    # Plot gradient norms if provided
    if grad_norms_dict is not None:
        ax = axes[metric_idx]
        plot_heatmap(grad_norms_dict, ax, "Gradient Norms", cmap="plasma")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_heatmap(metric_dict, ax, title, cmap="viridis"):
    """Helper function to plot a heatmap of metrics."""
    # Convert dictionary to matrix
    num_layers = len(metric_dict)
    num_heads = len(metric_dict[0])
    
    metric_matrix = np.zeros((num_layers, num_heads))
    
    for layer_idx, head_metrics in metric_dict.items():
        metric_matrix[layer_idx] = head_metrics.cpu().numpy()
    
    # Plot heatmap
    sns.heatmap(metric_matrix, ax=ax, cmap=cmap, annot=True, fmt=".2f")
    ax.set_title(title)
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")


def analyze_head_clustering(model, dataloader, device="cpu", n_samples=100):
    """
    Analyze the clustering of attention heads to identify redundant heads.
    
    Args:
        model: The adaptive transformer model
        dataloader: DataLoader yielding input_ids
        device: Device to run computation on
        n_samples: Number of samples to collect attention patterns from
        
    Returns:
        Dictionary mapping layer indices to matrices of cosine similarities between heads
    """
    # Set model to eval mode
    model.eval()
    
    # Collect attention patterns
    attention_patterns = {}
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= n_samples:
                break
                
            input_ids = batch[0].to(device)
            _ = model(input_ids)
            
            # For each layer, collect attention patterns
            for layer_idx, block in enumerate(model.blocks):
                attn_module = block["attn"]
                
                if not hasattr(attn_module, "attention_weights") or not attn_module.attention_weights:
                    continue
                
                if layer_idx not in attention_patterns:
                    attention_patterns[layer_idx] = {}
                
                # For each head, collect attention pattern
                for head_idx in range(attn_module.num_heads):
                    if head_idx not in attn_module.attention_weights:
                        continue
                    
                    attn_weights = attn_module.attention_weights[head_idx]
                    
                    if head_idx not in attention_patterns[layer_idx]:
                        attention_patterns[layer_idx][head_idx] = []
                    
                    # Flatten attention patterns and add to collection
                    flat_attn = attn_weights.view(attn_weights.size(0), -1)
                    attention_patterns[layer_idx][head_idx].append(flat_attn)
            
            sample_count += input_ids.size(0)
    
    # Compute average attention pattern for each head
    avg_patterns = {}
    
    for layer_idx, layer_patterns in attention_patterns.items():
        avg_patterns[layer_idx] = {}
        
        for head_idx, head_patterns in layer_patterns.items():
            # Concatenate all patterns
            all_patterns = torch.cat(head_patterns, dim=0)
            # Average across samples
            avg_pattern = all_patterns.mean(dim=0)
            avg_patterns[layer_idx][head_idx] = avg_pattern
    
    # Compute cosine similarity between heads in each layer
    similarity_dict = {}
    
    for layer_idx, layer_patterns in avg_patterns.items():
        num_heads = len(layer_patterns)
        similarity_matrix = torch.zeros((num_heads, num_heads), device=device)
        
        head_indices = sorted(layer_patterns.keys())
        
        for i, head_i in enumerate(head_indices):
            pattern_i = layer_patterns[head_i]
            
            for j, head_j in enumerate(head_indices):
                pattern_j = layer_patterns[head_j]
                
                # Compute cosine similarity
                similarity = F.cosine_similarity(pattern_i.unsqueeze(0), pattern_j.unsqueeze(0))
                similarity_matrix[i, j] = similarity
        
        similarity_dict[layer_idx] = similarity_matrix
    
    return similarity_dict


def visualize_head_similarities(similarity_dict, threshold=0.8, save_path=None):
    """
    Visualize cosine similarities between attention heads.
    
    Args:
        similarity_dict: Dictionary from analyze_head_clustering
        threshold: Similarity threshold to highlight as potentially redundant
        save_path: Optional path to save the figure
    """
    num_layers = len(similarity_dict)
    fig, axes = plt.subplots(num_layers, 1, figsize=(8, 4 * num_layers))
    
    if num_layers == 1:
        axes = [axes]
    
    for layer_idx, similarity_matrix in similarity_dict.items():
        ax = axes[layer_idx]
        
        # Plot similarity matrix
        im = ax.imshow(similarity_matrix.cpu().numpy(), cmap="coolwarm", vmin=-1, vmax=1)
        
        # Highlight potentially redundant heads
        for i in range(similarity_matrix.size(0)):
            for j in range(similarity_matrix.size(1)):
                if i != j and similarity_matrix[i, j] > threshold:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, 
                                              edgecolor="black", lw=2))
        
        ax.set_title(f"Layer {layer_idx} - Head Similarity")
        ax.set_xlabel("Head Index")
        ax.set_ylabel("Head Index")
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.show()


def recommend_pruning_growth(entropy_dict, importance_dict, grad_norms_dict,
                            entropy_threshold=1.5, importance_threshold=0.05, 
                            grad_norm_threshold=1e-3):
    """
    Recommend which heads to prune or grow based on metrics.
    
    Args:
        entropy_dict: Dictionary from compute_attention_entropy
        importance_dict: Dictionary from compute_head_importance
        grad_norms_dict: Dictionary from compute_gradient_norms
        entropy_threshold: Threshold for high entropy
        importance_threshold: Threshold for low importance
        grad_norm_threshold: Threshold for low gradient norms
        
    Returns:
        prune_candidates: List of (layer_idx, head_idx) tuples suggesting heads to prune
        grow_candidates: List of layer indices suggesting where to add heads
    """
    prune_candidates = []
    grow_candidates = set()
    
    # Collect pruning candidates
    for layer_idx in entropy_dict.keys():
        for head_idx in range(len(entropy_dict[layer_idx])):
            entropy = entropy_dict[layer_idx][head_idx].item()
            importance = importance_dict[layer_idx][head_idx].item() if importance_dict else 0
            grad_norm = grad_norms_dict[layer_idx][head_idx].item() if grad_norms_dict else 0
            
            # Criteria for pruning:
            # 1. High entropy (not focused)
            # 2. Low importance (minimal impact when masked)
            # 3. Low gradient norms (not learning much)
            if (entropy > entropy_threshold and 
                importance < importance_threshold and 
                grad_norm < grad_norm_threshold):
                prune_candidates.append((layer_idx, head_idx))
    
    # Collect growth candidates (layers with few pruning candidates)
    pruning_by_layer = {}
    for layer_idx, head_idx in prune_candidates:
        if layer_idx not in pruning_by_layer:
            pruning_by_layer[layer_idx] = 0
        pruning_by_layer[layer_idx] += 1
    
    # Layers with many active, useful heads might benefit from growth
    for layer_idx in entropy_dict.keys():
        num_heads = len(entropy_dict[layer_idx])
        num_pruned = pruning_by_layer.get(layer_idx, 0)
        
        # If less than 25% of heads are pruned, consider adding heads
        if num_pruned < num_heads * 0.25:
            grow_candidates.add(layer_idx)
    
    return prune_candidates, list(grow_candidates)