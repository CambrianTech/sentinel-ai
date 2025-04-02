import torch
import torch.nn.functional as F
import math
import os

def compute_perplexity(loss):
    return math.exp(loss)

def compute_loss(logits, targets):
    """
    Compute cross-entropy loss for language modeling.
    
    Args:
        logits: Model output logits [batch_size, sequence_length, vocab_size]
        targets: Target token IDs [batch_size, sequence_length]
        
    Returns:
        Loss tensor
    """
    # Shift targets for language modeling (predict next token)
    # We predict all tokens except the last one, and use all tokens except the first one as targets
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_targets = targets[:, 1:].contiguous()
    
    # Flatten the logits and targets
    vocab_size = shifted_logits.size(-1)
    flat_logits = shifted_logits.view(-1, vocab_size)
    flat_targets = shifted_targets.view(-1)
    
    # Compute loss with optional label smoothing
    loss = F.cross_entropy(flat_logits, flat_targets)
    
    return loss

def save_checkpoint(model, optimizer, head_lr_multipliers, step, epoch, path):
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "head_lr_multipliers": head_lr_multipliers,
        "train_step": step,
        "epoch": epoch
    }
    torch.save(state, path)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(model, optimizer, head_lr_multipliers, path, device):
    if not os.path.exists(path):
        print("No checkpoint found, starting fresh.")
        return model, optimizer, head_lr_multipliers, 0, 0

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    head_lr_multipliers = checkpoint.get("head_lr_multipliers", head_lr_multipliers)
    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("train_step", 0)
    print(f"Loaded checkpoint from {path}, resuming at epoch {epoch}, step {step}")
    return model, optimizer, head_lr_multipliers, step, epoch

def evaluate_model(model, eval_loader, device):
    """
    Evaluate model on evaluation dataset.
    
    Args:
        model: The model to evaluate
        eval_loader: DataLoader for evaluation data
        device: Device to run evaluation on
        
    Returns:
        average_loss: Average loss on evaluation set
        perplexity: Perplexity on evaluation set
    """
    model.eval()
    total_loss = 0
    total_steps = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch[0].to(device)
            targets = input_ids.clone()
            
            # Forward pass
            outputs = model(input_ids)
            
            # Compute loss
            loss = compute_loss(outputs, targets)
            
            # Update totals
            total_loss += loss.item()
            total_steps += 1
    
    # Calculate average loss and perplexity
    average_loss = total_loss / total_steps if total_steps > 0 else float('inf')
    perplexity = compute_perplexity(average_loss)
    
    model.train()
    return average_loss, perplexity

def adjust_learning_rates(optimizer, base_lr, step, warmup_steps=1000, decay_rate=0.95, decay_steps=1000):
    """
    Adjust learning rates based on step.
    
    Args:
        optimizer: PyTorch optimizer
        base_lr: Base learning rate
        step: Current training step
        warmup_steps: Number of warmup steps
        decay_rate: Exponential decay rate
        decay_steps: Steps between decay applications
        
    Returns:
        current_lr: Current learning rate after adjustment
    """
    # Warmup phase: linearly increase learning rate
    if step < warmup_steps:
        lr = base_lr * (step / warmup_steps)
    else:
        # Decay phase: exponentially decay learning rate
        steps_after_warmup = step - warmup_steps
        decay_factor = decay_rate ** (steps_after_warmup / decay_steps)
        lr = base_lr * decay_factor
    
    # Apply to all parameter groups
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr

def prune_inactive_heads(model, threshold=0.1):
    """
    Prune heads that have gate values below threshold.
    
    Args:
        model: The adaptive transformer model
        threshold: Gate value threshold below which heads are considered inactive
        
    Returns:
        num_pruned: Number of heads pruned
    """
    if not hasattr(model, "blocks"):
        return 0
        
    num_pruned = 0
    
    with torch.no_grad():
        for layer_idx, block in enumerate(model.blocks):
            attn_module = block["attn"]
            gates = attn_module.gate
            
            for head_idx, gate in enumerate(gates):
                if gate < threshold:
                    # Set gate to exactly zero
                    attn_module.gate[head_idx] = 0.0
                    num_pruned += 1
    
    return num_pruned

def count_active_heads(model):
    """
    Count active (non-pruned) attention heads in the model.
    
    Args:
        model: The adaptive transformer model
        
    Returns:
        active_heads: Number of active heads
        total_heads: Total number of heads
    """
    if not hasattr(model, "blocks"):
        return 0, 0
        
    total_heads = 0
    active_heads = 0
    
    for layer_idx, block in enumerate(model.blocks):
        attn_module = block["attn"]
        gates = attn_module.gate
        
        for head_idx, gate in enumerate(gates):
            total_heads += 1
            if gate > 0.01:  # Consider heads with gate > 0.01 as active
                active_heads += 1
    
    return active_heads, total_heads

def count_trainable_params(model):
    """
    Count trainable parameters in the model.
    
    Args:
        model: The model to count parameters for
        
    Returns:
        Count of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
