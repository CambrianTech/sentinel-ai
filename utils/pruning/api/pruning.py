"""
Core API for model pruning functionality.
Contains functions for head importance calculation, pruning, and fine-tuning.
"""

import torch
import numpy as np
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

def compute_head_importance(model, dataloader, num_batches=10, device="cuda"):
    """
    Compute importance scores for each attention head using a specified strategy.
    
    Args:
        model: The transformer model to analyze
        dataloader: DataLoader containing evaluation data
        num_batches: Number of batches to process for importance calculation
        device: Device to use for computation
        
    Returns:
        numpy array of shape [num_layers, num_heads] with importance scores
    """
    print("Computing head importance...")
    
    # Get model architecture details
    config = model.config
    num_layers = config.n_layer if hasattr(config, "n_layer") else config.num_hidden_layers
    num_heads = config.n_head if hasattr(config, "n_head") else config.num_attention_heads
    
    # For this simplified version, we're using random importance scores
    # In a real implementation, you would compute importance based on
    # attention entropy, weight magnitudes, or other metrics
    importance = np.random.rand(num_layers, num_heads)
    
    print(f"Computed importance for {num_layers} layers with {num_heads} heads each")
    return importance

def prune_heads(model, importance, pruning_percent=0.3, device="cuda"):
    """
    Prune the least important attention heads by creating a head mask.
    
    Args:
        model: The transformer model to prune
        importance: Array of shape [num_layers, num_heads] with importance scores
        pruning_percent: Fraction of heads to prune (0.0 to 1.0)
        device: Device to use for computation
        
    Returns:
        list of (layer_idx, head_idx) tuples of pruned heads
    """
    print(f"Pruning {pruning_percent*100:.1f}% of attention heads...")
    
    # Get model configuration
    config = model.config
    num_layers = config.n_layer if hasattr(config, "n_layer") else config.num_hidden_layers
    num_heads = config.n_head if hasattr(config, "n_head") else config.num_attention_heads
    
    # Reshape importance to 1D for ranking
    flat_importance = importance.flatten()
    
    # Determine how many heads to prune
    num_heads_total = num_layers * num_heads  
    k = int(num_heads_total * pruning_percent)
    
    if k <= 0:
        print("Pruning percentage too low, no heads will be pruned")
        return []
    
    # Find indices of least important heads
    indices = np.argsort(flat_importance)[:k]
    
    # Convert to (layer, head) pairs
    heads_to_prune = [(idx // num_heads, idx % num_heads) for idx in indices]
    
    # Create a mask to apply during forward pass
    head_mask = torch.ones(num_layers, num_heads).to(device)
    for layer, head in heads_to_prune:
        head_mask[layer, head] = 0.0
    
    # Store on the model for future use
    model.head_mask = head_mask
    model.pruned_heads = heads_to_prune
    
    # Monkey patch the forward method to use our head mask
    original_forward = model.forward
    
    def forward_with_head_mask(input_ids=None, **kwargs):
        # Add head_mask to kwargs
        kwargs['head_mask'] = model.head_mask
        return original_forward(input_ids, **kwargs)
    
    # Replace the forward method
    import types
    model.forward = types.MethodType(forward_with_head_mask, model)
    
    print(f"Pruned {len(heads_to_prune)} attention heads")
    return heads_to_prune

def fine_tune(model, train_dataloader, val_dataloader, num_epochs=3, 
              learning_rate=5e-5, max_steps=None, device="cuda", 
              eval_every=100, callbacks=None):
    """
    Fine-tune a pruned model to recover performance.
    
    Args:
        model: The pruned model to fine-tune
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        max_steps: Maximum number of training steps (for testing)
        device: Device to use for computation
        eval_every: Evaluate model every N steps
        callbacks: Optional dict of callback functions
        
    Returns:
        Tuple of (final_loss, final_perplexity)
    """
    print(f"Starting fine-tuning for {num_epochs} epochs...")
    
    # Set up callbacks
    if callbacks is None:
        callbacks = {}
    
    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Calculate total steps
    if max_steps:
        total_steps = min(len(train_dataloader) * num_epochs, max_steps)
    else:
        total_steps = len(train_dataloader) * num_epochs
        
    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=min(100, total_steps // 10), 
        num_training_steps=total_steps
    )
    
    # Track metrics
    best_val_loss = float('inf')
    step_count = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for step, (input_ids, attention_mask) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            # Check if we've reached max_steps
            if max_steps and step_count >= max_steps:
                break
                
            # Move to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids, 
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            # Compute loss
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            
            # Record loss
            train_loss += loss.item()
            
            # Call step callback if provided
            if 'on_step' in callbacks:
                callbacks['on_step'](step_count, loss.item())
                
            # Evaluate periodically
            if step_count % eval_every == 0 or (step == len(train_dataloader) - 1):
                val_loss, val_ppl = evaluate_model(model, val_dataloader, device=device)
                print(f"Step {step_count}/{total_steps} - Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
                
                # Call eval callback if provided
                if 'on_eval' in callbacks:
                    callbacks['on_eval'](step_count, val_loss, val_ppl)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if 'on_best_model' in callbacks:
                        callbacks['on_best_model'](model)
            
            # Increment step counter
            step_count += 1
        
        # Calculate average training loss for this epoch
        avg_train_loss = train_loss / (step + 1)
        print(f"Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f}")
        
        # Call epoch callback if provided
        if 'on_epoch' in callbacks:
            callbacks['on_epoch'](epoch, avg_train_loss)
    
    # Final evaluation
    final_loss, final_ppl = evaluate_model(model, val_dataloader, device=device)
    print(f"Final evaluation - Loss: {final_loss:.4f}, Perplexity: {final_ppl:.2f}")
    
    # Call completion callback if provided
    if 'on_complete' in callbacks:
        callbacks['on_complete'](final_loss, final_ppl)
    
    return final_loss, final_ppl

def evaluate_model(model, dataloader, device="cuda"):
    """
    Evaluate a model on the given dataloader.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing evaluation data
        device: Device to use for computation
        
    Returns:
        Tuple of (loss, perplexity)
    """
    model.eval()
    total_loss = 0
    total_elements = 0
    
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            
            # Get loss
            loss = outputs.loss
            
            # Accumulate loss
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_elements += batch_size
    
    # Calculate average loss and perplexity
    avg_loss = total_loss / total_elements if total_elements > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity