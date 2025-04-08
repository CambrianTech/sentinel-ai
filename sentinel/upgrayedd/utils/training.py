"""
Training utilities for fine-tuning and evaluating transformer models.
"""

import os
import torch
import torch.nn as nn
import logging
import math
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)

def fine_tune_model(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    tokenizer: Any,
    learning_rate: float = 5e-5,
    num_epochs: int = 3,
    device: str = "cuda",
    gradient_accumulation_steps: int = 1,
    use_differential_lr: bool = False,
    weight_decay: float = 0.01,
    warmup_steps: int = 0,
    max_grad_norm: float = 1.0,
    progress_tracker: Optional[Any] = None,
    eval_steps: Optional[int] = None,
    checkpoint_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fine-tune a transformer model.
    
    Args:
        model: The model to fine-tune
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        tokenizer: Tokenizer for the model
        learning_rate: Learning rate for optimization
        num_epochs: Number of training epochs
        device: Device to train on
        gradient_accumulation_steps: Number of steps to accumulate gradients
        use_differential_lr: Whether to use different learning rates for different heads
        weight_decay: Weight decay for AdamW optimizer
        warmup_steps: Number of warmup steps for learning rate scheduler
        max_grad_norm: Maximum gradient norm for gradient clipping
        progress_tracker: Optional progress tracker for metrics
        eval_steps: Evaluate every n steps (default: once per epoch)
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        Dictionary with training metrics
    """
    # Validate inputs
    if gradient_accumulation_steps < 1:
        logger.warning(f"Invalid gradient_accumulation_steps: {gradient_accumulation_steps}, setting to 1")
        gradient_accumulation_steps = 1
    
    # Prepare optimizer with differential learning rates if requested
    if use_differential_lr:
        param_groups = _create_differential_param_groups(model, learning_rate)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Prepare scheduler
    total_steps = len(train_dataloader) // gradient_accumulation_steps * num_epochs
    if warmup_steps < 0:
        warmup_steps = int(total_steps * 0.1)  # Default to 10% warmup
    
    # Create scheduler
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Set up evaluation steps
    if eval_steps is None:
        eval_steps = len(train_dataloader) // gradient_accumulation_steps
    
    # Training metrics
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_perplexities = []
    global_step = 0
    
    # Training loop
    logger.info(f"Starting fine-tuning for {num_epochs} epochs")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Progress bar for this epoch
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Prepare inputs
            if isinstance(batch, dict):
                inputs = {k: v.to(device) for k, v in batch.items()}
            else:
                # Assume the batch is a tuple with (input_ids, attention_mask)
                inputs = {
                    "input_ids": batch[0].to(device),
                    "attention_mask": batch[1].to(device) if len(batch) > 1 else None,
                    "labels": batch[0].to(device)  # Use input_ids as labels for LM
                }
            
            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update metrics
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
            
            # Only update parameters after accumulating enough gradients
            if (step + 1) % gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Update parameters
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Evaluate if needed
                if global_step % eval_steps == 0:
                    val_loss, val_ppl = evaluate_model(model, val_dataloader, device)
                    
                    # Log metrics
                    logger.info(f"Step {global_step}/{total_steps} - Val loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
                    
                    # Update tracker if provided
                    if progress_tracker:
                        progress_tracker.add_metrics(
                            step=global_step,
                            loss=val_loss,
                            perplexity=val_ppl
                        )
                    
                    # Save best model
                    if val_loss < best_val_loss and checkpoint_dir:
                        best_val_loss = val_loss
                        best_model_path = os.path.join(checkpoint_dir, "best_model")
                        model.save_pretrained(best_model_path)
                        tokenizer.save_pretrained(best_model_path)
                        logger.info(f"Saved best model to {best_model_path}")
                    
                    # Record metrics
                    val_losses.append(val_loss)
                    val_perplexities.append(val_ppl)
                    
                    # Back to training mode
                    model.train()
        
        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_epoch_loss)
        
        # Evaluate at the end of the epoch
        val_loss, val_ppl = evaluate_model(model, val_dataloader, device)
        val_losses.append(val_loss)
        val_perplexities.append(val_ppl)
        
        # Log epoch summary
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train loss: {avg_epoch_loss:.4f}, Val loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
        
        # Update tracker if provided
        if progress_tracker:
            progress_tracker.add_metrics(
                step=(epoch+1) * len(train_dataloader),
                loss=val_loss,
                perplexity=val_ppl
            )
        
        # Save checkpoint if requested
        if checkpoint_dir:
            epoch_checkpoint = os.path.join(checkpoint_dir, f"epoch_{epoch+1}")
            model.save_pretrained(epoch_checkpoint)
            tokenizer.save_pretrained(epoch_checkpoint)
            logger.info(f"Saved epoch checkpoint to {epoch_checkpoint}")
    
    # Final evaluation
    final_loss, final_ppl = evaluate_model(model, val_dataloader, device)
    
    # Collect results
    results = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_perplexities": val_perplexities,
        "final_loss": final_loss,
        "final_perplexity": final_ppl,
        "best_val_loss": best_val_loss,
        "epochs": num_epochs,
        "steps": global_step
    }
    
    logger.info(f"Fine-tuning complete. Final loss: {final_loss:.4f}, Final PPL: {final_ppl:.2f}")
    
    return results

def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda"
) -> Tuple[float, float]:
    """
    Evaluate a transformer model.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on
        
    Returns:
        Tuple of (loss, perplexity)
    """
    model.eval()
    device_obj = torch.device(device)
    
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Prepare inputs
            if isinstance(batch, dict):
                inputs = {k: v.to(device_obj) for k, v in batch.items()}
                if "labels" not in inputs and "input_ids" in inputs:
                    inputs["labels"] = inputs["input_ids"]
            else:
                # Assume the batch is a tuple with (input_ids, attention_mask)
                input_ids = batch[0].to(device_obj)
                attention_mask = batch[1].to(device_obj) if len(batch) > 1 else None
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": input_ids  # Use input_ids as labels for LM
                }
            
            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss
            
            # Update metrics
            batch_size = inputs["input_ids"].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    # Calculate metrics
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity

def _create_differential_param_groups(
    model: nn.Module, 
    base_lr: float
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with different learning rates for different model components.
    
    Args:
        model: The model to create parameter groups for
        base_lr: Base learning rate to scale from
        
    Returns:
        List of parameter dictionaries for the optimizer
    """
    # Categorize parameters
    attention_params = []
    ln_params = []
    ffn_params = []
    other_params = []
    
    # Group parameters by their role
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if any(x in name.lower() for x in ["attn", "attention"]):
            attention_params.append(param)
        elif any(x in name.lower() for x in ["ln", "layernorm", "layer_norm"]):
            ln_params.append(param)
        elif any(x in name.lower() for x in ["mlp", "ffn", "feed_forward"]):
            ffn_params.append(param)
        else:
            other_params.append(param)
    
    # Create parameter groups with different learning rates
    param_groups = [
        {"params": attention_params, "lr": base_lr * 0.8, "name": "attention"},  # Lower LR for attention
        {"params": ln_params, "lr": base_lr * 0.5, "name": "layernorm"},  # Even lower for LayerNorm
        {"params": ffn_params, "lr": base_lr * 1.2, "name": "ffn"},  # Higher for FFN
        {"params": other_params, "lr": base_lr, "name": "other"}  # Base LR for other params
    ]
    
    # Log parameter group sizes
    for group in param_groups:
        param_count = sum(p.numel() for p in group["params"])
        logger.info(f"Parameter group '{group['name']}': {param_count/1e6:.2f}M parameters, LR: {group['lr']}")
    
    return param_groups