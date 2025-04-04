# utils/training.py

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from .train_utils import compute_loss, compute_perplexity, evaluate_model, adjust_learning_rates, prune_inactive_heads, count_active_heads, count_trainable_params
from .checkpoint import save_checkpoint

from .head_metrics import (
    compute_attention_entropy,
    compute_head_importance,
    compute_gradient_norms,
    visualize_head_metrics,
    recommend_pruning_growth
)
from controller.controller_ann import ANNController
from .dynamic_architecture import integrate_dynamic_architecture

def train_model(
    model, tokenizer, train_ids, val_ids, device,
    epochs=1, batch_size=8, lr=5e-5, baseline_model=None,
    checkpoint_path="checkpoint.pth", 
    use_controller=True,
    controller_update_freq=500,
    metrics_visualization_path=None,
    use_dynamic_architecture=True,
    dynamic_arch_config=None
):
    """
    Train the adaptive transformer model with sentinel gates.
    
    Args:
        model: The adaptive transformer model
        tokenizer: Tokenizer for the model
        train_ids: Training data token IDs
        val_ids: Validation data token IDs
        device: Device to train on
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        baseline_model: Optional baseline model for comparison
        checkpoint_path: Path to save checkpoints
        use_controller: Whether to use the ANN controller for dynamic head management
        controller_update_freq: How often to update controller (steps)
        metrics_visualization_path: Optional path to save head metrics visualizations
        use_dynamic_architecture: Whether to use dynamic architecture adjustment during training
        dynamic_arch_config: Configuration dict for the dynamic architecture manager
    """
    # Initialize model for training
    model.train()
    
    # Create optimizer with per-parameter options
    optimizer = torch.optim.AdamW([
        # Main model parameters
        {'params': [p for n, p in model.named_parameters() 
                   if 'gate' not in n and p.requires_grad], 'lr': lr},
        # Gate parameters - use lower learning rate for stability
        {'params': [p for n, p in model.named_parameters() 
                  if 'gate' in n and p.requires_grad], 'lr': lr * 0.1}
    ])
    
    # Learning rate multipliers for fine-grained control
    head_lr_multipliers = {(layer_idx, head_idx): 1.0 
                         for layer_idx, block in enumerate(model.blocks)
                         for head_idx in range(block["attn"].num_heads)}

    # Create data loaders
    train_dataset = TensorDataset(torch.tensor(train_ids))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(torch.tensor(val_ids))
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
    val_input_ids = torch.tensor(val_ids).to(device)
    
    # Initialize controller if using
    if use_controller:
        num_layers = len(model.blocks)
        num_heads = model.blocks[0]["attn"].num_heads  # Assuming uniform heads per layer
        controller = ANNController(num_layers, num_heads, 
                                 config={"init_value": 3.0, "reg_weight": 1e-4})
        controller.to(device)
    
    # Training loop
    global_step = 0
    best_val_perplexity = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0.0
        reg_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            global_step += 1
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Get inputs
            input_ids = batch[0].to(device)

            # Forward pass
            logits = model(input_ids)
            
            # Compute main loss
            main_loss = compute_loss(logits, input_ids)
            
            # Add regularization if using controller
            if use_controller:
                # Gate regularization to encourage sparsity
                gate_reg = controller.regularization_loss()
                
                # Weighted sum of losses
                loss = main_loss + controller.reg_weight * gate_reg
                reg_loss += controller.reg_weight * gate_reg.item()
            else:
                loss = main_loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update metrics
            total_loss += main_loss.item()
            progress_bar.set_postfix(
                loss=main_loss.item(),
                reg=controller.reg_weight * gate_reg.item() if use_controller else 0.0
            )
            
            # Controller updates
            if use_controller and global_step % controller_update_freq == 0:
                # Compute metrics for controller
                with torch.no_grad():
                    # Get entropy for each attention head
                    entropy_dict = compute_attention_entropy(model, batch_size=4, seq_len=64, device=device)
                    
                    # For more accurate metrics, use a subset of validation data
                    if global_step % (controller_update_freq * 5) == 0:
                        # Compute head importance and gradient norms (more expensive)
                        importance_dict = compute_head_importance(
                            model, val_loader, compute_loss, device=device)
                        
                        grad_norms_dict = compute_gradient_norms(
                            model, train_loader, compute_loss, optimizer, device=device)
                        
                        # Create visualizations if requested
                        if metrics_visualization_path:
                            viz_path = f"{metrics_visualization_path}_step{global_step}.png"
                            visualize_head_metrics(
                                entropy_dict, importance_dict, grad_norms_dict, save_path=viz_path)
                        
                        # Get pruning and growth recommendations
                        prune_candidates, grow_candidates = recommend_pruning_growth(
                            entropy_dict, importance_dict, grad_norms_dict)
                        
                        print(f"\nStep {global_step} - Controller recommendations:")
                        print(f"  Prune candidates: {prune_candidates}")
                        print(f"  Growth candidates: {grow_candidates}")
                    else:
                        # Use just entropy for more frequent, lightweight updates
                        importance_dict = None
                        grad_norms_dict = None
                
                # Apply metrics-based updates
                metrics_dict = {
                    'entropy': torch.stack([entropy_dict[layer_idx] 
                                           for layer_idx in range(len(model.blocks))]),
                }
                
                if importance_dict is not None:
                    metrics_dict['importance'] = torch.stack(
                        [importance_dict[layer_idx] for layer_idx in range(len(model.blocks))])
                
                if grad_norms_dict is not None:
                    metrics_dict['grad_norm'] = torch.stack(
                        [grad_norms_dict[layer_idx] for layer_idx in range(len(model.blocks))])
                
                # Update controller based on metrics
                controller.update_gates(metrics_dict)
                
                # Apply updated gates to model
                current_gates = controller.forward()
                with torch.no_grad():
                    for layer_idx, block in enumerate(model.blocks):
                        block["attn"].gate.copy_(current_gates[layer_idx])
            
            # Dynamic architecture updates (pruning and growing)
            if use_dynamic_architecture and global_step % 1000 == 0:
                # Set default configuration if not provided
                if dynamic_arch_config is None:
                    dynamic_arch_config = {
                        'prune_frequency': 2000,
                        'grow_frequency': 5000,
                        'max_prune_per_step': 2,
                        'max_grow_per_step': 2,
                        'min_heads_per_layer': 4,
                        'max_heads_per_layer': 16,
                        'importance_threshold': 0.05,
                        'entropy_threshold': 1.5,
                        'performance_margin': 0.05
                    }
                
                # Integrate dynamic architecture management
                model, optimizer, head_lr_multipliers, arch_updated = integrate_dynamic_architecture(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    loss_fn=compute_loss,
                    optimizer=optimizer,
                    head_lr_multipliers=head_lr_multipliers,
                    global_step=global_step,
                    device=device,
                    **dynamic_arch_config
                )
                
                # If architecture was updated, save a special checkpoint
                if arch_updated:
                    arch_checkpoint_path = f"{checkpoint_path}.arch_step{global_step}"
                    save_checkpoint(arch_checkpoint_path, model, optimizer, 
                                  head_lr_multipliers, epoch+1, global_step)
        
        # End of epoch evaluations
        avg_train_loss = total_loss / len(train_loader)
        avg_reg_loss = reg_loss / len(train_loader) if reg_loss > 0 else 0.0

        # Evaluate on validation set
        val_loss, val_perplexity, baseline_perplexity = evaluate_model(
            model, val_input_ids, val_input_ids, baseline_model=baseline_model
        )

        # Count active heads and parameters
        active_heads = count_active_heads(model)
        param_count = count_trainable_params(model)

        # Report metrics
        print(f"\nEpoch {epoch+1}/{epochs} completed:")
        print(f"  Avg Train Loss: {avg_train_loss:.4f} (+ {avg_reg_loss:.4f} reg)")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Validation Perplexity: {val_perplexity:.2f}")
        if baseline_perplexity:
            print(f"  Baseline Perplexity: {baseline_perplexity:.2f}")
            perf_vs_baseline = (baseline_perplexity - val_perplexity) / baseline_perplexity * 100
            print(f"  Performance vs Baseline: {perf_vs_baseline:.2f}%")
        print(f"  Active Heads: {active_heads}")
        print(f"  Trainable Parameters: {param_count:,}")
        
        # Track best validation performance
        if val_perplexity < best_val_perplexity:
            best_val_perplexity = val_perplexity
            print(f"  New best validation perplexity! Saving checkpoint...")
            save_checkpoint(f"{checkpoint_path}.best", model, optimizer, 
                          head_lr_multipliers, epoch+1, global_step)

        # Adjust learning rates for different heads based on activity
        adjust_learning_rates(optimizer, head_lr_multipliers, model, lr)
        
        # Prune heads with gate values near zero
        prune_inactive_heads(model)

        # Save regular checkpoint
        save_checkpoint(checkpoint_path, model, optimizer, 
                      head_lr_multipliers, epoch+1, global_step)

    print("\nTraining complete! 🎉")
    print(f"Best validation perplexity: {best_val_perplexity:.2f}")
    if baseline_perplexity:
        perf_vs_baseline = (baseline_perplexity - best_val_perplexity) / baseline_perplexity * 100
        print(f"Performance vs Baseline: {perf_vs_baseline:.2f}%")
    print(f"Final active heads: {count_active_heads(model)}")
    print(f"Parameter efficiency: {count_trainable_params(model):,} parameters")


def compute_loss(logits, targets):
    logits = logits[:, :-1, :].contiguous()
    targets = targets[:, 1:].contiguous()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

def evaluate_model(model, input_ids, targets, baseline_model=None):
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        val_loss = compute_loss(logits, targets)
        val_perplexity = math.exp(val_loss.item())

        baseline_perplexity = None
        if baseline_model:
            baseline_logits = baseline_model(input_ids).logits
            baseline_loss = compute_loss(baseline_logits, targets)
            baseline_perplexity = math.exp(baseline_loss.item())

    model.train()
    return val_loss.item(), val_perplexity, baseline_perplexity

def count_active_heads(model, threshold=1e-2):
    active_heads = 0
    for block in model.blocks:
        attn = block["attn"]
        active_heads += sum(float(g) > threshold for g in attn.gate)
    return active_heads

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def adjust_learning_rates(optimizer, head_lr_multipliers, model, base_lr):
    for layer_idx, block in enumerate(model.blocks):
        attn = block["attn"]
        for head_idx in range(attn.num_heads):
            g = float(attn.gate[head_idx].item())
            key = (layer_idx, head_idx)
            if g < 0.1:
                head_lr_multipliers[key] = min(head_lr_multipliers[key] * 1.2, 5.0)
            elif g > 0.9:
                head_lr_multipliers[key] = max(head_lr_multipliers[key] * 0.8, 0.1)
            group_index = layer_idx * attn.num_heads + head_idx
            optimizer.param_groups[group_index]['lr'] = base_lr * head_lr_multipliers[key]

def prune_inactive_heads(model, threshold=1e-3):
    for block in model.blocks:
        attn = block["attn"]
        for head_idx in range(attn.num_heads):
            if attn.gate[head_idx] < threshold:
                attn.gate.data[head_idx] = 0.0
                for param in attn.W_q[head_idx].parameters(): param.requires_grad = False
                for param in attn.W_k[head_idx].parameters(): param.requires_grad = False
                for param in attn.W_v[head_idx].parameters(): param.requires_grad = False
                for param in attn.W_o[head_idx].parameters(): param.requires_grad = False
