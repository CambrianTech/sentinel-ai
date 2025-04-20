"""
Neural Plasticity Experiment Runner

This module provides a unified experiment runner for neural plasticity experiments
that can be used in both Colab and local environments.

Version: v0.0.56 (2025-04-19 23:30:00)
"""

import os
import sys
import json
import torch
import numpy as np
import platform
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from datetime import datetime
from tqdm import tqdm

# Check for Apple Silicon at module import time
IS_APPLE_SILICON = False
try:
    if platform.system() == "Darwin" and platform.processor() == "arm":
        IS_APPLE_SILICON = True
        print("🍎 Apple Silicon detected - enabling PyTorch/BLAS crash prevention")
        
        # Force single-threaded BLAS operations
        import os
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        
        # Try to force PyTorch to behave on Apple Silicon
        try:
            # Disable parallel CPU operations
            torch.set_num_threads(1)
            # Ensure the default device is CPU
            if torch.cuda.is_available():
                print("⚠️ CUDA detected on Apple Silicon - forcing CPU usage to prevent crashes")
                torch.__future__.set_overwrite_module_params_on_conversion(True)
        except (ImportError, AttributeError):
            pass
        
        # Configure matplotlib
        try:
            import matplotlib
            matplotlib.use('Agg')
            print("🎨 Switching to Agg matplotlib backend for improved stability")
        except (ImportError, RuntimeError):
            pass
except (ImportError, AttributeError):
    pass

# Import our core neural plasticity functions
from utils.neural_plasticity.core import (
    calculate_head_entropy,
    generate_pruning_mask,
    apply_pruning_mask,
    evaluate_model
)

def get_dataloader_builder(batch_size=4):
    """
    Create a function that returns train and evaluation dataloaders.
    Uses a simple dataset for testing purposes.
    """
    from transformers import AutoTokenizer
    import torch
    
    # Create synthetic data
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a world where technology dominates, humans seek connection.",
        "Once upon a time, there lived a wise king who ruled with compassion.",
        "The history of artificial intelligence dates back to ancient myths.",
        "Climate change is affecting ecosystems worldwide, leading to rising sea levels.",
        "The transformer architecture revolutionized natural language processing tasks.",
        "Neural plasticity allows models to adapt their structure during training.",
        "Deep learning models can recognize patterns in complex data.",
        "The attention mechanism focuses on different parts of the input sequence.",
        "Language models predict the next token based on previous context."
    ] * 10  # Repeat to create more samples
    
    def build_dataloaders(model_name, batch_size=batch_size):
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenize
        from torch.utils.data import TensorDataset, DataLoader
        
        encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        
        dataset = TensorDataset(input_ids, attention_mask)
        
        # Split into train and eval
        train_size = int(0.8 * len(dataset))
        eval_size = len(dataset) - train_size
        
        train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
        
        return train_dataloader, eval_dataloader, tokenizer
    
    # Return a function that will create dataloaders with the specified batch size
    return build_dataloaders

def create_model_and_entropy_baseline(model_name, device="cpu", warmup_steps=10):
    """Load model and create entropy baseline"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    print(f"Loading model: {model_name}...")
    
    # Use CPU if running on Apple Silicon
    if IS_APPLE_SILICON and device != "cpu":
        print("⚠️ Forcing CPU usage on Apple Silicon")
        device = "cpu"
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Move model to device
    model = model.to(device)
    
    # Create dummy input for entropy baseline
    dummy_input = tokenizer("This is a test", return_tensors="pt").to(device)
    
    # Put model in eval mode for baseline measurements
    model.eval()
    
    # Create attention maps for entropy baseline
    print("Running warmup iterations to establish baseline entropy...")
    
    # Perform warmup runs
    attention_maps = {}
    
    # Use a simple warmup loop to gather attention maps
    for i in range(warmup_steps):
        with torch.no_grad():
            outputs = model(**dummy_input, output_attentions=True)
        
        if hasattr(outputs, 'attentions') and outputs.attentions:
            # Process attentions from the model
            for layer_idx, layer_attention in enumerate(outputs.attentions):
                # Create or update layer attention maps
                if layer_idx not in attention_maps:
                    attention_maps[layer_idx] = []
                attention_maps[layer_idx].append(layer_attention.detach())
    
    # Calculate baseline entropy
    baseline_entropy = {}
    for layer_idx, layer_maps in attention_maps.items():
        # Stack maps for this layer
        stacked_maps = torch.cat(layer_maps, dim=0)
        layer_entropy = calculate_head_entropy(stacked_maps)
        baseline_entropy[layer_idx] = layer_entropy
    
    # Calculate perplexity on a simple input
    with torch.no_grad():
        outputs = model(**dummy_input, labels=dummy_input['input_ids'])
        baseline_loss = outputs.loss.item()
        baseline_perplexity = torch.exp(torch.tensor(baseline_loss)).item()
    
    metrics = {
        "baseline_entropy": baseline_entropy,
        "baseline_loss": baseline_loss,
        "baseline_perplexity": baseline_perplexity
    }
    
    print(f"Baseline perplexity: {baseline_perplexity:.4f}")
    
    # Create a simple visualization of the baseline entropy
    plot_baseline_entropy(baseline_entropy)
    
    return model, tokenizer, metrics

def plot_baseline_entropy(baseline_entropy):
    """Create a visualization of the baseline entropy"""
    # Convert dictionary to tensor for visualization
    layers = sorted(baseline_entropy.keys())
    
    # Determine shape of entropy values
    sample_entropy = baseline_entropy[layers[0]]
    
    # Handle multi-dimensional tensors
    if sample_entropy.dim() > 1:
        # For tensors with more than 1 dimension, we need to reduce extra dimensions
        num_heads = sample_entropy.size(0)
        entropy_tensor = torch.zeros((len(layers), num_heads))
        
        # Fill tensor with entropy values, reducing extra dimensions if needed
        for i, layer in enumerate(layers):
            layer_entropy = baseline_entropy[layer]
            if layer_entropy.dim() > 1:
                # Reduce to 1D by taking mean across extra dimensions
                reduced_entropy = layer_entropy.mean(dim=tuple(range(1, layer_entropy.dim())))
                entropy_tensor[i] = reduced_entropy
            else:
                entropy_tensor[i] = layer_entropy
    else:
        # Simple case: 1D or 0D tensors
        heads = 1
        entropy_tensor = torch.zeros((len(layers), heads))
        
        # Fill tensor with entropy values
        for i, layer in enumerate(layers):
            # Handle both 1D and 0D entropy values
            if baseline_entropy[layer].dim() > 0:
                entropy_tensor[i, 0] = baseline_entropy[layer].mean()
            else:
                entropy_tensor[i, 0] = baseline_entropy[layer]
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    im = plt.imshow(entropy_tensor.cpu().numpy(), cmap='viridis')
    plt.colorbar(im, label='Entropy')
    plt.xlabel('Head Index')
    plt.ylabel('Layer Index')
    plt.title('Baseline Attention Entropy')
    
    # Add text annotations
    for i in range(entropy_tensor.shape[0]):
        for j in range(entropy_tensor.shape[1]):
            plt.text(j, i, f'{entropy_tensor[i, j]:.2f}',
                    ha="center", va="center", color="w")
    
    # Save figure
    os.makedirs('test_output', exist_ok=True)
    plt.savefig('test_output/baseline_entropy.png')
    plt.close()
    
    print(f"✅ Saved baseline entropy visualization to test_output/baseline_entropy.png")

def run_neural_plasticity_experiment(
    model_name: str,
    device: str = "cpu",
    output_dir: str = "test_output/neural_plasticity",
    pruning_strategy: str = "entropy",
    prune_percent: float = 0.2,
    warmup_steps: int = 10,
    training_steps: int = 50,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    seed: int = 42,
    colab_mode: bool = False
):
    """
    Run a complete neural plasticity experiment.
    
    Args:
        model_name: Name of the model to use
        device: Device to use ('cpu' or 'cuda')
        output_dir: Directory to save results
        pruning_strategy: Strategy for pruning ('entropy' or 'magnitude')
        prune_percent: Percentage of heads to prune (0-1)
        warmup_steps: Number of warmup steps for baseline
        training_steps: Number of training steps
        batch_size: Batch size for training
        learning_rate: Learning rate for fine-tuning
        seed: Random seed for reproducibility
        colab_mode: Whether running in Colab (affects visualization)
    
    Returns:
        Dictionary of results
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Use CPU on Apple Silicon regardless of requested device
    if IS_APPLE_SILICON and device != "cpu":
        print("⚠️ Apple Silicon detected, forcing CPU usage")
        device = "cpu"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup experiment environment
    print(f"\n=== Neural Plasticity Experiment ===")
    print(f"Model: {model_name}")
    print(f"Pruning strategy: {pruning_strategy}")
    print(f"Pruning level: {prune_percent:.2f}")
    print(f"Training steps: {training_steps}")
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    
    # Get dataloader builder
    dataloader_builder = get_dataloader_builder(batch_size)
    
    # 1. Create model and establish baseline
    model, tokenizer, baseline_metrics = create_model_and_entropy_baseline(
        model_name, 
        device=device,
        warmup_steps=warmup_steps
    )
    
    # Save baseline metrics
    with open(os.path.join(output_dir, "baseline_metrics.json"), 'w') as f:
        # Convert tensors to lists for JSON serialization
        serializable_metrics = {}
        for key, value in baseline_metrics.items():
            if key == "baseline_entropy":
                serializable_entropy = {}
                for layer, entropy in value.items():
                    serializable_entropy[str(layer)] = entropy.cpu().numpy().tolist()
                serializable_metrics[key] = serializable_entropy
            else:
                serializable_metrics[key] = value
        json.dump(serializable_metrics, f, indent=2)
    
    # 2. Generate pruning mask based on entropy
    print(f"\nGenerating pruning mask using {pruning_strategy} strategy...")
    
    # Get entropy values and gradient norms
    entropy_values = {}
    for key, value in baseline_metrics["baseline_entropy"].items():
        entropy_values[key] = value
    
    # Process entropy into a single tensor for visualization and pruning
    num_layers = len(entropy_values)
    
    # Determine number of heads
    sample_entropy = entropy_values[0]
    if sample_entropy.dim() > 0:
        num_heads = sample_entropy.size(0)
    else:
        num_heads = 1
    
    # Stack entropy values for all layers
    entropy_tensor = torch.zeros((num_layers, num_heads))
    for layer_idx, layer in enumerate(sorted(entropy_values.keys())):
        entropy = entropy_values[layer]
        if entropy.dim() > 0:
            # For multi-dimensional entropy, use the first dimension
            if entropy.dim() > 1:
                # Reduce to 1D by taking mean across extra dimensions
                reduced_entropy = entropy.mean(dim=tuple(range(1, entropy.dim())))
                entropy_tensor[layer_idx] = reduced_entropy
            else:
                entropy_tensor[layer_idx] = entropy
        else:
            entropy_tensor[layer_idx, 0] = entropy
    
    # Use gradient norms as a placeholder for actual gradients
    # In a real implementation, we'd calculate actual gradients using dataloader
    grad_norms = torch.ones_like(entropy_tensor)
    
    # Generate pruning mask
    pruning_mask = generate_pruning_mask(
        grad_norm_values=grad_norms,
        entropy_values=entropy_tensor,
        prune_percent=prune_percent,
        strategy=pruning_strategy
    )
    
    # Visualize pruning mask
    plt.figure(figsize=(12, 8))
    plt.imshow(pruning_mask.cpu().numpy(), cmap='binary')
    plt.title(f'Pruning Mask ({pruning_strategy} strategy, {prune_percent:.2f} prune ratio)')
    plt.xlabel('Head Index')
    plt.ylabel('Layer Index')
    plt.colorbar(label='Pruned (1) / Active (0)')
    plt.savefig(os.path.join(output_dir, 'pruning_mask.png'))
    plt.close()
    
    # 3. Apply pruning to model
    print(f"\nApplying pruning to model...")
    pruned_heads = apply_pruning_mask(model, pruning_mask)
    
    print(f"Pruned {len(pruned_heads)} heads out of {pruning_mask.numel()}")
    
    # Display some pruned heads
    if pruned_heads:
        head_list = ", ".join([f"L{l}H{h}" for l, h in pruned_heads[:5]])
        if len(pruned_heads) > 5:
            head_list += f", and {len(pruned_heads)-5} more"
        print(f"Pruned heads: {head_list}")
    
    # 4. Evaluate pruned model
    print(f"\nEvaluating pruned model...")
    train_dataloader, eval_dataloader, _ = dataloader_builder(model_name)
    
    # Create dummy input for evaluation
    sample_batch = next(iter(eval_dataloader))
    input_ids = sample_batch[0].to(device)
    
    # Put model in eval mode
    model.eval()
    
    # Evaluate model
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        pruned_loss = outputs.loss.item()
        pruned_perplexity = torch.exp(torch.tensor(pruned_loss)).item()
    
    print(f"Pruned model perplexity: {pruned_perplexity:.4f}")
    
    # 5. Train pruned model (fine-tuning)
    print(f"\nFine-tuning pruned model for {training_steps} steps...")
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    training_metrics = {
        "step": [],
        "loss": [],
        "perplexity": [],
        "eval_loss": [],
        "eval_perplexity": []
    }
    
    # Create a progress bar
    progress_bar = tqdm(range(training_steps), desc="Training")
    
    # Main training loop
    for step in range(training_steps):
        # Get batch
        batch = next(iter(train_dataloader))
        input_ids = batch[0].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log metrics
        training_metrics["step"].append(step)
        training_metrics["loss"].append(loss.item())
        training_metrics["perplexity"].append(torch.exp(loss).item())
        
        # Evaluate occasionally
        if step % 10 == 0 or step == training_steps - 1:
            model.eval()
            with torch.no_grad():
                eval_batch = next(iter(eval_dataloader))
                eval_input_ids = eval_batch[0].to(device)
                eval_outputs = model(input_ids=eval_input_ids, labels=eval_input_ids)
                eval_loss = eval_outputs.loss.item()
                eval_perplexity = torch.exp(torch.tensor(eval_loss)).item()
                
                training_metrics["eval_loss"].append(eval_loss)
                training_metrics["eval_perplexity"].append(eval_perplexity)
                
                # Log to progress bar
                progress_bar.set_postfix({
                    "train_loss": f"{loss.item():.4f}",
                    "eval_loss": f"{eval_loss:.4f}",
                    "eval_ppl": f"{eval_perplexity:.2f}"
                })
            model.train()
        
        # Update progress bar
        progress_bar.update(1)
    
    # Close progress bar
    progress_bar.close()
    
    # 6. Evaluate fine-tuned model
    print(f"\nEvaluating fine-tuned model...")
    model.eval()
    
    # Evaluate on the same input as before
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        finetuned_loss = outputs.loss.item()
        finetuned_perplexity = torch.exp(torch.tensor(finetuned_loss)).item()
    
    print(f"Fine-tuned model perplexity: {finetuned_perplexity:.4f}")
    
    # 7. Get post-training entropy
    print(f"\nCalculating post-training entropy...")
    
    # Create attention maps for post-training entropy
    attention_maps = {}
    
    # Use a simple loop to gather attention maps
    for i in range(warmup_steps):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True)
        
        if hasattr(outputs, 'attentions') and outputs.attentions:
            # Process attentions from the model
            for layer_idx, layer_attention in enumerate(outputs.attentions):
                # Create or update layer attention maps
                if layer_idx not in attention_maps:
                    attention_maps[layer_idx] = []
                attention_maps[layer_idx].append(layer_attention.detach())
    
    # Calculate post-training entropy
    post_entropy = {}
    for layer_idx, layer_maps in attention_maps.items():
        # Stack maps for this layer
        stacked_maps = torch.cat(layer_maps, dim=0)
        layer_entropy = calculate_head_entropy(stacked_maps)
        post_entropy[layer_idx] = layer_entropy
    
    # 8. Create visualizations
    print(f"\nCreating visualizations...")
    
    # Plot training metrics
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(training_metrics["step"], training_metrics["loss"])
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot training perplexity
    plt.subplot(2, 2, 2)
    plt.plot(training_metrics["step"], training_metrics["perplexity"])
    plt.title('Training Perplexity')
    plt.xlabel('Step')
    plt.ylabel('Perplexity')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot evaluation loss
    plt.subplot(2, 2, 3)
    eval_steps = [step for step in training_metrics["step"] if step % 10 == 0 or step == training_steps - 1]
    if len(eval_steps) == len(training_metrics["eval_loss"]):
        plt.plot(eval_steps, training_metrics["eval_loss"])
        plt.title('Evaluation Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
    else:
        plt.text(0.5, 0.5, 'Insufficient evaluation data', ha='center', va='center')
        plt.title('Evaluation Loss (Not Available)')
    
    # Plot evaluation perplexity
    plt.subplot(2, 2, 4)
    if len(eval_steps) == len(training_metrics["eval_perplexity"]):
        plt.plot(eval_steps, training_metrics["eval_perplexity"])
        plt.title('Evaluation Perplexity')
        plt.xlabel('Step')
        plt.ylabel('Perplexity')
        plt.grid(True, linestyle='--', alpha=0.7)
    else:
        plt.text(0.5, 0.5, 'Insufficient evaluation data', ha='center', va='center')
        plt.title('Evaluation Perplexity (Not Available)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()
    
    # Plot pre-post entropy comparison
    num_layers = len(post_entropy)
    num_heads = post_entropy[0].shape[0] if post_entropy[0].dim() > 0 else 1
    
    # Stack entropy values for all layers (pre and post)
    pre_entropy_tensor = torch.zeros((num_layers, num_heads))
    post_entropy_tensor = torch.zeros((num_layers, num_heads))
    
    # Process pre-training entropy
    for layer_idx, layer in enumerate(sorted(baseline_metrics["baseline_entropy"].keys())):
        if layer < num_layers:  # Ensure we don't go out of bounds
            pre_entropy = baseline_metrics["baseline_entropy"][layer]
            if pre_entropy.dim() > 0:
                # For multi-dimensional entropy, use the first dimension
                if pre_entropy.dim() > 1:
                    # Reduce to 1D by taking mean across extra dimensions
                    reduced_entropy = pre_entropy.mean(dim=tuple(range(1, pre_entropy.dim())))
                    pre_entropy_tensor[layer_idx] = reduced_entropy
                else:
                    pre_entropy_tensor[layer_idx] = pre_entropy
            else:
                pre_entropy_tensor[layer_idx, 0] = pre_entropy
    
    # Process post-training entropy            
    for layer_idx, layer in enumerate(sorted(post_entropy.keys())):
        if layer < num_layers:  # Ensure we don't go out of bounds
            post_entropy_val = post_entropy[layer]
            if post_entropy_val.dim() > 0:
                # For multi-dimensional entropy, use the first dimension
                if post_entropy_val.dim() > 1:
                    # Reduce to 1D by taking mean across extra dimensions
                    reduced_entropy = post_entropy_val.mean(dim=tuple(range(1, post_entropy_val.dim())))
                    post_entropy_tensor[layer_idx] = reduced_entropy
                else:
                    post_entropy_tensor[layer_idx] = post_entropy_val
            else:
                post_entropy_tensor[layer_idx, 0] = post_entropy_val
    
    # Calculate deltas
    delta_entropy_tensor = post_entropy_tensor - pre_entropy_tensor
    
    # Create figures
    plt.figure(figsize=(18, 6))
    
    # Pre-training entropy
    plt.subplot(1, 3, 1)
    im = plt.imshow(pre_entropy_tensor.cpu().numpy(), cmap='viridis')
    plt.colorbar(im, label='Entropy')
    plt.title('Pre-training Entropy')
    plt.xlabel('Head Index')
    plt.ylabel('Layer Index')
    
    # Post-training entropy
    plt.subplot(1, 3, 2)
    im = plt.imshow(post_entropy_tensor.cpu().numpy(), cmap='viridis')
    plt.colorbar(im, label='Entropy')
    plt.title('Post-training Entropy')
    plt.xlabel('Head Index')
    plt.ylabel('Layer Index')
    
    # Delta entropy
    plt.subplot(1, 3, 3)
    im = plt.imshow(delta_entropy_tensor.cpu().numpy(), cmap='coolwarm')
    plt.colorbar(im, label='Entropy Change')
    plt.title('Entropy Change (Post - Pre)')
    plt.xlabel('Head Index')
    plt.ylabel('Layer Index')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_comparison.png'))
    plt.close()
    
    # 9. Plot metrics comparison for before/after pruning
    # Create bar chart of perplexity values for baseline, pruned, and fine-tuned models
    plt.figure(figsize=(10, 6))
    
    stages = ['Baseline', 'After Pruning', 'After Fine-tuning']
    perplexities = [
        baseline_metrics["baseline_perplexity"],
        pruned_perplexity,
        finetuned_perplexity
    ]
    
    # Calculate percent changes
    pruned_change = ((pruned_perplexity / baseline_metrics["baseline_perplexity"]) - 1) * 100
    finetuned_change = ((finetuned_perplexity / baseline_metrics["baseline_perplexity"]) - 1) * 100
    
    # Create bar chart
    bars = plt.bar(stages, perplexities, color=['green', 'red', 'blue'])
    
    # Add value labels
    for i, (bar, ppl) in enumerate(zip(bars, perplexities)):
        height = bar.get_height()
        if i == 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{ppl:.2f}', ha='center', va='bottom')
        elif i == 1:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{ppl:.2f}\n({pruned_change:+.1f}%)', ha='center', va='bottom')
        else:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{ppl:.2f}\n({finetuned_change:+.1f}%)', ha='center', va='bottom')
    
    plt.ylabel('Perplexity (lower is better)')
    plt.title('Model Perplexity Through Plasticity Cycle')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'perplexity_comparison.png'))
    plt.close()
    
    # 10. Collect results
    results = {
        "model_name": model_name,
        "pruning_strategy": pruning_strategy,
        "prune_percent": prune_percent,
        "training_steps": training_steps,
        "metrics": {
            "baseline": {
                "perplexity": baseline_metrics["baseline_perplexity"],
                "loss": baseline_metrics["baseline_loss"]
            },
            "post_pruning": {
                "perplexity": pruned_perplexity,
                "loss": pruned_loss,
                "pruned_heads": len(pruned_heads)
            },
            "final": {
                "perplexity": finetuned_perplexity,
                "loss": finetuned_loss
            }
        },
        "recovery_rate": (
            1.0 - max(0, finetuned_perplexity - baseline_metrics["baseline_perplexity"]) / 
            max(0.001, pruned_perplexity - baseline_metrics["baseline_perplexity"])
        )
    }
    
    # Save results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 11. Generate some text with the final model
    print(f"\nGenerating text with fine-tuned model...")
    
    prompts = [
        "The neural network model",
        "Artificial intelligence can",
        "Neural plasticity enables"
    ]
    
    generated_texts = []
    
    model.eval()
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                inputs["input_ids"],
                max_length=30,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )
            
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_texts.append({
            "prompt": prompt,
            "generated": generated_text
        })
        
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print()
    
    # Save generated texts
    with open(os.path.join(output_dir, 'generated_texts.json'), 'w') as f:
        json.dump(generated_texts, f, indent=2)
    
    # 12. Print summary
    print(f"\nNeural Plasticity Experiment Summary:")
    print(f"Model: {model_name}")
    print(f"Pruning: {pruning_strategy} at {prune_percent*100:.1f}% level")
    print(f"Pruned {len(pruned_heads)} heads ({len(pruned_heads)/pruning_mask.numel()*100:.1f}% of total)")
    print(f"Perplexity:")
    print(f"- Baseline: {baseline_metrics['baseline_perplexity']:.4f}")
    print(f"- After Pruning: {pruned_perplexity:.4f} ({pruned_change:+.1f}%)")
    print(f"- After Fine-tuning: {finetuned_perplexity:.4f} ({finetuned_change:+.1f}%)")
    
    recovery_rate = results["recovery_rate"] * 100
    print(f"Recovery rate: {recovery_rate:.1f}%")
    
    # Determine success level
    if finetuned_perplexity <= baseline_metrics["baseline_perplexity"] * 1.05:
        success_message = "SUCCESS! Model recovered fully after pruning."
    elif finetuned_perplexity <= baseline_metrics["baseline_perplexity"] * 1.15:
        success_message = "PARTIAL SUCCESS. Model recovered moderately well after pruning."
    else:
        success_message = "LIMITED SUCCESS. Model showed limited recovery after pruning."
    
    print(f"\nConclusion: {success_message}")
    print(f"Visualizations saved to: {output_dir}")
    
    return results