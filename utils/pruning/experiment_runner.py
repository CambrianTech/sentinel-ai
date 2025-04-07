"""Experiment runner for pruning and fine-tuning experiments

This module provides a simplified API for running pruning and fine-tuning experiments.
It handles all the complexity of loading models, datasets, and running the experiments.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up logging
import logging
logger = logging.getLogger(__name__)

# Try to import from the API package
try:
    # First, try the modular API
    from .api.data import load_wikitext, prepare_data, prepare_test_data
    from .api.pruning import compute_head_importance, prune_heads, fine_tune, evaluate_model
    from .api.entropy import collect_attention_distributions, entropy_based_pruning
    USING_FULL_MODULAR_API = True
    logger.info("Using full modular API")
except ImportError:
    # Fall back to simplified imports if needed
    try:
        from utils.pruning.api.data import load_wikitext, prepare_data, prepare_test_data
        from utils.pruning.api.pruning import compute_head_importance, prune_heads, fine_tune, evaluate_model
        # If we don't have entropy module, we'll use simplified implementation
        try:
            from utils.pruning.api.entropy import collect_attention_distributions, entropy_based_pruning
            USING_FULL_MODULAR_API = True
        except ImportError:
            # We'll use local implementations defined below
            USING_FULL_MODULAR_API = False
    except ImportError:
        # Fall back to minimal implementation
        USING_FULL_MODULAR_API = False
        logger.info("Using minimal implementation")

# Create a simple configuration class
@dataclass
class ExperimentConfig:
    """Configuration for a pruning and fine-tuning experiment"""
    model_name: str = "distilgpt2"
    pruning_percent: float = 0.3
    pruning_strategy: str = "entropy"
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    max_length: int = 128
    device: torch.device = None
    output_dir: str = "pruning_results"
    use_test_data: bool = False
    num_samples: int = 100
    prompt: str = "Once upon a time"
    
    def __post_init__(self):
        # Set device if not provided
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    def __str__(self):
        return (f"ExperimentConfig(model={self.model_name}, "
                f"pruning={self.pruning_percent}, "
                f"strategy={self.pruning_strategy}, "
                f"epochs={self.num_epochs})")


# Local implementation of collect_attention_distributions if needed
def local_collect_attention_distributions(model, dataloader, num_batches=5):
    """Collect attention distributions from the model for importance estimation."""
    # Move model to eval mode
    model.eval()
    
    # Store attention distributions
    attention_dists = {}
    
    # Register hooks to get attention weights
    hooks = []
    
    def attention_hook(module, input, output):
        # Get attention weights from output
        if isinstance(output, tuple) and len(output) > 1:
            # Some models return (context, attention_weights, ...)
            attention_weights = output[1]
            if attention_weights is not None:
                # Store attention weights by module
                module_id = id(module)
                if module_id not in attention_dists:
                    attention_dists[module_id] = []
                attention_dists[module_id].append(attention_weights.detach())
    
    # Attach hooks to all attention modules
    transformer_blocks = model.transformer.h if hasattr(model, 'transformer') and hasattr(model.transformer, 'h') else []
    
    for block in transformer_blocks:
        if hasattr(block, 'attn'):
            hook = block.attn.register_forward_hook(attention_hook)
            hooks.append(hook)
    
    # Collect attention patterns from batches
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            # Handle different batch formats
            if isinstance(batch, tuple) and len(batch) >= 2:
                input_ids, attention_mask = batch[0], batch[1]
            elif isinstance(batch, list) and len(batch) >= 2:
                input_ids, attention_mask = batch[0], batch[1]
            elif isinstance(batch, dict):
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask", None)
            else:
                # Skip this batch if format is unsupported
                print(f"Skipping unsupported batch format: {type(batch)}")
                continue
                
            # Forward pass to get attention weights
            device = model.device if hasattr(model, 'device') else next(model.parameters()).device
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            outputs = model(input_ids=input_ids, 
                           attention_mask=attention_mask, 
                           output_attentions=True)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Average attention distributions across batches
    avg_distributions = {}
    for module_id, dists in attention_dists.items():
        if dists:  # Check if we collected any distributions
            avg_distributions[module_id] = torch.cat(dists).mean(dim=0)
    
    return avg_distributions


# Local implementation of entropy_based_pruning if needed
def local_entropy_based_pruning(model, distributions, prune_ratio=0.3, safe_update_tensor_fn=None):
    """Prune attention heads based on entropy of attention distributions."""
    # Calculate entropy for each head
    head_entropies = []
    transformer_blocks = model.transformer.h if hasattr(model, 'transformer') and hasattr(model.transformer, 'h') else []
    
    for i, block in enumerate(transformer_blocks):
        if hasattr(block, 'attn'):
            module_id = id(block.attn)
            if module_id in distributions:
                attention_dist = distributions[module_id]
                
                # Calculate entropy for each head
                # Higher entropy → more uniform attention → less specialized → potentially less important
                for head_idx in range(attention_dist.shape[1]):
                    head_dist = attention_dist[:, head_idx]
                    # Add small epsilon to avoid log(0)
                    eps = 1e-10
                    head_dist = torch.clamp(head_dist, min=eps)
                    # Normalize to ensure it sums to 1
                    head_dist = head_dist / head_dist.sum(dim=-1, keepdim=True)
                    # Calculate entropy
                    entropy = -torch.sum(head_dist * torch.log(head_dist), dim=-1).mean().item()
                    head_entropies.append((i, head_idx, entropy))
    
    # Sort by descending entropy (higher entropy → less important)
    head_entropies.sort(key=lambda x: -x[2])
    
    # Determine number of heads to prune
    num_heads = len(head_entropies)
    heads_to_prune = int(num_heads * prune_ratio)
    
    # Prune heads by setting their mask to 0
    pruned_heads = []
    for i in range(heads_to_prune):
        layer_idx, head_idx, _ = head_entropies[i]
        
        # Access the block and set gate to 0 to prune the head
        if layer_idx < len(transformer_blocks):
            block = transformer_blocks[layer_idx]
            if hasattr(block, 'attn') and hasattr(block.attn, 'gate'):
                # Set gate to 0 to prune head
                block.attn.gate[head_idx] = 0.0
                pruned_heads.append((layer_idx, head_idx))
    
    print(f"Pruned {len(pruned_heads)} heads based on entropy")
    return pruned_heads


# Function to run the complete experiment
def run_experiment(config):
    """Run a complete pruning and fine-tuning experiment.
    
    Args:
        config: An ExperimentConfig object with experiment parameters
        
    Returns:
        Tuple of (model, tokenizer, summary_dict)
    """
    # We should address deprecation warnings properly rather than suppressing them
    
    print(f"Starting experiment with config: {config}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 1. Load model and tokenizer
    print(f"Loading model: {config.model_name}")
    try:
        # Load model with caching enabled (improves performance)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, 
            use_cache=True
        ).to(config.device)
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"Successfully loaded {config.model_name}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
        
    # 2. Prepare data
    print("Preparing data...")
    if config.use_test_data:
        # Use simple synthetic data for testing
        train_dataloader, eval_dataloader = prepare_test_data(tokenizer, max_length=config.max_length, batch_size=config.batch_size)
    else:
        # Use real data for full experiments
        train_dataloader = prepare_data(tokenizer, split="train", batch_size=config.batch_size, max_length=config.max_length)
        eval_dataloader = prepare_data(tokenizer, split="validation", batch_size=config.batch_size, max_length=config.max_length)
    
    # 3. Evaluate baseline model
    print("\nEvaluating baseline model...")
    baseline_metrics = evaluate_model(model, eval_dataloader)
    print(f"Baseline metrics: {baseline_metrics}")
    
    # Generate sample text with baseline model
    from .text_generator import generate_text
    
    # Get the default prompt from the config or use a fallback
    prompt = getattr(config, 'prompt', "Once upon a time")
    
    baseline_text = generate_text(model, tokenizer, prompt, max_length=50)
    print(f"Baseline generated text: {baseline_text}")
    
    # 4. Apply pruning strategy
    print(f"\nApplying {config.pruning_strategy} pruning with level {config.pruning_percent}")
    
    # Choose pruning strategy
    if config.pruning_strategy == "entropy":
        # Collect attention distributions (sample a few batches)
        if USING_FULL_MODULAR_API:
            # Use the imported functions from the modular API
            distributions = collect_attention_distributions(
                model,
                train_dataloader,
                num_batches=5  # Adjust based on dataset size
            )
            
            # Apply entropy-based pruning using the modular API implementation
            pruned_heads = entropy_based_pruning(
                model,
                distributions,
                prune_ratio=config.pruning_percent
            )
        else:
            # Use the locally defined functions
            distributions = local_collect_attention_distributions(
                model,
                train_dataloader,
                num_batches=5  # Adjust based on dataset size
            )
            
            # Apply entropy-based pruning using the local implementation
            pruned_heads = local_entropy_based_pruning(
                model,
                distributions,
                prune_ratio=config.pruning_percent
            )
    else:
        # Apply generic head importance based pruning
        head_importance = compute_head_importance(model, train_dataloader)
        pruned_heads = prune_heads(model, head_importance, config.pruning_percent)
    
    # 5. Evaluate pruned model
    print("\nEvaluating pruned model...")
    pruned_metrics = evaluate_model(model, eval_dataloader)
    print(f"Pruned metrics: {pruned_metrics}")
    
    # Generate sample text with pruned model
    pruned_text = generate_text(model, tokenizer, prompt, max_length=50)
    print(f"Pruned generated text: {pruned_text}")
    
    # 6. Fine-tune the pruned model
    print(f"\nFine-tuning pruned model for {config.num_epochs} epochs...")
    
    # Fine-tune the model
    training_history = fine_tune(
        model, 
        train_dataloader, 
        eval_dataloader, 
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate
    )
    
    # 7. Evaluate fine-tuned model
    print("\nEvaluating fine-tuned model...")
    finetuned_metrics = evaluate_model(model, eval_dataloader)
    print(f"Fine-tuned metrics: {finetuned_metrics}")
    
    # Generate sample text with fine-tuned model
    finetuned_text = generate_text(model, tokenizer, prompt, max_length=50)
    print(f"Fine-tuned generated text: {finetuned_text}")
    
    # 8. Calculate improvement
    # IMPORTANT: Handle different return formats from evaluate_model
    # It might return a tuple of (loss, perplexity) or a dict with keys "loss" and "perplexity"
    # This code is unit tested in utils/pruning/tests_metrics.py to prevent regressions
    if isinstance(baseline_metrics, tuple) and len(baseline_metrics) == 2:
        baseline_loss, baseline_perplexity = baseline_metrics
    elif isinstance(baseline_metrics, dict):
        baseline_loss = baseline_metrics["loss"]
        baseline_perplexity = baseline_metrics["perplexity"]
    else:
        raise ValueError(f"Unsupported metrics format: {type(baseline_metrics)}")
        
    if isinstance(pruned_metrics, tuple) and len(pruned_metrics) == 2:
        pruned_loss, pruned_perplexity = pruned_metrics
    elif isinstance(pruned_metrics, dict):
        pruned_loss = pruned_metrics["loss"]
        pruned_perplexity = pruned_metrics["perplexity"]
    else:
        raise ValueError(f"Unsupported metrics format: {type(pruned_metrics)}")
        
    if isinstance(finetuned_metrics, tuple) and len(finetuned_metrics) == 2:
        finetuned_loss, finetuned_perplexity = finetuned_metrics
    elif isinstance(finetuned_metrics, dict):
        finetuned_loss = finetuned_metrics["loss"]
        finetuned_perplexity = finetuned_metrics["perplexity"]
    else:
        raise ValueError(f"Unsupported metrics format: {type(finetuned_metrics)}")
    
    loss_change_pruning = pruned_loss - baseline_loss
    loss_change_finetuning = finetuned_loss - pruned_loss
    overall_improvement = ((baseline_loss - finetuned_loss) / baseline_loss) * 100
    
    print(f"\nOverall improvement: {overall_improvement:.2f}%")
    
    # 9. Create summary dictionary
    summary = {
        "baseline": {
            "loss": baseline_loss,
            "perplexity": baseline_perplexity
        },
        "pruned": {
            "loss": pruned_loss,
            "perplexity": pruned_perplexity
        },
        "finetuned": {
            "loss": finetuned_loss,
            "perplexity": finetuned_perplexity
        },
        "improvement": {
            "overall_percent": float(overall_improvement),
            "loss_change_pruning": float(loss_change_pruning),
            "loss_change_finetuning": float(loss_change_finetuning)
        },
        "text_samples": {
            "baseline": baseline_text,
            "pruned": pruned_text,
            "finetuned": finetuned_text
        },
        "training_history": training_history,
        "pruned_heads": len(pruned_heads)
    }
    
    # 10. Plot training history and metrics comparison
    try:
        # Create plot with metrics
        plt.figure(figsize=(12, 5))
        
        # Left subplot: Metrics comparison
        plt.subplot(1, 2, 1)
        stages = ['Baseline', 'Pruned', 'Fine-tuned']
        perplexity = [baseline_perplexity, pruned_perplexity, finetuned_perplexity]
        
        # Create bar chart
        bars = plt.bar(stages, perplexity, color=['blue', 'red', 'green'])
        plt.ylabel('Perplexity (lower is better)')
        plt.title('Perplexity Comparison')
        
        # Add value labels above bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Try to plot training history if available in right format
        if isinstance(training_history, dict) and 'train_loss' in training_history and len(training_history['train_loss']) > 0:
            # Right subplot: Training history
            plt.subplot(1, 2, 2)
            epochs = range(1, len(training_history['train_loss']) + 1)
            
            # Plot training loss
            plt.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss')
            
            # Plot eval loss if available
            if 'eval_loss' in training_history:
                plt.plot(epochs, training_history['eval_loss'], 'r-', label='Validation Loss')
                
            plt.title('Loss During Fine-tuning')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
        else:
            # If no training history, just show improvement percentage
            plt.subplot(1, 2, 2)
            plt.text(0.5, 0.5, f"Overall Improvement: {overall_improvement:.2f}%\n{len(pruned_heads)} heads pruned", 
                    horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig(f"{config.output_dir}/pruning_results.png")
        print(f"Saved visualization to {config.output_dir}/pruning_results.png")
    except Exception as e:
        print(f"Error plotting results: {e}")
        print("Continuing without plotting")
    
    return model, tokenizer, summary
