"""
Neural Plasticity Experiment

This module provides experiment utilities for running end-to-end neural
plasticity experiments, including baseline creation, data preparation,
and results analysis.

Version: v0.0.60 (2025-04-20 16:00:00)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from datetime import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader
from datasets import load_dataset

from .core import (
    calculate_head_entropy,
    calculate_head_gradients,
    detect_model_structure,
    evaluate_model,
    IS_APPLE_SILICON,
    IS_COLAB,
    HAS_GPU
)

from .training import run_plasticity_loop
from .visualization import (
    visualize_head_entropy,
    visualize_head_gradients,
    visualize_pruning_decisions
)


def get_dataloader_builder(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    model_name: str = "distilgpt2",
    max_length: int = 128,
    batch_size: int = 4
) -> Callable[[], Tuple[DataLoader, DataLoader]]:
    """
    Create a function that builds dataloaders for the specified dataset.

    Args:
        dataset_name: Name of the dataset to use
        dataset_config: Dataset configuration
        model_name: Model name for tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size for dataloaders

    Returns:
        Function that returns (train_dataloader, eval_dataloader)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def build_dataloaders() -> Tuple[DataLoader, DataLoader]:
        # Load datasets
        train_dataset = load_dataset(dataset_name, dataset_config, split="train")
        validation_dataset = load_dataset(dataset_name, dataset_config, split="validation")

        # Define tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length
            )

        # Tokenize datasets
        train_dataset = train_dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )
        validation_dataset = validation_dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

        # Add labels for language modeling
        def add_labels(examples):
            examples["labels"] = examples["input_ids"].copy()
            return examples

        train_dataset = train_dataset.map(add_labels)
        validation_dataset = validation_dataset.map(add_labels)

        # Set format
        train_dataset = train_dataset.with_format("torch")
        validation_dataset = validation_dataset.with_format("torch")

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=default_data_collator
        )

        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            collate_fn=default_data_collator
        )

        return train_dataloader, validation_dataloader

    return build_dataloaders


def create_model_and_entropy_baseline(
    model_name: str = "distilgpt2",
    device: Optional[str] = None,
    eval_dataloader: Optional[DataLoader] = None,
    dataloader_builder: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Create a model and calculate baseline entropy values.

    Args:
        model_name: Name of the model to use
        device: Device to use (auto-detected if None)
        eval_dataloader: Evaluation dataloader
        dataloader_builder: Function to build dataloaders if eval_dataloader is None

    Returns:
        Dictionary with model, entropy, gradient values, and evaluation metrics
    """
    # Determine appropriate device
    if device is None:
        if IS_COLAB and HAS_GPU:
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif IS_APPLE_SILICON:
            device = torch.device("cpu")
            print("Using CPU on Apple Silicon")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

    # Load model
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Create dataloader if not provided
    if eval_dataloader is None:
        if dataloader_builder is None:
            dataloader_builder = get_dataloader_builder(model_name=model_name)
        
        train_dataloader, eval_dataloader = dataloader_builder()
    
    # Calculate baseline entropy
    print("Calculating baseline entropy...")
    with torch.no_grad():
        batch = next(iter(eval_dataloader))
        # Move batch to device
        inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        # Forward pass with attention outputs
        outputs = model(**inputs, output_attentions=True)
        
        # Extract attention maps
        if hasattr(outputs, 'attentions') and outputs.attentions:
            # Calculate entropy for each layer
            entropy_values = torch.stack([
                calculate_head_entropy(layer_attn) 
                for layer_attn in outputs.attentions
            ])
        else:
            raise ValueError("Model does not output attention maps")
    
    # Calculate baseline gradients
    print("Calculating baseline gradients...")
    grad_norm_values = calculate_head_gradients(model, eval_dataloader)
    
    # Evaluate baseline model
    print("Evaluating baseline model...")
    baseline_metrics = evaluate_model(model, eval_dataloader)
    
    return {
        "model": model,
        "device": device,
        "entropy_values": entropy_values,
        "grad_norm_values": grad_norm_values,
        "baseline_metrics": baseline_metrics
    }


def plot_baseline_entropy(
    entropy_values: torch.Tensor,
    grad_norm_values: torch.Tensor,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot baseline entropy and gradient values.

    Args:
        entropy_values: Tensor of entropy values
        grad_norm_values: Tensor of gradient norm values
        figsize: Figure size
        save_path: Path to save the figure

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot entropy
    entropy_data = entropy_values.detach().cpu().numpy()
    im1 = ax1.imshow(entropy_data, cmap="viridis", aspect="auto")
    fig.colorbar(im1, ax=ax1, label='Entropy')
    ax1.set_title('Head Entropy (Higher = Less Focused)')
    ax1.set_xlabel('Head Index')
    ax1.set_ylabel('Layer Index')
    
    # Set proper colormap limits with non-zero range
    im1.set_clim(0, max(0.1, entropy_data.max()))
    
    # Plot gradients
    grad_data = grad_norm_values.detach().cpu().numpy()
    im2 = ax2.imshow(grad_data, cmap="plasma", aspect="auto")
    fig.colorbar(im2, ax=ax2, label='Gradient Norm')
    ax2.set_title('Head Gradient Norms (Higher = More Learning)')
    ax2.set_xlabel('Head Index')
    ax2.set_ylabel('Layer Index')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


def run_neural_plasticity_experiment(
    model_name: str = "distilgpt2",
    device: Optional[str] = None,
    output_dir: str = "plasticity_results",
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    pruning_strategy: str = "entropy",
    prune_percent: float = 0.2,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    training_steps: int = 500,
    use_differential_lr: bool = True,
    num_cycles: int = 3,
    create_visualizations: bool = True
) -> Dict[str, Any]:
    """
    Run a complete neural plasticity experiment.

    Args:
        model_name: Name of the model to use
        device: Device to use (auto-detected if None)
        output_dir: Directory to save results
        dataset_name: Name of the dataset to use
        dataset_config: Dataset configuration
        pruning_strategy: Pruning strategy to use
        prune_percent: Percentage of heads to prune (0-1)
        batch_size: Batch size for training
        learning_rate: Learning rate for training
        training_steps: Number of training steps per cycle
        use_differential_lr: Whether to use differential learning rates
        num_cycles: Number of plasticity cycles to run
        create_visualizations: Whether to create and save visualizations

    Returns:
        Dictionary with experiment results
    """
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = os.path.join(output_dir, f"{model_name.split('/')[-1]}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    if create_visualizations:
        visualization_dir = os.path.join(experiment_dir, "visualizations")
        os.makedirs(visualization_dir, exist_ok=True)
    
    # Create dataloader builder
    dataloader_builder = get_dataloader_builder(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        model_name=model_name,
        max_length=128,
        batch_size=batch_size
    )
    
    # Create dataloaders
    train_dataloader, eval_dataloader = dataloader_builder()
    
    # Create model and calculate baseline
    baseline = create_model_and_entropy_baseline(
        model_name=model_name,
        device=device,
        eval_dataloader=eval_dataloader
    )
    
    model = baseline["model"]
    device = baseline["device"]
    entropy_values = baseline["entropy_values"]
    grad_norm_values = baseline["grad_norm_values"]
    baseline_metrics = baseline["baseline_metrics"]
    
    # Save baseline metrics
    baseline_file = os.path.join(experiment_dir, "baseline_metrics.txt")
    with open(baseline_file, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Baseline Loss: {baseline_metrics['loss']:.4f}\n")
        f.write(f"Baseline Perplexity: {baseline_metrics['perplexity']:.2f}\n")
    
    # Create baseline visualization
    if create_visualizations:
        baseline_plot = plot_baseline_entropy(
            entropy_values=entropy_values,
            grad_norm_values=grad_norm_values,
            save_path=os.path.join(visualization_dir, "baseline_metrics.png")
        )
    
    # Run plasticity cycles
    cycle_results = []
    current_metrics = baseline_metrics
    best_metrics = baseline_metrics
    best_model_path = None
    
    for cycle in range(num_cycles):
        print(f"\n--- Plasticity Cycle {cycle+1}/{num_cycles} ---")
        
        # Run plasticity loop
        result = run_plasticity_loop(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            pruning_level=prune_percent,
            strategy=pruning_strategy,
            learning_rate=learning_rate,
            training_steps=training_steps,
            use_differential_lr=use_differential_lr
        )
        
        # Update current metrics
        current_metrics = result["final_metrics"]
        
        # Save cycle results
        cycle_results.append({
            "cycle": cycle + 1,
            "baseline_metrics": baseline_metrics,
            "pruned_metrics": result["pruned_metrics"],
            "final_metrics": result["final_metrics"],
            "pruned_heads": result["pruned_heads"],
            "perplexity_improvement": result["perplexity_improvement"],
            "recovery_rate": result["recovery_rate"]
        })
        
        # Create cycle visualization
        if create_visualizations:
            # Visualize pruning decisions
            pruning_viz = visualize_pruning_decisions(
                grad_norm_values=result["grad_norm_values"],
                pruning_mask=result["pruning_mask"],
                title=f"Cycle {cycle+1} Pruning Decisions ({len(result['pruned_heads'])} heads)",
                save_path=os.path.join(visualization_dir, f"cycle{cycle+1}_pruning.png")
            )
        
        # Check if this is the best model so far
        if current_metrics["perplexity"] < best_metrics["perplexity"]:
            best_metrics = current_metrics
            
            # Save best model
            best_model_path = os.path.join(experiment_dir, f"model_best_cycle{cycle+1}.pt")
            torch.save(model.state_dict(), best_model_path)
            
            print(f"New best model (cycle {cycle+1}) - Perplexity: {best_metrics['perplexity']:.2f}")
    
    # Calculate overall improvement
    overall_improvement = (baseline_metrics["perplexity"] - best_metrics["perplexity"]) / baseline_metrics["perplexity"]
    
    # Save final results
    results = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "pruning_strategy": pruning_strategy,
        "prune_percent": prune_percent,
        "learning_rate": learning_rate,
        "training_steps": training_steps,
        "num_cycles": num_cycles,
        "baseline_metrics": baseline_metrics,
        "best_metrics": best_metrics,
        "overall_improvement": overall_improvement,
        "cycle_results": cycle_results,
        "best_model_path": best_model_path
    }
    
    # Print final results
    print("\n=== Experiment Results ===")
    print(f"Baseline Perplexity: {baseline_metrics['perplexity']:.2f}")
    print(f"Final Perplexity: {best_metrics['perplexity']:.2f}")
    print(f"Overall Improvement: {overall_improvement*100:.2f}%")
    print(f"Results saved to: {experiment_dir}")
    
    return results