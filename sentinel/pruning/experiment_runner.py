"""
Experiment Runner for Pruning Transformers

This module provides a clean, modular interface for running pruning experiments
with transformer models.
"""

import os
import torch
import json
import dataclasses
from typing import Optional, Dict, List, Any, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path

from .model_manager import load_model, save_model
from .text_generator import generate_text
from .visualization import ProgressTracker

# Import from utils (will be refactored to live in sentinel)
from utils.pruning.api.pruning import compute_head_importance, prune_heads, fine_tune, evaluate_model
from utils.pruning.api.data import load_wikitext, prepare_data, prepare_test_data


@dataclass
class ExperimentConfig:
    """Configuration for pruning experiments."""
    model_name: str = "distilgpt2"
    pruning_percent: float = 0.3
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    prompt: str = "The quick brown fox jumps over the lazy dog. In recent years,"
    max_length: int = 100
    output_dir: str = "pruning_results"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_test_data: bool = False
    interactive: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def run_experiment(config: Optional[ExperimentConfig] = None, **kwargs) -> Tuple[torch.nn.Module, Any, Dict[str, Any]]:
    """
    Run a pruning and fine-tuning experiment with the specified configuration.
    
    Args:
        config: The experiment configuration
        **kwargs: Overrides for configuration parameters
        
    Returns:
        Tuple of (model, tokenizer, summary metrics)
    """
    # Use provided config or create a new one with kwargs
    if config is None:
        config = ExperimentConfig(**kwargs)
    else:
        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    print("Starting experiment...")
    print(f"Model: {config.model_name}")
    print(f"Pruning: {config.pruning_percent*100:.1f}%")
    print(f"Device: {config.device}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config.output_dir, "config.json")
    config.save(config_path)
    
    # Initialize progress tracker
    tracker = ProgressTracker()
    
    # Load model and tokenizer
    model, tokenizer = load_model(config.model_name, device=config.device)
    
    # Load and prepare data
    if config.use_test_data:
        print("Using small test dataset for quick validation")
        train_dataloader, val_dataloader = prepare_test_data(tokenizer, batch_size=config.batch_size)
    else:
        # Load real data
        train_data, val_data = load_wikitext()
        train_dataloader = prepare_data(tokenizer, train_data, batch_size=config.batch_size)
        val_dataloader = prepare_data(tokenizer, val_data, batch_size=config.batch_size)
    
    # Evaluate initial model
    print("\nEvaluating initial model...")
    initial_loss, initial_ppl = evaluate_model(model, val_dataloader, device=config.device)
    print(f"Initial model - Loss: {initial_loss:.4f}, Perplexity: {initial_ppl:.2f}")
    
    # Generate text with initial model
    initial_text = generate_text(model, tokenizer, config.prompt, max_length=config.max_length)
    print(f"\nInitial text generation:\n{initial_text}")
    
    # Record initial metrics
    tracker.update(0, initial_loss, initial_ppl, initial_text)
    
    # Compute head importance
    print("\nComputing head importance...")
    importance = compute_head_importance(model, val_dataloader, device=config.device)
    
    # Prune heads
    print("\nPruning heads...")
    pruned_heads = prune_heads(model, importance, pruning_percent=config.pruning_percent, device=config.device)
    
    # Store pruning results
    tracker.set_pruning_info(config.pruning_percent, pruned_heads)
    
    # Evaluate pruned model
    print("\nEvaluating pruned model...")
    pruned_loss, pruned_ppl = evaluate_model(model, val_dataloader, device=config.device)
    print(f"Pruned model - Loss: {pruned_loss:.4f}, Perplexity: {pruned_ppl:.2f}")
    
    # Generate text with pruned model
    pruned_text = generate_text(model, tokenizer, config.prompt, max_length=config.max_length)
    print(f"\nPruned model text generation:\n{pruned_text}")
    
    # Record pruned metrics
    tracker.update(1, pruned_loss, pruned_ppl, pruned_text)
    
    # Define callback functions for fine-tuning
    callbacks = {
        'on_step': lambda step, loss: None,  # No-op
        'on_eval': lambda step, loss, ppl: tracker.update(step + 2, loss, ppl)  # +2 because we already have steps 0 and 1
    }
    
    # Fine-tune the pruned model
    print("\nFine-tuning the pruned model...")
    final_loss, final_ppl = fine_tune(
        model, 
        train_dataloader, 
        val_dataloader, 
        num_epochs=config.num_epochs,
        device=config.device,
        callbacks=callbacks
    )
    
    # Generate text with fine-tuned model
    final_text = generate_text(model, tokenizer, config.prompt, max_length=config.max_length)
    print(f"\nFine-tuned model text generation:\n{final_text}")
    
    # Record final metrics if not already recorded by callbacks
    tracker.update(2 + config.num_epochs, final_loss, final_ppl, final_text)
    
    # Save metrics
    metrics_path = os.path.join(config.output_dir, "metrics.json")
    tracker.save_metrics(metrics_path)
    
    # Save plots
    plots_path = os.path.join(config.output_dir, "training_plots.png")
    tracker.save_plots(plots_path)
    
    # Calculate improvement
    initial_to_final = ((initial_ppl - final_ppl) / initial_ppl) * 100
    pruned_to_final = ((pruned_ppl - final_ppl) / pruned_ppl) * 100
    
    # Print summary
    print("\n=== Experiment Summary ===")
    print(f"Model: {config.model_name}")
    print(f"Pruning: {config.pruning_percent*100:.1f}% of heads pruned ({len(pruned_heads)} heads)")
    print(f"Initial perplexity: {initial_ppl:.2f}")
    print(f"After pruning perplexity: {pruned_ppl:.2f}")
    print(f"After fine-tuning perplexity: {final_ppl:.2f}")
    print(f"Overall improvement: {initial_to_final:.2f}%")
    print(f"Recovery from pruning: {pruned_to_final:.2f}%")
    
    # Save model
    model_path = os.path.join(config.output_dir, "model.pt")
    save_model(model, tokenizer, model_path)
    
    return model, tokenizer, tracker.get_summary()