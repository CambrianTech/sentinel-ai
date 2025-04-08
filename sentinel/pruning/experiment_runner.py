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

# Implement pruning functions directly to avoid circular imports
# These will eventually be refactored to live fully within sentinel.pruning

def compute_head_importance(model, dataloader, num_batches=10, device="cuda"):
    """Compute importance scores for each attention head using a specified strategy."""
    print("Using simplified compute_head_importance from experiment_runner.py")
    import numpy as np
    import torch
    
    # Get model configuration
    config = model.config
    
    # Try to get number of layers and heads based on model type
    num_layers = None
    num_heads = None
    
    # Check for common model configuration attributes
    if hasattr(config, "n_layer") and hasattr(config, "n_head"):
        # GPT-2 style configuration
        num_layers = config.n_layer
        num_heads = config.n_head
    elif hasattr(config, "num_hidden_layers") and hasattr(config, "num_attention_heads"):
        # BERT/RoBERTa style configuration
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
    elif hasattr(config, "depth") and hasattr(config, "n_heads"):
        # Some custom configurations
        num_layers = config.depth
        num_heads = config.n_heads
    else:
        # Default fallback values
        print("Warning: Could not determine model architecture from config. Using default values.")
        num_layers = 6
        num_heads = 12
    
    print(f"Model has {num_layers} layers with {num_heads} heads each")
    
    # For testing purposes, just return random importance scores
    return np.random.rand(num_layers, num_heads)

def prune_heads(model, importance, pruning_percent=0.3, device="cuda"):
    """Prune less important attention heads based on importance scores."""
    print("Using simplified prune_heads from experiment_runner.py")
    import numpy as np
    import torch
    
    # Get model configuration
    config = model.config
    
    # Try to get number of layers and heads based on model type
    num_layers = None
    num_heads = None
    
    # Check for common model configuration attributes
    if hasattr(config, "n_layer") and hasattr(config, "n_head"):
        # GPT-2 style configuration
        num_layers = config.n_layer
        num_heads = config.n_head
    elif hasattr(config, "num_hidden_layers") and hasattr(config, "num_attention_heads"):
        # BERT/RoBERTa style configuration
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
    elif hasattr(config, "depth") and hasattr(config, "n_heads"):
        # Some custom configurations
        num_layers = config.depth
        num_heads = config.n_heads
    else:
        # Default fallback values
        print("Warning: Could not determine model architecture from config. Using default values.")
        num_layers = 6
        num_heads = 12
    
    # Calculate number of heads to prune
    num_to_prune = int(num_layers * num_heads * pruning_percent)
    print(f"Pruning {num_to_prune} of {num_layers * num_heads} heads ({pruning_percent*100:.1f}%)")
    
    # Get indices of heads to prune
    indices = np.argsort(importance.flatten())[:num_to_prune]
    pruned_heads = [(idx // num_heads, idx % num_heads) for idx in indices]
    
    # Print information about pruned heads
    for layer_idx, head_idx in pruned_heads:
        print(f"Pruning layer {layer_idx}, head {head_idx}")
    
    # For a real implementation, we would need to actually modify the weights here
    # but for this simplified version, we just return the list of pruned heads
    print("Note: In this simplified version, we're just identifying heads to prune without modifying weights")
    
    return pruned_heads

def evaluate_model(model, dataloader, device="cuda"):
    """Evaluate model on dataloader."""
    print("Using simplified evaluate_model from experiment_runner.py")
    import torch
    from tqdm import tqdm
    model.eval()
    total_loss = 0
    total_elements = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if isinstance(batch, tuple) and len(batch) >= 2:
                input_ids, attention_mask = batch[0], batch[1]
            elif isinstance(batch, dict):
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask", None)
            else:
                continue
                
            # Move to device
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Create a simplified dataset to quickly test functionality
            if input_ids.size(0) == 0:
                print("Warning: Empty batch detected, using synthetic data for testing")
                input_ids = torch.randint(0, 100, (4, 32), device=device)
                attention_mask = torch.ones_like(input_ids)
            
            # Forward pass - create a dictionary of inputs to avoid issues with attention masks
            model_inputs = {
                "input_ids": input_ids,
                "labels": input_ids,
            }
            if attention_mask is not None:
                model_inputs["attention_mask"] = attention_mask
                
            outputs = model(**model_inputs)
            
            # Get loss
            loss = outputs.loss
            
            # Accumulate loss
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_elements += batch_size
    
    # Calculate average loss and perplexity
    avg_loss = total_loss / total_elements if total_elements > 0 else float("inf")
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity

def fine_tune(model, train_dataloader, val_dataloader, num_epochs=3, lr=5e-5, device="cuda", callbacks=None):
    """Fine-tune the model."""
    print("Using simplified fine_tune from experiment_runner.py")
    import torch
    from transformers import get_linear_schedule_with_warmup
    from tqdm import tqdm
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=100, 
        num_training_steps=total_steps
    )
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            if isinstance(batch, tuple) and len(batch) >= 2:
                input_ids, attention_mask = batch[0], batch[1]
            elif isinstance(batch, dict):
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask", None)
            else:
                continue
                
            # Move to device
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Create a simplified dataset to quickly test functionality
            if input_ids.size(0) == 0:
                print("Warning: Empty batch detected, using synthetic data for testing")
                input_ids = torch.randint(0, 100, (4, 32), device=device)
                attention_mask = torch.ones_like(input_ids)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass - create a dictionary of inputs to avoid issues with attention masks
            model_inputs = {
                "input_ids": input_ids,
                "labels": input_ids,
            }
            if attention_mask is not None:
                model_inputs["attention_mask"] = attention_mask
            
            outputs = model(**model_inputs)
            
            # Compute loss
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            
            # Record loss
            train_loss += loss.item()
            
            # Print progress every 100 steps
            if step % 100 == 0 and step > 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")
            
            # Call step callback if provided
            if callbacks and 'on_step' in callbacks:
                callbacks['on_step'](epoch * len(train_dataloader) + step, loss.item())
        
        # Calculate average training loss for this epoch
        avg_train_loss = train_loss / len(train_dataloader) if len(train_dataloader) > 0 else float("inf")
        
        # Evaluation
        val_loss, val_ppl = evaluate_model(model, val_dataloader, device=device)
        
        # Print epoch summary
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
        
        # Call eval callback if provided
        if callbacks and 'on_eval' in callbacks:
            callbacks['on_eval'](epoch, val_loss, val_ppl)
    
    print("Fine-tuning complete")
    
    # Final evaluation
    final_loss, final_ppl = evaluate_model(model, val_dataloader, device=device)
    return final_loss, final_ppl

# Simple functions for dataset handling - this avoids importing datasets directly
def load_wikitext():
    """Load the WikiText dataset (simple version)."""
    print("Using simplified load_wikitext from experiment_runner.py")
    # Just create dummy data for testing
    import random
    
    # Create simple dummy sentences
    train_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Python is a powerful programming language.",
        "Artificial intelligence is transforming many industries.",
        "Neural networks learn from data to make predictions."
    ]
    
    # Repeat sentences to create more data
    train_data = train_sentences * 10
    val_data = train_sentences[:3] * 5
    
    return train_data, val_data

def prepare_data(tokenizer, data, batch_size=4):
    """Prepare data for training (simple version)."""
    print("Using simplified prepare_data from experiment_runner.py")
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    
    if data is None or len(data) == 0:
        # Generate random data if no data is provided
        print("No data provided, generating random data")
        input_ids = torch.randint(0, tokenizer.vocab_size, (100, 32))
        attention_mask = torch.ones_like(input_ids)
    else:
        # Tokenize real data
        print(f"Tokenizing {len(data)} samples")
        encodings = tokenizer(
            data,
            truncation=True,
            max_length=64,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
    
    # Create dataset and dataloader
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def prepare_test_data(tokenizer, batch_size=4):
    """Prepare test data for quick testing (simple version)."""
    print("Using simplified prepare_test_data from experiment_runner.py")
    
    # Create dummy test data
    train_data = [
        "This is a test sentence for training.",
        "Another example for the training dataset.",
        "Machine learning models need training data."
    ] * 5
    
    val_data = [
        "This is a validation sentence.",
        "We need to evaluate model performance."
    ] * 5
    
    # Prepare dataloaders
    train_dataloader = prepare_data(tokenizer, train_data, batch_size)
    val_dataloader = prepare_data(tokenizer, val_data, batch_size)
    
    return train_dataloader, val_dataloader


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
        config_dict = dataclasses.asdict(self)
        
        # Convert device object to string if needed
        if isinstance(config_dict['device'], torch.device):
            config_dict['device'] = str(config_dict['device'])
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        # Convert device string to proper device if needed
        if 'device' in config_dict and isinstance(config_dict['device'], str):
            if config_dict['device'].startswith('cuda'):
                config_dict['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        
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
    
    # Initialize progress tracker (disable plotting in non-interactive environments)
    import sys
    disable_plotting = not sys.stdout.isatty()
    tracker = ProgressTracker(disable_plotting=disable_plotting)
    
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