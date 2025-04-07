#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pruning and Fine-Tuning Colab (v0.0.37)

This script demonstrates making a GPT-2 model smaller and more powerful by:
1. Applying pruning to remove less important attention heads
2. Fine-tuning the pruned model to recover performance
3. Showing clear metrics of improvement

It's designed to be run in Google Colab using real-world data (Wikitext).

Version History:
- v0.0.37 (April 2025): Complete rewrite with minimal dependencies for reliability
- v0.0.36 (April 2025): Simplified pruning implementation for better reliability 
- v0.0.35 (April 2025): Fixed in-place operation error in apply_head_pruning function
- v0.0.34 (April 2025): Fixed undefined variable error, visualization issues and enhanced CUDA error handling
- v0.0.33 (April 2025): Fixed visualization issues, improved model compatibility and enhanced error handling
- v0.0.32 (April 2025): Added CUDA error handling for Colab compatibility and memory management
- v0.0.31 (April 2025): Fixed get_strategy parameters issue and improved Colab compatibility 
- v0.0.30 (April 2025): Added OPT model support and chart improvements
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm.auto import tqdm
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import our API modules
from utils.pruning.api.pruning import (
    compute_head_importance,
    prune_heads,
    fine_tune,
    evaluate_model
)
from utils.pruning.api.data import (
    load_wikitext,
    prepare_data,
    prepare_test_data
)

# Configure device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Global variables
OUTPUT_DIR = "pruning_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ProgressTracker:
    """Track metrics throughout the pruning and fine-tuning process."""
    
    def __init__(self):
        self.metrics = {
            "loss": [],
            "perplexity": [],
            "steps": [],
            "pruning_level": None,
            "pruned_heads": [],
            "generation_samples": []
        }
        
        # Create visualizations
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 5))
        
    def update(self, step, loss, perplexity, generation_sample=None):
        """Update metrics with new values."""
        self.metrics["steps"].append(step)
        self.metrics["loss"].append(loss)
        self.metrics["perplexity"].append(perplexity)
        
        if generation_sample is not None:
            self.metrics["generation_samples"].append({
                "step": step,
                "text": generation_sample
            })
        
        # Update visualization
        self._update_plots()
        
    def set_pruning_info(self, level, pruned_heads):
        """Set pruning information."""
        self.metrics["pruning_level"] = level
        self.metrics["pruned_heads"] = pruned_heads
        
    def _update_plots(self):
        """Update visualization plots."""
        steps = self.metrics["steps"]
        loss = self.metrics["loss"]
        ppl = self.metrics["perplexity"]
        
        if not steps:
            return
            
        # Clear previous plots
        self.axes[0].clear()
        self.axes[1].clear()
        
        # Plot loss
        self.axes[0].plot(steps, loss, 'b-')
        self.axes[0].set_title('Loss')
        self.axes[0].set_xlabel('Step')
        self.axes[0].set_ylabel('Loss')
        self.axes[0].grid(True)
        
        # Plot perplexity
        self.axes[1].plot(steps, ppl, 'r-')
        self.axes[1].set_title('Perplexity (lower is better)')
        self.axes[1].set_xlabel('Step')
        self.axes[1].set_ylabel('Perplexity')
        self.axes[1].grid(True)
        
        self.fig.tight_layout()
        plt.draw()
        plt.pause(0.001)
        
    def save_plots(self, path):
        """Save plots to file."""
        plt.savefig(path)
        
    def save_metrics(self, path):
        """Save metrics to file."""
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def get_summary(self):
        """Return a summary of key metrics."""
        if not self.metrics["perplexity"]:
            return {}
            
        return {
            "pruning_level": self.metrics["pruning_level"],
            "pruned_heads_count": len(self.metrics["pruned_heads"]),
            "initial_loss": self.metrics["loss"][0] if self.metrics["loss"] else None,
            "final_loss": self.metrics["loss"][-1] if self.metrics["loss"] else None,
            "initial_perplexity": self.metrics["perplexity"][0] if self.metrics["perplexity"] else None,
            "final_perplexity": self.metrics["perplexity"][-1] if self.metrics["perplexity"] else None,
            "improvement_percent": ((self.metrics["perplexity"][0] - self.metrics["perplexity"][-1]) / 
                                   self.metrics["perplexity"][0] * 100) 
                                   if (self.metrics["perplexity"] and len(self.metrics["perplexity"]) > 1) else None
        }

def load_model(model_name="distilgpt2"):
    """Load model and tokenizer"""
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Move model to device
    model.to(DEVICE)
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {param_count/1e6:.2f}M parameters")
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100):
    """Generate text from model"""
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

def run_experiment(model_name="distilgpt2", 
                  pruning_percent=0.3, 
                  num_epochs=3, 
                  batch_size=4,
                  use_test_data=False):
    """Run the pruning and fine-tuning experiment"""
    print("Starting experiment...")
    
    # Initialize tracker
    tracker = ProgressTracker()
    
    # Load model and tokenizer
    model, tokenizer = load_model(model_name)
    
    # Get model architecture details
    config = model.config
    num_layers = config.n_layer if hasattr(config, "n_layer") else config.num_hidden_layers
    num_heads = config.n_head if hasattr(config, "n_head") else config.num_attention_heads
    
    print(f"Model has {num_layers} layers with {num_heads} heads per layer")
    
    # Load and prepare data
    if use_test_data:
        print("Using small test dataset for quick validation")
        train_dataloader, val_dataloader = prepare_test_data(tokenizer, batch_size=batch_size)
    else:
        # Load real data
        train_data, val_data = load_wikitext()
        train_dataloader = prepare_data(tokenizer, train_data, batch_size=batch_size)
        val_dataloader = prepare_data(tokenizer, val_data, batch_size=batch_size)
    
    # Evaluate initial model
    print("\nEvaluating initial model...")
    initial_loss, initial_ppl = evaluate_model(model, val_dataloader, device=DEVICE)
    print(f"Initial model - Loss: {initial_loss:.4f}, Perplexity: {initial_ppl:.2f}")
    
    # Generate text with initial model
    initial_prompt = "The quick brown fox jumps over the lazy dog. In recent years,"
    initial_text = generate_text(model, tokenizer, initial_prompt)
    print(f"\nInitial text generation:\n{initial_text}")
    
    # Record initial metrics
    tracker.update(0, initial_loss, initial_ppl, initial_text)
    
    # Compute head importance
    print("\nComputing head importance...")
    importance = compute_head_importance(model, val_dataloader, device=DEVICE)
    
    # Prune heads
    print("\nPruning heads...")
    pruned_heads = prune_heads(model, importance, pruning_percent=pruning_percent, device=DEVICE)
    
    # Store pruning results
    tracker.set_pruning_info(pruning_percent, pruned_heads)
    
    # Evaluate pruned model
    print("\nEvaluating pruned model...")
    pruned_loss, pruned_ppl = evaluate_model(model, val_dataloader, device=DEVICE)
    print(f"Pruned model - Loss: {pruned_loss:.4f}, Perplexity: {pruned_ppl:.2f}")
    
    # Generate text with pruned model
    pruned_text = generate_text(model, tokenizer, initial_prompt)
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
        num_epochs=num_epochs,
        device=DEVICE,
        callbacks=callbacks
    )
    
    # Generate text with fine-tuned model
    final_text = generate_text(model, tokenizer, initial_prompt)
    print(f"\nFine-tuned model text generation:\n{final_text}")
    
    # Record final metrics if not already recorded by callbacks
    tracker.update(2 + num_epochs, final_loss, final_ppl, final_text)
    
    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    tracker.save_metrics(metrics_path)
    
    # Save plots
    plots_path = os.path.join(OUTPUT_DIR, "training_plots.png")
    tracker.save_plots(plots_path)
    
    # Calculate improvement
    initial_to_final = ((initial_ppl - final_ppl) / initial_ppl) * 100
    pruned_to_final = ((pruned_ppl - final_ppl) / pruned_ppl) * 100
    
    # Print summary
    print("\n=== Experiment Summary ===")
    print(f"Model: {model_name}")
    print(f"Pruning: {pruning_percent*100:.1f}% of heads pruned ({len(pruned_heads)} heads)")
    print(f"Initial perplexity: {initial_ppl:.2f}")
    print(f"After pruning perplexity: {pruned_ppl:.2f}")
    print(f"After fine-tuning perplexity: {final_ppl:.2f}")
    print(f"Overall improvement: {initial_to_final:.2f}%")
    print(f"Recovery from pruning: {pruned_to_final:.2f}%")
    
    return model, tokenizer, tracker.get_summary()

def interactive_generate(model, tokenizer, prompt="", max_length=100):
    """Generate text from the model interactively"""
    if not prompt:
        prompt = input("Enter a prompt: ")
        
    generated_text = generate_text(model, tokenizer, prompt, max_length)
    print(f"\nGenerated text:\n{generated_text}")
    
    return generated_text

def main(args):
    """Main function to run the experiment with command line arguments"""
    # Run the experiment
    model, tokenizer, summary = run_experiment(
        model_name=args.model_name,
        pruning_percent=args.pruning_percent,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        use_test_data=args.test_mode
    )
    
    # Interactive generation if requested
    if args.interactive:
        print("\nEntering interactive generation mode. Type 'exit' to quit.")
        while True:
            prompt = input("\nEnter a prompt (or 'exit' to quit): ")
            if prompt.lower() == 'exit':
                break
            interactive_generate(model, tokenizer, prompt)
    
    return 0

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Prune and fine-tune a transformer model")
    
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                        help="Model name or path (default: distilgpt2)")
    
    parser.add_argument("--pruning_percent", type=float, default=0.3,
                        help="Percentage of heads to prune (default: 0.3)")
    
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of fine-tuning epochs (default: 3)")
    
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training (default: 4)")
    
    parser.add_argument("--test_mode", action="store_true",
                        help="Use small test dataset for quick validation")
    
    parser.add_argument("--interactive", action="store_true",
                        help="Enable interactive text generation after training")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))