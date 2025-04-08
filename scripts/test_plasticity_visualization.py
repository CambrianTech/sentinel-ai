#!/usr/bin/env python
"""
Test script for the neural plasticity visualization with head overlays.

This script demonstrates the new visualization feature that combines
gradient norm information with overlays showing pruned, revived, and vulnerable heads.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from torch.utils.data import DataLoader
from datasets import load_dataset

# Add project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from sentinel.pruning.plasticity_controller import create_plasticity_controller, PruningMode
from utils.pruning.visualization import plot_head_gradients_with_overlays


def test_visualization_direct():
    """Test the visualization function directly with synthetic data."""
    print("Testing visualization function with synthetic data...")
    
    # Create output directory for test results
    output_dir = "test_visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate synthetic data
    num_layers = 6
    heads_per_layer = 8
    grad_norms = np.random.rand(num_layers, heads_per_layer) * 0.1
    
    # Some heads have stronger gradients (learning more)
    grad_norms[2, 3] = 0.5
    grad_norms[4, 1] = 0.6
    grad_norms[0, 5] = 0.4
    
    # Create some pruned and revived heads
    pruned_heads = [(1, 2), (3, 4), (5, 1)]
    revived_heads = [(0, 2), (2, 6)]
    
    # Generate visualization
    fig = plot_head_gradients_with_overlays(
        grad_norms=grad_norms,
        pruned_heads=pruned_heads,
        revived_heads=revived_heads,
        vulnerable_threshold=0.03,
        title="Synthetic Test: Head Gradient Norms with Plasticity Status"
    )
    
    # Save figure
    fig.savefig(os.path.join(output_dir, "synthetic_test.png"), dpi=300, bbox_inches='tight')
    print(f"Synthetic test visualization saved to {output_dir}/synthetic_test.png")


def test_visualization_with_model():
    """Test the visualization with a real model and plasticity controller."""
    print("\nTesting visualization with real model and plasticity controller...")
    
    # Create output directory for test results
    output_dir = "test_visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load a small model
    model_name = "distilgpt2"  # Small model for quick testing
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set up tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load a small dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:50]")
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.with_format("torch")
    
    # Create dataloader
    dataloader = DataLoader(
        tokenized_dataset, 
        batch_size=2, 
        collate_fn=default_data_collator
    )
    
    # Create plasticity controller with aggressive settings for demonstration
    controller = create_plasticity_controller(
        model=model,
        mode=PruningMode.ADAPTIVE,
        high_entropy_threshold=0.7,  # Lower threshold to encourage pruning
        low_entropy_threshold=0.3,   # Higher threshold to encourage revival
        grad_threshold=1e-3,         # Higher threshold to encourage pruning
        min_zero_epochs=1            # Allow quicker revival for demo
    )
    
    # Run a few plasticity steps
    for i in range(3):
        print(f"\nRunning plasticity step {i+1}...")
        pruned, revived, metrics = controller.step(
            dataloader, 
            num_batches=1, 
            verbose=True,
            output_dir=output_dir  # This will generate and save visualizations
        )
        
        print(f"Step {i+1} results:")
        print(f"  Pruned: {len(pruned)} heads")
        print(f"  Revived: {len(revived)} heads")
        print(f"  Total pruned: {metrics['total_pruned']} heads")
        print(f"  Visualizations saved to: {output_dir}")
    
    # Generate a final custom visualization with a different threshold
    print("\nGenerating final custom visualization...")
    final_viz_path = os.path.join(output_dir, "final_custom_threshold.png")
    fig = controller.visualize_gradients_with_status(
        figsize=(14, 7),
        save_path=final_viz_path,
        vulnerable_threshold=0.005  # Different threshold to show more vulnerable heads
    )
    print(f"Final custom visualization saved to: {final_viz_path}")


if __name__ == "__main__":
    # Test both the direct function and with a real model
    test_visualization_direct()
    test_visualization_with_model()
    
    print("\nTests completed successfully!")
    print("Visualization types available:")
    print("1. Head gradient norms with pruning overlays: Shows gradients with ❌, ➕, and ⚠️ markers")
    print("2. Entropy dynamics: Shows attention entropy over time")
    print("3. Decision dynamics: Shows pruning/revival decisions over time")