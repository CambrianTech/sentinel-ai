#\!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Minimal Neural Plasticity Example

This script provides a minimal example of how to use the neural plasticity
functionality from the sentinel package.

Usage:
    python scripts/neural_plasticity/examples/minimal_example.py
"""

import os
import sys
import torch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path if needed
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Import from the sentinel package
from sentinel.pruning.plasticity_controller import create_plasticity_controller, PruningMode
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    """Run a minimal neural plasticity example."""
    # Load a model
    model_name = "distilgpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loaded model: {model_name}")
    
    # Create a plasticity controller
    controller = create_plasticity_controller(
        model=model,
        mode=PruningMode.ADAPTIVE,
        high_entropy_threshold=0.8,
        low_entropy_threshold=0.4,
        grad_threshold=1e-4
    )
    
    print(f"Created plasticity controller")
    
    # Generate some input data
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    inputs = tokenizer(
        ["This is a test sentence to analyze attention patterns."], 
        return_tensors="pt",
        padding=True
    )
    
    # Create a simple dataloader with one batch
    from torch.utils.data import TensorDataset, DataLoader
    
    dataset = TensorDataset(
        inputs["input_ids"], 
        inputs["attention_mask"]
    )
    dataloader = DataLoader(dataset, batch_size=1)
    
    # Collect head metrics
    print("Collecting head metrics...")
    entropy_values, grad_norm_values = controller.collect_head_metrics(dataloader)
    
    # Print metrics shape
    print(f"Entropy values shape: {entropy_values.shape}")
    print(f"Gradient norm values shape: {grad_norm_values.shape}")
    
    # Make pruning decisions
    print("Applying plasticity decisions...")
    pruned_heads, revived_heads = controller.apply_plasticity(
        entropy_values, 
        grad_norm_values,
        verbose=True
    )
    
    # Print pruning results
    print(f"Pruned {len(pruned_heads)} heads")
    if pruned_heads:
        print("Pruned heads:")
        for layer_idx, head_idx in pruned_heads:
            print(f"  Layer {layer_idx}, Head {head_idx}")
    
    # Get summary
    summary = controller.get_summary()
    print("\nPlasticity Summary:")
    print(f"Total Heads: {summary['total_heads']}")
    print(f"Pruned Heads: {summary['pruned_heads']}")
    print(f"Pruning Rate: {summary['pruning_rate']:.2%}")
    print(f"Model Size: {summary['model_size_mb']:.2f} MB")
    
    print("\nThis example shows how to use the plasticity controller to analyze and prune attention heads.")
    print("For a complete experiment, use the PlasticityExperiment class from sentinel.plasticity.plasticity_loop")


if __name__ == "__main__":
    main()
EOL < /dev/null