#!/usr/bin/env python
"""
Test script for the controller module.

This script demonstrates how to use the controller from the new sentinel package.
"""

import sys
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add the root directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import controller from the new location
from sentinel.controller import ControllerManager
from sentinel.controller.visualizations import GateVisualizer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test controller module")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="Model to use")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                      help="Device to use")
    parser.add_argument("--steps", type=int, default=10, help="Number of controller steps")
    return parser.parse_args()

def adapt_model_for_controller(model):
    """Adapt a HuggingFace model for use with the controller."""
    # This function adds gate parameters to the model's attention heads
    if hasattr(model, "transformer"):
        base_model = model.transformer
    elif hasattr(model, "model"):
        base_model = model.model
    else:
        base_model = model
    
    # We're assuming a standard transformer architecture
    # This part would need to be adjusted for specific models
    
    # Create blocks attribute expected by the controller
    model.blocks = []
    
    if hasattr(base_model, "h"):  # GPT-2 style
        layers = base_model.h
        for layer in layers:
            attn = layer.attn
            # Add gate parameter
            num_heads = attn.num_heads
            attn.gate = torch.nn.Parameter(torch.ones(num_heads))
            # Add to blocks
            model.blocks.append({"attn": attn})
    
    return model

def main():
    """Main function."""
    args = parse_args()
    
    # Load model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(args.device)
    
    # Adapt model for controller
    model = adapt_model_for_controller(model)
    
    # Create controller
    print("Creating controller")
    controller_config = {
        "controller_type": "ann",
        "controller_config": {
            "init_value": 3.0,
            "reg_weight": 1e-4
        },
        "update_frequency": 1,
        "warmup_steps": 0,
        "controller_lr": 0.1,
        "controller_lr_decay": 0.9,
        "controller_lr_decay_steps": 2,
        "enable_early_stopping": False
    }
    controller = ControllerManager(model, config=controller_config)
    
    # Create visualizer
    visualizer = GateVisualizer(output_dir="controller_output")
    
    # Run controller steps
    print(f"Running {args.steps} controller steps")
    for step in range(args.steps):
        # Create some random metrics
        metrics = {
            "entropy": torch.rand(len(model.blocks), model.blocks[0]["attn"].num_heads) * 2.0,
            "grad_norm": torch.rand(len(model.blocks), model.blocks[0]["attn"].num_heads) * 0.1
        }
        
        # Run a controller step
        result = controller.step(metrics_dict=metrics)
        
        # Print info
        print(f"Step {step+1}: Pruned {result['pruned_percent']:.1f}% of heads")
    
    # Visualize final state
    visualizer.visualize_gate_heatmap(model, title="Final Gate Values")
    
    print(f"Controller test complete. Gate heatmap saved to controller_output/gate_heatmap.png")
    print(f"Active gates per layer:")
    for layer_idx, heads in controller._get_active_gates().items():
        print(f"  Layer {layer_idx}: {len(heads)} active heads {heads}")

if __name__ == "__main__":
    main()