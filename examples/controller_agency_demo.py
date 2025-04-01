#!/usr/bin/env python
"""
Controller-Agency Integration Demo for Sentinel-AI

This example demonstrates how to:
1. Load a model with attention head agency
2. Set up a controller that respects agency state
3. Integrate agency with learning rate adjustments
4. Visualize the entire system working together

Usage:
  python controller_agency_demo.py
"""

import sys
import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import Sentinel-AI components
from models.loaders.gpt2_loader import load_gpt2_with_sentinel_gates
from models.agency_specialization import AgencySpecialization
from controller.controller_manager import ControllerManager
from controller.visualizations.agency_visualizer import AgencyVisualizer
from utils.head_lr_manager import HeadLRManager

def print_separator(title=""):
    """Print a separator with optional title."""
    width = 80
    if title:
        print("\n" + "=" * width)
        print(f"{title.center(width)}")
        print("=" * width)
    else:
        print("\n" + "-" * width)

def load_model_with_agency():
    """Load a model with agency-aware attention."""
    print("Loading model with agency-aware attention...")
    model, tokenizer = load_gpt2_with_sentinel_gates(
        model_name="gpt2",
        gate_init=1.0,
        norm_attn_output=True
    )
    
    # Apply agency specialization
    specialization = AgencySpecialization(model)
    specialization.initialize_specialization()
    
    return model, tokenizer, specialization

def setup_controller_and_lr_manager(model):
    """Set up controller and learning rate manager."""
    print("Setting up controller and learning rate manager...")
    
    # Create controller manager
    controller_config = {
        "controller_type": "ann",
        "update_frequency": 5,  # Update more frequently for demo
        "warmup_steps": 10,     # Short warmup for demo
        "controller_lr": 0.05,  # Higher learning rate for demo
        "controller_config": {
            "init_value": 2.0,  # Start with moderate gate values
            "reg_weight": 1e-4  # L1 regularization weight
        }
    }
    controller_manager = ControllerManager(model, controller_config)
    
    # Create optimizer (dummy for demo)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create learning rate manager
    head_lr_manager = HeadLRManager(
        model=model,
        optimizer=optimizer,
        base_lr=0.001,
        boost_factor=5.0,
        decay_factor=0.9,
        warmup_steps=5,     # Short warmup for demo
        cooldown_steps=20   # Short cooldown for demo
    )
    
    return controller_manager, head_lr_manager, optimizer

def simulate_training_step(model, controller_manager, head_lr_manager, specialization, step):
    """Simulate a training step with agency-controller interaction."""
    # Get agency state from the model
    agency_state = {}
    for layer_idx in range(model.num_layers):
        attn = model.blocks[layer_idx]["attn"]
        if hasattr(attn, "agency_signals"):
            for head_idx, signals in attn.agency_signals.items():
                agency_state[(layer_idx, head_idx)] = signals
    
    # Create dummy metrics dictionary
    batch_size = 4
    seq_len = 64
    device = next(model.parameters()).device
    
    dummy_metrics = {
        "entropy": torch.rand((model.num_layers, model.num_heads), device=device) * 3.0,
        "grad_norm": torch.rand((model.num_layers, model.num_heads), device=device) * 0.1,
        "head_importance": torch.rand((model.num_layers, model.num_heads), device=device)
    }
    
    # Update controller with agency state
    update_info = controller_manager.step(
        metrics_dict=dummy_metrics,
        head_lr_manager=head_lr_manager,
        agency_state=agency_state
    )
    
    # Log any agency signals emitted by controller
    if "agency_signals" in update_info and update_info["agency_signals"].get("count", 0) > 0:
        print(f"\nController emitted {update_info['agency_signals']['count']} agency signals:")
        for signal in update_info["agency_signals"].get("signals_emitted", []):
            print(f"  Layer {signal['layer']}, Head {signal['head']}: {signal['from_state']} → {signal['to_state']}")
    
    # Every few steps, simulate attention patterns that trigger state changes
    if step % 3 == 0:
        # Change some head states based on attention behavior patterns
        state_changes = []
        
        # Simulate some heads becoming overloaded from high activity
        if step % 9 == 0:
            overloaded_layer = step % model.num_layers
            overloaded_head = (step // 3) % model.num_heads
            model.set_head_state(overloaded_layer, overloaded_head, "overloaded")
            state_changes.append((overloaded_layer, overloaded_head, "active", "overloaded"))
        
        # Simulate some heads becoming misaligned
        if step % 6 == 3:
            misaligned_layer = (step // 3) % model.num_layers
            misaligned_head = (step + 1) % model.num_heads
            model.set_head_state(misaligned_layer, misaligned_head, "misaligned")
            state_changes.append((misaligned_layer, misaligned_head, "active", "misaligned"))
        
        # Log changes
        if state_changes:
            print(f"\nAttention behavior triggered {len(state_changes)} state changes:")
            for layer, head, from_state, to_state in state_changes:
                print(f"  Layer {layer}, Head {head}: {from_state} → {to_state}")
    
    # Return summary info
    return {
        "active_gates": len(update_info.get("active_gates", {}).get(0, [])),
        "total_heads": model.num_layers * model.num_heads,
        "state_changes": update_info.get("agency_signals", {}).get("count", 0)
    }

def main():
    print_separator("Sentinel-AI: Controller-Agency Integration Demo")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model with agency features
    model, tokenizer, specialization = load_model_with_agency()
    model = model.to(device)
    print(f"Model loaded with {model.num_layers} layers, {model.num_heads} heads per layer")
    
    # Set up controller and learning rate manager
    controller_manager, head_lr_manager, optimizer = setup_controller_and_lr_manager(model)
    
    # Create agency visualizer
    visualizer = AgencyVisualizer(model, controller_manager, head_lr_manager)
    
    # Simulate training steps
    print_separator("Simulating Training with Agency-Controller Integration")
    num_steps = 30
    
    for step in range(num_steps):
        print(f"\nStep {step+1}/{num_steps}")
        
        # Simulate a training step
        step_info = simulate_training_step(
            model, controller_manager, head_lr_manager, specialization, step
        )
        
        # Update visualizer history
        visualizer.update_history(step)
        
        # Print some stats
        print(f"Active gates: {step_info['active_gates']}/{step_info['total_heads']}")
        print(f"Controller-agency interactions: {step_info['state_changes']}")
        
        # Print current agency stats
        agency_report = model.get_agency_report()
        active_count = sum(layer_report.get("active_heads", 0) for layer_report in agency_report["layer_reports"].values())
        withdrawn_count = sum(layer_report.get("withdrawn_heads", 0) for layer_report in agency_report["layer_reports"].values())
        overloaded_count = sum(layer_report.get("overloaded_heads", 0) for layer_report in agency_report["layer_reports"].values())
        misaligned_count = sum(layer_report.get("misaligned_heads", 0) for layer_report in agency_report["layer_reports"].values())
        
        print(f"Agency states: {active_count} active, {overloaded_count} overloaded, {misaligned_count} misaligned, {withdrawn_count} withdrawn")
    
    # Create final visualization
    print_separator("Creating Final Visualization")
    dashboard = visualizer.create_dashboard()
    
    # Display plots
    plt.show()
    
    print("\nDemo complete!")
    print("The demo has shown how the controller dynamically adjusts gates based on agency states")
    print("and how attention heads signal state changes that influence controller decisions.")

if __name__ == "__main__":
    # Make sure the examples directory exists
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    main()