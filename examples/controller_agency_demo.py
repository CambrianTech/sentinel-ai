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

# Import Sentinel-AI components from the reorganized structure
# Note: Models are still in the original location
from models.loaders.gpt2_loader import load_gpt2_with_sentinel_gates
from models.agency_specialization import AgencySpecialization

# Controller has been moved to sentinel package
from sentinel.controller import ControllerManager
from sentinel.controller.visualizations import AgencyVisualizer

# The HeadLRManager is now in the pruning module
from utils.pruning.head_lr_manager import HeadLRManager

def print_separator(title=""):
    """Print a separator with optional title."""
    width = 80
    if title:
        print("\n" + "=" * width)
        print(f"{title.center(width)}")
        print("=" * width)
    else:
        print("\n" + "-" * width)

def main():
    """Run the controller-agency integration demo."""
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print_separator("Loading Model with Agency")
    
    # Load a small GPT-2 model with agency-enabled heads
    model = load_gpt2_with_sentinel_gates("distilgpt2", device=device)
    
    # Attach agency specialization to each attention head
    specialization = AgencySpecialization(model)
    
    # Assign task specializations to heads
    specialization.assign_specializations({
        (0, 0): "syntactic parsing",
        (0, 1): "semantic analysis",
        (0, 2): "topic classification",
        (1, 0): "entity recognition",
        (1, 1): "sentiment analysis",
        (1, 2): "coreference resolution",
        (2, 0): "logical reasoning",
        (2, 1): "factual recall",
        (2, 2): "summarization"
    })
    
    print("Model loaded with agency-aware attention heads")
    print(f"Model has {len(model.blocks)} layers with {model.blocks[0]['attn'].num_heads} heads each")
    
    # Initialize some heads with different agency states
    print("\nInitializing head agency states:")
    specialization.set_head_state(0, 0, "active", consent=True)
    specialization.set_head_state(0, 1, "active", consent=True)
    specialization.set_head_state(0, 2, "misaligned", consent=True)
    specialization.set_head_state(1, 0, "active", consent=True)
    specialization.set_head_state(1, 1, "overloaded", consent=True)
    specialization.set_head_state(1, 2, "active", consent=True)
    specialization.set_head_state(2, 0, "withdrawn", consent=False)
    specialization.set_head_state(2, 1, "active", consent=True)
    specialization.set_head_state(2, 2, "active", consent=True)
    
    print_separator("Setting Up Controller")
    
    # Set up controller with agency awareness
    controller_config = {
        "update_frequency": 1,  # Update every step for demo
        "controller_lr": 0.1,   # Higher learning rate for visible changes
        "warmup_steps": 0,      # No warmup for demo
        "enable_early_stopping": False
    }
    
    controller = ControllerManager(model, config=controller_config)
    
    # Create head LR manager for adjusting learning rates
    head_lr_manager = HeadLRManager(
        num_layers=len(model.blocks),
        num_heads=model.blocks[0]["attn"].num_heads,
        base_lr=0.0001,
        new_head_lr_factor=5.0
    )
    
    # Set up visualizer
    visualizer = AgencyVisualizer(model, controller, head_lr_manager)
    
    print_separator("Simulating Controller Updates with Agency")
    
    # Collect agency state information
    agency_state = specialization.get_agency_state()
    
    # Simulate controller steps
    for step in range(10):
        print(f"\nStep {step+1}:")
        
        # Generate random metrics for demonstration
        metrics = {
            "entropy": torch.rand(len(model.blocks), model.blocks[0]["attn"].num_heads) * 2.0,
            "grad_norm": torch.rand(len(model.blocks), model.blocks[0]["attn"].num_heads) * 0.1,
            "importance": torch.rand(len(model.blocks), model.blocks[0]["attn"].num_heads)
        }
        
        # Execute controller step with agency awareness
        result = controller.step(
            metrics_dict=metrics,
            head_lr_manager=head_lr_manager,
            agency_state=agency_state
        )
        
        # Check for agency signals from controller
        signals = result.get("agency_signals", {})
        if signals.get("count", 0) > 0:
            print(f"  Controller emitted {signals.get('count', 0)} signals to heads")
            for signal in signals.get("signals_emitted", []):
                print(f"  Layer {signal['layer']}, Head {signal['head']}: "
                      f"{signal['from_state']} → {signal['to_state']}")
        
        # Update agency states based on utilization and metrics
        for layer_idx in range(len(model.blocks)):
            for head_idx in range(model.blocks[0]["attn"].num_heads):
                key = (layer_idx, head_idx)
                if key in agency_state:
                    # Skip withdrawn heads
                    if agency_state[key].get("state") == "withdrawn":
                        continue
                        
                    # Randomly create stress for a head
                    if step == 3 and layer_idx == 1 and head_idx == 2:
                        print(f"  Head (1, 2) becoming overloaded due to high utilization")
                        specialization.set_head_state(1, 2, "overloaded", consent=True)
                        agency_state = specialization.get_agency_state()
                    
                    # Random recovery from stress
                    if step == 6 and layer_idx == 0 and head_idx == 2:
                        print(f"  Head (0, 2) recovering from misalignment")
                        specialization.set_head_state(0, 2, "active", consent=True)
                        agency_state = specialization.get_agency_state()
                    
                    # Random withdrawal
                    if step == 8 and layer_idx == 1 and head_idx == 1:
                        print(f"  Head (1, 1) withdrawing consent after continued overload")
                        specialization.set_head_state(1, 1, "withdrawn", consent=False)
                        agency_state = specialization.get_agency_state()
        
        # Update visualizer history
        visualizer.update_history(timestamp=step)
        
        # For demo purposes, we'll sleep to make changes visible
        time.sleep(0.5)
    
    print_separator("Generating Visualization")
    
    # Create visualization dashboard
    dashboard = visualizer.create_dashboard()
    
    # Show plots
    plt.show()
    
    print_separator("Demo Complete")

if __name__ == "__main__":
    main()