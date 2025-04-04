#!/usr/bin/env python
"""
Simplified training script for testing the reorganized imports
"""
import torch
from sentinel.models.loaders.loader import load_baseline_model, load_adaptive_model
from sentinel.controller.controller_manager import ControllerManager

def main():
    # Create a model
    device = torch.device("cpu")
    baseline_model = load_baseline_model("distilgpt2", device)
    model = load_adaptive_model("distilgpt2", baseline_model, device, debug=False, quiet=True)
    
    # Create controller
    controller = ControllerManager(model)
    
    print("Successfully imported and initialized model and controller.")
    print(f"Model type: {type(model)}")
    print(f"Controller type: {type(controller)}")

if __name__ == "__main__":
    main()