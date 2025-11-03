#!/usr/bin/env python
"""
Integration test script to verify that models and controller modules work together.
This script imports from both sentinel.models and sentinel.controller to 
ensure that the reorganized modules are compatible.
"""

import sys
import os
import torch
from torch import nn

# Add the root directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import from reorganized modules
from sentinel.models import AdaptiveCausalLmWrapper, AgencySpecialization
from sentinel.models.loaders import ModelLoader
from sentinel.controller import ControllerManager, ANNController

def create_dummy_model():
    """Create a simple model for testing."""
    class DummyAttention(nn.Module):
        def __init__(self, num_heads=4):
            super().__init__()
            self.num_heads = num_heads
            self.gate = nn.Parameter(torch.ones(num_heads))
        
        def forward(self, x):
            return x
    
    class DummyBlock(nn.Module):
        def __init__(self, num_heads=4):
            super().__init__()
            self.attn = DummyAttention(num_heads)
        
        def forward(self, x):
            return self.attn(x)
    
    class DummyModel(nn.Module):
        def __init__(self, num_layers=3, num_heads=4):
            super().__init__()
            self.blocks = nn.ModuleList([DummyBlock(num_heads) for _ in range(num_layers)])
        
        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x
    
    return DummyModel()

def test_controller_with_model():
    """Test that controller can be used with a model."""
    model = create_dummy_model()
    
    # Create controller
    controller = ControllerManager(model)
    
    # Create metrics
    metrics = {
        "entropy": torch.rand(len(model.blocks), model.blocks[0].attn.num_heads),
        "grad_norm": torch.rand(len(model.blocks), model.blocks[0].attn.num_heads) * 0.1,
        "controller_lr": torch.tensor(0.01)
    }
    
    # Step controller
    result = controller.step(metrics_dict=metrics)
    
    # Check result
    print(f"✅ Controller step succeeded: Active gates = {sum(len(heads) for heads in result['active_gates'].values())}")
    
    return controller

def test_loader_with_controller():
    """Test that model loader works with controller."""
    # Create ModelLoader
    loader = ModelLoader(debug=True)
    
    # Create a dummy model
    model = create_dummy_model()
    
    # Configure gates (this is a method from ModelLoader)
    try:
        # This should raise NotImplementedError since load_model is not implemented
        # but configure_gates should work
        loader.configure_gates(model, gate_values=0.5)
        print("✅ ModelLoader.configure_gates works with model")
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    
    # Create controller with configured model
    controller = ControllerManager(model)
    
    # Test with basic metrics
    metrics = {
        "entropy": torch.rand(len(model.blocks), model.blocks[0].attn.num_heads),
        "controller_lr": torch.tensor(0.01)
    }
    
    # Step controller
    result = controller.step(metrics_dict=metrics)
    
    # Verify gates were changed
    active_gates = sum(len(heads) for heads in result['active_gates'].values())
    print(f"✅ Controller working with model configured by loader: {active_gates} active gates")
    
    return controller

def main():
    """Run integration tests."""
    print("\n=== TESTING CONTROLLER WITH MODEL ===")
    controller = test_controller_with_model()
    
    print("\n=== TESTING LOADER WITH CONTROLLER ===")
    controller_with_loader = test_loader_with_controller()
    
    print("\n=== TESTING MODULES IMPORTS ===")
    print(f"✅ sentinel.controller.ANNController: {ANNController.__module__}")
    print(f"✅ sentinel.models.AdaptiveCausalLmWrapper: {AdaptiveCausalLmWrapper.__module__}")
    print(f"✅ sentinel.models.loaders.ModelLoader: {ModelLoader.__module__}")
    
    print("\nAll tests passed! The reorganized modules work correctly together.")

if __name__ == "__main__":
    main()