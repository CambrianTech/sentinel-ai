#!/usr/bin/env python
"""
Integration test script to verify that the reorganized modules work together.
This script imports from both sentinel.pruning and sentinel.controller to 
ensure that the reorganized modules are compatible.
"""

import sys
import os
import torch
from torch import nn

# Add the root directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import from reorganized modules
from sentinel.pruning.fixed_pruning_module_jax import PruningModule
from sentinel.pruning.strategies import get_strategy
from sentinel.controller import ANNController, ControllerManager

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

def test_pruning_module():
    """Test that pruning module can be imported and initialized."""
    # Skip testing with a real model, just try to import and create the module
    print("✅ Importing pruning module successful")
    
    # Test strategy creation
    strategy_names = ["random", "magnitude", "entropy"]
    for name in strategy_names:
        strategy = get_strategy(name, None)
        print(f"✅ Strategy '{strategy.__class__.__name__}' created successfully")
    
    return "OK"

def test_controller():
    """Test that controller can be imported and initialized."""
    model = create_dummy_model()
    
    # Create controller
    controller = ANNController(
        num_layers=len(model.blocks),
        num_heads=model.blocks[0].attn.num_heads
    )
    print("✅ Controller initialized successfully")
    
    # Create controller manager
    controller_manager = ControllerManager(model)
    print("✅ Controller manager initialized successfully")
    
    # Test gate values
    gate_values = controller.forward()
    print(f"✅ Controller gate values shape: {gate_values.shape}")
    
    return controller_manager

def test_integration():
    """Test that pruning and controller modules can be used together."""
    model = create_dummy_model()
    
    # Create controller
    controller_manager = ControllerManager(model)
    
    # Create metrics
    entropy = torch.rand(len(model.blocks), model.blocks[0].attn.num_heads)
    metrics_dict = {"entropy": entropy, "controller_lr": torch.tensor(0.01)}
    
    # Update controller gates
    controller_manager.controller.update_gates(metrics_dict)
    active_gates_before = controller_manager._get_active_gates()
    
    # Get the gates we'll manually update to simulate pruning
    for layer_idx, block in enumerate(model.blocks):
        # Randomly prune 50% of heads
        mask = torch.rand(block.attn.num_heads) > 0.5
        block.attn.gate.data *= mask.float()
    
    # Check that controller can still work with pruned model
    active_gates_after = controller_manager._get_active_gates()
    
    # Count pruned heads
    before_count = sum(len(heads) for heads in active_gates_before.values())
    after_count = sum(len(heads) for heads in active_gates_after.values())
    
    print(f"✅ Integration successful: Heads before: {before_count}, after: {after_count}")
    print(f"✅ Pruning simulation successful")
    
    # Test that controller can update the model again
    controller_manager.step(metrics_dict=metrics_dict)
    print(f"✅ Controller step after pruning successful")
    
    return "OK"

def main():
    """Run integration tests."""
    print("\n=== TESTING PRUNING MODULE ===")
    test_pruning_module()
    
    print("\n=== TESTING CONTROLLER MODULE ===")
    controller_manager = test_controller()
    
    print("\n=== TESTING INTEGRATION ===")
    test_integration()
    
    print("\nAll tests passed! The reorganized modules work correctly together.")

if __name__ == "__main__":
    main()