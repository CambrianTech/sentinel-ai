"""
Unit tests for magnitude-based pruning strategy.

These tests verify that the magnitude-based pruning strategy correctly identifies and
prunes attention heads based on the magnitude of their weights.
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

# Add project root to path if running this file directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

# Import the strategy to test
from sentinel.upgrayedd.strategies.magnitude import magnitude_based_pruning, collect_weight_magnitudes


# Mock transformer model for testing
class MockTransformerModel(nn.Module):
    def __init__(self, num_layers=4, num_heads=4):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.config = MagicMock()
        self.config.num_hidden_layers = num_layers
        self.config.num_attention_heads = num_heads
        
        # Create mock attention layers
        self.layers = nn.ModuleList([MockAttentionLayer(num_heads) for _ in range(num_layers)])
        
    def named_parameters(self):
        """Mock named_parameters to return weights with predictable magnitudes."""
        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                # Higher magnitude for higher head indices
                magnitude = 0.1 * (head_idx + 1)
                yield f"layers.{layer_idx}.attention.query.weight", torch.ones(64, 64) * magnitude
            
    def named_modules(self):
        modules = [("", self)]
        for i, layer in enumerate(self.layers):
            modules.append((f"layers.{i}.attention", layer))
        return modules


class MockAttentionLayer(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pruned_heads = set()
        
    def forward(self, x):
        # Mock attention computation
        return x


# Mock dataloader for testing
class MockDataLoader:
    def __init__(self, batch_size=2, seq_len=10, vocab_size=100):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
    def __iter__(self):
        # Return one batch for testing
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        attention_mask = torch.ones_like(input_ids)
        return iter([(input_ids, attention_mask)])


class TestMagnitudePruningStrategy(unittest.TestCase):
    """Test case for magnitude-based pruning strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MockTransformerModel(num_layers=4, num_heads=4)
        self.dataloader = MockDataLoader()
        
    def test_collect_weight_magnitudes(self):
        """Test that weight magnitudes are collected correctly."""
        # Test collection of magnitudes
        magnitudes = collect_weight_magnitudes(self.model)
        
        # Check shape
        self.assertEqual(magnitudes.shape, (4, 4))  # 4 layers, 4 heads
        
        # Check values (magnitudes should increase with head index)
        for layer_idx in range(4):
            for head_idx in range(3):
                self.assertLess(
                    magnitudes[layer_idx, head_idx], 
                    magnitudes[layer_idx, head_idx + 1],
                    f"Magnitude should increase with head index, but {magnitudes[layer_idx, head_idx]} >= {magnitudes[layer_idx, head_idx + 1]}"
                )
    
    def test_magnitude_pruning(self):
        """Test that pruning selects heads with smallest magnitudes."""
        # Test pruning
        prune_ratio = 0.25  # Prune 4 layers * 4 heads * 0.25 = 4 heads
        pruned_heads = magnitude_based_pruning(
            self.model, 
            self.dataloader,
            prune_ratio=prune_ratio,
            device="cpu"
        )
        
        # Check that the correct number of heads were pruned
        self.assertEqual(len(pruned_heads), 4)
        
        # Since we set up magnitudes to increase with head index,
        # we expect the lowest head indices to be pruned first
        for layer, head in pruned_heads:
            self.assertLess(head, 2, "Higher magnitude heads should not be pruned")


if __name__ == "__main__":
    unittest.main()