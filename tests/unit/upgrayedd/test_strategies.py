"""
Unit tests for pruning strategies in the sentinel.upgrayedd.strategies module.

These tests verify that the pruning strategies correctly identify and prune
attention heads according to their respective algorithms.
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

# Add project root to path if running this file directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

# Import strategies to be tested
from sentinel.upgrayedd.strategies.entropy import entropy_based_pruning, compute_attention_entropy
from sentinel.upgrayedd.strategies.magnitude import magnitude_based_pruning, collect_weight_magnitudes  
from sentinel.upgrayedd.strategies.random import random_pruning


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


class TestPruningStrategies(unittest.TestCase):
    """Test case for pruning strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MockTransformerModel(num_layers=4, num_heads=4)
        self.dataloader = MockDataLoader()
        
    def test_random_pruning(self):
        """Test that random pruning selects the correct number of heads."""
        prune_ratio = 0.25  # Prune 4 layers * 4 heads * 0.25 = 4 heads
        pruned_heads = random_pruning(
            self.model, 
            self.dataloader,
            prune_ratio=prune_ratio,
            device="cpu"
        )
        
        # Check that the correct number of heads were pruned
        self.assertEqual(len(pruned_heads), 4)
        
        # Check that all pruned heads are valid (layer, head) tuples
        for layer, head in pruned_heads:
            self.assertGreaterEqual(layer, 0)
            self.assertLess(layer, 4)
            self.assertGreaterEqual(head, 0)
            self.assertLess(head, 4)
    
    @patch("sentinel.upgrayedd.strategies.entropy.collect_attention_distributions")
    def test_entropy_pruning(self, mock_collect_attn):
        """Test that entropy pruning selects heads with highest entropy."""
        # Create mock attention distributions
        mock_attn_dists = {}
        for layer in range(4):
            # Create attention distributions where entropy increases with head index
            # So head 3 should have highest entropy and be pruned first
            attn = torch.zeros(2, 4, 10, 10)  # [batch, heads, seq, seq]
            for head in range(4):
                # More uniform (higher entropy) for higher head indices
                if head == 0:
                    # Very focused attention (low entropy)
                    attn[:, head, :, 0] = 0.9
                    attn[:, head, :, 1:] = 0.1 / 9
                else:
                    # Increasingly uniform attention (higher entropy)
                    attn[:, head] = torch.ones(10, 10) / (10 - head)
            
            mock_attn_dists[layer] = attn
            
        mock_collect_attn.return_value = mock_attn_dists
        
        # Test entropy pruning
        prune_ratio = 0.25  # Prune 4 layers * 4 heads * 0.25 = 4 heads
        pruned_heads = entropy_based_pruning(
            self.model, 
            self.dataloader,
            prune_ratio=prune_ratio,
            device="cpu"
        )
        
        # Check that the correct number of heads were pruned
        self.assertEqual(len(pruned_heads), 4)
        
        # Since we set up entropy to increase with head index,
        # we expect the highest head indices to be pruned first
        for layer, head in pruned_heads:
            self.assertGreaterEqual(head, 2, "Lower entropy heads should not be pruned")
    
    @patch("sentinel.upgrayedd.strategies.magnitude.collect_weight_magnitudes")
    def test_magnitude_pruning(self, mock_collect_magnitudes):
        """Test that magnitude pruning selects heads with smallest magnitudes."""
        # Create mock weight magnitudes for each layer and head
        mock_magnitudes = torch.zeros(4, 4)  # [layers, heads]
        for layer in range(4):
            for head in range(4):
                # Higher magnitude for higher head indices
                mock_magnitudes[layer, head] = 0.1 * (head + 1)
        
        mock_collect_magnitudes.return_value = mock_magnitudes
        
        # Test magnitude pruning
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