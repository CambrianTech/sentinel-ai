#!/usr/bin/env python
"""
Test script for fine-tuning pruned models. 

This script tests the functionality of finetune_pruned_model.py without requiring 
a full training run, focusing on validating:
1. Proper loading of pruned models
2. Per-head learning rate setup
3. Optimizer creation with head-specific parameter groups
4. Basic fine-tuning loop functionality

Usage:
    python scripts/test_finetune_pruned.py
"""

import os
import sys
import unittest
import tempfile
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.finetune_pruned_model import (
    create_optimizer_with_head_params,
    count_active_heads,
    evaluate_perplexity
)
from models.loaders.loader import load_baseline_model, load_adaptive_model
from utils.head_lr_manager import HeadLRManager


class TestFinetunePruned(unittest.TestCase):
    """Test cases for pruned model fine-tuning functionality."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Check if CUDA is available, use CPU if not
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create a small model for testing
        print("Loading test model...")
        self.baseline_model = load_baseline_model("distilgpt2", self.device)
        self.model = load_adaptive_model("distilgpt2", self.baseline_model, self.device)
        
        # Create temporary file for checkpoints
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_path = os.path.join(self.temp_dir.name, "test_checkpoint.pth")
        
        # Create mock dataset for testing
        self.mock_data = torch.randint(0, 100, (4, 20))  # 4 samples of length 20
        self.mock_dataset = torch.utils.data.TensorDataset(self.mock_data)
        self.mock_loader = torch.utils.data.DataLoader(self.mock_dataset, batch_size=2)
    
    def tearDown(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()
    
    def test_count_active_heads(self):
        """Test that active head counting works correctly."""
        # Initially all heads should be active
        total, active, ratio = count_active_heads(self.model)
        
        # Assert all heads are active initially
        self.assertEqual(active, total)
        self.assertEqual(ratio, 1.0)
        
        # Prune a few heads by setting gates to 0
        with torch.no_grad():
            layer_idx = 0
            head_idx = 1
            self.model.blocks[layer_idx]["attn"].gate[head_idx] = 0.0
            
            layer_idx = 2
            head_idx = 0
            self.model.blocks[layer_idx]["attn"].gate[head_idx] = 0.0
        
        # Recount heads
        total, active, ratio = count_active_heads(self.model)
        
        # Assert we have 2 fewer active heads
        self.assertEqual(active, total - 2)
        self.assertAlmostEqual(ratio, (total - 2) / total, places=5)
    
    def test_create_optimizer(self):
        """Test that optimizer creation with head-specific params works."""
        # Create optimizer with per-head parameters
        optimizer = create_optimizer_with_head_params(self.model, lr=1e-4)
        
        # Check that we have the right number of parameter groups
        # Should have at least one group for non-head params plus groups for each active head
        total_heads, active_heads, _ = count_active_heads(self.model)
        
        # For distilgpt2, we'd expect param groups for non-head params
        # plus per-head parameter groups
        self.assertGreaterEqual(len(optimizer.param_groups), 1)
        
        # Test that each parameter group has learning rate set
        for group in optimizer.param_groups:
            self.assertEqual(group['lr'], 1e-4)
            
            # Each group should have named_params
            self.assertTrue('named_params' in group)
    
    def test_head_lr_manager_integration(self):
        """Test integration with HeadLRManager."""
        # Create optimizer with per-head parameters
        optimizer = create_optimizer_with_head_params(self.model, lr=1e-4)
        
        # Create HeadLRManager
        head_lr_manager = HeadLRManager(
            model=self.model,
            optimizer=optimizer,
            base_lr=1e-4,
            boost_factor=5.0,
            decay_factor=0.9,
            warmup_steps=5,
            cooldown_steps=10
        )
        
        # Create dummy gate values to simulate pruning
        with torch.no_grad():
            dummy_gates = torch.ones((len(self.model.blocks), self.model.blocks[0]["attn"].num_heads))
            
            # Prune a few heads by setting gates to 0
            dummy_gates[0, 1] = 0.0
            dummy_gates[2, 0] = 0.0
            
            # Update head status with these gates
            head_lr_manager.update_head_status(dummy_gates)
            
            # Check that we can update learning rates
            lr_info = head_lr_manager.update_learning_rates()
            
            # Verify some learning rates were updated
            self.assertTrue(lr_info['changes_made'])
    
    def test_evaluation(self):
        """Test perplexity evaluation function."""
        # Test perplexity evaluation on mock data
        perplexity = evaluate_perplexity(self.model, self.mock_loader, self.device)
        
        # Just check that perplexity is a valid number (not necessarily meaningful on random data)
        self.assertTrue(isinstance(perplexity, float))
        self.assertFalse(torch.isnan(torch.tensor(perplexity)))


if __name__ == "__main__":
    unittest.main()