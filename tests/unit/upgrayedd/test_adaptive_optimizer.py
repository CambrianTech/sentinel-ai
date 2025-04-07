"""
Unit tests for the AdaptiveOptimizer class.

These tests verify that the adaptive optimizer correctly performs pruning,
fine-tuning, and evaluation of transformer models.
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
import tempfile
import json

# Add project root to path if running this file directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

# Import the optimizer classes
from sentinel.upgrayedd.optimizer.adaptive_optimizer import AdaptiveOptimizer, AdaptiveOptimizerConfig


class TestAdaptiveOptimizer(unittest.TestCase):
    """Test case for the AdaptiveOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a minimal config for testing
        self.config = AdaptiveOptimizerConfig(
            model_name="distilgpt2",
            pruning_ratio=0.2,
            strategy="random",
            epochs_per_cycle=1,
            max_cycles=1,
            device="cpu",
            batch_size=2,
            dataset="wikitext"
        )
        
        # Create a temporary directory for output
        self.temp_dir = tempfile.mkdtemp()
        self.config.output_dir = self.temp_dir
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch("sentinel.upgrayedd.optimizer.adaptive_optimizer.load_model_and_tokenizer")
    def test_initialization(self, mock_load_model):
        """Test that the optimizer initializes correctly."""
        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        # Create optimizer
        optimizer = AdaptiveOptimizer(self.config)
        
        # Check that config was correctly set
        self.assertEqual(optimizer.config.model_name, "distilgpt2")
        self.assertEqual(optimizer.config.pruning_ratio, 0.2)
        self.assertEqual(optimizer.config.strategy, "random")
        
        # Check that output directory was created
        self.assertTrue(os.path.exists(self.config.output_dir))
        
        # Check that config file was created
        self.assertTrue(os.path.exists(os.path.join(self.config.output_dir, "config.json")))
    
    @patch("sentinel.upgrayedd.optimizer.adaptive_optimizer.load_model_and_tokenizer")
    @patch("sentinel.upgrayedd.utils.data.load_and_prepare_data")
    def test_load_model_and_data(self, mock_load_data, mock_load_model):
        """Test that the optimizer loads model and data correctly."""
        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        # Mock data loaders
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_load_data.return_value = (mock_train_loader, mock_val_loader)
        
        # Create optimizer
        optimizer = AdaptiveOptimizer(self.config)
        
        # Test load_model
        model, tokenizer = optimizer.load_model()
        self.assertEqual(model, mock_model)
        self.assertEqual(tokenizer, mock_tokenizer)
        mock_load_model.assert_called_once_with(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            device=self.config.device
        )
        
        # Test load_data
        train_loader, val_loader = optimizer.load_data()
        self.assertEqual(train_loader, mock_train_loader)
        self.assertEqual(val_loader, mock_val_loader)
        mock_load_data.assert_called_once_with(
            self.config.dataset,
            optimizer.tokenizer,
            batch_size=self.config.batch_size,
            dataset_path=self.config.dataset_path
        )
    
    @patch("sentinel.upgrayedd.optimizer.adaptive_optimizer.load_model_and_tokenizer")
    @patch("sentinel.upgrayedd.optimizer.adaptive_optimizer.random_pruning")
    def test_prune_model(self, mock_prune, mock_load_model):
        """Test that the optimizer correctly prunes models."""
        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        # Mock pruning result
        mock_pruned_heads = [(0, 1), (1, 2)]
        mock_prune.return_value = mock_pruned_heads
        
        # Create optimizer
        optimizer = AdaptiveOptimizer(self.config)
        optimizer.val_dataloader = MagicMock()
        
        # Test prune_model
        pruned_heads = optimizer.prune_model()
        self.assertEqual(pruned_heads, mock_pruned_heads)
        mock_prune.assert_called_once_with(
            optimizer.model,
            optimizer.val_dataloader,
            prune_ratio=self.config.pruning_ratio,
            device=self.config.device
        )
    
    @patch("sentinel.upgrayedd.optimizer.adaptive_optimizer.load_model_and_tokenizer")
    @patch("sentinel.upgrayedd.optimizer.adaptive_optimizer.evaluate_model")
    @patch("sentinel.upgrayedd.optimizer.adaptive_optimizer.generate_text")
    def test_evaluate(self, mock_generate, mock_evaluate, mock_load_model):
        """Test that the optimizer correctly evaluates models."""
        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        # Mock evaluation results
        mock_evaluate.return_value = (2.0, 7.38)  # Loss, perplexity
        mock_generate.return_value = "Generated text for testing"
        
        # Create optimizer
        optimizer = AdaptiveOptimizer(self.config)
        optimizer.val_dataloader = MagicMock()
        
        # Test evaluate
        metrics = optimizer.evaluate()
        
        # Check evaluation results
        self.assertEqual(metrics["loss"], 2.0)
        self.assertEqual(metrics["perplexity"], 7.38)
        self.assertEqual(metrics["generated_text"], "Generated text for testing")
        
        # Check that evaluation methods were called
        mock_evaluate.assert_called_once_with(
            optimizer.model,
            optimizer.val_dataloader,
            device=self.config.device
        )
        mock_generate.assert_called_once()
    
    @patch("sentinel.upgrayedd.optimizer.adaptive_optimizer.load_model_and_tokenizer")
    def test_save_checkpoint(self, mock_load_model):
        """Test that the optimizer correctly saves checkpoints."""
        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        # Create optimizer
        optimizer = AdaptiveOptimizer(self.config)
        optimizer.current_cycle = 1
        optimizer.pruned_heads = [(0, 1), (1, 2)]
        optimizer.baseline_metrics = {"loss": 2.5, "perplexity": 12.18}
        optimizer.current_metrics = {"loss": 2.0, "perplexity": 7.38}
        
        # Test save_checkpoint
        checkpoint_dir = optimizer.save_checkpoint()
        
        # Check that checkpoint directory was created
        self.assertTrue(os.path.exists(checkpoint_dir))
        
        # Check that model and tokenizer were saved
        mock_model.save_pretrained.assert_called_once_with(checkpoint_dir)
        mock_tokenizer.save_pretrained.assert_called_once_with(checkpoint_dir)
        
        # Check that optimizer state was saved
        state_path = os.path.join(checkpoint_dir, "optimizer_state.json")
        self.assertTrue(os.path.exists(state_path))
        
        # Check state file contents
        with open(state_path, "r") as f:
            state = json.load(f)
            self.assertEqual(state["cycle"], 1)
            self.assertEqual(state["pruned_heads"], [(0, 1), (1, 2)])
            self.assertEqual(state["baseline_metrics"]["loss"], 2.5)
            self.assertEqual(state["current_metrics"]["perplexity"], 7.38)


if __name__ == "__main__":
    unittest.main()