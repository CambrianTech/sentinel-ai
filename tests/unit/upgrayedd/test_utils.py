"""
Unit tests for utility functions in the sentinel.upgrayedd.utils module.

These tests verify that the utility functions for data loading, model manipulation,
and training work correctly.
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
import tempfile

# Add project root to path if running this file directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

# Import utility functions to test
from sentinel.upgrayedd.utils.model_utils import load_model_and_tokenizer
from sentinel.upgrayedd.utils.data import load_and_prepare_data, _create_dummy_dataloaders
from sentinel.upgrayedd.utils.training import evaluate_model, fine_tune_model
from sentinel.upgrayedd.utils.generation import generate_text


class TestModelUtils(unittest.TestCase):
    """Test case for model utility functions."""
    
    @patch("sentinel.upgrayedd.utils.model_utils.AutoModelForCausalLM")
    @patch("sentinel.upgrayedd.utils.model_utils.AutoTokenizer")
    def test_load_model_and_tokenizer(self, mock_tokenizer_cls, mock_model_cls):
        """Test that models and tokenizers are loaded correctly."""
        # Create mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        # Test loading
        model, tokenizer = load_model_and_tokenizer(
            "distilgpt2",
            cache_dir=None,
            device="cpu"
        )
        
        # Check model loading
        mock_model_cls.from_pretrained.assert_called_once_with(
            "distilgpt2",
            cache_dir=None
        )
        
        # Check tokenizer loading
        mock_tokenizer_cls.from_pretrained.assert_called_once_with(
            "distilgpt2",
            cache_dir=None
        )
        
        # Check model is on correct device
        mock_model.to.assert_called_once_with("cpu")
        
        # Check returned values
        self.assertEqual(model, mock_model)
        self.assertEqual(tokenizer, mock_tokenizer)


class TestDataUtils(unittest.TestCase):
    """Test case for data utility functions."""
    
    def test_create_dummy_dataloaders(self):
        """Test that dummy dataloaders are created correctly."""
        # Create a tokenizer mock
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (15, 64)),
            "attention_mask": torch.ones(15, 64)
        }
        
        # Create dummy dataloaders
        train_loader, val_loader = _create_dummy_dataloaders(tokenizer, batch_size=2)
        
        # Check loader properties
        self.assertIsInstance(train_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(val_loader, torch.utils.data.DataLoader)
        self.assertEqual(train_loader.batch_size, 2)
        
        # Check that a batch can be retrieved
        batch = next(iter(train_loader))
        self.assertEqual(len(batch), 2)  # input_ids and attention_mask
        self.assertEqual(batch[0].shape[0], 2)  # batch size
    
    @patch("sentinel.upgrayedd.utils.data.load_dataset")
    def test_load_and_prepare_data_fallback(self, mock_load_dataset):
        """Test that data loading fallback works when dataset loading fails."""
        # Make load_dataset raise an exception
        mock_load_dataset.side_effect = Exception("Test exception")
        
        # Create a tokenizer mock
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (15, 64)),
            "attention_mask": torch.ones(15, 64)
        }
        
        # Test loading
        train_loader, val_loader = load_and_prepare_data(
            "wikitext",
            tokenizer,
            batch_size=2
        )
        
        # Check that fallback was used
        self.assertIsInstance(train_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(val_loader, torch.utils.data.DataLoader)


class TestTrainingUtils(unittest.TestCase):
    """Test case for training utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple model for testing
        self.model = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )
        
        # Create dummy data
        x = torch.randn(4, 10)
        y = torch.randn(4, 10)
        dataset = torch.utils.data.TensorDataset(x, y)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    
    @patch("sentinel.upgrayedd.utils.training.torch.nn.CrossEntropyLoss")
    def test_evaluate_model(self, mock_loss_fn):
        """Test model evaluation function."""
        # Create mock loss function
        mock_loss = MagicMock()
        mock_loss.return_value = torch.tensor(2.0)
        mock_loss_fn.return_value = mock_loss
        
        # Patch forward method to return logits and loss
        with patch.object(self.model, 'forward', return_value=MagicMock(loss=torch.tensor(2.0))):
            # Test evaluation
            loss, perplexity = evaluate_model(
                self.model,
                self.dataloader,
                device="cpu"
            )
            
            # Check results
            self.assertEqual(loss, 2.0)
            self.assertAlmostEqual(perplexity, 7.389, places=3)  # e^2


class TestGenerationUtils(unittest.TestCase):
    """Test case for text generation utility functions."""
    
    def test_generate_text(self):
        """Test text generation function."""
        # Create mock model and tokenizer
        model = MagicMock()
        tokenizer = MagicMock()
        
        # Configure mocks
        model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        tokenizer.decode.return_value = "Generated text for testing"
        
        # Test generation
        text = generate_text(
            model,
            tokenizer,
            "Test prompt",
            max_length=10
        )
        
        # Check result
        self.assertEqual(text, "Generated text for testing")
        
        # Check that model.generate was called
        model.generate.assert_called_once()
        
        # Check that tokenizer.decode was called
        tokenizer.decode.assert_called_once()


if __name__ == "__main__":
    unittest.main()