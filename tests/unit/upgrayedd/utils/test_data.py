"""
Unit tests for data loading utilities in the sentinel.upgrayedd.utils.data module.

These tests verify that data loading, tokenization, and dataloader creation 
work correctly.
"""

import os
import sys
import unittest
import torch
from unittest.mock import MagicMock, patch
import tempfile

# Add project root to path if running this file directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

# Import utility functions to test
from sentinel.upgrayedd.utils.data import load_and_prepare_data, load_dataset, _create_dummy_dataloaders


class TestDataUtils(unittest.TestCase):
    """Test case for data utility functions."""
    
    def test_load_dataset(self):
        """Test that the load_dataset function handles circular imports correctly."""
        # This is mainly a test that the function exists and is callable
        with patch('sentinel.upgrayedd.utils.data.sys.modules') as mock_modules:
            # Setup mock for testing the sys.modules branch
            mock_modules.__contains__.return_value = True
            mock_modules.__getitem__.return_value = MagicMock()
            mock_modules.__getitem__.return_value.load_dataset = MagicMock()
            
            # Call the function (it should use the mocked module)
            with self.assertRaises(Exception):  # Will raise because the mock doesn't actually load datasets
                load_dataset("test_dataset")
                
            # Check it accessed the module
            mock_modules.__contains__.assert_called_with('datasets')
            mock_modules.__getitem__.assert_called_with('datasets')
    
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


if __name__ == "__main__":
    unittest.main()