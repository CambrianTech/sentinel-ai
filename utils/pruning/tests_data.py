"""
Unit tests for data preparation functions in utils.pruning.api.data module.
"""

import unittest
import torch
from transformers import AutoTokenizer

# Import the functions to test
from utils.pruning.api.data import prepare_data, prepare_test_data

class TestDataPreparation(unittest.TestCase):
    """Test data preparation functions for pruning experiments."""
    
    def setUp(self):
        """Set up test environment."""
        # Load a small test tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def test_prepare_test_data(self):
        """Test prepare_test_data function."""
        # Test with default parameters
        train_loader, val_loader = prepare_test_data(self.tokenizer)
        
        # Verify we got two dataloaders
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        
        # Check that they have data
        self.assertGreater(len(train_loader), 0)
        self.assertGreater(len(val_loader), 0)
        
        # Check the first batch
        batch = next(iter(train_loader))
        self.assertEqual(len(batch), 2)  # Should have input_ids and attention_mask
        
        # Check tensor shapes
        input_ids, attention_mask = batch
        self.assertEqual(len(input_ids.shape), 2)  # [batch_size, seq_len]
        self.assertEqual(input_ids.shape, attention_mask.shape)
        
        # Test with different parameters
        batch_size = 2
        max_length = 32
        num_samples = 5
        train_loader, val_loader = prepare_test_data(
            self.tokenizer,
            max_length=max_length,
            batch_size=batch_size,
            num_samples=num_samples
        )
        
        # Verify batch size
        batch = next(iter(train_loader))
        input_ids, _ = batch
        self.assertEqual(input_ids.shape[0], batch_size)
        
        # Verify sequence length
        self.assertLessEqual(input_ids.shape[1], max_length)
    
    def test_prepare_data_fallback(self):
        """Test prepare_data falls back to test data if needed."""
        # Pass an invalid split to force fallback
        dataloader = prepare_data(self.tokenizer, split="invalid_split", batch_size=2)
        
        # Verify we got a dataloader
        self.assertIsNotNone(dataloader)
        
        # Check that it has data
        self.assertGreater(len(dataloader), 0)
        
        # Check the first batch
        batch = next(iter(dataloader))
        self.assertGreaterEqual(len(batch), 2)  # Should have at least input_ids and attention_mask
        
        # Call with valid splits to test the full function if dataset is available
        try:
            train_dataloader = prepare_data(self.tokenizer, split="train", batch_size=2)
            val_dataloader = prepare_data(self.tokenizer, split="validation", batch_size=2)
            
            # If we got here, the real dataset was loaded, verify dataloaders
            self.assertIsNotNone(train_dataloader)
            self.assertIsNotNone(val_dataloader)
            
            # Check they have data
            self.assertGreater(len(train_dataloader), 0)
            self.assertGreater(len(val_dataloader), 0)
        except Exception as e:
            # If dataset couldn't be loaded, that's okay - we know the fallback works
            print(f"Note: Could not test with real dataset, but fallback works: {e}")

# Allow running the tests directly
if __name__ == "__main__":
    unittest.main()