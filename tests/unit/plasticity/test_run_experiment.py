"""
Comprehensive tests for the neural plasticity run_experiment.py script.

This test suite verifies the functionality of the main neural plasticity experiment runner,
ensuring that it works properly across different environments and configurations.
"""

import os
import sys
import unittest
import tempfile
from unittest.mock import patch, MagicMock, ANY
import pytest
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import modules to test
from scripts.neural_plasticity.run_experiment import (
    parse_args, 
    create_simple_dataset, 
    create_dataloader_builder,
    main
)

class TestRunExperiment(unittest.TestCase):
    """Test the run_experiment.py script."""
    
    def setUp(self):
        """Setup for tests."""
        # Create a temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Cleanup after tests."""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('scripts.neural_plasticity.run_experiment.PlasticityExperiment')
    def test_main_quick_test(self, mock_experiment):
        """Test main function with quick test flag."""
        # Setup mock experiment
        mock_instance = MagicMock()
        mock_experiment.return_value = mock_instance
        mock_instance.run_experiment.return_value = {
            'metrics': {
                'baseline': {'loss': 10.0, 'perplexity': 1000.0},
                'final': {'loss': 5.0, 'perplexity': 500.0}
            },
            'pruned_heads': [(0, 0, 0.1), (0, 1, 0.2)]
        }
        
        # Mock arguments
        with patch('sys.argv', ['run_experiment.py', '--quick_test', '--output_dir', self.temp_dir]):
            # Run main function
            result = main()
            
            # Verify PlasticityExperiment was created with correct parameters
            mock_experiment.assert_called_once_with(
                model_name=ANY,
                output_dir=self.temp_dir,
                device=ANY,
                adaptive_model=True
            )
            
            # Verify run_experiment was called
            mock_instance.run_experiment.assert_called_once()
            
            # Verify result was returned
            self.assertEqual(result['metrics']['baseline']['perplexity'], 1000.0)
            self.assertEqual(result['metrics']['final']['perplexity'], 500.0)
            self.assertEqual(len(result['pruned_heads']), 2)
    
    def test_create_dataloader_builder_with_synthetic_data(self):
        """Test creating dataloader builder with synthetic data."""
        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.vocab_size = 10000
        
        # Mock args
        class MockArgs:
            model_name = "distilgpt2"
            dataset = "wikitext"
            dataset_config = "wikitext-2-raw-v1"
            max_length = 32
            batch_size = 2
        
        args = MockArgs()
        
        # Create builder with quick_test=True
        builder = create_dataloader_builder(args, tokenizer, quick_test=True)
        
        # Verify function was returned
        self.assertTrue(callable(builder))
    
    @patch('scripts.neural_plasticity.run_experiment.load_dataset')
    def test_dataloader_error_handling(self, mock_load_dataset):
        """Test error handling when loading dataset fails."""
        # Mock load_dataset to raise exception
        mock_load_dataset.side_effect = Exception("Dataset not found")
        
        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.vocab_size = 10000
        
        # Mock args
        class MockArgs:
            model_name = "distilgpt2"
            dataset = "wikitext"
            dataset_config = "wikitext-2-raw-v1"
            max_length = 32
            batch_size = 2
        
        args = MockArgs()
        
        # Create builder with quick_test=False (to force real dataset loading attempt)
        builder = create_dataloader_builder(args, tokenizer, quick_test=False)
        
        # Get dataloaders - should fall back to synthetic data
        with patch('scripts.neural_plasticity.run_experiment.logger') as mock_logger:
            train_dataloader, eval_dataloader = builder()
            
            # Verify fallback message was logged
            mock_logger.error.assert_called_with(f"Error loading dataset: Dataset not found")
            mock_logger.info.assert_called_with("Falling back to synthetic dataset")
        
        # Verify dataloaders were created
        self.assertIsNotNone(train_dataloader)
        self.assertIsNotNone(eval_dataloader)

    def test_simple_dataset_creation(self):
        """Test creating a simple synthetic dataset without requiring real models."""
        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.vocab_size = 10000
        
        # Create dataset
        dataset = create_simple_dataset(tokenizer, num_samples=5, seq_length=16)
        
        # Verify dataset properties
        self.assertEqual(len(dataset), 5)
        self.assertIn('input_ids', dataset.features)
        self.assertIn('attention_mask', dataset.features)
        self.assertIn('labels', dataset.features)
        
        # Check shapes
        self.assertEqual(dataset[0]['input_ids'].shape[0], 16)
        self.assertEqual(dataset[0]['attention_mask'].shape[0], 16)
        self.assertEqual(dataset[0]['labels'].shape[0], 16)

if __name__ == '__main__':
    unittest.main()