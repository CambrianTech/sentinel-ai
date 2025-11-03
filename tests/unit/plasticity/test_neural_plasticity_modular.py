"""
Tests for the modular neural plasticity implementation.
"""

import os
import sys
import unittest
import tempfile
import pytest
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import from modular implementation
from scripts.neural_plasticity.run_experiment import (
    parse_args,
    create_simple_dataset,
    create_dataloader_builder
)

class TestNeuralPlasticityModular(unittest.TestCase):
    """Test the modular neural plasticity implementation."""
    
    def test_imports(self):
        """Test that all necessary modules can be imported."""
        try:
            from scripts.neural_plasticity.run_experiment import main
            from scripts.neural_plasticity.examples.minimal_example import main as minimal_main
            from sentinel.plasticity.plasticity_loop import PlasticityExperiment
            
            self.assertTrue(True)  # If we get here, imports succeeded
        except ImportError as e:
            self.fail(f"Import error: {e}")
    
    def test_argument_parser(self):
        """Test that argument parsing works correctly."""
        # Save original argv
        original_argv = sys.argv
        
        try:
            # Test with minimal arguments
            sys.argv = ['run_experiment.py', '--quick_test']
            args = parse_args()
            
            # Check default values
            self.assertEqual(args.model_name, 'distilgpt2')
            self.assertEqual(args.dataset, 'wikitext')
            self.assertEqual(args.batch_size, 4)
            self.assertEqual(args.pruning_strategy, 'entropy')
            self.assertEqual(args.pruning_level, 0.2)
            self.assertTrue(args.quick_test)
            
            # Test with custom arguments
            sys.argv = ['run_experiment.py', 
                       '--model_name', 'gpt2', 
                       '--dataset', 'tiny_shakespeare',
                       '--pruning_strategy', 'magnitude',
                       '--pruning_level', '0.3',
                       '--batch_size', '8']
            args = parse_args()
            
            # Check custom values
            self.assertEqual(args.model_name, 'gpt2')
            self.assertEqual(args.dataset, 'tiny_shakespeare')
            self.assertEqual(args.batch_size, 8)
            self.assertEqual(args.pruning_strategy, 'magnitude')
            self.assertEqual(args.pruning_level, 0.3)
            self.assertFalse(args.quick_test)
        finally:
            # Restore original argv
            sys.argv = original_argv
    
    @pytest.mark.skip(reason="Requires model downloads")
    def test_simple_dataset_creation(self):
        """Test creating a simple synthetic dataset."""
        from transformers import AutoTokenizer
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        
        # Create dataset
        dataset = create_simple_dataset(tokenizer, num_samples=10, seq_length=32)
        
        # Verify dataset properties
        self.assertEqual(len(dataset), 10)
        self.assertTrue('input_ids' in dataset.features)
        self.assertTrue('attention_mask' in dataset.features)
        self.assertTrue('labels' in dataset.features)
        
        # Check shapes
        self.assertEqual(dataset[0]['input_ids'].shape[0], 32)
        self.assertEqual(dataset[0]['attention_mask'].shape[0], 32)
        self.assertEqual(dataset[0]['labels'].shape[0], 32)
    
    @pytest.mark.skip(reason="Requires model downloads")
    def test_dataloader_builder(self):
        """Test the dataloader builder function."""
        from transformers import AutoTokenizer
        import torch
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        
        # Create mock args
        class Args:
            model_name = "distilgpt2"
            dataset = "wikitext"
            dataset_config = "wikitext-2-raw-v1"
            max_length = 32
            batch_size = 2
        
        args = Args()
        
        # Create dataloader builder with quick_test=True
        builder = create_dataloader_builder(args, tokenizer, quick_test=True)
        
        # Get dataloaders
        train_dataloader, eval_dataloader = builder()
        
        # Check dataloaders exist
        self.assertIsNotNone(train_dataloader)
        self.assertIsNotNone(eval_dataloader)
        
        # Get a batch from each
        train_batch = next(iter(train_dataloader))
        eval_batch = next(iter(eval_dataloader))
        
        # Verify batch contents
        self.assertTrue('input_ids' in train_batch)
        self.assertTrue('attention_mask' in train_batch)
        self.assertTrue('labels' in train_batch)
        
        # Check batch size
        self.assertEqual(train_batch['input_ids'].shape[0], 2)
        self.assertEqual(eval_batch['input_ids'].shape[0], 2)


if __name__ == "__main__":
    unittest.main()