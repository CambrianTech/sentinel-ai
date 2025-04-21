#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the neural plasticity run_experiment functionality
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO)

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Import from the modular implementation
from scripts.neural_plasticity.run_experiment import (
    parse_args,
    create_simple_dataset,
    create_dataloader_builder
)

# Import from sentinel package
from sentinel.pruning.plasticity_controller import PlasticityController, PruningMode
from sentinel.plasticity.plasticity_loop import PlasticityExperiment
from transformers import AutoTokenizer


class TestRunExperiment(unittest.TestCase):
    """Test the neural plasticity run_experiment functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.model_name = "distilgpt2"
        self.temp_dir = tempfile.mkdtemp()
    
    def test_parse_args(self):
        """Test the argument parser."""
        # Mock sys.argv
        orig_argv = sys.argv
        try:
            sys.argv = ['run_experiment.py', '--model_name', 'distilgpt2', '--quick_test']
            args = parse_args()
            
            self.assertEqual(args.model_name, 'distilgpt2')
            self.assertTrue(args.quick_test)
            self.assertEqual(args.pruning_strategy, 'entropy')  # Default value
            self.assertEqual(args.pruning_level, 0.2)  # Default value
        finally:
            sys.argv = orig_argv
    
    def test_create_simple_dataset(self):
        """Test creation of simple synthetic dataset."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Create a synthetic dataset
        dataset = create_simple_dataset(tokenizer, num_samples=5, seq_length=8)
        
        # Check that it has the correct shape and attributes
        self.assertEqual(len(dataset), 5)  # 5 samples
        self.assertEqual(dataset[0]['input_ids'].shape[0], 8)  # Sequence length
        self.assertTrue('input_ids' in dataset[0])
        self.assertTrue('attention_mask' in dataset[0])
        self.assertTrue('labels' in dataset[0])
    
    def test_create_dataloader_builder(self):
        """Test creation of dataloader builder function."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Mock args
        class Args:
            model_name = self.model_name
            dataset = "wikitext"
            dataset_config = "wikitext-2-raw-v1"
            max_length = 8
            batch_size = 2
        
        args = Args()
        
        # Create dataloader builder for quick test
        builder = create_dataloader_builder(args, tokenizer, quick_test=True)
        
        # Get dataloaders
        train_loader, eval_loader = builder()
        
        # Check dataloaders
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(eval_loader)
        
        # Check batch size
        batch = next(iter(train_loader))
        self.assertEqual(batch['input_ids'].shape[0], 2)  # Batch size is 2


if __name__ == "__main__":
    unittest.main()