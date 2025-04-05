#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the evaluation module.

Tests the functionality of the text generation and evaluation utilities.
"""

import unittest
import os
import tempfile
import json
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.insert(0, project_root)

from utils.evaluation import evaluate_text_coherence, save_generated_samples


class TestEvaluationModule(unittest.TestCase):
    """Test cases for the evaluation module functions."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test samples
        self.samples = [
            {
                "prompt": "Once upon a time",
                "generated": "Once upon a time there was a kingdom far away where dragons and humans lived in harmony.",
                "sample_idx": 0
            },
            {
                "prompt": "The scientists discovered",
                "generated": "The scientists discovered a new species of deep sea creatures living near hydrothermal vents.",
                "sample_idx": 0
            },
            {
                "prompt": "In a world where",
                "generated": "In a world where technology has replaced human interaction, people seek meaningful connections.",
                "sample_idx": 0
            }
        ]
        
        # Create sample with repetition
        self.repetitive_sample = {
            "prompt": "Test prompt",
            "generated": "This is a test with repetition repetition repetition of words. This is a test with repetition.",
            "sample_idx": 0
        }
        
        # Create sample with error
        self.error_sample = {
            "prompt": "Error prompt",
            "error": "Generation failed with error XYZ"
        }
    
    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_evaluate_text_coherence(self):
        """Test coherence evaluation of generated text."""
        # Evaluate the samples
        evaluated = evaluate_text_coherence(self.samples)
        
        # Check that metrics were added to each sample
        for sample in evaluated:
            self.assertIn("metrics", sample)
            self.assertIn("length", sample["metrics"])
            self.assertIn("tokens", sample["metrics"])
            self.assertIn("repetition_2gram", sample["metrics"])
            self.assertIn("repetition_3gram", sample["metrics"])
            self.assertIn("prompt_relevance", sample["metrics"])
    
    def test_repetition_detection(self):
        """Test that the repetition metrics correctly identify repetitive text."""
        # Evaluate the repetitive sample
        evaluated = evaluate_text_coherence([self.repetitive_sample])
        
        # Check that repetition is detected
        self.assertGreater(evaluated[0]["metrics"]["repetition_2gram"], 0.1)
    
    def test_prompt_relevance(self):
        """Test that prompt relevance is correctly calculated."""
        # Create a sample that uses different words than the prompt
        irrelevant_sample = {
            "prompt": "Quantum physics",
            "generated": "The weather was beautiful today with clear skies and sunshine."
        }
        
        # Evaluate the sample
        evaluated = evaluate_text_coherence([irrelevant_sample])
        
        # Check that prompt relevance is low
        self.assertLess(evaluated[0]["metrics"]["prompt_relevance"], 0.5)
    
    def test_error_handling(self):
        """Test that samples with errors are handled correctly."""
        # Evaluate a mix of normal and error samples
        mixed_samples = self.samples + [self.error_sample]
        evaluated = evaluate_text_coherence(mixed_samples)
        
        # Check that we have the right number of results
        self.assertEqual(len(evaluated), len(mixed_samples))
        
        # Check that the error sample was passed through without adding metrics
        error_result = evaluated[-1]
        self.assertIn("error", error_result)
        self.assertNotIn("metrics", error_result)
    
    def test_save_generated_samples(self):
        """Test saving generated samples to a file."""
        # Create a samples dictionary
        samples_dict = {
            "baseline": self.samples,
            "pruned_0.3": self.samples[:-1] + [self.error_sample],
            "finetuned": self.samples
        }
        
        # Add metrics to the samples
        for strategy in samples_dict:
            samples_dict[strategy] = evaluate_text_coherence(samples_dict[strategy])
        
        # Save the samples
        output_path = os.path.join(self.temp_dir, "test_samples.txt")
        save_generated_samples(samples_dict, output_path, "TEST SAMPLES")
        
        # Check that the file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Check the file content
        with open(output_path, "r") as f:
            content = f.read()
            
            # Check that the title is there
            self.assertIn("TEST SAMPLES", content)
            
            # Check that all configurations are included
            for strategy in samples_dict:
                self.assertIn(f"Configuration: {strategy}", content)
            
            # Check that prompts are included
            for sample in self.samples:
                self.assertIn(f"Prompt: {sample['prompt']}", content)
            
            # Check that the error message is included
            self.assertIn("ERROR:", content)
            
            # Check that metrics are included
            self.assertIn("Metrics:", content)


if __name__ == "__main__":
    unittest.main()