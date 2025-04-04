"""
Tests for memory management stability functions.
"""

import unittest
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root))

# Import the module to test
from utils.pruning.stability.memory_management import (
    estimate_model_memory,
    recommend_batch_size,
    optimize_training_parameters
)


class TestMemoryManagement(unittest.TestCase):
    """Test memory management functions."""
    
    def test_memory_estimation(self):
        """Test memory estimation for different models."""
        # Test small models
        small_memory = estimate_model_memory("distilgpt2")
        self.assertIsInstance(small_memory, dict)
        self.assertIn("total_gb", small_memory)
        self.assertLess(small_memory["total_gb"], 4.0)  # Should be less than 4GB
        
        # Test medium models
        medium_memory = estimate_model_memory("gpt2-medium")
        self.assertGreater(medium_memory["total_gb"], small_memory["total_gb"])
    
    def test_batch_size_recommendation(self):
        """Test batch size recommendation."""
        # Test with small model and limited memory
        small_bs = recommend_batch_size("distilgpt2", available_memory_gb=2.0)
        self.assertLessEqual(small_bs, 16)  # Reasonable batch size for small GPU
        
        # Test with large model and limited memory
        large_bs = recommend_batch_size("gpt2-large", available_memory_gb=4.0)
        self.assertLessEqual(large_bs, 4)  # Should be small for large model
        
        # Test with plenty of memory
        large_mem_bs = recommend_batch_size("distilgpt2", available_memory_gb=24.0)
        self.assertGreaterEqual(large_mem_bs, 16)  # Can be larger with more memory
    
    def test_training_parameter_optimization(self):
        """Test training parameter optimization."""
        # This tests the comprehensive optimization function
        try:
            # Should run without errors
            params = optimize_training_parameters("distilgpt2", available_memory_gb=4.0)
            self.assertIn("batch_size", params)
            self.assertIn("sequence_length", params)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"optimize_training_parameters raised {type(e)} unexpectedly")


if __name__ == "__main__":
    unittest.main()