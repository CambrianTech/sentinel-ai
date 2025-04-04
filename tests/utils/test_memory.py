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
    estimate_model_memory_usage,
    optimize_batch_size,
    set_jax_memory_preallocation
)


class TestMemoryManagement(unittest.TestCase):
    """Test memory management functions."""
    
    def test_memory_estimation(self):
        """Test memory estimation for different models."""
        # Test small models
        small_memory = estimate_model_memory_usage("distilgpt2")
        self.assertLess(small_memory, 2.0)  # Should be less than 2GB
        
        # Test medium models
        medium_memory = estimate_model_memory_usage("gpt2-medium")
        self.assertGreater(medium_memory, small_memory)
        
        # Test with explicit parameter count
        explicit_memory = estimate_model_memory_usage(parameters=350e6)
        self.assertGreater(explicit_memory, 1.0)  # Should be at least 1GB
    
    def test_batch_size_optimization(self):
        """Test batch size optimization."""
        # Test with small model and limited memory
        small_bs = optimize_batch_size("distilgpt2", gpu_memory_gb=2.0)
        self.assertLessEqual(small_bs, 16)  # Reasonable batch size for small GPU
        
        # Test with large model and limited memory
        large_bs = optimize_batch_size("gpt2-large", gpu_memory_gb=4.0)
        self.assertLessEqual(large_bs, 4)  # Should be small for large model
        
        # Test with plenty of memory
        large_mem_bs = optimize_batch_size("distilgpt2", gpu_memory_gb=24.0)
        self.assertGreaterEqual(large_mem_bs, 16)  # Can be larger with more memory
    
    def test_jax_memory_preallocation(self):
        """Test JAX memory preallocation setting."""
        # This is mostly a placeholder test since we can't easily test
        # the actual effect without running JAX
        try:
            # Should run without errors
            set_jax_memory_preallocation(0.8)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"set_jax_memory_preallocation raised {type(e)} unexpectedly")


if __name__ == "__main__":
    unittest.main()