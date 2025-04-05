"""
Tests for the pruning module structure and functionality.
This will help ensure our refactoring doesn't break existing functionality.
"""

import os
import sys
import unittest
import tempfile

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sentinel.pruning.fixed_pruning_module import FixedPruningModule
from sentinel.pruning.fixed_pruning_module_jax import PruningModule
from sentinel.pruning.strategies import get_strategy


class TestPruningStructure(unittest.TestCase):
    """Test the structure and basic functionality of the pruning module."""
    
    def test_fixed_pruning_module_exists(self):
        """Test that the FixedPruningModule class exists and can be instantiated."""
        module = FixedPruningModule(model_name="distilgpt2")
        self.assertIsNotNone(module)
        self.assertEqual(module.model_name, "distilgpt2")
    
    def test_jax_pruning_module_exists(self):
        """Test that the JAX PruningModule class exists and can be instantiated."""
        module = PruningModule(model_name="distilgpt2")
        self.assertIsNotNone(module)
        self.assertEqual(module.model_name, "distilgpt2")
    
    def test_pruning_strategies_exist(self):
        """Test that the pruning strategies are available."""
        strategy_names = ["random", "magnitude", "entropy"]
        for name in strategy_names:
            strategy_class = get_strategy(name, None)
            self.assertIsNotNone(strategy_class)
    
    def test_get_strategy_function(self):
        """Test the get_strategy function returns different strategy classes."""
        random_strategy = get_strategy("random", None)
        magnitude_strategy = get_strategy("magnitude", None)
        entropy_strategy = get_strategy("entropy", None)
        
        # Check they are different classes
        self.assertNotEqual(random_strategy.__class__, magnitude_strategy.__class__)
        self.assertNotEqual(random_strategy.__class__, entropy_strategy.__class__)
        self.assertNotEqual(magnitude_strategy.__class__, entropy_strategy.__class__)


if __name__ == "__main__":
    unittest.main()