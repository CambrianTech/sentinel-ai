"""
Tests for the neural plasticity functionality.
"""

import os
import sys
import unittest
import tempfile

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

class TestNeuralPlasticity(unittest.TestCase):
    """Test the neural plasticity functionality."""
    
    def test_imports(self):
        """Test that all necessary modules can be imported."""
        try:
            from sentinel.pruning.fixed_pruning_module_jax import PruningModule
            from sentinel.pruning.strategies import get_strategy as get_pruning_strategy
            from sentinel.pruning.growth import (
                grow_attention_heads_gradually, 
                determine_active_heads,
                get_strategy as get_growth_strategy
            )
            from sentinel.pruning.head_lr_manager import HeadLRManager
            from utils.train_utils import FineTuner
            
            self.assertTrue(True)  # If we get here, imports succeeded
        except ImportError as e:
            self.fail(f"Import error: {e}")
    
    def test_plasticity_components(self):
        """Test that the core components of neural plasticity exist."""
        # Import components
        from sentinel.pruning.fixed_pruning_module_jax import PruningModule
        from sentinel.pruning.growth import grow_attention_heads_gradually, determine_active_heads
        
        # Create a test module
        module = PruningModule(model_name="distilgpt2")
        
        # Check that core functions exist and have the right signature
        self.assertTrue(hasattr(module, "load_model"))
        self.assertTrue(hasattr(module, "prune_head"))
        self.assertTrue(hasattr(module, "evaluate_perplexity"))
        self.assertTrue(hasattr(module, "generate_text"))
        
        # Check growth module functions exist
        self.assertTrue(callable(grow_attention_heads_gradually))
        self.assertTrue(callable(determine_active_heads))
    
    def test_pruning_strategies(self):
        """Test that pruning strategies exist and work."""
        from sentinel.pruning.strategies import get_strategy
        
        # Check that each strategy can be retrieved
        for strategy_name in ["random", "magnitude", "entropy"]:
            strategy = get_strategy(strategy_name, None)
            self.assertIsNotNone(strategy)
            self.assertTrue(hasattr(strategy, "get_head_importance"))
    
    def test_growth_strategies(self):
        """Test that growth strategies exist and work."""
        from sentinel.pruning.growth import get_strategy
        from sentinel.pruning.fixed_pruning_module_jax import PruningModule
        
        # Create a mock pruning module
        module = PruningModule(model_name="distilgpt2")
        
        # Check that each strategy can be retrieved
        for strategy_name in ["random", "gradient_sensitivity", "entropy_gap", "balanced"]:
            strategy = get_strategy(strategy_name, module)
            self.assertIsNotNone(strategy)


if __name__ == "__main__":
    unittest.main()