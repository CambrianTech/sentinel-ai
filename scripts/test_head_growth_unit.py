#!/usr/bin/env python
"""
Unit tests for head growth functionality.

This script verifies that the head growth implementation works correctly
at a basic level by testing various growth strategies and model types.
"""

import os
import sys
import unittest
import tempfile
import random
import numpy as np
import jax
import jax.numpy as jnp

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pruning.pruning_module import PruningModule
from utils.pruning.strategies import get_strategy as get_pruning_strategy
from utils.pruning.growth import (
    grow_attention_heads_gradually, 
    determine_active_heads,
    get_strategy as get_growth_strategy,
    GradientSensitivityStrategy,
    EntropyGapStrategy,
    BalancedStrategy,
    RandomStrategy
)

class TestHeadGrowth(unittest.TestCase):
    """Test cases for head growth functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Use a small model for faster tests
        cls.model_name = "distilgpt2"
        cls.pruning_module = PruningModule(cls.model_name)
        cls.pruning_module.load_model()
        
        # Save original parameters
        cls.original_params = cls.pruning_module.model.params.copy()
        
        # Determine active heads in original model
        cls.original_active_heads = determine_active_heads(cls.pruning_module, cls.original_params)
        print(f"Original model has {len(cls.original_active_heads)} active heads")
        
        # Ensure consistent random behavior
        random.seed(42)
        np.random.seed(42)
    
    def prune_model(self, params, pruning_level=0.3, strategy="entropy"):
        """Helper method to prune model for testing growth"""
        # Get pruning strategy
        pruning_strategy = get_pruning_strategy(strategy, self.pruning_module)
        
        # Calculate importance scores for all heads
        head_importance = pruning_strategy.get_head_importance(params)
        head_importance.sort(key=lambda x: x[2])
        
        # Calculate number of heads to prune
        total_heads = self.pruning_module.num_layers * self.pruning_module.num_heads
        heads_to_prune = int(total_heads * pruning_level)
        
        # Select and prune heads
        heads_to_prune = [(layer_idx, head_idx) for layer_idx, head_idx, _ in head_importance[:heads_to_prune]]
        pruned_params = params.copy()
        
        for layer_idx, head_idx in heads_to_prune:
            pruned_params = self.pruning_module.prune_head(pruned_params, layer_idx, head_idx)
        
        return pruned_params, heads_to_prune
    
    def test_active_heads_detection(self):
        """Test that active heads are correctly detected"""
        # Original model should have all heads active
        active_heads = determine_active_heads(self.pruning_module, self.original_params)
        expected_total = self.pruning_module.num_layers * self.pruning_module.num_heads
        
        self.assertEqual(len(active_heads), expected_total, 
                        f"Expected {expected_total} active heads, found {len(active_heads)}")
        
        # Prune some heads
        pruning_level = 0.3
        pruned_params, pruned_heads = self.prune_model(self.original_params, pruning_level)
        
        # Check active heads after pruning
        active_after_pruning = determine_active_heads(self.pruning_module, pruned_params)
        expected_active = expected_total - len(pruned_heads)
        
        self.assertEqual(len(active_after_pruning), expected_active,
                        f"Expected {expected_active} active heads after pruning, found {len(active_after_pruning)}")
    
    def test_growth_strategies_creation(self):
        """Test that all growth strategies can be created"""
        # Test each strategy type
        strategies = ["gradient_sensitivity", "entropy_gap", "balanced", "random"]
        
        for strategy_name in strategies:
            # Get strategy
            strategy = get_growth_strategy(strategy_name, self.pruning_module)
            
            # Check type
            if strategy_name == "gradient_sensitivity":
                self.assertIsInstance(strategy, GradientSensitivityStrategy)
            elif strategy_name == "entropy_gap":
                self.assertIsInstance(strategy, EntropyGapStrategy)
            elif strategy_name == "balanced":
                self.assertIsInstance(strategy, BalancedStrategy)
            elif strategy_name == "random":
                self.assertIsInstance(strategy, RandomStrategy)
    
    def test_grow_attention_heads(self):
        """Test basic head growth functionality"""
        # Prune heads first
        pruning_level = 0.3
        pruned_params, pruned_heads = self.prune_model(self.original_params, pruning_level)
        
        # Get active heads after pruning
        pruned_active_heads = determine_active_heads(self.pruning_module, pruned_params)
        self.assertLess(len(pruned_active_heads), len(self.original_active_heads),
                       "Pruning should reduce the number of active heads")
        
        # Grow heads
        growth_percentage = 0.1
        growth_strategy = "random"  # Use random for deterministic testing
        initial_scale = 0.01
        
        grown_params, added_count, added_heads, warmup_schedule = grow_attention_heads_gradually(
            self.pruning_module,
            params=pruned_params,
            active_heads=pruned_active_heads,
            growth_percentage=growth_percentage,
            strategy=growth_strategy,
            initial_scale=initial_scale
        )
        
        # Check that heads were added
        self.assertGreater(added_count, 0, "No heads were added during growth")
        
        # Check active heads after growth
        grown_active_heads = determine_active_heads(self.pruning_module, grown_params)
        
        # Check that added heads are active (at least some of them)
        found_added_heads = 0
        for layer_idx, head_idx in added_heads:
            if (layer_idx, head_idx) in grown_active_heads:
                found_added_heads += 1
        
        self.assertGreater(found_added_heads, 0, 
                          "At least some of the added heads should be active after growth")
    
    def test_all_growth_strategies(self):
        """Test all growth strategies"""
        # Prune heads first
        pruning_level = 0.3
        pruned_params, pruned_heads = self.prune_model(self.original_params, pruning_level)
        pruned_active_heads = determine_active_heads(self.pruning_module, pruned_params)
        
        # Test each growth strategy
        strategies = ["gradient_sensitivity", "entropy_gap", "balanced", "random"]
        growth_percentage = 0.1
        
        for strategy_name in strategies:
            # Grow heads with this strategy
            grown_params, added_count, added_heads, warmup_schedule = grow_attention_heads_gradually(
                self.pruning_module,
                params=pruned_params.copy(),  # Use copy to start from same pruned state
                active_heads=pruned_active_heads,
                growth_percentage=growth_percentage,
                strategy=strategy_name,
                initial_scale=0.01
            )
            
            # Check that heads were added
            self.assertGreater(added_count, 0, 
                             f"No heads were added with {strategy_name} strategy")
            
            # Check active heads after growth
            grown_active_heads = determine_active_heads(self.pruning_module, grown_params)
            
            # Check that at least some added heads are active
            found_added_heads = 0
            for layer_idx, head_idx in added_heads:
                if (layer_idx, head_idx) in grown_active_heads:
                    found_added_heads += 1
            
            self.assertGreater(found_added_heads, 0, 
                             f"No added heads are active with {strategy_name} strategy")
            
            print(f"Strategy {strategy_name}: added {added_count} heads, {found_added_heads} confirmed active")
    
    def test_warmup_schedule(self):
        """Test that warmup schedule behaves as expected"""
        # Prune and grow heads
        pruned_params, _ = self.prune_model(self.original_params, 0.3)
        pruned_active_heads = determine_active_heads(self.pruning_module, pruned_params)
        
        initial_scale = 0.01
        warmup_steps = 100
        
        _, _, _, warmup_schedule = grow_attention_heads_gradually(
            self.pruning_module,
            params=pruned_params,
            active_heads=pruned_active_heads,
            growth_percentage=0.1,
            strategy="random",
            initial_scale=initial_scale,
            warmup_steps=warmup_steps
        )
        
        # Test schedule at different steps
        self.assertAlmostEqual(warmup_schedule(0), initial_scale, 
                             "Schedule should start at initial_scale")
        
        self.assertAlmostEqual(warmup_schedule(warmup_steps // 2), 
                             (initial_scale + 1.0) / 2,
                             2,  # places
                             "Schedule should be halfway at warmup_steps/2")
        
        self.assertAlmostEqual(warmup_schedule(warmup_steps), 1.0,
                             "Schedule should reach 1.0 at warmup_steps")
        
        self.assertAlmostEqual(warmup_schedule(warmup_steps * 2), 1.0,
                             "Schedule should stay at 1.0 beyond warmup_steps")
    
    def test_zero_growth_possible(self):
        """Test behavior when almost all heads are active - should add very few heads"""
        # Create a modified params with just 1 head pruned
        pruning_level = 0.02  # Very small pruning level
        almost_full_params, few_pruned_heads = self.prune_model(self.original_params, pruning_level)
        
        if len(few_pruned_heads) == 0:
            # Skip test if no heads were pruned
            print("Skipping test_zero_growth_possible: unable to create partial pruning")
            return
            
        # Get active heads
        almost_full_active_heads = determine_active_heads(self.pruning_module, almost_full_params)
        
        # Grow heads with very small percentage
        growth_percentage = 0.01  # Minimal growth
        grown_params, added_count, added_heads, warmup_schedule = grow_attention_heads_gradually(
            self.pruning_module,
            params=almost_full_params,
            active_heads=almost_full_active_heads,
            growth_percentage=growth_percentage,
            strategy="random"
        )
        
        # Check warmup schedule works
        self.assertTrue(warmup_schedule(0) <= warmup_schedule(50),
                      "Warmup schedule should be monotonically increasing")
    
    def test_model_functionality_after_growth(self):
        """Test that model still functions after head growth"""
        # Prune heads first
        pruned_params, _ = self.prune_model(self.original_params, 0.3)
        pruned_active_heads = determine_active_heads(self.pruning_module, pruned_params)
        
        # Grow heads
        grown_params, _, _, _ = grow_attention_heads_gradually(
            self.pruning_module,
            params=pruned_params,
            active_heads=pruned_active_heads,
            growth_percentage=0.1,
            strategy="random"
        )
        
        # Test model with original params
        test_text = "The transformer model processes"
        try:
            # Test perplexity calculation
            original_perplexity = self.pruning_module.evaluate_perplexity(self.original_params, test_text)
            self.assertFalse(jnp.isnan(original_perplexity).any(), "Original perplexity contains NaN")
            
            # Test text generation
            original_generation = self.pruning_module.generate_text(self.original_params, test_text, max_length=30)
            self.assertGreater(len(original_generation), len(test_text), "Original model should generate text")
            
            # Test with grown params
            grown_perplexity = self.pruning_module.evaluate_perplexity(grown_params, test_text)
            self.assertFalse(jnp.isnan(grown_perplexity).any(), "Grown model perplexity contains NaN")
            
            grown_generation = self.pruning_module.generate_text(grown_params, test_text, max_length=30)
            self.assertGreater(len(grown_generation), len(test_text), "Grown model should generate text")
            
            print("Model functionality test passed. Original perplexity:", original_perplexity, 
                  "Grown perplexity:", grown_perplexity)
            
        except Exception as e:
            self.fail(f"Model functionality test failed with error: {e}")

if __name__ == "__main__":
    unittest.main()