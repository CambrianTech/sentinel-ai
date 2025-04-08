"""
Unit tests for the PlasticityController.

To run these tests:
    pytest -xvs sentinel/pruning/tests/test_plasticity_controller.py
"""

import torch
import pytest
import numpy as np
from transformers import AutoModelForCausalLM

from sentinel.pruning.plasticity_controller import (
    PlasticityController, PlasticityDecision, create_plasticity_controller
)
from sentinel.pruning.dual_mode_pruning import PruningMode


class TestPlasticityController:
    @pytest.fixture(scope="class")
    def model(self):
        """Load a small GPT-2 model for testing."""
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        return model
    
    @pytest.fixture
    def controller(self, model):
        """Create a plasticity controller with default settings."""
        return create_plasticity_controller(
            model=model,
            mode=PruningMode.ADAPTIVE,
            high_entropy_threshold=0.8,
            low_entropy_threshold=0.4,
            grad_threshold=1e-3,
            min_zero_epochs=1
        )
    
    def test_controller_creation(self, model):
        """Test that the controller is created correctly."""
        controller = create_plasticity_controller(model)
        
        # Check controller attributes
        assert controller.model == model
        assert controller.total_layers == len(model.transformer.h)
        assert controller.heads_per_layer == model.transformer.h[0].attn.n_head
        assert controller.mode == PruningMode.ADAPTIVE
        
        # Check default threshold values
        assert controller.high_entropy_threshold == 0.8
        assert controller.low_entropy_threshold == 0.4
        assert controller.grad_threshold == 1e-3
        
    def test_decide_head_fate_keep(self):
        """Test that the head fate decision correctly keeps heads."""
        # Create a mock controller
        controller = PlasticityController(
            model=None, 
            total_layers=1, 
            heads_per_layer=1,
            high_entropy_threshold=0.8,
            low_entropy_threshold=0.4,
            grad_threshold=1e-3
        )
        
        # Test cases for keeping active heads
        # Case 1: Low entropy, high gradient - clearly useful head
        head_stats = {'is_zeroed': False, 'zeroed_epochs': 0}
        decision = controller._decide_head_fate(head_stats, entropy=0.3, grad_norm=1e-2)
        assert decision == PlasticityDecision.KEEP
        
        # Case 2: Low entropy, low gradient - good focus but not learning much
        head_stats = {'is_zeroed': False, 'zeroed_epochs': 0}
        decision = controller._decide_head_fate(head_stats, entropy=0.3, grad_norm=1e-4)
        assert decision == PlasticityDecision.KEEP
        
        # Case 3: High entropy but still high gradient - noisy but learning something
        head_stats = {'is_zeroed': False, 'zeroed_epochs': 0}
        decision = controller._decide_head_fate(head_stats, entropy=0.9, grad_norm=1e-2)
        assert decision == PlasticityDecision.KEEP
        
    def test_decide_head_fate_prune(self):
        """Test that the head fate decision correctly prunes heads."""
        # Create a mock controller
        controller = PlasticityController(
            model=None, 
            total_layers=1, 
            heads_per_layer=1,
            high_entropy_threshold=0.6,  # Lower threshold for more pruning
            low_entropy_threshold=0.3,
            grad_threshold=1e-3
        )
        
        # Test case for pruning: High entropy, low gradient - not useful
        head_stats = {'is_zeroed': False, 'zeroed_epochs': 0}
        decision = controller._decide_head_fate(head_stats, entropy=0.7, grad_norm=5e-4)
        assert decision == PlasticityDecision.PRUNE
        
        # Edge case: Exactly at the threshold
        head_stats = {'is_zeroed': False, 'zeroed_epochs': 0}
        decision = controller._decide_head_fate(head_stats, entropy=0.6, grad_norm=1e-3)
        assert decision == PlasticityDecision.KEEP  # Should be kept (not strict inequality)
        
        # Test with even more aggressive thresholds
        controller.high_entropy_threshold = 0.4
        controller.grad_threshold = 1e-2
        
        # This should now be pruned with the new thresholds
        head_stats = {'is_zeroed': False, 'zeroed_epochs': 0}
        decision = controller._decide_head_fate(head_stats, entropy=0.5, grad_norm=5e-3)
        assert decision == PlasticityDecision.PRUNE
        
    def test_decide_head_fate_revive(self):
        """Test that the head fate decision correctly revives heads."""
        # Create a mock controller
        controller = PlasticityController(
            model=None, 
            total_layers=1, 
            heads_per_layer=1,
            high_entropy_threshold=0.8,
            low_entropy_threshold=0.4,
            grad_threshold=1e-3,
            min_zero_epochs=1,
            mode=PruningMode.ADAPTIVE
        )
        
        # Test case for revival: Low entropy, high gradient - potentially useful
        head_stats = {'is_zeroed': True, 'zeroed_epochs': 1}
        decision = controller._decide_head_fate(head_stats, entropy=0.3, grad_norm=3e-3)
        assert decision == PlasticityDecision.REVIVE
        
        # Test case: Not enough epochs zeroed
        head_stats = {'is_zeroed': True, 'zeroed_epochs': 0}
        decision = controller._decide_head_fate(head_stats, entropy=0.3, grad_norm=3e-3)
        assert decision == PlasticityDecision.KEEP
        
        # Test case: Not enough gradient
        head_stats = {'is_zeroed': True, 'zeroed_epochs': 1}
        decision = controller._decide_head_fate(head_stats, entropy=0.3, grad_norm=1.5e-3)
        assert decision == PlasticityDecision.KEEP  # Not enough gradient (needs 2x threshold)
        
        # Test case: Entropy too high
        head_stats = {'is_zeroed': True, 'zeroed_epochs': 1}
        decision = controller._decide_head_fate(head_stats, entropy=0.5, grad_norm=3e-3)
        assert decision == PlasticityDecision.KEEP
        
        # Test in compressed mode (should never revive)
        controller.mode = PruningMode.COMPRESSED
        head_stats = {'is_zeroed': True, 'zeroed_epochs': 1}
        decision = controller._decide_head_fate(head_stats, entropy=0.3, grad_norm=3e-3)
        assert decision == PlasticityDecision.KEEP
        
    def test_thresholds_with_realistic_values(self):
        """Test thresholds with realistic entropy and gradient values."""
        controller = PlasticityController(
            model=None, 
            total_layers=1, 
            heads_per_layer=1,
            high_entropy_threshold=0.4,  # More aggressive threshold
            low_entropy_threshold=0.2,
            grad_threshold=1e-3,
            min_zero_epochs=1
        )
        
        # Simulated realistic data (based on empirical observations)
        realistic_values = [
            # entropy, grad_norm, is_zeroed, expected_decision
            (0.35, 1.2e-3, False, PlasticityDecision.KEEP),     # Below high entropy threshold
            (0.45, 1.2e-3, False, PlasticityDecision.KEEP),     # Above high entropy but high gradient
            (0.45, 0.8e-3, False, PlasticityDecision.PRUNE),    # Should be pruned
            (0.25, 0.8e-3, False, PlasticityDecision.KEEP),     # Low entropy, low gradient
            (0.15, 2.5e-3, True, PlasticityDecision.REVIVE),    # Should be revived
            (0.15, 1.5e-3, True, PlasticityDecision.KEEP),      # Not enough gradient to revive
            (0.25, 2.5e-3, True, PlasticityDecision.KEEP),      # Entropy too high to revive
        ]
        
        for entropy, grad_norm, is_zeroed, expected in realistic_values:
            head_stats = {'is_zeroed': is_zeroed, 'zeroed_epochs': 1 if is_zeroed else 0}
            decision = controller._decide_head_fate(head_stats, entropy, grad_norm)
            assert decision == expected, f"Failed for entropy={entropy}, grad_norm={grad_norm}, is_zeroed={is_zeroed}"
            
    def test_compute_entropy(self):
        """Test entropy computation."""
        controller = PlasticityController(
            model=None, 
            total_layers=1, 
            heads_per_layer=1
        )
        
        # Create uniform attention (maximum entropy)
        seq_len = 10
        uniform_attn = torch.ones(1, seq_len, seq_len) / seq_len
        entropy = controller._compute_entropy([uniform_attn])
        
        # Should be close to 1.0 (max normalized entropy)
        assert abs(entropy - 1.0) < 1e-5
        
        # Create peaked attention (low entropy)
        peaked_attn = torch.zeros(1, seq_len, seq_len)
        peaked_attn[0, 0, 0] = 0.9  # 90% attention to one token
        for i in range(1, seq_len):
            peaked_attn[0, 0, i] = 0.1 / (seq_len - 1)  # Distribute remaining 10%
        
        entropy = controller._compute_entropy([peaked_attn])
        
        # Should be much lower than 1.0
        assert entropy < 0.5


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])