"""
Tests for neural plasticity module.

This file contains unit tests for the neural plasticity utilities,
ensuring that the core algorithms, visualization, and training 
functions work properly.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Import module functions for testing
from utils.neural_plasticity.core import (
    calculate_head_entropy,
    calculate_head_gradients,
    generate_pruning_mask,
    apply_pruning_mask,
    evaluate_model
)

from utils.neural_plasticity.visualization import (
    visualize_head_entropy,
    visualize_head_gradients,
    visualize_pruning_decisions,
    visualize_training_metrics,
    visualize_attention_patterns
)


class TestNeuralPlasticityCore:
    """Tests for the core neural plasticity functions."""
    
    def test_calculate_head_entropy(self):
        """Test entropy calculation for attention heads."""
        # Create sample attention tensor: [batch, heads, seq_len, seq_len]
        batch_size, num_heads, seq_len = 2, 4, 10
        attention = torch.zeros(batch_size, num_heads, seq_len, seq_len)
        
        # Set different patterns for different heads
        # Head 0: Uniform (high entropy)
        attention[:, 0] = torch.ones(batch_size, seq_len, seq_len) / seq_len
        
        # Head 1: Diagonal (medium entropy)
        for b in range(batch_size):
            for i in range(seq_len):
                attention[b, 1, i, i] = 0.8
                for j in range(seq_len):
                    if i != j:
                        attention[b, 1, i, j] = 0.2 / (seq_len - 1)
        
        # Head 2: Focused on first token (low entropy)
        attention[:, 2, :, 0] = 0.9
        attention[:, 2, :, 1:] = 0.1 / (seq_len - 1)
        
        # Head 3: Random
        attention[:, 3] = torch.rand(batch_size, seq_len, seq_len)
        # Normalize to make it a proper probability distribution
        attention[:, 3] = attention[:, 3] / attention[:, 3].sum(dim=-1, keepdim=True)
        
        # Calculate entropy
        entropy = calculate_head_entropy(attention)
        
        # Check shape
        assert entropy.shape == (num_heads,), f"Expected shape {(num_heads,)}, got {entropy.shape}"
        
        # Check entropy values
        assert entropy[0] > entropy[1] > entropy[2], "Expected decreasing entropy: uniform > diagonal > focused"
        
        # Ensure values are in range [0, 1]
        assert torch.all(entropy >= 0) and torch.all(entropy <= 1), "Entropy values should be in range [0, 1]"
    
    def test_generate_pruning_mask(self):
        """Test pruning mask generation with different strategies."""
        # Create sample gradient norms: [layers, heads]
        num_layers, num_heads = 3, 4
        grad_norms = torch.tensor([
            [0.1, 0.5, 0.2, 0.8],  # Layer 0
            [0.3, 0.1, 0.7, 0.2],  # Layer 1
            [0.9, 0.4, 0.1, 0.6]   # Layer 2
        ])
        
        # Sample entropy values: [layers, heads]
        entropy = torch.tensor([
            [0.8, 0.3, 0.7, 0.2],  # Layer 0
            [0.5, 0.9, 0.1, 0.6],  # Layer 1
            [0.2, 0.4, 0.8, 0.3]   # Layer 2
        ])
        
        # Test gradient-based pruning
        prune_percent = 0.25  # Prune 3 out of 12 heads
        mask_gradient = generate_pruning_mask(
            grad_norm_values=grad_norms,
            prune_percent=prune_percent,
            strategy="gradient"
        )
        
        # Check shape
        assert mask_gradient.shape == (num_layers, num_heads), f"Expected shape {(num_layers, num_heads)}, got {mask_gradient.shape}"
        
        # Check number of pruned heads
        assert mask_gradient.sum().item() == 3, f"Expected 3 pruned heads, got {mask_gradient.sum().item()}"
        
        # Test entropy-based pruning
        mask_entropy = generate_pruning_mask(
            grad_norm_values=grad_norms,  # Not used but required
            prune_percent=prune_percent,
            strategy="entropy",
            entropy_values=entropy
        )
        
        # Check number of pruned heads
        assert mask_entropy.sum().item() == 3, f"Expected 3 pruned heads, got {mask_entropy.sum().item()}"
        
        # Test combined pruning
        mask_combined = generate_pruning_mask(
            grad_norm_values=grad_norms,
            prune_percent=prune_percent,
            strategy="combined",
            entropy_values=entropy
        )
        
        # Check number of pruned heads
        assert mask_combined.sum().item() == 3, f"Expected 3 pruned heads, got {mask_combined.sum().item()}"
        
        # Test random pruning
        torch.manual_seed(42)  # For reproducibility
        mask_random = generate_pruning_mask(
            grad_norm_values=grad_norms,
            prune_percent=prune_percent,
            strategy="random"
        )
        
        # Check number of pruned heads
        assert mask_random.sum().item() == 3, f"Expected 3 pruned heads, got {mask_random.sum().item()}"


class TestNeuralPlasticityVisualization:
    """Tests for the neural plasticity visualization functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for visualization tests."""
        num_layers, num_heads = 3, 4
        grad_norms = torch.tensor([
            [0.1, 0.5, 0.2, 0.8],  # Layer 0
            [0.3, 0.1, 0.7, 0.2],  # Layer 1
            [0.9, 0.4, 0.1, 0.6]   # Layer 2
        ])
        
        entropy = torch.tensor([
            [0.8, 0.3, 0.7, 0.2],  # Layer 0
            [0.5, 0.9, 0.1, 0.6],  # Layer 1
            [0.2, 0.4, 0.8, 0.3]   # Layer 2
        ])
        
        pruning_mask = torch.zeros_like(grad_norms, dtype=torch.bool)
        pruning_mask[0, 0] = True  # Layer 0, Head 0
        pruning_mask[1, 1] = True  # Layer 1, Head 1
        pruning_mask[2, 2] = True  # Layer 2, Head 2
        
        pruned_heads = [(0, 0), (1, 1), (2, 2)]
        
        metrics_history = {
            "train_loss": [2.0, 1.8, 1.6, 1.5, 1.4, 1.3, 1.2],
            "eval_loss": [2.1, 1.9, 1.7, 1.6, 1.5, 1.4, 1.3],
            "perplexity": [8.2, 7.5, 6.2, 5.5, 4.8, 4.2, 3.8],
            "sparsity": [0.0, 0.1, 0.2, 0.25, 0.25, 0.25, 0.25],
            "step": [0, 50, 100, 150, 200, 250, 300]
        }
        
        # Sample attention tensor: [batch, heads, seq_len, seq_len]
        batch_size, seq_len = 2, 10
        attention = torch.zeros(batch_size, num_heads, seq_len, seq_len)
        
        # Head 0: Uniform
        attention[:, 0] = torch.ones(batch_size, seq_len, seq_len) / seq_len
        
        # Head 1: Diagonal
        for b in range(batch_size):
            for i in range(seq_len):
                attention[b, 1, i, i] = 0.8
                for j in range(seq_len):
                    if i != j:
                        attention[b, 1, i, j] = 0.2 / (seq_len - 1)
        
        # Head 2: Focused
        attention[:, 2, :, 0] = 0.9
        attention[:, 2, :, 1:] = 0.1 / (seq_len - 1)
        
        # Head 3: Random
        attention[:, 3] = torch.rand(batch_size, seq_len, seq_len)
        # Normalize
        attention[:, 3] = attention[:, 3] / attention[:, 3].sum(dim=-1, keepdim=True)
        
        return {
            "grad_norms": grad_norms,
            "entropy": entropy,
            "pruning_mask": pruning_mask,
            "pruned_heads": pruned_heads,
            "metrics_history": metrics_history,
            "attention": attention
        }
    
    def test_visualize_head_entropy(self, sample_data):
        """Test entropy visualization."""
        # Create plot
        fig = visualize_head_entropy(
            entropy_values=sample_data["entropy"],
            title="Test Entropy Visualization",
            annotate=True
        )
        
        # Check if figure was created
        assert isinstance(fig, plt.Figure), "Expected matplotlib.figure.Figure object"
        
        # Close figure to avoid memory leaks
        plt.close(fig)
    
    def test_visualize_head_gradients(self, sample_data):
        """Test gradient visualization."""
        # Create plot
        fig = visualize_head_gradients(
            grad_norm_values=sample_data["grad_norms"],
            pruned_heads=sample_data["pruned_heads"],
            title="Test Gradient Visualization"
        )
        
        # Check if figure was created
        assert isinstance(fig, plt.Figure), "Expected matplotlib.figure.Figure object"
        
        # Close figure to avoid memory leaks
        plt.close(fig)
    
    def test_visualize_pruning_decisions(self, sample_data):
        """Test pruning decisions visualization."""
        # Create plot
        fig = visualize_pruning_decisions(
            grad_norm_values=sample_data["grad_norms"],
            pruning_mask=sample_data["pruning_mask"],
            title="Test Pruning Visualization"
        )
        
        # Check if figure was created
        assert isinstance(fig, plt.Figure), "Expected matplotlib.figure.Figure object"
        
        # Close figure to avoid memory leaks
        plt.close(fig)
    
    def test_visualize_training_metrics(self, sample_data):
        """Test training metrics visualization."""
        # Create plot
        fig = visualize_training_metrics(
            metrics_history=sample_data["metrics_history"],
            title="Test Training Metrics"
        )
        
        # Check if figure was created
        assert isinstance(fig, plt.Figure), "Expected matplotlib.figure.Figure object"
        
        # Close figure to avoid memory leaks
        plt.close(fig)
    
    def test_visualize_attention_patterns(self, sample_data):
        """Test attention pattern visualization."""
        # Single head
        fig1 = visualize_attention_patterns(
            attention_maps=sample_data["attention"],
            layer_idx=0,
            head_idx=1,
            title="Test Single Head Attention"
        )
        
        # Check if figure was created
        assert isinstance(fig1, plt.Figure), "Expected matplotlib.figure.Figure object"
        
        # Multiple heads
        fig2 = visualize_attention_patterns(
            attention_maps=sample_data["attention"],
            layer_idx=0,
            head_idx=None,  # Show all heads
            title="Test Multiple Heads Attention",
            num_heads=4
        )
        
        # Check if figure was created
        assert isinstance(fig2, plt.Figure), "Expected matplotlib.figure.Figure object"
        
        # Close figures to avoid memory leaks
        plt.close(fig1)
        plt.close(fig2)