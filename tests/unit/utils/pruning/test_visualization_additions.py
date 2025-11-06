"""
Unit tests for visualization_additions.py utilities.

Tests the visualization functions used in neural plasticity demonstrations
to ensure they handle various input types and edge cases properly.
"""

import os
import sys
import unittest
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')  # Use non-interactive backend for testing

from utils.pruning.visualization_additions import (
    visualize_gradient_norms,
    visualize_attention_matrix,
    visualize_entropy_heatmap,
    visualize_normalized_entropy,
    visualize_entropy_vs_gradient,
    visualize_training_progress
)


class TestVisualizationAdditions(unittest.TestCase):
    """Test suite for the visualization additions utility functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create test directory for saved figures
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_output')
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create test tensors
        self.grad_norms = torch.rand(6, 12)  # 6 layers, 12 heads
        self.entropy_values = torch.rand(6, 12)
        self.attention = torch.rand(1, 12, 32, 32)  # batch=1, heads=12, seq_len=32
        self.max_entropy = torch.tensor(np.log(32))  # log(seq_len)
        
        # Create test metrics history
        self.metrics_history = {
            "step": list(range(0, 100, 10)),
            "train_loss": [1.0 - i*0.01 for i in range(10)],
            "eval_loss": [1.1 - i*0.01 for i in range(10)],
            "pruned_heads": [i % 3 for i in range(10)],
            "revived_heads": [i % 2 for i in range(10)],
            "sparsity": [0.1 + i*0.01 for i in range(10)],
            "epoch": [1 + i//4 for i in range(10)],
            "perplexity": [10.0 - i*0.2 for i in range(10)]
        }
    
    def test_visualize_gradient_norms(self):
        """Test gradient norm visualization."""
        pruned_heads = [(0, 1), (2, 3)]
        revived_heads = [(1, 2)]
        
        # Test with torch tensor
        save_path = os.path.join(self.test_dir, 'grad_norms_torch.png')
        fig = visualize_gradient_norms(
            self.grad_norms, 
            pruned_heads=pruned_heads,
            revived_heads=revived_heads,
            title="Test Gradient Norms",
            save_path=save_path
        )
        self.assertTrue(isinstance(fig, plt.Figure))
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
        
        # Test with numpy array
        save_path = os.path.join(self.test_dir, 'grad_norms_numpy.png')
        fig = visualize_gradient_norms(
            self.grad_norms.numpy(),
            pruned_heads=pruned_heads,
            title="Test Gradient Norms (numpy)",
            save_path=save_path
        )
        self.assertTrue(isinstance(fig, plt.Figure))
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
        
        # Test with no markers
        save_path = os.path.join(self.test_dir, 'grad_norms_no_markers.png')
        fig = visualize_gradient_norms(
            self.grad_norms,
            title="Test Gradient Norms (no markers)",
            save_path=save_path
        )
        self.assertTrue(isinstance(fig, plt.Figure))
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
    
    def test_visualize_attention_matrix(self):
        """Test attention matrix visualization."""
        save_path = os.path.join(self.test_dir, 'attention.png')
        fig = visualize_attention_matrix(
            self.attention,
            layer_idx=0,
            head_idx=0,
            title="Test Attention Pattern",
            save_path=save_path
        )
        self.assertTrue(isinstance(fig, plt.Figure))
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
        
        # Test with different head
        save_path = os.path.join(self.test_dir, 'attention_head5.png')
        fig = visualize_attention_matrix(
            self.attention,
            layer_idx=0,
            head_idx=5,
            title="Test Attention Pattern (Head 5)",
            save_path=save_path
        )
        self.assertTrue(isinstance(fig, plt.Figure))
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
    
    def test_visualize_entropy_heatmap(self):
        """Test entropy heatmap visualization."""
        save_path = os.path.join(self.test_dir, 'entropy_heatmap.png')
        fig = visualize_entropy_heatmap(
            self.entropy_values,
            title="Test Entropy Heatmap",
            save_path=save_path
        )
        self.assertTrue(isinstance(fig, plt.Figure))
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
        
        # Test with min_value
        save_path = os.path.join(self.test_dir, 'entropy_heatmap_min.png')
        fig = visualize_entropy_heatmap(
            self.entropy_values,
            min_value=0.2,
            annotate=False,
            title="Test Entropy Heatmap (min=0.2)",
            save_path=save_path
        )
        self.assertTrue(isinstance(fig, plt.Figure))
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
    
    def test_visualize_normalized_entropy(self):
        """Test normalized entropy visualization."""
        save_path = os.path.join(self.test_dir, 'normalized_entropy.png')
        fig = visualize_normalized_entropy(
            self.entropy_values,
            self.max_entropy,
            title="Test Normalized Entropy",
            save_path=save_path
        )
        self.assertTrue(isinstance(fig, plt.Figure))
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
        
        # Test with scalar max entropy
        save_path = os.path.join(self.test_dir, 'normalized_entropy_scalar.png')
        fig = visualize_normalized_entropy(
            self.entropy_values,
            self.max_entropy.item(),
            title="Test Normalized Entropy (scalar max)",
            save_path=save_path
        )
        self.assertTrue(isinstance(fig, plt.Figure))
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
    
    def test_visualize_entropy_vs_gradient(self):
        """Test entropy vs gradient visualization."""
        save_path = os.path.join(self.test_dir, 'entropy_vs_gradient.png')
        fig = visualize_entropy_vs_gradient(
            self.entropy_values,
            self.grad_norms,
            title="Test Entropy vs Gradient",
            save_path=save_path
        )
        self.assertTrue(isinstance(fig, plt.Figure))
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
        
        # Test with numpy arrays
        save_path = os.path.join(self.test_dir, 'entropy_vs_gradient_numpy.png')
        fig = visualize_entropy_vs_gradient(
            self.entropy_values.numpy(),
            self.grad_norms.numpy(),
            title="Test Entropy vs Gradient (numpy)",
            save_path=save_path
        )
        self.assertTrue(isinstance(fig, plt.Figure))
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
    
    def test_visualize_training_progress(self):
        """Test training progress visualization."""
        save_path = os.path.join(self.test_dir, 'training_progress.png')
        fig = visualize_training_progress(
            self.metrics_history,
            max_display_points=10,
            figsize=(10, 8),
            save_path=save_path
        )
        self.assertTrue(isinstance(fig, plt.Figure))
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
        
        # Test with missing perplexity
        metrics_no_perplexity = self.metrics_history.copy()
        metrics_no_perplexity.pop("perplexity")
        
        save_path = os.path.join(self.test_dir, 'training_progress_no_perplexity.png')
        fig = visualize_training_progress(
            metrics_no_perplexity,
            max_display_points=10,
            figsize=(10, 8),
            save_path=save_path
        )
        self.assertTrue(isinstance(fig, plt.Figure))
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
        
        # Test with missing revived_heads
        metrics_no_revived = self.metrics_history.copy()
        metrics_no_revived.pop("revived_heads")
        
        save_path = os.path.join(self.test_dir, 'training_progress_no_revived.png')
        fig = visualize_training_progress(
            metrics_no_revived,
            max_display_points=10,
            figsize=(10, 8),
            save_path=save_path
        )
        self.assertTrue(isinstance(fig, plt.Figure))
        self.assertTrue(os.path.exists(save_path))
        plt.close(fig)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with empty tensor
        empty_tensor = torch.zeros(0, 0)
        with self.assertRaises(Exception):  # Should raise an exception for empty tensor
            visualize_gradient_norms(empty_tensor)
        
        # Test with invalid dimensions for attention matrix
        invalid_attn = torch.rand(1, 1)  # Wrong dimensions
        with self.assertRaises(Exception):  # Should raise an exception
            visualize_attention_matrix(invalid_attn)
    
    def tearDown(self):
        """Clean up after tests."""
        # Close all figures
        plt.close('all')


if __name__ == '__main__':
    unittest.main()