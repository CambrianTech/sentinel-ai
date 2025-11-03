"""
Unit tests for the visualization utilities in utils/colab/visualizations.py.
"""

import unittest
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Skip tests if IPython is not available (needed for display widgets)
try:
    from IPython.display import HTML, display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

# Skip tests if torch is not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipIf(not IPYTHON_AVAILABLE, "IPython not available")
class TestPersistentDisplay(unittest.TestCase):
    """Test the PersistentDisplay class."""
    
    def setUp(self):
        """Set up test environment."""
        from utils.colab.visualizations import PersistentDisplay
        self.persistent_display = PersistentDisplay(
            title="Test Display",
            show_timestamp=False,
            display_id="test_display"
        )
    
    def test_initialization(self):
        """Test that the display initializes correctly."""
        self.assertEqual(self.persistent_display.title, "Test Display")
        self.assertEqual(self.persistent_display.display_id, "test_display")
        self.assertEqual(self.persistent_display.update_count, 0)
    
    def test_update(self):
        """Test updating the display content."""
        # This shouldn't raise an exception even though we're not in a notebook
        try:
            self.persistent_display.update("Test content", clear=False)
            self.assertEqual(self.persistent_display.update_count, 1)
        except Exception as e:
            self.fail(f"update() raised {type(e).__name__} unexpectedly!")
    
    def test_update_with_metrics(self):
        """Test updating with metrics dictionary."""
        metrics = {
            "loss": 0.5,
            "accuracy": 0.8,
            "step": 100
        }
        
        # Test different layouts
        for layout in ["grid", "list", "table"]:
            try:
                self.persistent_display.update_with_metrics(
                    metrics, 
                    layout=layout,
                    clear=False
                )
            except Exception as e:
                self.fail(f"update_with_metrics() with {layout} layout raised {type(e).__name__} unexpectedly!")


@unittest.skipIf(not IPYTHON_AVAILABLE, "IPython not available")
class TestTrainingMonitor(unittest.TestCase):
    """Test the TrainingMonitor class."""
    
    def setUp(self):
        """Set up test environment."""
        from utils.colab.visualizations import TrainingMonitor
        self.training_monitor = TrainingMonitor(
            title="Test Training Monitor",
            metrics_to_track=["loss", "accuracy", "step"],
            show_timestamp=False
        )
    
    def test_initialization(self):
        """Test that the monitor initializes correctly."""
        self.assertEqual(self.training_monitor.title, "Test Training Monitor")
        self.assertEqual(list(self.training_monitor.history.keys()), ["loss", "accuracy", "step"])
    
    def test_update_metrics(self):
        """Test updating metrics."""
        metrics = {
            "loss": 0.5,
            "accuracy": 0.8,
            "step": 100
        }
        
        try:
            # Should update without plotting
            self.training_monitor.update_metrics(metrics, plot=False)
            
            # Check if history is updated correctly
            self.assertEqual(self.training_monitor.history["loss"], [0.5])
            self.assertEqual(self.training_monitor.history["accuracy"], [0.8])
            self.assertEqual(self.training_monitor.history["step"], [100])
            
            # Update again with new values
            self.training_monitor.update_metrics({"loss": 0.4, "accuracy": 0.85, "step": 200}, plot=False)
            
            # Check if history is appended correctly
            self.assertEqual(self.training_monitor.history["loss"], [0.5, 0.4])
            self.assertEqual(self.training_monitor.history["accuracy"], [0.8, 0.85])
            self.assertEqual(self.training_monitor.history["step"], [100, 200])
            
        except Exception as e:
            self.fail(f"update_metrics() raised {type(e).__name__} unexpectedly!")
    
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_update_with_figure(self):
        """Test updating with a figure."""
        # Create a simple matplotlib figure
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([1, 2, 3], [4, 5, 6])
        
        try:
            # Test updating with a figure directly
            self.training_monitor.update_with_figure(fig, caption="Test Figure", clear=False)
            
            # Test with a function that returns a figure
            def create_figure():
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.scatter([1, 2, 3], [4, 5, 6])
                return fig
            
            self.training_monitor.update_with_figure(create_figure, caption="Function Figure", clear=False)
            
        except Exception as e:
            self.fail(f"update_with_figure() raised {type(e).__name__} unexpectedly!")
        finally:
            plt.close("all")


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
class TestVisualizationFunctions(unittest.TestCase):
    """Test the visualization functions."""
    
    def setUp(self):
        """Set up test data."""
        import torch
        import numpy as np
        
        # Create test data
        self.grad_tensor = torch.rand(6, 12)  # 6 layers, 12 heads per layer
        self.entropy_tensor = torch.rand(6, 12)
        self.attention_tensor = torch.rand(1, 12, 32, 32)  # batch=1, heads=12, seq_len=32x32
        
        # Create output directory for test figures
        self.test_dir = os.path.join(os.path.dirname(__file__), "test_output")
        os.makedirs(self.test_dir, exist_ok=True)
    
    def test_visualize_gradient_norms(self):
        """Test gradient norm visualization."""
        from utils.colab.visualizations import visualize_gradient_norms
        
        # Test with PyTorch tensor
        save_path = os.path.join(self.test_dir, "gradient_norms.png")
        fig = visualize_gradient_norms(
            self.grad_tensor,
            pruned_heads=[(0, 1), (2, 3)],
            revived_heads=[(1, 2)],
            title="Test Gradient Norms"
        )
        fig.savefig(save_path)
        plt.close(fig)
        
        # Verify file was created
        self.assertTrue(os.path.exists(save_path))
        
        # Test with numpy array
        save_path = os.path.join(self.test_dir, "gradient_norms_numpy.png")
        fig = visualize_gradient_norms(
            self.grad_tensor.numpy(),
            title="Test Gradient Norms (NumPy)"
        )
        fig.savefig(save_path)
        plt.close(fig)
        
        # Verify file was created
        self.assertTrue(os.path.exists(save_path))
    
    def test_visualize_head_entropy(self):
        """Test entropy visualization."""
        from utils.colab.visualizations import visualize_head_entropy
        
        # Test with PyTorch tensor
        save_path = os.path.join(self.test_dir, "entropy.png")
        fig = visualize_head_entropy(
            self.entropy_tensor,
            title="Test Entropy",
            annotate=True
        )
        fig.savefig(save_path)
        plt.close(fig)
        
        # Verify file was created
        self.assertTrue(os.path.exists(save_path))
        
        # Test with numpy array
        save_path = os.path.join(self.test_dir, "entropy_numpy.png")
        fig = visualize_head_entropy(
            self.entropy_tensor.numpy(),
            title="Test Entropy (NumPy)",
            annotate=False
        )
        fig.savefig(save_path)
        plt.close(fig)
        
        # Verify file was created
        self.assertTrue(os.path.exists(save_path))
    
    def test_visualize_attention_heatmap(self):
        """Test attention heatmap visualization."""
        from utils.colab.visualizations import visualize_attention_heatmap
        
        # Test with PyTorch tensor
        save_path = os.path.join(self.test_dir, "attention.png")
        fig = visualize_attention_heatmap(
            self.attention_tensor,
            layer_idx=0,
            head_idx=0,
            title="Test Attention"
        )
        fig.savefig(save_path)
        plt.close(fig)
        
        # Verify file was created
        self.assertTrue(os.path.exists(save_path))
        
        # Test with numpy array
        save_path = os.path.join(self.test_dir, "attention_numpy.png")
        fig = visualize_attention_heatmap(
            self.attention_tensor.numpy(),
            layer_idx=0,
            head_idx=1,
            title="Test Attention (NumPy)",
            show_colorbar=False
        )
        fig.savefig(save_path)
        plt.close(fig)
        
        # Verify file was created
        self.assertTrue(os.path.exists(save_path))
    
    def tearDown(self):
        """Clean up after tests."""
        plt.close("all")


if __name__ == "__main__":
    unittest.main()