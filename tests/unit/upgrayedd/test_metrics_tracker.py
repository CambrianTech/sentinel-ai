"""
Unit tests for the ProgressTracker class in sentinel.upgrayedd.metrics.tracker.

These tests verify that the metrics tracking and visualization functionality
works correctly.
"""

import os
import sys
import unittest
import torch
from unittest.mock import MagicMock, patch
import tempfile
import json

# Add project root to path if running this file directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

# Import the tracker class
from sentinel.upgrayedd.metrics.tracker import ProgressTracker


class TestProgressTracker(unittest.TestCase):
    """Test case for the ProgressTracker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for output
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a tracker
        self.tracker = ProgressTracker(
            output_dir=self.temp_dir,
            disable_plotting=True
        )
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_add_metrics(self):
        """Test that metrics are correctly added to the tracker."""
        # Add metrics
        self.tracker.add_metrics(
            step=0,
            loss=2.5,
            perplexity=12.18
        )
        
        self.tracker.add_metrics(
            step=10,
            loss=2.0,
            perplexity=7.38,
            pruned_heads=[(0, 1), (1, 2)]
        )
        
        # Check tracker state
        self.assertEqual(len(self.tracker.steps), 2)
        self.assertEqual(self.tracker.steps[0], 0)
        self.assertEqual(self.tracker.steps[1], 10)
        
        self.assertEqual(len(self.tracker.losses), 2)
        self.assertEqual(self.tracker.losses[0], 2.5)
        self.assertEqual(self.tracker.losses[1], 2.0)
        
        self.assertEqual(len(self.tracker.perplexities), 2)
        self.assertEqual(self.tracker.perplexities[0], 12.18)
        self.assertEqual(self.tracker.perplexities[1], 7.38)
        
        self.assertEqual(len(self.tracker.pruned_heads), 2)
        self.assertEqual(self.tracker.pruned_heads[0], [])
        self.assertEqual(self.tracker.pruned_heads[1], [(0, 1), (1, 2)])
    
    def test_add_generated_text(self):
        """Test that generated text is correctly added to the tracker."""
        # Add text
        self.tracker.add_generated_text(
            "Generated text sample 1",
            step=0
        )
        
        self.tracker.add_generated_text(
            "Generated text sample 2",
            step=10
        )
        
        # Check tracker state
        self.assertEqual(len(self.tracker.generated_texts), 2)
        self.assertEqual(self.tracker.generated_texts[0], "Generated text sample 1")
        self.assertEqual(self.tracker.generated_texts[1], "Generated text sample 2")
        
        self.assertEqual(len(self.tracker.text_steps), 2)
        self.assertEqual(self.tracker.text_steps[0], 0)
        self.assertEqual(self.tracker.text_steps[1], 10)
    
    def test_save_and_load(self):
        """Test that tracker state can be saved and loaded."""
        # Add some data
        self.tracker.add_metrics(step=0, loss=2.5, perplexity=12.18)
        self.tracker.add_metrics(step=10, loss=2.0, perplexity=7.38)
        self.tracker.add_generated_text("Sample text", step=5)
        
        # Create a file for saving
        file_path = os.path.join(self.temp_dir, "tracker_state.json")
        
        # Save tracker state
        self.tracker.save(file_path)
        
        # Check that file was created
        self.assertTrue(os.path.exists(file_path))
        
        # Create a new tracker
        new_tracker = ProgressTracker(
            output_dir=self.temp_dir,
            disable_plotting=True
        )
        
        # Load state
        new_tracker.load(file_path)
        
        # Check that state was loaded correctly
        self.assertEqual(len(new_tracker.steps), 2)
        self.assertEqual(new_tracker.steps[0], 0)
        self.assertEqual(new_tracker.steps[1], 10)
        
        self.assertEqual(len(new_tracker.losses), 2)
        self.assertEqual(new_tracker.losses[0], 2.5)
        self.assertEqual(new_tracker.losses[1], 2.0)
        
        self.assertEqual(len(new_tracker.perplexities), 2)
        self.assertEqual(new_tracker.perplexities[0], 12.18)
        self.assertEqual(new_tracker.perplexities[1], 7.38)
        
        self.assertEqual(len(new_tracker.generated_texts), 1)
        self.assertEqual(new_tracker.generated_texts[0], "Sample text")
    
    @patch("sentinel.upgrayedd.metrics.tracker.plt.figure")
    @patch("sentinel.upgrayedd.metrics.tracker.plt.savefig")
    def test_create_plots(self, mock_savefig, mock_figure):
        """Test that plots are created correctly."""
        # Add some data
        self.tracker.add_metrics(step=0, loss=2.5, perplexity=12.18)
        self.tracker.add_metrics(step=10, loss=2.0, perplexity=7.38)
        
        # Enable plotting for this test
        self.tracker.disable_plotting = False
        
        # Create plots
        self.tracker.create_plots()
        
        # Check that figure was created at least twice (loss and perplexity plots)
        self.assertGreaterEqual(mock_figure.call_count, 2)
        
        # Check that savefig was called at least twice
        self.assertGreaterEqual(mock_savefig.call_count, 2)
        
        # Check that plot files were created (if not mocked)
        if not mock_savefig.called:
            self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "loss_plot.png")))
            self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "perplexity_plot.png")))


if __name__ == "__main__":
    unittest.main()