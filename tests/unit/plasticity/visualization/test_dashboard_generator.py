"""
Tests for the neural plasticity dashboard generator.

This test suite verifies that dashboards are generated correctly with the expected content.
"""

import os
import sys
import unittest
import tempfile
import json
from unittest.mock import patch, MagicMock
import pytest
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

# Import the dashboard generator
from scripts.neural_plasticity.visualization.dashboard_generator import generate_dashboard

class TestDashboardGenerator(unittest.TestCase):
    """Test the dashboard generator."""
    
    def setUp(self):
        """Setup for tests."""
        # Create a temporary directory for experiment data
        self.temp_dir = tempfile.mkdtemp()
        self.dashboard_dir = os.path.join(self.temp_dir, "dashboards")
        os.makedirs(self.dashboard_dir, exist_ok=True)
        
        # Create mock experiment data
        os.makedirs(os.path.join(self.temp_dir, "entropy_0.2_test"), exist_ok=True)
        
        # Create metrics.json
        metrics = {
            "baseline": {
                "loss": 5.0,
                "perplexity": 150.0
            },
            "post_pruning": {
                "loss": 5.5,
                "perplexity": 200.0
            },
            "final": {
                "loss": 4.0,
                "perplexity": 50.0
            }
        }
        
        with open(os.path.join(self.temp_dir, "entropy_0.2_test", "metrics.json"), "w") as f:
            json.dump(metrics, f)
        
        # Create pruned_heads.json
        pruned_heads = [
            [0, 0, 0.5],
            [0, 1, 0.6],
            [1, 0, 0.7]
        ]
        
        with open(os.path.join(self.temp_dir, "entropy_0.2_test", "pruned_heads.json"), "w") as f:
            json.dump(pruned_heads, f)
        
    def tearDown(self):
        """Cleanup after tests."""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_dashboard(self):
        """Test generating a dashboard with mock data."""
        # Define output path
        output_path = os.path.join(self.dashboard_dir, "dashboard.html")
        
        # Generate dashboard
        generate_dashboard(
            experiment_dir=self.temp_dir,
            output_path=output_path,
            model_name="distilgpt2",
            pruning_strategy="entropy",
            pruning_level=0.2
        )
        
        # Verify dashboard file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Check contents of the dashboard
        with open(output_path, "r") as f:
            content = f.read()
            
            # Check for expected content
            self.assertIn("<title>Neural Plasticity Dashboard</title>", content)
            self.assertIn("distilgpt2", content)
            self.assertIn("entropy", content)
            self.assertIn("0.2", content)
    
    @patch('scripts.neural_plasticity.visualization.dashboard_generator.os.path.exists')
    def test_dashboard_with_missing_data(self, mock_exists):
        """Test dashboard generation with missing data."""
        # Mock os.path.exists to return False for metrics.json
        mock_exists.side_effect = lambda path: "metrics.json" not in path
        
        # Define output path
        output_path = os.path.join(self.dashboard_dir, "dashboard_missing.html")
        
        # Generate dashboard with missing data
        generate_dashboard(
            experiment_dir=self.temp_dir,
            output_path=output_path,
            model_name="distilgpt2",
            pruning_strategy="entropy",
            pruning_level=0.2
        )
        
        # Verify dashboard file was created even with missing data
        self.assertTrue(os.path.exists(output_path))
        
        # Check contents of the dashboard
        with open(output_path, "r") as f:
            content = f.read()
            
            # Check that dashboard was created with default content
            self.assertIn("<title>Neural Plasticity Dashboard</title>", content)
            self.assertIn("distilgpt2", content)
            
            # Check that it notes missing data
            self.assertIn("No metrics found", content)
            
    def test_dashboard_with_real_data_structure(self):
        """Test that dashboard generation handles the expected experiment directory structure."""
        # Create a more realistic experiment directory structure
        exp_subdir = os.path.join(self.temp_dir, "entropy_0.2_test", "cycle_1")
        os.makedirs(exp_subdir, exist_ok=True)
        
        # Create cycle metrics
        cycle_metrics = {
            "loss": 4.5,
            "perplexity": 75.0
        }
        
        with open(os.path.join(exp_subdir, "metrics_cycle1.json"), "w") as f:
            json.dump(cycle_metrics, f)
        
        # Define output path
        output_path = os.path.join(self.dashboard_dir, "dashboard_full.html")
        
        # Generate dashboard
        generate_dashboard(
            experiment_dir=self.temp_dir,
            output_path=output_path,
            model_name="distilgpt2",
            pruning_strategy="entropy",
            pruning_level=0.2
        )
        
        # Verify dashboard file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify content
        with open(output_path, "r") as f:
            content = f.read()
            self.assertIn("<title>Neural Plasticity Dashboard</title>", content)
            self.assertIn("distilgpt2", content)

if __name__ == '__main__':
    unittest.main()