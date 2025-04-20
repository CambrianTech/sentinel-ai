#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the simple neural plasticity experiment functionality.
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path

# Add project root to path for imports
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.neural_plasticity import simple_experiment
import pytest


class TestSimpleExperiment(unittest.TestCase):
    """Test the simple neural plasticity experiment."""
    
    def test_imports(self):
        """Test that all necessary modules can be imported."""
        self.assertTrue(hasattr(simple_experiment, "run_experiment"))
    
    @pytest.mark.skip(reason="Requires model downloads and GPU resources")
    def test_quick_experiment(self):
        """Test that the quick experiment runs successfully."""
        # Create a temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run a minimal experiment
            result = simple_experiment.run_experiment(
                model_name="distilgpt2",
                output_dir=temp_dir,
                quick_test=True,
                fine_tuning_steps=5,
                pruning_level=0.1
            )
            
            # Check that the result contains expected keys
            self.assertIsNotNone(result)
            self.assertIn("metrics", result)
            self.assertIn("pruned_heads", result)
            
            # Check that log files were created
            log_file = os.path.join(temp_dir, "experiment.log")
            self.assertTrue(os.path.exists(log_file))


if __name__ == "__main__":
    unittest.main()