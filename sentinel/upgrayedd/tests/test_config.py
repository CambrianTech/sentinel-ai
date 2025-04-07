"""
Tests for the AdaptiveOptimizerConfig class.
"""

import os
import unittest
import torch
import tempfile
import json
import sys

# Add parent directory to path if running this file directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

# Import the config class
from sentinel.upgrayedd.optimizer.adaptive_optimizer import AdaptiveOptimizerConfig


class TestAdaptiveOptimizerConfig(unittest.TestCase):
    """Test case for the AdaptiveOptimizerConfig class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = AdaptiveOptimizerConfig()
        
        # Check default values
        self.assertEqual(config.model_name, "distilgpt2")
        self.assertEqual(config.pruning_ratio, 0.3)
        self.assertEqual(config.strategy, "entropy")
        self.assertEqual(config.batch_size, 4)
        
        # Default device should be set to cuda if available, cpu otherwise
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(config.device, expected_device)

    def test_custom_values(self):
        """Test that custom values are set correctly."""
        # Create config with custom values
        config = AdaptiveOptimizerConfig(
            model_name="gpt2",
            pruning_ratio=0.5,
            strategy="magnitude",
            batch_size=8,
            device="cpu"
        )
        
        # Check custom values
        self.assertEqual(config.model_name, "gpt2")
        self.assertEqual(config.pruning_ratio, 0.5)
        self.assertEqual(config.strategy, "magnitude")
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.device, "cpu")

    def test_to_dict(self):
        """Test the to_dict method."""
        config = AdaptiveOptimizerConfig(
            model_name="gpt2",
            device="cuda"
        )
        
        # Convert to dict
        config_dict = config.to_dict()
        
        # Check type and content
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["model_name"], "gpt2")
        self.assertEqual(config_dict["device"], "cuda")

    def test_from_dict(self):
        """Test the from_dict class method."""
        config_dict = {
            "model_name": "gpt2-medium",
            "pruning_ratio": 0.4,
            "device": "cpu"
        }
        
        # Create config from dict
        config = AdaptiveOptimizerConfig.from_dict(config_dict)
        
        # Check values
        self.assertEqual(config.model_name, "gpt2-medium")
        self.assertEqual(config.pruning_ratio, 0.4)
        self.assertEqual(config.device, "cpu")
        
        # Default values should still be set for other parameters
        self.assertEqual(config.strategy, "entropy")

    def test_save_and_load(self):
        """Test saving and loading config to/from JSON file."""
        # Create config
        config = AdaptiveOptimizerConfig(
            model_name="gpt2-large",
            pruning_ratio=0.6,
            strategy="random"
        )
        
        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
            
            try:
                # Save config
                config.save(temp_path)
                
                # Check that file was created
                self.assertTrue(os.path.exists(temp_path))
                
                # Check file contents
                with open(temp_path, 'r') as f:
                    saved_dict = json.load(f)
                    self.assertEqual(saved_dict["model_name"], "gpt2-large")
                    self.assertEqual(saved_dict["pruning_ratio"], 0.6)
                    self.assertEqual(saved_dict["strategy"], "random")
                
                # Load config
                loaded_config = AdaptiveOptimizerConfig.load(temp_path)
                
                # Check values
                self.assertEqual(loaded_config.model_name, "gpt2-large")
                self.assertEqual(loaded_config.pruning_ratio, 0.6)
                self.assertEqual(loaded_config.strategy, "random")
            
            finally:
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    def test_device_handling(self):
        """Test that device is handled correctly."""
        # Test with torch.device object
        device_obj = torch.device("cpu")
        config = AdaptiveOptimizerConfig(device=device_obj)
        
        # Convert to dict - should convert device to string
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict["device"], str)
        self.assertEqual(config_dict["device"], "cpu")
        
        # Test loading with device string
        config_dict = {"device": "cuda:0" if torch.cuda.is_available() else "cpu"}
        config = AdaptiveOptimizerConfig.from_dict(config_dict)
        self.assertIsInstance(config.device, str)


if __name__ == "__main__":
    unittest.main()