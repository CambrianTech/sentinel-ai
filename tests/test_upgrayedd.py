#!/usr/bin/env python
"""
Unit tests for the upgrayedd.py script.

These tests verify that the model upgrader properly handles:
1. Loading different model types
2. Configuration validation
3. Error handling
4. CLI argument parsing
5. Mock optimization cycle
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock, mock_open

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to test
from scripts.upgrayedd import ModelUpgrader, parse_args, load_json_config


class TestUpgrayedd(unittest.TestCase):
    """Test cases for the upgrayedd.py script"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_model_name = "test-model"
        
        # Mock configuration
        self.test_config = {
            "dataset": "test-dataset",
            "cycles": 3,
            "pruning_level": 0.2,
            "growth_ratio": 0.4,
            "controller_config": {
                "controller_type": "ann"
            }
        }

    def tearDown(self):
        """Tear down test fixtures"""
        # Clean up temp directory if needed
        if os.path.exists(self.temp_dir):
            # In a real test, would use shutil.rmtree(self.temp_dir)
            pass

    @patch('scripts.upgrayedd.logger')
    def test_init(self, mock_logger):
        """Test ModelUpgrader initialization"""
        upgrader = ModelUpgrader(
            model_name=self.mock_model_name,
            output_dir=self.temp_dir,
            config=self.test_config
        )
        
        # Verify the instance was created correctly
        self.assertEqual(upgrader.model_name, self.mock_model_name)
        self.assertEqual(upgrader.config, self.test_config)
        self.assertTrue(mock_logger.info.called)

    @patch('scripts.upgrayedd.os.makedirs')
    @patch('scripts.upgrayedd.logger')
    def test_directory_creation(self, mock_logger, mock_makedirs):
        """Test output directory creation"""
        upgrader = ModelUpgrader(
            model_name=self.mock_model_name,
            output_dir=self.temp_dir,
            config=self.test_config
        )
        
        # Verify directories were created
        self.assertEqual(mock_makedirs.call_count, 5)  # main dir + 4 subdirs
        
    @patch('builtins.open', new_callable=mock_open)
    @patch('scripts.upgrayedd.json.dump')
    @patch('scripts.upgrayedd.logger')
    def test_save_config(self, mock_logger, mock_json_dump, mock_file):
        """Test configuration saving"""
        upgrader = ModelUpgrader(
            model_name=self.mock_model_name,
            output_dir=self.temp_dir,
            config=self.test_config
        )
        
        # Call the method being tested
        upgrader.save_config()
        
        # Check that file was opened and config was written
        mock_file.assert_called_once()
        mock_json_dump.assert_called_once()

    @patch('scripts.upgrayedd.importlib.util.find_spec')
    @patch('scripts.upgrayedd.logger')
    def test_validate_model_structure(self, mock_logger, mock_find_spec):
        """Test model structure validation"""
        upgrader = ModelUpgrader(
            model_name=self.mock_model_name,
            output_dir=self.temp_dir,
            config=self.test_config
        )
        
        # Mock a model with transformer attributes
        mock_model = MagicMock()
        mock_model.config.model_type = "gpt2"
        upgrader.model = mock_model
        
        # Test validation
        result = upgrader._validate_model_structure()
        self.assertTrue(result)
        
        # Test with unsupported model
        mock_model.config.model_type = "unsupported_type"
        mock_model.transformer = None
        mock_model.model = None
        mock_model.encoder = None
        mock_model.decoder = None
        result = upgrader._validate_model_structure()
        self.assertFalse(result)

    @patch('scripts.upgrayedd.logger')
    def test_dry_run(self, mock_logger):
        """Test dry run mode"""
        # Set up configuration with dry run enabled
        dry_run_config = self.test_config.copy()
        dry_run_config["dry_run"] = True
        
        upgrader = ModelUpgrader(
            model_name=self.mock_model_name,
            output_dir=self.temp_dir,
            config=dry_run_config
        )
        
        # Mock methods that should be skipped in dry run
        upgrader.load_model_and_tokenizer = MagicMock(return_value=True)
        upgrader.inject_adaptive_modules = MagicMock(return_value=True)
        upgrader.load_dataset = MagicMock(return_value=True)
        upgrader.setup_optimization_cycle = MagicMock(return_value=True)
        upgrader.run_optimization = MagicMock(return_value={})
        upgrader.save_upgraded_model = MagicMock(return_value=True)
        
        # Run the upgrade process
        result = upgrader.upgrade()
        
        # Verify the process completed successfully
        self.assertTrue(result)
        
        # Verify optimization was not actually run in dry run mode
        upgrader.load_model_and_tokenizer.assert_called_once()
        upgrader.inject_adaptive_modules.assert_called_once()
        upgrader.setup_optimization_cycle.assert_called_once()
        upgrader.run_optimization.assert_not_called()
        upgrader.save_upgraded_model.assert_not_called()

    @patch('scripts.upgrayedd.open', new_callable=mock_open, read_data='{"test": 123}')
    def test_load_json_config(self, mock_file):
        """Test loading configuration from JSON file"""
        # Call the function being tested
        result = load_json_config("dummy_path.json")
        
        # Check that file was opened and JSON was parsed
        mock_file.assert_called_once_with("dummy_path.json", 'r')
        self.assertEqual(result, {"test": 123})

    @patch('scripts.upgrayedd.argparse.ArgumentParser.parse_args')
    def test_parse_args(self, mock_args):
        """Test command line argument parsing"""
        # Set up mock arguments
        mock_args.return_value = MagicMock(
            model="gpt2",
            dataset="tiny_shakespeare",
            cycles=5,
            pruning_level=0.3,
            growth_ratio=0.5,
            controller_type="ann",
            dry_run=False
        )
        
        # Call the function being tested
        args = parse_args()
        
        # Check that arguments were parsed correctly
        self.assertEqual(args.model, "gpt2")
        self.assertEqual(args.dataset, "tiny_shakespeare")
        self.assertEqual(args.cycles, 5)
        self.assertEqual(args.pruning_level, 0.3)
        self.assertEqual(args.growth_ratio, 0.5)
        self.assertEqual(args.controller_type, "ann")
        self.assertEqual(args.dry_run, False)

    @patch('scripts.upgrayedd.ModelUpgrader')
    @patch('scripts.upgrayedd.parse_args')
    @patch('scripts.upgrayedd.load_json_config')
    def test_main(self, mock_load_config, mock_parse_args, mock_upgrader):
        """Test main function"""
        # Set up mock arguments
        mock_args = MagicMock(
            model="gpt2",
            dataset="tiny_shakespeare",
            cycles=5,
            pruning_level=0.3,
            growth_ratio=0.5,
            controller_type="ann",
            dry_run=False,
            json_config=None,
            output_dir="./output"
        )
        mock_parse_args.return_value = mock_args
        
        # Mock the upgrader
        mock_instance = MagicMock()
        mock_instance.upgrade.return_value = True
        mock_upgrader.return_value = mock_instance
        
        # Import main after patching
        from scripts.upgrayedd import main
        
        # Call the function being tested
        result = main()
        
        # Check that upgrader was created with correct arguments
        mock_upgrader.assert_called_once()
        args = mock_upgrader.call_args[1]
        self.assertEqual(args["model_name"], "gpt2")
        self.assertEqual(args["output_dir"], "./output")
        
        # Check that upgrade was called and returned success
        mock_instance.upgrade.assert_called_once()
        self.assertEqual(result, 0)


if __name__ == '__main__':
    unittest.main()