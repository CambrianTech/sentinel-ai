"""
Tests for the Colab integration utilities.

These tests ensure that the Colab integration utilities work correctly
in both Colab and local environments.
"""

import os
import sys
import unittest
import tempfile
import json
from unittest.mock import patch, MagicMock, PropertyMock
import pytest
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

# Import modules to test
from scripts.neural_plasticity.colab.integration import (
    is_colab,
    is_apple_silicon,
    has_gpu,
    get_environment_info,
    get_output_dir,
    save_experiment_results,
    load_experiment_results
)

class TestColabIntegration(unittest.TestCase):
    """Test the Colab integration utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temp directory for outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_is_colab_detection(self):
        """Test Colab environment detection."""
        # Mock the google.colab module not existing
        with patch.dict(sys.modules, {'google.colab': None}):
            with patch('importlib.util.find_spec', return_value=None):
                self.assertFalse(is_colab())
        
        # Mock the google.colab module existing
        mock_colab = MagicMock()
        with patch.dict(sys.modules, {'google.colab': mock_colab}):
            with patch('importlib.util.find_spec', return_value=MagicMock()):
                self.assertTrue(is_colab())
    
    @patch('platform.system')
    @patch('platform.processor')
    def test_is_apple_silicon_detection(self, mock_processor, mock_system):
        """Test Apple Silicon detection."""
        # Not Apple Silicon
        mock_system.return_value = 'Darwin'
        mock_processor.return_value = 'i386'
        self.assertFalse(is_apple_silicon())
        
        # Not macOS
        mock_system.return_value = 'Linux'
        mock_processor.return_value = 'arm'
        self.assertFalse(is_apple_silicon())
        
        # Apple Silicon
        mock_system.return_value = 'Darwin'
        mock_processor.return_value = 'arm'
        self.assertTrue(is_apple_silicon())
    
    @patch('torch.cuda.is_available')
    def test_has_gpu_detection(self, mock_is_available):
        """Test GPU detection."""
        # No GPU
        mock_is_available.return_value = False
        self.assertFalse(has_gpu())
        
        # GPU available
        mock_is_available.return_value = True
        self.assertTrue(has_gpu())
        
        # Handle ImportError
        with patch.dict(sys.modules, {'torch': None}):
            with patch('importlib.util.find_spec', return_value=None):
                self.assertFalse(has_gpu())
    
    @patch('scripts.neural_plasticity.colab.integration.is_colab')
    @patch('scripts.neural_plasticity.colab.integration.is_apple_silicon')
    @patch('scripts.neural_plasticity.colab.integration.has_gpu')
    def test_get_environment_info(self, mock_has_gpu, mock_is_apple_silicon, mock_is_colab):
        """Test environment info collection."""
        # Set up mocks
        mock_is_colab.return_value = False
        mock_is_apple_silicon.return_value = True
        mock_has_gpu.return_value = False
        
        # Get environment info
        info = get_environment_info()
        
        # Check basic fields
        self.assertFalse(info['is_colab'])
        self.assertTrue(info['is_apple_silicon'])
        self.assertFalse(info['has_gpu'])
        self.assertIn('platform', info)
        self.assertIn('python_version', info)
        self.assertIn('timestamp', info)
        
        # Check with GPU available
        mock_has_gpu.return_value = True
        with patch('torch.cuda.get_device_name', return_value='NVIDIA T4'):
            with patch('torch.cuda.device_count', return_value=1):
                # Create a mock object with a string value for cuda version
                cuda_version = '11.7'
                mock_version = MagicMock()
                mock_version.cuda = cuda_version
                
                with patch('torch.version', mock_version):
                    info = get_environment_info()
                    self.assertTrue(info['has_gpu'])
                    self.assertEqual(info['gpu_type'], 'NVIDIA T4')
                    self.assertEqual(info['gpu_count'], 1)
                    self.assertEqual(info['cuda_version'], '11.7')
    
    @patch('scripts.neural_plasticity.colab.integration.is_colab')
    def test_get_output_dir_local(self, mock_is_colab):
        """Test getting output directory in local environment."""
        # Set up mocks
        mock_is_colab.return_value = False
        
        # Get output directory
        output_dir = get_output_dir('test_experiment', base_dir=self.temp_dir)
        
        # Check output directory
        self.assertTrue(os.path.exists(output_dir))
        self.assertTrue(output_dir.startswith(self.temp_dir))
        self.assertIn('test_experiment', output_dir)
        
        # Check environment info file
        env_info_path = os.path.join(output_dir, 'environment_info.json')
        self.assertTrue(os.path.exists(env_info_path))
        
        # Check content of environment info file
        with open(env_info_path, 'r') as f:
            env_info = json.load(f)
            self.assertIn('is_colab', env_info)
            self.assertIn('is_apple_silicon', env_info)
            self.assertIn('platform', env_info)
    
    @patch('scripts.neural_plasticity.colab.integration.is_colab')
    def test_save_load_experiment_results(self, mock_is_colab):
        """Test saving and loading experiment results."""
        # Set up mocks
        mock_is_colab.return_value = False
        
        # Create test results
        results = {
            'metrics': {
                'baseline': {'loss': 10.0, 'perplexity': 1000.0},
                'final': {'loss': 5.0, 'perplexity': 500.0}
            },
            'pruned_heads': [(0, 0, 0.1), (0, 1, 0.2)],
            'hyperparameters': {
                'model': 'distilgpt2',
                'strategy': 'entropy',
                'pruning_level': 0.2
            }
        }
        
        # Save results
        results_path = save_experiment_results(results, 'test_save', output_dir=self.temp_dir)
        
        # Check results file
        self.assertTrue(os.path.exists(results_path))
        
        # Load results
        loaded_results = load_experiment_results(results_path)
        
        # Check loaded results
        self.assertEqual(loaded_results['metrics']['baseline']['loss'], 10.0)
        self.assertEqual(loaded_results['metrics']['final']['perplexity'], 500.0)
        self.assertEqual(len(loaded_results['pruned_heads']), 2)
        self.assertEqual(loaded_results['hyperparameters']['model'], 'distilgpt2')
        
        # Test loading from directory
        loaded_results_dir = load_experiment_results(os.path.dirname(results_path))
        self.assertEqual(loaded_results_dir['metrics']['baseline']['loss'], 10.0)

if __name__ == '__main__':
    unittest.main()