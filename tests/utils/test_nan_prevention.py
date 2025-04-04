"""
Tests for NaN prevention stability functions.
"""

import unittest
import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root))

# Import the module to test
from utils.pruning.stability.nan_prevention import (
    replace_nans,
    is_safe_loss,
    create_safe_loss_fn,
    replace_nans_in_gradient
)


class TestNanPrevention(unittest.TestCase):
    """Test NaN prevention functions."""
    
    def test_replace_nans(self):
        """Test NaN replacement in arrays."""
        # Create test array with NaNs
        test_array = np.array([1.0, 2.0, np.nan, 4.0, np.nan])
        
        # Replace NaNs with 0.0
        cleaned = replace_nans(test_array, replacement=0.0)
        
        # Check if NaNs are replaced
        self.assertTrue(np.array_equal(cleaned, np.array([1.0, 2.0, 0.0, 4.0, 0.0])))
        
        # Test with different replacement
        cleaned2 = replace_nans(test_array, replacement=-1.0)
        self.assertTrue(np.array_equal(cleaned2, np.array([1.0, 2.0, -1.0, 4.0, -1.0])))
    
    def test_is_safe_loss(self):
        """Test loss safety checking."""
        # Safe loss
        self.assertTrue(is_safe_loss(1.5))
        self.assertTrue(is_safe_loss(np.array(2.0)))
        
        # Unsafe loss (NaN)
        self.assertFalse(is_safe_loss(np.nan))
        self.assertFalse(is_safe_loss(np.array(np.nan)))
        
        # Unsafe loss (Inf)
        self.assertFalse(is_safe_loss(np.inf))
        self.assertFalse(is_safe_loss(np.array(np.inf)))
        
        # Unsafe loss (negative large value)
        self.assertFalse(is_safe_loss(-1e10))
    
    def test_safe_loss_fn(self):
        """Test safe loss function creation."""
        # Create a dummy loss function that sometimes returns NaN
        def dummy_loss(params, batch):
            if batch == "nan":
                return np.nan
            return 1.0
        
        # Create safe version
        safe_loss = create_safe_loss_fn(dummy_loss)
        
        # Test with regular input
        self.assertEqual(safe_loss(None, "good"), 1.0)
        
        # Test with NaN-producing input
        self.assertEqual(safe_loss(None, "nan"), 10.0)  # Default fallback loss
        
        # Test with custom fallback
        safe_loss_custom = create_safe_loss_fn(dummy_loss, fallback=5.0)
        self.assertEqual(safe_loss_custom(None, "nan"), 5.0)
    
    def test_replace_nans_in_gradient(self):
        """Test NaN replacement in gradients."""
        # Create a dummy gradient with some NaNs
        dummy_grads = {
            'layer1': {
                'weight': np.array([1.0, np.nan, 3.0]),
                'bias': np.array([np.nan, 2.0])
            },
            'layer2': {
                'weight': np.array([[1.0, 2.0], [np.nan, 4.0]]),
                'bias': np.array([5.0, 6.0])
            }
        }
        
        # Replace NaNs with zeros
        cleaned_grads = replace_nans_in_gradient(dummy_grads)
        
        # Check if NaNs are replaced
        self.assertEqual(cleaned_grads['layer1']['weight'][1], 0.0)
        self.assertEqual(cleaned_grads['layer1']['bias'][0], 0.0)
        self.assertEqual(cleaned_grads['layer2']['weight'][1, 0], 0.0)
        
        # Original values should be untouched
        self.assertEqual(cleaned_grads['layer1']['weight'][0], 1.0)
        self.assertEqual(cleaned_grads['layer1']['weight'][2], 3.0)
        self.assertEqual(cleaned_grads['layer1']['bias'][1], 2.0)
        self.assertEqual(cleaned_grads['layer2']['bias'][0], 5.0)


if __name__ == "__main__":
    unittest.main()