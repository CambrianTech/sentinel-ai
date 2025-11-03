"""
Tests for the models module structure and functionality.
This will help ensure our refactoring doesn't break existing functionality.
"""

import os
import sys
import unittest

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sentinel.models import AdaptiveCausalLmWrapper, AgencySpecialization, SpecializationRegistry
from sentinel.models.loaders import ModelLoader


class TestModelsStructure(unittest.TestCase):
    """Test the structure and basic functionality of the models module."""
    
    def test_imports(self):
        """Test that the key classes can be imported."""
        # These imports should work without errors
        self.assertIsNotNone(AdaptiveCausalLmWrapper)
        self.assertIsNotNone(AgencySpecialization)
        self.assertIsNotNone(SpecializationRegistry)
        self.assertIsNotNone(ModelLoader)
    
    def test_backward_compatibility(self):
        """Test that the backward compatibility imports work."""
        # Import from the old location
        import models
        
        # They should import something, not None
        self.assertTrue(hasattr(models, 'AdaptiveCausalLmWrapper'))
        self.assertTrue(hasattr(models, 'AgencySpecialization'))
        
        # Should be able to import these without errors
        from models import AdaptiveCausalLmWrapper as OldWrapper
        from models.agency_specialization import AgencySpecialization as OldSpecialization
        
        # Check not None
        self.assertIsNotNone(OldWrapper)
        self.assertIsNotNone(OldSpecialization)
    
    def test_model_loader_class(self):
        """Test that the ModelLoader base class has the expected methods."""
        # Create instance
        loader = ModelLoader(debug=True)
        
        # Check methods
        self.assertTrue(hasattr(loader, 'load_model'))
        self.assertTrue(hasattr(loader, 'configure_gates'))
        
        # Check attributes
        self.assertTrue(hasattr(loader, 'use_optimized'))
        self.assertTrue(hasattr(loader, 'debug'))


if __name__ == '__main__':
    unittest.main()