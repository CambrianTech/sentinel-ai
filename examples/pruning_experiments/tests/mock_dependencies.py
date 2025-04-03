"""
Mock dependencies for testing the pruning experiment framework.

This module provides mock implementations of external dependencies to avoid
import conflicts or network requests during testing.
"""

import sys
import unittest.mock as mock
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).parents[3].absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


def apply_mocks():
    """Apply all necessary mocks for testing."""
    # We need to mock these modules before they are imported
    mock_modules = {
        'datasets': mock.MagicMock(),
        'datasets.load_dataset': mock.MagicMock(),
        'transformers': mock.MagicMock(),
        'transformers.FlaxAutoModelForCausalLM': mock.MagicMock(),
    }
    
    # Apply the mocks
    for module_name, mock_obj in mock_modules.items():
        if module_name not in sys.modules:
            sys.modules[module_name] = mock_obj
        
    # Special case for datasets.load_dataset which is imported directly
    sys.modules['datasets'].load_dataset = mock.MagicMock()
    
    return mock_modules