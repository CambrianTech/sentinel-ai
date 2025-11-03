"""
Run the visualization tests.

This script runs the visualization_additions tests and reports the results.
It handles dependency issues gracefully to avoid CI/CD failures.
"""

import unittest
import sys

try:
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies not available ({e}). Tests will be skipped.")
    DEPENDENCIES_AVAILABLE = False


def run_tests():
    """Run the visualization tests."""
    if not DEPENDENCIES_AVAILABLE:
        print("Skipping tests due to missing dependencies.")
        print("To run tests, install: torch, numpy, matplotlib")
        return 0
    
    # Import test modules
    from test_visualization_additions import TestVisualizationAdditions
    
    # Create test suite
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestVisualizationAdditions))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())