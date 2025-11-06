#!/usr/bin/env python
"""
Run tests for the neural plasticity module.

This script runs unit tests for the neural plasticity module,
covering core algorithms, visualization, and training.

Usage:
  python tests/unit/utils/run_neural_plasticity_tests.py
"""

import os
import sys
import unittest
import pytest

# Add parent directory to path to allow importing from project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))


if __name__ == "__main__":
    # Run all tests in the test_neural_plasticity.py file
    pytest.main(['-xvs', os.path.join(os.path.dirname(__file__), 'test_neural_plasticity.py')])