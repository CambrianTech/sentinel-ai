#!/usr/bin/env python
"""
CI-specific test runner for sentinel-ai project.

This script runs tests in a way that works with CI systems, handling edge cases
and ensuring clean exit codes for automation.
"""

import os
import sys
import unittest
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

def run_tests():
    """Run all tests in a CI-compatible way."""
    print("Running CI tests for sentinel-ai")
    
    # Get test directories - we'll run each separately for stability
    base_test_dir = os.path.join(project_root, 'tests')
    unit_test_dir = os.path.join(base_test_dir, 'unit')
    
    # Get all subdirectories in unit tests
    test_dirs = []
    for entry in os.scandir(unit_test_dir):
        if entry.is_dir() and not entry.name.startswith('__'):
            test_dirs.append(entry.path)
    
    # Run each test subdirectory separately with proper error handling
    all_success = True
    
    for test_dir in test_dirs:
        dir_name = os.path.basename(test_dir)
        print(f"\n===== Running tests in {dir_name} =====")
        
        # Special handling for models dir which can have import issues
        if dir_name == "models":
            models_test_file = os.path.join(test_dir, "test_models_structure.py")
            if os.path.exists(models_test_file):
                print(f"Running models test directly: {models_test_file}")
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "unittest", models_test_file],
                        check=False,
                        capture_output=True,
                        text=True
                    )
                    print(result.stdout)
                    if result.stderr:
                        print(f"STDERR: {result.stderr}")
                    
                    # Success is determined by returncode
                    success = (result.returncode == 0)
                    all_success = all_success and success
                    
                    if success:
                        print("✅ Tests PASSED")
                    else:
                        print("❌ Tests FAILED")
                    
                    # Continue to next directory
                    continue
                except Exception as e:
                    print(f"Error running models test directly: {e}")
                    # Fall through to standard test discovery as backup
        
        # Use standard unittest discovery for other dirs
        loader = unittest.TestLoader()
        suite = loader.discover(test_dir, pattern="test_*.py")
        
        # Count tests
        test_count = 0
        for test_case in suite:
            for test in test_case:
                test_count += 1
        
        if test_count == 0:
            print(f"No tests found in {dir_name}")
            continue
            
        print(f"Running {test_count} tests in {dir_name}")
        
        # Run the suite
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Check success
        success = result.wasSuccessful()
        all_success = all_success and success
        
        if success:
            print(f"✅ All {dir_name} tests PASSED")
        else:
            print(f"❌ Some {dir_name} tests FAILED")
    
    # Final summary
    if all_success:
        print("\n✅ All tests PASSED")
        return 0
    else:
        print("\n❌ Some tests FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())