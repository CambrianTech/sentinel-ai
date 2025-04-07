#!/usr/bin/env python
"""
Test runner for sentinel-ai project.

This script runs all tests or specific test categories, with support for
coverage analysis and different verbosity levels.
"""

import os
import sys
import argparse
import unittest
import importlib.util
from typing import List, Optional

# Add project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import dependencies if available
try:
    import pytest
    has_pytest = True
except ImportError:
    has_pytest = False

try:
    import coverage
    has_coverage = True
except ImportError:
    has_coverage = False

def discover_tests(test_dirs: List[str], pattern: str = 'test_*.py') -> unittest.TestSuite:
    """
    Discover tests in the specified directories matching the pattern.
    
    Args:
        test_dirs: List of directories to search for tests
        pattern: Pattern to match test files
        
    Returns:
        TestSuite containing all discovered tests
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            print(f"Discovering tests in {test_dir}...")
            test_suite = loader.discover(test_dir, pattern=pattern)
            suite.addTest(test_suite)
        else:
            print(f"Warning: Test directory {test_dir} does not exist")
    
    return suite

def run_unittest_suite(suite: unittest.TestSuite, verbosity: int = 2) -> bool:
    """
    Run a test suite using unittest runner.
    
    Args:
        suite: TestSuite to run
        verbosity: Verbosity level (1=quiet, 2=verbose)
        
    Returns:
        True if all tests passed, False otherwise
    """
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result.wasSuccessful()

def run_pytest(test_dirs: List[str], verbosity: str = '-v') -> bool:
    """
    Run tests using pytest.
    
    Args:
        test_dirs: List of directories to run tests from
        verbosity: Verbosity flag for pytest
        
    Returns:
        True if all tests passed, False otherwise
    """
    if not has_pytest:
        print("Warning: pytest not available, falling back to unittest")
        return run_unittest_suite(discover_tests(test_dirs))
    
    args = [verbosity]
    args.extend(test_dirs)
    return pytest.main(args) == 0

def run_coverage(
    test_dirs: List[str], 
    source_dirs: List[str] = ['sentinel'], 
    omit: Optional[List[str]] = None,
    html_report: bool = False
) -> bool:
    """
    Run tests with coverage analysis.
    
    Args:
        test_dirs: List of directories to run tests from
        source_dirs: List of source directories to measure coverage for
        omit: List of patterns to omit from coverage
        html_report: Whether to generate an HTML coverage report
        
    Returns:
        True if all tests passed, False otherwise
    """
    if not has_coverage:
        print("Warning: coverage not available, running tests without coverage")
        return run_unittest_suite(discover_tests(test_dirs))
    
    # Default patterns to omit
    if omit is None:
        omit = [
            '*/venv/*',
            '*/site-packages/*',
            '*/tests/*',
            '*/__pycache__/*',
            '*.pyc',
        ]
    
    # Start coverage measurement
    cov = coverage.Coverage(source=source_dirs, omit=omit)
    cov.start()
    
    # Run tests
    result = run_unittest_suite(discover_tests(test_dirs))
    
    # Stop coverage measurement
    cov.stop()
    cov.save()
    
    # Print coverage report
    print("\nCoverage Report:")
    cov.report()
    
    # Generate HTML report if requested
    if html_report:
        html_dir = os.path.join(project_root, 'htmlcov')
        print(f"\nGenerating HTML coverage report in {html_dir}")
        cov.html_report(directory=html_dir)
    
    return result

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description='Run tests for sentinel-ai project')
    
    parser.add_argument(
        '--category', '-c', 
        choices=['all', 'unit', 'integration', 'colab'],
        default='all',
        help='Test category to run (default: all)'
    )
    
    parser.add_argument(
        '--coverage', 
        action='store_true',
        help='Run tests with coverage analysis'
    )
    
    parser.add_argument(
        '--html-report', 
        action='store_true',
        help='Generate HTML coverage report'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Run tests with verbose output'
    )
    
    parser.add_argument(
        '--dirs',
        type=str,
        nargs='+',
        help='Specific test directories to run'
    )
    
    return parser.parse_args()

def get_test_dirs(category: str, custom_dirs: Optional[List[str]] = None) -> List[str]:
    """
    Get test directories based on category or custom dirs.
    
    Args:
        category: Test category (all, unit, integration, colab)
        custom_dirs: Custom test directories to use instead of category
        
    Returns:
        List of test directories to run
    """
    # Use custom dirs if provided
    if custom_dirs:
        return custom_dirs
    
    # Base test directory
    base_dir = os.path.join(project_root, 'tests')
    
    if category == 'all':
        # Test all categories
        return [
            os.path.join(base_dir, 'unit'),
            os.path.join(base_dir, 'integration'),
            os.path.join(base_dir, 'colab'),
        ]
    elif category == 'unit':
        # Only unit tests
        return [os.path.join(base_dir, 'unit')]
    elif category == 'integration':
        # Only integration tests
        return [os.path.join(base_dir, 'integration')]
    elif category == 'colab':
        # Only colab notebook tests
        return [os.path.join(base_dir, 'colab')]
    else:
        # Default to all tests
        return [base_dir]

def main() -> int:
    """
    Main entry point for test runner.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Parse arguments
    args = parse_args()
    
    # Determine test directories
    test_dirs = get_test_dirs(args.category, args.dirs)
    
    # Set verbosity level
    verbosity = 2 if args.verbose else 1
    
    print(f"\nRunning tests in:\n- " + "\n- ".join(test_dirs))
    
    # Run tests with or without coverage
    if args.coverage:
        success = run_coverage(
            test_dirs, 
            source_dirs=['sentinel'],
            html_report=args.html_report
        )
    else:
        suite = discover_tests(test_dirs)
        success = run_unittest_suite(suite, verbosity=verbosity)
    
    # Print summary
    if success:
        print("\n✅ All tests passed!\n")
        return 0
    else:
        print("\n❌ Tests failed!\n")
        return 1

if __name__ == '__main__':
    sys.exit(main())