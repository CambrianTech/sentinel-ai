#!/usr/bin/env python
"""
Comprehensive test runner for the Sentinel AI codebase.

This script runs all tests and generates a coverage report to ensure
our code quality remains high. It's designed to be used in CI pipelines
or locally for development.

Usage:
    python run_tests.py [--verbose] [--unit-only] [--integration-only] [--coverage]
"""

import os
import sys
import argparse
import subprocess
import tempfile


def run_command(command, verbose=False):
    """Run a command and return the output."""
    if verbose:
        print(f"Running: {' '.join(command)}")
    
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if verbose or result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
    
    return result.returncode


def run_unit_tests(args):
    """Run all unit tests."""
    print("\n=== Running Unit Tests ===\n")
    
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests/unit"]
    
    if args.verbose:
        command.append("-v")
    
    return run_command(command, verbose=args.verbose)


def run_integration_tests(args):
    """Run all integration tests."""
    print("\n=== Running Integration Tests ===\n")
    
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests/integration"]
    
    if args.verbose:
        command.append("-v")
    
    return run_command(command, verbose=args.verbose)


def run_functional_tests(args):
    """Run the functional test scripts."""
    print("\n=== Running Functional Tests ===\n")
    
    test_scripts = [
        "test_dataset_loading.py",
        "test_run_experiment.py",
        "test_mock_upgrader.py",
        "test_model_support.py",
        "test_neural_plasticity.py",
        "test_optimization_comparison.py"
    ]
    
    success = True
    for script in test_scripts:
        script_path = os.path.join(os.getcwd(), script)
        if os.path.exists(script_path):
            print(f"\nRunning {script}...")
            command = [sys.executable, "-u", script_path]
            result = run_command(command, verbose=args.verbose)
            if result != 0:
                success = False
        else:
            print(f"Warning: Test script {script} not found")
    
    return 0 if success else 1


def run_colab_tests(args):
    """Run tests for colab notebooks."""
    print("\n=== Running Colab Notebook Tests ===\n")
    
    # List of supported colab scripts with command-line arguments
    colab_scripts_with_args = [
        # Test PruningAndFineTuningColab.py with super_simple mode
        {
            "path": "colab_notebooks/PruningAndFineTuningColab.py",
            "args": ["--test_mode", "--super_simple", "--model_name", "distilgpt2"],
            "description": "with super_simple mode"
        }
    ]
    
    # Test scripts with arguments
    success = True
    for script in colab_scripts_with_args:
        if os.path.exists(script["path"]):
            print(f"\nTesting {script['path']} {script['description']}...")
            command = [sys.executable, "-u", script["path"]] + script["args"]
            result = run_command(command, verbose=args.verbose)
            if result != 0:
                success = False
                print(f"❌ Test failed for {script['path']}")
            else:
                print(f"✅ Test passed for {script['path']}")
        else:
            print(f"⚠️ Warning: Colab script {script['path']} not found")
    
    # Create a test module for scripts without command-line arguments
    scripts_without_args = [
        {
            "path": "colab_notebooks/UpgrayeddAPI.py",
            "description": "by monkey patching for testing"
        }
    ]
    
    print("\nTesting notebooks without command-line support:")
    for script in scripts_without_args:
        if os.path.exists(script["path"]):
            # Create a temporary test script
            test_script = f"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Monkey patch functions that require user input or slow operations
with patch('torch.cuda.is_available', return_value=False), \\
     patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=MagicMock()), \\
     patch('transformers.AutoTokenizer.from_pretrained', return_value=MagicMock()):
    
    # Import the module (will use the monkey patched functions)
    try:
        import {script["path"].replace("/", ".").replace(".py", "")}
        print("✅ Successfully imported {script["path"]}")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error importing {script["path"]}: {{e}}")
        sys.exit(1)
"""
            
            # Write the test script to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
                f.write(test_script)
                temp_script = f.name
            
            try:
                # Run the test script
                print(f"\nTesting {script['path']} {script['description']}...")
                command = [sys.executable, "-u", temp_script]
                result = run_command(command, verbose=args.verbose)
                if result != 0:
                    success = False
                    print(f"❌ Test failed for {script['path']}")
                else:
                    print(f"✅ Test passed for {script['path']}")
            finally:
                # Clean up
                os.unlink(temp_script)
        else:
            print(f"⚠️ Warning: Colab script {script['path']} not found")
    
    # If we need to test notebook conversion and execution
    notebook_files = [
        "colab_notebooks/UpgrayeddColab.ipynb",
        "colab_notebooks/UpgrayeddContinuous.ipynb", 
        "colab_notebooks/PruningAndFineTuningColab.ipynb"
    ]
    
    # Just count how many notebook files we have
    notebook_count = sum(1 for nb in notebook_files if os.path.exists(nb))
    print(f"\nFound {notebook_count}/{len(notebook_files)} Jupyter notebooks")
    
    if args.verbose:
        print("\nChecking Jupyter notebook files:")
        for notebook in notebook_files:
            if os.path.exists(notebook):
                print(f"  ✅ {notebook} exists")
            else:
                print(f"  ❌ {notebook} not found")
    
    return 0 if success else 1


def run_coverage(args):
    """Run tests with coverage."""
    if not args.coverage:
        return 0
    
    print("\n=== Running Coverage Analysis ===\n")
    
    try:
        import coverage
    except ImportError:
        print("Coverage package not found. Install with pip install coverage")
        return 1
    
    # Create a coverage configuration file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.coveragerc', delete=False) as f:
        f.write("""
[run]
source =
    sentinel/upgrayedd
    sentinel/pruning
    colab_notebooks

omit =
    */tests/*
    */test_*
    */__pycache__/*
    colab_notebooks/*.ipynb

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if __name__ == .__main__.:
    pass
    raise ImportError
        """)
        coveragerc = f.name
    
    try:
        # Run coverage
        command = [
            "coverage", "run", "--rcfile", coveragerc,
            "-m", "unittest", "discover"
        ]
        result = run_command(command, verbose=args.verbose)
        
        # Generate report
        if result == 0:
            print("\nCoverage Report:")
            report_command = ["coverage", "report", "--rcfile", coveragerc]
            run_command(report_command, verbose=True)
        
        return result
    finally:
        # Clean up
        os.unlink(coveragerc)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run tests for Sentinel AI")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--coverage", action="store_true", help="Run coverage analysis")
    args = parser.parse_args()
    
    # Track overall success
    success = True
    
    # Run tests based on arguments
    if args.unit_only:
        # Run only unit tests
        result = run_unit_tests(args)
        success = result == 0
    elif args.integration_only:
        # Run only integration tests
        result = run_integration_tests(args)
        success = result == 0
    else:
        # Run all tests
        unit_result = run_unit_tests(args)
        integration_result = run_integration_tests(args)
        functional_result = run_functional_tests(args)
        colab_result = run_colab_tests(args)
        
        success = (
            unit_result == 0 and
            integration_result == 0 and
            functional_result == 0 and
            colab_result == 0
        )
    
    # Run coverage if requested
    if args.coverage:
        coverage_result = run_coverage(args)
        success = success and coverage_result == 0
    
    # Exit with appropriate code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())