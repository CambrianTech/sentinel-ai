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
    
    colab_script = "colab_notebooks/PruningAndFineTuningColab.py"
    if os.path.exists(colab_script):
        print(f"Testing {colab_script} with super_simple mode...")
        command = [
            sys.executable, "-u", colab_script,
            "--test_mode", "--super_simple", "--model_name", "distilgpt2"
        ]
        return run_command(command, verbose=args.verbose)
    else:
        print(f"Warning: Colab script {colab_script} not found")
        return 0


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

omit =
    */tests/*
    */test_*
    */__pycache__/*

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