#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Repository Reorganization Master Script

This script orchestrates the entire repository reorganization process by:
1. Analyzing the current repository structure
2. Creating the new directory structure
3. Moving files to their new locations
4. Updating imports and references
5. Running tests to ensure functionality
6. Generating reports on the reorganization

Usage:
    python scripts/reorganize_repository.py [--phase PHASE] [--dry-run]
"""

import os
import sys
import argparse
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'reorganization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger('reorganize')

# Define the phases of reorganization
PHASES = [
    'analyze',     # Analyze current structure
    'structure',   # Create new directory structure
    'core',        # Move core modules
    'scripts',     # Create/update entry point scripts
    'tests',       # Move and update tests
    'outputs',     # Reorganize output directories
    'cleanup',     # Remove stubs and temporary files
    'validate'     # Validate the reorganization
]

def run_command(command, description=None):
    """Run a shell command and log the output."""
    if description:
        logger.info(description)
    
    logger.info(f"Running command: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        logger.info(f"Command output:\n{result.stdout}")
        return True, result.stdout
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with error code {e.returncode}:\n{e.stderr}")
        return False, e.stderr

def run_phase_analyze(dry_run=False):
    """Run the analysis phase."""
    logger.info("=== PHASE: ANALYZE ===")
    
    # Run output directory analysis
    run_command(
        "python scripts/analyze_output_directories.py --output-file output_analysis.json",
        "Analyzing output directories"
    )
    
    # Run test coverage analysis
    run_command(
        "python scripts/analyze_test_coverage.py --output-file test_coverage.json",
        "Analyzing test coverage"
    )
    
    logger.info("Analysis phase complete. Review output_analysis.json and test_coverage.json for details.")
    return True

def run_phase_structure(dry_run=False):
    """Run the structure creation phase."""
    logger.info("=== PHASE: STRUCTURE ===")
    
    # Create the new directory structure
    cmd = "python scripts/create_new_structure.py"
    if dry_run:
        logger.info(f"DRY RUN: Would run '{cmd}'")
        return True
    
    success, _ = run_command(cmd, "Creating new directory structure")
    return success

def run_phase_core(dry_run=False):
    """Run the core module movement phase."""
    logger.info("=== PHASE: CORE ===")
    
    # Move core modules in stages
    modules = [
        'models',
        'controller',
        'pruning',
        'adaptive',
        'data'
    ]
    
    all_success = True
    for module in modules:
        cmd = f"python scripts/move_files.py --verbose {module}"
        if dry_run:
            cmd += " --dry-run"
        
        success, _ = run_command(cmd, f"Moving {module} module files")
        all_success = all_success and success
    
    return all_success

def run_phase_scripts(dry_run=False):
    """Run the entry point scripts phase."""
    logger.info("=== PHASE: SCRIPTS ===")
    
    # Create compatibility redirects for main entry points
    entry_points = [
        ('main.py', 'scripts/entry_points/inference.py'),
        ('train.py', 'scripts/entry_points/train.py'),
        ('generate_samples.py', 'scripts/entry_points/generate.py')
    ]
    
    all_success = True
    
    # Create redirects
    for source, target in entry_points:
        if not os.path.exists(target):
            logger.error(f"Target script {target} does not exist")
            all_success = False
            continue
        
        if dry_run:
            logger.info(f"DRY RUN: Would create compatibility redirect from {source} to {target}")
            continue
        
        try:
            # Create a redirect script
            with open(source, 'w') as f:
                f.write(f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DEPRECATED: This script has been moved to {target}
This file is kept for backward compatibility and will be removed in a future version.
"""

import sys
import warnings

warnings.warn(
    "This script has been moved to {target}. "
    "Please update your usage.",
    DeprecationWarning,
    stacklevel=2
)

# Forward to the new script
import subprocess
cmd = ["python", "{target}"] + sys.argv[1:]
sys.exit(subprocess.run(cmd).returncode)
''')
            logger.info(f"Created compatibility redirect from {source} to {target}")
        except Exception as e:
            logger.error(f"Failed to create compatibility redirect: {e}")
            all_success = False
    
    return all_success

def run_phase_tests(dry_run=False):
    """Run the test movement phase."""
    logger.info("=== PHASE: TESTS ===")
    
    # Move test files
    cmd = "python scripts/move_files.py --verbose test"
    if dry_run:
        cmd += " --dry-run"
    
    success, _ = run_command(cmd, "Moving test files")
    
    # Run tests to verify functionality
    if not dry_run and success:
        run_command("python -m unittest discover -s tests", "Running tests after reorganization")
    
    return success

def run_phase_outputs(dry_run=False):
    """Run the output directory reorganization phase."""
    logger.info("=== PHASE: OUTPUTS ===")
    
    # Create output directories (don't move files - just update scripts)
    dirs = [
        'experiments/results/pruning',
        'experiments/results/profiling',
        'experiments/results/validation',
        'experiments/results/plasticity',
        'experiments/results/benchmark',
        'experiments/results/demo'
    ]
    
    for dir_path in dirs:
        if dry_run:
            logger.info(f"DRY RUN: Would create directory {dir_path}")
        else:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory {dir_path}")
    
    logger.info("Output directories created. Scripts need to be manually updated to use these paths.")
    return True

def run_phase_cleanup(dry_run=False):
    """Run the cleanup phase."""
    logger.info("=== PHASE: CLEANUP ===")
    
    # This phase would remove compatibility stubs after ensuring everything works
    logger.info("CLEANUP phase requires manual verification and execution.")
    logger.info("Do not run this phase until all tests pass and functionality is verified.")
    
    if not dry_run:
        logger.warning("Cleanup phase should be run manually after thorough testing.")
    
    return True

def run_phase_validate(dry_run=False):
    """Run the validation phase."""
    logger.info("=== PHASE: VALIDATE ===")
    
    # Create a validation test script that imports from both old and new paths
    validation_script = "validation_test.py"
    if not dry_run:
        with open(validation_script, 'w') as f:
            f.write('''#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Validation script for repository reorganization.
This script attempts to import modules from both old and new paths
to ensure compatibility.
"""

import sys
import importlib

# Track successes and failures
successes = []
failures = []

def try_import(module_path):
    try:
        module = importlib.import_module(module_path)
        print(f"SUCCESS: Imported {module_path}")
        successes.append(module_path)
        return True
    except Exception as e:
        print(f"FAILURE: Could not import {module_path}: {e}")
        failures.append((module_path, str(e)))
        return False

# Test imports from old paths (just a small sample)
old_imports = [
    "utils.pruning.pruning_module",
    "utils.adaptive.adaptive_plasticity",
    "models.adaptive_transformer",
    "controller.controller_manager",
    "sentinel_data.dataset_loader"
]

# Test imports from new paths
new_imports = [
    "sentinel.pruning.pruning_module",
    "sentinel.plasticity.adaptive.adaptive_plasticity",
    "sentinel.models.adaptive.transformer",
    "sentinel.controller.controller_manager",
    "sentinel.data.loaders.dataset_loader"
]

print("=== Testing Old Imports ===")
for module_path in old_imports:
    try_import(module_path)

print("\n=== Testing New Imports ===")
for module_path in new_imports:
    try_import(module_path)

print("\n=== Summary ===")
print(f"Successful imports: {len(successes)}/{len(old_imports) + len(new_imports)}")
print(f"Failed imports: {len(failures)}/{len(old_imports) + len(new_imports)}")

if failures:
    print("\nFailures:")
    for module_path, error in failures:
        print(f"  {module_path}: {error}")
    sys.exit(1)
else:
    print("\nAll imports successful!")
    sys.exit(0)
''')
    
    if dry_run:
        logger.info(f"DRY RUN: Would run validation script {validation_script}")
        return True
    
    # Run the validation script
    success, _ = run_command(f"python {validation_script}", "Running validation script")
    
    # Run entry point tests
    if success:
        entry_points = [
            "scripts/entry_points/inference.py",
            "scripts/entry_points/train.py"
        ]
        
        for script in entry_points:
            if os.path.exists(script):
                script_success, _ = run_command(f"python {script} --help", f"Testing entry point {script}")
                success = success and script_success
    
    # Clean up validation script
    try:
        os.remove(validation_script)
    except:
        pass
    
    return success

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Repository Reorganization Master Script")
    parser.add_argument('--phase', choices=PHASES, help="Only run the specified phase")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    logger.info("Starting repository reorganization")
    
    if args.dry_run:
        logger.info("DRY RUN mode - no changes will be made")
    
    phases_to_run = [args.phase] if args.phase else PHASES
    phase_functions = {
        'analyze': run_phase_analyze,
        'structure': run_phase_structure,
        'core': run_phase_core,
        'scripts': run_phase_scripts,
        'tests': run_phase_tests,
        'outputs': run_phase_outputs,
        'cleanup': run_phase_cleanup,
        'validate': run_phase_validate
    }
    
    results = {}
    for phase in phases_to_run:
        logger.info(f"Running phase: {phase}")
        
        # Run the phase function
        try:
            success = phase_functions[phase](args.dry_run)
            results[phase] = success
            
            if not success:
                logger.error(f"Phase {phase} failed")
                if phase != phases_to_run[-1]:  # Not the last phase
                    if input("Continue to next phase? (y/n): ").lower() != 'y':
                        logger.info("Aborting reorganization")
                        break
        except Exception as e:
            logger.error(f"Phase {phase} failed with exception: {e}")
            results[phase] = False
            if phase != phases_to_run[-1]:  # Not the last phase
                if input("Continue to next phase? (y/n): ").lower() != 'y':
                    logger.info("Aborting reorganization")
                    break
    
    # Print summary
    logger.info("\n=== Reorganization Summary ===")
    for phase, success in results.items():
        status = "SUCCESS" if success else "FAILURE"
        logger.info(f"{phase}: {status}")
    
    all_success = all(results.values())
    if all_success:
        logger.info("All phases completed successfully")
    else:
        logger.warning("Some phases failed - check the log for details")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())