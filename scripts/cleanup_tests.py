#!/usr/bin/env python
"""
Cleanup script to move scattered test files into a proper test directory.

This script:
1. Identifies test_*.py files in the root directory
2. Creates a tests/ directory if it doesn't exist
3. Moves the identified files to the tests/ directory
4. Leaves any existing tests in place
"""

import os
import sys
import shutil
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def create_test_directory(test_dir="tests"):
    """Create the test directory if it doesn't exist."""
    if not os.path.exists(test_dir):
        logger.info(f"Creating directory: {test_dir}")
        os.makedirs(test_dir)
    return os.path.abspath(test_dir)

def find_test_files(root_dir="."):
    """Find all test_*.py files in the root directory."""
    test_files = []
    for filename in os.listdir(root_dir):
        if filename.startswith("test_") and filename.endswith(".py") and os.path.isfile(os.path.join(root_dir, filename)):
            test_files.append(filename)
    
    logger.info(f"Found {len(test_files)} test files in root directory")
    return test_files

def move_test_files(test_files, test_dir, backup=True):
    """Move test files to the test directory with optional backup."""
    moved_files = []
    backup_dir = os.path.join(test_dir, "backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    for filename in test_files:
        source = os.path.abspath(filename)
        destination = os.path.join(test_dir, filename)
        
        # Create backup if requested
        if backup and os.path.exists(destination):
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            
            backup_file = os.path.join(backup_dir, filename)
            logger.info(f"Backing up existing file: {destination} -> {backup_file}")
            shutil.copy2(destination, backup_file)
        
        # Move the file
        logger.info(f"Moving: {source} -> {destination}")
        shutil.move(source, destination)
        moved_files.append(filename)
    
    return moved_files

def main():
    parser = argparse.ArgumentParser(description="Clean up test files by moving them to a tests/ directory")
    parser.add_argument("--test-dir", type=str, default="tests",
                      help="Directory to move test files to (default: tests/)")
    parser.add_argument("--no-backup", action="store_true",
                      help="Don't backup existing files in the test directory")
    args = parser.parse_args()
    
    # Create test directory
    test_dir = create_test_directory(args.test_dir)
    
    # Find test files in root directory
    test_files = find_test_files()
    
    if not test_files:
        logger.info("No test files found in root directory")
        return
    
    # Move files
    moved_files = move_test_files(test_files, test_dir, backup=not args.no_backup)
    
    logger.info(f"Moved {len(moved_files)} test files to {test_dir}")
    logger.info("Cleanup complete")

if __name__ == "__main__":
    main()