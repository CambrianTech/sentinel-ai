#!/usr/bin/env python
"""
Neural Plasticity Redundant Files Purge Script

This script reads the PURGE_LIST.txt file and removes files that are marked for deletion.
It helps consolidate the neural plasticity implementation by removing redundant files.

Usage:
    python scripts/neural_plasticity/purge_redundant_files.py
    python scripts/neural_plasticity/purge_redundant_files.py --dry-run
    python scripts/neural_plasticity/purge_redundant_files.py --force
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

def read_purge_list(purge_list_path):
    """Read the purge list file and return a list of files to remove."""
    try:
        with open(purge_list_path, 'r') as f:
            lines = f.readlines()
        
        # Extract valid file paths from the list (ignore comments and empty lines)
        files_to_remove = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('EOL'):
                # Check if the line is a valid file path
                if os.path.exists(line):
                    files_to_remove.append(line)
                # Try to handle the case with project root
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                abs_path = os.path.join(project_root, line.lstrip('/'))
                if os.path.exists(abs_path):
                    files_to_remove.append(abs_path)
                
        return files_to_remove
    except Exception as e:
        print(f"Error reading purge list: {e}")
        return []

def purge_files(files_to_remove, dry_run=False, force=False):
    """Remove files in the purge list."""
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                if dry_run:
                    print(f"Would remove file: {file_path}")
                else:
                    try:
                        os.remove(file_path)
                        print(f"Removed file: {file_path}")
                    except Exception as e:
                        print(f"Error removing file {file_path}: {e}")
            elif os.path.isdir(file_path):
                if dry_run:
                    print(f"Would remove directory: {file_path}")
                else:
                    try:
                        if force:
                            shutil.rmtree(file_path)
                            print(f"Removed directory: {file_path}")
                        else:
                            # Check if directory is empty
                            if not os.listdir(file_path):
                                os.rmdir(file_path)
                                print(f"Removed empty directory: {file_path}")
                            else:
                                print(f"Directory not empty, use --force to remove: {file_path}")
                    except Exception as e:
                        print(f"Error removing directory {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Purge redundant files from the neural plasticity implementation.')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be removed without actually removing')
    parser.add_argument('--force', action='store_true', help='Force removal of non-empty directories')
    args = parser.parse_args()
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    purge_list_path = os.path.join(project_root, 'PURGE_LIST.txt')
    
    # Check if purge list exists
    if not os.path.exists(purge_list_path):
        print(f"Purge list not found: {purge_list_path}")
        return 1
    
    # Read purge list
    files_to_remove = read_purge_list(purge_list_path)
    
    if not files_to_remove:
        print("No files to remove.")
        return 0
    
    # Print summary
    print(f"Found {len(files_to_remove)} files/directories to remove.")
    
    # Purge files
    purge_files(files_to_remove, dry_run=args.dry_run, force=args.force)
    
    # Print completion message
    if args.dry_run:
        print(f"Dry run completed. Would have removed {len(files_to_remove)} files/directories.")
    else:
        print(f"Purge completed. Removed files and directories as specified in {purge_list_path}.")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())