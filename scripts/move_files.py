#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Movement Tool

This script helps move files from their current locations to new locations
according to the repository reorganization plan. It:

1. Creates the new directory if it doesn't exist
2. Copies the file to the new location with updated imports
3. Creates a compatibility stub at the old location
4. Logs the movement for later reference

Usage:
    python scripts/move_files.py [--dry-run] [--verbose] [module_name]
    
Arguments:
    --dry-run   Show what would be done without making changes
    --verbose   Show detailed information about each operation
    module_name Only move files for the specified module (e.g., 'pruning')
"""

import os
import re
import sys
import shutil
import argparse
from pathlib import Path
import importlib.util
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('file_movement.log')
    ]
)
logger = logging.getLogger('move_files')

# Load file movement mappings
def load_movement_plan(filename='FILE_MOVEMENT_PLAN.md'):
    """Load file movement mappings from the plan file."""
    movement_map = {}
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Find table sections
    table_pattern = r'\|\s*Current Path\s*\|\s*New Path\s*\|\s*\n\|[-\s|]+\n((?:\|[^|]+\|[^|]+\|\s*\n)+)'
    tables = re.findall(table_pattern, content)
    
    # Parse each table
    for table in tables:
        rows = table.strip().split('\n')
        for row in rows:
            cells = row.split('|')
            if len(cells) >= 3:
                source = cells[1].strip()
                target = cells[2].strip()
                
                # Skip entries with 'preserve filenames' note
                if 'preserve filenames' in target:
                    source_dir = source.rstrip('*').rstrip('/')
                    target_dir = target.split('(')[0].strip().rstrip('/')
                    
                    # Handle these separately
                    continue
                
                # Convert to proper paths
                source = source.lstrip('/')
                target = target.lstrip('/')
                
                # Skip non-path entries
                if not source or not target:
                    continue
                
                movement_map[source] = target
    
    return movement_map

# Update import statements
def update_imports(content, old_root_module, new_root_module):
    """Update import statements in file content."""
    # Regular imports
    pattern = r'from\s+{0}(\.\w+)+\s+import'.format(re.escape(old_root_module))
    replacement = r'from {0}\1 import'.format(new_root_module)
    content = re.sub(pattern, replacement, content)
    
    # Star imports
    pattern = r'from\s+{0}(\.\w+)*\s+import\s+\*'.format(re.escape(old_root_module))
    replacement = r'from {0}\1 import *'.format(new_root_module)
    content = re.sub(pattern, replacement, content)
    
    # Simple imports
    pattern = r'import\s+{0}(\.\w+)*'.format(re.escape(old_root_module))
    replacement = r'import {0}\1'.format(new_root_module)
    content = re.sub(pattern, replacement, content)
    
    return content

def create_stub_file(original_path, new_import_path):
    """Create a stub file for backward compatibility."""
    module_name = os.path.splitext(os.path.basename(original_path))[0]
    new_module_path = new_import_path.replace('/', '.')
    
    stub_content = f'''"""
DEPRECATED: This module has moved to {new_module_path}
This import stub will be removed in a future version.
"""
import warnings
warnings.warn(
    "Importing from {os.path.dirname(original_path).replace('/', '.')}.{module_name} is deprecated. "
    "Use {new_module_path} instead.",
    DeprecationWarning, 
    stacklevel=2
)

from {new_module_path} import *
'''
    
    with open(original_path, 'w') as f:
        f.write(stub_content)
    
    logger.info(f"Created stub file at {original_path}")

def move_file(source_path, target_path, dry_run=False, verbose=False):
    """Move a file from source to target with import updates."""
    # Ensure target directory exists
    target_dir = os.path.dirname(target_path)
    if not os.path.exists(target_dir) and not dry_run:
        os.makedirs(target_dir, exist_ok=True)
        if verbose:
            logger.info(f"Created directory: {target_dir}")
    
    # Read source file
    with open(source_path, 'r') as f:
        content = f.read()
    
    # Update imports
    old_module_parts = source_path.split('/')
    new_module_parts = target_path.split('/')
    
    # Handle special cases based on file paths
    if source_path.startswith('utils/'):
        content = update_imports(content, 'utils', 'sentinel.utils')
        content = update_imports(content, 'utils.pruning', 'sentinel.pruning')
        content = update_imports(content, 'utils.adaptive', 'sentinel.plasticity.adaptive')
    
    if source_path.startswith('models/'):
        content = update_imports(content, 'models', 'sentinel.models')
    
    if source_path.startswith('controller/'):
        content = update_imports(content, 'controller', 'sentinel.controller')
    
    if source_path.startswith('sentinel_data/'):
        content = update_imports(content, 'sentinel_data', 'sentinel.data')
    
    # Add more import update patterns as needed
    
    # Write to target location
    if not dry_run:
        with open(target_path, 'w') as f:
            f.write(content)
        logger.info(f"Moved file from {source_path} to {target_path}")
    else:
        logger.info(f"Would move file from {source_path} to {target_path}")
    
    # Create stub file for backward compatibility
    if not dry_run:
        # First, make sure the original directory still exists
        original_dir = os.path.dirname(source_path)
        if not os.path.exists(original_dir):
            os.makedirs(original_dir, exist_ok=True)
        
        # Create the stub
        new_module_path = os.path.splitext(target_path)[0].replace('/', '.')
        create_stub_file(source_path, new_module_path)
    else:
        logger.info(f"Would create stub file at {source_path}")
    
    return {
        "source": source_path,
        "target": target_path,
        "timestamp": datetime.now().isoformat()
    }

def move_files_from_plan(dry_run=False, verbose=False, module_filter=None):
    """Move files according to the movement plan."""
    movement_plan = load_movement_plan()
    results = []
    
    for source, target in movement_plan.items():
        # Skip if we're filtering by module and this isn't in the module
        if module_filter and module_filter not in source and module_filter not in target:
            continue
        
        # Handle glob patterns with preserve filenames note
        if '*' in source or 'preserve filenames' in target:
            # This is handled separately
            continue
        
        # Skip if source doesn't exist (might be a directory pattern)
        if not os.path.exists(source):
            if verbose:
                logger.warning(f"Source file does not exist: {source}")
            continue
        
        # If source is a file, move it
        if os.path.isfile(source):
            result = move_file(source, target, dry_run, verbose)
            results.append(result)
        
    # Save movement log
    if not dry_run:
        with open('file_movement_log.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    logger.info(f"Processed {len(results)} files")
    return results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="File Movement Tool")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be done without making changes")
    parser.add_argument('--verbose', action='store_true', help="Show detailed information")
    parser.add_argument('module', nargs='?', help="Only move files for the specified module")
    
    args = parser.parse_args()
    
    logger.info("Starting file movement")
    logger.info(f"Dry run: {args.dry_run}")
    if args.module:
        logger.info(f"Filtering by module: {args.module}")
    
    move_files_from_plan(args.dry_run, args.verbose, args.module)
    
    logger.info("File movement complete")

if __name__ == "__main__":
    main()