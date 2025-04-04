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
    
    if verbose:
        logger.info(f"Loaded movement plan with {len(movement_plan)} entries")
    
    # For now, let's print what we loaded from the movement plan
    if verbose:
        for source, target in movement_plan.items():
            logger.info(f"Plan entry: {source} -> {target}")
    
    # Handle specific modules based on the module_filter
    if module_filter == 'models':
        sources = [
            '/models/adaptive_transformer.py',
            '/models/agency_specialization.py',
            '/models/bloom_adapter.py',
            '/models/llama_adapter.py',
            '/models/optimized_attention.py',
            '/models/specialization_registry.py',
            '/models/unet_transformer.py',
            '/models/unet_transformer_optimized.py'
        ]
        targets = [
            '/sentinel/models/adaptive/transformer.py',
            '/sentinel/models/adaptive/agency_specialization.py',
            '/sentinel/models/adaptive/bloom_adapter.py',
            '/sentinel/models/adaptive/llama_adapter.py',
            '/sentinel/models/utils/optimized_attention.py',
            '/sentinel/models/utils/specialization_registry.py',
            '/sentinel/models/adaptive/unet_transformer.py',
            '/sentinel/models/adaptive/unet_transformer_optimized.py'
        ]
        
        # Add loaders
        loader_files = [f for f in os.listdir('models/loaders') if f.endswith('.py')]
        for f in loader_files:
            sources.append(f'/models/loaders/{f}')
            targets.append(f'/sentinel/models/loaders/{f}')
            
        # Add optimized files
        if os.path.exists('models/optimized'):
            optimized_files = [f for f in os.listdir('models/optimized') if f.endswith('.py')]
            for f in optimized_files:
                sources.append(f'/models/optimized/{f}')
                targets.append(f'/sentinel/models/optimized/{f}')
                
    elif module_filter == 'controller':
        sources = [
            '/controller/controller_ann.py',
            '/controller/controller_manager.py'
        ]
        targets = [
            '/sentinel/controller/controller_ann.py',
            '/sentinel/controller/controller_manager.py'
        ]
        
        # Add metrics and visualization files
        if os.path.exists('controller/metrics'):
            metrics_files = [f for f in os.listdir('controller/metrics') if f.endswith('.py')]
            for f in metrics_files:
                sources.append(f'/controller/metrics/{f}')
                targets.append(f'/sentinel/controller/metrics/{f}')
                
        if os.path.exists('controller/visualizations'):
            vis_files = [f for f in os.listdir('controller/visualizations') if f.endswith('.py')]
            for f in vis_files:
                sources.append(f'/controller/visualizations/{f}')
                targets.append(f'/sentinel/controller/visualizations/{f}')
                
    elif module_filter == 'pruning':
        sources = [
            '/utils/pruning/pruning_module.py',
            '/utils/pruning/strategies.py',
            '/utils/pruning/fine_tuner.py',
            '/utils/pruning/fine_tuner_improved.py',
            '/utils/pruning/fine_tuner_consolidated.py',
            '/utils/pruning/experiment.py',
            '/utils/pruning/results_manager.py',
            '/utils/pruning/benchmark.py',
            '/utils/pruning/visualization.py',
            '/utils/pruning/growth.py',
            '/utils/pruning/head_lr_manager.py'
        ]
        targets = [
            '/sentinel/pruning/pruning_module.py',
            '/sentinel/pruning/strategies/base.py',
            '/sentinel/pruning/fine_tuning/base.py',
            '/sentinel/pruning/fine_tuning/improved.py',
            '/sentinel/pruning/fine_tuning/consolidated.py',
            '/sentinel/pruning/experiment.py',
            '/sentinel/pruning/results_manager.py',
            '/sentinel/pruning/benchmark.py',
            '/sentinel/pruning/visualization.py',
            '/sentinel/pruning/growth.py',
            '/sentinel/pruning/head_lr_manager.py'
        ]
        
        # Add stability files
        if os.path.exists('utils/pruning/stability'):
            stability_files = [f for f in os.listdir('utils/pruning/stability') if f.endswith('.py')]
            for f in stability_files:
                sources.append(f'/utils/pruning/stability/{f}')
                targets.append(f'/sentinel/pruning/stability/{f}')
                
    elif module_filter == 'adaptive':
        sources = [
            '/utils/adaptive/adaptive_plasticity.py',
            '/utils/adaptive/__init__.py'
        ]
        targets = [
            '/sentinel/plasticity/adaptive/adaptive_plasticity.py',
            '/sentinel/plasticity/adaptive/__init__.py'
        ]
                
    elif module_filter == 'data':
        sources = [
            '/sentinel_data/__init__.py',
            '/sentinel_data/dataset_loader.py',
            '/sentinel_data/eval.py'
        ]
        targets = [
            '/sentinel/data/__init__.py',
            '/sentinel/data/loaders/dataset_loader.py',
            '/sentinel/data/eval.py'
        ]
        
        # Add custom data loaders
        if os.path.exists('custdata/loaders'):
            custom_files = [f for f in os.listdir('custdata/loaders') if f.endswith('.py')]
            for f in custom_files:
                sources.append(f'/custdata/loaders/{f}')
                targets.append(f'/sentinel/data/loaders/custom/{f}')
    
    # Process files to move
    for i, source in enumerate(sources):
        target = targets[i]
        
        # Remove leading slash for file operations
        source_path = source.lstrip('/')
        target_path = target.lstrip('/')
        
        # Skip if source doesn't exist
        if not os.path.exists(source_path):
            if verbose:
                logger.warning(f"Source file does not exist: {source_path}")
            continue
        
        # If source is a file, move it
        if os.path.isfile(source_path):
            result = move_file(source_path, target_path, dry_run, verbose)
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