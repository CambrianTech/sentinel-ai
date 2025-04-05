#!/usr/bin/env python
"""
Update imports from sdata to sdata in the project's files.

This script scans Python files in the project and updates imports from sdata to sdata.
"""

import os
import sys
import re

# Files to exclude from processing
EXCLUDE_DIRS = [
    '.git',
    '.venv',
    '__pycache__',
    'venv',
    'node_modules',
]

def should_process(path):
    """Check if file should be processed"""
    # Skip excluded directories
    for exclude_dir in EXCLUDE_DIRS:
        if exclude_dir in path:
            return False
    
    # Only process Python files
    if not path.endswith('.py'):
        return False
    
    return True

def update_imports(content):
    """Update imports from sdata to sdata"""
    # Update import statements
    content = re.sub(
        r'from\s+sentinel_data(\s+|\.)([^\s]+)',
        r'from sdata\1\2',
        content
    )
    
    # Update import statements (without from)
    content = re.sub(
        r'import\s+sentinel_data(\s+|\.)([^\s]+)',
        r'import sdata\1\2',
        content
    )
    
    # Update simple import
    content = re.sub(
        r'import\s+sentinel_data',
        r'import sdata',
        content
    )
    
    # Update variable references
    content = re.sub(
        r'sentinel_data\.([a-zA-Z0-9_]+)',
        r'sdata.\1',
        content
    )
    
    return content

def process_file(file_path):
    """Process a single file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if file contains sentinel_data
        if 'sentinel_data' not in content:
            return False
        
        # Update imports
        new_content = update_imports(content)
        
        # Write back to file if changed
        if new_content != content:
            with open(file_path, 'w') as f:
                f.write(new_content)
            return True
        
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function"""
    count = 0
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for root, dirs, files in os.walk(base_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            file_path = os.path.join(root, file)
            
            if should_process(file_path):
                if process_file(file_path):
                    print(f"Updated: {file_path}")
                    count += 1
    
    print(f"Updated {count} files.")

if __name__ == "__main__":
    main()