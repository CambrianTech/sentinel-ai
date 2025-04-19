#!/usr/bin/env python

"""
Disk space cleanup utility for Sentinel AI

This script helps manage disk space by cleaning up old/temp files
and providing information about space usage in the project.
"""

import os
import glob
import shutil
import argparse
from datetime import datetime, timedelta

def format_size(size_bytes):
    """Format size in bytes to human-readable string"""
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def scan_large_files(directory, min_size_mb=10, exclude=None):
    """Scan for large files that might be candidates for cleanup"""
    if exclude is None:
        exclude = ['.git', 'venv', '.venv', '__pycache__']
    
    large_files = []
    min_size_bytes = min_size_mb * 1024 * 1024
    
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude]
        
        for file in files:
            file_path = os.path.join(root, file)
            try:
                size = os.path.getsize(file_path)
                if size >= min_size_bytes:
                    large_files.append((file_path, size))
            except (FileNotFoundError, PermissionError):
                continue
    
    # Sort by size (largest first)
    large_files.sort(key=lambda x: x[1], reverse=True)
    return large_files

def clean_temp_files(directory, extensions=None, older_than_days=7, dry_run=True):
    """Clean temporary files with specified extensions that are older than X days"""
    if extensions is None:
        extensions = ['.tmp', '.log', '.bak', '.swp', '.pyc', '.pyo']
    
    cutoff_date = datetime.now() - timedelta(days=older_than_days)
    cleaned_files = []
    cleaned_size = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                try:
                    # Check file modification time
                    mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if mod_time < cutoff_date:
                        size = os.path.getsize(file_path)
                        cleaned_size += size
                        cleaned_files.append((file_path, size))
                        
                        if not dry_run:
                            os.remove(file_path)
                except (FileNotFoundError, PermissionError):
                    continue
    
    return cleaned_files, cleaned_size

def clean_model_cache(cache_dir='~/.cache/huggingface', dry_run=True):
    """Clean Hugging Face model cache"""
    cache_dir = os.path.expanduser(cache_dir)
    if not os.path.exists(cache_dir):
        return [], 0
    
    # Get list of cached models
    model_dirs = glob.glob(os.path.join(cache_dir, 'hub', 'models--*'))
    
    cleaned_dirs = []
    cleaned_size = 0
    
    for model_dir in model_dirs:
        try:
            # Get size
            size = get_dir_size(model_dir)
            model_name = os.path.basename(model_dir).replace('models--', '').replace('--', '/')
            cleaned_dirs.append((model_dir, size, model_name))
            cleaned_size += size
            
            if not dry_run:
                shutil.rmtree(model_dir)
        except (FileNotFoundError, PermissionError):
            continue
    
    return cleaned_dirs, cleaned_size

def get_dir_size(path):
    """Get the total size of a directory"""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):  # Skip symbolic links
                try:
                    total_size += os.path.getsize(fp)
                except (FileNotFoundError, PermissionError):
                    continue
    return total_size

def main():
    parser = argparse.ArgumentParser(description='Disk space management utility')
    parser.add_argument('--scan', action='store_true', help='Scan for large files')
    parser.add_argument('--clean-temp', action='store_true', help='Clean temporary files')
    parser.add_argument('--clean-cache', action='store_true', help='Clean model cache')
    parser.add_argument('--min-size', type=int, default=10, help='Minimum file size in MB for scanning')
    parser.add_argument('--older-than', type=int, default=7, help='Clean files older than X days')
    parser.add_argument('--no-dry-run', action='store_true', help='Actually delete files (default is dry run)')
    parser.add_argument('--directory', type=str, default='.', help='Target directory')
    
    args = parser.parse_args()
    dry_run = not args.no_dry_run
    
    if dry_run:
        print("DRY RUN MODE - No files will be deleted")
        print("Use --no-dry-run to actually delete files")
        print()
    
    if not (args.scan or args.clean_temp or args.clean_cache):
        # If no action specified, default to scanning
        args.scan = True
    
    # Get total disk usage of directory
    total_size = get_dir_size(args.directory)
    print(f"Total project size: {format_size(total_size)}")
    print()
    
    if args.scan:
        print(f"Scanning for files larger than {args.min_size} MB...")
        large_files = scan_large_files(args.directory, min_size_mb=args.min_size)
        
        if large_files:
            print(f"Found {len(large_files)} large files:")
            for file_path, size in large_files:
                print(f"{format_size(size):>10} - {file_path}")
        else:
            print("No large files found.")
        print()
    
    if args.clean_temp:
        print(f"Finding temporary files older than {args.older_than} days...")
        cleaned_files, cleaned_size = clean_temp_files(
            args.directory, 
            older_than_days=args.older_than,
            dry_run=dry_run
        )
        
        if cleaned_files:
            action = "Would clean" if dry_run else "Cleaned"
            print(f"{action} {len(cleaned_files)} files, freeing {format_size(cleaned_size)}:")
            for file_path, size in cleaned_files:
                print(f"{format_size(size):>10} - {file_path}")
        else:
            print("No temporary files to clean.")
        print()
    
    if args.clean_cache:
        print("Finding cached models...")
        cleaned_dirs, cleaned_size = clean_model_cache(dry_run=dry_run)
        
        if cleaned_dirs:
            action = "Would clean" if dry_run else "Cleaned"
            print(f"{action} {len(cleaned_dirs)} cached models, freeing {format_size(cleaned_size)}:")
            for dir_path, size, model_name in cleaned_dirs:
                print(f"{format_size(size):>10} - {model_name}")
        else:
            print("No cached models to clean.")
        print()
    
    if dry_run:
        print("This was a dry run. Use --no-dry-run to actually delete files.")

if __name__ == '__main__':
    main()