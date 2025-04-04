#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Output Directory Analysis Tool

This script analyzes the various output directories in the repository to:
1. Find all output directories and files
2. Trace which scripts generate these outputs
3. Identify output patterns and formats
4. Recommend a standardized output structure
5. Generate a migration plan for output files

Usage:
    python scripts/analyze_output_directories.py [--output-file FILENAME]
"""

import os
import re
import sys
import json
import argparse
from collections import defaultdict
from pathlib import Path

# Output directories to analyze
OUTPUT_DIRS = [
    'output',
    'profiling_results',
    'pruning_results',
    'validation_results',
    'demo_results',
    'optimization_results'
]

def find_output_files(root_dir='.'):
    """Find all output files in the specified directories."""
    output_files = []
    
    for output_dir in OUTPUT_DIRS:
        dir_path = os.path.join(root_dir, output_dir)
        if not os.path.exists(dir_path):
            continue
        
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, root_dir)
                
                output_files.append({
                    'file_path': filepath,
                    'rel_path': rel_path,
                    'directory': output_dir,
                    'extension': os.path.splitext(file)[1].lower(),
                    'filename': file
                })
    
    return output_files

def find_output_refs_in_file(file_path, output_dirs):
    """Find references to output directories in a source file."""
    # Skip if the file doesn't exist
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return []
    
    refs = []
    with open(file_path, 'r', errors='ignore') as f:
        try:
            content = f.read()
            
            # Find output directory references
            for output_dir in output_dirs:
                # Look for paths with the output directory
                pattern = r'[\'"](' + re.escape(output_dir) + r'[\\/][^\'"]*)[\'"]'
                matches = re.findall(pattern, content)
                
                for match in matches:
                    refs.append({
                        'dir': output_dir,
                        'path': match,
                        'source_file': file_path
                    })
                
                # Also look for os.path.join with the output directory
                pattern = r'os\.path\.join\([\'"]' + re.escape(output_dir) + r'[\'"]'
                if re.search(pattern, content):
                    refs.append({
                        'dir': output_dir,
                        'path': f"{output_dir}/...",
                        'source_file': file_path,
                        'is_join': True
                    })
        except UnicodeDecodeError:
            # Skip binary files
            pass
    
    return refs

def find_output_generators(root_dir='.'):
    """Find scripts that generate output files."""
    output_generators = []
    
    # Find all Python files
    for root, dirs, files in os.walk(root_dir):
        # Skip venv, .git, etc.
        if any(excluded in root for excluded in ['venv', '.venv', '.git', '__pycache__']):
            continue
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                
                # Find references to output directories
                refs = find_output_refs_in_file(filepath, OUTPUT_DIRS)
                
                if refs:
                    output_generators.append({
                        'file_path': filepath,
                        'output_refs': refs
                    })
    
    return output_generators

def extract_output_patterns(output_files):
    """Extract patterns from output file paths."""
    patterns = defaultdict(int)
    
    for file in output_files:
        path_parts = file['rel_path'].split('/')
        
        # Skip the first part (output directory name)
        if len(path_parts) > 1:
            pattern = '/'.join([p if not re.search(r'\d{8}', p) else '*DATE*' for p in path_parts[1:2]])
            patterns[pattern] += 1
    
    return dict(patterns)

def analyze_output_formats(output_files):
    """Analyze output file formats/extensions."""
    formats = defaultdict(int)
    
    for file in output_files:
        formats[file['extension']] += 1
    
    return dict(formats)

def map_generators_to_outputs(output_generators, output_files):
    """Map output generators to specific output files/directories."""
    mapping = defaultdict(list)
    
    # Create a lookup of output directories and paths
    output_paths = defaultdict(list)
    for file in output_files:
        dir_name = file['directory']
        path_parts = file['rel_path'].split('/')
        
        # Add all parent paths
        for i in range(1, len(path_parts) + 1):
            partial_path = '/'.join(path_parts[:i])
            output_paths[partial_path].append(file)
    
    # Map generators to outputs
    for generator in output_generators:
        file_path = generator['file_path']
        
        for ref in generator['output_refs']:
            output_dir = ref['dir']
            output_path = ref['path']
            
            # Find matching output files
            matches = []
            
            if ref.get('is_join', False):
                # For os.path.join references, we need to match by directory
                for path, files in output_paths.items():
                    if path.startswith(output_dir):
                        matches.extend(files)
            else:
                # For direct path references
                for path, files in output_paths.items():
                    if path.startswith(output_path) or output_path.startswith(path):
                        matches.extend(files)
            
            # Remove duplicates
            unique_matches = []
            seen_paths = set()
            for file in matches:
                if file['rel_path'] not in seen_paths:
                    seen_paths.add(file['rel_path'])
                    unique_matches.append(file)
            
            mapping[file_path].extend(unique_matches)
    
    return mapping

def generate_standardized_structure(output_files, generator_mapping):
    """Generate a standardized output directory structure."""
    # Categorize outputs
    categories = {
        'pruning': [],
        'profiling': [],
        'validation': [],
        'plasticity': [],
        'benchmark': [],
        'demo': [],
        'other': []
    }
    
    for file in output_files:
        path = file['rel_path']
        
        # Categorize based on path or directory
        if 'pruning' in path:
            categories['pruning'].append(file)
        elif 'profiling' in path:
            categories['profiling'].append(file)
        elif 'validation' in path:
            categories['validation'].append(file)
        elif 'plasticity' in path or 'adaptive' in path:
            categories['plasticity'].append(file)
        elif 'benchmark' in path:
            categories['benchmark'].append(file)
        elif 'demo' in path:
            categories['demo'].append(file)
        else:
            categories['other'].append(file)
    
    # Generate proposed structure
    proposed_structure = {
        "experiments/results/pruning/": categories['pruning'],
        "experiments/results/profiling/": categories['profiling'],
        "experiments/results/validation/": categories['validation'],
        "experiments/results/plasticity/": categories['plasticity'],
        "experiments/results/benchmark/": categories['benchmark'],
        "experiments/results/demo/": categories['demo'],
        "experiments/results/other/": categories['other']
    }
    
    return proposed_structure

def create_migration_plan(output_files, proposed_structure):
    """Create a migration plan for output files."""
    migration_plan = []
    
    # Flatten the proposed structure
    file_to_target = {}
    for target_dir, files in proposed_structure.items():
        for file in files:
            # Extract the filename and subdirectories
            path_parts = file['rel_path'].split('/')
            source_dir = path_parts[0]
            
            # Keep subdirectories after the first one
            if len(path_parts) > 2:
                subdirs = '/'.join(path_parts[1:-1])
                target_path = f"{target_dir}{subdirs}/{path_parts[-1]}"
            else:
                target_path = f"{target_dir}{path_parts[-1]}"
            
            file_to_target[file['rel_path']] = target_path
    
    # Create migration entries
    for file in output_files:
        source_path = file['rel_path']
        if source_path in file_to_target:
            target_path = file_to_target[source_path]
            
            migration_plan.append({
                "source": source_path,
                "target": target_path
            })
    
    return migration_plan

def generate_output_report(output_files, output_generators, generator_mapping, proposed_structure, migration_plan, output_file=None):
    """Generate a comprehensive output directory analysis report."""
    # Calculate statistics
    num_output_files = len(output_files)
    num_generators = len(output_generators)
    
    extension_stats = analyze_output_formats(output_files)
    pattern_stats = extract_output_patterns(output_files)
    
    # Create report
    report = {
        "statistics": {
            "total_output_files": num_output_files,
            "total_generators": num_generators,
            "output_formats": extension_stats,
            "output_patterns": pattern_stats
        },
        "output_files": output_files,
        "output_generators": output_generators,
        "proposed_structure": {
            target: [f['rel_path'] for f in files] 
            for target, files in proposed_structure.items()
        },
        "migration_plan": migration_plan
    }
    
    # Output report
    report_json = json.dumps(report, indent=2)
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_json)
    
    # Print human-readable summary
    print("\n=== OUTPUT DIRECTORY ANALYSIS ===\n")
    print(f"Total output files: {num_output_files}")
    print(f"Total generator scripts: {num_generators}")
    
    print("\nOutput formats:")
    for ext, count in sorted(extension_stats.items(), key=lambda x: -x[1]):
        print(f"  {ext or 'no extension'}: {count}")
    
    print("\nCommon output patterns:")
    for pattern, count in sorted(pattern_stats.items(), key=lambda x: -x[1])[:10]:
        print(f"  {pattern}: {count}")
    
    print("\nProposed directory structure:")
    for target, files in proposed_structure.items():
        print(f"  {target}: {len(files)} files")
    
    print("\nTop generators by output count:")
    generator_counts = {}
    for gen_path, files in generator_mapping.items():
        generator_counts[gen_path] = len(files)
    
    for gen_path, count in sorted(generator_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {gen_path}: {count} files")
    
    return report

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Output Directory Analysis Tool")
    parser.add_argument('--output-file', type=str, help="Output file for the JSON report")
    
    args = parser.parse_args()
    
    # Find all output files
    output_files = find_output_files()
    print(f"Found {len(output_files)} output files")
    
    # Find all output generators
    output_generators = find_output_generators()
    print(f"Found {len(output_generators)} output generators")
    
    # Map generators to outputs
    generator_mapping = map_generators_to_outputs(output_generators, output_files)
    
    # Generate standardized structure
    proposed_structure = generate_standardized_structure(output_files, generator_mapping)
    
    # Create migration plan
    migration_plan = create_migration_plan(output_files, proposed_structure)
    
    # Generate report
    generate_output_report(
        output_files, 
        output_generators, 
        generator_mapping, 
        proposed_structure, 
        migration_plan,
        args.output_file
    )

if __name__ == "__main__":
    main()