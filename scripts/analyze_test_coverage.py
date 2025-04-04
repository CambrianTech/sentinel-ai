#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Coverage Analysis Tool

This script analyzes the test coverage of the codebase by:
1. Finding all Python modules in the repository
2. Finding all test files that test those modules
3. Identifying modules that lack test coverage
4. Generating a test coverage report
5. Recommending areas that need more testing

Usage:
    python scripts/analyze_test_coverage.py [--output-file FILENAME]
"""

import os
import re
import sys
import argparse
import json
from collections import defaultdict
from pathlib import Path

def find_python_modules(root_dir='.', exclude_dirs=None):
    """Find all Python modules in the repository."""
    if exclude_dirs is None:
        exclude_dirs = ['venv', '.venv', '.git', '__pycache__', 'tests']
    
    modules = []
    
    for root, dirs, files in os.walk(root_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith('.py'):
                # Skip test files
                if file.startswith('test_') or file.endswith('_test.py'):
                    continue
                
                # Skip __init__.py files
                if file == '__init__.py':
                    continue
                
                filepath = os.path.join(root, file)
                
                # Convert to module path
                module_path = os.path.splitext(filepath)[0].replace('/', '.')
                if module_path.startswith('./'):
                    module_path = module_path[2:]
                
                modules.append({
                    'file_path': filepath,
                    'module_path': module_path,
                    'name': os.path.splitext(file)[0]
                })
    
    return modules

def find_test_files(root_dir='.', exclude_dirs=None):
    """Find all test files in the repository."""
    if exclude_dirs is None:
        exclude_dirs = ['venv', '.venv', '.git', '__pycache__']
    
    test_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.startswith('test_') or file.endswith('_test.py'):
                filepath = os.path.join(root, file)
                
                test_files.append({
                    'file_path': filepath,
                    'name': os.path.splitext(file)[0]
                })
    
    return test_files

def analyze_test_file(test_file):
    """Analyze a test file to find which modules it tests."""
    with open(test_file['file_path'], 'r') as f:
        content = f.read()
    
    # Extract import statements
    import_pattern = r'(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))'
    imports = re.findall(import_pattern, content)
    
    # Extract imported modules
    modules = set()
    for from_import, direct_import in imports:
        if from_import:
            # Handle 'from x.y.z import a'
            parts = from_import.split('.')
            modules.add(parts[0])  # Add the top-level module
            
            # Add the full module path
            if len(parts) > 1:
                modules.add(from_import)
        
        if direct_import:
            # Handle 'import x.y.z'
            parts = direct_import.split('.')
            modules.add(parts[0])  # Add the top-level module
            
            # Add the full module path
            if len(parts) > 1:
                modules.add(direct_import)
    
    # Look for class and function tests
    # This regex looks for test_X or test_X_Y patterns
    test_pattern = r'(?:def|class)\s+test_(\w+)'
    test_targets = re.findall(test_pattern, content)
    
    return {
        'test_file': test_file,
        'imported_modules': list(modules),
        'test_targets': test_targets
    }

def find_module_test_coverage(modules, test_files):
    """Find test coverage for each module."""
    
    # Analyze all test files
    test_analyses = [analyze_test_file(test_file) for test_file in test_files]
    
    # Create a mapping of module to test files
    module_to_tests = defaultdict(list)
    for analysis in test_analyses:
        for module in analysis['imported_modules']:
            module_to_tests[module].append(analysis['test_file'])
    
    # Check coverage for each module
    coverage = []
    for module in modules:
        # Find test files that import this module
        tests = []
        module_name = module['name']
        module_path = module['module_path']
        
        # Check for exact module path match
        if module_path in module_to_tests:
            tests.extend(module_to_tests[module_path])
        
        # Check for module name match in test targets
        for analysis in test_analyses:
            for target in analysis['test_targets']:
                # Convert test target to potential module name
                target_parts = target.split('_')
                if module_name.lower() in [p.lower() for p in target_parts]:
                    tests.append(analysis['test_file'])
                    break
        
        # Remove duplicates
        unique_tests = []
        test_paths = set()
        for test in tests:
            if test['file_path'] not in test_paths:
                test_paths.add(test['file_path'])
                unique_tests.append(test)
        
        coverage.append({
            'module': module,
            'tests': unique_tests,
            'has_tests': len(unique_tests) > 0
        })
    
    return coverage

def classify_modules(modules):
    """Classify modules by their component/category."""
    classifications = defaultdict(list)
    
    for module in modules:
        path = module['file_path']
        
        # Classify based on path
        if path.startswith('models/'):
            classifications['models'].append(module)
        elif path.startswith('controller/'):
            classifications['controller'].append(module)
        elif path.startswith('utils/pruning/'):
            classifications['pruning'].append(module)
        elif path.startswith('utils/adaptive/'):
            classifications['plasticity'].append(module)
        elif path.startswith('sentinel_data/'):
            classifications['data'].append(module)
        elif path.startswith('utils/'):
            classifications['utils'].append(module)
        elif path.startswith('scripts/'):
            classifications['scripts'].append(module)
        else:
            classifications['other'].append(module)
    
    return classifications

def generate_coverage_report(coverage, output_file=None):
    """Generate a test coverage report."""
    # Group by category
    modules_by_category = classify_modules([c['module'] for c in coverage])
    
    # Create coverage statistics by category
    category_stats = {}
    for category, modules in modules_by_category.items():
        modules_with_tests = 0
        modules_without_tests = 0
        
        for module in modules:
            # Find coverage entry for this module
            for c in coverage:
                if c['module'] == module:
                    if c['has_tests']:
                        modules_with_tests += 1
                    else:
                        modules_without_tests += 1
                    break
        
        total_modules = modules_with_tests + modules_without_tests
        coverage_percent = 0 if total_modules == 0 else (modules_with_tests / total_modules) * 100
        
        category_stats[category] = {
            'total_modules': total_modules,
            'modules_with_tests': modules_with_tests,
            'modules_without_tests': modules_without_tests,
            'coverage_percent': coverage_percent
        }
    
    # Generate report
    report = {
        'total_modules': len(coverage),
        'modules_with_tests': sum(1 for c in coverage if c['has_tests']),
        'modules_without_tests': sum(1 for c in coverage if not c['has_tests']),
        'category_stats': category_stats,
        'untested_modules': [
            {
                'module': c['module']['module_path'],
                'file_path': c['module']['file_path']
            }
            for c in coverage if not c['has_tests']
        ]
    }
    
    # Calculate overall coverage
    report['overall_coverage_percent'] = 0 if len(coverage) == 0 else (report['modules_with_tests'] / len(coverage)) * 100
    
    # Output to file or stdout
    report_json = json.dumps(report, indent=2)
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_json)
    else:
        print(report_json)
    
    # Print human-readable summary
    print("\n=== TEST COVERAGE SUMMARY ===\n")
    print(f"Total modules: {report['total_modules']}")
    print(f"Modules with tests: {report['modules_with_tests']}")
    print(f"Modules without tests: {report['modules_without_tests']}")
    print(f"Overall coverage: {report['overall_coverage_percent']:.1f}%\n")
    
    print("Coverage by category:")
    categories = sorted(category_stats.keys())
    for category in categories:
        stats = category_stats[category]
        print(f"  {category}: {stats['coverage_percent']:.1f}% ({stats['modules_with_tests']}/{stats['total_modules']})")
    
    print("\nTop 10 modules needing tests:")
    for i, module in enumerate(report['untested_modules'][:10]):
        print(f"  {i+1}. {module['module']}")
    
    if len(report['untested_modules']) > 10:
        print(f"  ... and {len(report['untested_modules']) - 10} more")
    
    return report

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Coverage Analysis Tool")
    parser.add_argument('--output-file', type=str, help="Output file for the JSON report")
    
    args = parser.parse_args()
    
    # Find all Python modules
    modules = find_python_modules()
    print(f"Found {len(modules)} Python modules")
    
    # Find all test files
    test_files = find_test_files()
    print(f"Found {len(test_files)} test files")
    
    # Analyze test coverage
    coverage = find_module_test_coverage(modules, test_files)
    
    # Generate report
    generate_coverage_report(coverage, args.output_file)

if __name__ == "__main__":
    main()