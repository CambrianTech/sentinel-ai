#!/usr/bin/env python
"""
Fix dataset imports to avoid circular import issues.

This script pre-imports the datasets module and key functions to avoid
circular import issues when used in other modules. Run this before
importing any code that uses datasets.
"""

import os
import sys
import importlib
import types

def fix_dataset_imports():
    """
    Pre-import datasets module to avoid circular imports.
    
    Returns:
        True if successful, False otherwise
    """
    print("Fixing dataset imports...")
    
    # Check if datasets is already imported
    if 'datasets' in sys.modules:
        print("datasets module already imported")
        return True
        
    try:
        # First create and install a minimal mock module to prevent circular imports
        print("Creating initial mock datasets module...")
        mock_datasets = types.ModuleType('datasets')
        mock_datasets.__path__ = []
            
        # Add required attributes to break circular imports
        mock_datasets.ArrowBasedBuilder = type('ArrowBasedBuilder', (), {})
        mock_datasets.GeneratorBasedBuilder = type('GeneratorBasedBuilder', (), {})
        mock_datasets.Value = lambda *args, **kwargs: None
        mock_datasets.Features = lambda *args, **kwargs: {}
            
        # Install the mock module
        sys.modules['datasets'] = mock_datasets
        
        # Now try to safely import the real module
        try:
            # Now try importing the real dataset loader directly
            from datasets.load import load_dataset
            
            # Add the real load_dataset to our mock
            mock_datasets.load_dataset = load_dataset
            print("Successfully added load_dataset to datasets module")
            return True
        except ImportError as e:
            print(f"Failed to import load_dataset: {e}")
            
            # Try an alternative approach
            try:
                print("Trying alternative import approach...")
                import importlib.util
                
                # Check if the module exists in the environment
                spec = importlib.util.find_spec('datasets')
                if spec is None:
                    print("datasets package not found in environment")
                    return False
                
                # Try to load just the load.py module
                load_spec = importlib.util.find_spec('datasets.load')
                if load_spec:
                    load_module = importlib.util.module_from_spec(load_spec)
                    load_spec.loader.exec_module(load_module)
                    
                    # Add load_dataset to our mock
                    if hasattr(load_module, 'load_dataset'):
                        mock_datasets.load_dataset = load_module.load_dataset
                        print("Successfully added load_dataset via spec loader")
                        return True
                    else:
                        print("load_dataset not found in datasets.load module")
                
                return False
            except Exception as alt_e:
                print(f"Alternative import failed: {alt_e}")
                return False
                
    except Exception as e:
        print(f"Failed to fix dataset imports: {e}")
        return False
        
if __name__ == "__main__":
    success = fix_dataset_imports()
    print(f"Dataset import fix: {'SUCCESS' if success else 'FAILED'}")
    
    # If successful, show what is now available
    if success and 'datasets' in sys.modules:
        datasets = sys.modules['datasets']
        print("\nAvailable attributes in datasets module:")
        print(f"- load_dataset: {'✓' if hasattr(datasets, 'load_dataset') else '✗'}")
        print(f"- ArrowBasedBuilder: {'✓' if hasattr(datasets, 'ArrowBasedBuilder') else '✗'}")
        print(f"- GeneratorBasedBuilder: {'✓' if hasattr(datasets, 'GeneratorBasedBuilder') else '✗'}")
        
        # Try using load_dataset
        try:
            print("\nTrying to load a small dataset...")
            ds = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="validation[:10]")
            print(f"Successfully loaded dataset with {len(ds)} samples")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            
    sys.exit(0 if success else 1)