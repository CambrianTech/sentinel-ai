#!/usr/bin/env python
"""
Fix Neural Plasticity Demo Dataset Imports

This script specifically fixes the dataset import conflict in the NeuralPlasticityDemo notebook
by implementing a proper resolution strategy.

Usage:
  python scripts/fix_neural_plasticity_datasets.py [notebook_path] [output_path]
  
Default paths:
- Input: notebooks/NeuralPlasticityDemo.ipynb
- Output: notebooks/NeuralPlasticityDemo_datasets_fixed.ipynb

Requirements:
  - nbformat: Install with `pip install nbformat`
"""

import os
import sys
import re

# Check for required dependencies
try:
    import nbformat
    from nbformat.v4 import new_code_cell
except ImportError:
    print("Error: This script requires nbformat.")
    print("Please install it with: pip install nbformat")
    sys.exit(1)

def fix_dataset_imports(notebook_path, output_path):
    """
    Fix dataset import conflicts in the Neural Plasticity Demo notebook.
    
    Args:
        notebook_path: Path to the original notebook
        output_path: Path to save the fixed notebook
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Fixing dataset imports in Neural Plasticity Demo...")
    print(f"Input: {notebook_path}")
    print(f"Output: {output_path}")
    
    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        print(f"Successfully read notebook: {len(notebook.cells)} cells")
        
        # Add dataset fix cell at the beginning
        fix_dataset_cell = new_code_cell("""# Fix dataset import conflicts
import sys
import types

# Check if datasets is already imported
if 'datasets' not in sys.modules:
    print("Creating datasets module to prevent import conflicts...")
    # Create mock module
    mock_datasets = types.ModuleType('datasets')
    mock_datasets.__path__ = []
    
    # Add required attributes
    mock_datasets.ArrowBasedBuilder = type('ArrowBasedBuilder', (), {})
    mock_datasets.GeneratorBasedBuilder = type('GeneratorBasedBuilder', (), {})
    mock_datasets.Value = lambda *args, **kwargs: None
    mock_datasets.Features = lambda *args, **kwargs: {}
    
    # Install the mock module
    sys.modules['datasets'] = mock_datasets
    
    # Try to import the real load_dataset function
    try:
        from datasets.load import load_dataset
        mock_datasets.load_dataset = load_dataset
        print("Successfully added load_dataset to datasets module")
    except ImportError as e:
        print(f"Failed to import load_dataset: {e}")
        
        # Try importing from our custom module
        try:
            from sdata import load_dataset
            mock_datasets.load_dataset = load_dataset
            print("Using sdata.load_dataset as fallback")
        except ImportError:
            try:
                from sentinel_data import load_dataset
                mock_datasets.load_dataset = load_dataset
                print("Using sentinel_data.load_dataset as fallback")
            except ImportError:
                print("WARNING: Could not import any dataset loading function")
else:
    print("datasets module already imported")""")
        
        # Insert dataset fix cell at the beginning, after any %matplotlib magic 
        # and import cells but before model loading
        import_cell_idx = None
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code' and 'import torch' in cell.source:
                import_cell_idx = i
                break
        
        if import_cell_idx is not None:
            notebook.cells.insert(import_cell_idx + 1, fix_dataset_cell)
            print(f"Added dataset fix cell after cell {import_cell_idx}")
        else:
            # If no import cell found, add at the beginning
            notebook.cells.insert(1, fix_dataset_cell)  # Insert after title
            print("Added dataset fix cell at the beginning")
        
        # Update all instances of 'from datasets import load_dataset'
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code' and 'from datasets import load_dataset' in cell.source:
                # Replace with our fixed version
                cell.source = cell.source.replace(
                    'from datasets import load_dataset', 
                    '# Dataset loading is already handled by the fix_dataset_imports cell\ntry:\n    load_dataset = sys.modules["datasets"].load_dataset\nexcept (KeyError, AttributeError):\n    from datasets import load_dataset'
                )
                print(f"Updated dataset import in cell {i}")
        
        # Write the fixed notebook
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(notebook, f)
        
        print(f"Successfully saved fixed notebook to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error fixing notebook: {e}")
        return False

if __name__ == "__main__":
    # Default paths
    default_input = os.path.join("notebooks", "NeuralPlasticityDemo.ipynb")
    default_output = os.path.join("notebooks", "NeuralPlasticityDemo_datasets_fixed.ipynb")
    
    # Get paths from command line arguments if provided
    input_path = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_path = sys.argv[2] if len(sys.argv) > 2 else default_output
    
    # Run the fix
    success = fix_dataset_imports(input_path, output_path)
    
    if success:
        print("✅ Successfully fixed dataset imports in Neural Plasticity Demo notebook")
        print(f"Original notebook: {input_path}")
        print(f"Fixed notebook: {output_path}")
    else:
        print("❌ Failed to fix dataset imports in Neural Plasticity Demo notebook")
        
    sys.exit(0 if success else 1)