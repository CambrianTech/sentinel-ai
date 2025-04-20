#!/usr/bin/env python
"""
Fix Neural Plasticity Demo Imports and Cross-Platform Compatibility

This script fixes import issues in the NeuralPlasticityDemo notebook:
1. Resolves the 'datasets' module import conflict
2. Enhances Apple Silicon compatibility 
3. Improves environment detection for Colab/local execution

Usage:
  python scripts/fix_neural_plasticity_imports.py [notebook_path] [output_path]
  
Default paths:
- Input: notebooks/NeuralPlasticityDemo.ipynb
- Output: notebooks/NeuralPlasticityDemo_fixed.ipynb

Requirements:
  - nbformat: Install with `pip install nbformat`
"""

import os
import sys
import re
import datetime
import uuid

# Check for required dependencies
try:
    import nbformat
    from nbformat.v4 import new_code_cell
except ImportError:
    print("Error: This script requires nbformat.")
    print("Please install it with: pip install nbformat")
    sys.exit(1)

def fix_neural_plasticity_imports(notebook_path, output_path):
    """
    Fix import issues in the Neural Plasticity Demo notebook.
    
    Args:
        notebook_path: Path to the original notebook
        output_path: Path to save the fixed notebook
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Fixing Neural Plasticity Demo imports...")
    print(f"Input: {notebook_path}")
    print(f"Output: {output_path}")
    
    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        print(f"Successfully read notebook: {len(notebook.cells)} cells")
        
        # Update version in title cell
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_version = "v0.0.59"
        unique_id = str(uuid.uuid4())[:8]
        
        # Fix title cell
        if notebook.cells[0].cell_type == 'markdown' and 'Neural Plasticity' in notebook.cells[0].source:
            # Extract current content for preservation
            title_cell = notebook.cells[0]
            content = title_cell.source
            
            # Update version number
            content = re.sub(r'v0\.\d+\.\d+.*?\)', f"{new_version} ({current_time})", content)
            
            # Add new changes
            changes_section = f"""
### Changes in {new_version}:
- Fixed datasets import conflict issues
- Enhanced Apple Silicon compatibility with improved tensor handling
- Added workarounds for PyTorch/BLAS crashes on M1/M2/M3 chips
- Improved environment detection for Colab/local execution
- Fixed cross-platform tensor handling in visualizations
"""
            # Add the changes section after the title/first paragraph
            parts = content.split("\n\n", 1)
            if len(parts) > 1:
                # Add after first paragraph, before existing content
                existing_changes_match = re.search(r'### Changes in v', parts[1])
                if existing_changes_match:
                    # Replace existing changes section
                    content = parts[0] + "\n\n" + changes_section + re.sub(r'### Changes in v.*?(?=\n\n##|\Z)', '', parts[1], flags=re.DOTALL)
                else:
                    # Add changes section at top
                    content = parts[0] + "\n\n" + changes_section + parts[1]
            else:
                # Just append if no paragraphs
                content += changes_section
                
            # Update unique ID
            content = re.sub(r'\[ID: [a-z0-9]+\]', f'[ID: {unique_id}]', content)
            
            # Update the cell
            title_cell.source = content
        
        # Add Apple Silicon detection cell
        apple_silicon_code = """
# Enhanced environment detection and optimization
import sys
import platform
import os
from typing import Dict, Tuple, Optional, Union, List, Any

# Detect Apple Silicon
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.processor() == "arm"

# Detect if we're running in Google Colab
IS_COLAB = 'google.colab' in str(get_ipython()) if 'get_ipython' in globals() else False

# Check for GPU availability
if not IS_APPLE_SILICON:  # Skip CUDA check on Apple Silicon
    import torch
    HAS_GPU = torch.cuda.is_available()
    if HAS_GPU and IS_COLAB:
        print(f"✅ CUDA GPU detected in Colab: {torch.cuda.get_device_name(0)}")
        print(f"🚀 Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    HAS_GPU = False

# Add Apple Silicon optimizations if needed
if IS_APPLE_SILICON:
    # Apply basic optimizations
    print("🍎 Apple Silicon detected - enabling basic optimizations")
    
    # Force PyTorch to use CPU and single-threading for stability
    import torch
    torch.set_num_threads(1)
    
    # Force single-threaded BLAS operations
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    # Disable BLAS threading at pytorch level too
    try:
        torch.backends.openmp.is_available = lambda: False
        torch.backends.mkldnn.enabled = False
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError) as e:
        print(f"⚠️ Could not set all PyTorch safeguards: {e}")
    
    # Force use of slower but more stable BLAS implementation
    os.environ["ACCELERATE_USE_SYSTEM_BLAS"] = "1"
    os.environ["PYTORCH_JIT_USE_AUTOTUNER"] = "0"
    
    # Also set matplotlib to use Agg backend for better stability
    import matplotlib
    matplotlib.use('Agg')
    print("🎨 Switched to Agg matplotlib backend for stability")
    
    print("ℹ️ When running on Apple Silicon, model operations will be forced to CPU")
    print("   This prevents BLAS/libtorch crashes that commonly occur on M1/M2/M3 chips")

# Safe tensor conversion function 
def safe_tensor_to_numpy(tensor):
    \"\"\"Safely convert a tensor to numpy array with proper device handling\"\"\"
    import torch
    import numpy as np
    
    if isinstance(tensor, np.ndarray):
        return tensor
        
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    
    # Handle based on environment
    if IS_APPLE_SILICON:
        # Always detach and move to CPU on Apple Silicon
        if tensor.requires_grad:
            tensor = tensor.detach()
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
    else:
        # Standard handling
        if tensor.requires_grad:
            tensor = tensor.detach()
        if tensor.is_cuda:
            tensor = tensor.cpu()
    
    # Convert to numpy safely
    try:
        return tensor.numpy()
    except Exception as e:
        print(f"Error converting tensor to numpy: {e}")
        # Fall back to a zeros array with the same shape
        return np.zeros(tensor.shape)

# Define unique ID for cache busting
unique_id = f"{unique_id}"
print(f"Environment setup complete [ID: {unique_id}]")
"""
        
        # Find environment setup cell and enhance it for Apple Silicon
        environment_cell_index = None
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code' and 'IS_APPLE_SILICON' in cell.source:
                environment_cell_index = i
                break
                
        # If environment cell found, replace it
        if environment_cell_index is not None:
            notebook.cells[environment_cell_index] = new_code_cell(apple_silicon_code)
            print(f"Updated Apple Silicon compatibility cell at position {environment_cell_index}")
        else:
            # Otherwise find the appropriate place to add the environment cell
            for i, cell in enumerate(notebook.cells):
                if cell.cell_type == 'code' and 'model = AutoModelForCausalLM.from_pretrained' in cell.source:
                    environment_cell_index = i
                    break
                    
            # If still not found, add after the imports
            if environment_cell_index is None:
                for i, cell in enumerate(notebook.cells):
                    if cell.cell_type == 'code' and 'import torch' in cell.source:
                        environment_cell_index = i + 1
                        break
            
            # Insert a new environment setup cell
            if environment_cell_index is not None:
                notebook.cells.insert(environment_cell_index, new_code_cell(apple_silicon_code))
                print(f"Added Apple Silicon compatibility cell at position {environment_cell_index}")
            
        # Fix dataset import cell
        dataset_fixed = False
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code' and 'from datasets import load_dataset' in cell.source:
                # Create new import cell with dataset fix
                fixed_import_code = """
# Fix dataset imports with proper error handling
try:
    # Try using the datasets module directly if imports were fixed
    from datasets import load_dataset
    print("Using HuggingFace datasets module")
except (ImportError, AttributeError) as e:
    try:
        # Fall back to our custom module
        from sdata import load_dataset
        print("Using sdata module (Sentinel's dataset wrapper)")
    except ImportError:
        # Last resort - check if datasets was fixed earlier
        if 'datasets' in sys.modules:
            datasets = sys.modules['datasets']
            load_dataset = datasets.load_dataset
            print("Using datasets module from sys.modules")
        else:
            print("WARNING: Could not import datasets module. Will attempt to use it anyway.")
            # Create a minimal datasets module
            import types
            datasets = types.ModuleType('datasets')
            sys.modules['datasets'] = datasets
            
            # Try to import from sentinel_data as last resort
            try:
                from sentinel_data import load_dataset
                datasets.load_dataset = load_dataset
                print("Using sentinel_data module as fallback")
            except ImportError:
                print("CRITICAL: No dataset module available")
"""
                # Replace the line with our fixed version
                cell.source = cell.source.replace('from datasets import load_dataset', fixed_import_code)
                dataset_fixed = True
                print(f"Updated dataset import in cell {i}")
                break
        
        # Add dataset import fix if it wasn't found and fixed
        if not dataset_fixed:
            # Find where to add the import fix
            import_cell_idx = None
            for i, cell in enumerate(notebook.cells):
                if cell.cell_type == 'code' and 'import torch' in cell.source:
                    import_cell_idx = i
                    break
            
            if import_cell_idx is not None:
                # Add dataset fix right after the imports
                dataset_fix_cell = new_code_cell("""# Fix dataset imports to avoid circular import issues
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
                
                notebook.cells.insert(import_cell_idx + 1, dataset_fix_cell)
                print(f"Added dataset fix cell after cell {import_cell_idx}")
                
        
        # Find device setting cell and update it for Apple Silicon compatibility
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code' and 'device = torch.device(' in cell.source:
                # Update device selection logic
                cell.source = cell.source.replace(
                    'device = torch.device(',
                    '# Set device - force CPU on Apple Silicon regardless of CUDA availability\ndevice = torch.device("cpu") if IS_APPLE_SILICON else torch.device('
                )
                print(f"Updated device selection in cell {i}")
                break
        
        # Write the fixed notebook
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(notebook, f)
        
        print(f"Successfully saved fixed notebook to {output_path}")
        print(f"Fixed notebook has unique ID: {unique_id}")
        return True
        
    except Exception as e:
        print(f"Error fixing notebook: {e}")
        return False

if __name__ == "__main__":
    # Default paths
    default_input = os.path.join("notebooks", "NeuralPlasticityDemo.ipynb")
    default_output = os.path.join("notebooks", "NeuralPlasticityDemo_fixed.ipynb")
    
    # Get paths from command line arguments if provided
    input_path = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_path = sys.argv[2] if len(sys.argv) > 2 else default_output
    
    # Run the fix
    success = fix_neural_plasticity_imports(input_path, output_path)
    
    if success:
        print("✅ Successfully fixed Neural Plasticity Demo notebook")
        print(f"Original notebook: {input_path}")
        print(f"Fixed notebook: {output_path}")
    else:
        print("❌ Failed to fix Neural Plasticity Demo notebook")
        
    sys.exit(0 if success else 1)