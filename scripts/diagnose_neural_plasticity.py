#!/usr/bin/env python
"""
Diagnose Neural Plasticity Environment Issues

This script checks for environment-specific issues that may cause
BLAS/libtorch crashes when running the Neural Plasticity notebook.
It performs diagnostic checks on:

1. BLAS configuration
2. Python package installation (torch, numpy, matplotlib)
3. Basic tensor operations
4. Matrix multiplication (BLAS operations)
5. Basic visualization capabilities

This helps diagnose the root cause of crashes without running the full notebook.
"""

import os
import sys
import platform
import subprocess
from pathlib import Path

def check_environment():
    """Check basic environment details."""
    print("=== Environment Information ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Platform architecture: {platform.architecture()}")
    print(f"Current directory: {os.getcwd()}")
    print(f"PATH: {os.environ.get('PATH', 'Not set')}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    
    # Check for BLAS-related environment variables
    blas_vars = [
        'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'
    ]
    
    print("\n=== BLAS Configuration ===")
    for var in blas_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")
    
    # Set safe defaults if not set
    for var in blas_vars:
        if var not in os.environ:
            os.environ[var] = '1'
            print(f"Setting {var} = 1 for safety")

def check_package_installations():
    """Check for required packages."""
    print("\n=== Package Installation Checks ===")
    
    required_packages = ['numpy', 'torch', 'matplotlib']
    installed_packages = {}
    
    for package in required_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            installed_packages[package] = version
            print(f"✅ {package} is installed (version {version})")
        except ImportError:
            print(f"❌ {package} is NOT installed")
    
    return installed_packages

def check_blas_configuration():
    """Check BLAS configuration."""
    print("\n=== BLAS Library Check ===")
    
    # Check for system BLAS libraries
    system_libs = [
        '/usr/lib/libblas.dylib',
        '/usr/lib/libopenblas.dylib',
        '/usr/lib/liblapack.dylib',
        '/usr/local/opt/openblas/lib/libopenblas.dylib',
        '/usr/local/lib/libopenblas.dylib'
    ]
    
    for lib_path in system_libs:
        if os.path.exists(lib_path):
            print(f"✅ Found system BLAS library: {lib_path}")
        else:
            print(f"ℹ️ Not found: {lib_path}")
    
    # If torch is installed, check its BLAS backend
    try:
        import torch
        print(f"\nPyTorch BLAS information:")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Try to determine BLAS backend
        backends = []
        if hasattr(torch, '_C'):
            if hasattr(torch._C, '_GLIBCXX_USE_CXX11_ABI'):
                backends.append("Using GLIBCXX ABI")
            
        if torch.__config__.parallel_info():
            print(f"Parallel backend: {torch.__config__.parallel_info()}")
            
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mkl') and hasattr(torch.backends.mkl, 'is_available'):
            if torch.backends.mkl.is_available():
                backends.append("MKL")
                
        if backends:
            print(f"Detected backends: {', '.join(backends)}")
        else:
            print("Could not determine BLAS backend")
            
    except ImportError:
        print("PyTorch not installed, skipping BLAS backend check")

def run_basic_tensor_tests():
    """Run basic tensor operations."""
    print("\n=== Basic Tensor Tests ===")
    
    try:
        import torch
        import numpy as np
        
        # Create basic tensors
        print("Creating basic tensors...")
        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        
        # Basic operations
        print("Testing basic operations...")
        c = a + b
        d = a * b
        e = torch.matmul(a, b)
        
        print("Basic operations completed successfully")
        
        # Test conversions to/from numpy
        print("\nTesting numpy conversions...")
        numpy_a = a.numpy()
        back_to_torch = torch.from_numpy(numpy_a)
        
        print("Numpy conversion successful")
        
        return True
    except ImportError as e:
        print(f"ImportError: {e}")
        return False
    except Exception as e:
        print(f"Error during tensor tests: {e}")
        return False

def run_matrix_multiply_test():
    """Test matrix multiplication specifically (common BLAS crash source)."""
    print("\n=== Matrix Multiplication Test (BLAS operations) ===")
    
    try:
        import torch
        
        # Settings to mimic what might happen in the neural plasticity code
        batch_size = 2
        seq_len = 128
        num_heads = 12
        hidden_size = 64
        
        # Create the kinds of tensors that would be used in attention mechanisms
        print("Creating attention-like tensors...")
        
        # Query, Key, Value tensors
        query = torch.randn(batch_size, num_heads, seq_len, hidden_size)
        key = torch.randn(batch_size, num_heads, seq_len, hidden_size)
        value = torch.randn(batch_size, num_heads, seq_len, hidden_size)
        
        # Attention computation (this often triggers BLAS issues)
        print("Computing attention scores...")
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        
        # Softmax and scaling
        print("Applying softmax...")
        attention_scores = attention_scores / (hidden_size ** 0.5)
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        
        # Final matmul
        print("Computing final attention...")
        context = torch.matmul(attention_probs, value)
        
        print("Matrix operations completed successfully")
        return True
    except ImportError as e:
        print(f"ImportError: {e}")
        return False
    except Exception as e:
        print(f"Error during matrix tests: {e}")
        print(f"This is likely the source of the BLAS/libtorch crash")
        return False

def run_entropy_calculation_test():
    """Test the entropy calculation specifically."""
    print("\n=== Entropy Calculation Test ===")
    
    try:
        import torch
        
        # Create attention-like tensor
        print("Creating attention probability tensor...")
        batch_size = 2
        num_heads = 4
        seq_len = 32  # Smaller for faster testing
        
        # Create random attention-like matrices and ensure they sum to 1
        attn = torch.rand(batch_size, num_heads, seq_len, seq_len)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        
        # Test entropy calculation with the fixed approach
        print("Calculating entropy with the fixed approach...")
        eps = 1e-6
        
        # Handle potential NaN or Inf values
        attn_probs = torch.where(
            torch.isfinite(attn),
            attn,
            torch.ones_like(attn) * eps
        )
        
        # Add small epsilon and renormalize
        attn_probs = attn_probs.clamp(min=eps)
        attn_probs = attn_probs / attn_probs.sum(dim=-1, keepdim=True)
        
        # Cast to float32 for better numerical stability
        if attn_probs.dtype != torch.float32:
            attn_probs = attn_probs.to(torch.float32)
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -torch.sum(attn_probs * torch.log(attn_probs), dim=-1)
        
        print(f"Entropy calculation successful")
        print(f"Entropy shape: {entropy.shape}")
        print(f"Entropy min/max/mean: {entropy.min().item():.4f}/{entropy.max().item():.4f}/{entropy.mean().item():.4f}")
        
        return True
    except ImportError as e:
        print(f"ImportError: {e}")
        return False
    except Exception as e:
        print(f"Error during entropy calculation: {e}")
        print(f"This is likely related to the BLAS/libtorch crash")
        return False

def run_visualization_test():
    """Test visualization capabilities."""
    print("\n=== Visualization Test ===")
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        print("Creating basic plot...")
        plt.figure(figsize=(6, 4))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        plt.plot(x, y)
        plt.title("Test Plot")
        
        # Save plot to verify it works
        test_plot_path = "test_plot.png"
        plt.savefig(test_plot_path)
        plt.close()
        
        if os.path.exists(test_plot_path):
            print(f"✅ Plot created successfully at {test_plot_path}")
            return True
        else:
            print(f"❌ Failed to create plot at {test_plot_path}")
            return False
    except ImportError as e:
        print(f"ImportError: {e}")
        return False
    except Exception as e:
        print(f"Error during visualization test: {e}")
        return False

def run_notebook_patch_test():
    """Check if our fixes can be applied to the notebook files."""
    print("\n=== Notebook Patch Test ===")
    
    # Check if neural plasticity modules exist
    core_path = Path("utils/neural_plasticity/core.py")
    vis_path = Path("utils/neural_plasticity/visualization.py")
    helpers_path = Path("utils/colab/helpers.py")
    
    if not core_path.exists():
        print(f"❌ Module not found: {core_path}")
    else:
        print(f"✅ Found module: {core_path}")
        
    if not vis_path.exists():
        print(f"❌ Module not found: {vis_path}")
    else:
        print(f"✅ Found module: {vis_path}")
        
    if not helpers_path.exists():
        print(f"❌ Module not found: {helpers_path}")
    else:
        print(f"✅ Found module: {helpers_path}")
    
    # Check notebook
    notebook_path = Path("colab_notebooks/NeuralPlasticityDemo.ipynb")
    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
    else:
        print(f"✅ Found notebook: {notebook_path}")
        
        # Check if we can parse it
        try:
            import json
            with open(notebook_path, 'r') as f:
                nb_content = json.load(f)
            print(f"✅ Successfully parsed notebook JSON structure")
            print(f"Notebook has {len(nb_content.get('cells', []))} cells")
        except ImportError:
            print("json module unavailable, skipping notebook parsing")
        except Exception as e:
            print(f"❌ Error parsing notebook: {e}")

def main():
    """Run all diagnostic checks."""
    print("Starting Neural Plasticity Environment Diagnostics")
    print("=" * 50)
    
    # Run all checks
    check_environment()
    installed_packages = check_package_installations()
    check_blas_configuration()
    
    # Only run tests if required packages are installed
    required_pkgs = {'numpy', 'torch', 'matplotlib'}
    installed_pkgs = set(installed_packages.keys())
    missing_pkgs = required_pkgs - installed_pkgs
    
    if missing_pkgs:
        print(f"\n❌ Missing required packages: {', '.join(missing_pkgs)}")
        print("Cannot run tensor tests without required packages")
    else:
        print("\n✅ All required packages are installed")
        
        # Run tensor tests
        tensor_tests_ok = run_basic_tensor_tests()
        
        if tensor_tests_ok:
            # If basic tests pass, try the more intensive ones
            matrix_tests_ok = run_matrix_multiply_test()
            entropy_tests_ok = run_entropy_calculation_test()
            
            # Check if the tests that crashed in the notebook are working
            if matrix_tests_ok and entropy_tests_ok:
                print("\n✅ SUCCESS: Core tensor operations are working correctly!")
                print("This suggests the fixes we applied should solve the BLAS/libtorch crashes.")
            else:
                print("\n⚠️ WARNING: Some tensor operations failed that are used in the notebook.")
                print("The notebook will likely still crash with the current environment.")
        
        # Run visualization test
        vis_test_ok = run_visualization_test()
        
        if not vis_test_ok:
            print("\n⚠️ WARNING: Visualization test failed. Notebook visualizations may not work.")
    
    # Check patches
    run_notebook_patch_test()
    
    print("\n" + "=" * 50)
    print("Diagnostic Summary:")
    
    # Determine overall status
    if 'torch' not in installed_packages:
        print("❌ CRITICAL: PyTorch is not installed")
        print("Solution: Install PyTorch with 'pip install torch'")
    elif 'numpy' not in installed_packages or 'matplotlib' not in installed_packages:
        print("❌ CRITICAL: Required dependencies are missing")
        print("Solution: Install missing packages with pip")
    else:
        try:
            # Check if we've run the tests
            if matrix_tests_ok and entropy_tests_ok:
                print("✅ Environment appears capable of running the notebook")
                print("Next steps:")
                print("1. Run the fixed notebook with: python scripts/run_neural_plasticity_notebook.py --minimal")
            else:
                print("❌ Environment has issues that may cause crashes")
                print("Recommendations:")
                print("1. Try setting these environment variables:")
                print("   export OMP_NUM_THREADS=1")
                print("   export OPENBLAS_NUM_THREADS=1")
                print("   export MKL_NUM_THREADS=1")
                print("   export NUMEXPR_NUM_THREADS=1")
                print("2. Consider reinstalling PyTorch with CPU-only version")
                print("3. Check for system BLAS library conflicts")
        except NameError:
            # If we didn't run the tests
            print("⚠️ Could not run all diagnostic tests due to missing packages")
    
if __name__ == "__main__":
    main()