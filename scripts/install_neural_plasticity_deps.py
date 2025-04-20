#!/usr/bin/env python
"""
Install Neural Plasticity Dependencies

This script installs all dependencies needed to run the Neural Plasticity Demo notebook.

Usage:
  # Activate the virtual environment first
  source .venv/bin/activate  # On Linux/Mac
  .venv\Scripts\activate     # On Windows
  
  # Then run the install script
  python scripts/install_neural_plasticity_deps.py
"""

import os
import sys
import subprocess
import platform

def check_dependency(package_name):
    """Check if a Python package is installed."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def install_dependencies():
    """Install all required dependencies for Neural Plasticity Demo."""
    print("Installing Neural Plasticity Demo dependencies...")
    
    # Core dependencies
    dependencies = [
        "torch",
        "transformers",
        "datasets",
        "matplotlib",
        "seaborn",
        "nbformat",
        "jupyter"
    ]
    
    # Platform-specific dependencies
    is_apple_silicon = platform.system() == "Darwin" and platform.processor() == "arm"
    if is_apple_silicon:
        print("🍎 Detected Apple Silicon - adding platform-specific dependencies")
        dependencies.extend([
            "numpy",
            "pyarrow"  # Required for datasets on Apple Silicon
        ])
    
    # Check what's missing
    missing_deps = [dep for dep in dependencies if not check_dependency(dep)]
    
    if not missing_deps:
        print("✅ All dependencies are already installed.")
        return True
    
    # Install missing dependencies
    print(f"Missing dependencies: {', '.join(missing_deps)}")
    try:
        # Use pip to install the missing dependencies
        cmd = [sys.executable, "-m", "pip", "install"] + missing_deps
        print(f"Running: {' '.join(cmd)}")
        
        process = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("Dependencies installed successfully:")
        for dep in missing_deps:
            print(f"  - {dep}")
            
        # Special instructions for datasets and its arrow backend
        if "datasets" in missing_deps:
            print("\nSetting up datasets with proper arrow backend...")
            subprocess.run(
                [sys.executable, "-c", "import datasets; datasets.enable_caching()"],
                check=False
            )
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        print("\nOutput:")
        print(e.stdout)
        print("\nErrors:")
        print(e.stderr)
        print("\nPlease install the dependencies manually:")
        print(f"pip install {' '.join(missing_deps)}")
        return False
    
if __name__ == "__main__":
    success = install_dependencies()
    
    if success:
        print("\n✅ All dependencies installed successfully.")
        print("You can now run the Neural Plasticity Demo.")
        print("\nTo fix and run the notebook:")
        print("  python scripts/run_neural_plasticity_notebook_e2e.py")
        print("\nTo fix only the dataset imports:")
        print("  python scripts/fix_neural_plasticity_datasets.py")
    else:
        print("\n❌ Failed to install all dependencies.")
        sys.exit(1)