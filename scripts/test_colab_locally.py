#!/usr/bin/env python
"""
Test Upgrayedd Colab notebooks locally.

This script validates and optionally runs Upgrayedd Colab notebooks locally,
making it easier to ensure that notebooks will work before pushing to GitHub.

Usage:
  python scripts/test_colab_locally.py [--notebook NOTEBOOK] [--execute] [--mock-gpu]
  
Options:
  --notebook NOTEBOOK  Specific notebook to test (defaults to all)
  --execute            Actually execute the notebook cells (not just validate)
  --mock-gpu           Mock GPU environment for testing CUDA code paths
  --max-cells N        Maximum number of cells to execute (for partial testing)
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add repository root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("Upgrayedd")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Upgrayedd Colab notebooks locally")
    parser.add_argument(
        "--notebook", 
        type=str, 
        help="Specific notebook to test (defaults to all)"
    )
    parser.add_argument(
        "--execute", 
        action="store_true", 
        help="Actually execute the notebook cells (not just validate)"
    )
    parser.add_argument(
        "--mock-gpu", 
        action="store_true", 
        help="Mock GPU environment for testing CUDA code paths"
    )
    parser.add_argument(
        "--max-cells", 
        type=int, 
        help="Maximum number of cells to execute (for partial testing)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    try:
        # Import utilities
        from upgrayedd.utils.local_testing import (
            mock_gpu_environment,
            restore_gpu_environment,
            run_notebook_cells,
            validate_notebook_compatibility
        )
        
        # Mock GPU if requested
        original_is_available = None
        if args.mock_gpu:
            logger.info("Mocking GPU environment for testing")
            import torch
            original_is_available = mock_gpu_environment()
            logger.info(f"GPU available: {torch.cuda.is_available()}")
            logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
            
        try:
            # Find notebooks to test
            notebook_dir = os.path.join(repo_root, "colab_notebooks")
            if not os.path.exists(notebook_dir):
                logger.error(f"Notebook directory not found: {notebook_dir}")
                return 1
                
            notebooks = []
            if args.notebook:
                # Test specific notebook
                notebook_path = args.notebook
                if not os.path.exists(notebook_path):
                    # Try with colab_notebooks directory
                    notebook_path = os.path.join(notebook_dir, args.notebook)
                    
                if not os.path.exists(notebook_path):
                    logger.error(f"Notebook not found: {args.notebook}")
                    return 1
                    
                notebooks = [notebook_path]
            else:
                # Test all notebooks
                for filename in os.listdir(notebook_dir):
                    if filename.endswith(".ipynb"):
                        notebooks.append(os.path.join(notebook_dir, filename))
                        
            logger.info(f"Found {len(notebooks)} notebooks to test")
            
            # Validate notebooks
            validation_results = validate_notebook_compatibility()
            
            # Print validation results
            print("\nNotebook Validation Results:")
            print("=" * 80)
            for notebook, result in validation_results.items():
                status = "✅ Compatible" if result["can_run_locally"] else "❌ Not compatible"
                print(f"{notebook}: {status}")
                if not result["can_run_locally"] and "colab_specific_features" in result:
                    print(f"  Issues: {', '.join(result['colab_specific_features'])}")
                    
            if args.execute:
                # Execute notebooks
                print("\nNotebook Execution Results:")
                print("=" * 80)
                
                for notebook_path in notebooks:
                    notebook_name = os.path.basename(notebook_path)
                    validation = validation_results.get(notebook_name, {})
                    
                    # Skip notebooks that can't run locally
                    if validation and not validation.get("can_run_locally", False):
                        print(f"{notebook_name}: ⚠️ Skipping - not compatible with local execution")
                        continue
                        
                    print(f"{notebook_name}: Running...")
                    
                    # Execute notebook
                    result = run_notebook_cells(
                        notebook_path,
                        max_cells=args.max_cells
                    )
                    
                    # Print result
                    if result["success"]:
                        print(f"{notebook_name}: ✅ Successfully executed")
                        if "outputs" in result:
                            print(f"  Cells executed: {len(result['outputs'])}")
                    else:
                        print(f"{notebook_name}: ❌ Execution failed")
                        if "error" in result:
                            print(f"  Error: {result['error']}")
                            
            return 0
            
        finally:
            # Restore GPU environment if mocked
            if args.mock_gpu and original_is_available:
                restore_gpu_environment(original_is_available)
                
    except ImportError as e:
        logger.error(f"Failed to import necessary modules: {e}")
        logger.error("Make sure upgrayedd package is installed")
        return 1
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())