"""
Utilities for local testing of Upgrayedd functionality that runs on Colab.

This module provides functions for:
1. Creating CPU-friendly versions of experiments
2. Mocking GPU environments
3. Running smaller-scale validation tests
4. Simulating Colab notebook execution
"""

import os
import sys
import json
import logging
import tempfile
from typing import Dict, Any, Optional, List, Union

import torch
import numpy as np
from tqdm import tqdm

logger = logging.getLogger("Upgrayedd")


def create_test_config(
    full_config: Dict[str, Any],
    scale_factor: float = 0.1,
    force_cpu: bool = True,
    use_small_model: bool = True
) -> Dict[str, Any]:
    """
    Create a scaled-down test configuration for local testing.
    
    Args:
        full_config: The full configuration that would run on Colab
        scale_factor: Factor to scale down data and computation size
        force_cpu: Whether to force CPU usage regardless of GPU availability
        use_small_model: Whether to use a smaller model for testing
        
    Returns:
        A test configuration suitable for local testing
    """
    # Create a copy of the config
    test_config = dict(full_config)
    
    # Force CPU if requested
    if force_cpu:
        test_config["device"] = "cpu"
        
    # Use smaller model if requested
    if use_small_model and "model_name" in test_config:
        if test_config["model_name"].startswith("gpt2"):
            test_config["model_name"] = "distilgpt2"
        elif "bloom" in test_config["model_name"].lower():
            test_config["model_name"] = "bigscience/bloom-560m"
        elif "opt" in test_config["model_name"].lower():
            test_config["model_name"] = "facebook/opt-125m"
        elif "llama" in test_config["model_name"].lower():
            test_config["model_name"] = "openlm-research/open_llama_3b_v2"
        
    # Scale down batch size
    if "batch_size" in test_config:
        test_config["batch_size"] = max(1, int(test_config["batch_size"] * scale_factor))
        
    # Scale down sequence length
    if "max_length" in test_config:
        test_config["max_length"] = max(32, int(test_config["max_length"] * scale_factor))
        
    # Reduce number of cycles
    if "cycles" in test_config:
        test_config["cycles"] = min(2, test_config["cycles"])
        
    # Use tiny dataset
    if "dataset" in test_config:
        test_config["dataset"] = "tiny_shakespeare"
        
    # Reduce epochs
    if "epochs" in test_config:
        test_config["epochs"] = 1
        
    # Disable expensive operations
    test_config["visualize"] = False
    test_config["compress_model"] = False
    
    return test_config


def mock_gpu_environment():
    """
    Mock a GPU environment for testing code that checks for GPU.
    
    This function patches torch.cuda.is_available to return True and
    mocks some basic CUDA functionality. Useful for testing code paths
    that would only run on a GPU environment.
    
    Returns:
        The original torch.cuda.is_available function for restoration
    """
    original_is_available = torch.cuda.is_available
    
    # Create a device class that mimics CUDA device
    class MockCUDADevice:
        def __init__(self, index=0):
            self.index = index
            self.name = "Mock Tesla T4"
            
        def __str__(self):
            return self.name
    
    # Mock torch.cuda functions
    torch.cuda.is_available = lambda: True
    torch.cuda.current_device = lambda: 0
    torch.cuda.device_count = lambda: 1
    torch.cuda.get_device_name = lambda idx: "Mock Tesla T4"
    torch.cuda.get_device_properties = lambda idx: MockCUDADevice(idx)
    
    # Return the original function for restoration
    return original_is_available


def restore_gpu_environment(original_is_available):
    """
    Restore the original GPU environment after mocking.
    
    Args:
        original_is_available: The original torch.cuda.is_available function
    """
    torch.cuda.is_available = original_is_available


def run_notebook_cells(notebook_path: str, max_cells: Optional[int] = None) -> Dict[str, Any]:
    """
    Execute a Jupyter notebook programmatically for testing.
    
    Args:
        notebook_path: Path to the notebook file
        max_cells: Maximum number of cells to execute (None for all)
        
    Returns:
        Dict containing cell outputs and execution results
    """
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
        
        # Load the notebook
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
            
        # Create a preprocessor
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        
        # Execute only a subset of cells if requested
        if max_cells is not None:
            nb.cells = nb.cells[:max_cells]
            
        # Execute the notebook
        ep.preprocess(nb, {"metadata": {"path": os.path.dirname(notebook_path)}})
        
        # Collect outputs
        outputs = {}
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == "code" and cell.outputs:
                outputs[f"cell_{i}"] = cell.outputs
                
        return {
            "success": True,
            "outputs": outputs,
            "notebook": nb
        }
        
    except Exception as e:
        logger.error(f"Error running notebook: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def simulate_colab_experiment(
    config: Dict[str, Any],
    max_cycles: int = 2,
    mock_gpu: bool = True
) -> Dict[str, Any]:
    """
    Simulate running an Upgrayedd experiment as if on Colab.
    
    This function provides a local test environment that mimics
    Colab execution, making it easier to test code before deployment.
    
    Args:
        config: Experiment configuration
        max_cycles: Maximum number of cycles to run
        mock_gpu: Whether to mock a GPU environment
        
    Returns:
        Dict containing experiment results and metrics
    """
    # Create a temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set output directory
        config["output_dir"] = temp_dir
        
        # Mock GPU if requested
        original_is_available = None
        if mock_gpu:
            original_is_available = mock_gpu_environment()
            
        try:
            # Import Upgrayedd
            from upgrayedd.core import transform_model
            
            # Run the experiment
            config["cycles"] = max_cycles
            results = transform_model(
                model_name=config.get("model_name", "distilgpt2"),
                output_dir=config["output_dir"],
                device=config.get("device", "cpu"),
                config=config,
                mode="fixed"
            )
            
            # Load any generated metrics files
            metrics_files = {}
            for filename in os.listdir(temp_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(temp_dir, filename)
                    with open(file_path, "r") as f:
                        metrics_files[filename] = json.load(f)
                        
            return {
                "success": True,
                "results": results,
                "metrics_files": metrics_files
            }
            
        except Exception as e:
            logger.error(f"Error in simulated experiment: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
        finally:
            # Restore GPU environment if mocked
            if mock_gpu and original_is_available:
                restore_gpu_environment(original_is_available)
                

def validate_notebook_compatibility():
    """
    Check that all Upgrayedd notebooks can run locally.
    
    This function attempts to load and validate all notebooks
    in the colab_notebooks directory to ensure they can run locally.
    
    Returns:
        Dict mapping notebook names to validation results
    """
    results = {}
    notebook_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "colab_notebooks")
    
    if not os.path.exists(notebook_dir):
        logger.warning(f"Notebook directory not found: {notebook_dir}")
        return {}
        
    for filename in os.listdir(notebook_dir):
        if filename.endswith(".ipynb"):
            notebook_path = os.path.join(notebook_dir, filename)
            
            try:
                # Load notebook
                import nbformat
                with open(notebook_path, "r", encoding="utf-8") as f:
                    nb = nbformat.read(f, as_version=4)
                    
                # Check for specific Colab-only features
                colab_specific_features = []
                
                for cell in nb.cells:
                    if cell.cell_type == "code":
                        source = cell.source
                        
                        # Check for Colab-specific imports
                        if "from google.colab import" in source:
                            colab_specific_features.append("google.colab import")
                            
                        # Check for drive mounting
                        if "drive.mount" in source:
                            colab_specific_features.append("drive mounting")
                            
                        # Check for Colab forms
                        if "@param" in source:
                            colab_specific_features.append("@param forms")
                            
                # Create validation result
                results[filename] = {
                    "valid": len(colab_specific_features) == 0,
                    "colab_specific_features": colab_specific_features,
                    "can_run_locally": len(colab_specific_features) == 0
                }
                
            except Exception as e:
                results[filename] = {
                    "valid": False,
                    "error": str(e),
                    "can_run_locally": False
                }
                
    return results


if __name__ == "__main__":
    # Simple test when run directly
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Upgrayedd local testing utilities")
    
    # Create a test config
    test_config = create_test_config({
        "model_name": "gpt2",
        "dataset": "wikitext",
        "batch_size": 16,
        "max_length": 512,
        "cycles": 10,
        "epochs": 3,
        "visualize": True,
        "compress_model": True
    })
    
    print("\nTest Config:")
    for key, value in test_config.items():
        print(f"  {key}: {value}")
        
    # Validate notebooks
    notebook_results = validate_notebook_compatibility()
    
    print("\nNotebook Compatibility:")
    for notebook, result in notebook_results.items():
        status = "✅ Compatible" if result["can_run_locally"] else "❌ Not compatible"
        print(f"  {notebook}: {status}")
        if not result["can_run_locally"] and "colab_specific_features" in result:
            print(f"    Issues: {', '.join(result['colab_specific_features'])}")
            
    print("\nLocal testing utilities ready!")