#!/usr/bin/env python
"""
Test script for pruning visualization functions.

This script tests the visualization functions for pruning, especially
the new gradient overlay visualization.

Usage:
    python -m utils.pruning.stability.test_visualization

Options:
    --output_dir DIR   Directory to save visualization outputs (default: ./test_viz)
    --verbose          Enable verbose logging
"""

import argparse
import logging
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from utils.pruning.visualization import plot_head_gradients_with_overlays


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test pruning visualization functions")
    parser.add_argument("--output_dir", type=str, default="./test_viz",
                        help="Directory to save visualization outputs")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    return parser.parse_args()


def test_gradient_overlay_visualization(output_dir):
    """Test the gradient overlay visualization function."""
    # Create test data
    num_layers = 6
    heads_per_layer = 12
    
    # Create random gradient norms
    grad_norms = np.random.rand(num_layers, heads_per_layer)
    
    # Generate some test pruned, revived, and vulnerable heads
    pruned_heads = [(0, 2), (1, 5), (2, 8), (3, 3), (4, 10)]
    revived_heads = [(0, 7), (2, 2), (4, 5)]
    vulnerable_heads = [(1, 1), (3, 9), (5, 11)]
    
    # Ensure the visualization handles both numpy and torch tensors
    tensor_tests = [
        ("numpy_array", grad_norms, "Gradient Overlay (NumPy array input)"),
        ("torch_tensor", torch.tensor(grad_norms), "Gradient Overlay (PyTorch tensor input)"),
        ("flattened_array", grad_norms.flatten(), "Gradient Overlay (Flattened array input)")
    ]
    
    results = []
    
    for name, tensor, title in tensor_tests:
        try:
            # Test creating the visualization
            fig = plot_head_gradients_with_overlays(
                grad_norms=tensor,
                pruned_heads=pruned_heads,
                revived_heads=revived_heads,
                vulnerable_heads=vulnerable_heads,
                title=title
            )
            
            # Save the figure to the output directory
            filename = f"grad_overlay_{name}.png"
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath)
            plt.close(fig)
            
            logging.info(f"✅ Visualization test passed for {name}, saved to {filepath}")
            results.append(True)
        except Exception as e:
            logging.error(f"❌ Visualization test failed for {name}: {str(e)}")
            results.append(False)
    
    # Test auto-detection of vulnerable heads
    try:
        # Create new gradient norms where some heads have very low values
        auto_detect_norms = np.random.rand(num_layers, heads_per_layer)
        # Set some heads to very low values that should be detected as vulnerable
        auto_detect_norms[0, 3] = 0.005
        auto_detect_norms[1, 2] = 0.003
        auto_detect_norms[4, 7] = 0.001
        
        fig = plot_head_gradients_with_overlays(
            grad_norms=auto_detect_norms,
            pruned_heads=pruned_heads,
            revived_heads=revived_heads,
            vulnerable_threshold=0.01,  # Heads with norms below this are vulnerable
            title="Gradient Overlay with Auto-detected Vulnerable Heads"
        )
        
        # Save the figure to the output directory
        filepath = os.path.join(output_dir, "grad_overlay_auto_detect.png")
        fig.savefig(filepath)
        plt.close(fig)
        
        logging.info(f"✅ Auto-detection test passed, saved to {filepath}")
        results.append(True)
    except Exception as e:
        logging.error(f"❌ Auto-detection test failed: {str(e)}")
        results.append(False)
    
    return all(results)


def main():
    """Run visualization tests."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving visualization outputs to {output_dir.absolute()}")
    
    # Run visualization tests
    overlay_result = test_gradient_overlay_visualization(output_dir)
    
    # Report results
    if overlay_result:
        logging.info("🎉 All visualization tests PASSED!")
        return 0
    else:
        logging.error("❌ Some visualization tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())