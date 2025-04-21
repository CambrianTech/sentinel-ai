#!/usr/bin/env python
"""
Test Script for Neural Plasticity Dashboard in Colab Notebooks

Tests the integration with Colab notebooks by simulating a notebook environment
and ensuring the dashboard can be displayed properly.

Usage:
    python scripts/test_colab_integration.py

Author: Claude <noreply@anthropic.com>
Version: v0.0.1 (2025-04-20)
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

# Add root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the dashboard cell code
from utils.colab.neural_plasticity_dashboard_cell import (
    display_neural_plasticity_dashboard,
    create_synthetic_experiment_data
)

def test_notebook_cell():
    """
    Test the notebook cell to ensure it works properly.
    """
    # Create a synthetic experiment dataset
    experiment = create_synthetic_experiment_data()
    
    # Create output directory
    output_dir = "viz_notebook_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Display the dashboard - this would normally be shown in a notebook
    print("Generating visualization...")
    fig = display_neural_plasticity_dashboard(
        experiment=experiment,
        output_dir=output_dir,
        show_quote=True
    )
    
    # In a notebook environment, the visualization would be displayed
    # Here we'll just save it to a file to verify it worked
    save_path = os.path.join(output_dir, "notebook_dashboard_test.png")
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    
    # Verify the file was created
    if os.path.exists(save_path):
        print(f"✅ Successfully generated notebook visualization: {save_path}")
    else:
        print("❌ Failed to generate notebook visualization")
    
    return fig

if __name__ == "__main__":
    print("Testing Neural Plasticity Dashboard for Colab Notebooks...")
    test_notebook_cell()
    print("Test completed successfully!")