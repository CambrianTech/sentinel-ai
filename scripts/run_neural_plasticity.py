#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural Plasticity Experiment Runner

This script executes neural plasticity experiments using a modular architecture
that works consistently in both command-line and notebook environments.
All outputs are stored in the /output directory with a standardized structure.

Version: v0.0.34 (2025-04-20 16:30:00)
"""

import os
import sys
import argparse
from datetime import datetime

# Add project root to path if needed
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Import the neural plasticity experiment
    from sentinel.plasticity.neural_plasticity_experiment import NeuralPlasticityExperiment
    
    # Create argument parser to override default output directory
    parser = argparse.ArgumentParser(description="Run neural plasticity experiment")
    parser.add_argument(
        "--output_dir", 
        type=str,
        default=None,
        help="Output directory. If not specified, defaults to /output/neural_plasticity_[timestamp]"
    )
    
    # Parse only the output directory argument before passing to experiment
    args, remaining = parser.parse_known_args()
    
    # Set standard output directory in /output if not specified
    if args.output_dir is None:
        # Ensure /output directory exists in project root
        output_base_dir = os.path.join(project_root, "output")
        if not os.path.exists(output_base_dir):
            os.makedirs(output_base_dir, exist_ok=True)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_base_dir, f"neural_plasticity_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        sys.argv.extend(["--output_dir", output_dir])
    
    # Run the main function with all arguments
    NeuralPlasticityExperiment.main()