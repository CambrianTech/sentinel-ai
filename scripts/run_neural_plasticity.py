#\!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural Plasticity Experiment Runner

This script provides a frontend to the modular neural plasticity implementation
in scripts/neural_plasticity/. It maintains backward compatibility with existing
command-line arguments while using the consolidated implementation.

Usage:
    python scripts/run_neural_plasticity.py --model_name distilgpt2 --pruning_level 0.2
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path if needed
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Import and run the modular implementation
    # This redirects to the consolidated implementation
    from scripts.neural_plasticity.run_experiment import main

    # Run the main function from the modular implementation
    main()
