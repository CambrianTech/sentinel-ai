# config/paths.py
"""
Global path configuration for the Sentinel-AI project.

This module defines paths for various data directories and ensures
they exist. These paths are used throughout the project for consistency.

Version: v0.0.25 (2025-04-20 23:50:00)
"""

import os
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Base paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# Neural plasticity specific paths
NEURAL_PLASTICITY_DIR = os.path.join(OUTPUT_DIR, "neural_plasticity")
PLASTICITY_MODELS_DIR = os.path.join(NEURAL_PLASTICITY_DIR, "models")
PLASTICITY_DASHBOARDS_DIR = os.path.join(NEURAL_PLASTICITY_DIR, "dashboards")
PLASTICITY_VISUALIZATIONS_DIR = os.path.join(NEURAL_PLASTICITY_DIR, "visualizations")

# Other experiment directories
EXPERIMENT_DIR = os.path.join(OUTPUT_DIR, "experiments")
TEST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "test_output")

# Required dependent directories
DIRECTORIES = [
    DATA_DIR,
    CHECKPOINT_DIR,
    PLOTS_DIR,
    OUTPUT_DIR,
    LOGS_DIR,
    NEURAL_PLASTICITY_DIR,
    PLASTICITY_MODELS_DIR,
    PLASTICITY_DASHBOARDS_DIR,
    PLASTICITY_VISUALIZATIONS_DIR,
    EXPERIMENT_DIR,
    TEST_OUTPUT_DIR
]

# Ensure all directories exist
for directory in DIRECTORIES:
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create directory {directory}: {e}")
