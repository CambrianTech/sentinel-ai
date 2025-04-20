#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural Plasticity Experiment Runner

This script runs a complete neural plasticity experiment using real models and data.
It does NOT use any simulated components and makes all decisions based on mathematical
criteria rather than hardcoded schedules.

Usage:
    source .venv/bin/activate
    python run_experiment.py
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("neural_plasticity_experiment.log")
    ]
)

logger = logging.getLogger(__name__)

# Import neural plasticity experiment module
from utils.neural_plasticity.experiment import NeuralPlasticityExperiment

# Main experiment parameters
MODEL_NAME = "distilgpt2"  # Use a real model from HuggingFace
DATASET = "wikitext"  # Use a real dataset
DATASET_CONFIG = "wikitext-2-raw-v1"
OUTPUT_DIR = "neural_plasticity_output/complete_experiment"
BATCH_SIZE = 2
MAX_LENGTH = 64
WARMUP_PATIENCE = 10
MAX_WARMUP_STEPS = 30
TRAINING_STEPS = 30
PRUNING_LEVEL = 0.15
PRUNING_STRATEGY = "combined"

def main():
    """Run the complete neural plasticity experiment."""
    logger.info("=" * 80)
    logger.info("NEURAL PLASTICITY EXPERIMENT - Complete Implementation")
    logger.info("=" * 80)
    logger.info(f"Using model: {MODEL_NAME}")
    logger.info(f"Using dataset: {DATASET}/{DATASET_CONFIG}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # Create the experiment
        experiment = NeuralPlasticityExperiment(
            model_name=MODEL_NAME,
            dataset=DATASET,
            dataset_config=DATASET_CONFIG,
            output_dir=OUTPUT_DIR,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            pruning_level=PRUNING_LEVEL,
            pruning_strategy=PRUNING_STRATEGY,
            learning_rate=5e-5,
            device=None,  # Auto-detect
            verbose=True,
            save_results=True,
            show_samples=True,
            use_dashboard=False,
            dashboard_dir=os.path.join(OUTPUT_DIR, "dashboards")
        )
        
        # Setup experiment
        logger.info("Setting up experiment...")
        experiment.setup()
        
        # Run warmup phase
        logger.info("Running warmup phase...")
        experiment.run_warmup(
            max_epochs=1,
            patience=WARMUP_PATIENCE,
            min_steps=10,
            max_steps=MAX_WARMUP_STEPS
        )
        
        # Analyze attention patterns
        logger.info("Analyzing attention patterns...")
        experiment.analyze_attention()
        
        # Run pruning cycle
        logger.info("Running pruning cycle...")
        experiment.run_pruning_cycle(training_steps=TRAINING_STEPS)
        
        # Evaluate final model
        logger.info("Evaluating final model...")
        eval_metrics = experiment.evaluate()
        
        # Generate examples
        logger.info("Generating examples...")
        generated_texts = experiment.generate_examples()
        
        # Log improvement
        baseline_perplexity = experiment.baseline_perplexity
        final_perplexity = experiment.final_perplexity
        
        improvement = ((baseline_perplexity - final_perplexity) / baseline_perplexity) * 100
        
        logger.info("=" * 80)
        logger.info("EXPERIMENT RESULTS:")
        logger.info(f"Baseline perplexity: {baseline_perplexity:.2f}")
        logger.info(f"Final perplexity: {final_perplexity:.2f}")
        logger.info(f"Improvement: {improvement:.2f}%")
        logger.info(f"Pruned heads: {len(experiment.pruned_heads)} out of {6*12} ({len(experiment.pruned_heads)/72*100:.2f}%)")
        logger.info("=" * 80)
        
        # Display dashboard location
        dashboard_path = os.path.join(OUTPUT_DIR, "dashboards", "neural_plasticity_dashboard.html")
        logger.info(f"Dashboard available at: {dashboard_path}")
        
        # Return success
        return 0
        
    except Exception as e:
        logger.error(f"Error running experiment: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())