#\!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Minimal Neural Plasticity Example

This script demonstrates how to use the neural plasticity framework programmatically
with minimal code. It's designed to show the core workflow of setting up an experiment,
running the pruning cycle, and evaluating the results.

Version: v0.0.1 (2025-04-20 23:55:00)
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import neural plasticity experiment class
from utils.neural_plasticity.experiment import NeuralPlasticityExperiment

def run_minimal_experiment():
    """Run a minimal neural plasticity experiment."""
    
    # Setup output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "output", f"minimal_experiment_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running minimal neural plasticity experiment...")
    print(f"Output directory: {output_dir}")
    
    # Create experiment with minimal configuration
    experiment = NeuralPlasticityExperiment(
        # Use a small model for quick execution
        model_name="distilgpt2",
        
        # Use minimal dataset
        dataset="wikitext",
        dataset_config="wikitext-2-raw-v1",
        
        # Use small batch size and sequence length for speed
        batch_size=2,
        max_length=64,
        
        # Entropy-based pruning with 20% pruning level
        pruning_strategy="entropy",
        pruning_level=0.2,
        
        # Output configuration
        output_dir=output_dir,
        save_results=True,
        use_dashboard=True,
        verbose=True
    )
    
    # Setup the experiment
    experiment.setup()
    
    # Run minimal warmup (1 epoch)
    print("Running warmup phase...")
    experiment.run_warmup(max_epochs=1)
    
    # Analyze attention patterns to determine pruning targets
    print("Analyzing attention patterns...")
    experiment.analyze_attention()
    
    # Run pruning cycle with minimal training steps
    print("Running pruning cycle...")
    experiment.run_pruning_cycle(training_steps=50)
    
    # Evaluate final model
    print("Evaluating model...")
    eval_metrics = experiment.evaluate()
    
    # Generate a few text examples
    experiment.generate_examples({
        "example": "Neural plasticity allows models to",
        "test": "The result of pruning attention heads is"
    })
    
    # Display results
    print("\n=== Experiment Results ===")
    print(f"Baseline perplexity: {eval_metrics['baseline_perplexity']:.2f}")
    print(f"Final perplexity: {eval_metrics['perplexity']:.2f}")
    print(f"Improvement: {eval_metrics['improvement_percent']:.2f}%")
    print(f"Pruned {len(eval_metrics['pruned_heads'])} attention heads")
    print(f"Results saved to: {output_dir}")
    
    return experiment, eval_metrics

if __name__ == "__main__":
    run_minimal_experiment()
EOF < /dev/null