#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural Plasticity Dashboard Generator

This script creates a comprehensive dashboard for the neural plasticity
process, including warmup, stabilization, pruning, and fine-tuning phases.

Created for demonstration purposes using simulated but realistic data
based on actual experiment patterns.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import visualization utilities
from utils.colab.visualizations import visualize_complete_training_process

def create_simulated_data():
    """
    Create simulated but realistic training data for demonstration purposes.
    This follows the patterns of real neural plasticity training, but can be
    generated quickly without running a full experiment.
    """
    np.random.seed(42)  # For reproducibility
    
    # Create experiment dictionary structure
    experiment = {
        'warmup': {
            'losses': [],
            'stabilization_point': 75,
            'is_stable': True,
            'steps_without_decrease': 25,
            'polynomial_fit': {'coefficients': [0.0001, -0.02, 1.0, 4.5]}
        },
        'pruning': {
            'training_metrics': {
                'train_loss': [],
                'perplexity': [],
                'sparsity': [],
                'pruned_heads': []
            },
            'pruned_heads': [(i, j) for i in range(6) for j in range(2)],
            'total_heads': 72,
        },
        'fine_tuning': {
            'training_metrics': {
                'train_loss': [],
                'perplexity': []
            }
        },
        'baseline_metrics': {
            'loss': 4.2,
            'perplexity': 66.7
        },
        'final_metrics': {
            'loss': 3.5,
            'perplexity': 33.1
        },
        'improvements': {
            'loss': 16.7,
            'perplexity': 50.4
        }
    }
    
    # Generate warmup phase losses: starting high and gradually stabilizing
    initial_warmup_loss = 6.5
    num_warmup_steps = 150
    
    for i in range(num_warmup_steps):
        # Exponential decay with noise
        decay_rate = 0.97
        noise_factor = 0.4
        loss = initial_warmup_loss * (decay_rate ** (i/10))
        loss += np.random.normal(0, noise_factor * (1 - i/num_warmup_steps))
        loss = max(0.5, loss)  # Keep it positive
        experiment['warmup']['losses'].append(float(loss))
    
    # Generate pruning phase losses: spike after pruning, gradual recovery
    num_pruning_steps = 100
    last_warmup_loss = experiment['warmup']['losses'][-1]
    pruning_spike = last_warmup_loss * 1.8  # Loss spike after pruning
    
    for i in range(num_pruning_steps):
        # Recovery curve with noise
        recovery_rate = 0.98
        noise_factor = 0.2
        loss = pruning_spike * (recovery_rate ** i) 
        loss += np.random.normal(0, noise_factor)
        loss = max(0.5, loss)
        experiment['pruning']['training_metrics']['train_loss'].append(float(loss))
        
        # Generate perplexity scores: related to loss values
        perplexity = np.exp(loss) - np.random.uniform(0, 1)
        experiment['pruning']['training_metrics']['perplexity'].append(float(perplexity))
        
        # Generate sparsity increasing over time
        max_sparsity = 20.0  # 20% sparsity
        step_sparsity = max_sparsity * min(1.0, (i+1) / (num_pruning_steps * 0.7))
        experiment['pruning']['training_metrics']['sparsity'].append(float(step_sparsity))
        
        # Generate pruned head counts increasing over time
        max_pruned = 12  # Total heads to prune
        step_pruned = int(max_pruned * min(1.0, (i+1) / (num_pruning_steps * 0.7)))
        experiment['pruning']['training_metrics']['pruned_heads'].append(step_pruned)
    
    # Generate fine-tuning phase losses: continued improvement
    num_fine_tuning_steps = 200
    last_pruning_loss = experiment['pruning']['training_metrics']['train_loss'][-1]
    
    for i in range(num_fine_tuning_steps):
        # Gradual improvement curve with noise
        improvement_rate = 0.997
        noise_factor = 0.1
        loss = last_pruning_loss * (improvement_rate ** i)
        loss += np.random.normal(0, noise_factor)
        loss = max(0.5, loss)
        experiment['fine_tuning']['training_metrics']['train_loss'].append(float(loss))
        
        # Generate perplexity scores: related to loss values
        perplexity = np.exp(loss) - np.random.uniform(0, 1)
        experiment['fine_tuning']['training_metrics']['perplexity'].append(float(perplexity))
    
    return experiment


def main():
    """
    Main function to generate the dashboard visualization.
    """
    print("Generating Neural Plasticity Dashboard...")
    
    # Create output directory
    output_dir = Path("neural_plasticity_dashboard")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create simulated but realistic experiment data
    experiment_data = create_simulated_data()
    
    # Create comprehensive visualization
    print("Creating comprehensive visualization dashboard...")
    fig = visualize_complete_training_process(
        experiment=experiment_data,
        output_dir=output_dir,
        title="Neural Plasticity Complete Training Process",
        show_plot=False,
        figsize=(14, 10),
        show_quote=True
    )
    
    # Save visualization
    print(f"✅ Dashboard visualization saved to: {output_dir}")
    
    # Try to display figure in an appropriate way for the environment
    try:
        # For Jupyter/IPython environment
        from IPython.display import display
        display(fig)
    except (ImportError, NameError):
        # For standard Python script environment
        try:
            # Use a non-blocking method to show the plot if possible
            plt.ion()
            plt.show()
            plt.pause(0.001)  # Small pause to render
        except Exception:
            print("Visualization created but cannot be displayed in current environment.")
            print(f"Please open the saved PNG file in {output_dir}")
    
    print("\nNeural Plasticity Dashboard generated successfully!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()