#!/usr/bin/env python
"""
Test Script for Complete Neural Plasticity Visualization

Tests the comprehensive visualization of the neural plasticity process
with synthetic data to ensure it works correctly.

Usage:
    python scripts/test_complete_visualization.py --output_dir viz_final_output

Author: Claude <noreply@anthropic.com>
Version: v0.0.1 (2025-04-20)
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

# Add root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import visualization utilities
from utils.colab.visualizations import (
    visualize_complete_training_process,
    extract_complete_training_data,
    generate_inspirational_quote
)
# Import from neural_plasticity module instead
from utils.neural_plasticity.visualization import visualize_warmup_dashboard

def create_synthetic_experiment_data():
    """
    Create synthetic data to mimic a neural plasticity experiment.
    
    Returns:
        Dictionary with experiment results
    """
    # Generate simulated loss curves with realistic patterns
    # Warmup: initial rapid decrease, then plateau
    warmup_steps = 20
    warmup_losses = [5.0 - 0.5 * np.sqrt(i) + 0.05 * np.random.randn() for i in range(1, warmup_steps + 1)]
    
    # Pruning: slight increase, then decrease
    pruning_steps = 15
    pruning_losses = [warmup_losses[-1] + 0.1 * np.sin(i/5) + 0.05 * np.random.randn() for i in range(1, pruning_steps + 1)]
    
    # Fine-tuning: steady decrease
    fine_tuning_steps = 25
    fine_tuning_losses = [pruning_losses[-1] - 0.03 * i + 0.05 * np.random.randn() for i in range(1, fine_tuning_steps + 1)]
    
    # Generate perplexity scores (inverse relationship with loss, roughly)
    warmup_perplexity = [np.exp(loss) for loss in warmup_losses][-5:]  # Only measure at end of warmup
    pruning_perplexity = [np.exp(loss) for loss in pruning_losses]
    fine_tuning_perplexity = [np.exp(loss) for loss in fine_tuning_losses]
    
    # Generate pruning metrics
    sparsity_values = [0.0]
    for i in range(1, pruning_steps):
        # Gradually increase sparsity to about 40%
        if i < pruning_steps - 3:
            sparsity_values.append(min(40.0, sparsity_values[-1] + np.random.uniform(2.0, 4.0)))
        else:
            sparsity_values.append(sparsity_values[-1])  # Plateau at the end
    
    # Calculate pruned head counts (assuming 144 total heads = 12 layers * 12 heads)
    total_heads = 144
    pruned_head_counts = [int(total_heads * sparsity / 100.0) for sparsity in sparsity_values]
    
    # Create experiment data structure
    experiment_data = {
        'warmup': {
            'losses': warmup_losses,
            'is_stable': True,
            'steps_without_decrease': 3,
            'initial_loss': warmup_losses[0],
            'final_loss': warmup_losses[-1],
            'stabilization_point': warmup_steps - 3,
            'segment_analysis': {
                'segment_size': 5,
                'first_segment_avg': np.mean(warmup_losses[:5]),
                'last_segment_avg': np.mean(warmup_losses[-5:]),
                'improvement': 100 * (1 - np.mean(warmup_losses[-5:]) / np.mean(warmup_losses[:5])),
                'still_improving': False
            }
        },
        'pruning': {
            'training_metrics': {
                'train_loss': pruning_losses,
                'step': list(range(len(pruning_losses))),
                'perplexity': pruning_perplexity,
                'pruned_heads': pruned_head_counts,
                'sparsity': sparsity_values
            },
            'baseline_metrics': {
                'loss': warmup_losses[-1],
                'perplexity': warmup_perplexity[-1] if warmup_perplexity else 0
            },
            'pruned_metrics': {
                'loss': pruning_losses[-1],
                'perplexity': pruning_perplexity[-1] if pruning_perplexity else 0
            }
        },
        'fine_tuning': {
            'training_metrics': {
                'train_loss': fine_tuning_losses,
                'step': list(range(len(fine_tuning_losses))),
                'perplexity': fine_tuning_perplexity
            },
            'final_metrics': {
                'loss': fine_tuning_losses[-1],
                'perplexity': fine_tuning_perplexity[-1] if fine_tuning_perplexity else 0
            }
        }
    }
    
    return experiment_data


def test_visualization(output_dir=None, show_plot=True):
    """
    Test the complete neural plasticity visualization with synthetic data.
    
    Args:
        output_dir: Directory to save the visualization
        show_plot: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    from utils.colab.visualizations import visualize_complete_training_process
    
    # Create synthetic data
    experiment_data = create_synthetic_experiment_data()
    
    # Generate visualization
    fig = visualize_complete_training_process(
        experiment=experiment_data,
        output_dir=output_dir,
        title="Complete Neural Plasticity Training Process (Synthetic Data)",
        show_plot=show_plot,
        show_quote=True
    )
    
    return fig


def test_dashboard_generator(output_dir=None, show_plot=True):
    """
    Test the neural plasticity dashboard generator with synthetic data.
    
    Args:
        output_dir: Directory to save the visualization
        show_plot: Whether to display the plot
        
    Returns:
        Dictionary of generated figures
    """
    # Import the dashboard generator
    from scripts.neural_plasticity_dashboard import generate_dashboards
    
    # Create synthetic data
    experiment_data = create_synthetic_experiment_data()
    
    # Generate dashboards
    figures = generate_dashboards(
        experiment=experiment_data,
        output_dir=output_dir,
        show=show_plot
    )
    
    return figures


def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Test neural plasticity visualizations")
    parser.add_argument('--output_dir', type=str, default='viz_test_output', help='Directory to save visualizations')
    parser.add_argument('--no_show', action='store_true', help='Do not display plots (useful for headless environments)')
    parser.add_argument('--test_all', action='store_true', help='Run all tests')
    parser.add_argument('--test_visualization', action='store_true', help='Test the complete visualization')
    parser.add_argument('--test_dashboard', action='store_true', help='Test the dashboard generator')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {os.path.abspath(args.output_dir)}")
    
    # Determine which tests to run
    run_visualization = args.test_all or args.test_visualization or not (args.test_all or args.test_visualization or args.test_dashboard)
    run_dashboard = args.test_all or args.test_dashboard
    
    # Run tests
    if run_visualization:
        print("Testing complete visualization...")
        viz_fig = test_visualization(args.output_dir, show_plot=not args.no_show)
        
        # Print information about the generated file
        if args.output_dir:
            files = [f for f in os.listdir(args.output_dir) if f.startswith("neural_plasticity_process")]
            if files:
                print(f"✅ Generated visualization: {os.path.join(args.output_dir, files[0])}")
    
    if run_dashboard:
        print("\nTesting dashboard generator...")
        dashboard_output_dir = os.path.join(args.output_dir, "dashboards") if args.output_dir else None
        figures = test_dashboard_generator(dashboard_output_dir, show_plot=not args.no_show)
        
        # Print information about the generated files
        if dashboard_output_dir and os.path.exists(dashboard_output_dir):
            print(f"✅ Generated dashboards in: {os.path.abspath(dashboard_output_dir)}")
            
    # Display plots if not in headless mode
    if not args.no_show:
        print("\nDisplaying plots (close windows to continue)...")
        plt.show()
    
    print("\nTests completed successfully!")


if __name__ == "__main__":
    main()