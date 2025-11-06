"""
Test script for the WandbDashboard integration

This script tests the Weights & Biases integration for neural plasticity experiments.
It creates a simple mock experiment and tests all dashboard features.

Version: v0.0.1 (2025-04-20 25:45:00)
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import WandbDashboard
from utils.neural_plasticity.dashboard.wandb_integration import WandbDashboard

def run_mock_experiment(
    project_name: str = "neural-plasticity-test",
    experiment_name: str = None,
    output_dir: str = None,
    mode: str = "offline"
) -> None:
    """
    Run a mock experiment to test the wandb dashboard integration.
    
    Args:
        project_name: Name of the wandb project
        experiment_name: Name of this experiment run
        output_dir: Directory for wandb files (used for offline mode)
        mode: wandb mode ("online" or "offline")
    """
    # Create timestamp for experiment name if not provided
    if experiment_name is None:
        import time
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        experiment_name = f"test-dashboard-{timestamp}"
    
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.join(project_root, "test_output", "wandb")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create configuration for the experiment
    config = {
        "model_name": "distilgpt2",
        "dataset": "wikitext/wikitext-2-raw-v1",
        "pruning_strategy": "entropy",
        "pruning_level": 0.2,
        "learning_rate": 5e-5,
        "batch_size": 4,
        "cycles": 3,
        "environment": "test"
    }
    
    logger.info(f"Creating WandbDashboard with project: {project_name}, experiment: {experiment_name}")
    
    # Create dashboard
    dashboard = WandbDashboard(
        project_name=project_name,
        experiment_name=experiment_name,
        output_dir=output_dir,
        config=config,
        mode=mode,
        tags=["test", "neural-plasticity"]
    )
    
    # Get callbacks
    metrics_callback = dashboard.get_metrics_callback()
    sample_callback = dashboard.get_sample_callback()
    
    # Set initial phase
    dashboard.set_phase("setup")
    
    # Simulate experiment setup
    logger.info("Simulating experiment setup...")
    time.sleep(1)
    
    # Log metrics for setup phase
    metrics_callback(0, {
        "phase": "setup",
        "status": "initializing",
        "memory_usage": 1250,
        "initialization_time": 0.5
    })
    
    # Simulate model loading
    logger.info("Simulating model loading...")
    time.sleep(1)
    
    # Log more metrics
    metrics_callback(1, {
        "status": "loading_model",
        "memory_usage": 2560,
        "model_load_time": 1.2
    })
    
    # Start warmup phase
    logger.info("Starting warmup phase...")
    dashboard.set_phase("warmup")
    
    # Simulate warmup training
    warmup_losses = [5.2, 4.8, 4.5, 4.2, 4.0, 3.9, 3.85, 3.8, 3.78, 3.75]
    
    for step, loss in enumerate(warmup_losses):
        logger.info(f"Warmup step {step+1}/{len(warmup_losses)}, loss: {loss:.4f}")
        metrics_callback(step + 2, {
            "phase": "warmup",
            "warmup_loss": loss,
            "learning_rate": 5e-5,
            "step": step
        })
        time.sleep(0.5)
    
    # Switch to analysis phase
    logger.info("Starting analysis phase...")
    dashboard.set_phase("analysis")
    
    # Create mock entropy and gradient values
    num_layers = 6
    num_heads = 12
    
    # Generate random entropy values
    entropy_values = np.random.rand(num_layers, num_heads) * 2  # Range 0-2
    
    # Generate random gradient values
    gradient_values = np.random.rand(num_layers, num_heads) * 5  # Range 0-5
    
    # Log the entropy and gradient heatmaps
    metrics_callback(12, {
        "phase": "analysis",
        "entropy_values": entropy_values,
        "grad_norm_values": gradient_values
    })
    
    time.sleep(1)
    
    # Start pruning phase
    logger.info("Starting pruning phase...")
    dashboard.set_phase("pruning")
    
    # Simulate pruning decisions
    pruned_heads = [(0, 2), (1, 5), (2, 3), (3, 8), (4, 1), (5, 10)]
    
    # Log pruning decision
    pruning_decision = {
        "strategy": "entropy",
        "pruning_level": 0.2,
        "pruned_heads": pruned_heads,
        "cycle": 1
    }
    dashboard.log_pruning_decision(pruning_decision)
    
    # Simulate training after pruning
    logger.info("Simulating training after pruning...")
    
    # Initial perplexity after pruning
    initial_perplexity = 25.5
    
    # Training steps
    for step in range(10):
        # Calculate decreasing loss and perplexity over time
        train_loss = 3.5 - (step * 0.05)
        eval_loss = 3.6 - (step * 0.04)
        perplexity = initial_perplexity - (step * 0.8)
        
        logger.info(f"Training step {step+1}/10, "
                   f"loss: {train_loss:.4f}, "
                   f"eval_loss: {eval_loss:.4f}, "
                   f"perplexity: {perplexity:.2f}")
        
        metrics_callback(step + 13, {
            "phase": "pruning",
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "perplexity": perplexity,
            "sparsity": len(pruned_heads) / (num_layers * num_heads),
            "learning_rate": 5e-5 * (1 - step/10),  # Decreasing learning rate
            "step": step
        })
        
        time.sleep(0.5)
    
    # Log text samples
    logger.info("Logging text generation samples...")
    
    sample_callback(24, {
        "input_text": "Once upon a time",
        "generated_text": "Once upon a time, there was a magical kingdom ruled by a wise and just king. The king had three daughters, each more beautiful than the last.",
        "predicted_tokens": [
            "Once", "upon", "a", "time,", "there", "was", "a", "magical", "kingdom", 
            "ruled", "by", "a", "wise", "and", "just", "king.", "The", "king", "had", 
            "three", "daughters,", "each", "more", "beautiful", "than", "the", "last."
        ]
    })
    
    sample_callback(25, {
        "input_text": "In the future",
        "generated_text": "In the future, humanity has spread throughout the galaxy, establishing colonies on distant planets and developing advanced technologies.",
        "predicted_tokens": [
            "In", "the", "future,", "humanity", "has", "spread", "throughout", "the", 
            "galaxy,", "establishing", "colonies", "on", "distant", "planets", "and", 
            "developing", "advanced", "technologies."
        ]
    })
    
    # Switch to evaluation phase
    logger.info("Starting evaluation phase...")
    dashboard.set_phase("evaluation")
    
    # Log final metrics
    final_metrics = {
        "baseline_perplexity": 25.5,
        "final_perplexity": 18.2,
        "improvement_percent": 28.6,
        "execution_time": 125.3,
        "pruned_heads_count": len(pruned_heads),
        "sparsity": len(pruned_heads) / (num_layers * num_heads)
    }
    
    metrics_callback(26, final_metrics)
    
    time.sleep(1)
    
    # Complete the experiment
    logger.info("Completing experiment...")
    dashboard.set_phase("complete")
    
    metrics_callback(27, {
        "status": "completed",
        "message": "Experiment completed successfully"
    })
    
    # Finish the dashboard
    logger.info("Finishing dashboard...")
    dashboard.finish()
    
    logger.info("Mock experiment completed!")

def main():
    """Main function for running the test script."""
    parser = argparse.ArgumentParser(description="Test the WandbDashboard integration")
    
    parser.add_argument("--project", type=str, default="neural-plasticity-test",
                      help="Name of the wandb project")
    parser.add_argument("--name", type=str, default=None,
                      help="Name of the experiment run (default: auto-generated)")
    parser.add_argument("--output-dir", type=str, default=None,
                      help="Directory for wandb files (default: test_output/wandb)")
    parser.add_argument("--mode", type=str, default="offline", choices=["online", "offline"],
                      help="wandb mode (online or offline)")
    
    args = parser.parse_args()
    
    run_mock_experiment(
        project_name=args.project,
        experiment_name=args.name,
        output_dir=args.output_dir,
        mode=args.mode
    )

if __name__ == "__main__":
    main()