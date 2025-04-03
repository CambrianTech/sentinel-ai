#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example of how to integrate the modular experiment framework with main.py

This script demonstrates:
1. Running a pruning experiment
2. Extracting the fine-tuned parameters
3. Running inference with main.py using the fine-tuned model
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import custom modules
from utils.pruning.experiment import PruningExperiment
from utils.pruning.environment import Environment
from utils.checkpoint import save_checkpoint


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Pruning integration example")
    
    # Model selection
    parser.add_argument("--model", type=str, default="distilgpt2",
                       help="Model to test (default: distilgpt2)")
                       
    # Experiment parameters
    parser.add_argument("--strategy", type=str, default="attention",
                       help="Pruning strategy to use (default: attention)")
    parser.add_argument("--pruning_level", type=float, default=0.3,
                       help="Pruning level to use (default: 0.3)")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of fine-tuning epochs")
    
    # Integration options
    parser.add_argument("--output_dir", type=str, default="integration_output",
                       help="Directory to save results and checkpoints")
    parser.add_argument("--prompt", type=str, 
                        default="Artificial intelligence will transform society by",
                        help="Prompt to use for evaluation")
    parser.add_argument("--run_inference", action="store_true",
                        help="Run inference with main.py after experiment")
    
    return parser.parse_args()


def save_model_for_inference(pruning_module, params, output_dir, model_name="pruned_model"):
    """
    Save the pruned and fine-tuned model for use with main.py
    
    Args:
        pruning_module: The pruning module instance
        params: The model parameters to save
        output_dir: Directory to save the model
        model_name: Name for the saved model
        
    Returns:
        Path to the saved model
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get model and gate values from params
    model = pruning_module.model
    
    # Convert JAX params to PyTorch (simplified example - actual implementation may differ)
    # This is a placeholder for the actual conversion logic
    try:
        # Get gate values (assuming gates are part of the params)
        gate_values = {}
        for i in range(pruning_module.num_layers):
            layer_gates = []
            for h in range(pruning_module.num_heads):
                # Calculate gate value for each head (implementation depends on model structure)
                # In a real implementation, this would extract gate values from the params
                gate_value = 0.0  # Default to zero (pruned)
                
                # This is just a placeholder logic - real logic depends on model structure
                if f"layer_{i}" in params and f"head_{h}" in params[f"layer_{i}"]:
                    gate_value = 1.0  # Active head
                
                layer_gates.append(gate_value)
            gate_values[i] = torch.tensor(layer_gates)
        
        # Create dummy PyTorch model for demonstration
        # In a real implementation, this would be a proper conversion of the pruned model
        dummy_model = {
            "gate_values": gate_values,
            "pruning_level": args.pruning_level,
            "strategy": args.strategy,
            "model_name": model_name
        }
        
        # Save model to disk
        checkpoint_path = output_dir / f"{model_name}.pth"
        torch.save(dummy_model, checkpoint_path)
        
        logger.info(f"Model saved to {checkpoint_path}")
        return checkpoint_path
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def run_inference_with_main(model_path, args):
    """
    Run inference using main.py with the saved model
    
    Args:
        model_path: Path to the saved model
        args: Command-line arguments
        
    Returns:
        None
    """
    try:
        # Import main module
        from main import generate_text
        from transformers import AutoTokenizer
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load the saved model (simplified example)
        loaded_model = torch.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # In a real implementation, this would convert the loaded model
        # to a format that can be used with generate_text
        
        # Run inference
        logger.info(f"Running inference with prompt: {args.prompt}")
        # In this example, we're just printing the loaded model info
        # In a real implementation, this would call generate_text
        logger.info(f"Model info: {json.dumps(loaded_model, indent=2)}")
        
        logger.info("Inference completed")
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        import traceback
        logger.error(traceback.format_exc())


def main():
    """Main entry point"""
    global args
    args = parse_args()
    
    # Print system information
    env = Environment()
    env.print_info()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create experiment
    experiment = PruningExperiment(
        results_dir=str(output_dir / "experiment_results"),
        use_improved_fine_tuner=True,
        detect_environment=True,
        optimize_memory=True
    )
    
    # Run experiment
    logger.info(f"Running experiment with model: {args.model}")
    result = experiment.run_single_experiment(
        model=args.model,
        strategy=args.strategy,
        pruning_level=args.pruning_level,
        prompt=args.prompt,
        fine_tuning_epochs=args.epochs,
        save_results=True
    )
    
    # Print summary
    logger.info("\nExperiment Summary:")
    
    baseline_perplexity = result["stages"]["baseline"]["perplexity"]
    logger.info(f"Baseline perplexity: {baseline_perplexity:.4f}")
    
    pruned_perplexity = result["stages"]["pruned"]["perplexity"]
    logger.info(f"Pruned perplexity: {pruned_perplexity:.4f}")
    
    if "fine_tuned" in result["stages"]:
        fine_tuned_perplexity = result["stages"]["fine_tuned"]["perplexity"]
        logger.info(f"Fine-tuned perplexity: {fine_tuned_perplexity:.4f}")
        
        # Plot results
        logger.info("Plotting results...")
        fig = experiment.plot_results()
        
        # Save the plot
        plot_path = output_dir / f"experiment_results.png"
        fig.savefig(plot_path)
        logger.info(f"Plot saved to {plot_path}")
        
        # Save model for inference (if requested)
        if args.run_inference:
            logger.info("Saving model for inference...")
            # In a real implementation, this would extract the fine-tuned params from result
            # Here we're just passing a dummy value for demonstration
            
            # Get the pruning module from the experiment
            pruning_module = experiment.pruning_module if hasattr(experiment, 'pruning_module') else None
            
            if pruning_module:
                # Save model
                model_path = save_model_for_inference(
                    pruning_module,
                    {}, # Dummy params - in reality, would use fine-tuned params from result
                    output_dir / "inference_models",
                    f"{args.model.replace('/', '_')}_{args.strategy}_{args.pruning_level}"
                )
                
                # Run inference with main.py
                if model_path:
                    logger.info("Running inference with main.py...")
                    run_inference_with_main(model_path, args)
            else:
                logger.error("Pruning module not found in experiment")
    else:
        logger.info("Fine-tuning was not performed or failed")
    
    logger.info("Integration example completed")


if __name__ == "__main__":
    main()