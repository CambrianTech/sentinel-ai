#!/usr/bin/env python
"""
Test script for the consolidated FineTuner implementation.

This script tests the consolidated FineTuner with different stability levels
and compares it with the original implementations for compatibility.
"""

import os
import sys
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fine_tuner_test")

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import pruning utilities
from utils.pruning import PruningModule, FineTuner


def test_fine_tuner(model_name, stability_level, use_synthetic_data=True):
    """
    Test the consolidated FineTuner with the specified model and stability level.
    
    Args:
        model_name: Name of the model to test
        stability_level: Stability level (0, 1, or 2)
        use_synthetic_data: Whether to use synthetic data
        
    Returns:
        Dictionary with test results
    """
    logger.info(f"Testing consolidated FineTuner with {model_name} at stability level {stability_level}")
    
    # Create pruning module
    pruning_module = PruningModule(model_name)
    if not pruning_module.load_model():
        logger.error(f"Failed to load model {model_name}")
        return {"success": False, "error": "Failed to load model"}
    
    # Set model name for proper detection in fine-tuner
    pruning_module.model_name = model_name
    
    try:
        # Create fine-tuner with specified stability level
        fine_tuner = FineTuner(
            pruning_module=pruning_module,
            dataset_name="wikitext",
            dataset_config="wikitext-2-v1",
            batch_size=2,
            stability_level=stability_level,
            use_synthetic_data=use_synthetic_data
        )
        
        # Fine-tune for a single epoch
        params, metrics = fine_tuner.fine_tune(
            pruned_params=pruning_module.original_params,
            num_epochs=1,
            learning_rate=5e-5,
            evaluate_interval=5
        )
        
        # Evaluate results
        prompt = "Artificial intelligence will transform"
        try:
            generated = pruning_module.generate_text(params, prompt, max_length=30)
            perplexity = pruning_module.evaluate_perplexity(params, prompt)
            logger.info(f"Perplexity: {perplexity:.4f}")
            logger.info(f"Generated text: {generated}")
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
        
        return {
            "success": True,
            "model": model_name,
            "stability_level": stability_level,
            "metrics": metrics,
            "final_params": params
        }
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "model": model_name,
            "stability_level": stability_level,
            "error": str(e)
        }


def compare_stability_levels(model_name="distilgpt2", use_synthetic_data=True):
    """
    Compare different stability levels with the same model.
    
    Args:
        model_name: Name of the model to test
        use_synthetic_data: Whether to use synthetic data
        
    Returns:
        Dictionary with comparison results
    """
    logger.info(f"Comparing stability levels for model {model_name}")
    
    results = {}
    
    # Test each stability level
    for level in [0, 1, 2]:
        logger.info(f"Testing stability level {level}...")
        result = test_fine_tuner(model_name, level, use_synthetic_data)
        results[f"level_{level}"] = result
    
    # Plot comparison if all tests succeeded
    if all(r.get("success", False) for r in results.values()):
        plot_comparison(results)
    
    return results


def plot_comparison(results):
    """
    Plot comparison of training metrics across stability levels.
    
    Args:
        results: Dictionary with test results for different stability levels
    """
    # Reset matplotlib parameters
    plt.rcParams.update(plt.rcParamsDefault)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot losses
    ax1 = axes[0]
    for level, result in results.items():
        if result.get("success", False) and "metrics" in result:
            metrics = result["metrics"]
            epochs = [m["epoch"] for m in metrics]
            losses = [m["loss"] for m in metrics]
            level_num = level.split("_")[1]
            
            ax1.plot(epochs, losses, "o-", label=f"Stability Level {level_num}", linewidth=2)
    
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss by Stability Level")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()
    
    # Plot NaN counts
    ax2 = axes[1]
    has_nan_data = False
    
    for level, result in results.items():
        if result.get("success", False) and "metrics" in result:
            metrics = result["metrics"]
            epochs = [m["epoch"] for m in metrics]
            nan_counts = [m.get("nan_count", 0) for m in metrics]
            
            if any(nan_counts):
                has_nan_data = True
                
            level_num = level.split("_")[1]
            ax2.plot(epochs, nan_counts, "o-", label=f"Stability Level {level_num}", linewidth=2)
    
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("NaN Count")
    ax2.set_title("NaN Losses by Stability Level")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()
    
    if not has_nan_data:
        ax2.text(0.5, 0.5, "No NaN losses detected in any stability level",
                ha="center", va="center", fontsize=12, color="green")
    
    # Save figure
    fig.tight_layout(pad=2.0)
    fig.savefig("stability_level_comparison.png")
    logger.info("Comparison plot saved as stability_level_comparison.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Test the consolidated FineTuner implementation")
    parser.add_argument("--model", type=str, default="distilgpt2", help="Model to test with")
    parser.add_argument("--stability-level", type=int, default=None, 
                       help="Stability level to test (if not specified, all levels will be compared)")
    parser.add_argument("--real-data", action="store_true", help="Use real data instead of synthetic data")
    args = parser.parse_args()
    
    # Use synthetic data by default
    use_synthetic_data = not args.real_data
    
    # Test specific stability level or compare all
    if args.stability_level is not None:
        if args.stability_level not in [0, 1, 2]:
            logger.error(f"Invalid stability level: {args.stability_level}. Must be 0, 1, or 2.")
            return
            
        test_fine_tuner(args.model, args.stability_level, use_synthetic_data)
    else:
        compare_stability_levels(args.model, use_synthetic_data)


if __name__ == "__main__":
    main()