#!/usr/bin/env python
"""
Test script to investigate NaN issues with OPT models after heavy pruning.

This script specifically focuses on OPT models with high pruning levels
to identify and address numerical stability issues during fine-tuning.
"""

import os
import sys
import logging
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("opt_pruning_stability_test")

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import pruning utilities
from utils.pruning import PruningModule, get_strategy, FineTuner, ImprovedFineTuner
from utils.pruning.stability import nan_prevention

def setup_environment():
    """Configure environment for testing"""
    # Ensure consistent random seed for reproducibility
    np.random.seed(42)
    # Set JAX flags for better numerical stability
    os.environ["JAX_ENABLE_X64"] = "True"  # Enable double precision

def test_opt_pruning_stability(
    model_name: str = "facebook/opt-125m",
    pruning_level: float = 0.5,
    pruning_strategy: str = "random",
    learning_rate: float = 5e-6,
    batch_size: int = 2,
    sequence_length: int = 32,
    use_fp16: bool = False,
    gradient_clip_norm: float = 0.5,
    stop_on_nan: bool = False
):
    """
    Test OPT model pruning and fine-tuning with focus on numerical stability
    
    Args:
        model_name: Name of the OPT model to test
        pruning_level: Proportion of heads to prune (0.0 to 1.0)
        pruning_strategy: Pruning strategy to use
        learning_rate: Learning rate for fine-tuning
        batch_size: Batch size for fine-tuning
        sequence_length: Maximum sequence length
        use_fp16: Whether to use mixed precision (FP16)
        gradient_clip_norm: Gradient clipping norm
        stop_on_nan: Whether to stop on first NaN encounter
    """
    logger.info(f"Testing OPT model stability: {model_name}")
    logger.info(f"Configuration: pruning={pruning_level}, strategy={pruning_strategy}, lr={learning_rate}")
    
    # Initialize pruning module
    pruning_module = PruningModule(model_name)
    if not pruning_module.load_model():
        logger.error(f"Failed to load model {model_name}")
        return
    
    logger.info(f"Model loaded successfully. Layers: {pruning_module.num_layers}, Heads: {pruning_module.num_heads}")
    
    # Count parameters
    total_params = 0
    for param_dict in pruning_module.original_params.values():
        total_params += sum(p.size for p in jax.tree_util.tree_leaves(param_dict))
    logger.info(f"Total parameters: {total_params:,}")
    
    # Run baseline evaluation
    logger.info("Running baseline evaluation...")
    prompt = "Artificial intelligence will transform society by"
    
    try:
        baseline_perplexity = pruning_module.evaluate_perplexity(pruning_module.original_params, prompt)
        logger.info(f"Baseline perplexity: {baseline_perplexity:.4f}")
        
        baseline_text = pruning_module.generate_text(pruning_module.original_params, prompt)
        logger.info(f"Baseline generated: {baseline_text}")
    except Exception as e:
        logger.error(f"Error in baseline evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Apply pruning
    logger.info(f"Applying {pruning_strategy} pruning at {pruning_level:.2f} level...")
    
    try:
        # Create pruning strategy
        strategy = get_strategy(pruning_strategy, pruning_module, prompt)
        
        # Calculate head importance
        all_head_importance = strategy.get_head_importance(pruning_module.original_params)
        all_head_importance.sort(key=lambda x: x[2])
        
        # Calculate heads to prune
        total_heads = pruning_module.num_layers * pruning_module.num_heads
        heads_to_prune = int(total_heads * pruning_level)
        logger.info(f"Pruning {heads_to_prune} out of {total_heads} heads")
        
        # Get head indices to prune
        head_indices = [(l, h) for l, h, _ in all_head_importance[:heads_to_prune]]
        
        # Prune heads
        pruned_params = strategy.prune_heads(pruning_module.original_params, head_indices)
        
        # Evaluate pruned model
        pruned_perplexity = pruning_module.evaluate_perplexity(pruned_params, prompt)
        logger.info(f"Pruned perplexity: {pruned_perplexity:.4f}")
        
        pruned_text = pruning_module.generate_text(pruned_params, prompt)
        logger.info(f"Pruned generated: {pruned_text}")
        
        # Check if pruning caused significant perplexity increase
        if pruned_perplexity > baseline_perplexity * 5:
            logger.warning(f"Pruning caused a significant perplexity increase: {pruned_perplexity:.2f} vs {baseline_perplexity:.2f}")
    
    except Exception as e:
        logger.error(f"Error in pruning phase: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Fine-tune with different configurations to identify stability issues
    fine_tuning_configs = [
        {
            "name": "Standard",
            "use_improved": True,
            "stability_level": 1,
            "learning_rate": learning_rate,
            "gradient_clip_norm": 1.0
        },
        {
            "name": "Conservative",
            "use_improved": True,
            "stability_level": 2,
            "learning_rate": learning_rate / 2,
            "gradient_clip_norm": gradient_clip_norm
        },
        {
            "name": "Ultra-Conservative",
            "use_improved": True,
            "stability_level": 2,
            "learning_rate": learning_rate / 5,
            "gradient_clip_norm": gradient_clip_norm / 2,
            "use_synthetic_data": True,
            "sequence_length": min(sequence_length, 32),
            "batch_size": 1
        }
    ]
    
    results = []
    
    # Test each fine-tuning configuration
    for config in fine_tuning_configs:
        logger.info(f"\nTesting {config['name']} fine-tuning configuration...")
        
        # Create appropriate fine-tuner
        if config["use_improved"]:
            fine_tuner = ImprovedFineTuner(
                pruning_module,
                dataset_name="wikitext",
                dataset_config="wikitext-2-v1",
                batch_size=config.get("batch_size", batch_size),
                stability_level=config.get("stability_level", 1)
            )
            if "use_synthetic_data" in config:
                fine_tuner.use_synthetic_data = config["use_synthetic_data"]
            if "sequence_length" in config:
                fine_tuner.max_seq_length = config["sequence_length"]
        else:
            fine_tuner = FineTuner(
                pruning_module,
                dataset_name="wikitext",
                batch_size=config.get("batch_size", batch_size)
            )
        
        # Monitor for NaNs
        nan_counts = []
        last_loss = 0.0
        
        # Custom callback to monitor training
        def training_callback(epoch, step, loss, state):
            nonlocal nan_counts, last_loss
            
            # Check for NaN loss
            is_nan = np.isnan(loss) or np.isinf(loss)
            if is_nan:
                nan_counts.append((epoch, step))
                logger.warning(f"NaN detected at epoch {epoch}, step {step}")
                if stop_on_nan:
                    return False  # Stop training
            
            # Check for exploding loss
            if not is_nan and last_loss > 0 and loss > last_loss * 10:
                logger.warning(f"Exploding loss detected: {loss:.4f} vs previous {last_loss:.4f}")
            
            last_loss = float(loss) if not is_nan else last_loss
            return True  # Continue training
        
        try:
            # Fine-tune the model
            fine_tuned_params, metrics = fine_tuner.fine_tune(
                pruned_params,
                num_epochs=1,  # Just one epoch for testing
                learning_rate=config["learning_rate"],
                evaluate_interval=5,
                callback=training_callback
            )
            
            # Evaluate fine-tuned model
            fine_tuned_perplexity = pruning_module.evaluate_perplexity(fine_tuned_params, prompt)
            logger.info(f"Fine-tuned perplexity: {fine_tuned_perplexity:.4f}")
            
            fine_tuned_text = pruning_module.generate_text(fine_tuned_params, prompt)
            logger.info(f"Fine-tuned generated: {fine_tuned_text}")
            
            # Record results
            results.append({
                "config": config["name"],
                "baseline_perplexity": baseline_perplexity,
                "pruned_perplexity": pruned_perplexity,
                "fine_tuned_perplexity": fine_tuned_perplexity,
                "nan_count": len(nan_counts),
                "success": not np.isnan(fine_tuned_perplexity) and not np.isinf(fine_tuned_perplexity)
            })
            
        except Exception as e:
            logger.error(f"Error in fine-tuning with {config['name']} configuration: {e}")
            import traceback
            traceback.print_exc()
            
            # Record failure
            results.append({
                "config": config["name"],
                "baseline_perplexity": baseline_perplexity,
                "pruned_perplexity": pruned_perplexity,
                "fine_tuned_perplexity": float('nan'),
                "nan_count": len(nan_counts),
                "success": False,
                "error": str(e)
            })
    
    # Summarize results
    logger.info("\n=== Stability Test Results ===")
    logger.info(f"Model: {model_name}")
    logger.info(f"Pruning: {pruning_level:.2f} with {pruning_strategy} strategy")
    
    for result in results:
        success_mark = "✅" if result["success"] else "❌"
        logger.info(f"{success_mark} {result['config']}: {result['nan_count']} NaNs, " +
                   f"Perplexity={result['fine_tuned_perplexity']:.4f}")
    
    # Identify best configuration
    successful_configs = [r for r in results if r["success"]]
    if successful_configs:
        best_config = min(successful_configs, key=lambda r: r["fine_tuned_perplexity"])
        logger.info(f"\nBest configuration: {best_config['config']} with " +
                   f"perplexity={best_config['fine_tuned_perplexity']:.4f}")
    else:
        logger.warning("No successful configurations found.")
        
    return results


def main():
    """Main function to parse arguments and run tests"""
    parser = argparse.ArgumentParser(
        description="Test OPT model pruning stability with focus on numerical issues"
    )
    parser.add_argument("--model", type=str, default="facebook/opt-125m",
                       help="OPT model to test")
    parser.add_argument("--pruning-levels", nargs="+", type=float, default=[0.3, 0.5, 0.7],
                       help="Pruning levels to test")
    parser.add_argument("--strategy", type=str, default="random",
                       choices=["random", "magnitude", "entropy"],
                       help="Pruning strategy to use")
    parser.add_argument("--learning-rate", type=float, default=5e-6,
                       help="Base learning rate for fine-tuning")
    parser.add_argument("--output", type=str, default="opt_stability_results.json",
                       help="Output file for test results")
    
    args = parser.parse_args()
    
    # Set up environment
    setup_environment()
    
    # Run tests for different pruning levels
    all_results = {}
    for level in args.pruning_levels:
        logger.info(f"\n=== Testing pruning level {level:.2f} ===\n")
        results = test_opt_pruning_stability(
            model_name=args.model,
            pruning_level=level,
            pruning_strategy=args.strategy,
            learning_rate=args.learning_rate
        )
        all_results[f"level_{level:.2f}"] = results
    
    # Save results
    import json
    
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)
    
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)
    
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    # Import JAX here to avoid module not found errors
    import jax
    main()