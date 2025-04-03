#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the improved fine-tuner with OPT models

This script serves as a test harness for the improved fine-tuner's ability to handle
OPT models which were previously experiencing NaN loss and other stability issues
during fine-tuning. This script can be run locally or on Colab to verify the improvements.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import custom modules
from utils.pruning import (
    Environment, 
    PruningModule,
    FineTuner,
    ImprovedFineTuner,
    get_strategy
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test improved fine-tuner with OPT models")
    
    # Model selection
    parser.add_argument("--model", type=str, default="facebook/opt-350m",
                       help="Model to test (default: facebook/opt-350m)")
                       
    # Pruning parameters
    parser.add_argument("--strategy", type=str, default="random",
                       help="Pruning strategy to use (default: random)")
    parser.add_argument("--pruning_level", type=float, default=0.3,
                       help="Pruning level to use (default: 0.3)")
    
    # Fine-tuning parameters
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of fine-tuning epochs")
    parser.add_argument("--dataset", type=str, default="wikitext",
                       help="Dataset to use for fine-tuning")
    parser.add_argument("--dataset_config", type=str, default="wikitext-103-v1",
                       help="Dataset configuration")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate for fine-tuning")
    parser.add_argument("--use_improved", action="store_true",
                       help="Force use of improved fine-tuner")
    parser.add_argument("--use_original", action="store_true",
                       help="Force use of original fine-tuner")
    
    # Other parameters
    parser.add_argument("--prompt", type=str, 
                        default="Artificial intelligence will transform society by",
                        help="Prompt to use for evaluation")
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Initialize environment
    env = Environment()
    env.print_info()
    
    if "opt" in args.model.lower() and "1.3b" in args.model.lower():
        logger.warning(
            "\n======================================================\n"
            "WARNING: Testing with OPT-1.3B model which is known to\n"
            "cause NaN issues with the original fine-tuner\n"
            "======================================================\n"
        )
        time.sleep(2)  # Pause to make sure user sees the warning
    
    # Initialize pruning module
    logger.info(f"Loading model {args.model}...")
    pruning_module = PruningModule(args.model)
    if not pruning_module.load_model():
        logger.error(f"Failed to load model {args.model}")
        return
    
    # Evaluate baseline model
    logger.info("Evaluating baseline model...")
    original_params = pruning_module.original_params
    
    # Evaluate perplexity and generation
    perplexity_baseline = pruning_module.evaluate_perplexity(original_params, args.prompt)
    logger.info(f"Baseline perplexity: {perplexity_baseline:.4f}")
    
    generated_baseline = pruning_module.generate_text(original_params, args.prompt)
    logger.info(f"Baseline generated: {generated_baseline}")
    
    # Apply pruning
    logger.info(f"Applying {args.strategy} pruning at {args.pruning_level:.2f} level...")
    
    # Get strategy
    pruning_strat = get_strategy(args.strategy, pruning_module, args.prompt)
    
    # Calculate importance scores
    logger.info("Calculating head importance...")
    all_head_importance = pruning_strat.get_head_importance(original_params)
    
    # Sort by importance (ascending)
    all_head_importance.sort(key=lambda x: x[2])
    
    # Determine number of heads to prune
    total_heads = pruning_module.num_layers * pruning_module.num_heads
    heads_to_prune = int(total_heads * args.pruning_level)
    logger.info(f"Pruning {heads_to_prune} out of {total_heads} heads")
    
    # Get head indices to prune (least important first)
    head_indices = [(l, h) for l, h, _ in all_head_importance[:heads_to_prune]]
    
    # Prune heads
    pruned_params = pruning_strat.prune_heads(original_params, head_indices)
    
    # Evaluate after pruning
    perplexity_pruned = pruning_module.evaluate_perplexity(pruned_params, args.prompt)
    logger.info(f"Post-pruning perplexity: {perplexity_pruned:.4f}")
    
    generated_pruned = pruning_module.generate_text(pruned_params, args.prompt)
    logger.info(f"Post-pruning generated: {generated_pruned}")
    
    # Choose fine-tuner based on arguments or model type
    use_improved = args.use_improved or "opt" in args.model.lower() or "large" in args.model.lower() or "1.3b" in args.model.lower()
    use_original = args.use_original
    
    if use_improved and use_original:
        logger.error("Cannot use both --use_improved and --use_original at the same time")
        return
    
    # Run fine-tuning with original tuner
    if use_original:
        logger.info(f"\n\n{'='*50}")
        logger.info("Testing original fine-tuner")
        logger.info(f"{'='*50}\n")
        
        fine_tuner = FineTuner(
            pruning_module, 
            dataset_name=args.dataset,
            batch_size=args.batch_size
        )
        
        try:
            tuned_params_original, metrics_original = fine_tuner.fine_tune(
                pruned_params,
                num_epochs=args.epochs,
                learning_rate=args.learning_rate,
                evaluate_interval=5
            )
            
            # Evaluate fine-tuned model
            perplexity_tuned_original = pruning_module.evaluate_perplexity(tuned_params_original, args.prompt)
            logger.info(f"Post-fine-tuning perplexity (original tuner): {perplexity_tuned_original:.4f}")
            
            generated_tuned_original = pruning_module.generate_text(tuned_params_original, args.prompt)
            logger.info(f"Post-fine-tuning generated (original tuner): {generated_tuned_original}")
            
        except Exception as e:
            logger.error(f"Error with original fine-tuner: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Run fine-tuning with improved tuner
    if use_improved:
        logger.info(f"\n\n{'='*50}")
        logger.info("Testing improved fine-tuner")
        logger.info(f"{'='*50}\n")
        
        improved_fine_tuner = ImprovedFineTuner(
            pruning_module, 
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            batch_size=args.batch_size
        )
        
        try:
            tuned_params_improved, metrics_improved = improved_fine_tuner.fine_tune(
                pruned_params,
                num_epochs=args.epochs,
                learning_rate=args.learning_rate,
                evaluate_interval=5
            )
            
            # Evaluate fine-tuned model
            perplexity_tuned_improved = pruning_module.evaluate_perplexity(tuned_params_improved, args.prompt)
            logger.info(f"Post-fine-tuning perplexity (improved tuner): {perplexity_tuned_improved:.4f}")
            
            generated_tuned_improved = pruning_module.generate_text(tuned_params_improved, args.prompt)
            logger.info(f"Post-fine-tuning generated (improved tuner): {generated_tuned_improved}")
            
        except Exception as e:
            logger.error(f"Error with improved fine-tuner: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
    logger.info(f"\n\n{'='*50}")
    logger.info("Testing completed")
    logger.info(f"{'='*50}\n")


if __name__ == "__main__":
    main()