#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run Adaptive Neural Plasticity Experiment with ANN Controller

This script provides a command-line interface for running adaptive neural
plasticity experiments with the ANN controller for dynamic attention head
management and comprehensive multi-phase visualizations.

Version: v0.1.0 (2025-04-20 22:30:00)
"""

import os
import sys
import logging
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
plasticity_dir = os.path.dirname(script_dir)
sentinel_dir = os.path.dirname(plasticity_dir)
project_root = os.path.dirname(sentinel_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("adaptive_plasticity")

def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run adaptive neural plasticity experiment with ANN controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Model name or path (e.g., gpt2, distilgpt2)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (cpu, cuda, auto). Auto-detected if not specified.")
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, default="wikitext",
                        help="Dataset name (e.g., wikitext, cnn_dailymail)")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1",
                        help="Dataset configuration")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    
    # Experiment configuration
    parser.add_argument("--warmup_steps", type=int, default=200,
                        help="Number of warmup steps")
    parser.add_argument("--finetuning_steps", type=int, default=200,
                        help="Number of fine-tuning steps per cycle")
    parser.add_argument("--analysis_steps", type=int, default=10,
                        help="Number of steps for analysis phase")
    parser.add_argument("--cycles", type=int, default=3,
                        help="Number of pruning cycles")
    parser.add_argument("--pruning_level", type=float, default=0.2,
                        help="Pruning level (0.0 to 1.0)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for training")
    
    # Controller configuration
    parser.add_argument("--controller_reg_weight", type=float, default=1e-4,
                        help="L1 regularization weight for controller")
    parser.add_argument("--controller_lr", type=float, default=0.01,
                        help="Controller learning rate")
    parser.add_argument("--entropy_threshold", type=float, default=1.5,
                        help="Entropy threshold for pruning")
    parser.add_argument("--importance_threshold", type=float, default=0.7,
                        help="Importance threshold for pruning")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for experiment results")
    parser.add_argument("--save_model", action="store_true", default=False,
                        help="Save model after experiment")
    parser.add_argument("--no_visualize", action="store_true", default=False,
                        help="Disable visualization generation")
    parser.add_argument("--quick_test", action="store_true", default=False,
                        help="Run quick test with minimal data")
    
    return parser.parse_args()

def main():
    """Main function to run the experiment."""
    args = parse_args()
    
    try:
        # Import required modules
        from sentinel.plasticity.controller_ann.adaptive_experiment import AdaptiveNeuralPlasticityExperiment
        from transformers import AutoTokenizer, default_data_collator
        from torch.utils.data import DataLoader
        from datasets import load_dataset
        
        # Configure output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.output_dir is None:
            output_dir = os.path.join(project_root, "experiment_output", 
                                     "neural_plasticity", f"run_{timestamp}")
        else:
            output_dir = args.output_dir
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure device
        if args.device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device
        
        logger.info(f"Using device: {device}")
        logger.info(f"Output directory: {output_dir}")
        
        # Set up controller configuration
        controller_config = {
            "init_value": 3.0,
            "reg_weight": args.controller_reg_weight,
            "controller_lr": args.controller_lr,
            "entropy_threshold": args.entropy_threshold,
            "importance_threshold": args.importance_threshold
        }
        
        # Create experiment instance
        experiment = AdaptiveNeuralPlasticityExperiment(
            output_dir=output_dir,
            device=device,
            model_name=args.model_name,
            adaptive_model=True,
            experiment_name=f"adaptive_{args.model_name.split('/')[-1]}_{timestamp}",
            controller_config=controller_config
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create dataloader builder
        def create_dataloader_builder():
            """Create a function that builds train and eval dataloaders."""
            
            def build_dataloaders():
                """Build train and eval dataloaders."""
                # For quick tests, use a small subset of the dataset
                if args.quick_test:
                    logger.info("Loading small dataset for quick test")
                    train_dataset = load_dataset(args.dataset, args.dataset_config, split="train[:100]")
                    eval_dataset = load_dataset(args.dataset, args.dataset_config, split="validation[:20]")
                else:
                    logger.info(f"Loading dataset: {args.dataset}/{args.dataset_config}")
                    train_dataset = load_dataset(args.dataset, args.dataset_config, split="train")
                    eval_dataset = load_dataset(args.dataset, args.dataset_config, split="validation")
                
                # Define tokenization function
                def tokenize_function(examples):
                    if "text" in examples:
                        text_key = "text"
                    else:
                        # Try to find a suitable text column
                        text_cols = [col for col in examples.keys() 
                                  if col.lower() in ["text", "content", "document", "article"]]
                        if text_cols:
                            text_key = text_cols[0]
                        else:
                            raise ValueError(f"Could not find text column in dataset. Available columns: {list(examples.keys())}")
                    
                    return tokenizer(
                        examples[text_key], 
                        padding="max_length", 
                        truncation=True, 
                        max_length=args.max_length
                    )
                
                # Process datasets
                train_dataset = train_dataset.map(tokenize_function, batched=True)
                eval_dataset = eval_dataset.map(tokenize_function, batched=True)
                
                # Remove original text columns
                text_columns = [col for col in train_dataset.column_names 
                              if col.lower() in ["text", "content", "document", "article"]]
                if text_columns:
                    train_dataset = train_dataset.remove_columns(text_columns)
                    eval_dataset = eval_dataset.remove_columns(text_columns)
                
                # Add labels for language modeling
                def add_labels(examples):
                    examples["labels"] = examples["input_ids"].copy()
                    return examples
                
                train_dataset = train_dataset.map(add_labels)
                eval_dataset = eval_dataset.map(add_labels)
                
                # Set torch format
                train_dataset = train_dataset.with_format("torch")
                eval_dataset = eval_dataset.with_format("torch")
                
                # Create dataloaders
                train_dataloader = DataLoader(
                    train_dataset, 
                    batch_size=args.batch_size, 
                    shuffle=True, 
                    collate_fn=default_data_collator
                )
                
                eval_dataloader = DataLoader(
                    eval_dataset, 
                    batch_size=args.batch_size, 
                    collate_fn=default_data_collator
                )
                
                return train_dataloader, eval_dataloader
            
            return build_dataloaders
        
        # Create dataloader builder
        dataloader_builder = create_dataloader_builder()
        
        # Adjust steps for quick test
        if args.quick_test:
            args.warmup_steps = 50
            args.finetuning_steps = 50
            args.analysis_steps = 5
        
        # Run the experiment
        logger.info("Starting experiment...")
        results = experiment.run_full_experiment(
            dataloader_builder_fn=dataloader_builder,
            warmup_steps=args.warmup_steps,
            finetuning_steps=args.finetuning_steps,
            analysis_steps=args.analysis_steps,
            pruning_cycles=args.cycles,
            pruning_level=args.pruning_level,
            learning_rate=args.learning_rate,
            config=controller_config
        )
        
        # Print summary
        logger.info("Experiment completed successfully!")
        logger.info("Summary:")
        logger.info(f"  Baseline perplexity: {results['summary']['baseline_perplexity']:.2f}")
        logger.info(f"  Final perplexity: {results['summary']['final_perplexity']:.2f}")
        logger.info(f"  Improvement: {results['summary']['improvement_percent']:.2f}%")
        logger.info(f"  Pruned heads: {results['summary']['pruned_heads']}/{results['summary']['total_heads']}")
        logger.info(f"  Sparsity: {results['summary']['sparsity']:.1f}%")
        logger.info(f"Results saved to: {output_dir}")
        
        # Save model if requested
        if args.save_model:
            logger.info("Saving model...")
            model_path = os.path.join(output_dir, "final_model")
            experiment.model.save_pretrained(model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Save controller if available
            if experiment.controller:
                controller_path = os.path.join(output_dir, "controller.pt")
                torch.save(experiment.controller.state_dict(), controller_path)
                logger.info(f"Controller saved to {controller_path}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Error running experiment: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())