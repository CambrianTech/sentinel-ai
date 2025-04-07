#!/usr/bin/env python
"""
Example script demonstrating the Upgrayedd modular framework.

This script shows how to use the upgrayedd package to optimize a model
using different pruning strategies and configuration options.
"""

import argparse
import json
import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

from upgrayedd import UpgrayeddPipeline, transform_model
from upgrayedd.config import UpgrayeddConfig
from upgrayedd.strategies import get_strategy


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Upgrayedd model optimization")
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="distilgpt2",
        help="Model name or path (default: distilgpt2)"
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="wikitext", 
        choices=["wikitext", "shakespeare", "custom"],
        help="Dataset to use for optimization"
    )
    
    parser.add_argument(
        "--custom_dataset_path", 
        type=str, 
        help="Path to custom dataset (required if dataset=custom)"
    )
    
    parser.add_argument(
        "--strategy", 
        type=str, 
        default="entropy", 
        choices=["random", "entropy", "magnitude"],
        help="Pruning strategy to use"
    )
    
    parser.add_argument(
        "--pruning_ratio", 
        type=float, 
        default=0.3,
        help="Fraction of heads to prune (0.0 to 1.0)"
    )
    
    parser.add_argument(
        "--growth_ratio", 
        type=float, 
        default=0.1,
        help="Fraction of pruned heads to regrow (0.0 to 1.0)"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=1,
        help="Number of training epochs after each pruning cycle"
    )
    
    parser.add_argument(
        "--cycles", 
        type=int, 
        default=3,
        help="Number of prune-train cycles to perform"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./upgrayedd_output",
        help="Directory to save optimized model and metrics"
    )
    
    parser.add_argument(
        "--config_file", 
        type=str,
        help="Path to JSON or YAML configuration file (overrides command line args)"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Device to use (cpu, cuda, cuda:0, etc.)"
    )
    
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    return parser.parse_args()


def setup_logging(log_level):
    """Set up logging with the specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_config_from_file(config_file):
    """Load configuration from a JSON or YAML file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    if config_file.endswith(".json"):
        with open(config_file, "r") as f:
            return json.load(f)
    elif config_file.endswith((".yaml", ".yml")):
        try:
            import yaml
            with open(config_file, "r") as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML is required to load YAML config files")
    else:
        raise ValueError("Config file must be JSON or YAML")


def main():
    """Main entry point for the example script."""
    args = parse_args()
    setup_logging(args.log_level)
    
    # Load config from file if provided
    config_dict = vars(args)
    if args.config_file:
        file_config = load_config_from_file(args.config_file)
        config_dict.update(file_config)
    
    # Create config object
    config = UpgrayeddConfig(**config_dict)
    
    # Create pruning strategy
    strategy = get_strategy(
        config.strategy,
        pruning_ratio=config.pruning_ratio,
        growth_ratio=config.growth_ratio,
        min_heads=1,
        seed=42
    )
    
    logging.info(f"Running Upgrayedd with {config.strategy} strategy")
    logging.info(f"Model: {config.model_name}")
    logging.info(f"Dataset: {config.dataset}")
    logging.info(f"Pruning ratio: {config.pruning_ratio}, Growth ratio: {config.growth_ratio}")
    
    # Set device
    if config.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device
    
    logging.info(f"Using device: {device}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = model.to(device)
    
    # Option 1: Use transform_model high-level function
    optimized_model = transform_model(
        model=model,
        tokenizer=tokenizer,
        dataset_name=config.dataset,
        custom_dataset_path=config.custom_dataset_path,
        strategy=strategy,
        num_cycles=config.cycles,
        epochs_per_cycle=config.epochs,
        output_dir=config.output_dir,
        device=device
    )
    
    # Option 2: Use UpgrayeddPipeline for more control
    # pipeline = UpgrayeddPipeline(
    #     model=model,
    #     tokenizer=tokenizer,
    #     strategy=strategy, 
    #     device=device,
    #     output_dir=config.output_dir
    # )
    # 
    # # Prepare data
    # dataloader = pipeline.prepare_dataset(
    #     dataset_name=config.dataset,
    #     custom_dataset_path=config.custom_dataset_path
    # )
    # 
    # # Run optimization
    # optimized_model = pipeline.run_optimization(
    #     dataloader=dataloader, 
    #     num_cycles=config.cycles,
    #     epochs_per_cycle=config.epochs
    # )
    # 
    # # Save model and metrics
    # pipeline.save_model(optimized_model)
    # pipeline.save_metrics()
    
    logging.info(f"Optimization complete. Model saved to {config.output_dir}")
    
    # Print summary
    print("\nUpgrayedd Optimization Summary:")
    print(f"- Original model: {config.model_name}")
    print(f"- Strategy used: {config.strategy}")
    print(f"- Number of cycles: {config.cycles}")
    print(f"- Output directory: {config.output_dir}")
    print(f"- For detailed metrics, see {os.path.join(config.output_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()