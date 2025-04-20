#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural Plasticity Full Experiment with Comprehensive Visualization

This script runs a complete neural plasticity experiment and generates
comprehensive visualizations of the entire process.

Usage:
    python scripts/run_neural_plasticity_full.py --model_name distilgpt2 --num_epochs 3 --output_dir viz_experiment
"""

import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    default_data_collator, 
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader
from datasets import load_dataset

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import neural plasticity modules
from utils.neural_plasticity import NeuralPlasticity
from utils.neural_plasticity.visualization import VisualizationReporter
from utils.colab.visualizations import visualize_complete_training_process

# Constants
DATE_FORMAT = "%Y%m%d-%H%M%S"
DEFAULT_OUTPUT_DIR = os.path.join("viz_experiment", f"run_{datetime.now().strftime(DATE_FORMAT)}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run neural plasticity experiment with visualization")
    
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                       help="Model name or path (default: distilgpt2)")
    
    parser.add_argument("--dataset", type=str, default="wikitext",
                       help="Dataset name (default: wikitext)")
    
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1",
                       help="Dataset configuration (default: wikitext-2-raw-v1)")
    
    parser.add_argument("--max_length", type=int, default=128,
                       help="Max sequence length (default: 128)")
    
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size (default: 4)")
    
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of epochs (default: 3)")
    
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate (default: 5e-5)")
    
    parser.add_argument("--warmup_steps", type=int, default=50,
                       help="Warmup steps (default: 50)")
    
    parser.add_argument("--eval_interval", type=int, default=50,
                       help="Evaluation interval in steps (default: 50)")
    
    parser.add_argument("--pruning_level", type=float, default=0.2,
                       help="Pruning level (default: 0.2)")
    
    parser.add_argument("--strategy", type=str, default="combined",
                       choices=["gradient", "entropy", "random", "combined"],
                       help="Pruning strategy (default: combined)")

    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    
    parser.add_argument("--device", type=str, default=None,
                       help="Device (default: auto)")
    
    return parser.parse_args()


def load_and_prepare_data(args):
    """Load and prepare datasets."""
    print(f"Loading dataset: {args.dataset}/{args.dataset_config}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    train_dataset = load_dataset(args.dataset, args.dataset_config, split="train")
    validation_dataset = load_dataset(args.dataset, args.dataset_config, split="validation")
    
    # Define tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=args.max_length
        )
    
    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    validation_dataset = validation_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Add labels for language modeling
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
    
    train_dataset = train_dataset.map(add_labels)
    validation_dataset = validation_dataset.map(add_labels)
    
    # Set format
    train_dataset = train_dataset.with_format("torch")
    validation_dataset = validation_dataset.with_format("torch")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=default_data_collator
    )
    
    validation_dataloader = DataLoader(
        validation_dataset, 
        batch_size=args.batch_size, 
        collate_fn=default_data_collator
    )
    
    return tokenizer, train_dataloader, validation_dataloader


def run_neural_plasticity_experiment(args):
    """Run the neural plasticity experiment with comprehensive visualization."""
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Get environment information
    env_info = NeuralPlasticity.get_environment_info()
    print("Environment:")
    for key, value in env_info.items():
        print(f"  {key}: {value}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment parameters
    with open(output_dir / "parameters.txt", "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    
    # Load and prepare data
    tokenizer, train_dataloader, validation_dataloader = load_and_prepare_data(args)
    
    # Create visualization reporter
    reporter = VisualizationReporter(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir / "visualizations",
        save_visualizations=True,
        verbose=True
    )
    
    # Create directories for different phases
    warmup_dir = output_dir / "warmup"
    pruning_dir = output_dir / "pruning"
    fine_tuning_dir = output_dir / "fine_tuning"
    visualization_dir = output_dir / "visualizations"
    
    for directory in [warmup_dir, pruning_dir, fine_tuning_dir, visualization_dir]:
        directory.mkdir(exist_ok=True)
    
    #--- PHASE 1: WARMUP ---#
    print("\n=== Starting Warmup Phase ===")
    warmup_results = NeuralPlasticity.run_warmup_training(
        model=model,
        train_dataloader=train_dataloader,
        max_epochs=1,
        learning_rate=args.learning_rate,
        patience=args.warmup_steps,
        device=device,
        verbose=True,
        save_visualizations=True,
        output_dir=warmup_dir
    )
    
    # Display warmup results
    reporter.display_warmup_results(warmup_results)
    
    # Baseline evaluation
    baseline_metrics = NeuralPlasticity.evaluate_model_performance(
        model=model,
        dataloader=validation_dataloader,
        device=device
    )
    print(f"Baseline metrics after warmup: Loss = {baseline_metrics['loss']:.4f}, Perplexity = {baseline_metrics['perplexity']:.2f}")
    
    #--- PHASE 2: PRUNING & FINE-TUNING ---#
    print("\n=== Starting Pruning Phase ===")
    pruning_results = NeuralPlasticity.run_pruning_cycle(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=validation_dataloader,
        pruning_level=args.pruning_level,
        strategy=args.strategy,
        learning_rate=args.learning_rate,
        training_steps=args.eval_interval * 2,
        save_visualizations=True,
        output_dir=pruning_dir
    )
    
    # Display pruning results
    reporter.display_pruning_results(pruning_results)
    
    # Print pruning summary
    pruned_heads = pruning_results.get("pruned_heads", [])
    total_heads = pruning_results.get("total_heads", 0)
    pruning_rate = len(pruned_heads) / total_heads if total_heads > 0 else 0
    print(f"Pruned {len(pruned_heads)} out of {total_heads} heads ({pruning_rate:.2%})")
    
    #--- PHASE 3: ADDITIONAL FINE-TUNING ---#
    print("\n=== Starting Fine-tuning Phase ===")
    fine_tuning_results = NeuralPlasticity.train_pruned_model(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=validation_dataloader,
        pruned_heads=pruned_heads,
        learning_rate=args.learning_rate / 2,  # Lower learning rate for fine-tuning
        steps=args.eval_interval * 4  # More steps for fine-tuning
    )
    
    # Final evaluation
    final_metrics = NeuralPlasticity.evaluate_model_performance(
        model=model,
        dataloader=validation_dataloader,
        device=device
    )
    print(f"Final metrics: Loss = {final_metrics['loss']:.4f}, Perplexity = {final_metrics['perplexity']:.2f}")
    
    # Calculate improvements
    loss_improvement = (baseline_metrics['loss'] - final_metrics['loss']) / baseline_metrics['loss'] * 100
    perplexity_improvement = (baseline_metrics['perplexity'] - final_metrics['perplexity']) / baseline_metrics['perplexity'] * 100
    
    print(f"Improvements:")
    print(f"  Loss: {loss_improvement:.2f}%")
    print(f"  Perplexity: {perplexity_improvement:.2f}%")
    
    # Compile experiment results
    experiment_results = {
        'warmup': warmup_results,
        'pruning': pruning_results,
        'fine_tuning': {
            'training_metrics': fine_tuning_results,
            'final_metrics': final_metrics
        },
        'baseline_metrics': baseline_metrics,
        'final_metrics': final_metrics,
        'improvements': {
            'loss': loss_improvement,
            'perplexity': perplexity_improvement
        },
        'pruning_info': {
            'pruned_heads': pruned_heads,
            'total_heads': total_heads,
            'pruning_rate': pruning_rate,
            'pruning_level': args.pruning_level,
            'strategy': args.strategy
        }
    }
    
    # Save experiment results
    torch.save(experiment_results, output_dir / "experiment_results.pt")
    
    # Generate comprehensive visualization
    print("\n=== Generating Comprehensive Visualizations ===")
    
    # Generate complete process dashboard
    complete_fig = visualize_complete_training_process(
        experiment=experiment_results,
        output_dir=output_dir / "visualizations",
        title="Complete Neural Plasticity Training Process",
        show_plot=False
    )
    
    # Generate all dashboards
    reporter.generate_comprehensive_dashboard(
        experiment=experiment_results,
        output_dir=output_dir / "dashboards"
    )
    
    # Save model and tokenizer
    model.save_pretrained(output_dir / "final_model")
    tokenizer.save_pretrained(output_dir / "final_model")
    
    print(f"\nExperiment completed. Results saved to: {output_dir}")
    print(f"Comprehensive visualizations saved to: {output_dir / 'visualizations'}")
    print(f"Dashboards saved to: {output_dir / 'dashboards'}")
    
    # Return experiment directory for further analysis
    return output_dir


def main():
    """Main function."""
    args = parse_args()
    
    print("Starting Neural Plasticity Experiment")
    print(f"Output directory: {args.output_dir}")
    
    output_dir = run_neural_plasticity_experiment(args)
    
    print("\nExperiment completed successfully!")
    print(f"To view results, open: {output_dir}/visualizations/neural_plasticity_process_*.png")


if __name__ == "__main__":
    main()