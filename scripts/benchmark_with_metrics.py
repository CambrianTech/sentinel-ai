#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark With Comprehensive Metrics

This script runs benchmarks with the comprehensive metrics collection system, tracking
pruning patterns, gate values, head importance, and performance metrics.

Usage:
    python scripts/benchmark_with_metrics.py --model_name gpt2 --output_dir ./benchmark_results
"""

import os
import sys
import argparse
import torch
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import necessary modules
from sentinel.utils.metric_collection import MetricCollector
# The loader functions are in the original models.loaders module, not yet in sentinel
from models.loaders.loader import load_baseline_model, load_adaptive_model
from utils.pruning.pruning_module import PruningModule


def setup_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark with comprehensive metrics")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                       help="Name of the model to benchmark")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run on (cpu, cuda)")
    
    # Pruning configuration
    parser.add_argument("--pruning_strategies", type=str, default="entropy,magnitude,random",
                       help="Comma-separated list of pruning strategies to benchmark")
    parser.add_argument("--pruning_levels", type=str, default="0.1,0.3,0.5",
                       help="Comma-separated list of pruning levels to benchmark")
    
    # Evaluation configuration
    parser.add_argument("--eval_dataset", type=str, default=None,
                       help="Optional dataset for evaluation")
    parser.add_argument("--eval_samples", type=int, default=100,
                       help="Number of evaluation samples")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for evaluation")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                       help="Directory to save results")
    parser.add_argument("--verbose", action="store_true",
                       help="Print verbose output")
    
    return parser.parse_args()


def prepare_model(args):
    """Prepare the model for benchmarking."""
    print(f"Loading baseline model: {args.model_name}")
    
    # Load baseline model
    baseline_model = load_baseline_model(args.model_name, args.device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create adaptive model
    print("Creating adaptive model")
    adaptive_model = load_adaptive_model(args.model_name, baseline_model, args.device, debug=args.verbose)
    
    return baseline_model, adaptive_model, tokenizer


def prepare_evaluation_data(tokenizer, args):
    """Prepare evaluation data."""
    if args.eval_dataset:
        # Load custom dataset if provided
        try:
            from datasets import load_dataset
            
            # Try to load from Hugging Face datasets
            dataset = load_dataset(args.eval_dataset, split="validation")
            
            # Get text column (try common column names)
            text_column = None
            for column in ["text", "content", "sentence", "input_text"]:
                if column in dataset.column_names:
                    text_column = column
                    break
            
            if text_column is None:
                print(f"Warning: Could not find text column in dataset. Using first column: {dataset.column_names[0]}")
                text_column = dataset.column_names[0]
            
            # Extract texts
            texts = dataset[text_column][:args.eval_samples]
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to synthetic data")
            texts = None
    else:
        texts = None
    
    # If no dataset or loading failed, create synthetic data
    if texts is None:
        print("Using synthetic evaluation data")
        
        # Create synthetic prompts
        prompts = [
            "The quick brown fox",
            "In a world where",
            "Once upon a time",
            "The history of artificial intelligence",
            "Climate change is",
        ]
        
        # Repeat prompts to get enough samples
        texts = []
        while len(texts) < args.eval_samples:
            texts.extend(prompts)
        texts = texts[:args.eval_samples]
    
    # Tokenize texts
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt"
    )
    
    # Create dataloaders
    from torch.utils.data import TensorDataset, DataLoader
    
    dataset = TensorDataset(tokenized["input_ids"], tokenized["attention_mask"])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    return dataloader


def benchmark_model(model, dataloader, tokenizer, collector, args, strategy=None, pruning_level=None):
    """Benchmark a model with comprehensive metrics collection."""
    device = next(model.parameters()).device
    model.eval()
    
    # Initialize pruning if requested
    if strategy is not None and pruning_level is not None:
        print(f"\nBenchmarking with {strategy} pruning at level {pruning_level}")
        
        # Direct implementation of pruning for adaptive model
        try:
            # Count total heads
            total_heads = 0
            for block in model.blocks:
                if hasattr(block, "attn") and hasattr(block["attn"], "gate"):
                    total_heads += len(block["attn"].gate)
            
            # Apply simple pruning (set gates to 0)
            num_to_prune = int(total_heads * pruning_level)
            num_pruned = 0
            
            # Very basic pruning based on strategy
            if strategy == "random":
                # Random pruning
                import random
                head_indices = []
                for layer_idx, block in enumerate(model.blocks):
                    for head_idx in range(len(block["attn"].gate)):
                        head_indices.append((layer_idx, head_idx))
                
                # Shuffle and select heads to prune
                random.shuffle(head_indices)
                prune_indices = head_indices[:num_to_prune]
                
                # Apply pruning
                for layer_idx, head_idx in prune_indices:
                    model.blocks[layer_idx]["attn"].gate[head_idx] = 0.0
                    num_pruned += 1
                    
            elif strategy == "magnitude" or strategy == "entropy":
                # For simplicity, just zero out the first num_to_prune heads
                # In a real implementation, we would use entropy or magnitude measurements
                remaining = num_to_prune
                for layer_idx, block in enumerate(model.blocks):
                    for head_idx in range(len(block["attn"].gate)):
                        if remaining > 0:
                            model.blocks[layer_idx]["attn"].gate[head_idx] = 0.0
                            num_pruned += 1
                            remaining -= 1
            
            print(f"Pruned {num_pruned}/{total_heads} heads ({num_pruned/total_heads:.1%})")
        
        except Exception as e:
            print(f"Error applying pruning: {e}")
            return {
                "loss": float('nan'),
                "perplexity": float('nan'),
                "elapsed_time": 0,
            }
    else:
        print("\nBenchmarking baseline model (no pruning)")
    
    # Create loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Track metrics
    all_losses = []
    all_perplexities = []
    step = 0
    
    # Define function to compute perplexity from loss
    def compute_perplexity(loss):
        return torch.exp(loss).item()
    
    # Run evaluation
    start_time = time.time()
    
    with torch.no_grad():
        for batch_num, (input_ids, attention_mask) in enumerate(dataloader):
            # Move batch to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Create shifted targets for language modeling (predict next token)
            targets = input_ids.clone()
            targets[:, :-1] = input_ids[:, 1:]  # Shift right
            targets[:, -1] = tokenizer.pad_token_id  # Pad last token
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Extract logits
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            
            # Calculate loss
            shift_logits = logits[:, :-1, :]
            shift_targets = targets[:, :-1]
            loss = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)), shift_targets.reshape(-1))
            
            # Calculate perplexity
            perplexity = compute_perplexity(loss)
            
            # Collect comprehensive metrics
            collector.collect_step_metrics(
                model=model,
                step=step,
                phase="eval",
                inputs={"input_ids": input_ids, "attention_mask": attention_mask},
                labels=shift_targets,
                logits=shift_logits,
                additional_metrics={
                    "eval/batch": batch_num,
                    "eval/loss": loss.item(),
                    "eval/perplexity": perplexity,
                    "eval/strategy": strategy or "baseline",
                    "eval/pruning_level": pruning_level or 0.0
                }
            )
            
            # Store metrics
            all_losses.append(loss.item())
            all_perplexities.append(perplexity)
            
            step += 1
            
            if args.verbose:
                print(f"Batch {batch_num+1}/{len(dataloader)}, "
                     f"Loss: {loss.item():.4f}, Perplexity: {perplexity:.4f}")
    
    end_time = time.time()
    
    # Calculate average metrics
    avg_loss = sum(all_losses) / len(all_losses)
    avg_perplexity = sum(all_perplexities) / len(all_perplexities)
    elapsed_time = end_time - start_time
    
    print(f"Evaluation completed in {elapsed_time:.2f}s")
    print(f"Loss: {avg_loss:.4f}, Perplexity: {avg_perplexity:.4f}")
    
    return {
        "loss": avg_loss,
        "perplexity": avg_perplexity,
        "elapsed_time": elapsed_time,
    }


def main():
    """Main function."""
    args = setup_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.model_name.replace('/', '_')}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare model and data
    baseline_model, adaptive_model, tokenizer = prepare_model(args)
    dataloader = prepare_evaluation_data(tokenizer, args)
    
    # Initialize metric collector
    collector = MetricCollector(
        output_dir=output_dir,
        model_name=args.model_name,
        track_gate_values=True,
        track_head_metrics=True,
        track_performance=True,
        track_pruning_patterns=True,
        compare_with_static=True,
        log_level="INFO"
    )
    
    # Parse pruning configurations
    pruning_strategies = args.pruning_strategies.split(",")
    pruning_levels = [float(level) for level in args.pruning_levels.split(",")]
    
    # Save benchmark configuration
    config = {
        "model_name": args.model_name,
        "device": args.device,
        "max_length": args.max_length,
        "eval_dataset": args.eval_dataset,
        "eval_samples": args.eval_samples,
        "batch_size": args.batch_size,
        "pruning_strategies": pruning_strategies,
        "pruning_levels": pruning_levels,
        "timestamp": timestamp
    }
    
    with open(os.path.join(output_dir, "benchmark_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Start with baseline (no pruning)
    print("\n" + "="*50)
    print(f"Benchmarking model: {args.model_name}")
    print("="*50)
    
    baseline_results = benchmark_model(
        model=adaptive_model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        collector=collector,
        args=args
    )
    
    # Register baseline results as a "strategy" for comparison
    collector.register_static_pruning_metrics("baseline", baseline_results)
    
    # Run benchmarks for each pruning strategy and level
    all_results = {
        "baseline": baseline_results
    }
    
    for strategy in pruning_strategies:
        strategy_results = {}
        
        for level in pruning_levels:
            print("\n" + "="*50)
            print(f"Benchmarking {strategy} pruning at level {level}")
            print("="*50)
            
            # Clone model to avoid interference between runs
            model_clone = load_adaptive_model(args.model_name, baseline_model, args.device, debug=False)
            
            # Run benchmark
            results = benchmark_model(
                model=model_clone,
                dataloader=dataloader,
                tokenizer=tokenizer,
                collector=collector,
                args=args,
                strategy=strategy,
                pruning_level=level
            )
            
            strategy_results[str(level)] = results
            
            # Register results for comparison
            collector.register_static_pruning_metrics(f"{strategy}_{level}", results)
            
            # Clear memory
            del model_clone
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        all_results[strategy] = strategy_results
    
    # Save all results
    with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Generate comprehensive analysis
    print("\n" + "="*50)
    print("Generating comprehensive analysis")
    print("="*50)
    
    report = collector.generate_report()
    
    # Create visualizations
    collector.visualize_metrics()
    
    # Save metrics to CSV for external analysis
    collector.save_metrics_csv()
    
    print("\n" + "="*50)
    print(f"Benchmark complete. Results saved to {output_dir}")
    print("="*50)
    
    # Print summary of best strategies
    if "static_comparison" in report and "overall_winner" in report["static_comparison"]:
        winner = report["static_comparison"]["overall_winner"]
        print(f"\nOverall best strategy: {winner}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())