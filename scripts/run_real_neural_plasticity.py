#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real Neural Plasticity Experiment - End-to-End Implementation

This script runs a COMPLETE neural plasticity experiment using REAL models and REAL datasets.
It executes the full process from warmup through pruning to fine-tuning while generating
comprehensive visualizations of all decisions.

The experiment consists of:
1. Loading a real HuggingFace transformer model
2. Preparing a real dataset (wikitext by default)
3. Running a warmup phase with mathematical stabilization detection
4. Performing pruning based on entropy and gradient metrics
5. Fine-tuning the pruned model
6. Evaluating the model with actual text generation
7. Generating comprehensive HTML dashboards showing all decisions

This script works identically in both local and Google Colab environments.

Usage:
    # Activate virtual environment first
    source .venv/bin/activate
    
    # Run with default parameters (distilgpt2 on wikitext)
    python scripts/run_real_neural_plasticity.py
    
    # Run with custom parameters
    python scripts/run_real_neural_plasticity.py --model_name gpt2 --dataset wikitext --dataset_config wikitext-2-raw-v1
    
    # Adjust pruning strategy
    python scripts/run_real_neural_plasticity.py --pruning_strategy combined --pruning_level 0.3
    
Version: v0.0.68 (2025-04-20 18:55:00)
"""

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import time
import gc
import json
import logging
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import transformers components
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    default_data_collator, 
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader
from datasets import load_dataset

# Add project root to path to ensure imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)

# Import neural plasticity modules
from utils.neural_plasticity import NeuralPlasticity
from utils.neural_plasticity.visualization import VisualizationReporter
from utils.neural_plasticity.core import (
    calculate_head_entropy,
    calculate_head_gradients,
    detect_model_structure,
    evaluate_model,
    generate_pruning_mask,
    apply_pruning_mask,
    IS_APPLE_SILICON,
    IS_COLAB,
    HAS_GPU
)
from utils.neural_plasticity.training import run_warmup_phase, run_plasticity_loop
from utils.neural_plasticity.experiment import run_neural_plasticity_experiment

# Constants
DATE_FORMAT = "%Y%m%d-%H%M%S"
DEFAULT_OUTPUT_DIR = os.path.join("experiment_results", f"run_{datetime.now().strftime(DATE_FORMAT)}")
VERSION = "v0.0.68 (2025-04-20 18:55:00)"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run complete neural plasticity experiment with real models and data")
    
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
    
    parser.add_argument("--num_epochs", type=int, default=1,
                       help="Number of epochs (default: 1)")
    
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate (default: 5e-5)")
    
    parser.add_argument("--warmup_patience", type=int, default=20,
                       help="Warmup patience for stabilization detection (default: 20)")
    
    parser.add_argument("--pruning_level", type=float, default=0.2,
                       help="Pruning level (default: 0.2)")
    
    parser.add_argument("--pruning_strategy", type=str, default="combined",
                       choices=["gradient", "entropy", "random", "combined"],
                       help="Pruning strategy (default: combined)")

    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    
    parser.add_argument("--device", type=str, default=None,
                       help="Device (default: auto)")
    
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    parser.add_argument("--training_steps", type=int, default=100,
                       help="Training steps per phase (default: 100)")
    
    parser.add_argument("--eval_steps", type=int, default=10,
                       help="Evaluation steps (default: 10)")
    
    parser.add_argument("--generate_html", action="store_true", default=True,
                       help="Generate HTML dashboard (default: True)")
    
    parser.add_argument("--sample_texts", type=int, default=3,
                       help="Number of sample texts to generate for evaluation (default: 3)")
    
    return parser.parse_args()


def setup_environment(seed=42):
    """Set up environment for reproducibility and logging."""
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Detect execution environment
    env_info = NeuralPlasticity.get_environment_info()
    
    logger.info(f"Neural Plasticity Experiment {VERSION}")
    logger.info(f"Environment Information:")
    logger.info(f"  Platform: {env_info['platform']}")
    logger.info(f"  Python Version: {env_info['python_version']}")
    logger.info(f"  PyTorch Version: {env_info['pytorch_version']}")
    logger.info(f"  Device: {env_info['device']}")
    
    if env_info['is_colab']:
        logger.info(f"  Running in Google Colab")
        if env_info['has_gpu']:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"  GPU: {gpu_name}")
            # Show GPU memory
            logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    if env_info['is_apple_silicon']:
        logger.info(f"  Running on Apple Silicon")
        logger.info(f"  Using CPU for inference to avoid instability")
    
    return env_info


def load_and_prepare_data(args, tokenizer):
    """Load and prepare real datasets from HuggingFace."""
    logger.info(f"Loading dataset: {args.dataset}/{args.dataset_config}")
    
    # Load datasets
    try:
        train_dataset = load_dataset(args.dataset, args.dataset_config, split="train")
        validation_dataset = load_dataset(args.dataset, args.dataset_config, split="validation")
        
        logger.info(f"Loaded training dataset with {len(train_dataset)} examples")
        logger.info(f"Loaded validation dataset with {len(validation_dataset)} examples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
    
    # Define tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=args.max_length
        )
    
    # Tokenize datasets
    logger.info("Tokenizing datasets...")
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
    
    # Save a few example prompts for text generation tests
    sample_texts = []
    
    # Safely extract sample texts
    try:
        # Convert to int to avoid numpy type issues
        num_samples = min(args.sample_texts, len(validation_dataset))
        sample_ids = np.random.choice(len(validation_dataset), num_samples, replace=False)
        
        for i in range(num_samples):
            # Convert numpy int to python int
            idx = int(sample_ids[i])
            sample = validation_dataset[idx]
            input_ids = sample["input_ids"]
            # Take first 25 tokens as prompt
            sample_text = tokenizer.decode(input_ids[:25])
            sample_texts.append(sample_text)
    except Exception as e:
        logger.warning(f"Error extracting sample texts: {e}")
        # Fallback to simple text prompts if we couldn't extract from dataset
        sample_texts = [
            "The transformer architecture has",
            "In the context of neural networks"
        ][:args.sample_texts]
    
    logger.info(f"Created data loaders with batch size {args.batch_size}")
    
    return train_dataloader, validation_dataloader, sample_texts


def generate_text_samples(model, tokenizer, prompts, device=None, max_length=50):
    """Generate text samples to evaluate model capabilities."""
    if not prompts:
        return []
        
    if device is None:
        device = next(model.parameters()).device
    
    # Save model state
    model.eval()
    
    generated_texts = []
    for prompt in prompts:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate
        try:
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    do_sample=True,
                    top_p=0.92,
                    temperature=0.8,
                    num_return_sequences=1
                )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_texts.append(generated_text)
                
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            generated_texts.append(f"[Error: {str(e)}]")
    
    return generated_texts


def run_experiment(args):
    """Run the complete neural plasticity experiment."""
    start_time = time.time()
    
    # Setup environment
    env_info = setup_environment(args.seed)
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        if env_info["is_apple_silicon"]:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment parameters
    params_file = output_dir / "parameters.json"
    with open(params_file, "w") as f:
        params = {
            "model_name": args.model_name,
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "warmup_patience": args.warmup_patience,
            "pruning_level": args.pruning_level,
            "pruning_strategy": args.pruning_strategy,
            "training_steps": args.training_steps,
            "seed": args.seed,
            "device": str(device),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": VERSION,
            "environment": {
                "is_apple_silicon": env_info["is_apple_silicon"],
                "is_colab": env_info["is_colab"],
                "has_gpu": env_info["has_gpu"],
                "platform": env_info["platform"],
                "python_version": env_info["python_version"],
                "pytorch_version": env_info["pytorch_version"]
            }
        }
        json.dump(params, f, indent=2)
    
    logger.info(f"Saved experiment parameters to {params_file}")
    
    # Create subdirectories
    dirs = {
        "warmup": output_dir / "warmup",
        "pruning": output_dir / "pruning",
        "fine_tuning": output_dir / "fine_tuning", 
        "visualizations": output_dir / "visualizations",
        "inference": output_dir / "inference",
        "metrics": output_dir / "metrics",
        "html": output_dir / "html",
        "model": output_dir / "model"
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(exist_ok=True)
    
    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        model = model.to(device)
        
        # Get model structure
        num_layers, num_heads = detect_model_structure(model)
        logger.info(f"Model loaded: {num_layers} layers with {num_heads} attention heads each")
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {param_count:,}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None
    
    # Prepare datasets
    try:
        train_dataloader, validation_dataloader, sample_prompts = load_and_prepare_data(args, tokenizer)
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return None
    
    # Create visualization reporter
    reporter = VisualizationReporter(
        model=model,
        tokenizer=tokenizer,
        output_dir=dirs["visualizations"],
        save_visualizations=True,
        verbose=True
    )
    
    # Initialize metrics tracker
    experiment_metrics = {
        "phases": [],
        "warmup": {
            "losses": [],
            "steps": [],
            "stabilization_point": None
        },
        "pruning": {
            "pruned_heads": [],
            "metrics_before": None,
            "metrics_after": None
        },
        "fine_tuning": {
            "losses": [],
            "steps": []
        },
        "inference": {
            "baseline": None,
            "after_warmup": None,
            "after_pruning": None, 
            "after_fine_tuning": None
        },
        "baseline_metrics": None,
        "final_metrics": None
    }
    
    # Baseline evaluation
    logger.info("Evaluating baseline model...")
    baseline_metrics = evaluate_model(
        model=model,
        dataloader=validation_dataloader,
        device=device,
        max_eval_steps=args.eval_steps
    )
    logger.info(f"Baseline metrics: Loss = {baseline_metrics['loss']:.4f}, Perplexity = {baseline_metrics['perplexity']:.2f}")
    
    # Generate baseline text samples
    logger.info("Generating baseline text samples...")
    baseline_samples = generate_text_samples(
        model=model,
        tokenizer=tokenizer,
        prompts=sample_prompts,
        device=device
    )
    
    # Save baseline samples
    with open(dirs["inference"] / "baseline_samples.txt", "w") as f:
        for i, (prompt, sample) in enumerate(zip(sample_prompts, baseline_samples)):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Generated: {sample}\n\n")
    
    # Store baseline metrics
    experiment_metrics["baseline_metrics"] = baseline_metrics
    experiment_metrics["inference"]["baseline"] = baseline_samples
    
    # PHASE 1: WARMUP TRAINING
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: WARMUP TRAINING")
    logger.info("="*80)
    
    logger.info("Starting warmup phase...")
    warmup_results = run_warmup_phase(
        model=model,
        train_dataloader=train_dataloader,
        max_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        patience=args.warmup_patience,
        device=device,
        verbose=True,
        save_visualizations=True,
        output_dir=dirs["warmup"]
    )
    
    # Store warmup metrics
    experiment_metrics["warmup"]["losses"] = warmup_results.get("losses", [])
    experiment_metrics["warmup"]["steps"] = list(range(len(experiment_metrics["warmup"]["losses"])))
    experiment_metrics["warmup"]["stabilization_point"] = warmup_results.get("stabilization_point")
    
    # Add phase to timeline
    experiment_metrics["phases"].append({
        "name": "warmup",
        "start_step": 0,
        "end_step": len(experiment_metrics["warmup"]["losses"]),
        "stabilization_point": warmup_results.get("stabilization_point")
    })
    
    # Visualize warmup results
    reporter.display_warmup_results(warmup_results)
    
    # Evaluate after warmup
    logger.info("Evaluating model after warmup...")
    warmup_metrics = evaluate_model(
        model=model,
        dataloader=validation_dataloader,
        device=device,
        max_eval_steps=args.eval_steps
    )
    logger.info(f"After warmup metrics: Loss = {warmup_metrics['loss']:.4f}, Perplexity = {warmup_metrics['perplexity']:.2f}")
    
    # Generate text samples after warmup
    logger.info("Generating text samples after warmup...")
    warmup_samples = generate_text_samples(
        model=model,
        tokenizer=tokenizer,
        prompts=sample_prompts,
        device=device
    )
    
    # Save warmup samples
    with open(dirs["inference"] / "warmup_samples.txt", "w") as f:
        for i, (prompt, sample) in enumerate(zip(sample_prompts, warmup_samples)):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Generated: {sample}\n\n")
    
    # Store warmup inference results
    experiment_metrics["inference"]["after_warmup"] = warmup_samples
    
    # PHASE 2: PRUNING
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: PRUNING")
    logger.info("="*80)
    
    # Let's calculate head importance metrics
    logger.info("Calculating head importance metrics...")
    importance_metrics = NeuralPlasticity.calculate_head_importance(
        model=model,
        dataloader=train_dataloader,
        num_batches=5,
        mode=args.pruning_strategy
    )
    
    entropy_values = importance_metrics["entropy"]
    grad_norm_values = importance_metrics["gradients"]
    
    # Generate pruning mask
    logger.info(f"Generating pruning mask with {args.pruning_strategy} strategy at {args.pruning_level:.1%} level...")
    pruning_mask = generate_pruning_mask(
        grad_norm_values=grad_norm_values,
        entropy_values=entropy_values,
        prune_percent=args.pruning_level,
        strategy=args.pruning_strategy
    )
    
    # Visualize pruning decisions
    logger.info("Visualizing pruning decisions...")
    reporter.visualize_pruning_decisions(
        grad_norm_values=grad_norm_values,
        entropy_values=entropy_values,
        pruning_mask=pruning_mask,
        save_path=dirs["pruning"] / "pruning_decisions.png"
    )
    
    # Apply pruning
    logger.info("Applying pruning to model...")
    pruned_heads = apply_pruning_mask(model, pruning_mask)
    
    # Count pruned heads
    pruned_count = len(pruned_heads)
    total_heads = num_layers * num_heads
    pruning_rate = pruned_count / total_heads
    
    logger.info(f"Pruned {pruned_count} out of {total_heads} heads ({pruning_rate:.2%})")
    
    # Save pruned heads information
    with open(dirs["pruning"] / "pruned_heads.json", "w") as f:
        json.dump({"pruned_heads": [(int(l), int(h)) for l, h in pruned_heads]}, f, indent=2)
    
    # Store pruning information
    experiment_metrics["pruning"]["pruned_heads"] = [(int(l), int(h)) for l, h in pruned_heads]
    
    # Evaluate immediately after pruning
    logger.info("Evaluating model immediately after pruning...")
    post_pruning_metrics = evaluate_model(
        model=model,
        dataloader=validation_dataloader,
        device=device,
        max_eval_steps=args.eval_steps
    )
    logger.info(f"After pruning metrics: Loss = {post_pruning_metrics['loss']:.4f}, Perplexity = {post_pruning_metrics['perplexity']:.2f}")
    
    # Generate text samples after pruning
    logger.info("Generating text samples after pruning...")
    pruning_samples = generate_text_samples(
        model=model,
        tokenizer=tokenizer,
        prompts=sample_prompts,
        device=device
    )
    
    # Save pruning samples
    with open(dirs["inference"] / "pruning_samples.txt", "w") as f:
        for i, (prompt, sample) in enumerate(zip(sample_prompts, pruning_samples)):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Generated: {sample}\n\n")
    
    # Store pruning metrics and inference results
    experiment_metrics["pruning"]["metrics_before"] = warmup_metrics
    experiment_metrics["pruning"]["metrics_after"] = post_pruning_metrics
    experiment_metrics["inference"]["after_pruning"] = pruning_samples
    
    # Add phase to timeline
    last_step = experiment_metrics["phases"][-1]["end_step"]
    experiment_metrics["phases"].append({
        "name": "pruning",
        "start_step": last_step,
        "end_step": last_step + 1,
        "pruned_heads": pruned_count,
        "pruning_rate": pruning_rate
    })
    
    # PHASE 3: FINE-TUNING
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: FINE-TUNING")
    logger.info("="*80)
    
    # Fine-tune the pruned model
    logger.info(f"Fine-tuning model for {args.training_steps} steps...")
    
    try:
        fine_tuning_results = NeuralPlasticity.train_pruned_model(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=validation_dataloader,
            pruned_heads=pruned_heads,
            learning_rate=args.learning_rate / 2,  # Lower learning rate for fine-tuning
            steps=args.training_steps
        )
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        # Create fallback results
        fine_tuning_results = {
            "train_losses": [post_pruning_metrics["loss"]],
            "eval_losses": [post_pruning_metrics["loss"]],
            "perplexities": [post_pruning_metrics["perplexity"]],
            "steps": [0]
        }
        logger.warning("Using fallback fine-tuning results due to error")
    
    # Store fine-tuning metrics
    experiment_metrics["fine_tuning"]["losses"] = fine_tuning_results.get("train_losses", [])
    experiment_metrics["fine_tuning"]["steps"] = list(range(len(experiment_metrics["fine_tuning"]["losses"])))
    
    # Add phase to timeline
    last_step = experiment_metrics["phases"][-1]["end_step"]
    experiment_metrics["phases"].append({
        "name": "fine_tuning",
        "start_step": last_step,
        "end_step": last_step + len(experiment_metrics["fine_tuning"]["losses"])
    })
    
    # Final evaluation
    logger.info("Performing final evaluation...")
    final_metrics = evaluate_model(
        model=model,
        dataloader=validation_dataloader,
        device=device,
        max_eval_steps=args.eval_steps
    )
    logger.info(f"Final metrics: Loss = {final_metrics['loss']:.4f}, Perplexity = {final_metrics['perplexity']:.2f}")
    
    # Generate final text samples
    logger.info("Generating final text samples...")
    final_samples = generate_text_samples(
        model=model,
        tokenizer=tokenizer,
        prompts=sample_prompts,
        device=device
    )
    
    # Save final samples
    with open(dirs["inference"] / "final_samples.txt", "w") as f:
        for i, (prompt, sample) in enumerate(zip(sample_prompts, final_samples)):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Generated: {sample}\n\n")
    
    # Store final metrics and inference results
    experiment_metrics["final_metrics"] = final_metrics
    experiment_metrics["inference"]["after_fine_tuning"] = final_samples
    
    # Calculate improvements
    loss_improvement = (baseline_metrics['loss'] - final_metrics['loss']) / baseline_metrics['loss'] * 100
    perplexity_improvement = (baseline_metrics['perplexity'] - final_metrics['perplexity']) / baseline_metrics['perplexity'] * 100
    
    experiment_metrics["improvements"] = {
        "loss": loss_improvement,
        "perplexity": perplexity_improvement
    }
    
    logger.info("\n" + "="*50)
    logger.info("EXPERIMENT RESULTS")
    logger.info("="*50)
    logger.info(f"Baseline metrics: Loss = {baseline_metrics['loss']:.4f}, Perplexity = {baseline_metrics['perplexity']:.2f}")
    logger.info(f"After warmup: Loss = {warmup_metrics['loss']:.4f}, Perplexity = {warmup_metrics['perplexity']:.2f}")
    logger.info(f"After pruning: Loss = {post_pruning_metrics['loss']:.4f}, Perplexity = {post_pruning_metrics['perplexity']:.2f}")
    logger.info(f"Final metrics: Loss = {final_metrics['loss']:.4f}, Perplexity = {final_metrics['perplexity']:.2f}")
    logger.info(f"Improvements: Loss = {loss_improvement:.2f}%, Perplexity = {perplexity_improvement:.2f}%")
    
    # Save model and tokenizer
    logger.info("Saving model and tokenizer...")
    model.save_pretrained(dirs["model"])
    tokenizer.save_pretrained(dirs["model"])
    
    # Save all metrics
    with open(dirs["metrics"] / "experiment_metrics.json", "w") as f:
        # Convert any tensor values to python types
        serializable_metrics = {}
        for key, value in experiment_metrics.items():
            if isinstance(value, dict):
                serializable_metrics[key] = {}
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        serializable_metrics[key][k] = v.cpu().numpy().tolist()
                    elif isinstance(v, np.ndarray):
                        serializable_metrics[key][k] = v.tolist()
                    else:
                        serializable_metrics[key][k] = v
            elif isinstance(value, torch.Tensor):
                serializable_metrics[key] = value.cpu().numpy().tolist()
            elif isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            else:
                serializable_metrics[key] = value
                
        json.dump(serializable_metrics, f, indent=2)
    
    # Generate comprehensive visualizations
    logger.info("\nGenerating comprehensive visualizations...")
    
    try:
        # Generate complete process dashboard
        logger.info("Generating complete process dashboard...")
        try:
            NeuralPlasticity.visualize_complete_process(
                experiment=experiment_metrics,
                output_dir=dirs["visualizations"],
                show_plot=False
            )
            logger.info("✓ Visualization created successfully")
        except Exception as e:
            logger.error(f"Error generating process visualization: {e}")
            # Create minimal visualization with matplotlib directly
            try:
                # Create a simple visualization as fallback
                plt.figure(figsize=(12, 6))
                
                # Plot warmup losses if available
                if experiment_metrics["warmup"]["losses"]:
                    plt.subplot(2, 2, 1)
                    plt.plot(experiment_metrics["warmup"]["losses"])
                    plt.title("Warmup Loss")
                    plt.xlabel("Steps")
                    plt.ylabel("Loss")
                
                # Plot fine-tuning losses if available
                if experiment_metrics["fine_tuning"]["losses"]:
                    plt.subplot(2, 2, 2)
                    plt.plot(experiment_metrics["fine_tuning"]["losses"])
                    plt.title("Fine-tuning Loss")
                    plt.xlabel("Steps")
                    plt.ylabel("Loss")
                
                # Save figure
                plt.tight_layout()
                plt.savefig(dirs["visualizations"] / "simple_visualization.png")
                logger.info(f"✓ Created simple visualization as fallback")
            except Exception as viz_err:
                logger.error(f"Error creating fallback visualization: {viz_err}")
        
        # Generate full dashboard
        if args.generate_html:
            logger.info("Generating HTML dashboard...")
            
            try:
                dashboard_path = NeuralPlasticity.generate_dashboards(
                    experiment=experiment_metrics,
                    output_dir=dirs["html"]
                )
                logger.info(f"✓ Generated HTML dashboard at: {dirs['html']}")
            except Exception as e:
                logger.error(f"Error generating HTML dashboard: {e}")
                
                # Create a simple HTML file as fallback
                try:
                    simple_html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Neural Plasticity Experiment Results</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 20px; }}
                            h1 {{ color: #2c3e50; }}
                            .metrics {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
                            .improvement {{ color: green; }}
                        </style>
                    </head>
                    <body>
                        <h1>Neural Plasticity Experiment Results</h1>
                        <p>Model: {args.model_name}</p>
                        <p>Pruning strategy: {args.pruning_strategy} at {args.pruning_level*100:.1f}% level</p>
                        
                        <div class="metrics">
                            <h2>Metrics:</h2>
                            <p>Baseline loss: {baseline_metrics['loss']:.4f}, Perplexity: {baseline_metrics['perplexity']:.2f}</p>
                            <p>After pruning loss: {post_pruning_metrics['loss']:.4f}, Perplexity: {post_pruning_metrics['perplexity']:.2f}</p>
                            <p>Final loss: {final_metrics['loss']:.4f}, Perplexity: {final_metrics['perplexity']:.2f}</p>
                            <p>Improvements: Loss <span class="improvement">{loss_improvement:.2f}%</span>, 
                               Perplexity <span class="improvement">{perplexity_improvement:.2f}%</span></p>
                        </div>
                        
                        <h2>Pruning Information:</h2>
                        <p>Pruned {pruned_count} out of {total_heads} heads ({pruning_rate:.2%})</p>
                        
                        <p><em>Generated at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</em></p>
                    </body>
                    </html>
                    """
                    
                    with open(dirs["html"] / "simple_results.html", "w") as f:
                        f.write(simple_html)
                    logger.info(f"✓ Created simple HTML report as fallback")
                except Exception as html_err:
                    logger.error(f"Error creating simple HTML report: {html_err}")
    except Exception as e:
        logger.error(f"Error in visualization phase: {e}")
    
    # Calculate total runtime
    end_time = time.time()
    runtime_seconds = end_time - start_time
    hours, remainder = divmod(runtime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info("\n" + "="*80)
    logger.info(f"EXPERIMENT COMPLETED")
    logger.info("="*80)
    logger.info(f"Total runtime: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"HTML dashboard: {dirs['html'] / 'neural_plasticity_dashboard.html'}")
    
    return output_dir


def main():
    """Main function."""
    # Record start time for overall script
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_args()
    
    # Make the output directory path absolute if needed
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(args.output_dir)
    
    # Create minimal output directory structure in case of early failure
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
        
        # Add file handler to logger
        log_file = os.path.join(args.output_dir, "logs", f"experiment_{datetime.now().strftime(DATE_FORMAT)}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not set up logging to file: {e}")
    
    # Display script header
    print(f"\n{'='*80}")
    print(f"NEURAL PLASTICITY EXPERIMENT {VERSION}")
    print(f"{'='*80}")
    print(f"Output directory: {args.output_dir}")
    
    # Run experiment
    try:
        output_dir = run_experiment(args)
        
        # Record overall execution time
        end_time = time.time()
        total_time = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if output_dir:
            print(f"\n{'='*80}")
            print(f"EXPERIMENT COMPLETED SUCCESSFULLY!")
            print(f"{'='*80}")
            print(f"Total execution time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
            print(f"Results saved to: {output_dir}")
            
            # Check if HTML dashboard exists and provide appropriate message
            html_path = os.path.join(output_dir, "html", "neural_plasticity_dashboard.html")
            simple_html_path = os.path.join(output_dir, "html", "simple_results.html")
            
            if os.path.exists(html_path):
                print(f"HTML dashboard: {html_path}")
            elif os.path.exists(simple_html_path):
                print(f"Simple HTML report: {simple_html_path}")
            else:
                print(f"No HTML dashboard was generated - check logs for errors")
                
            print(f"\nTo run this experiment on Colab:")
            print(f"1. Upload this script to Colab")
            print(f"2. Run with: !python run_real_neural_plasticity.py --device cuda")
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        logger.warning("Experiment interrupted by user")
    except Exception as e:
        # Record error execution time
        end_time = time.time()
        total_time = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\n{'='*80}")
        print(f"ERROR RUNNING EXPERIMENT: {str(e)}")
        print(f"{'='*80}")
        print(f"Execution time before error: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        print(f"Check logs for details: {args.output_dir}/logs/")
        
        logger.error(f"Error running experiment: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()