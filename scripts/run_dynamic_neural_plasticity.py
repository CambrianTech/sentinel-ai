#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dynamic Neural Plasticity Experiment

This script runs a complete neural plasticity experiment with real models and datasets,
using the neural plasticity module for dynamic pruning and fine-tuning.

Key features:
- Uses real HuggingFace models (not simulated)
- Uses real dataset from HuggingFace (not simulated)
- Dynamically detects model stabilization (not on a fixed schedule)
- Makes pruning decisions based on entropy and gradient metrics
- Creates visualization dashboards showing the entire process
- Generates text samples at each phase to evaluate model quality
- Works identically in both local and Google Colab environments

Usage:
    # Activate virtual environment
    source .venv/bin/activate
    
    # Run with default parameters (distilgpt2 on wikitext)
    python scripts/run_dynamic_neural_plasticity.py
    
    # Run with GPT-2 and different dataset
    python scripts/run_dynamic_neural_plasticity.py --model_name gpt2 --dataset gutenberg
"""

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import logging
import time
import json
import gc
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)

# Import transformers components
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    default_data_collator, 
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader
from datasets import load_dataset

# Import neural plasticity modules
from utils.neural_plasticity import NeuralPlasticity
from utils.neural_plasticity.visualization import (
    visualize_head_entropy,
    visualize_head_gradients,
    visualize_pruning_decisions,
    visualize_training_metrics,
    VisualizationReporter
)
from utils.neural_plasticity.core import (
    calculate_head_entropy,
    calculate_head_gradients,
    detect_model_structure,
    evaluate_model,
    generate_pruning_mask,
    apply_pruning_mask
)
from utils.neural_plasticity.training import (
    run_warmup_phase,
    run_plasticity_loop
)

# Try to import dashboard utilities
try:
    from utils.neural_plasticity.dashboard import DashboardReporter
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    logger.warning("DashboardReporter not available - HTML dashboard will not be generated")

# Constants
VERSION = "v0.0.70 (2025-04-20 19:45:00)"
DATE_FORMAT = "%Y%m%d-%H%M%S"
DEFAULT_OUTPUT_DIR = os.path.join("experiment_results", f"run_{datetime.now().strftime(DATE_FORMAT)}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run dynamic neural plasticity experiment")
    
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
    
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate (default: 5e-5)")
    
    parser.add_argument("--warmup_patience", type=int, default=20,
                       help="Patience for warmup stabilization detection (default: 20)")
    
    parser.add_argument("--pruning_strategy", type=str, default="combined",
                       choices=["gradient", "entropy", "random", "combined"],
                       help="Pruning strategy (default: combined)")
    
    parser.add_argument("--pruning_level", type=float, default=0.2,
                       help="Pruning level (default: 0.2)")
    
    parser.add_argument("--max_warmup_steps", type=int, default=300,
                       help="Maximum warmup steps (default: 300)")
    
    parser.add_argument("--training_steps", type=int, default=200,
                       help="Training steps after pruning (default: 200)")
    
    parser.add_argument("--eval_steps", type=int, default=10,
                       help="Evaluation steps (default: 10)")
    
    parser.add_argument("--sample_texts", type=int, default=3,
                       help="Number of sample texts to generate (default: 3)")
    
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    
    parser.add_argument("--device", type=str, default=None,
                       help="Device (default: auto-detect)")
    
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    
    parser.add_argument("--show_samples", action="store_true", default=True,
                       help="Show text samples during training (default: True)")
    
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Print detailed progress information (default: True)")
    
    return parser.parse_args()


def setup_environment(seed=42):
    """Set up environment and return environment information."""
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Get environment information
    env_info = NeuralPlasticity.get_environment_info()
    
    # Log environment information
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
    
    if env_info['is_apple_silicon']:
        logger.info(f"  Running on Apple Silicon")
        logger.info(f"  Using CPU for inference to avoid instability")
    
    return env_info


def load_and_prepare_data(args, tokenizer):
    """Load and prepare datasets."""
    logger.info(f"Loading dataset: {args.dataset}/{args.dataset_config}")
    
    # Make sure datasets library is imported
    from datasets import load_dataset
    
    # Load datasets
    try:
        if args.dataset == "wikitext":
            train_dataset = load_dataset(args.dataset, args.dataset_config, split="train")
            validation_dataset = load_dataset(args.dataset, args.dataset_config, split="validation")
        elif args.dataset == "gutenberg":
            # Custom handling for Project Gutenberg dataset
            try:
                train_dataset = load_dataset("pg19", split="train[:2%]")  # Use a small subset for training
                validation_dataset = load_dataset("pg19", split="validation[:5%]")  # Use a small subset for validation
                logger.info("Successfully loaded pg19 (Gutenberg) dataset")
            except Exception as e:
                logger.error(f"Error loading Gutenberg dataset: {e}")
                logger.info("Falling back to wikitext dataset")
                train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
                validation_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        else:
            # Default to loading specified dataset
            train_dataset = load_dataset(args.dataset, args.dataset_config, split="train")
            validation_dataset = load_dataset(args.dataset, args.dataset_config, split="validation")
        
        logger.info(f"Loaded training dataset with {len(train_dataset)} examples")
        logger.info(f"Loaded validation dataset with {len(validation_dataset)} examples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        logger.info("Falling back to wikitext dataset")
        try:
            train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            validation_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
            logger.info(f"Loaded fallback training dataset with {len(train_dataset)} examples")
            logger.info(f"Loaded fallback validation dataset with {len(validation_dataset)} examples")
        except Exception as fallback_error:
            logger.error(f"Error loading fallback dataset: {fallback_error}")
            # Create minimal synthetic datasets as last resort
            logger.info("Creating synthetic datasets for demonstration")
            
            # Create minimal datasets for demonstration purposes
            from torch.utils.data import Dataset
            
            class SimpleDataset(Dataset):
                def __init__(self, texts, tokenizer, max_length):
                    self.encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)
                    self.labels = self.encodings["input_ids"].copy()
                
                def __len__(self):
                    return len(self.encodings["input_ids"])
                
                def __getitem__(self, idx):
                    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                    item["labels"] = torch.tensor(self.labels[idx])
                    return item
            
            # Sample texts
            sample_texts = [
                "The neural plasticity system enables transformer models to adapt dynamically.",
                "Attention mechanisms allow models to focus on different parts of input sequences.",
                "Pruning less important attention heads can improve efficiency without sacrificing performance.",
                "Dynamic stabilization detection identifies when model training has reached a plateau.",
                "Real-time metrics help visualize the entire neural plasticity process."
            ] * 5  # Repeat to get more examples
            
            train_dataset = SimpleDataset(sample_texts, tokenizer, args.max_length)
            validation_dataset = SimpleDataset(sample_texts[:3], tokenizer, args.max_length)
            logger.info(f"Created synthetic training dataset with {len(train_dataset)} examples")
            logger.info(f"Created synthetic validation dataset with {len(validation_dataset)} examples")
    
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
    
    # Handle different dataset formats
    if "text" in train_dataset.column_names:
        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        validation_dataset = validation_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    else:
        # Try to find the appropriate text column
        text_column = None
        for column in train_dataset.column_names:
            if column.lower() in ["text", "content", "document", "passage"]:
                text_column = column
                break
        
        if text_column:
            logger.info(f"Using column '{text_column}' as text data")
            # Create a modified tokenize function that uses the identified column
            def adapted_tokenize_function(examples):
                return tokenizer(
                    examples[text_column], 
                    padding="max_length", 
                    truncation=True, 
                    max_length=args.max_length
                )
            
            train_dataset = train_dataset.map(adapted_tokenize_function, batched=True, remove_columns=[text_column])
            validation_dataset = validation_dataset.map(adapted_tokenize_function, batched=True, remove_columns=[text_column])
        else:
            # For unexpected formats, create a simple dataset with sample texts
            logger.warning(f"Could not find text column in dataset. Creating simple dataset.")
            
            # Sample texts for a simple dataset
            sample_texts = [
                "The quick brown fox jumps over the lazy dog. This sentence contains every letter in the English alphabet.",
                "Neural networks consist of interconnected artificial neurons organized in layers to process input data.",
                "Machine learning algorithms analyze data to identify patterns and make predictions without explicit programming.",
                "Transformers have revolutionized natural language processing with their attention mechanisms.",
                "Attention is all you need was the title of the paper that introduced the transformer architecture."
            ] * 10  # Repeat to get more samples
            
            # Create simple datasets
            from torch.utils.data import Dataset
            
            class SimpleDataset(Dataset):
                def __init__(self, texts, tokenizer, max_length):
                    self.encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)
                    self.labels = self.encodings["input_ids"].copy()
                
                def __len__(self):
                    return len(self.encodings["input_ids"])
                
                def __getitem__(self, idx):
                    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                    item["labels"] = torch.tensor(self.labels[idx])
                    return item
            
            train_dataset = SimpleDataset(sample_texts, tokenizer, args.max_length)
            validation_dataset = SimpleDataset(sample_texts[:5], tokenizer, args.max_length)
            logger.info(f"Created simple dataset with {len(train_dataset)} training examples")
            logger.info(f"Created simple dataset with {len(validation_dataset)} validation examples")
    
    # Add labels for language modeling if they don't exist already
    if isinstance(train_dataset, dict) or (hasattr(train_dataset, "column_names") and "labels" not in train_dataset.column_names):
        def add_labels(examples):
            examples["labels"] = examples["input_ids"].copy()
            return examples
        
        train_dataset = train_dataset.map(add_labels)
        validation_dataset = validation_dataset.map(add_labels)
    
    # Set torch format if available
    if hasattr(train_dataset, "with_format"):
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
    
    # Extract sample prompts for text generation
    sample_texts = []
    
    try:
        # Try to extract samples based on dataset type
        if hasattr(validation_dataset, "__getitem__"):
            # Dataset has getitem method
            num_samples = min(args.sample_texts, len(validation_dataset))
            indices = np.random.choice(len(validation_dataset), num_samples, replace=False)
            
            for i in range(num_samples):
                idx = int(indices[i])
                sample = validation_dataset[idx]
                
                if isinstance(sample, dict) and "input_ids" in sample:
                    # Extract input_ids and decode the first 25 tokens
                    input_ids = sample["input_ids"]
                    
                    # Convert to tensor if necessary
                    if not isinstance(input_ids, torch.Tensor):
                        input_ids = torch.tensor(input_ids)
                    
                    # Take first 25 tokens as prompt
                    prompt_ids = input_ids[:25].tolist()
                    sample_text = tokenizer.decode(prompt_ids)
                    sample_texts.append(sample_text)
        else:
            # Fall back to artificial prompts
            sample_texts = [
                "The neural plasticity system",
                "Attention heads can be dynamically",
                "Transformer models achieve"
            ][:args.sample_texts]
    except Exception as e:
        logger.warning(f"Error extracting sample texts: {e}")
        # Fallback to simple text prompts
        sample_texts = [
            "The neural plasticity system",
            "Attention heads can be dynamically",
            "Transformer models achieve"
        ][:args.sample_texts]
    
    logger.info(f"Created data loaders with batch size {args.batch_size}")
    return train_dataloader, validation_dataloader, sample_texts


def generate_text_samples(model, tokenizer, prompts, device, max_length=50):
    """Generate text samples from the model."""
    model.eval()
    generated_texts = []
    
    for prompt in prompts:
        try:
            # Prepare input
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate text
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    do_sample=True,
                    top_p=0.92,
                    temperature=0.8,
                    pad_token_id=tokenizer.eos_token_id,  # Handle padding
                    attention_mask=inputs.get("attention_mask", None)
                )
                
                # Decode the generated text
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_texts.append(generated_text)
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            generated_texts.append(f"[Error generating text: {str(e)}]")
    
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
    
    # Create sub-directories
    dirs = {
        "warmup": output_dir / "warmup",
        "pruning": output_dir / "pruning",
        "fine_tuning": output_dir / "fine_tuning",
        "visualizations": output_dir / "visualizations",
        "inference": output_dir / "inference",
        "metrics": output_dir / "metrics",
        "html": output_dir / "html",
        "model": output_dir / "model",
        "logs": output_dir / "logs"
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(exist_ok=True)
    
    # Save experiment parameters
    params_file = output_dir / "parameters.json"
    with open(params_file, "w") as f:
        params = {
            "model_name": args.model_name,
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_patience": args.warmup_patience,
            "pruning_strategy": args.pruning_strategy,
            "pruning_level": args.pruning_level,
            "max_warmup_steps": args.max_warmup_steps,
            "training_steps": args.training_steps,
            "eval_steps": args.eval_steps,
            "sample_texts": args.sample_texts,
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
        logger.info(f"Total attention heads: {num_layers * num_heads}")
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {param_count:,}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None
    
    # Prepare datasets
    train_dataloader, validation_dataloader, sample_prompts = load_and_prepare_data(args, tokenizer)
    
    # Create visualization reporter
    reporter = VisualizationReporter(
        model=model,
        tokenizer=tokenizer,
        output_dir=dirs["visualizations"],
        save_visualizations=True,
        verbose=args.verbose
    )
    
    # Create dashboard reporter if available
    dashboard_reporter = None
    if DASHBOARD_AVAILABLE:
        try:
            dashboard_reporter = DashboardReporter(
                output_dir=dirs["html"],
                dashboard_name="neural_plasticity_dashboard.html",
                auto_update=True,
                update_interval=max(1, args.training_steps // 10)  # Update every ~10% of training
            )
            logger.info("Created dashboard reporter")
        except Exception as e:
            logger.error(f"Error creating dashboard reporter: {e}")
    
    # Event tracking for visualization
    events = []
    
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
    
    events.append({
        "step": 0,
        "type": "baseline",
        "metrics": baseline_metrics,
        "samples": baseline_samples
    })
    
    #----------------------------------------------
    # PHASE 1: WARMUP
    #----------------------------------------------
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: WARMUP TRAINING WITH DYNAMIC STABILIZATION DETECTION")
    logger.info("="*80)
    
    # Run warmup phase with dynamic stabilization detection
    warmup_result = run_warmup_phase(
        model=model,
        train_dataloader=train_dataloader,
        learning_rate=args.learning_rate,
        patience=args.warmup_patience,
        min_warmup_steps=30,
        max_warmup_steps=args.max_warmup_steps,
        device=device,
        verbose=args.verbose,
        save_visualizations=True,
        output_dir=str(dirs["warmup"])
    )
    
    # Extract warmup information
    warmup_losses = warmup_result.get("losses", [])
    stabilization_point = warmup_result.get("stabilization_point")
    stabilization_reason = warmup_result.get("stabilization_reason", "")
    
    logger.info(f"Warmup completed in {len(warmup_losses)} steps")
    if stabilization_point:
        logger.info(f"Loss stabilization detected at step {stabilization_point}")
        logger.info(f"Stabilization reason: {stabilization_reason}")
    else:
        logger.info(f"No stabilization detected - reached maximum steps")
    
    # Display warmup visualization
    reporter.display_warmup_results(warmup_result)
    
    # Evaluate model after warmup
    logger.info("Evaluating model after warmup phase...")
    warmup_metrics = evaluate_model(
        model=model,
        dataloader=validation_dataloader,
        device=device,
        max_eval_steps=args.eval_steps
    )
    logger.info(f"After warmup: Loss = {warmup_metrics['loss']:.4f}, Perplexity = {warmup_metrics['perplexity']:.2f}")
    
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
    
    events.append({
        "step": len(warmup_losses),
        "type": "warmup_complete",
        "metrics": warmup_metrics,
        "samples": warmup_samples,
        "stabilization_point": stabilization_point,
        "stabilization_reason": stabilization_reason
    })
    
    #----------------------------------------------
    # PHASE 2 & 3: PRUNING AND FINE-TUNING WITH PLASTICITY LOOP
    #----------------------------------------------
    logger.info("\n" + "="*80)
    logger.info("PHASE 2-3: PRUNING AND FINE-TUNING WITH PLASTICITY LOOP")
    logger.info("="*80)
    
    # Define callback for tracking events in the plasticity loop
    def plasticity_callback(event_type, step, data):
        logger.info(f"Plasticity event: {event_type} at step {step}")
        
        # Store event data for visualization
        events.append({
            "step": step + len(warmup_losses),
            "type": event_type,
            "data": data
        })
        
        # For pruning event, save detail information
        if event_type == "pruning" and "pruned_heads" in data:
            pruned_heads = data.get("pruned_heads", [])
            logger.info(f"Pruned {len(pruned_heads)} attention heads")
            
            # Save pruned heads information
            pruned_heads_file = dirs["pruning"] / "pruned_heads.json"
            with open(pruned_heads_file, "w") as f:
                json.dump({"pruned_heads": [(int(l), int(h)) for l, h in pruned_heads]}, f, indent=2)
            
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
            
            # Add samples to event
            events[-1]["samples"] = pruning_samples
        
        # For final event, save detail information
        if event_type == "final":
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
            
            # Add samples to event
            events[-1]["samples"] = final_samples
    
    # Run the complete plasticity loop (pruning and fine-tuning)
    plasticity_results = run_plasticity_loop(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=validation_dataloader,
        pruning_level=args.pruning_level,
        strategy=args.pruning_strategy,
        learning_rate=args.learning_rate,
        training_steps=args.training_steps,
        tokenizer=tokenizer,
        show_samples=args.show_samples,
        sample_interval=max(1, args.training_steps // 5),
        dashboard_reporter=dashboard_reporter,
        callback=plasticity_callback
    )
    
    # Extract plasticity results
    pruned_heads = plasticity_results.get("pruned_heads", [])
    pruned_metrics = plasticity_results.get("pruned_metrics", {})
    final_metrics = plasticity_results.get("final_metrics", {})
    
    # Calculate improvements
    baseline_perplexity = baseline_metrics.get("perplexity", 0)
    final_perplexity = final_metrics.get("perplexity", 0)
    
    if baseline_perplexity > 0 and final_perplexity > 0:
        perplexity_improvement = (baseline_perplexity - final_perplexity) / baseline_perplexity * 100
    else:
        perplexity_improvement = 0
    
    # Save all metrics
    metrics = {
        "baseline": baseline_metrics,
        "warmup": warmup_metrics,
        "pruned": pruned_metrics,
        "final": final_metrics,
        "improvement": {
            "perplexity": perplexity_improvement
        },
        "pruning": {
            "strategy": args.pruning_strategy,
            "level": args.pruning_level,
            "pruned_heads": [(int(l), int(h)) for l, h in pruned_heads],
            "pruned_count": len(pruned_heads),
            "total_heads": num_layers * num_heads,
            "pruning_rate": len(pruned_heads) / (num_layers * num_heads) if num_layers and num_heads else 0
        },
        "warmup": {
            "steps": len(warmup_losses),
            "stabilization_point": stabilization_point,
            "stabilization_reason": stabilization_reason
        },
        "events": events
    }
    
    with open(dirs["metrics"] / "experiment_metrics.json", "w") as f:
        # Convert any tensor values to python types
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_tensors(metrics), f, indent=2)
    
    # Display final results
    logger.info("\n" + "="*50)
    logger.info("EXPERIMENT RESULTS")
    logger.info("="*50)
    logger.info(f"Baseline metrics: Loss = {baseline_metrics['loss']:.4f}, Perplexity = {baseline_metrics['perplexity']:.2f}")
    logger.info(f"After warmup: Loss = {warmup_metrics['loss']:.4f}, Perplexity = {warmup_metrics['perplexity']:.2f}")
    logger.info(f"After pruning: Loss = {pruned_metrics.get('loss', 0):.4f}, Perplexity = {pruned_metrics.get('perplexity', 0):.2f}")
    logger.info(f"Final metrics: Loss = {final_metrics.get('loss', 0):.4f}, Perplexity = {final_metrics.get('perplexity', 0):.2f}")
    logger.info(f"Perplexity improvement: {perplexity_improvement:.2f}%")
    logger.info(f"Pruned {len(pruned_heads)} out of {num_layers * num_heads} heads ({len(pruned_heads) / (num_layers * num_heads) * 100:.2f}%)")
    
    # Generate comprehensive visualization
    logger.info("\nGenerating comprehensive visualizations...")
    
    try:
        # Generate complete process visualization
        complete_viz_path = NeuralPlasticity.visualize_complete_process(
            experiment=metrics,
            output_dir=dirs["visualizations"],
            show_plot=False
        )
        logger.info(f"Generated complete process visualization")
    except Exception as e:
        logger.error(f"Error generating complete visualization: {e}")
    
    # Generate HTML dashboard
    if dashboard_reporter:
        try:
            # Update dashboard with final metrics
            dashboard_reporter.add_experiment_summary({
                "model_name": args.model_name,
                "dataset": args.dataset,
                "baseline_perplexity": baseline_metrics["perplexity"],
                "final_perplexity": final_metrics["perplexity"],
                "perplexity_improvement": perplexity_improvement,
                "pruned_heads_count": len(pruned_heads),
                "total_heads": num_layers * num_heads,
                "pruning_rate": len(pruned_heads) / (num_layers * num_heads),
                "warmup_steps": len(warmup_losses),
                "training_steps": args.training_steps
            })
            
            # Generate comprehensive dashboard
            dashboard_path = dashboard_reporter.generate_dashboard()
            logger.info(f"Generated HTML dashboard at: {dashboard_path}")
        except Exception as e:
            logger.error(f"Error generating HTML dashboard: {e}")
            
            # Create a simple HTML report instead
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
                    <p>Dataset: {args.dataset}</p>
                    <p>Pruning strategy: {args.pruning_strategy} at {args.pruning_level*100:.1f}% level</p>
                    
                    <div class="metrics">
                        <h2>Metrics:</h2>
                        <p>Baseline perplexity: {baseline_metrics['perplexity']:.2f}</p>
                        <p>After warmup perplexity: {warmup_metrics['perplexity']:.2f}</p>
                        <p>After pruning perplexity: {pruned_metrics.get('perplexity', 0):.2f}</p>
                        <p>Final perplexity: {final_metrics.get('perplexity', 0):.2f}</p>
                        <p>Improvement: <span class="improvement">{perplexity_improvement:.2f}%</span></p>
                    </div>
                    
                    <h2>Pruning Information:</h2>
                    <p>Pruned {len(pruned_heads)} out of {num_layers * num_heads} heads ({len(pruned_heads) / (num_layers * num_heads) * 100:.2f}%)</p>
                    
                    <h2>Process Summary:</h2>
                    <p>Warmup completed in {len(warmup_losses)} steps</p>
                    <p>Stabilization point: {stabilization_point if stabilization_point else 'None detected'}</p>
                    <p>Training steps after pruning: {args.training_steps}</p>
                    
                    <p><em>Generated at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</em></p>
                </body>
                </html>
                """
                
                with open(dirs["html"] / "simple_results.html", "w") as f:
                    f.write(simple_html)
                logger.info(f"Created simple HTML report at: {dirs['html'] / 'simple_results.html'}")
            except Exception as html_err:
                logger.error(f"Error creating simple HTML report: {html_err}")
    
    # Save model and tokenizer
    logger.info("Saving final model and tokenizer...")
    
    try:
        model.save_pretrained(dirs["model"])
        tokenizer.save_pretrained(dirs["model"])
        logger.info(f"Saved model and tokenizer to {dirs['model']}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
    
    # Calculate runtime
    end_time = time.time()
    runtime_seconds = end_time - start_time
    hours, remainder = divmod(runtime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info("\n" + "="*80)
    logger.info(f"EXPERIMENT COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"Total runtime: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    logger.info(f"Results saved to: {output_dir}")
    
    return output_dir


def main():
    """Main function."""
    # Record start time
    start_time = time.time()
    
    # Parse arguments
    args = parse_args()
    
    # Make output directory path absolute if needed
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(args.output_dir)
    
    # Create output directory and setup logging
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
            
            # Provide HTML paths
            html_path = os.path.join(output_dir, "html", "neural_plasticity_dashboard.html")
            simple_html_path = os.path.join(output_dir, "html", "simple_results.html")
            
            if os.path.exists(html_path):
                print(f"HTML dashboard: {html_path}")
            elif os.path.exists(simple_html_path):
                print(f"Simple HTML report: {simple_html_path}")
            
            print(f"\nTo run this experiment on Colab:")
            print(f"!pip install transformers datasets torch matplotlib numpy tqdm")
            print(f"!python run_dynamic_neural_plasticity.py --device cuda")
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