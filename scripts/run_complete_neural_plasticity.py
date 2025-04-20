#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete Neural Plasticity Experiment - End-to-End Implementation

This script runs a complete neural plasticity experiment using REAL models and REAL datasets.
It executes the full process from warmup through pruning to fine-tuning while generating
comprehensive visualizations of all decisions.

The experiment consists of:
1. Loading a real HuggingFace transformer model
2. Preparing a real dataset (wikitext by default)
3. Running a warmup phase with stabilization detection
4. Performing dynamic pruning based on entropy and gradient metrics
5. Fine-tuning the pruned model
6. Generating comprehensive visualizations and dashboards
7. Evaluating the model after each phase with actual text generation

The entire script works identically whether run locally or in Google Colab.

Usage:
    # Activate virtual environment first
    source .venv/bin/activate
    
    # Run with default parameters (distilgpt2 on wikitext)
    python scripts/run_complete_neural_plasticity.py
    
    # Run with custom parameters
    python scripts/run_complete_neural_plasticity.py --model_name gpt2 --dataset wikitext --dataset_config wikitext-2-raw-v1 --num_epochs 2
    
    # Run with a different pruning strategy
    python scripts/run_complete_neural_plasticity.py --strategy combined --pruning_level 0.3
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import neural plasticity modules
from utils.neural_plasticity import NeuralPlasticity
from utils.neural_plasticity.visualization import VisualizationReporter
from utils.neural_plasticity.core import (
    calculate_head_entropy,
    calculate_head_gradients,
    detect_model_structure,
    evaluate_model,
    generate_pruning_mask,
    apply_pruning_mask
)
from utils.neural_plasticity.training import run_warmup_phase, run_plasticity_loop
from utils.colab.visualizations import visualize_complete_training_process

# Constants
DATE_FORMAT = "%Y%m%d-%H%M%S"
DEFAULT_OUTPUT_DIR = os.path.join("experiment_output", f"run_{datetime.now().strftime(DATE_FORMAT)}")
VERSION = "v0.0.5 (2025-04-20 18:30:00)"

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
    
    parser.add_argument("--show_visualizations", action="store_true",
                       help="Show visualizations during run (default: False)")
    
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    return parser.parse_args()


def setup_environment(seed=42):
    """Set up environment for reproducibility."""
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Detect environment
    is_colab = 'google.colab' in sys.modules
    if is_colab:
        print("🌎 Running in Google Colab environment")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 Using GPU: {gpu_name}")
            # Try to display more GPU information
            try:
                import subprocess
                subprocess.run(["nvidia-smi"], check=False)
            except Exception:
                pass
    else:
        print("💻 Running in local environment")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 Using GPU: {gpu_name}")
    
    # Check if running on Apple Silicon
    is_apple_silicon = False
    try:
        import platform
        if platform.system() == "Darwin" and platform.processor() == "arm":
            is_apple_silicon = True
            print("🍎 Running on Apple Silicon")
    except Exception:
        pass
    
    return {
        "is_colab": is_colab,
        "is_apple_silicon": is_apple_silicon,
        "has_gpu": torch.cuda.is_available()
    }


def load_and_prepare_data(args):
    """Load and prepare real datasets."""
    print(f"\n📚 Loading dataset: {args.dataset}/{args.dataset_config}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    train_dataset = load_dataset(args.dataset, args.dataset_config, split="train")
    validation_dataset = load_dataset(args.dataset, args.dataset_config, split="validation")
    
    print(f"✓ Loaded training dataset with {len(train_dataset)} examples")
    print(f"✓ Loaded validation dataset with {len(validation_dataset)} examples")
    
    # Define tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=args.max_length
        )
    
    # Tokenize datasets
    print("Tokenizing datasets...")
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
    
    # Save a few text examples for inference evaluation
    sample_texts = []
    for i, example in enumerate(validation_dataset):
        if i >= 5:  # Just save 5 examples
            break
        # Decode sample text
        sample_text = tokenizer.decode(example["input_ids"][:25])  # First 25 tokens
        sample_texts.append(sample_text)
    
    print(f"✓ Created data loaders with batch size {args.batch_size}")
    
    return tokenizer, train_dataloader, validation_dataloader, sample_texts


def generate_sample_texts(
    model, 
    tokenizer, 
    sample_prompts, 
    max_length=50, 
    num_return_sequences=1,
    device=None
):
    """Generate sample texts using the model."""
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    generated_texts = []
    for prompt in sample_prompts:
        # Tokenize the prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        # Generate text
        with torch.no_grad():
            try:
                output = model.generate(
                    input_ids,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    top_p=0.92,
                    temperature=0.8
                )
                
                # Decode the generated text
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                generated_texts.append(generated_text)
            except Exception as e:
                print(f"Error generating text: {e}")
                generated_texts.append(f"Error: {str(e)}")
    
    return generated_texts


def run_complete_neural_plasticity_experiment(args):
    """Run the complete neural plasticity experiment with real models and data."""
    
    # Print experiment header
    print(f"\n{'='*80}")
    print(f"🧠 NEURAL PLASTICITY EXPERIMENT {VERSION}")
    print(f"{'='*80}")
    
    # Setup environment
    env_info = setup_environment(args.seed)
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"💻 Using device: {device}")
    
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
            "warmup_steps": args.warmup_steps,
            "eval_interval": args.eval_interval,
            "pruning_level": args.pruning_level,
            "strategy": args.strategy,
            "seed": args.seed,
            "device": str(device),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": VERSION,
            "environment": env_info
        }
        json.dump(params, f, indent=2)
    
    print(f"📝 Saved experiment parameters to {params_file}")
    
    # Load model
    print(f"\n📦 Loading model: {args.model_name}")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        model = model.to(device)
        print(f"✓ Successfully loaded model")
        
        # Print model size information
        num_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Model size: {num_params/1e6:.2f}M parameters")
        
        # Detect model structure
        num_layers, num_heads = detect_model_structure(model)
        print(f"📊 Model structure: {num_layers} layers with {num_heads} attention heads each")
        print(f"📊 Total attention heads: {num_layers * num_heads}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Load and prepare data
    tokenizer, train_dataloader, validation_dataloader, sample_prompts = load_and_prepare_data(args)
    
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
    inference_dir = output_dir / "inference"
    
    for directory in [warmup_dir, pruning_dir, fine_tuning_dir, visualization_dir, inference_dir]:
        directory.mkdir(exist_ok=True)
    
    # Track all metrics for visualization
    all_metrics = {
        "warmup": {
            "losses": [],
            "steps": [],
            "perplexity": []
        },
        "pruning": {
            "losses": [],
            "steps": [],
            "perplexity": []
        },
        "fine_tuning": {
            "losses": [],
            "steps": [],
            "perplexity": []
        },
        "inference": {
            "baseline": [],
            "after_warmup": [],
            "after_pruning": [],
            "after_fine_tuning": []
        }
    }
    
    # Generate text with the initial model (baseline)
    print("\n📝 Generating text with initial model (baseline)...")
    baseline_texts = generate_sample_texts(model, tokenizer, sample_prompts, device=device)
    all_metrics["inference"]["baseline"] = baseline_texts
    
    # Save baseline text samples
    with open(inference_dir / "baseline_texts.txt", "w") as f:
        for i, (prompt, text) in enumerate(zip(sample_prompts, baseline_texts)):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Generated: {text}\n\n")
    
    # Baseline evaluation
    print("\n📊 Running baseline evaluation...")
    baseline_metrics = evaluate_model(
        model=model,
        dataloader=validation_dataloader,
        device=device
    )
    print(f"Baseline metrics: Loss = {baseline_metrics['loss']:.4f}, Perplexity = {baseline_metrics['perplexity']:.2f}")
    
    #----------------------------------------------
    # PHASE 1: WARMUP TRAINING
    #----------------------------------------------
    print(f"\n{'='*80}")
    print(f"🔥 PHASE 1: WARMUP TRAINING")
    print(f"{'='*80}")
    
    print("Starting warmup phase training...")
    warmup_result = run_warmup_phase(
        model=model,
        train_dataloader=train_dataloader,
        max_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        patience=args.warmup_steps,
        device=device,
        verbose=True,
        save_visualizations=True,
        output_dir=warmup_dir
    )
    
    # Extract warmup results
    warmup_losses = warmup_result.get("losses", [])
    warmup_steps = list(range(len(warmup_losses)))
    stabilization_point = warmup_result.get("stabilization_point", None)
    
    # Store warmup metrics
    all_metrics["warmup"]["losses"] = warmup_losses
    all_metrics["warmup"]["steps"] = warmup_steps
    
    # Display warmup visualization
    reporter.display_warmup_results(warmup_result)
    
    # Evaluate model after warmup
    print("\n📊 Evaluating model after warmup phase...")
    warmup_eval_metrics = evaluate_model(
        model=model,
        dataloader=validation_dataloader,
        device=device
    )
    print(f"After warmup: Loss = {warmup_eval_metrics['loss']:.4f}, Perplexity = {warmup_eval_metrics['perplexity']:.2f}")
    all_metrics["warmup"]["perplexity"] = warmup_eval_metrics['perplexity']
    
    # Generate text after warmup
    print("\n📝 Generating text after warmup phase...")
    warmup_texts = generate_sample_texts(model, tokenizer, sample_prompts, device=device)
    all_metrics["inference"]["after_warmup"] = warmup_texts
    
    # Save warmup text samples
    with open(inference_dir / "warmup_texts.txt", "w") as f:
        for i, (prompt, text) in enumerate(zip(sample_prompts, warmup_texts)):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Generated: {text}\n\n")
    
    #----------------------------------------------
    # PHASE 2: PRUNING
    #----------------------------------------------
    print(f"\n{'='*80}")
    print(f"✂️ PHASE 2: PRUNING")
    print(f"{'='*80}")
    
    # Calculate attention patterns and gradients
    print("\n📊 Calculating head importance metrics...")
    
    # Collect attention patterns for entropy calculation
    attention_maps = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(validation_dataloader, desc="Collecting attention patterns", total=min(5, len(validation_dataloader))):
            if len(attention_maps) >= 5:  # Limit to 5 batches for speed
                break
                
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with output_attentions=True
            outputs = model(**batch, output_attentions=True)
            
            # Extract attention patterns (all layers, all heads)
            batch_attention = outputs.attentions
            if batch_attention:
                attention_maps.extend(batch_attention)
    
    # Convert attention maps to tensor
    if attention_maps:
        # Stack along batch dimension
        stacked_attention = torch.stack(attention_maps)
        
        # Calculate entropy for each head
        print("Calculating head entropy...")
        entropy_values = calculate_head_entropy(stacked_attention)
        print(f"✓ Calculated entropy for {entropy_values.shape[0]} layers, {entropy_values.shape[1]} heads")
    else:
        print("❌ Could not extract attention patterns for entropy calculation")
        # Create dummy entropy values
        entropy_values = torch.rand((num_layers, num_heads), device=device)
    
    # Calculate gradients
    print("Calculating head gradients...")
    grad_norm_values = calculate_head_gradients(
        model=model,
        dataloader=train_dataloader,
        num_batches=3,
        device=device
    )
    print(f"✓ Calculated gradients for {grad_norm_values.shape[0]} layers, {grad_norm_values.shape[1]} heads")
    
    # Visualize head importance metrics
    reporter.visualize_head_metrics(entropy_values, grad_norm_values)
    
    # Generate pruning mask
    print(f"\nGenerating pruning mask using {args.strategy} strategy at {args.pruning_level*100:.1f}% level...")
    pruning_mask = generate_pruning_mask(
        grad_norm_values=grad_norm_values,
        entropy_values=entropy_values,
        prune_percent=args.pruning_level,
        strategy=args.strategy
    )
    
    # Visualize pruning decisions
    total_heads = pruning_mask.numel()
    pruned_count = pruning_mask.sum().item()
    pruning_rate = pruned_count / total_heads
    print(f"Pruning {pruned_count} out of {total_heads} heads ({pruning_rate:.2%})")
    
    reporter.visualize_pruning_decisions(entropy_values, grad_norm_values, pruning_mask)
    
    # Apply pruning
    print("\nApplying pruning mask to model...")
    pruned_heads = apply_pruning_mask(model, pruning_mask)
    print(f"✓ Pruned {len(pruned_heads)} attention heads")
    
    # Save pruned head information
    with open(pruning_dir / "pruned_heads.json", "w") as f:
        json.dump({"pruned_heads": [(int(l), int(h)) for l, h in pruned_heads]}, f, indent=2)
    
    # Evaluate model immediately after pruning (before fine-tuning)
    print("\n📊 Evaluating model immediately after pruning...")
    after_pruning_metrics = evaluate_model(
        model=model,
        dataloader=validation_dataloader,
        device=device
    )
    print(f"After pruning (before fine-tuning): Loss = {after_pruning_metrics['loss']:.4f}, Perplexity = {after_pruning_metrics['perplexity']:.2f}")
    
    # Generate text after pruning (before fine-tuning)
    print("\n📝 Generating text after pruning (before fine-tuning)...")
    after_pruning_texts = generate_sample_texts(model, tokenizer, sample_prompts, device=device)
    all_metrics["inference"]["after_pruning"] = after_pruning_texts
    
    # Save post-pruning text samples
    with open(inference_dir / "after_pruning_texts.txt", "w") as f:
        for i, (prompt, text) in enumerate(zip(sample_prompts, after_pruning_texts)):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Generated: {text}\n\n")
    
    #----------------------------------------------
    # PHASE 3: FINE-TUNING
    #----------------------------------------------
    print(f"\n{'='*80}")
    print(f"🔄 PHASE 3: FINE-TUNING")
    print(f"{'='*80}")
    
    # Fine-tune the pruned model
    print("\nFine-tuning pruned model...")
    
    # Setup optimizer with lower learning rate for fine-tuning
    fine_tuning_lr = args.learning_rate / 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=fine_tuning_lr)
    
    # Number of steps for fine-tuning (one epoch or specified steps)
    total_steps = min(args.eval_interval * 2, len(train_dataloader))
    
    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,  # 10% of steps for warmup
        num_training_steps=total_steps
    )
    
    # Fine-tuning loop
    model.train()
    fine_tuning_losses = []
    
    for step, batch in enumerate(tqdm(train_dataloader, desc="Fine-tuning", total=total_steps)):
        if step >= total_steps:
            break
            
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Record loss
        fine_tuning_losses.append(loss.item())
        
        # Evaluate periodically
        if (step + 1) % (total_steps // 5) == 0 or step == total_steps - 1:
            # Quick evaluation
            model.eval()
            with torch.no_grad():
                eval_batch = next(iter(validation_dataloader))
                eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                eval_outputs = model(**eval_batch)
                eval_loss = eval_outputs.loss.item()
                perplexity = torch.exp(torch.tensor(eval_loss)).item()
                
                print(f"Step {step+1}/{total_steps}: Loss = {loss.item():.4f}, Eval Loss = {eval_loss:.4f}, Perplexity = {perplexity:.2f}")
                
                # Store metrics
                all_metrics["fine_tuning"]["losses"].append(loss.item())
                all_metrics["fine_tuning"]["steps"].append(step)
                all_metrics["fine_tuning"]["perplexity"].append(perplexity)
            
            # Back to training mode
            model.train()
    
    # Final evaluation after fine-tuning
    print("\n📊 Evaluating model after fine-tuning...")
    final_metrics = evaluate_model(
        model=model,
        dataloader=validation_dataloader,
        device=device
    )
    print(f"After fine-tuning: Loss = {final_metrics['loss']:.4f}, Perplexity = {final_metrics['perplexity']:.2f}")
    
    # Generate text after fine-tuning
    print("\n📝 Generating text after fine-tuning...")
    final_texts = generate_sample_texts(model, tokenizer, sample_prompts, device=device)
    all_metrics["inference"]["after_fine_tuning"] = final_texts
    
    # Save final text samples
    with open(inference_dir / "final_texts.txt", "w") as f:
        for i, (prompt, text) in enumerate(zip(sample_prompts, final_texts)):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Generated: {text}\n\n")
    
    # Calculate improvements
    loss_improvement = (baseline_metrics['loss'] - final_metrics['loss']) / baseline_metrics['loss'] * 100
    perplexity_improvement = (baseline_metrics['perplexity'] - final_metrics['perplexity']) / baseline_metrics['perplexity'] * 100
    
    print(f"\n📈 IMPROVEMENTS:")
    print(f"  Loss: {loss_improvement:.2f}%")
    print(f"  Perplexity: {perplexity_improvement:.2f}%")
    
    # Compile experiment results
    experiment_results = {
        'warmup': {
            'metrics': warmup_result,
            'final_metrics': warmup_eval_metrics
        },
        'pruning': {
            'entropy_values': entropy_values.cpu().numpy().tolist() if isinstance(entropy_values, torch.Tensor) else None,
            'gradient_values': grad_norm_values.cpu().numpy().tolist() if isinstance(grad_norm_values, torch.Tensor) else None,
            'pruning_mask': pruning_mask.cpu().numpy().tolist() if isinstance(pruning_mask, torch.Tensor) else None,
            'pruned_heads': [(int(l), int(h)) for l, h in pruned_heads],
            'metrics': after_pruning_metrics
        },
        'fine_tuning': {
            'losses': fine_tuning_losses,
            'final_metrics': final_metrics
        },
        'baseline_metrics': baseline_metrics,
        'final_metrics': final_metrics,
        'improvements': {
            'loss': loss_improvement,
            'perplexity': perplexity_improvement
        },
        'inference': {
            'sample_prompts': sample_prompts,
            'baseline': baseline_texts,
            'after_warmup': warmup_texts,
            'after_pruning': after_pruning_texts,
            'after_fine_tuning': final_texts
        },
        'experiment_params': params
    }
    
    # Save experiment results
    results_file = output_dir / "experiment_results.json"
    with open(results_file, "w") as f:
        # Convert any non-serializable objects
        serializable_results = {}
        for key, value in experiment_results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        serializable_results[key][k] = v.cpu().numpy().tolist()
                    elif isinstance(v, np.ndarray):
                        serializable_results[key][k] = v.tolist()
                    else:
                        serializable_results[key][k] = v
            else:
                if isinstance(value, torch.Tensor):
                    serializable_results[key] = value.cpu().numpy().tolist()
                elif isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                else:
                    serializable_results[key] = value
                    
        json.dump(serializable_results, f, indent=2)
    
    print(f"📝 Saved experiment results to {results_file}")
    
    # Generate comprehensive visualization
    print("\n🎨 Generating comprehensive visualizations...")
    
    # Create complete training process visualization
    try:
        visualization_file = reporter.generate_comprehensive_dashboard(
            experiment=experiment_results,
            output_dir=output_dir / "dashboards"
        )
        print(f"✓ Generated comprehensive dashboard: {visualization_file}")
    except Exception as e:
        print(f"❌ Error generating comprehensive dashboard: {e}")
    
    # Save model and tokenizer
    model_dir = output_dir / "final_model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"💾 Saved model and tokenizer to {model_dir}")
    
    print(f"\n{'='*80}")
    print(f"✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"📂 Results directory: {output_dir}")
    print(f"📊 Dashboards: {output_dir / 'dashboards'}")
    print(f"🖼️ Visualizations: {output_dir / 'visualizations'}")
    print(f"📝 Generated texts: {output_dir / 'inference'}")
    
    # Return experiment directory for further analysis
    return output_dir


def main():
    """Main function."""
    # Get start time
    start_time = time.time()
    
    # Parse arguments
    args = parse_args()
    
    print(f"Neural Plasticity Experiment {VERSION}")
    print(f"Output directory: {args.output_dir}")
    
    # Run experiment
    output_dir = run_complete_neural_plasticity_experiment(args)
    
    # Calculate and print execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n⏱️ Total execution time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    
    if output_dir:
        print(f"\nTo view results, open: {output_dir}/dashboards/neural_plasticity_dashboard.html")


if __name__ == "__main__":
    main()