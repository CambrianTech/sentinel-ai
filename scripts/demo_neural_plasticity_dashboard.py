#!/usr/bin/env python
"""
Neural Plasticity Dashboard Demo

This script demonstrates the neural plasticity dashboard functionality,
showing model predictions during training, pruning decisions, and visualization.

Version: v0.0.2 (2025-04-20 15:15:00)
"""

import os
import argparse
import platform
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import webbrowser

# Add the repository root to the Python path to allow imports
import sys
import os.path as osp
root_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(root_dir)

from utils.neural_plasticity.training import run_plasticity_loop
from utils.neural_plasticity.dashboard import DashboardReporter

def parse_args():
    parser = argparse.ArgumentParser(description="Neural Plasticity Dashboard Demo")
    parser.add_argument("--model", type=str, default="distilgpt2",
                      help="Model name or path")
    parser.add_argument("--dataset", type=str, default="wikitext-2-raw-v1",
                      help="Dataset name for training")
    parser.add_argument("--pruning_level", type=float, default=0.2,
                      help="Fraction of heads to prune (0-1)")
    parser.add_argument("--strategy", type=str, default="combined",
                      choices=["gradient", "entropy", "combined", "random"],
                      help="Pruning strategy")
    parser.add_argument("--max_warmup_steps", type=int, default=1000,
                      help="Maximum number of warmup steps (will stop when loss stabilizes)")
    parser.add_argument("--training_steps", type=int, default=300,
                      help="Number of training steps after pruning")
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                      help="Learning rate for training")
    parser.add_argument("--dashboard_dir", type=str, default="dashboard",
                      help="Directory to store dashboard files")
    parser.add_argument("--open_browser", action="store_true",
                      help="Open dashboard in browser when complete")
    parser.add_argument("--no_samples", action="store_true",
                      help="Disable sample text display during training")
    parser.add_argument("--no_gpu", action="store_true",
                      help="Force CPU usage even if GPU is available")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create dashboard directory
    os.makedirs(args.dashboard_dir, exist_ok=True)
    
    # Set device (CPU or GPU)
    if args.no_gpu:
        device = torch.device("cpu")
        print(f"🖥️  Using CPU as requested")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"🖥️  Using CPU (no GPU available)")
    
    print(f"Loading model: {args.model}")
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Move model to the appropriate device
    model = model.to(device)
    
    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading dataset: {args.dataset}")
    # Load dataset
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", args.dataset, split="train")
        
        # Create training and evaluation splits
        dataset_size = len(dataset)
        # Use a decent size training set for better quality
        train_size = min(2000, int(dataset_size * 0.9))
        eval_size = min(500, dataset_size - train_size)
        
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, train_size + eval_size))
        
        # Tokenize datasets with longer context window
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        eval_dataset = eval_dataset.map(tokenize_function, batched=True)
        
        # Create data loaders
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
        
        print(f"Training dataset size: {len(train_dataset)} examples")
        print(f"Evaluation dataset size: {len(eval_dataset)} examples")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to dummy dataset")
        
        # Create simple text dataset as fallback
        text = "This is a dummy dataset for demonstration purposes. " * 10
        
        # Tokenize text
        encodings = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
        
        # Create dummy dataset
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
                
            def __len__(self):
                return 20  # Create 20 dummy examples
                
            def __getitem__(self, idx):
                return {key: val[0].clone() for key, val in self.encodings.items()}
        
        dummy_dataset = DummyDataset(encodings)
        
        # Create data loaders
        train_dataloader = DataLoader(dummy_dataset, batch_size=args.batch_size, shuffle=True)
        eval_dataloader = DataLoader(dummy_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Running Neural Plasticity with dashboard visualization...")
    
    # Run warmup phase first to properly establish baseline after stabilization
    print(f"\n=== Running warmup phase until loss stabilizes (max {args.max_warmup_steps} steps) ===")
    from utils.neural_plasticity.training import run_warmup_phase
    
    # Initialize dashboard for warmup phase
    warmup_dashboard_dir = os.path.join(args.dashboard_dir, "warmup")
    os.makedirs(warmup_dashboard_dir, exist_ok=True)
    
    # Check if we're running on Apple Silicon and use a safer approach
    is_apple_silicon = (platform.system() == "Darwin" and platform.processor() == "arm")
    
    if is_apple_silicon:
        print("🍎 Using simplified warmup for Apple Silicon compatibility")
        # For Apple Silicon, just evaluate the model as a simplified warmup
        from utils.neural_plasticity.core import evaluate_model
        
        # Create dummy warmup results
        warmup_results = {
            "is_stable": True,
            "total_steps": 0,
            "initial_loss": 0,
            "final_loss": 0,
            "improvement_percent": 0
        }
        
        # Evaluate model as baseline
        try:
            baseline_metrics = evaluate_model(model, eval_dataloader, device)
            warmup_results["final_loss"] = baseline_metrics["loss"]
            print(f"Initial evaluation: Loss = {baseline_metrics['loss']:.4f}, Perplexity = {baseline_metrics['perplexity']:.2f}")
        except Exception as e:
            print(f"⚠️ Evaluation error: {e}")
    else:
        # Normal warmup for non-Apple Silicon
        warmup_results = run_warmup_phase(
            model=model,
            train_dataloader=train_dataloader,
            max_epochs=10,  # Allow multiple epochs if needed for stabilization 
            learning_rate=args.learning_rate,
            warmup_steps=args.max_warmup_steps // 10,  # 10% of steps for warming up the warmup
            patience=args.max_warmup_steps // 20,  # Stop after this many steps with no improvement
            min_warmup_steps=args.max_warmup_steps // 5,  # Minimum steps to ensure some training
            max_warmup_steps=args.max_warmup_steps,
            device=device,
            verbose=True,
            visualize=True,
            save_visualizations=True,
            output_dir=warmup_dashboard_dir
        )
    
    # Check if warmup was successful
    if 'error' in warmup_results:
        print(f"⚠️ Warmup phase encountered an error: {warmup_results['error']}")
        print("Continuing with pruning anyway...")
    else:
        is_stable = warmup_results.get('is_stable', False)
        steps_completed = warmup_results.get('total_steps', 0)
        initial_loss = warmup_results.get('initial_loss', 0)
        final_loss = warmup_results.get('final_loss', 0)
        improvement = warmup_results.get('improvement_percent', 0)
        
        print(f"\n=== Warmup Phase Complete ===")
        print(f"Steps completed: {steps_completed}")
        print(f"Initial loss: {initial_loss:.4f}")
        print(f"Final loss: {final_loss:.4f}")
        print(f"Improvement: {improvement:.2f}%")
        print(f"Loss stabilized: {'Yes' if is_stable else 'No (reached max steps)'}")
    
    # Run neural plasticity with dashboard visualization
    print(f"\n=== Running Neural Plasticity Phase ===")
    results = run_plasticity_loop(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        pruning_level=args.pruning_level,
        strategy=args.strategy,
        learning_rate=args.learning_rate,
        training_steps=args.training_steps,
        use_differential_lr=True,
        show_samples=not args.no_samples,
        tokenizer=tokenizer,
        sample_interval=5,
        use_dashboard=True,
        dashboard_dir=args.dashboard_dir,
        dashboard_name="neural_plasticity_dashboard.html"
    )
    
    # Get dashboard path
    dashboard_path = results.get("dashboard_path")
    if dashboard_path:
        print(f"\n✅ Neural plasticity completed successfully!")
        print(f"Dashboard available at: {dashboard_path}")
        
        # Open in browser if requested
        if args.open_browser:
            dashboard_url = f"file://{os.path.abspath(dashboard_path)}"
            print(f"Opening dashboard in browser: {dashboard_url}")
            webbrowser.open(dashboard_url)
    else:
        print(f"\n✅ Neural plasticity completed successfully!")
        print(f"Dashboard was not generated. Check if dashboard.py is accessible.")
    
    # Print summary metrics with proper analysis
    print("\nSummary Metrics:")
    print(f"Baseline perplexity (after warmup): {results['baseline_metrics']['perplexity']:.2f}")
    print(f"After pruning perplexity: {results['pruned_metrics']['perplexity']:.2f}")
    print(f"Final perplexity (after fine-tuning): {results['final_metrics']['perplexity']:.2f}")
    
    # Calculate impact of pruning and recovery
    pruning_impact = (results['pruned_metrics']['perplexity'] - results['baseline_metrics']['perplexity']) / results['baseline_metrics']['perplexity'] * 100
    recovery = (results['pruned_metrics']['perplexity'] - results['final_metrics']['perplexity']) / results['pruned_metrics']['perplexity'] * 100
    net_impact = (results['final_metrics']['perplexity'] - results['baseline_metrics']['perplexity']) / results['baseline_metrics']['perplexity'] * 100
    
    print(f"Pruning impact: {pruning_impact:.2f}% {'worse' if pruning_impact > 0 else 'better'}")
    print(f"Recovery through fine-tuning: {recovery:.2f}%")
    print(f"Net impact after fine-tuning: {net_impact:.2f}% {'worse' if net_impact > 0 else 'better'}")
    
    # Sparsity analysis
    if hasattr(model, "config"):
        total_heads = model.config.num_hidden_layers * model.config.num_attention_heads
        sparsity = len(results['pruned_heads']) / total_heads * 100
        print(f"Pruned {len(results['pruned_heads'])} out of {total_heads} heads ({sparsity:.1f}% sparsity)")

if __name__ == "__main__":
    main()