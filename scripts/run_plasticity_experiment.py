#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural Plasticity Experiment Runner

This script runs a neural plasticity experiment using the modular implementation
from the sentinel package, focusing on entropy-guided pruning and dynamic head
stabilization to create more efficient transformer models.

Usage:
    # Activate virtual environment
    source .venv/bin/activate
    
    # Run with minimal settings (fast)
    python scripts/run_plasticity_experiment.py --quick
    
    # Run with custom parameters
    python scripts/run_plasticity_experiment.py --model_name distilgpt2 --cycles 3
"""

import os
import sys
import argparse
import torch
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from torch.utils.data import DataLoader
from datasets import load_dataset

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Import the modular plasticity implementation
from sentinel.pruning.plasticity_controller import create_plasticity_controller
from sentinel.pruning.dual_mode_pruning import PruningMode, get_model_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run neural plasticity experiment")
    
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
    
    parser.add_argument("--cycles", type=int, default=5,
                      help="Number of plasticity cycles to run (default: 5)")
    
    parser.add_argument("--high_entropy", type=float, default=0.8,
                      help="High entropy threshold for pruning (default: 0.8)")
    
    parser.add_argument("--low_entropy", type=float, default=0.4,
                      help="Low entropy threshold for revival (default: 0.4)")
    
    parser.add_argument("--grad_threshold", type=float, default=1e-3,
                      help="Gradient threshold for pruning decisions (default: 1e-3)")
                      
    parser.add_argument("--mode", type=str, default="adaptive", choices=["adaptive", "compressed"],
                      help="Pruning mode (default: adaptive)")
    
    parser.add_argument("--output_dir", type=str, 
                      default=os.path.join("plasticity_experiment", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                      help="Output directory")
    
    parser.add_argument("--device", type=str, default=None,
                      help="Device (default: auto-detect)")
                      
    parser.add_argument("--quick", action="store_true",
                      help="Run a quick experiment with smaller parameters")
    
    return parser.parse_args()


def main():
    """Main function to run the experiment."""
    # Parse arguments
    args = parse_args()
    
    # If quick mode is specified, override parameters for a faster run
    if args.quick:
        args.max_length = 32
        args.batch_size = 2
        args.cycles = 3
        print("🏃 Running in quick mode with reduced parameters")
    
    print(f"\n{'='*80}")
    print(f"NEURAL PLASTICITY EXPERIMENT USING MODULAR IMPLEMENTATION")
    print(f"{'='*80}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine appropriate device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    try:
        # Load model and tokenizer
        print(f"Loading model: {args.model_name}")
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        model = model.to(device)
        
        # Get model structure
        model_info = get_model_info(model)
        print(f"Model loaded with {model_info['nonzero_params']:,} parameters")
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load dataset
        print(f"Loading dataset: {args.dataset}/{args.dataset_config}")
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
        print("Tokenizing datasets...")
        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        validation_dataset = validation_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        
        # Set format for PyTorch
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
        
        print(f"Train dataset size: {len(train_dataset)} examples")
        print(f"Validation dataset size: {len(validation_dataset)} examples")
        
        # Create plasticity controller
        print("\nCreating plasticity controller...")
        controller = create_plasticity_controller(
            model=model,
            mode=PruningMode.ADAPTIVE if args.mode == "adaptive" else PruningMode.COMPRESSED,
            high_entropy_threshold=args.high_entropy,
            low_entropy_threshold=args.low_entropy,
            grad_threshold=args.grad_threshold
        )
        
        # Run plasticity cycles
        print(f"\nRunning {args.cycles} plasticity cycles...")
        
        cycle_metrics = []
        
        for cycle in range(args.cycles):
            print(f"\nCycle {cycle+1}/{args.cycles}")
            
            # Create cycle directory
            cycle_dir = os.path.join(args.output_dir, f"cycle_{cycle+1}")
            os.makedirs(cycle_dir, exist_ok=True)
            
            # Run plasticity step
            pruned_heads, revived_heads, metrics = controller.step(
                dataloader=train_dataloader,
                num_batches=2,  # Process limited batches for efficiency
                verbose=True,
                output_dir=cycle_dir
            )
            
            # Process results
            print(f"  Pruned {len(pruned_heads)} heads, Revived {len(revived_heads)} heads")
            print(f"  Total pruned: {metrics['total_pruned']} heads")
            print(f"  Model sparsity: {metrics['sparsity']:.4f}")
            
            # Store metrics for analysis
            cycle_metrics.append({
                "cycle": cycle + 1,
                "pruned_heads": len(pruned_heads),
                "revived_heads": len(revived_heads),
                "total_pruned": metrics["total_pruned"],
                "sparsity": metrics["sparsity"]
            })
            
            # Evaluate after pruning
            print("  Evaluating pruned model...")
            model.eval()
            with torch.no_grad():
                eval_loss = 0.0
                eval_steps = 0
                
                for batch in validation_dataloader:
                    # Only process a few batches for quick evaluation
                    if eval_steps >= 5:
                        break
                        
                    # Move batch to device
                    inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    
                    # Forward pass
                    outputs = model(**inputs)
                    loss = outputs.loss
                    
                    eval_loss += loss.item()
                    eval_steps += 1
                
                # Calculate average loss and perplexity
                if eval_steps > 0:
                    avg_loss = eval_loss / eval_steps
                    perplexity = torch.exp(torch.tensor(avg_loss)).item()
                    
                    print(f"  Evaluation Loss: {avg_loss:.4f}")
                    print(f"  Perplexity: {perplexity:.2f}")
                    
                    # Add to metrics
                    cycle_metrics[-1]["eval_loss"] = avg_loss
                    cycle_metrics[-1]["perplexity"] = perplexity
            
            # Generate sample text
            print("  Generating sample text...")
            model.eval()
            
            # Use a simple prompt
            prompt = "The neural plasticity system allows models to"
            
            # Tokenize prompt
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            
            # Generate continuation
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids,
                    max_length=50,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.92
                )
                
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            print(f"  Sample: {generated_text}")
            
            # Save sample to file
            with open(os.path.join(cycle_dir, "sample.txt"), "w") as f:
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Generated: {generated_text}\n")
            
            # Save cycle metrics to file
            import json
            with open(os.path.join(cycle_dir, "metrics.json"), "w") as f:
                json.dump(cycle_metrics[-1], f, indent=2)
        
        # Save final model if requested
        if args.cycles > 0:
            print("\nSaving final pruned model...")
            model_dir = os.path.join(args.output_dir, "model")
            os.makedirs(model_dir, exist_ok=True)
            
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            
            # Save model info
            final_info = get_model_info(model)
            with open(os.path.join(model_dir, "model_info.json"), "w") as f:
                json.dump({
                    "model_name": args.model_name,
                    "pruning_mode": args.mode,
                    "total_parameters": final_info["total_params"],
                    "nonzero_parameters": final_info["nonzero_params"],
                    "sparsity": final_info["sparsity"],
                    "size_mb": final_info["size_mb"],
                    "saved_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
        
        # Save all metrics to file
        with open(os.path.join(args.output_dir, "all_metrics.json"), "w") as f:
            json.dump(cycle_metrics, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR RUNNING EXPERIMENT: {str(e)}")
        print(f"{'='*80}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()