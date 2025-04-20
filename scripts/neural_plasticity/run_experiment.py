#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural Plasticity Experiment Runner

This script runs a complete neural plasticity experiment using the sentinel
implementation, not the utils implementation or any of the other scattered scripts.

This consolidates the scattered implementations into a single, well-structured approach
using the proper sentinel package.

Usage:
    python scripts/neural_plasticity/run_experiment.py --model_name distilgpt2 --pruning_level 0.2
"""

import os
import sys
import torch
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path if needed
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Import from sentinel package (not utils)
from sentinel.pruning.plasticity_controller import PlasticityController, create_plasticity_controller, PruningMode
from sentinel.plasticity.plasticity_loop import PlasticityExperiment
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from torch.utils.data import DataLoader
from datasets import load_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Neural Plasticity Experiment Runner")
    
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                      help="Model name (default: distilgpt2)")
    
    parser.add_argument("--dataset", type=str, default="wikitext",
                      help="Dataset name (default: wikitext)")
    
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1",
                      help="Dataset configuration (default: wikitext-2-raw-v1)")
    
    parser.add_argument("--max_length", type=int, default=128,
                      help="Maximum sequence length (default: 128)")
    
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Batch size (default: 4)")
    
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                      help="Learning rate (default: 5e-5)")
    
    parser.add_argument("--pruning_strategy", type=str, default="entropy",
                      choices=["entropy", "magnitude"],
                      help="Pruning strategy (default: entropy)")
    
    parser.add_argument("--pruning_level", type=float, default=0.2,
                      help="Pruning level (default: 0.2)")
    
    parser.add_argument("--fine_tuning_steps", type=int, default=500,
                      help="Number of fine-tuning steps (default: 500)")
    
    parser.add_argument("--output_dir", type=str, default=None,
                      help="Output directory (default: generated based on timestamp)")
    
    parser.add_argument("--device", type=str, default=None,
                      help="Device (default: auto-detect)")
    
    parser.add_argument("--pruning_mode", type=str, default="adaptive",
                      choices=["adaptive", "compressed"],
                      help="Pruning mode (default: adaptive)")
    
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed (default: 42)")
    
    parser.add_argument("--verbose", action="store_true", default=True,
                      help="Print verbose output (default: True)")
    
    parser.add_argument("--quick_test", action="store_true", default=False,
                      help="Run a quick test with minimal steps (default: False)")
                      
    parser.add_argument("--visualize", action="store_true", default=True,
                      help="Generate visualizations during execution (default: True)")
                      
    parser.add_argument("--no_dashboard", action="store_true", default=False,
                      help="Skip HTML dashboard generation (default: False)")
    
    return parser.parse_args()


def create_simple_dataset(tokenizer, num_samples=50, seq_length=64):
    """Create a simple dataset for testing using actual coherent text."""
    from datasets import Dataset
    import numpy as np
    
    # Use real pre-defined sentences that make sense for language modeling
    # This ensures the model can actually predict the sequence, giving realistic perplexity
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a world where technology continues to advance, humans seek deeper connections.",
        "Artificial intelligence systems are designed to process and learn from vast amounts of data.",
        "The history of computing can be traced back to ancient calculating devices.",
        "Neural networks consist of interconnected layers of artificial neurons.",
        "Machine learning algorithms improve through experience without explicit programming.",
        "The field of deep learning has revolutionized computer vision and natural language processing.",
        "Transformer models use attention mechanisms to process sequential data efficiently.",
        "Natural language understanding remains a challenging problem in artificial intelligence.",
        "The development of large language models has led to significant advances in text generation."
    ]
    
    # Create enough samples by repeating and combining these texts
    all_texts = []
    for _ in range(num_samples // len(texts) + 1):
        all_texts.extend(texts)
    all_texts = all_texts[:num_samples]
    
    # Tokenize the texts with padding and truncation
    encodings = tokenizer(all_texts, padding="max_length", truncation=True, 
                         max_length=seq_length, return_tensors="pt")
    
    # Extract input_ids and attention_mask
    input_ids = encodings["input_ids"].numpy()
    attention_mask = encodings["attention_mask"].numpy()
    
    # For causal language modeling, labels are the same as input_ids
    labels = input_ids.copy()
    
    # Create dataset with dictionaries that HuggingFace data collator expects
    features = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
    # Create HuggingFace dataset
    dataset = Dataset.from_dict(features)
    return dataset.with_format("torch")


def create_dataloader_builder(args, tokenizer, quick_test=False):
    """Create a function that builds dataloaders."""
    def build_dataloaders(batch_size=args.batch_size):
        # For quick tests, use a small subset of a real dataset
        if quick_test:
            try:
                # Use wikitext as a small but real dataset instead of synthetic data
                logger.info("Loading small wikitext sample for quick test...")
                
                # Always use real text data, just with a small sample for quick testing
                train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:100]")
                eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation[:20]")
                
                # Check if dataset has a text column
                if "text" in train_dataset.column_names:
                    # Define tokenization function
                    def tokenize_function(examples):
                        return tokenizer(
                            examples["text"], 
                            padding="max_length", 
                            truncation=True, 
                            max_length=args.max_length
                        )
                    
                    # Process datasets
                    train_dataset = train_dataset.map(tokenize_function, batched=True)
                    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
                    
                    # Remove original text columns
                    train_dataset = train_dataset.remove_columns(["text"])
                    eval_dataset = eval_dataset.remove_columns(["text"])
                    
                    # Add labels for language modeling
                    def add_labels(examples):
                        examples["labels"] = examples["input_ids"].copy()
                        return examples
                    
                    train_dataset = train_dataset.map(add_labels)
                    eval_dataset = eval_dataset.map(add_labels)
                    
                    # Set torch format
                    train_dataset = train_dataset.with_format("torch")
                    eval_dataset = eval_dataset.with_format("torch")
                    
                    logger.info(f"Using real wikitext data (small sample) for quick test")
                else:
                    raise ValueError("Wikitext dataset missing 'text' column")
            except Exception as e:
                # Only use the text-based synthetic dataset as fallback, not random tokens
                logger.warning(f"Failed to load wikitext for quick test: {e}. Using fallback text dataset.")
                train_dataset = create_simple_dataset(tokenizer, num_samples=50, seq_length=64)
                eval_dataset = create_simple_dataset(tokenizer, num_samples=10, seq_length=64)
            
        else:
            # Load real dataset
            logger.info(f"Loading dataset: {args.dataset}/{args.dataset_config}...")
            try:
                train_dataset = load_dataset(args.dataset, args.dataset_config, split="train")
                validation_dataset = load_dataset(args.dataset, args.dataset_config, split="validation")
                
                # Check if dataset has a text column
                if "text" not in train_dataset.column_names:
                    logger.warning(f"Dataset {args.dataset} does not have a 'text' column. Trying to find another text column...")
                    text_col = None
                    for col in train_dataset.column_names:
                        if col.lower() in ["content", "document", "article"]:
                            text_col = col
                            break
                    
                    if text_col is None:
                        logger.warning("Could not find a text column. Creating synthetic dataset instead.")
                        # Create synthetic datasets instead of recursive call
                        train_dataset = create_simple_dataset(tokenizer, num_samples=50, seq_length=64)
                        eval_dataset = create_simple_dataset(tokenizer, num_samples=10, seq_length=64)
                        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=default_data_collator), DataLoader(eval_dataset, batch_size=batch_size, collate_fn=default_data_collator)
                    
                    logger.info(f"Using column '{text_col}' as text data")
                    
                    # Define tokenization function using the identified column
                    def tokenize_function(examples):
                        return tokenizer(
                            examples[text_col], 
                            padding="max_length", 
                            truncation=True, 
                            max_length=args.max_length
                        )
                        
                else:
                    # Standard tokenization function
                    def tokenize_function(examples):
                        return tokenizer(
                            examples["text"], 
                            padding="max_length", 
                            truncation=True, 
                            max_length=args.max_length
                        )
                
                # Process datasets
                train_dataset = train_dataset.map(tokenize_function, batched=True)
                validation_dataset = validation_dataset.map(tokenize_function, batched=True)
                
                # Remove original text columns
                text_columns = []
                if "text" in train_dataset.column_names:
                    text_columns.append("text")
                elif 'text_col' in locals() and text_col is not None:
                    text_columns.append(text_col)
                else:
                    text_columns = [col for col in train_dataset.column_names 
                                  if "text" in col.lower()]
                
                if text_columns:
                    train_dataset = train_dataset.remove_columns(text_columns)
                    validation_dataset = validation_dataset.remove_columns(text_columns)
                
                # Add labels for language modeling
                def add_labels(examples):
                    examples["labels"] = examples["input_ids"].copy()
                    return examples
                
                train_dataset = train_dataset.map(add_labels)
                validation_dataset = validation_dataset.map(add_labels)
                
                # Set torch format
                train_dataset = train_dataset.with_format("torch")
                validation_dataset = validation_dataset.with_format("torch")
                
            except Exception as e:
                logger.error(f"Error loading dataset: {e}")
                logger.info("Falling back to synthetic dataset")
                # Create synthetic datasets instead of recursive call
                train_dataset = create_simple_dataset(tokenizer, num_samples=50, seq_length=64)
                eval_dataset = create_simple_dataset(tokenizer, num_samples=10, seq_length=64)
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=default_data_collator
        )
        
        eval_dataloader = DataLoader(
            validation_dataset if 'validation_dataset' in locals() else eval_dataset, 
            batch_size=batch_size, 
            collate_fn=default_data_collator
        )
        
        return train_dataloader, eval_dataloader
    
    return build_dataloaders


def main():
    """Main function to run the experiment."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory with timestamp if not provided
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use the /output directory as requested
        args.output_dir = os.path.join(
            "output",
            f"run_{timestamp}"
        )
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a timestamp-based experiment ID for consistent naming across functions
    experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.experiment_id = f"{args.pruning_strategy}_{args.pruning_level}_{experiment_timestamp}"
    
    # Create experiment and visualization directories
    experiment_dir = os.path.join(args.output_dir, args.experiment_id)
    visualization_dir = os.path.join(experiment_dir, "visualizations")
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Add file handler to logger
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "experiment.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log experiment parameters
    logger.info(f"Neural Plasticity Experiment")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Dataset: {args.dataset}/{args.dataset_config}")
    logger.info(f"Pruning Strategy: {args.pruning_strategy}")
    logger.info(f"Pruning Level: {args.pruning_level}")
    logger.info(f"Output Directory: {args.output_dir}")
    
    # Check if we're in quick test mode
    if args.quick_test:
        logger.info("QUICK TEST MODE: Using reduced steps and synthetic data")
        # Override parameters for quick testing
        args.fine_tuning_steps = 20
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create the experiment instance using the sentinel implementation
    experiment = PlasticityExperiment(
        model_name=args.model_name,
        output_dir=args.output_dir,
        device=args.device,
        adaptive_model=True
    )
    
    # Create dataloader builder
    dataloader_builder = create_dataloader_builder(args, tokenizer, quick_test=args.quick_test)
    
    # Set pruning mode
    pruning_mode = PruningMode.ADAPTIVE if args.pruning_mode == "adaptive" else PruningMode.COMPRESSED
    
    try:
        # Run the experiment
        logger.info("Running experiment...")
        # Pass the consistent experiment ID to ensure visualizations are saved in the right place
        results = experiment.run_experiment(
            pruning_strategy=args.pruning_strategy,
            pruning_level=args.pruning_level,
            dataloader_builder_fn=dataloader_builder,
            fine_tuning_steps=args.fine_tuning_steps,
            learning_rate=args.learning_rate,
            use_differential_lr=True,
            output_dir=args.output_dir,
            generate_visualizations=args.visualize,
            experiment_id=args.experiment_id  # Pass the consistent experiment ID
        )
        
        # Output results
        logger.info("Experiment completed!")
        logger.info(f"Baseline metrics: {results.get('metrics', {}).get('baseline', {})}")
        logger.info(f"Final metrics: {results.get('metrics', {}).get('final', {})}")
        
        # Calculate improvement if possible
        baseline_perplexity = results.get('metrics', {}).get('baseline', {}).get('perplexity', 0)
        final_perplexity = results.get('metrics', {}).get('final', {}).get('perplexity', 0)
        
        if baseline_perplexity > 0 and final_perplexity > 0:
            improvement = (baseline_perplexity - final_perplexity) / baseline_perplexity * 100
            logger.info(f"Perplexity improvement: {improvement:.2f}%")
        
        # Summary and next steps
        pruned_heads_count = len(results.get('pruned_heads', []))
        logger.info(f"Pruned {pruned_heads_count} attention heads")
        
        # Generate dashboard if not explicitly disabled
        if not args.no_dashboard:
            try:
                from scripts.neural_plasticity.visualization.dashboard_generator import create_neural_plasticity_dashboard
                dashboard_dir = os.path.join(args.output_dir, "dashboards")
                os.makedirs(dashboard_dir, exist_ok=True)
                dashboard_path = os.path.join(dashboard_dir, "dashboard.html")
                
                # Load experiment data
                from scripts.neural_plasticity.visualization.dashboard_generator import load_experiment_data
                experiment_data = load_experiment_data(args.output_dir)
                
                logger.info(f"Generating dashboard at {dashboard_path}...")
                create_neural_plasticity_dashboard(experiment_data, dashboard_path)
                logger.info(f"Dashboard generated successfully at {dashboard_path}")
                
                # Generate enhanced dashboard
                try:
                    from scripts.neural_plasticity.visualization.enhanced_dashboard import generate_dashboard as generate_enhanced_dashboard
                    from scripts.neural_plasticity.visualization.enhanced_dashboard import load_experiment_data
                    
                    enhanced_dashboard_path = os.path.join(dashboard_dir, "enhanced_dashboard.html")
                    logger.info(f"Generating enhanced dashboard at {enhanced_dashboard_path}...")
                    
                    # Load experiment data
                    experiment_data = load_experiment_data(args.output_dir)
                    
                    # Generate enhanced dashboard
                    generate_enhanced_dashboard(experiment_data, enhanced_dashboard_path)
                    logger.info(f"Enhanced dashboard generated successfully at {enhanced_dashboard_path}")
                except Exception as e:
                    logger.warning(f"Error generating enhanced dashboard: {e}")
                
                # Generate comprehensive dashboard
                try:
                    from scripts.neural_plasticity.visualization.comprehensive_dashboard import generate_comprehensive_dashboard
                    
                    comprehensive_dir = os.path.join(dashboard_dir, "comprehensive")
                    logger.info(f"Generating comprehensive dashboard in {comprehensive_dir}...")
                    
                    # Load experiment data
                    from scripts.neural_plasticity.visualization.comprehensive_dashboard import load_experiment_data
                    experiment_data = load_experiment_data(args.output_dir)
                    
                    # Generate dashboard
                    outputs = generate_comprehensive_dashboard(experiment_data, comprehensive_dir)
                    
                    if 'complete_process' in outputs:
                        logger.info(f"Complete process visualization generated at: {outputs['complete_process']}")
                    
                    logger.info(f"Comprehensive dashboard generated in {comprehensive_dir}")
                except Exception as e:
                    logger.warning(f"Error generating comprehensive dashboard: {e}")
                    
            except Exception as e:
                logger.error(f"Error generating dashboard: {e}")
        else:
            logger.info("HTML dashboard generation skipped (--no_dashboard flag set)")
            
        # Final log message
        logger.info(f"Results saved to: {args.output_dir}")
        
        # Construct the path to the visualizations directory
        experiment_viz_dir = os.path.join(args.output_dir, args.experiment_id, "visualizations")
        
        # Print helpful next steps
        print("\nExperiment completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Visualizations saved to: {experiment_viz_dir}")
        print("\nNext steps:")
        print(f"- View log file: {os.path.join(args.output_dir, 'experiment.log')}")
        print(f"- Check metrics: {os.path.join(args.output_dir, args.experiment_id, 'metrics.json')}")
        print(f"- View visualizations: {experiment_viz_dir}")
        
        # Direct to visualizations based on what was generated
        if args.visualize:
            viz_dir = os.path.join(args.output_dir, args.experiment_id, "visualizations")
            print("\nVisualizations:")
            print(f"- Direct visualizations: {viz_dir}")
            print(f"  - Training Progress: {os.path.join(viz_dir, 'training_progress_complete.png')}")
            print(f"  - Entropy Heatmaps: {os.path.join(viz_dir, 'entropy_heatmap_*.png')}")
            print(f"  - Entropy Changes: {os.path.join(viz_dir, 'entropy_changes.png')}")
            print(f"  - Pruned Heads: {os.path.join(viz_dir, 'pruned_heads*.png')}")
            print(f"  - Recovery Analysis: {os.path.join(viz_dir, 'recovery_analysis.png')}")
        
        # Direct to dashboards if they were generated
        if not args.no_dashboard:
            print("\nDashboards (HTML):")
            print(f"- Simple dashboard: {os.path.join(args.output_dir, 'dashboards/dashboard.html')}")
            print(f"- Enhanced dashboard: {os.path.join(args.output_dir, 'dashboards/enhanced_dashboard.html')}")
            print(f"- Comprehensive dashboard: {os.path.join(args.output_dir, 'dashboards/comprehensive/complete_process/neural_plasticity_process.png')}")
        
        print("\nRun examples:")
        print("- With visualizations (default):")
        print(f"  python scripts/run_neural_plasticity.py --model_name distilgpt2 --pruning_level 0.2")
        print("- Skip HTML dashboards (images only):")
        print(f"  python scripts/run_neural_plasticity.py --no_dashboard")
        print("- Quick test with synthetic data:")
        print(f"  python scripts/run_neural_plasticity.py --quick_test")
        print("- No visualizations:")
        print(f"  python scripts/run_neural_plasticity.py --visualize=False")
        
        return results
        
    except Exception as e:
        logger.error(f"Error running experiment: {e}", exc_info=True)
        print(f"Error running experiment: {e}")
        return None


if __name__ == "__main__":
    main()