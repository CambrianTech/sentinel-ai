#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified Neural Plasticity Experiment Runner

This script uses the NeuralPlasticityExperiment class to run a complete
neural plasticity experiment with real models and data, generating HTML dashboards.

Key features:
- Uses real HuggingFace models (not simulated)
- Uses real dataset from HuggingFace (not simulated)
- Dynamically detects model stabilization (not on a fixed schedule)
- Makes pruning decisions based on entropy and gradient metrics
- Creates HTML dashboards showing the entire process
- Generates text samples at each phase to evaluate model quality

Usage:
    # Activate virtual environment
    source .venv/bin/activate
    
    # Run with small parameters for quick test
    python scripts/run_neural_plasticity_simple.py
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)

# Import NeuralPlasticityExperiment
from utils.neural_plasticity.experiment import NeuralPlasticityExperiment

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run neural plasticity experiment")
    
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                       help="Model name or path (default: distilgpt2)")
    
    parser.add_argument("--dataset", type=str, default="wikitext",
                       help="Dataset name (default: wikitext)")
    
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1",
                       help="Dataset configuration (default: wikitext-2-raw-v1)")
    
    parser.add_argument("--max_length", type=int, default=64,
                       help="Max sequence length (default: 64)")
    
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size (default: 2)")
    
    parser.add_argument("--pruning_level", type=float, default=0.15,
                       help="Pruning level (default: 0.15)")
    
    parser.add_argument("--pruning_strategy", type=str, default="combined",
                       choices=["gradient", "entropy", "random", "combined"],
                       help="Pruning strategy (default: combined)")
    
    parser.add_argument("--output_dir", type=str, 
                       default=os.path.join("plasticity_experiment", "simple", f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
                       help="Output directory")
    
    parser.add_argument("--use_dashboard", action="store_true", default=True,
                       help="Generate HTML dashboard (default: True)")
    
    parser.add_argument("--device", type=str, default=None,
                       help="Device (default: auto-detect)")
    
    return parser.parse_args()

def main():
    """Main function to run the experiment."""
    # Parse arguments
    args = parse_args()
    
    print(f"\n{'='*80}")
    print(f"SIMPLIFIED NEURAL PLASTICITY EXPERIMENT")
    print(f"{'='*80}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        # Create the experiment
        experiment = NeuralPlasticityExperiment(
            model_name=args.model_name,
            dataset=args.dataset,
            dataset_config=args.dataset_config,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
            pruning_level=args.pruning_level,
            pruning_strategy=args.pruning_strategy,
            learning_rate=5e-5,
            device=args.device,
            verbose=True,
            save_results=True,
            show_samples=True,
            sample_interval=5,
            use_dashboard=args.use_dashboard,
            dashboard_dir=os.path.join(args.output_dir, "dashboards")
        )
        
        # Setup experiment
        print("Setting up experiment...")
        experiment.setup()
        
        # Run warmup phase
        print("\nRunning warmup phase...")
        experiment.run_warmup(max_epochs=1, patience=10, min_steps=20, max_steps=30)
        
        # Analyze attention patterns
        print("\nAnalyzing attention patterns...")
        experiment.analyze_attention()
        
        # Run pruning cycle
        print("\nRunning pruning cycle...")
        experiment.run_pruning_cycle(training_steps=25)
        
        # Generate comprehensive visualization
        print("\nGenerating comprehensive visualization...")
        experiment.visualize_metrics_dashboard(
            save_path=os.path.join(args.output_dir, "visualizations", "complete_dashboard.png")
        )
        
        # Final evaluation
        print("\nPerforming final evaluation...")
        eval_metrics = experiment.evaluate()
        
        # Generate examples
        print("\nGenerating text examples...")
        generated_texts = experiment.generate_examples()
        
        # Save metadata
        experiment.save_metadata()
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"Results saved to: {args.output_dir}")
        print(f"Dashboard available at: {os.path.join(args.output_dir, 'dashboards')}")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR RUNNING EXPERIMENT: {str(e)}")
        print(f"{'='*80}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()