#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pruning and Fine-Tuning Colab (v0.0.44)

This script demonstrates making a GPT-2 model smaller and more powerful by:
1. Applying pruning to remove less important attention heads
2. Fine-tuning the pruned model to recover performance
3. Showing clear metrics of improvement

It's designed to be run in Google Colab using real-world data (Wikitext).

IMPORTANT USAGE NOTE:
For quick testing of the modular API, use:
    python PruningAndFineTuningColab.py --test_mode --super_simple

The full experiment workflow is under development - while the modular API is
ready and functional, running the complete pruning pipeline still requires
additional debugging.

Version History:
- v0.0.44 (April 2025): Fixed Colab repository URL and branch selection for reliable execution
- v0.0.43 (April 2025): Fixed entropy pruning implementation to handle API availability gracefully
- v0.0.42 (April 2025): Added super_simple test mode and improved error handling
- v0.0.41 (April 2025): Modularized code using sentinel.pruning package
- v0.0.40 (April 2025): Improve robustness for different model architectures
- v0.0.39 (April 2025): Fix TypeError in run_experiment function call
- v0.0.38 (April 2025): Fix ValueError in generate_text function
- v0.0.37 (April 2025): Complete rewrite with minimal dependencies for reliability
- v0.0.36 (April 2025): Simplified pruning implementation for better reliability 
- v0.0.35 (April 2025): Fixed in-place operation error in apply_head_pruning function
- v0.0.34 (April 2025): Fixed undefined variable error, visualization issues and enhanced CUDA error handling
- v0.0.33 (April 2025): Fixed visualization issues, improved model compatibility and enhanced error handling
- v0.0.32 (April 2025): Added CUDA error handling for Colab compatibility and memory management
- v0.0.31 (April 2025): Fixed get_strategy parameters issue and improved Colab compatibility 
- v0.0.30 (April 2025): Added OPT model support and chart improvements
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Install required packages
try:
    import transformers
    import matplotlib
    import tqdm
    import numpy
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call(["pip", "install", "-q", "transformers==4.38.0", "datasets==2.17.0", "torch", "matplotlib", "tqdm"])

# Add project root to path for imports
# First determine the project root directory
if os.path.exists('/content'):
    # We're in Colab
    project_root = '/content/sentinel-ai'
    # Clone the repo if running in Colab and not already cloned
    if not os.path.exists(project_root):
        import subprocess
        subprocess.check_call(["git", "clone", "https://github.com/your-username/sentinel-ai.git", project_root])
else:
    # We're running locally - find the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir.endswith('colab_notebooks'):
        # We're in the colab_notebooks directory
        project_root = os.path.dirname(current_dir)
    else:
        # Not sure where we are - use the working directory
        project_root = os.getcwd()
        # Check if we're in the sentinel-ai directory
        if not os.path.basename(project_root) == 'sentinel-ai':
            # Try the parent directory
            parent_dir = os.path.dirname(project_root)
            if os.path.basename(parent_dir) == 'sentinel-ai':
                project_root = parent_dir

# Add the project root to the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

# Import the modular sentinel.pruning API
try:
    from sentinel.pruning.experiment_runner import run_experiment, ExperimentConfig
    from sentinel.pruning.text_generator import interactive_generate
    print("Successfully imported sentinel.pruning modules")
except ImportError as e:
    print(f"Failed to import sentinel.pruning modules: {e}")
    print("This notebook requires the modular sentinel.pruning package.")
    print("Make sure you've pulled the latest code from the repository.")
    print("Falling back to direct API imports...")
    
    # Fall back to the old API imports if sentinel.pruning is not available
    from utils.pruning.api.pruning import compute_head_importance, prune_heads, fine_tune, evaluate_model
    from utils.pruning.api.data import load_wikitext, prepare_data, prepare_test_data

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories for saving results
os.makedirs("pruning_results", exist_ok=True)


def main(args):
    """Main function to run the experiment with command line arguments"""
    # Create experiment configuration
    config = ExperimentConfig(
        model_name=args.model_name,
        pruning_percent=args.pruning_percent,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        use_test_data=args.test_mode,
        device=device
    )
    
    # If in test mode, use a super-simplified workflow just to verify imports
    if args.test_mode and args.super_simple:
        print("\nRunning in super simple test mode to verify imports and API")
        print("Args:", args)
        
        try:
            print("\nStarting super_simple test mode...")
            
            # Load model and tokenizer
            print("Loading model and tokenizer...")
            from sentinel.pruning.model_manager import load_model
            model, tokenizer = load_model(config.model_name, device=config.device)
            print(f"Successfully loaded {config.model_name} model")
            
            # Generate text
            print("\nGenerating text...")
            from sentinel.pruning.text_generator import generate_text
            text = generate_text(model, tokenizer, "The quick brown fox", max_length=50)
            print(f"Generated text: {text}")
            
            # Create a progress tracker
            print("\nTesting progress tracker...")
            from sentinel.pruning.visualization import ProgressTracker
            tracker = ProgressTracker(disable_plotting=True)
            tracker.update(0, 5.0, 150.0, text)
            print("Progress tracker test successful")
            
            print("\nAPI test completed successfully!")
            return 0
        except Exception as e:
            print(f"\nError in simplified test: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Run the full experiment
    try:
        print("\nRunning experiment with modular API and improved error handling...")
        print("Note: Entropy pruning will gracefully fall back to alternative methods if needed")
        model, tokenizer, summary = run_experiment(config)
        
        # Interactive generation if requested
        if args.interactive:
            print("\nEntering interactive generation mode. Type 'exit' to quit.")
            while True:
                prompt = input("\nEnter a prompt (or 'exit' to quit): ")
                if prompt.lower() == 'exit':
                    break
                interactive_generate(model, tokenizer, prompt)
        
        return 0
    except Exception as e:
        print(f"\nError in experiment: {e}")
        import traceback
        traceback.print_exc()
        
        if "collect_attention_distributions" in str(e) or "entropy_based_pruning" in str(e):
            print("\nNOTE: If you're seeing an error with entropy pruning functions, make sure")
            print("you're using the latest version of the benchmark_with_metrics.py script that")
            print("has the fix for handling different API availability scenarios.")
        
        return 1


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Prune and fine-tune a transformer model")
    
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                        help="Model name or path (default: distilgpt2)")
    
    parser.add_argument("--pruning_percent", type=float, default=0.3,
                        help="Percentage of heads to prune (default: 0.3)")
    
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of fine-tuning epochs (default: 3)")
    
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training (default: 4)")
    
    parser.add_argument("--test_mode", action="store_true",
                        help="Use small test dataset for quick validation")
    
    parser.add_argument("--super_simple", action="store_true",
                        help="Run a super simplified test of the API (must be used with --test_mode)")
    
    parser.add_argument("--interactive", action="store_true",
                        help="Enable interactive text generation after training")
    
    return parser.parse_args()


if __name__ == "__main__":
    # When run as a script, parse arguments and run main
    args = parse_args()
    sys.exit(main(args))
else:
    # When run in a notebook, use default parameters
    
    # Configure experiment parameters
    MODEL_NAME = "distilgpt2"  # Smaller GPT-2 model for faster demonstration
    PRUNING_PERCENT = 0.3  # Percentage of heads to prune (0-1)
    NUM_EPOCHS = 3  # Number of fine-tuning epochs 
    BATCH_SIZE = 4  # Batch size for training and evaluation
    
    # Create experiment config
    experiment_config = ExperimentConfig(
        model_name=MODEL_NAME,
        pruning_percent=PRUNING_PERCENT,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        device=device,
        output_dir="pruning_results"
    )
    
    # Run the experiment
    model, tokenizer, summary = run_experiment(experiment_config)
    
    # Function for interactive generation
    def generate_interactive(prompt=None, max_length=100):
        """Generate text from the model interactively"""
        return interactive_generate(model, tokenizer, prompt, max_length)