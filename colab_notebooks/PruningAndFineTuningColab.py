#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pruning and Fine-Tuning Colab (v0.0.41)

This script demonstrates making a GPT-2 model smaller and more powerful by:
1. Applying pruning to remove less important attention heads
2. Fine-tuning the pruned model to recover performance
3. Showing clear metrics of improvement

It's designed to be run in Google Colab using real-world data (Wikitext).

Version History:
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
    !pip install -q transformers==4.38.0 datasets==2.17.0 torch matplotlib tqdm

# Add project root to path for imports
if not any(p.endswith('sentinel-ai') for p in sys.path):
    # For Google Colab - handle case where the notebook is running in a different directory
    if os.path.exists('/content'):
        # Clone the repo if running in Colab and not already cloned
        if not os.path.exists('/content/sentinel-ai'):
            !git clone https://github.com/your-username/sentinel-ai.git /content/sentinel-ai
        sys.path.append('/content/sentinel-ai')
    else:
        # Add parent directory to path if running locally
        sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

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
    
    # Run the experiment
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