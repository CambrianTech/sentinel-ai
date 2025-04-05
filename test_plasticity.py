#!/usr/bin/env python
"""
Test script for neural plasticity functionality.
"""

import os
import sys
import torch
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our fixed modules
from utils.pruning.fixed_pruning_module_jax import PruningModule
from utils.pruning.growth import determine_active_heads
from sentinel_data.dataset_loader import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Test Neural Plasticity Functionality")
    
    parser.add_argument("--model_name", type=str, default="distilgpt2",
                      help="Model name (default: distilgpt2)")
    parser.add_argument("--dataset", type=str, default="tiny_shakespeare",
                      help="Dataset name (default: tiny_shakespeare)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to use (default: cuda if available, else cpu)")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose output")
    
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Testing neural plasticity with model {args.model_name}")
    
    # Load model
    print("Loading model...")
    pruning_module = PruningModule(args.model_name, device=args.device)
    success = pruning_module.load_model()
    
    if not success:
        print("Failed to load model!")
        return False
    
    print(f"Model loaded successfully with {pruning_module.num_layers} layers and {pruning_module.num_heads} heads per layer")
    
    # Load dataset
    print(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(args.dataset, pruning_module.tokenizer)
    
    # Check for model.params attribute
    if hasattr(pruning_module.model, 'params'):
        print("✅ model.params attribute exists")
    else:
        print("❌ model.params attribute is missing")
    
    # Check active heads
    print("Testing determine_active_heads function...")
    active_heads = determine_active_heads(pruning_module, pruning_module.params)
    print(f"Found {len(active_heads)} active heads")
    
    # Test text generation
    print("\nTesting text generation...")
    try:
        prompt = "The future of artificial intelligence is"
        generation = pruning_module.generate_text(pruning_module.params, prompt, max_length=50)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generation}")
        print("✅ Text generation successful")
    except Exception as e:
        print(f"❌ Text generation failed: {e}")
    
    # Test perplexity calculation
    print("\nTesting perplexity calculation...")
    try:
        text = "The model needs to process this text efficiently."
        perplexity = pruning_module.evaluate_perplexity(pruning_module.params, text)
        print(f"Text: {text}")
        print(f"Perplexity: {perplexity}")
        print("✅ Perplexity calculation successful")
    except Exception as e:
        print(f"❌ Perplexity calculation failed: {e}")
    
    # Test pruning
    print("\nTesting head pruning...")
    try:
        layer_idx = 0
        head_idx = 0
        original_params = pruning_module.params
        
        # Get active heads before pruning
        before_heads = determine_active_heads(pruning_module, original_params)
        
        # Prune one head
        print(f"Pruning head: layer={layer_idx}, head={head_idx}")
        pruned_params = pruning_module.prune_head(original_params, layer_idx, head_idx)
        
        # Get active heads after pruning
        after_heads = determine_active_heads(pruning_module, pruned_params)
        
        # Compare head counts
        print(f"Active heads before: {len(before_heads)}, after: {len(after_heads)}")
        
        # Head should be removed
        should_be_removed = (layer_idx, head_idx) not in after_heads
        print(f"Head {layer_idx}, {head_idx} removed: {should_be_removed}")
        
        if len(before_heads) > len(after_heads) and should_be_removed:
            print("✅ Head pruning successful")
        else:
            print("❌ Head pruning failed")
    except Exception as e:
        print(f"❌ Head pruning failed: {e}")
    
    print("\nAll tests complete!")
    return True

if __name__ == "__main__":
    main()