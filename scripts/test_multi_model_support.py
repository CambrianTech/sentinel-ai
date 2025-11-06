#!/usr/bin/env python
"""
Test script for multi-model support in fine-tuning.

This script verifies that different model architectures can be loaded and adapted
by our framework. It tests a small model of each supported architecture type.

Usage:
    python scripts/test_multi_model_support.py --models gpt2,opt,pythia,bloom,llama
"""

import os
import sys
import argparse
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.loaders.loader import load_baseline_model, load_adaptive_model
from scripts.finetune_pruned_model import create_optimizer_with_head_params, count_active_heads

# Map of model types to small examples
MODEL_EXAMPLES = {
    "gpt2": "distilgpt2",
    "opt": "facebook/opt-125m",
    "pythia": "EleutherAI/pythia-70m",
    "bloom": "bigscience/bloom-560m",
    "llama": "meta-llama/Llama-2-7b-hf"  # Need Hugging Face access token for this one
}

def test_model_loading(model_type, device="cuda"):
    """Test loading and adapting a model of the specified type."""
    if model_type not in MODEL_EXAMPLES:
        print(f"❌ Unknown model type: {model_type}")
        return False
    
    model_name = MODEL_EXAMPLES[model_type]
    print(f"\n--- Testing {model_type} model: {model_name} ---")
    
    try:
        # Load the baseline model
        print(f"Loading baseline model...")
        baseline_model = load_baseline_model(model_name, device)
        
        # Create the adaptive model
        print(f"Creating adaptive model...")
        adaptive_model = load_adaptive_model(model_name, baseline_model, device)
        
        # Check model structure
        total_heads, active_heads, ratio = count_active_heads(adaptive_model)
        print(f"Model has {active_heads}/{total_heads} active heads ({ratio:.2%})")
        
        # Test forward pass with a small input
        print("Testing forward pass...")
        input_ids = torch.randint(0, 1000, (1, 16), device=device)
        with torch.no_grad():
            outputs = adaptive_model(input_ids)
        print(f"Forward pass succeeded, output shape: {outputs.shape}")
        
        # Test optimizer creation
        print("Testing optimizer creation...")
        optimizer = create_optimizer_with_head_params(adaptive_model, 1e-5)
        param_groups = len(optimizer.param_groups)
        print(f"Created optimizer with {param_groups} parameter groups")
        
        # Free memory
        del baseline_model
        del adaptive_model
        del optimizer
        torch.cuda.empty_cache()
        
        print(f"✅ Successfully tested {model_type} model")
        return True
        
    except Exception as e:
        print(f"❌ Error testing {model_type} model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test multi-model support")
    parser.add_argument("--models", type=str, default="gpt2,opt,pythia,bloom", 
                       help="Comma-separated list of model types to test")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None,
                      help="Device to use (defaults to CUDA if available)")
    args = parser.parse_args()
    
    # Determine device
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Parse models to test
    model_types = [m.strip() for m in args.models.split(",")]
    
    # Run tests
    results = []
    for model_type in model_types:
        success = test_model_loading(model_type, device)
        results.append((model_type, success))
    
    # Print summary
    print("\n--- Results ---")
    for model_type, success in results:
        status = "✅ Passed" if success else "❌ Failed"
        print(f"{model_type}: {status}")
    
    # Exit with error code if any test failed
    failed = any(not success for _, success in results)
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())