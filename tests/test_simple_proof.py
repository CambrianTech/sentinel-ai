#!/usr/bin/env python
"""
Simple proof-of-concept test for Sentinel-AI model functionality.
Tests baseline, adaptive, and pruned models to verify cross-model compatibility.
"""

import os
import sys
import torch
import logging
import numpy as np
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Setup paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# Import model loaders
try:
    from sentinel.models.loaders.loader import load_baseline_model, load_adaptive_model
    print("Using sentinel namespace imports")
except ImportError:
    try:
        from models.loaders.loader import load_baseline_model, load_adaptive_model
        print("Using models directory imports")
    except ImportError:
        print("ERROR: Could not import loader functions")
        sys.exit(1)

def apply_pruning(model, pruning_level=0.5):
    """Apply random pruning to a model."""
    if not hasattr(model, "blocks"):
        return model
    
    num_layers = len(model.blocks)
    num_heads = model.blocks[0]["attn"].num_heads
    total_heads = num_layers * num_heads
    heads_to_prune = int(total_heads * pruning_level)
    
    print(f"Pruning {heads_to_prune} of {total_heads} heads ({pruning_level*100:.0f}%)")
    
    # Get all heads
    all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    
    # Randomly select heads to prune
    indices_to_prune = np.random.choice(len(all_heads), heads_to_prune, replace=False)
    
    # Apply pruning
    with torch.no_grad():
        for idx in indices_to_prune:
            layer_idx, head_idx = all_heads[idx]
            model.blocks[layer_idx]["attn"].gate[head_idx] = torch.tensor(0.001, device=model.device)
    
    return model

def test_model(model_name, prompt="The future of AI is"):
    """Test a model with baseline, adaptive, and pruned configurations."""
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load baseline model
    print(f"Loading baseline model: {model_name}")
    baseline_model = load_baseline_model(model_name, device)
    
    # Test baseline generation
    print("\nGenerating with baseline model:")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = baseline_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=30,
            do_sample=True,
            temperature=0.7
        )
    baseline_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  {baseline_text}")
    
    # Load adaptive model
    print("\nConverting to adaptive model")
    adaptive_model = load_adaptive_model(model_name, baseline_model, device)
    
    # Test adaptive generation
    print("\nGenerating with adaptive model:")
    with torch.no_grad():
        outputs = adaptive_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=30,
            do_sample=True,
            temperature=0.7
        )
    adaptive_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  {adaptive_text}")
    
    # Apply pruning and test
    print("\nApplying pruning to model")
    pruned_model = apply_pruning(adaptive_model)
    
    # Test pruned generation
    print("\nGenerating with pruned model:")
    with torch.no_grad():
        outputs = pruned_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=30,
            do_sample=True,
            temperature=0.7
        )
    pruned_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  {pruned_text}")
    
    # Assess results
    print("\nResults:")
    print(f"  Baseline text length: {len(baseline_text)} chars")
    print(f"  Adaptive text length: {len(adaptive_text)} chars")
    print(f"  Pruned text length: {len(pruned_text)} chars")
    
    # Success is measured by all three models producing text
    baseline_ok = len(baseline_text) > len(prompt) + 5
    adaptive_ok = len(adaptive_text) > len(prompt) + 5  
    pruned_ok = len(pruned_text) > len(prompt) + 5
    
    all_ok = baseline_ok and adaptive_ok and pruned_ok
    
    if all_ok:
        print("\n✅ SUCCESS: All models successfully generated text")
    else:
        print("\n❌ FAILURE: Some models failed to generate text properly")
        if not baseline_ok:
            print("  - Baseline model failed")
        if not adaptive_ok:
            print("  - Adaptive model failed")
        if not pruned_ok:
            print("  - Pruned model failed")
    
    return all_ok

def main():
    # Test each model family
    test_models = [
        "distilgpt2"  # Quick test with GPT-2
    ]
    
    results = {}
    for model in test_models:
        results[model] = test_model(model)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*60}")
    for model, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{model}: {status}")

if __name__ == "__main__":
    main()