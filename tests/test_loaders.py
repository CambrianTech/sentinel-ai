#!/usr/bin/env python
"""
Direct test of model loaders
"""
import os
import sys
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the loader functions
from sentinel.models.loaders.loader import load_baseline_model, load_adaptive_model

# Test models and prompts
test_configs = [
    {
        "model_name": "distilgpt2",
        "prompt": "The future of artificial intelligence is",
        "max_length": 30
    },
    {
        "model_name": "gpt2",
        "prompt": "Once upon a time there was a",
        "max_length": 30
    }
]

def test_model(model_name, prompt, max_length=30):
    """Test a model with direct loader calls"""
    print(f"\nTESTING MODEL: {model_name}")
    print("=" * 70)
    
    # Load the model
    print(f"Loading baseline model...")
    device = torch.device("cpu")
    baseline_model = load_baseline_model(model_name, device)
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test baseline model
    print(f"\nBASELINE MODEL TEST")
    print("-" * 70)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = baseline_model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id
        )
    baseline_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {baseline_text}")
    
    # Load and test adaptive model
    print(f"\nADAPTIVE MODEL TEST")
    print("-" * 70)
    print(f"Loading adaptive model...")
    adaptive_model = load_adaptive_model(model_name, baseline_model, device, quiet=True)
    
    with torch.no_grad():
        outputs = adaptive_model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id
        )
    adaptive_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {adaptive_text}")
    
    # Return generation results for comparison
    return {
        "model_name": model_name,
        "baseline_text": baseline_text,
        "adaptive_text": adaptive_text
    }

def main():
    """Run all tests"""
    results = []
    
    for config in test_configs:
        try:
            result = test_model(**config)
            results.append(result)
        except Exception as e:
            print(f"Error testing {config['model_name']}: {e}")
    
    # Print summary
    print("\nTEST SUMMARY")
    print("=" * 70)
    for result in results:
        print(f"Model: {result['model_name']}")
        print(f"  Baseline output is coherent: {'Yes' if len(result['baseline_text']) > len(config['prompt']) + 10 else 'No'}")
        print(f"  Adaptive output is coherent: {'Yes' if len(result['adaptive_text']) > len(config['prompt']) + 10 else 'No'}")
        
if __name__ == "__main__":
    main()