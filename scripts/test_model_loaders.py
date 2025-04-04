#!/usr/bin/env python3
"""
Simple test script to check if the model loaders are working.
"""
import sys
import argparse
import torch
import os
from transformers import AutoTokenizer

# Add the parent directory to the Python path to allow importing the sentinel module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.loaders.loader import load_baseline_model, load_adaptive_model

def test_model_loaders(models, device="cpu"):
    """Test loading different models with the respective loaders."""
    results = {}
    
    for model_name in models:
        print(f"\n✓ Testing model: {model_name}")
        try:
            # Step 1: Load the tokenizer
            print(f"  Loading tokenizer for {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"  ✓ Tokenizer loaded successfully!")
                
            # Step 2: Load the baseline model
            print(f"  Loading baseline model {model_name}...")
            baseline_model = load_baseline_model(model_name, device=device)
            print(f"  ✓ Baseline model loaded successfully!")
            
            # Step 3: Load the adaptive model
            print(f"  Converting to adaptive model...")
            adaptive_model = load_adaptive_model(model_name, baseline_model, device=device)
            print(f"  ✓ Adaptive model loaded successfully!")
            
            # Step 4: Test text generation
            print(f"  Testing text generation...")
            prompt = "Once upon a time,"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                # Generate text with both models
                baseline_output = baseline_model.generate(
                    **inputs, max_length=25, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id
                )
                adaptive_output = adaptive_model.generate(
                    **inputs, max_length=25, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id
                )
            
            # Step 5: Decode outputs
            baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
            adaptive_text = tokenizer.decode(adaptive_output[0], skip_special_tokens=True)
            
            print(f"  ✓ Baseline output: {baseline_text}")
            print(f"  ✓ Adaptive output: {adaptive_text}")
            
            results[model_name] = {
                "status": "✅ SUCCESS",
                "baseline_text": baseline_text,
                "adaptive_text": adaptive_text
            }
            
        except Exception as e:
            print(f"  ❌ ERROR: {str(e)}")
            results[model_name] = {
                "status": "❌ FAILED",
                "error": str(e)
            }
    
    # Print summary
    print("\n" + "="*80)
    print(" MODEL LOADER TEST SUMMARY ".center(80, "="))
    print("="*80)
    
    for model_name, result in results.items():
        print(f"{model_name}: {result['status']}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model loaders")
    parser.add_argument("--models", type=str, default="gpt2",
                        help="Comma-separated list of models to test (e.g., gpt2,facebook/opt-125m)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to load models on")
    
    args = parser.parse_args()
    models = [model.strip() for model in args.models.split(",")]
    
    test_model_loaders(models, device=args.device)