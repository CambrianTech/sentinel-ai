#!/usr/bin/env python3
"""
Simple test script to check if the model loaders are working.
Comprehensive test that exercises the core functionality of the sentinel-ai framework.
"""
import sys
import argparse
import torch
import os
import time
from transformers import AutoTokenizer

# Add the parent directory to the Python path to allow importing the sentinel module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Use the sentinel package directly instead of the deprecated models.loaders.loader
from sentinel.models.loaders.loader import load_baseline_model, load_adaptive_model

def test_model_loaders(models, device="cpu", comprehensive=False):
    """
    Test loading different models with the respective loaders.
    
    Args:
        models: List of model names to test
        device: Device to load models on ('cpu' or 'cuda')
        comprehensive: If True, run more comprehensive tests
    """
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
            
            # Run comprehensive tests if requested
            if comprehensive:
                comp_results = run_comprehensive_tests(
                    model_name, baseline_model, adaptive_model, tokenizer, device
                )
                results[model_name]["comprehensive"] = comp_results
            
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

def run_comprehensive_tests(model_name, baseline_model, adaptive_model, tokenizer, device):
    """Run more comprehensive tests on the models to ensure core functionality."""
    print(f"\n📊 Running comprehensive tests for {model_name}...")
    
    results = {}
    
    # Test 1: Additional text generation with longer context
    try:
        prompt = "In a world where artificial intelligence has become self-aware,"
        input_ids = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate with both models
        print("  ⏱️  Generating longer text...")
        start_time = time.time()
        with torch.no_grad():
            baseline_output = baseline_model.generate(
                **input_ids, max_length=50, num_return_sequences=1, 
                pad_token_id=tokenizer.pad_token_id,
                temperature=0.8
            )
            adaptive_output = adaptive_model.generate(
                **input_ids, max_length=50, num_return_sequences=1, 
                pad_token_id=tokenizer.pad_token_id,
                temperature=0.8
            )
        gen_time = time.time() - start_time
        
        # Get output text
        baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
        adaptive_text = tokenizer.decode(adaptive_output[0], skip_special_tokens=True)
        
        print(f"  ✓ Generation time: {gen_time:.2f}s")
        print(f"  ✓ Baseline: {baseline_text[:100]}...")
        print(f"  ✓ Adaptive: {adaptive_text[:100]}...")
        
        results["long_generation"] = {
            "status": "✅ SUCCESS",
            "baseline_text": baseline_text,
            "adaptive_text": adaptive_text,
            "generation_time": gen_time
        }
    except Exception as e:
        print(f"  ❌ Long generation test failed: {str(e)}")
        results["long_generation"] = {
            "status": "❌ FAILED",
            "error": str(e)
        }
    
    # Test 2: Model attention mechanism (access internal attributes)
    try:
        print("  ⏱️  Testing model attention...")
        # Check if we can access attention gates in the adaptive model
        found_gates = False
        
        for name, module in adaptive_model.named_modules():
            if "gate" in name:
                found_gates = True
                gate_values = getattr(module, "weight", None) or getattr(module, "gate", None)
                if gate_values is not None:
                    print(f"  ✓ Found attention gates: {name} with shape {gate_values.shape}")
                    break
        
        if found_gates:
            results["attention_gates"] = {
                "status": "✅ SUCCESS",
                "found_gates": found_gates
            }
        else:
            print("  ⚠️  Could not find attention gates in the model")
            results["attention_gates"] = {
                "status": "⚠️ WARNING",
                "found_gates": found_gates
            }
    except Exception as e:
        print(f"  ❌ Attention test failed: {str(e)}")
        results["attention_gates"] = {
            "status": "❌ FAILED",
            "error": str(e)
        }
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model loaders")
    parser.add_argument("--models", type=str, default="gpt2",
                        help="Comma-separated list of models to test (e.g., gpt2,facebook/opt-125m)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to load models on")
    parser.add_argument("--comprehensive", action="store_true",
                        help="Run comprehensive tests")
    
    args = parser.parse_args()
    models = [model.strip() for model in args.models.split(",")]
    
    test_model_loaders(models, device=args.device, comprehensive=args.comprehensive)