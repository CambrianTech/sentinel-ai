#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify which models we can successfully load and run inference with.

Usage:
    python test_model_support.py
"""

import argparse
import sys
import torch
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from models.loaders.loader import load_baseline_model, load_adaptive_model


def test_model(model_name, device="cpu", verbose=True):
    """Test loading and inference with a specific model."""
    print(f"\n{'='*50}")
    print(f"Testing model: {model_name}")
    print(f"{'='*50}")
    
    results = {
        "model_name": model_name,
        "loaded_baseline": False,
        "loaded_adaptive": False,
        "inference_success": False,
        "error": None
    }
    
    try:
        # Step 1: Load baseline model
        print(f"Loading baseline model...")
        baseline_model = load_baseline_model(model_name, device)
        results["loaded_baseline"] = True
        
        # Step 2: Load adaptive model
        print(f"Loading adaptive model...")
        adaptive_model = load_adaptive_model(model_name, baseline_model, device, debug=verbose)
        results["loaded_adaptive"] = True
        
        # Step 3: Test inference
        print(f"Testing inference...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create sample input
        prompt = "The transformer model architecture revolutionized"
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        
        # Run inference
        with torch.no_grad():
            outputs = adaptive_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.size(1) + 10,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results["inference_success"] = True
        
        # Print success
        print(f"✅ Successfully loaded and ran inference:")
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        results["error"] = str(e)
    
    return results


def main():
    """Run tests on multiple models."""
    parser = argparse.ArgumentParser(description="Test model loading and inference")
    parser.add_argument("--device", type=str, default="cpu", 
                        help="Device to run on (default: cpu)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output")
    args = parser.parse_args()
    
    # Models to test from each family
    models_to_test = [
        # GPT-2 family
        "distilgpt2",
        "gpt2",
        "gpt2-medium",
        
        # OPT family
        "facebook/opt-125m",
        "facebook/opt-350m",
        
        # Pythia/GPT-NeoX family
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-160m",
        
        # BLOOM family
        "bigscience/bloom-560m",
        
        # Llama family (limited by API access)
        # "meta-llama/Llama-2-7b-hf",  # Requires token
    ]
    
    # Run tests
    results = []
    for model_name in models_to_test:
        result = test_model(model_name, device=args.device, verbose=args.verbose)
        results.append(result)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF MODEL SUPPORT")
    print("="*80)
    
    # Display as a table
    headers = ["Model", "Baseline", "Adaptive", "Inference", "Error"]
    # Calculate column widths
    widths = [
        max(len(headers[0]), max(len(r["model_name"]) for r in results)),
        max(len(headers[1]), 10),
        max(len(headers[2]), 10),
        max(len(headers[3]), 10),
        max(len(headers[4]), 20)
    ]
    
    # Print header
    header_row = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    separator = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    print(header_row)
    print(separator)
    
    # Print results
    for result in results:
        row = [
            result["model_name"],
            "✅" if result["loaded_baseline"] else "❌",
            "✅" if result["loaded_adaptive"] else "❌",
            "✅" if result["inference_success"] else "❌",
            result["error"][:20] + "..." if result["error"] and len(result["error"]) > 20 else (result["error"] or "")
        ]
        print("| " + " | ".join(str(r).ljust(w) for r, w in zip(row, widths)) + " |")
    
    # Print legend
    print("\n✅ = Success, ❌ = Failed")


if __name__ == "__main__":
    main()