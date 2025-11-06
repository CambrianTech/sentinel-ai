#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate sample outputs from each supported model.

Usage:
    python generate_samples.py
"""

import sys
import torch
from pathlib import Path
import textwrap

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from models.loaders.loader import load_baseline_model, load_adaptive_model
from transformers import AutoTokenizer


def generate_sample(model_name, prompt, max_length=200, device="cpu"):
    """Generate a sample from the specified model."""
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load baseline model
        print(f"Loading baseline model...")
        baseline_model = load_baseline_model(model_name, device)
        
        # Load adaptive model
        print(f"Loading adaptive model...")
        adaptive_model = load_adaptive_model(model_name, baseline_model, device, quiet=True)
        
        # Tokenize input
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # Calculate approximately how many tokens we need for 200 characters
        # Assuming ~4 characters per token on average
        target_new_tokens = 50  # This should give roughly 200 characters
        
        # Generate text
        print(f"Generating sample...")
        with torch.no_grad():
            output_ids = adaptive_model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_length=input_ids.size(1) + target_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1
            )
        
        # Decode output
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Get only the generated part (without the prompt)
        generated_only = generated_text[len(prompt):]
        
        # Ensure we have approximately 200 characters
        char_limit = 200
        if len(generated_only) > char_limit:
            generated_only = generated_only[:char_limit]
        
        # Print result with formatting
        print(f"\nPrompt: {prompt}")
        print(f"\nGenerated ({len(generated_only)} chars): ")
        print("-" * 70)
        print(textwrap.fill(generated_only, width=70))
        print("-" * 70)
        
        # Return just the generated part
        return generated_only
        
    except Exception as e:
        print(f"Error generating sample from {model_name}: {e}")
        return f"Error: {str(e)}"


def main():
    # Common prompt for all models
    prompt = "The adaptive transformer architecture enables AI systems to"
    
    # List of models to test
    models = [
        "distilgpt2",
        "gpt2",
        "gpt2-medium",
        "facebook/opt-125m",
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-160m",
        "bigscience/bloom-560m"
    ]
    
    # Generate and collect results
    results = {}
    for model_name in models:
        result = generate_sample(model_name, prompt)
        results[model_name] = result
    
    # Print summary table
    print("\n\n" + "="*100)
    print("SUMMARY OF MODEL OUTPUTS")
    print("="*100)
    
    for model_name, output in results.items():
        # Format for easier reading
        short_name = model_name.split('/')[-1]
        print(f"{short_name.ljust(15)}: {output[:100]}...")
    
    print("\nFull prompt: " + prompt)


if __name__ == "__main__":
    main()