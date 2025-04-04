#!/usr/bin/env python
"""
Very minimal test script using direct module imports.
"""
import os
import sys
import torch
import inspect
from transformers import AutoTokenizer

print("Current directory:", os.getcwd())

# Try Direct Imports
try:
    # Import model loaders 
    from models.loaders.loader import load_baseline_model, load_adaptive_model
    print("Successfully imported from old namespace")
    
    # Test model loading
    model_name = "distilgpt2"
    device = torch.device("cpu")
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading baseline model: {model_name}")
    baseline_model = load_baseline_model(model_name, device)
    print("Baseline model loaded successfully")
    
    print(f"Converting to adaptive model")
    adaptive_model = load_adaptive_model(model_name, baseline_model, device)
    print("Adaptive model loaded successfully")
    
    # Test generation
    prompt = "The future of AI is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print("Generating with baseline model...")
    with torch.no_grad():
        baseline_outputs = baseline_model.generate(
            **inputs,
            max_length=len(inputs.input_ids[0]) + 10,
            do_sample=True,
            temperature=0.7,
        )
    baseline_text = tokenizer.decode(baseline_outputs[0], skip_special_tokens=True)
    print(f"Baseline output: {baseline_text}")
    
    print("Generating with adaptive model...")
    with torch.no_grad():
        adaptive_outputs = adaptive_model.generate(
            **inputs,
            max_length=len(inputs.input_ids[0]) + 10,
            do_sample=True,
            temperature=0.7,
        )
    adaptive_text = tokenizer.decode(adaptive_outputs[0], skip_special_tokens=True)
    print(f"Adaptive output: {adaptive_text}")
    
    # Examine model structure
    print("\nExamining adaptive model:")
    print("Type:", type(adaptive_model))
    
    if hasattr(adaptive_model, "blocks"):
        print(f"Number of blocks: {len(adaptive_model.blocks)}")
        print(f"Number of heads per block: {adaptive_model.blocks[0]['attn'].num_heads}")
    
    # Print how the token-boosting logic was fixed
    try:
        forward_func = adaptive_model.forward
        print("\nAdaptive model forward function source:")
        print(inspect.getsource(forward_func))
    except Exception as e:
        print(f"Could not get source for forward method: {e}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()