#!/usr/bin/env python
"""
Simple test script for verifying model loading and generation.
"""

import os
import sys
import torch
import traceback
from transformers import AutoTokenizer

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Python version:", sys.version)
print("Python path:", sys.path)
print("Current directory:", os.getcwd())

# Check if modules are where we expect them
try:
    # List available modules
    print("\nListing modules in models directory:")
    if os.path.exists("models"):
        print(os.listdir("models"))
    else:
        print("models/ directory not found")
    
    print("\nListing modules in sentinel directory:")
    if os.path.exists("sentinel"):
        print(os.listdir("sentinel"))
    else:
        print("sentinel/ directory not found")
        
    # List contents of models/loaders
    if os.path.exists("models/loaders"):
        print("\nListing modules in models/loaders:")
        print(os.listdir("models/loaders"))
    
    # List contents of sentinel/models/loaders
    if os.path.exists("sentinel/models/loaders"):
        print("\nListing modules in sentinel/models/loaders:")
        print(os.listdir("sentinel/models/loaders"))
except Exception as e:
    print(f"Error listing directories: {e}")

# Now attempt imports
try:
    # Try importing from the new namespace
    from sentinel.models.loaders.loader import load_baseline_model, load_adaptive_model
    print("\nSuccessfully imported from sentinel namespace")
except ImportError as e:
    print(f"\nImport error from sentinel namespace: {e}")
    traceback.print_exc()
    
    try:
        # Fallback to old imports
        print("\nFalling back to old import paths")
        from models.loaders.loader import load_baseline_model, load_adaptive_model
        print("Successfully imported from old namespace")
    except ImportError as e2:
        print(f"Import error from old namespace too: {e2}")
        traceback.print_exc()
        print("\nCannot proceed without loader functions. Exiting.")
        sys.exit(1)

# Test model loading with a simple model
model_name = "distilgpt2"
device = torch.device("cpu")

print(f"Loading tokenizer for {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading baseline model: {model_name}")
baseline_model = load_baseline_model(model_name, device)
print(f"Baseline model loaded: {type(baseline_model)}")

print("Converting to adaptive model")
adaptive_model = load_adaptive_model(model_name, baseline_model, device)
print(f"Adaptive model loaded: {type(adaptive_model)}")

# Generate text with baseline model
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("\nGenerating with baseline model...")
with torch.no_grad():
    baseline_outputs = baseline_model.generate(
        **inputs,
        max_length=len(inputs.input_ids[0]) + 20,
        do_sample=True,
        temperature=0.7
    )
baseline_text = tokenizer.decode(baseline_outputs[0], skip_special_tokens=True)
print(f"Baseline output: {baseline_text}")

print("\nGenerating with adaptive model...")
with torch.no_grad():
    adaptive_outputs = adaptive_model.generate(
        **inputs,
        max_length=len(inputs.input_ids[0]) + 20,
        do_sample=True,
        temperature=0.7
    )
adaptive_text = tokenizer.decode(adaptive_outputs[0], skip_special_tokens=True)
print(f"Adaptive output: {adaptive_text}")

print("\nTest completed successfully!")