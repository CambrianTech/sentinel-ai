#!/usr/bin/env python
"""
Test script to verify model loading functionality.
"""

import os
import sys
import torch
from transformers import AutoTokenizer

# Add the root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions from the new namespace (sentinel)
print("Importing from sentinel namespace...")
from sentinel.models.loaders.loader import load_baseline_model, load_adaptive_model

# Test loading a simple model
model_name = "distilgpt2"
device = torch.device("cpu")

print(f"\nLoading baseline model: {model_name}")
baseline_model = load_baseline_model(model_name, device)
print(f"Baseline model loaded: {type(baseline_model)}")

print(f"\nConverting to adaptive model")
adaptive_model = load_adaptive_model(model_name, baseline_model, device)
print(f"Adaptive model loaded: {type(adaptive_model)}")

# Test text generation with the baseline model
prompt = "The future of AI is"
print(f"\nTokenizing prompt: '{prompt}'")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("\nGenerating with baseline model...")
with torch.no_grad():
    baseline_outputs = baseline_model.generate(
        **inputs,
        max_length=len(inputs.input_ids[0]) + 10,
        do_sample=True
    )
baseline_text = tokenizer.decode(baseline_outputs[0], skip_special_tokens=True)
print(f"Baseline output: {baseline_text}")

# Test text generation with the adaptive model
print("\nGenerating with adaptive model...")
with torch.no_grad():
    adaptive_outputs = adaptive_model.generate(
        **inputs,
        max_length=len(inputs.input_ids[0]) + 10,
        do_sample=True
    )
adaptive_text = tokenizer.decode(adaptive_outputs[0], skip_special_tokens=True)
print(f"Adaptive output: {adaptive_text}")

print("\nDone!")