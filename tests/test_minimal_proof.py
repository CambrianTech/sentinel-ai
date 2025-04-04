#!/usr/bin/env python
"""
Minimal proof test for Sentinel-AI model functionality.
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

print("TESTING MODEL FUNCTIONALITY")
print("==========================")

# Load a basic HuggingFace model
model_name = "distilgpt2"
print(f"Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate text with the model
prompt = "The future of AI is"
print(f"Prompt: '{prompt}'")

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=30,
    do_sample=True,
    temperature=0.7
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")

if len(generated_text) > len(prompt) + 5:
    print("\n✅ SUCCESS: Model successfully generated text")
else:
    print("\n❌ FAILURE: Model failed to generate text properly")