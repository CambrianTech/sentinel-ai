#!/usr/bin/env python
"""
Simple test for HuggingFace model generation.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "distilgpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Print details about the model
print(f"Model: {model_name}")
print(f"Model type: {type(model)}")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

# Prepare input
prompt = "The future of AI is"
print(f"\nPrompt: {prompt}")
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
print("Generating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=30,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode and print output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nGenerated: {generated_text}")
print("\nDone!")