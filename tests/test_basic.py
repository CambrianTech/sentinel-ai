#!/usr/bin/env python
"""
Test baseline model loading and generation with a simple GPT-2 model.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Basic model test
model_name = "distilgpt2"
device = torch.device("cpu")

print(f"Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Test generation
prompt = "The future of AI is"
print(f"Prompt: {prompt}")

inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=0.7
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")
print("Test completed!")