#!/usr/bin/env python
from transformers import AutoModelForCausalLM, AutoTokenizer

models = ["distilgpt2", "gpt2"]
prompt = "The future of AI is"

for model_name in models:
    print(f"\nTesting model: {model_name}")
    print("-" * 50)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate text
    outputs = model.generate(
        **inputs,
        max_length=50,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and print result
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print("-" * 50)