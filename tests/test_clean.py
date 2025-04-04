#!/usr/bin/env python
"""
Test script with stderr suppression to see actual program flow.
"""
import os
import sys
import torch
import contextlib
import io
import tempfile
from transformers import AutoTokenizer

# Function to temporarily suppress stderr messages
@contextlib.contextmanager
def suppress_stderr():
    # Create a temporary file to redirect stderr
    stderr_fd = sys.stderr.fileno()
    with tempfile.NamedTemporaryFile(mode='w+') as temp:
        stderr_backup = os.dup(stderr_fd)
        sys.stderr.flush()
        try:
            os.dup2(temp.fileno(), stderr_fd)
            yield
        finally:
            sys.stderr.flush()
            os.dup2(stderr_backup, stderr_fd)
            os.close(stderr_backup)

# Print basic environment info
print("Python version:", sys.version)
print("Current directory:", os.getcwd())

try:
    # Import model loaders - suppress deprecation warnings
    with suppress_stderr():
        from models.loaders.loader import load_baseline_model, load_adaptive_model
    print("Successfully imported loader functions")
    
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
    
    # Test generation with error capture
    prompt = "The future of AI is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print("Generating with baseline model...")
    with suppress_stderr(), torch.no_grad():
        baseline_outputs = baseline_model.generate(
            **inputs,
            max_length=len(inputs.input_ids[0]) + 10,
            do_sample=True,
            temperature=0.7,
        )
    baseline_text = tokenizer.decode(baseline_outputs[0], skip_special_tokens=True)
    print(f"Baseline output: {baseline_text}")
    
    print("Generating with adaptive model...")
    with suppress_stderr(), torch.no_grad():
        adaptive_outputs = adaptive_model.generate(
            **inputs,
            max_length=len(inputs.input_ids[0]) + 10,
            do_sample=True,
            temperature=0.7,
        )
    adaptive_text = tokenizer.decode(adaptive_outputs[0], skip_special_tokens=True)
    print(f"Adaptive output: {adaptive_text}")
    
    # Check for successful text generation
    if len(adaptive_text.strip()) > len(prompt.strip()):
        print("\n✅ Adaptive model successfully generated text")
        
        # Check the quality of generated text
        if len(adaptive_text) > len(baseline_text) * 0.7:
            print("✅ Adaptive model output is comparable to baseline in length")
        else:
            print("⚠️ Adaptive model output is significantly shorter than baseline")
            
        # Look for degeneration patterns in adaptive output
        tokens = adaptive_text[len(prompt):].split()
        unique_tokens = set(tokens)
        if len(unique_tokens) < len(tokens) * 0.5:
            print("⚠️ Adaptive model output shows repetition")
        else:
            print("✅ Adaptive model output shows good diversity")
    else:
        print("\n❌ Adaptive model failed to generate meaningful text")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()