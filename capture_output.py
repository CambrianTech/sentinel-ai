#!/usr/bin/env python
import subprocess
import sys

cmd = [
    "python", "-c", 
    "from transformers import AutoModelForCausalLM, AutoTokenizer; model_name='distilgpt2'; tokenizer = AutoTokenizer.from_pretrained(model_name); model = AutoModelForCausalLM.from_pretrained(model_name); prompt='The future of AI is'; inputs = tokenizer(prompt, return_tensors='pt'); outputs = model.generate(**inputs, max_length=50); print('OUTPUT_BEGINS_HERE'); print(tokenizer.decode(outputs[0], skip_special_tokens=True)); print('OUTPUT_ENDS_HERE')"
]

# Run command and capture output
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
stdout, stderr = process.communicate()

# Find output between markers
if "OUTPUT_BEGINS_HERE" in stdout and "OUTPUT_ENDS_HERE" in stdout:
    start_marker = stdout.find("OUTPUT_BEGINS_HERE") + len("OUTPUT_BEGINS_HERE")
    end_marker = stdout.find("OUTPUT_ENDS_HERE")
    output = stdout[start_marker:end_marker].strip()
    print("Generated Text:")
    print(output)
else:
    print("Failed to capture output")