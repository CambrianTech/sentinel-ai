#!/usr/bin/env python
"""
Test model loaders with output capture to filter warnings
"""
import subprocess
import sys
import os

def run_test(model_name, prompt, max_length=30):
    """Run test on a single model and capture output"""
    test_script = f"""
import os
import sys
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the loader functions
from sentinel.models.loaders.loader import load_baseline_model, load_adaptive_model

# Parameters
model_name = "{model_name}"
prompt = "{prompt}"
max_length = {max_length}

print("TEST_START")
print(f"Testing model: {{model_name}}")

# Load the model
device = torch.device("cpu")
baseline_model = load_baseline_model(model_name, device)

# Load tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Test baseline model
print("BASELINE_START")
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = baseline_model.generate(
        **inputs,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id
    )
baseline_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prompt: {{prompt}}")
print(f"Generated: {{baseline_text}}")
print("BASELINE_END")

# Load and test adaptive model
print("ADAPTIVE_START")
adaptive_model = load_adaptive_model(model_name, baseline_model, device, quiet=True)

with torch.no_grad():
    outputs = adaptive_model.generate(
        **inputs,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id
    )
adaptive_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prompt: {{prompt}}")
print(f"Generated: {{adaptive_text}}")
print("ADAPTIVE_END")
print("TEST_END")
"""

    # Create temporary script file
    script_file = "temp_test_script.py"
    with open(script_file, "w") as f:
        f.write(test_script)
    
    # Run script and capture output
    try:
        output = subprocess.check_output(["python", script_file], stderr=subprocess.STDOUT, text=True)
        
        # Parse output to get test results
        results = {}
        if "TEST_START" in output and "TEST_END" in output:
            # Extract baseline results
            if "BASELINE_START" in output and "BASELINE_END" in output:
                baseline_start = output.find("BASELINE_START") + len("BASELINE_START")
                baseline_end = output.find("BASELINE_END")
                baseline_output = output[baseline_start:baseline_end].strip()
                
                # Extract generated text
                if "Generated:" in baseline_output:
                    generated_idx = baseline_output.find("Generated:") + len("Generated:")
                    results["baseline_text"] = baseline_output[generated_idx:].strip()
            
            # Extract adaptive results
            if "ADAPTIVE_START" in output and "ADAPTIVE_END" in output:
                adaptive_start = output.find("ADAPTIVE_START") + len("ADAPTIVE_START")
                adaptive_end = output.find("ADAPTIVE_END")
                adaptive_output = output[adaptive_start:adaptive_end].strip()
                
                # Extract generated text
                if "Generated:" in adaptive_output:
                    generated_idx = adaptive_output.find("Generated:") + len("Generated:")
                    results["adaptive_text"] = adaptive_output[generated_idx:].strip()
            
            results["success"] = True
        else:
            results["success"] = False
            results["error"] = "Test markers not found in output"
        
        # Clean up
        os.remove(script_file)
        return results
    
    except subprocess.CalledProcessError as e:
        # Clean up
        if os.path.exists(script_file):
            os.remove(script_file)
        
        return {
            "success": False,
            "error": str(e),
            "output": e.output
        }

def main():
    """Run tests on multiple models"""
    tests = [
        {"model_name": "distilgpt2", "prompt": "The future of artificial intelligence is", "max_length": 30},
        {"model_name": "gpt2", "prompt": "Once upon a time there was a", "max_length": 30},
        {"model_name": "facebook/opt-125m", "prompt": "The meaning of life is", "max_length": 30},
        {"model_name": "EleutherAI/pythia-70m", "prompt": "In a world where robots have become", "max_length": 30}
    ]
    
    results = []
    
    for test in tests:
        print(f"\nTesting model: {test['model_name']}")
        print("-" * 70)
        
        result = run_test(**test)
        result["model_name"] = test["model_name"]
        result["prompt"] = test["prompt"]
        
        if result["success"]:
            print("✅ Test completed successfully")
            
            # Baseline model output
            if "baseline_text" in result:
                print("\nBaseline model output:")
                print(f"  {result['baseline_text']}")
            else:
                print("❌ Baseline model output not found")
            
            # Adaptive model output
            if "adaptive_text" in result:
                print("\nAdaptive model output:")
                print(f"  {result['adaptive_text']}")
            else:
                print("❌ Adaptive model output not found")
        else:
            print(f"❌ Test failed: {result.get('error', 'Unknown error')}")
        
        results.append(result)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for result in results:
        model_name = result["model_name"]
        prompt = result["prompt"]
        
        print(f"\nModel: {model_name}")
        if result["success"]:
            # Check if baseline output extends beyond prompt
            if "baseline_text" in result and result["baseline_text"].startswith(prompt):
                baseline_extension = result["baseline_text"][len(prompt):].strip()
                baseline_ok = len(baseline_extension) > 10
            else:
                baseline_ok = False
                
            # Check if adaptive output extends beyond prompt
            if "adaptive_text" in result and result["adaptive_text"].startswith(prompt):
                adaptive_extension = result["adaptive_text"][len(prompt):].strip()
                adaptive_ok = len(adaptive_extension) > 10
            else:
                adaptive_ok = False
            
            print(f"  Baseline output coherent: {'✅ Yes' if baseline_ok else '❌ No'}")
            print(f"  Adaptive output coherent: {'✅ Yes' if adaptive_ok else '❌ No'}")
        else:
            print("  ❌ Test failed")

if __name__ == "__main__":
    main()