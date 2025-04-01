#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Colab Wrapper for Pruning Efficacy Comparison

This script:
1. Sets up a Google Colab environment
2. Clones the Sentinel-AI repository
3. Installs dependencies 
4. Runs the pruning comparison experiment with optimal settings for T4 GPU

Usage:
1. Upload this script to Google Colab
2. Run all cells
3. Results will be saved to Google Drive if mounted

Note: This script is designed to be run in a Google Colab notebook with T4 GPU.
"""

# @title Setup Environment
# @markdown Run this cell to set up the environment for pruning comparison

import os
import sys
import time
import subprocess
import warnings
warnings.filterwarnings('ignore')

# Check if running in Colab
IN_COLAB = 'google.colab' in sys.modules
if not IN_COLAB:
    print("This script is designed to be run in Google Colab")
    print("For local execution, use the pruning_agency_comparison.py script directly")

# Check for GPU
if IN_COLAB:
    gpu_info = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    if 'T4' in gpu_info:
        print("T4 GPU detected")
    elif 'GPU' in gpu_info:
        print("GPU detected, but not T4. Script is optimized for T4.")
    else:
        print("No GPU detected! This script requires a GPU runtime.")
        print("Go to Runtime > Change runtime type and select GPU")
        raise SystemError("No GPU detected")

# Clone the repository if needed
if not os.path.exists('sentinel-ai'):
    print("Cloning Sentinel-AI repository...")
    
    # Clone the repo
    !git clone https://github.com/CambrianTech/sentinel-ai.git
    
    # Move into the directory
    %cd sentinel-ai
else:
    # Move into the directory if not already there
    if 'sentinel-ai' not in os.getcwd():
        %cd sentinel-ai
    
    # Pull latest changes
    print("Updating Sentinel-AI repository...")
    !git pull

# Install dependencies
print("Installing dependencies...")
!pip install -q torch transformers matplotlib seaborn numpy psutil

# @title Configure and Run Pruning Comparison
# @markdown Adjust these parameters for your experiment

# @markdown ### Model Configuration
model_name = "gpt2"  # @param ["gpt2", "distilgpt2"]
device = "cuda"  # @param ["cuda", "cpu"]
precision = "float16"  # @param ["float32", "float16", "bfloat16"]

# @markdown ### Pruning Configuration
pruning_method = "entropy"  # @param ["entropy", "random", "magnitude"]
pruning_levels = "0,10,20,30,40,50,60,70"  # @param {type:"string"}

# @markdown ### Generation Parameters
num_tokens = 50  # @param {type:"slider", min:10, max:200, step:10}
temperatures = "0.7,1.0"  # @param {type:"string"}
max_prompts = 5  # @param {type:"slider", min:1, max:20, step:1}

# @markdown ### Experiment Configuration
iterations = 3  # @param {type:"slider", min:1, max:10, step:1}
save_outputs = True  # @param {type:"boolean"}
memory_logging = True  # @param {type:"boolean"}

# Create command
cmd = [
    "python", "scripts/pruning_comparison/pruning_agency_comparison.py",
    f"--model_name={model_name}",
    f"--device={device}",
    f"--precision={precision}",
    f"--pruning_method={pruning_method}",
    f"--pruning_levels={pruning_levels}",
    f"--num_tokens={num_tokens}",
    f"--temperatures={temperatures}",
    f"--max_prompts={max_prompts}",
    f"--iterations={iterations}"
]

if save_outputs:
    cmd.append("--save_outputs")
    
if memory_logging:
    cmd.append("--memory_logging")
    
# Print and execute command
print("Running pruning comparison with command:")
print(" ".join(cmd))
print("\n" + "="*80 + "\n")

start_time = time.time()
!{" ".join(cmd)}
end_time = time.time()

print(f"\nExperiment completed in {end_time - start_time:.2f} seconds")

# @title Save Results to Google Drive
# @markdown Run this after the experiment completes to save results to Google Drive
mount_drive = True  # @param {type:"boolean"}
drive_folder = "Sentinel_AI_Results"  # @param {type:"string"}

if mount_drive:
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Create folder if it doesn't exist
    drive_path = f"/content/drive/My Drive/{drive_folder}"
    os.makedirs(drive_path, exist_ok=True)
    
    # Copy results to Drive
    latest_results = "validation_results/pruning_agency/latest"
    if os.path.exists(latest_results):
        print(f"Copying results to Google Drive: {drive_path}")
        !cp -r {latest_results} {drive_path}/pruning_results_{time.strftime("%Y%m%d_%H%M%S")}
        print("Results successfully copied to Google Drive")
    else:
        print("No results found to copy. Run the experiment first.")