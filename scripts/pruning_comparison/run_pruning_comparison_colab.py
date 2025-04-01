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

# @title Display Visualizations
# @markdown Run this cell to display the experiment results
import glob
import json
import matplotlib.pyplot as plt
from IPython.display import display, Image
from pathlib import Path

def display_results():
    # Find the latest run directory
    latest_runs = sorted(glob.glob("validation_results/pruning_agency/run_*"))
    if not latest_runs:
        print("No results found. Run the experiment first.")
        return
        
    latest_run = latest_runs[-1]
    print(f"Displaying results from: {latest_run}")
    
    # Display temperature-specific visualizations
    temp_dirs = glob.glob(f"{latest_run}/temp_*")
    if temp_dirs:
        # Temperature-specific visualizations exist
        for temp_dir in sorted(temp_dirs):
            temp = Path(temp_dir).name.replace("temp_", "")
            print(f"\n== Results for temperature {temp} ==")
            
            # Show comprehensive summary
            comp_summary = f"{temp_dir}/comprehensive_summary.png"
            if os.path.exists(comp_summary):
                display(Image(comp_summary))
            
            # Show relative improvement
            rel_improvement = f"{temp_dir}/relative_improvement.png"
            if os.path.exists(rel_improvement):
                display(Image(rel_improvement))
        
        # Display temperature comparison charts
        print("\n== Temperature Comparisons ==")
        temp_comparison = f"{latest_run}/temperature_improvement_comparison.png"
        if os.path.exists(temp_comparison):
            display(Image(temp_comparison))
            
        heatmap = f"{latest_run}/improvement_heatmap.png"
        if os.path.exists(heatmap):
            display(Image(heatmap))
    else:
        # Single temperature visualizations (older format)
        print("\n== Results ==")
        for img_path in glob.glob(f"{latest_run}/*.png"):
            display(Image(img_path))
    
    # Print key findings if results file exists
    results_file = f"{latest_run}/pruning_comparison_results.json"
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            try:
                results = json.load(f)
                
                print("\n== Key Findings ==")
                # Check if new format with temperatures
                temp_keys = [k for k in results.keys() if k.startswith("temperature_")]
                
                if temp_keys:
                    # New format with temperatures
                    for temp_key in temp_keys:
                        temp = float(temp_key.split("_")[1])
                        print(f"\nTemperature {temp}:")
                        
                        pruning_levels = sorted([
                            int(l) for l in results[temp_key]["baseline"].keys() 
                            if l.isdigit() and int(l) > 0
                        ])
                        
                        for level in pruning_levels:
                            try:
                                baseline_speed = results[temp_key]["baseline"][str(level)]["tokens_per_second"]["mean"]
                                agency_speed = results[temp_key]["agency"][str(level)]["tokens_per_second"]["mean"]
                                
                                baseline_ppl = results[temp_key]["baseline"][str(level)]["perplexity"]["mean"]
                                agency_ppl = results[temp_key]["agency"][str(level)]["perplexity"]["mean"]
                                
                                speed_imp = ((agency_speed / baseline_speed) - 1) * 100
                                quality_imp = ((baseline_ppl / agency_ppl) - 1) * 100
                                
                                print(f"  At {level}% pruning: Agency is {speed_imp:.1f}% faster and {quality_imp:.1f}% better quality")
                            except (KeyError, TypeError) as e:
                                # Skip levels with missing data
                                continue
                else:
                    # Old format without temperatures
                    print("\nAggregate results:")
                    
                    pruning_levels = sorted([
                        int(l) for l in results["baseline"].keys() 
                        if l.isdigit() and int(l) > 0
                    ])
                    
                    for level in pruning_levels:
                        try:
                            baseline_speed = results["baseline"][str(level)]["tokens_per_second"]
                            agency_speed = results["agency"][str(level)]["tokens_per_second"]
                            
                            baseline_ppl = results["baseline"][str(level)]["perplexity"]
                            agency_ppl = results["agency"][str(level)]["perplexity"]
                            
                            speed_imp = ((agency_speed / baseline_speed) - 1) * 100
                            quality_imp = ((baseline_ppl / agency_ppl) - 1) * 100
                            
                            print(f"  At {level}% pruning: Agency is {speed_imp:.1f}% faster and {quality_imp:.1f}% better quality")
                        except (KeyError, TypeError) as e:
                            # Skip levels with missing data
                            continue
            except json.JSONDecodeError:
                print("Error parsing results file.")

# Call the function to display results
display_results()

# @title Save Results to Google Drive (Optional)
# @markdown Run this after the experiment completes to save results to Google Drive
mount_drive = False  # @param {type:"boolean"}
drive_folder = "Sentinel_AI_Results"  # @param {type:"string"}

if mount_drive:
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Create folder if it doesn't exist
    drive_path = f"/content/drive/My Drive/{drive_folder}"
    os.makedirs(drive_path, exist_ok=True)
    
    # Copy results to Drive
    latest_results = sorted(glob.glob("validation_results/pruning_agency/run_*"))
    if latest_results:
        latest = latest_results[-1]
        print(f"Copying results to Google Drive: {drive_path}")
        !cp -r {latest} {drive_path}/pruning_results_{time.strftime("%Y%m%d_%H%M%S")}
        print("Results successfully copied to Google Drive")
    else:
        print("No results found to copy. Run the experiment first.")