#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CPU Test Script for Pruning Comparison

This script runs a minimal test of the pruning comparison framework
on CPU with reduced parameters for quick validation.

Usage:
    python scripts/pruning_comparison/run_pruning_cpu_test.py

Optional arguments:
    --debug: Run with even fewer parameters for faster debugging
    --help: Show this help message
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Run a minimal CPU test of the pruning comparison framework")
    parser.add_argument("--debug", action="store_true", help="Run with even fewer parameters for faster debugging")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure working directory is the project root
    project_root = Path(__file__).parent.parent.parent.absolute()
    os.chdir(project_root)
    
    # Ensure we're working with the latest code from main branch
    try:
        if os.path.exists(".git"):
            print("Updating from main branch...")
            subprocess.run(["git", "fetch", "origin", "main"], check=False)
            subprocess.run(["git", "merge", "origin/main"], check=False)
    except Exception as e:
        print(f"Note: Unable to update from main branch: {e}")
        print("Continuing with current code...")
    
    # Ensure output directory exists
    output_dir = Path("validation_results/pruning_agency")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure prompts file exists
    if not os.path.exists('datasets/eval_prompts.txt'):
        print("Creating sample evaluation prompts...")
        os.makedirs('datasets', exist_ok=True)
        
        sample_prompts = [
            "Write a short summary of artificial intelligence.",
            "Explain how neural networks function in simple terms.",
            "What are the ethical implications of large language models?",
            "Describe the concept of attention in neural networks."
        ]
        
        with open('datasets/eval_prompts.txt', 'w') as f:
            f.write('\n'.join(sample_prompts))
        print(f"Created prompt file with {len(sample_prompts)} prompts")
    
    # Configure test parameters
    if args.debug:
        # Ultra minimal for fast debugging
        cmd = [
            "python", "scripts/pruning_comparison/pruning_agency_comparison.py",
            "--model_name=gpt2",
            "--device=cpu",
            "--pruning_levels=0,50",  # Just baseline and 50% pruning
            "--num_tokens=10",        # Generate very few tokens
            "--max_prompts=1",        # Use only 1 prompt
            "--iterations=1",         # Single iteration
            "--save_outputs"          # Save the generated text
        ]
        print("Running in DEBUG mode with minimal parameters")
    else:
        # Standard CPU test
        cmd = [
            "python", "scripts/pruning_comparison/pruning_agency_comparison.py",
            "--model_name=gpt2",
            "--device=cpu",
            "--pruning_levels=0,20,50",  # Baseline and two pruning levels
            "--num_tokens=30",          # Generate fewer tokens
            "--max_prompts=2",          # Use only 2 prompts
            "--iterations=1",           # Single iteration
            "--save_outputs",           # Save the generated text
            "--output_dir=validation_results/pruning_agency"
        ]
    
    # Print and execute command
    print("\n==== Pruning Comparison CPU Test ====")
    print("Running pruning comparison CPU test with command:")
    print(" ".join(cmd))
    print("\n" + "="*80 + "\n")
    
    start_time = time.time()
    subprocess.run(cmd)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"\nCPU test completed in {duration:.2f} seconds")
    
    # If successful, suggest next steps
    if duration > 0:  # Simple check that the process ran
        print("\nNext steps:")
        print("1. Examine the results in validation_results/pruning_agency/latest/")
        print("2. If the test was successful, run the full comparison on Colab with:")
        print("   python scripts/pruning_comparison/run_pruning_comparison_colab.py")
        print("\nTo run an even quicker test for debugging, use:")
        print("   python scripts/pruning_comparison/run_pruning_cpu_test.py --debug")

if __name__ == "__main__":
    main()