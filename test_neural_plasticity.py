#!/usr/bin/env python
"""
Simple test script for the neural plasticity implementation
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom dataset loader
from sdata.dataset_loader import load_dataset
print("Dataset module imported successfully")

# Test the dataset loader with a tokenizer
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    dataset = load_dataset("test", tokenizer)
    print("Dataset loaded successfully")
    
    # Get evaluation samples
    samples = dataset.get_evaluation_samples(3)
    print(f"Evaluation samples: {samples}")
    
    print("Test completed successfully")
except Exception as e:
    print(f"Error: {e}")