#!/usr/bin/env python
"""
Test dataset loading with workaround for circular imports.

This script tests the loading of datasets using our workaround for circular imports,
which is implemented in sentinel.upgrayedd.utils.data.
"""

import os
import sys
import torch
from transformers import AutoTokenizer

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the data module from sentinel.upgrayedd
from sentinel.upgrayedd.utils.data import load_and_prepare_data

def main():
    """Main test function."""
    print("Testing dataset loading with circular import workaround")
    
    # Load a small tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
    # Set pad token for GPT-2 models which don't have one by default
    if tokenizer.pad_token is None:
        print("Setting pad token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test loading with dummy data (fallback if dataset loading fails)
    print("\nTesting dummy data fallback:")
    train_loader, val_loader = load_and_prepare_data(
        "dummy_dataset_that_doesnt_exist",
        tokenizer,
        batch_size=2
    )
    
    # Print batch information
    print(f"Train loader has {len(train_loader)} batches")
    print(f"Val loader has {len(val_loader)} batches")
    
    # Check a sample batch
    print("\nSample batch from train loader:")
    sample_batch = next(iter(train_loader))
    for i, tensor in enumerate(sample_batch):
        print(f"Tensor {i} shape: {tensor.shape}")
    
    # Test with an actual dataset
    print("\nTesting with actual 'wikitext' dataset:")
    try:
        wiki_train, wiki_val = load_and_prepare_data(
            "wikitext",
            tokenizer,
            batch_size=2
        )
        print(f"WikiText train loader has {len(wiki_train)} batches")
        print(f"WikiText val loader has {len(wiki_val)} batches")
        success = True
    except Exception as e:
        print(f"Error loading WikiText: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    # Return status
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())