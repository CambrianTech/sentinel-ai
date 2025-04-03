#!/usr/bin/env python
"""
Test script for NaN prevention mechanisms.

This script can be used to test the NaN prevention mechanisms locally
before deploying to Colab or other environments.

Usage:
    python -m utils.pruning.stability.test_nan_prevention

Options:
    --model MODEL    Model to test with (default: gpt2)
    --batch_size N   Batch size for testing (default: 2)
    --seq_length N   Sequence length for testing (default: 128)
    --verbose        Enable verbose logging
"""

import argparse
import logging
import sys
import numpy as np

from utils.pruning.stability import test_nan_safety

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test NaN prevention mechanisms")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model to test with (default: gpt2)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for testing (default: 2)")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length for testing (default: 128)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    return parser.parse_args()

def main():
    """Run tests for NaN prevention."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    
    # Log test parameters
    logging.info(f"Testing NaN prevention with parameters:")
    logging.info(f"  - Model: {args.model}")
    logging.info(f"  - Batch size: {args.batch_size}")
    logging.info(f"  - Sequence length: {args.seq_length}")
    
    # Run the test
    result = test_nan_safety(
        model_name=args.model,
        batch_size=args.batch_size,
        sequence_length=args.seq_length
    )
    
    # Report results
    if result:
        logging.info("All tests PASSED!")
        return 0
    else:
        logging.error("Tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())