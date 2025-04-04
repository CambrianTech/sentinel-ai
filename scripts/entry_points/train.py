#\!/usr/bin/env python
"""
Training entry point for the sentinel-ai package.

This script is a wrapper around the train.py file and provides the same functionality
but as an installable entry point.
"""

import sys
import os

# Add the sentinel-ai root directory to the Python path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

def main():
    """Run the main training function."""
    # Import here to avoid circular imports
    from train import main as _main
    return _main()

if __name__ == "__main__":
    main()
