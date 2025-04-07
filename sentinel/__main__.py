#!/usr/bin/env python
"""
Main entry point for Sentinel-AI package.
This allows running 'python -m sentinel' as a shortcut to upgrayedd.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    from scripts.upgrayedd import main
    sys.exit(main())