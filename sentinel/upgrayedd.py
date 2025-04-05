#!/usr/bin/env python
"""
Module wrapper for upgrayedd.py script.
Allows importing and using the upgrader programmatically.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from scripts
from scripts.upgrayedd import ModelUpgrader, main, banner

# Re-export for convenience
__all__ = ["ModelUpgrader", "main", "banner"]

# Example usage
if __name__ == "__main__":
    from scripts.upgrayedd import main
    sys.exit(main())