#!/usr/bin/env python
"""
Minimal test script for verifying imports and basic functionality.
"""

import os
import sys
import torch
import traceback

# Add the root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Python version:", sys.version)
print("Current directory:", os.getcwd())

# Check directory structure
print("\nExploring directory structure:")
if os.path.exists("models"):
    print("models/ exists, contents:", os.listdir("models"))
else:
    print("models/ directory doesn't exist")

if os.path.exists("sentinel"):
    print("sentinel/ exists, contents:", os.listdir("sentinel"))
else:
    print("sentinel/ directory doesn't exist")

if os.path.exists("sentinel/models"):
    print("sentinel/models/ exists, contents:", os.listdir("sentinel/models"))

# Try importing modules directly
print("\nAttempting direct imports:")
try:
    import models
    print("Successfully imported 'models' package")
except ImportError as e:
    print(f"Failed to import 'models': {e}")

try:
    import sentinel
    print("Successfully imported 'sentinel' package")
except ImportError as e:
    print(f"Failed to import 'sentinel': {e}")

# Try importing loader modules
print("\nAttempting to import loaders:")
try:
    from models.loaders import loader
    print("Successfully imported from models.loaders.loader")
    print("Functions in loader:", [f for f in dir(loader) if not f.startswith('_')])
except ImportError as e:
    print(f"Failed to import from models.loaders.loader: {e}")
    traceback.print_exc()

try:
    from sentinel.models.loaders import loader as sentinel_loader
    print("Successfully imported from sentinel.models.loaders.loader")
    print("Functions in sentinel_loader:", [f for f in dir(sentinel_loader) if not f.startswith('_')])
except ImportError as e:
    print(f"Failed to import from sentinel.models.loaders.loader: {e}")
    traceback.print_exc()

# Try imports for individual functions
print("\nAttempting to import specific functions:")
try:
    from models.loaders.loader import load_baseline_model
    print("Successfully imported load_baseline_model from old path")
except ImportError as e:
    print(f"Failed to import load_baseline_model from old path: {e}")

try:
    from sentinel.models.loaders.loader import load_baseline_model
    print("Successfully imported load_baseline_model from new path")
except ImportError as e:
    print(f"Failed to import load_baseline_model from new path: {e}")

print("\nTest complete")