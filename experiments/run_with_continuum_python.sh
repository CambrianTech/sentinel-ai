#!/bin/bash
# Run Python scripts using Continuum's Python environment
# This environment has all Sentinel dependencies installed

set -euo pipefail

# Path to Continuum's Python environment
CONTINUUM_PATH="/Volumes/FlashGordon/cambrian/continuum/src/debug/jtag"
TRAIN_WRAPPER="${CONTINUUM_PATH}/.continuum/genome/python/train-wrapper.sh"

# Add Sentinel-AI to Python path
export PYTHONPATH="/Volumes/FlashGordon/cambrian/sentinel-ai:${PYTHONPATH:-}"

# Run the command with Continuum's Python
"${TRAIN_WRAPPER}" "$@"
