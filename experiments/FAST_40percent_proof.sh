#!/bin/bash
# FAST 40% Pruning Proof (2-3 hours on M1)
# This is the ONE GOOD WIN - proves Sentinel works ASAP

set -e

echo "============================================"
echo "SENTINEL-AI: FAST 40% PRUNING PROOF"
echo "============================================"
echo "Goal: Prove 40% pruning in 2-3 hours"
echo "Model: DistilGPT-2 (smaller, faster)"
echo "Dataset: WikiText-2 (standard benchmark)"
echo "Device: MPS (M1 GPU)"
echo ""
echo "Started: $(date)"
echo ""

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="experiments/results/fast_40percent_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

# Use the upgrayedd tool for fast iteration
python3 scripts/upgrayedd.py \
  --model distilgpt2 \
  --dataset wikitext \
  --pruning-level 0.4 \
  --strategy entropy \
  --cycles 3 \
  --epochs 2 \
  --batch-size 4 \
  --learning-rate 5e-5 \
  --device mps \
  --output-dir "${OUTPUT_DIR}" \
  --save-model \
  --use-dashboard \
  --verbose

echo ""
echo "============================================"
echo "EXPERIMENT COMPLETE"
echo "============================================"
echo "Completed: $(date)"
echo ""
echo "Results: ${OUTPUT_DIR}"
echo ""
echo "Check dashboard.html for visualizations"
echo ""
