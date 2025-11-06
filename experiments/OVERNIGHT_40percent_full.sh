#!/bin/bash
# OVERNIGHT 40% Pruning - Full Reproduction
# Runs overnight (~6-8 hours) with complete dataset
# This is the DEFINITIVE proof for the paper

set -e

echo "============================================"
echo "SENTINEL-AI: OVERNIGHT 40% PRUNING PROOF"
echo "============================================"
echo "This is the full reproduction for publication"
echo "Model: GPT-2 (full model, not distil)"
echo "Dataset: WikiText-2 (complete)"
echo "Device: MPS (M1 GPU)"
echo "Expected runtime: 6-8 hours"
echo ""
echo "Started: $(date)"
echo ""

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="experiments/results/overnight_40percent_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

# Log everything
LOG_FILE="${OUTPUT_DIR}/experiment.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Output directory: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo ""

# Run the REAL Sentinel pruning benchmark (using Continuum's Python environment)
experiments/run_with_continuum_python.sh scripts/benchmark_pruning.py \
  --model gpt2 \
  --dataset wikitext \
  --pruning_levels 0.0 0.1 0.2 0.3 0.4 \
  --strategies entropy \
  --max_length 256 \
  --device mps \
  --output_dir "${OUTPUT_DIR}" \
  --seed 42

EXIT_CODE=$?

echo ""
echo "============================================"
echo "EXPERIMENT COMPLETE"
echo "============================================"
echo "Completed: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS"
    echo ""
    echo "Results location:"
    echo "  ${OUTPUT_DIR}"
    echo ""
    echo "Key files:"
    echo "  - results.json (raw data)"
    echo "  - SUMMARY.txt (paper summary)"
    echo "  - figures/ (publication figures)"
    echo "  - models/ (checkpoints)"
    echo ""
    echo "Next steps:"
    echo "  1. Review SUMMARY.txt"
    echo "  2. Check figures for paper"
    echo "  3. Commit results to git"
    echo ""
else
    echo "❌ FAILED (exit code: ${EXIT_CODE})"
    echo "Check log: ${LOG_FILE}"
fi

# Send notification (if terminal-notifier installed)
if command -v terminal-notifier &> /dev/null; then
    if [ $EXIT_CODE -eq 0 ]; then
        terminal-notifier -title "Sentinel-AI" -message "40% pruning experiment complete! ✅"
    else
        terminal-notifier -title "Sentinel-AI" -message "Experiment failed ❌"
    fi
fi

exit $EXIT_CODE
