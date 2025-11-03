#!/bin/bash
# Reproduce the 40% pruning experiment from April 2025
# This script demonstrates that GPT-2 can be pruned by 40% with minimal quality impact

set -e  # Exit on error

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="experiments/results/pruning_40percent_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "Sentinel-AI: 40% Pruning Reproduction"
echo "============================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "Started: $(date)"
echo ""

# Run the neural plasticity experiment with 40% pruning
# Using the original parameters from April 2025
python scripts/run_neural_plasticity.py \
  --model_name gpt2 \
  --dataset wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --pruning_strategy entropy \
  --pruning_level 0.4 \
  --cycles 3 \
  --training_steps 500 \
  --learning_rate 5e-5 \
  --batch_size 8 \
  --max_length 128 \
  --output_dir "${OUTPUT_DIR}" \
  --save_model \
  --use_dashboard \
  --verbose

echo ""
echo "============================================"
echo "Experiment Complete!"
echo "============================================"
echo "Completed: $(date)"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Key files:"
echo "  - ${OUTPUT_DIR}/dashboard.html (interactive results)"
echo "  - ${OUTPUT_DIR}/metrics/ (raw metrics)"
echo "  - ${OUTPUT_DIR}/models/ (pruned model checkpoints)"
echo "  - ${OUTPUT_DIR}/visualizations/ (figures)"
echo ""
echo "Expected results (from April 2025):"
echo "  - 30-40% of attention heads pruned"
echo "  - Perplexity improvement: baseline → pruned → regrown"
echo "  - Minimal quality loss with significant parameter reduction"
echo ""
