#!/bin/bash
# Script to test all supported model architectures in Sentinel-AI
# This script runs the comprehensive model test for each family

set -e
SCRIPT_DIR=$(dirname "$0")
ROOT_DIR="$SCRIPT_DIR/.."
OUTPUT_DIR="$ROOT_DIR/test_results"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/test_run_$TIMESTAMP.log"

echo "===== Starting Sentinel-AI Model Tests =====" | tee -a "$LOG_FILE"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "Device: ${DEVICE:-cpu}" | tee -a "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"

# Test function
run_test() {
  family=$1
  echo -e "\n\n===== Testing $family Models =====" | tee -a "$LOG_FILE"
  
  python "$ROOT_DIR/test_all_models_comprehensive.py" \
    --family "$family" \
    --device "${DEVICE:-cpu}" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee -a "$LOG_FILE"
    
  echo "===== Completed $family Models =====" | tee -a "$LOG_FILE"
}

# Run tests for each model family
run_test "GPT-2"
run_test "OPT"  
run_test "Pythia"
run_test "BLOOM"

# Only run Llama tests if credentials are available
if [ -f "$HOME/.huggingface/token" ]; then
  run_test "Llama"
else
  echo "Skipping Llama models (no HuggingFace token found)" | tee -a "$LOG_FILE"
fi

echo -e "\n\n===== All Tests Complete =====" | tee -a "$LOG_FILE"
echo "See results in $OUTPUT_DIR" | tee -a "$LOG_FILE"

# Generate combined report
echo "Generating combined report..." | tee -a "$LOG_FILE"
python "$ROOT_DIR/test_all_models_comprehensive.py" \
  --output_dir "$OUTPUT_DIR" \
  2>&1 | tee -a "$LOG_FILE"

echo "Test suite completed successfully!" | tee -a "$LOG_FILE"