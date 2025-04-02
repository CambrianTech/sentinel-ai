#!/bin/bash

# Run Multi-Model Profiling Tool
# This script makes it easy to run the profile_full_model.py script with
# multi-model comparison enabled and various default settings.

# Default values
MODELS="gpt2,gpt2-medium,facebook/opt-125m,bigscience/bloom-560m"
OUTPUT_DIR="profiling_results/multi_model"
DEVICE="cuda"
ITERATIONS=3
TOKENS=20
OPTIMIZATION_LEVEL=2
VISUALIZE=true

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --models)
      MODELS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --tokens)
      TOKENS="$2"
      shift 2
      ;;
    --optimization-level)
      OPTIMIZATION_LEVEL="$2"
      shift 2
      ;;
    --no-visualize)
      VISUALIZE=false
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python scripts/profile_full_model.py 
  --multi_model_comparison 
  --compare_models \"$MODELS\" 
  --device $DEVICE 
  --iterations $ITERATIONS 
  --generated_tokens $TOKENS 
  --optimization_level $OPTIMIZATION_LEVEL 
  --output_dir \"$OUTPUT_DIR\""

# Add visualize flag if needed
if [ "$VISUALIZE" = true ]; then
  CMD="$CMD --visualize"
fi

# Display and run command
echo "Running command:"
echo "$CMD"
echo ""
echo "This will compare the following models: ${MODELS//,/, }"
echo ""
eval "$CMD"

# Display results location
echo ""
echo "Results saved to $OUTPUT_DIR"
if [ "$VISUALIZE" = true ]; then
  echo "Visualizations saved to $OUTPUT_DIR"
  # List generated image files
  echo "Generated visualizations:"
  find "$OUTPUT_DIR" -name "*.png" -exec basename {} \;
fi