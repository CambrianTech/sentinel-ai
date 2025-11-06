#!/bin/bash
# Run the benchmark_with_metrics.py script with the modular pruning API
# This script uses the improved benchmark_with_metrics.py which avoids circular imports
# by using either the modular sentinel.pruning API or a mock datasets approach

# Capture start time
START_TIME=$(date +%s)

# Print help function
print_help() {
    echo "Usage: ./scripts/run_benchmark.sh [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --model_name MODEL       Model name to benchmark (default: distilgpt2)"
    echo "  --output_dir DIR         Directory to save results (default: ./benchmark_results)"
    echo "  --pruning_strategies S   Comma-separated list of pruning strategies (default: random,entropy,magnitude)"
    echo "  --pruning_levels L       Comma-separated list of pruning levels (default: 0.1,0.3,0.5)"
    echo "  --learning_steps N       Number of learning steps after pruning (default: 500)"
    echo "  --learning_rate LR       Learning rate for fine-tuning (default: 5e-6)"
    echo "  --early_stop_patience P  Early stopping patience (default: 15)"
    echo "  --eval_interval EI       Evaluate every N steps (default: 25)"
    echo "  --batch_size BS          Batch size for training (default: 4)"
    echo "  --eval_samples ES        Number of evaluation samples (default: 100)"
    echo "  --max_length ML          Maximum sequence length (default: 256)"
    echo "  --help                   Show this help message"
    echo ""
    echo "BOOLEAN FLAGS:"
    echo "  --use_adaptive_lr        Use different learning rates for different parts of the model"
    echo "  --use_real_data          Use real data instead of synthetic data"
    echo "  --save_checkpoints       Save model checkpoints during fine-tuning"
    echo "  --verbose                Print verbose output"
    echo ""
    echo "Example:"
    echo "  ./scripts/run_benchmark.sh --model_name distilgpt2 --pruning_strategies random,entropy \\"
    echo "    --pruning_levels 0.1,0.3 --learning_steps 50 --verbose"
}

# Default values
MODEL_NAME="distilgpt2"
OUTPUT_DIR="./benchmark_results"
PRUNING_STRATEGIES="random,entropy,magnitude"
PRUNING_LEVELS="0.1,0.3,0.5"
LEARNING_STEPS=500
LEARNING_RATE=0.000005
EARLY_STOP_PATIENCE=15
EVAL_INTERVAL=25
BATCH_SIZE=4
EVAL_DATASET=""
EVAL_SAMPLES=100
MAX_LENGTH=256
USE_ADAPTIVE_LR=""
USE_REAL_DATA=""
SAVE_CHECKPOINTS=""
VERBOSE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --pruning_strategies)
            PRUNING_STRATEGIES="$2"
            shift 2
            ;;
        --pruning_levels)
            PRUNING_LEVELS="$2"
            shift 2
            ;;
        --learning_steps)
            LEARNING_STEPS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --early_stop_patience)
            EARLY_STOP_PATIENCE="$2"
            shift 2
            ;;
        --eval_interval)
            EVAL_INTERVAL="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --eval_dataset)
            EVAL_DATASET="$2"
            shift 2
            ;;
        --eval_samples)
            EVAL_SAMPLES="$2"
            shift 2
            ;;
        --max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --use_adaptive_lr)
            USE_ADAPTIVE_LR="--use_adaptive_lr"
            shift
            ;;
        --use_real_data)
            USE_REAL_DATA="--use_real_data"
            shift
            ;;
        --save_checkpoints)
            SAVE_CHECKPOINTS="--save_checkpoints"
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --help)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            print_help
            exit 1
            ;;
    esac
done

# Construct eval_dataset parameter if provided
EVAL_DATASET_PARAM=""
if [ -n "$EVAL_DATASET" ]; then
    EVAL_DATASET_PARAM="--eval_dataset $EVAL_DATASET"
fi

# Build the command
CMD="python scripts/benchmark_with_metrics.py \
    --model_name $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --pruning_strategies \"$PRUNING_STRATEGIES\" \
    --pruning_levels \"$PRUNING_LEVELS\" \
    --learning_steps $LEARNING_STEPS \
    --learning_rate $LEARNING_RATE \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --eval_interval $EVAL_INTERVAL \
    --batch_size $BATCH_SIZE \
    $EVAL_DATASET_PARAM \
    --eval_samples $EVAL_SAMPLES \
    --max_length $MAX_LENGTH \
    $USE_ADAPTIVE_LR \
    $USE_REAL_DATA \
    $SAVE_CHECKPOINTS \
    $VERBOSE"

# Print command
echo "Running benchmark with command:"
echo "$CMD"
echo "----------------------------"

# Run the command
eval "$CMD"

# Calculate and print elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))
echo "----------------------------"
echo "Benchmark completed in: ${HOURS}h ${MINUTES}m ${SECONDS}s"