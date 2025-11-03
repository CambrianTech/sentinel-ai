#!/bin/bash
# Launcher for continuous training sessions

set -e

echo "========================================================================"
echo "SENTINEL-AI CONTINUOUS TRAINING LAUNCHER"
echo "========================================================================"
echo

# Kill existing sessions if requested
if [[ "$1" == "kill" ]]; then
    echo "Killing existing training sessions..."
    tmux kill-session -t sentinel-unpruned 2>/dev/null || true
    tmux kill-session -t sentinel-pruned 2>/dev/null || true
    echo "✅ Sessions killed"
    exit 0
fi

# List active sessions
if [[ "$1" == "list" ]]; then
    echo "Active training sessions:"
    tmux list-sessions 2>/dev/null | grep sentinel || echo "  (none)"
    exit 0
fi

# Attach to a session
if [[ "$1" == "attach" ]]; then
    if [[ -z "$2" ]]; then
        echo "Usage: ./start_training.sh attach <unpruned|pruned>"
        exit 1
    fi
    tmux attach-session -t "sentinel-$2"
    exit 0
fi

# Show logs
if [[ "$1" == "logs" ]]; then
    if [[ -z "$2" ]]; then
        echo "Usage: ./start_training.sh logs <unpruned|pruned>"
        exit 1
    fi
    MODE="$2"
    LOG_DIR="./checkpoints/gpt2_${MODE}_wikitext"
    if [[ -f "$LOG_DIR/training.log" ]]; then
        tail -f "$LOG_DIR/training.log"
    else
        echo "No log file found at $LOG_DIR/training.log"
    fi
    exit 0
fi

# Show samples
if [[ "$1" == "samples" ]]; then
    if [[ -z "$2" ]]; then
        echo "Usage: ./start_training.sh samples <unpruned|pruned>"
        exit 1
    fi
    MODE="$2"
    SAMPLES_FILE="./checkpoints/gpt2_${MODE}_wikitext/text_samples.txt"
    if [[ -f "$SAMPLES_FILE" ]]; then
        tail -f "$SAMPLES_FILE"
    else
        echo "No samples file found at $SAMPLES_FILE"
    fi
    exit 0
fi

# Start training
echo "Starting training sessions..."
echo

# Check if sessions already exist
if tmux has-session -t sentinel-unpruned 2>/dev/null; then
    echo "⚠️  Session 'sentinel-unpruned' already exists"
else
    echo "🚀 Starting UNPRUNED model training..."
    tmux new-session -d -s sentinel-unpruned \
        "experiments/run_with_continuum_python.sh train_continuously.py --model gpt2 --mode unpruned --dataset wikitext 2>&1 | tee ./checkpoints/gpt2_unpruned_wikitext/training.log"
    echo "   ✅ Session: sentinel-unpruned"
fi

if tmux has-session -t sentinel-pruned 2>/dev/null; then
    echo "⚠️  Session 'sentinel-pruned' already exists"
else
    echo "🚀 Starting PRUNED model training (40% magnitude pruning)..."
    tmux new-session -d -s sentinel-pruned \
        "experiments/run_with_continuum_python.sh train_continuously.py --model gpt2 --mode pruned --pruning-level 0.4 --dataset wikitext 2>&1 | tee ./checkpoints/gpt2_pruned_wikitext/training.log"
    echo "   ✅ Session: sentinel-pruned"
fi

echo
echo "========================================================================"
echo "TRAINING STARTED"
echo "========================================================================"
echo
echo "📊 Monitor progress:"
echo "   Unpruned samples: ./start_training.sh samples unpruned"
echo "   Pruned samples:   ./start_training.sh samples pruned"
echo "   Unpruned logs:    ./start_training.sh logs unpruned"
echo "   Pruned logs:      ./start_training.sh logs pruned"
echo
echo "🔗 Attach to sessions:"
echo "   Unpruned: tmux attach -t sentinel-unpruned"
echo "   Pruned:   tmux attach -t sentinel-pruned"
echo "   (Ctrl+B then D to detach)"
echo
echo "📁 Checkpoints:"
echo "   Unpruned: ./checkpoints/gpt2_unpruned_wikitext/checkpoints/"
echo "   Pruned:   ./checkpoints/gpt2_pruned_wikitext/checkpoints/"
echo
echo "🛑 Stop training:"
echo "   ./start_training.sh kill"
echo
echo "📋 List sessions:"
echo "   ./start_training.sh list"
echo "========================================================================"
