#!/bin/bash
# Monitor training progress for both models

while true; do
    clear
    echo "========================================================================"
    echo "SENTINEL-AI TRAINING MONITOR - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================================"
    echo

    # Check sessions
    echo "📊 ACTIVE SESSIONS:"
    tmux list-sessions 2>/dev/null | grep sentinel || echo "  (none running)"
    echo

    # Unpruned model
    echo "========================================================================"
    echo "UNPRUNED MODEL (GPT-2 Baseline)"
    echo "========================================================================"
    UNPRUNED_LOG="./checkpoints/gpt2_unpruned_wikitext/training.log"
    if [[ -f "$UNPRUNED_LOG" ]]; then
        echo "Latest training steps:"
        tail -15 "$UNPRUNED_LOG" | grep -E "(Step|Loss|Epoch|Generating)" || echo "  (initializing...)"
        echo
        echo "Latest samples:"
        tail -20 "./checkpoints/gpt2_unpruned_wikitext/text_samples.txt" 2>/dev/null | head -10 || echo "  (no samples yet)"
    else
        echo "  ⏳ Waiting for unpruned model to start..."
    fi
    echo

    # Pruned model
    echo "========================================================================"
    echo "PRUNED MODEL (GPT-2 with 40% Magnitude Pruning)"
    echo "========================================================================"
    PRUNED_LOG="./checkpoints/gpt2_pruned_wikitext/training.log"
    if [[ -f "$PRUNED_LOG" ]]; then
        echo "Latest training steps:"
        tail -15 "$PRUNED_LOG" | grep -E "(Step|Loss|Epoch|Generating|Pruned)" || echo "  (initializing...)"
        echo
        echo "Latest samples:"
        tail -20 "./checkpoints/gpt2_pruned_wikitext/text_samples.txt" 2>/dev/null | head -10 || echo "  (no samples yet)"
    else
        echo "  ⏳ Waiting for pruned model to start..."
    fi
    echo

    echo "========================================================================"
    echo "Press Ctrl+C to exit | Updates every 10 seconds"
    echo "========================================================================"

    sleep 10
done
