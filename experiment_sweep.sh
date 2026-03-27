#!/usr/bin/env bash
# Full experiment sweep for paper evidence
# Runs sequentially (all need the GPU)
set -euo pipefail

cd ~/sentinel-ai
source .venv/bin/activate

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="output/sweep_${TIMESTAMP}.log"
mkdir -p output

log() { echo "$(date '+%H:%M:%S') | $1" | tee -a "$LOG"; }

log "=== SENTINEL-AI EXPERIMENT SWEEP ==="
log "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"
log ""

# --- Experiment 1: Strategy comparison on gpt2-medium ---
log ">>> EXP 1/6: gpt2-medium ENTROPY 30% (baseline strategy)"
python3 scripts/run_neural_plasticity.py \
  --model_name gpt2-medium \
  --pruning_strategy entropy \
  --pruning_level 0.3 \
  --training_steps 500 \
  --cycles 3 \
  --device cuda 2>&1 | tee -a "$LOG" | tail -5
log "<<< EXP 1 DONE"
log ""

log ">>> EXP 2/6: gpt2-medium MAGNITUDE 30%"
python3 scripts/run_neural_plasticity.py \
  --model_name gpt2-medium \
  --pruning_strategy magnitude \
  --pruning_level 0.3 \
  --training_steps 500 \
  --cycles 3 \
  --device cuda 2>&1 | tee -a "$LOG" | tail -5
log "<<< EXP 2 DONE"
log ""

log ">>> EXP 3/6: gpt2-medium RANDOM 30%"
python3 scripts/run_neural_plasticity.py \
  --model_name gpt2-medium \
  --pruning_strategy random \
  --pruning_level 0.3 \
  --training_steps 500 \
  --cycles 3 \
  --device cuda 2>&1 | tee -a "$LOG" | tail -5
log "<<< EXP 3 DONE"
log ""

# --- Experiment 4: Close the 40% recovery gap with more training ---
log ">>> EXP 4/6: gpt2-large 40% entropy, 2000 steps (recovery test)"
python3 scripts/run_neural_plasticity.py \
  --model_name gpt2-large \
  --pruning_strategy entropy \
  --pruning_level 0.4 \
  --training_steps 2000 \
  --cycles 3 \
  --device cuda 2>&1 | tee -a "$LOG" | tail -5
log "<<< EXP 4 DONE"
log ""

# --- Experiment 5: Qwen2.5-7B (the big one) ---
log ">>> EXP 5/6: Qwen2.5-7B 30% entropy (cross-architecture at scale)"
python3 scripts/run_neural_plasticity.py \
  --model_name Qwen/Qwen2.5-7B \
  --pruning_strategy entropy \
  --pruning_level 0.3 \
  --training_steps 500 \
  --cycles 3 \
  --device cuda 2>&1 | tee -a "$LOG" | tail -5
log "<<< EXP 5 DONE"
log ""

# --- Experiment 6: Qwen2.5-3B with more training ---
log ">>> EXP 6/6: Qwen2.5-3B 30% entropy, 1000 steps"
python3 scripts/run_neural_plasticity.py \
  --model_name Qwen/Qwen2.5-3B \
  --pruning_strategy entropy \
  --pruning_level 0.3 \
  --training_steps 1000 \
  --cycles 3 \
  --device cuda 2>&1 | tee -a "$LOG" | tail -5
log "<<< EXP 6 DONE"
log ""

log "=== SWEEP COMPLETE ==="
log ""

# --- Collect results summary ---
log "=== RESULTS SUMMARY ==="
for dir in output/neural_plasticity_*/; do
  if [ -f "$dir/model/model_info.txt" ]; then
    log "---"
    cat "$dir/model/model_info.txt" | tee -a "$LOG"
  fi
done

log ""
log "Full logs: $LOG"
log "Output dirs: output/neural_plasticity_*"
