#!/usr/bin/env bash
# Full experiment sweep for paper evidence — v2
# Runs sequentially (all need the GPU)
# Continues on failure (no set -e)

cd ~/sentinel-ai
source .venv/bin/activate

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="output/sweep_v2_${TIMESTAMP}.log"
mkdir -p output

log() { echo "$(date '+%H:%M:%S') | $1" | tee -a "$LOG"; }

run_exp() {
  local name="$1"; shift
  log ">>> $name"
  if python3 scripts/run_neural_plasticity.py "$@" 2>&1 | tee -a "$LOG" | grep -E '(perplexity|Improvement|Error|FAIL|completed)' | tail -5; then
    log "<<< $name DONE"
  else
    log "<<< $name FAILED (see log)"
  fi
  log ""
}

log "=== SENTINEL-AI EXPERIMENT SWEEP v2 ==="
log "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"
log ""

# --- Strategy comparison: entropy vs gradient vs random on gpt2-medium ---
run_exp "EXP 1/7: gpt2-medium ENTROPY 30%" \
  --model_name gpt2-medium --pruning_strategy entropy --pruning_level 0.3 --training_steps 500 --cycles 3 --device cuda

run_exp "EXP 2/7: gpt2-medium GRADIENT 30%" \
  --model_name gpt2-medium --pruning_strategy gradient --pruning_level 0.3 --training_steps 500 --cycles 3 --device cuda

run_exp "EXP 3/7: gpt2-medium RANDOM 30%" \
  --model_name gpt2-medium --pruning_strategy random --pruning_level 0.3 --training_steps 500 --cycles 3 --device cuda

# --- Close the 40% recovery gap with more training ---
run_exp "EXP 4/7: gpt2-large 40% entropy, 2000 steps" \
  --model_name gpt2-large --pruning_strategy entropy --pruning_level 0.4 --training_steps 2000 --cycles 3 --device cuda

# --- Qwen2.5-7B (the big one) ---
run_exp "EXP 5/7: Qwen2.5-7B 30% entropy" \
  --model_name Qwen/Qwen2.5-7B --pruning_strategy entropy --pruning_level 0.3 --training_steps 500 --cycles 3 --device cuda

# --- Qwen2.5-3B with more training ---
run_exp "EXP 6/7: Qwen2.5-3B 30% entropy, 1000 steps" \
  --model_name Qwen/Qwen2.5-3B --pruning_strategy entropy --pruning_level 0.3 --training_steps 1000 --cycles 3 --device cuda

# --- Combined strategy on gpt2-medium ---
run_exp "EXP 7/7: gpt2-medium COMBINED 30%" \
  --model_name gpt2-medium --pruning_strategy combined --pruning_level 0.3 --training_steps 500 --cycles 3 --device cuda

log "=== SWEEP v2 COMPLETE ==="
log ""

# --- Collect all results ---
log "=== RESULTS SUMMARY ==="
for dir in $(ls -dt output/neural_plasticity_*/); do
  if [ -f "${dir}model/model_info.txt" ]; then
    log "---"
    cat "${dir}model/model_info.txt" | tee -a "$LOG"
  fi
done
