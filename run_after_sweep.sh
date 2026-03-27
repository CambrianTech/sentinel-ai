#!/usr/bin/env bash
# Wait for sweep to finish, then run self-directed experiment
cd ~/sentinel-ai
source .venv/bin/activate

echo "Waiting for sweep to finish..."
while pgrep -f experiment_sweep > /dev/null; do
    sleep 30
done
echo "Sweep done. Starting self-directed experiment."

echo "=== SELF-DIRECTED: gpt2-medium ==="
python3 experiments/experiment_self_directed.py --model_name gpt2-medium 2>&1 | tee output/self_directed_gpt2medium.log

echo "=== SELF-DIRECTED: Qwen2.5-3B ==="
python3 experiments/experiment_self_directed.py --model_name Qwen/Qwen2.5-3B 2>&1 | tee output/self_directed_qwen3b.log

echo "=== ALL DONE ==="
