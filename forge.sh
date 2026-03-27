#!/usr/bin/env bash
#
# forge.sh — One-command model forging pipeline
#
# Usage:
#   ./forge.sh Qwen/Qwen2.5-7B                    # General forging
#   ./forge.sh Qwen/Qwen2.5-7B --strategy combined # Specific strategy
#   ./forge.sh --batch forge_list.txt               # Batch mode
#
# Output: output/forged/<model-name>/
#   ├── model/           (forged model checkpoint)
#   ├── results.json     (before/after metrics)
#   ├── model_card.md    (auto-generated HF model card)
#   ├── figures/          (entropy heatmaps, recovery curves)
#   └── benchmark/        (perplexity, generation samples)

set -uo pipefail
cd "$(dirname "$0")"

# Ensure venv
if [ ! -d ".venv" ]; then
    echo "Run ./setup.sh first"
    exit 1
fi
source .venv/bin/activate

# Defaults
STRATEGY="combined"
PRUNING_LEVEL="0.3"
STEPS="1000"
CYCLES="3"
DEVICE="cuda"
BATCH_FILE=""
PUBLISH=false
DOMAIN="general"
EARLY_STOP=""
RESUME=""

# Parse args
MODEL=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy) STRATEGY="$2"; shift 2 ;;
        --pruning) PRUNING_LEVEL="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --cycles) CYCLES="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --batch) BATCH_FILE="$2"; shift 2 ;;
        --publish) PUBLISH=true; shift ;;
        --domain) DOMAIN="$2"; shift 2 ;;
        --early-stop) EARLY_STOP="$2"; shift 2 ;;
        --resume) RESUME="$2"; shift 2 ;;
        *) MODEL="$1"; shift ;;
    esac
done

forge_model() {
    local model="$1"
    local model_slug=$(basename "$model" | tr '[:upper:]' '[:lower:]')
    local output_dir="output/forged/${model_slug}"
    local timestamp=$(date +%Y%m%d_%H%M%S)

    echo ""
    echo "============================================================"
    echo "  FORGING: $model"
    echo "  Strategy: $STRATEGY, Pruning: $PRUNING_LEVEL, Steps: $STEPS, Cycles: $CYCLES"
    echo "  Output: $output_dir"
    echo "============================================================"
    echo ""

    mkdir -p "$output_dir/figures" "$output_dir/benchmark"

    # Step 1: Run plasticity experiment
    # If resuming, use saved model weights as the starting point
    local effective_model="$model"
    if [ -n "$RESUME" ] && [ -d "$RESUME/model" ]; then
        effective_model="$RESUME/model"
        echo "[RESUME] Continuing from $RESUME/model"
    fi

    echo "[1/5] Running experiential plasticity..."
    .venv/bin/python3 scripts/run_neural_plasticity.py \
        --model_name "$effective_model" \
        --pruning_strategy "$STRATEGY" \
        --pruning_level "$PRUNING_LEVEL" \
        --training_steps "$STEPS" \
        --cycles "$CYCLES" \
        --save_model \
        ${EARLY_STOP:+--early_stop "$EARLY_STOP"} \
        --device "$DEVICE" 2>&1 | tee "$output_dir/forge.log"

    # Find the experiment output directory (most recent)
    local exp_dir=$(ls -dt output/neural_plasticity_*/ 2>/dev/null | head -1)
    if [ -z "$exp_dir" ]; then
        echo "ERROR: No experiment output found"
        return 1
    fi

    # Step 2: Collect results
    echo "[2/5] Collecting results..."
    local model_info="$exp_dir/model/model_info.txt"
    if [ -f "$model_info" ]; then
        cp "$model_info" "$output_dir/"

        # Extract metrics for results.json
        local final_ppl=$(grep "After Training:.*Perplexity" "$output_dir/forge.log" | tail -1 | sed "s/.*Perplexity = //" | tr -d " ")
        # Fallback: last evaluation perplexity
        [ -z "$final_ppl" ] && final_ppl=$(grep "Evaluation:.*Perplexity" "$output_dir/forge.log" | tail -1 | sed "s/.*Perplexity = //" | tr -d " ")
        local strategy=$(grep "Pruning Strategy" "$model_info" | awk '{print $NF}')
        local level=$(grep "Pruning Level" "$model_info" | awk '{print $NF}')

        # Get baseline from log
        local baseline_ppl=$(grep "Baseline:.*Perplexity" "$output_dir/forge.log" | head -1 | sed "s/.*Perplexity = //" | tr -d " ")

        .venv/bin/python3 -c "
import json
results = {
    'model': '$model',
    'strategy': '$strategy',
    'pruning_level': float('$level'),
    'cycles': $CYCLES,
    'training_steps': $STEPS,
    'baseline_ppl': float('${baseline_ppl:-0}'),
    'final_ppl': float('${final_ppl:-0}'),
    'forged_at': '$(date -Iseconds)',
    'device': '$DEVICE',
}
if results['baseline_ppl'] > 0:
    results['improvement_pct'] = round(
        (results['baseline_ppl'] - results['final_ppl']) / results['baseline_ppl'] * 100, 2
    )
json.dump(results, open('$output_dir/results.json', 'w'), indent=2)
print(json.dumps(results, indent=2))
"
    fi

    # Step 3: Copy figures
    echo "[3/5] Collecting figures..."
    cp "$exp_dir"/*.png "$output_dir/figures/" 2>/dev/null || true
    cp "$exp_dir"/cycle_*/*.png "$output_dir/figures/" 2>/dev/null || true
    cp "$exp_dir"/visualizations/*.png "$output_dir/figures/" 2>/dev/null || true

    # Copy generation samples
    cp "$exp_dir"/generation/*.txt "$output_dir/benchmark/" 2>/dev/null || true

    # Step 4: Copy model weights
    echo "[4/5] Copying model weights..."
    local model_src="${exp_dir}models/final_model"
    if [ -d "$model_src" ]; then
        mkdir -p "$output_dir/model"
        cp -r "$model_src"/* "$output_dir/model/"
        local weight_count=$(ls "$output_dir/model/"*.safetensors 2>/dev/null | wc -l)
        echo "  Copied $weight_count safetensors files + config/tokenizer"
    else
        echo "  WARNING: No model weights found at $model_src"
        echo "  The model was not saved — check --save_model flag"
    fi

    # Step 5: Generate model card
    echo "[5/5] Generating model card..."
    .venv/bin/python3 -c "
import json

results = json.load(open('$output_dir/results.json'))
model = results['model']
slug = '$model_slug'
baseline = results.get('baseline_ppl', 0)
final = results.get('final_ppl', 0)
improvement = results.get('improvement_pct', 0)
strategy = results.get('strategy', 'combined')
level = results.get('pruning_level', 0.3)
cycles = results.get('cycles', 3)
steps = results.get('training_steps', 1000)

card = f'''---
tags:
- continuum
- experiential-plasticity
- forged
base_model: {model}
---

# {slug}-forged

A **forged** version of [{model}](https://huggingface.co/{model}) — optimized through [Experiential Plasticity](https://github.com/CambrianTech/sentinel-ai).

## Results

| Metric | Value |
|--------|-------|
| Baseline Perplexity | {baseline:.2f} |
| **Forged Perplexity** | **{final:.2f}** |
| **Improvement** | **{improvement:+.1f}%** |
| Pruning Strategy | {strategy} |
| Pruning Level | {level:.0%} |
| Cycles | {cycles} |
| Training Steps/Cycle | {steps} |

This model is **{abs(improvement):.1f}% {\"better\" if improvement > 0 else \"within\"}** than the base model after removing {level:.0%} of attention heads and retraining. The remaining heads specialized to compensate, producing a more efficient architecture.

## What is Experiential Plasticity?

Iterative entropy-based pruning with retraining — attention heads that contribute minimal information are removed, then the model retrains to compensate. Remaining heads specialize and the model emerges smaller and more capable. Like biological synaptic pruning during brain development.

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(\"continuum-ai/{slug}-forged\",
    torch_dtype=\"auto\", device_map=\"auto\")
tokenizer = AutoTokenizer.from_pretrained(\"continuum-ai/{slug}-forged\")
```

## Reproduce

```bash
git clone https://github.com/CambrianTech/sentinel-ai && cd sentinel-ai && ./setup.sh
source .venv/bin/activate
python scripts/run_neural_plasticity.py \\\\
  --model_name {model} --pruning_strategy {strategy} \\\\
  --pruning_level {level} --training_steps {steps} --cycles {cycles}
```

## Part of continuum

**Research:**
- [Experiential Plasticity Paper](https://github.com/CambrianTech/continuum/blob/main/docs/papers/EXPERIENTIAL-PLASTICITY.md)
- [Neural Plasticity in Transformers](https://github.com/CambrianTech/continuum/blob/main/docs/papers/SENTINEL-AI-NEURAL-PLASTICITY.md)
- [Plasticity Compaction](https://github.com/CambrianTech/continuum/blob/main/docs/papers/PLASTICITY-COMPACTION-MOE.md)

[sentinel-ai](https://github.com/CambrianTech/sentinel-ai) | [continuum](https://github.com/CambrianTech/continuum) | [HuggingFace](https://huggingface.co/continuum-ai)
'''

with open('$output_dir/model_card.md', 'w') as f:
    f.write(card)
print('Model card generated: $output_dir/model_card.md')
"

    echo ""
    echo "============================================================"
    echo "  FORGING COMPLETE: $model"
    echo "  Results: $output_dir/results.json"
    echo "  Model card: $output_dir/model_card.md"
    echo "  Figures: $output_dir/figures/"
    echo "  Model: $output_dir/model/"
    echo "============================================================"

    # Publish to HuggingFace if requested
    if [ "$PUBLISH" = true ]; then
        echo ""
        echo "[PUBLISH] Uploading to HuggingFace..."
        .venv/bin/python3 publish_forged.py "$output_dir" --domain "$DOMAIN"
    fi
}

# Batch mode
if [ -n "$BATCH_FILE" ]; then
    echo "=== BATCH FORGING FROM $BATCH_FILE ==="
    while IFS=' ' read -r model args; do
        [ -z "$model" ] && continue
        [[ "$model" == \#* ]] && continue
        forge_model "$model"
    done < "$BATCH_FILE"
    echo "=== BATCH COMPLETE ==="
    exit 0
fi

# Single model
if [ -z "$MODEL" ]; then
    echo "Usage: ./forge.sh <model_name> [--strategy combined] [--pruning 0.3] [--steps 1000] [--cycles 3] [--domain general] [--publish] [--early-stop 0.5] [--resume output/forged/qwen2.5-7b]"
    echo "       ./forge.sh --batch forge_list.txt"
    exit 1
fi

forge_model "$MODEL"
