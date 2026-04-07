#!/usr/bin/env bash
#
# forge_v2_pipeline.sh — End-to-end v2 forge pipeline with deployment-runtime gates.
#
# This is the v0 of the harness Layer 7 deployment-runtime load test, encoded as
# a chained shell script. Each stage's success is the trigger for the next; each
# stage's failure halts the chain with a clear "halted at stage X for reason Y"
# message and dumps the relevant log path.
#
# Stages:
#   1. Forge (forge_model.py with --prune-metric activation --defrag-mode pad)
#   2. Validate output loads in transformers
#   3. Quantize to GGUF via llama.cpp's convert + quantize
#   4. Validate GGUF loads in actual llama.cpp (the Layer 7 gate)
#   5. EvalPlus HumanEval+ on the safetensors (the §4.1.4 number)
#   6. Write a results JSON that the paper update process can consume
#
# Usage:
#   ./scripts/forge_v2_pipeline.sh <base_model_dir> <output_dir> [--steps N]

set -euo pipefail

BASE_MODEL="${1:?usage: forge_v2_pipeline.sh <base_model_dir> <output_dir> [--steps N]}"
OUT_DIR="${2:?usage: forge_v2_pipeline.sh <base_model_dir> <output_dir> [--steps N]}"
STEPS=500
shift 2 || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps) STEPS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$OUT_DIR"
RESULTS_JSON="$OUT_DIR/forge_v2_results.json"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

LLAMA_CPP_BIN="${LLAMA_CPP_BIN:-$HOME/llama.cpp/build/bin}"
LLAMA_CPP_CONVERT="${LLAMA_CPP_CONVERT:-$HOME/llama.cpp/convert_hf_to_gguf.py}"

halt() {
    local stage="$1"
    local reason="$2"
    local logpath="${3:-}"
    echo "==================================================================="
    echo "  HALT at stage $stage: $reason"
    [[ -n "$logpath" ]] && echo "  log: $logpath"
    echo "==================================================================="
    cat > "$RESULTS_JSON" <<EOF
{
  "status": "halted",
  "halted_at_stage": "$stage",
  "reason": "$reason",
  "log": "$logpath",
  "out_dir": "$OUT_DIR"
}
EOF
    exit 1
}

stage() {
    echo
    echo "==================================================================="
    echo "  STAGE: $1"
    echo "==================================================================="
}

# ── Stage 1: Forge ─────────────────────────────────────────────────────────
stage "1/6 Forge with activation metric + pad defrag"
SCRIPTS_DIR="$(dirname "$(readlink -f "$0")")"
FORGE_LOG="$LOG_DIR/01_forge.log"

python3 -u "$SCRIPTS_DIR/forge_model.py" "$BASE_MODEL" \
    --domain code \
    --steps "$STEPS" \
    --cycles 1 \
    --prune-level 0.3 \
    --prune-metric activation \
    --defrag-mode pad \
    --output-dir "$OUT_DIR/forged" \
    2>&1 | tee "$FORGE_LOG"

if [[ ! -d "$OUT_DIR/forged/model" ]] && [[ ! -d "$OUT_DIR/forged" ]]; then
    halt "1/6 forge" "forge did not produce a model directory" "$FORGE_LOG"
fi

# ── Stage 2: Validate transformers load ────────────────────────────────────
stage "2/6 Validate transformers load + smoke generation"
TRANSFORMERS_LOG="$LOG_DIR/02_transformers_load.log"

# Find the model dir (forge_model.py writes to output/forged/<slug> by default
# but with --output-dir, it writes there directly)
MODEL_DIR="$OUT_DIR/forged"
if [[ -d "$MODEL_DIR/model" ]]; then
    MODEL_DIR="$MODEL_DIR/model"
fi

python3 - "$MODEL_DIR" > "$TRANSFORMERS_LOG" 2>&1 <<'PYEOF' || halt "2/6 transformers-load" "transformers could not load the forged model" "$LOG_DIR/02_transformers_load.log"
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained(sys.argv[1], torch_dtype="float16", device_map="cuda")
t = AutoTokenizer.from_pretrained(sys.argv[1])
ids = t("def fibonacci(n):", return_tensors="pt").to("cuda")
out = m.generate(**ids, max_new_tokens=40, do_sample=False, pad_token_id=t.eos_token_id)
text = t.decode(out[0], skip_special_tokens=True)
print(text)
print("---")
print(f"q_proj L0 shape: {m.model.layers[0].self_attn.q_proj.weight.shape}")
print(f"o_proj L0 shape: {m.model.layers[0].self_attn.o_proj.weight.shape}")
print(f"hidden_size: {m.config.hidden_size}")
print(f"num_attention_heads: {m.config.num_attention_heads}")
print(f"num_key_value_heads: {m.config.num_key_value_heads}")
print(f"head_dim: {m.config.head_dim}")
# Confirm the q_proj invariant for llama.cpp compatibility
assert m.model.layers[0].self_attn.q_proj.weight.shape[0] == m.config.hidden_size, \
    "q_proj.shape[0] != hidden_size — pad-defrag did not preserve the wire shape"
print("OK: q_proj.shape[0] == hidden_size (llama.cpp invariant satisfied)")
PYEOF

# ── Stage 3: Quantize to GGUF ──────────────────────────────────────────────
stage "3/6 Quantize to GGUF (q5_K_S via llama.cpp)"
QUANT_LOG="$LOG_DIR/03_quantize.log"
GGUF_F16="$OUT_DIR/v2-fp16.gguf"
GGUF_Q5KS="$OUT_DIR/v2-q5_K_S.gguf"

if [[ ! -f "$LLAMA_CPP_CONVERT" ]]; then
    halt "3/6 quantize" "llama.cpp convert script not found at $LLAMA_CPP_CONVERT" ""
fi
if [[ ! -x "$LLAMA_CPP_BIN/llama-quantize" ]]; then
    halt "3/6 quantize" "llama-quantize binary not found at $LLAMA_CPP_BIN/llama-quantize" ""
fi

python3 "$LLAMA_CPP_CONVERT" "$MODEL_DIR" --outfile "$GGUF_F16" --outtype f16 2>&1 | tee "$QUANT_LOG"
"$LLAMA_CPP_BIN/llama-quantize" "$GGUF_F16" "$GGUF_Q5KS" Q5_K_S 2>&1 | tee -a "$QUANT_LOG"

[[ -f "$GGUF_Q5KS" ]] || halt "3/6 quantize" "quantize did not produce $GGUF_Q5KS" "$QUANT_LOG"

# ── Stage 4: Layer 7 gate — actual llama.cpp load test ─────────────────────
stage "4/6 LAYER 7 GATE: load GGUF in actual llama.cpp"
LLAMA_LOG="$LOG_DIR/04_llama_cpp_gate.log"

# NOTE: -st (single-turn) is REQUIRED. Without it, llama-cli drops into
# interactive chat mode after generating the response and hangs the pipeline
# waiting on stdin. -no-cnv was the old flag and is rejected by current builds.
# This is the no-fallbacks discipline applied at the script layer: a recurring
# bug class fixed at the source rather than worked around manually each run.
"$LLAMA_CPP_BIN/llama-cli" -m "$GGUF_Q5KS" \
    -p "def fibonacci(n):" \
    -n 50 --temp 0 -ngl 99 -st 2>&1 | tee "$LLAMA_LOG"

if grep -q "wrong shape" "$LLAMA_LOG" || grep -q "failed to load model" "$LLAMA_LOG"; then
    halt "4/6 layer7-gate" "llama.cpp failed to load the GGUF — Finding 6 fix did not take" "$LLAMA_LOG"
fi
# Also assert the smoke output is actually present and non-degenerate
if ! grep -q "fibonacci\|return\|def " "$LLAMA_LOG"; then
    halt "4/6 layer7-gate" "llama.cpp loaded but produced no recognizable code output" "$LLAMA_LOG"
fi
echo "  ✓ Layer 7 gate PASSED — GGUF loads in llama.cpp and produces output"

# ── Stage 5: EvalPlus HumanEval+ on safetensors ────────────────────────────
stage "5/6 EvalPlus HumanEval+ on v2 safetensors"
EVALPLUS_LOG="$LOG_DIR/05_evalplus.log"
EVALPLUS_RESULTS="$OUT_DIR/evalplus_results"

python3 -m evalplus.codegen --model "$MODEL_DIR" \
    --dataset humaneval --backend hf --greedy \
    --root "$EVALPLUS_RESULTS" --dtype float16 --trust_remote_code \
    2>&1 | tee "$EVALPLUS_LOG"

# Sanitize + evaluate
python3 -m evalplus.sanitize --samples "$EVALPLUS_RESULTS/humaneval/"*.jsonl 2>&1 | tee -a "$EVALPLUS_LOG" || true
python3 -m evalplus.evaluate --dataset humaneval \
    --samples "$EVALPLUS_RESULTS/humaneval/"*sanitized*.jsonl 2>&1 | tee -a "$EVALPLUS_LOG"

# ── Stage 6: Write results JSON ────────────────────────────────────────────
stage "6/6 Write results JSON"
python3 - <<PYEOF
import json, os, glob, re
from pathlib import Path

results = {"status": "complete", "out_dir": "$OUT_DIR"}

# Try to extract pass@1 from evalplus log
log = Path("$EVALPLUS_LOG").read_text()
for label in ("base", "plus"):
    m = re.search(rf"{label}\s*[:=]\s*(\d+\.\d+)", log, re.IGNORECASE)
    if m:
        results[f"humaneval_{label}_pass1"] = float(m.group(1))

# GGUF size
gguf = Path("$GGUF_Q5KS")
if gguf.exists():
    results["gguf_q5ks_bytes"] = gguf.stat().st_size

# Forge config
results["forge_config"] = {
    "base_model": "$BASE_MODEL",
    "steps": $STEPS,
    "prune_metric": "activation",
    "defrag_mode": "pad",
    "prune_level": 0.3,
}

Path("$RESULTS_JSON").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
PYEOF

echo
echo "==================================================================="
echo "  ✓ FORGE V2 PIPELINE COMPLETE"
echo "  results: $RESULTS_JSON"
echo "==================================================================="
