#!/usr/bin/env python3
"""expert_activation_profile.py — calibration-aware expert importance for MoE forge.

Loads an unmodified MoE base model in 8-bit on GPU, hooks every layer's
router gate output, runs a calibration corpus through the model, and
records per-layer per-expert activation counts (how often each expert
was in the top-k routing decision across all tokens of all calibration
inputs).

Output: JSON with shape
    {
      "model": "<base model path>",
      "calibration_corpus": "<jsonl path>",
      "calibration_examples": <int>,
      "calibration_tokens": <int>,
      "num_hidden_layers": <int>,
      "num_experts": <int>,
      "num_experts_per_tok": <int>,
      "activation_counts": {
          "0": [<count for expert 0>, ..., <count for expert num_experts-1>],
          "1": [...],
          ...
      }
    }

This file is the input to cpu_expert_prune_v2.py --importance-json, which
uses the per-layer activation counts as the expert importance metric
instead of the router gate row L2 norm.

Why this matters: router gate row L2 norm measures the magnitude of the
projection vector that scores each expert during routing. It correlates
with overall expert capability across the training distribution but is
NOT the same as "this expert fires on Python code." For task-specific
forges (a coder model is forged against code-heavy tasks), the
activation count on a code-heavy calibration corpus is the structurally
correct importance metric — it directly measures which experts the
router actually fires on the workload the artifact will be used for.

This is the §4.1.3.4 fix (expert level) of the same lesson §4.1.3.1
applied to dense heads: an architectural-only importance metric is
task-misaligned and produces lower held-out scores than a calibration-
aware metric does.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _ts() -> str:
    return time.strftime("[%H:%M:%S]")


def _log(msg: str) -> None:
    print(f"{_ts()} {msg}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", help="Path to or HF id of the base MoE model")
    p.add_argument("--calibration-data", required=True,
                   help="JSONL file with {'text': ...} entries")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--max-examples", type=int, default=None,
                   help="Cap on calibration examples (default: all)")
    p.add_argument("--max-length", type=int, default=2048,
                   help="Max sequence length per example (default: 2048)")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    model_path = args.model
    out_path = Path(args.output)

    # Load calibration corpus
    _log(f"loading calibration corpus from {args.calibration_data}")
    examples = []
    with open(args.calibration_data) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            text = d.get("text") or d.get("content") or ""
            if text:
                examples.append(text)
    if args.max_examples:
        examples = examples[: args.max_examples]
    _log(f"  {len(examples)} examples")

    _log(f"loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    _log(f"loading base model in 8-bit on {args.device}")
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_cfg,
        device_map={"": args.device},
        trust_remote_code=True,
    )
    model.eval()
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_experts = cfg.num_experts
    num_experts_per_tok = cfg.num_experts_per_tok
    _log(f"  arch: {type(model).__name__}")
    _log(f"  layers={num_layers} experts={num_experts} top_k={num_experts_per_tok}")

    # Per-layer activation counters (CPU int64, length num_experts each)
    counts = {li: torch.zeros(num_experts, dtype=torch.int64) for li in range(num_layers)}
    hooks = []

    def make_hook(layer_idx: int):
        def hook(module, inp, out):
            # `out` is the gate's output: [batch, seq, num_experts] (router logits)
            # Pick top-k expert indices per token, accumulate counts.
            x = out[0] if isinstance(out, tuple) else out
            flat = x.reshape(-1, x.shape[-1])  # [tokens, num_experts]
            if flat.shape[-1] != num_experts:
                return  # not a router gate (defensive)
            topk = flat.topk(num_experts_per_tok, dim=-1).indices.flatten()
            counts[layer_idx] += torch.bincount(
                topk.detach().cpu(), minlength=num_experts
            )
        return hook

    # Register hooks on each MoE layer's router gate
    registered = 0
    for li in range(num_layers):
        try:
            gate = model.model.layers[li].mlp.gate
            h = gate.register_forward_hook(make_hook(li))
            hooks.append(h)
            registered += 1
        except AttributeError:
            _log(f"  WARN: layer {li} has no mlp.gate")
    _log(f"  hooks registered on {registered}/{num_layers} layers")
    if registered == 0:
        _log("  FATAL: no router gates found")
        return 1

    _log(f"running {len(examples)} calibration examples through base model")
    total_tokens = 0
    t0 = time.time()
    with torch.inference_mode():
        for i, text in enumerate(examples):
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )
            input_ids = enc["input_ids"].to(args.device)
            total_tokens += input_ids.shape[1]
            try:
                model(input_ids=input_ids, use_cache=False)
            except Exception as e:
                _log(f"  example {i} failed: {e}")
                continue
            if (i + 1) % 25 == 0 or i == len(examples) - 1:
                elapsed = time.time() - t0
                _log(f"  {i+1}/{len(examples)} examples, {total_tokens} tokens, {elapsed:.1f}s")

    for h in hooks:
        h.remove()

    # Build output
    out_data = {
        "model": str(model_path),
        "calibration_corpus": str(args.calibration_data),
        "calibration_examples": len(examples),
        "calibration_tokens": int(total_tokens),
        "num_hidden_layers": int(num_layers),
        "num_experts": int(num_experts),
        "num_experts_per_tok": int(num_experts_per_tok),
        "activation_counts": {
            str(li): counts[li].tolist() for li in range(num_layers)
        },
        "metric_version": "v1.activation_count",
        "tool": "expert_activation_profile.py",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    _log(f"wrote {out_path}")

    # Quick stats: per-layer top-10 activation counts
    _log("per-layer top expert counts (sample):")
    for li in [0, 23, 47]:
        sorted_idxs = counts[li].argsort(descending=True)[:5].tolist()
        sorted_vals = counts[li][sorted_idxs].tolist()
        zeros = (counts[li] == 0).sum().item()
        _log(f"  layer {li}: top-5 experts {sorted_idxs} counts {sorted_vals}, zeros {zeros}/{num_experts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
