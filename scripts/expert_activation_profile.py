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


# ── Importable API ──────────────────────────────────────────────────────────
#
# Two callable entry points + one private inner. Both entry points produce the
# same JSON output and have strict, non-overlapping contracts:
#
#   profile_experts_from_path(model_path, ...) — used by the CLI and any
#       caller that wants this script to load the model itself in 8-bit on
#       GPU. Loads tokenizer, loads model with BitsAndBytesConfig, then
#       delegates to _profile_inner.
#
#   profile_experts(model, tokenizer, ...) — used by the family-adapter
#       set. Caller provides an already-loaded model + tokenizer; this
#       function does NOT touch model loading. Delegates to _profile_inner.
#
# Both write the importance JSON to the output path AND return the data dict.


def _read_calibration_corpus(calibration_data: Path, max_examples: int | None) -> list[str]:
    examples: list[str] = []
    with open(calibration_data) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            text = d.get("text") or d.get("content") or ""
            if text:
                examples.append(text)
    if max_examples:
        examples = examples[:max_examples]
    return examples


def _moe_geometry(model) -> tuple[int, int, int]:
    """Return (num_layers, num_experts, num_experts_per_tok) from model.config.

    MoE field names vary across families: Qwen3MoE/Olmoe use num_experts,
    GraniteMoE/Mixtral use num_local_experts, DeepSeek-V2 uses
    n_routed_experts. Probes in order. Raises if neither path resolves —
    failure is loud, never silently substituted with a default.
    """
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    # MoE field names vary across families: Qwen3MoE/Olmoe use num_experts,
    # GraniteMoE/Mixtral use num_local_experts, DeepSeek-V2 uses
    # n_routed_experts. Try in order.
    num_experts = (
        getattr(cfg, "num_experts", None)
        or getattr(cfg, "num_local_experts", None)
        or getattr(cfg, "n_routed_experts", None)
    )
    num_experts_per_tok = (
        getattr(cfg, "num_experts_per_tok", None)
        or getattr(cfg, "num_active_experts", None)
    )
    if num_experts is None:
        raise ValueError(
            f"could not find expert count on {type(cfg).__name__}; "
            f"checked num_experts / num_local_experts / n_routed_experts"
        )
    if num_experts_per_tok is None:
        raise ValueError(
            f"could not find num_experts_per_tok on {type(cfg).__name__}"
        )
    return num_layers, num_experts, num_experts_per_tok


def _resolve_attr_path(obj, path: str):
    """Walk a dotted attribute path on a torch module. Returns None if
    any segment is missing — caller decides whether that's fatal."""
    cur = obj
    for seg in path.split("."):
        cur = getattr(cur, seg, None)
        if cur is None:
            return None
    return cur


def _profile_inner(
    *,
    model,
    tokenizer,
    examples: list[str],
    max_length: int,
    device: str,
    model_label: str,
    corpus_label: str,
    output: Path,
    gate_attr_path: str = "mlp.gate",
) -> dict:
    """The actual profiling work. Caller provides loaded model + tokenizer
    + already-read calibration examples. Both entry points wrap this."""
    num_layers, num_experts, num_experts_per_tok = _moe_geometry(model)
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

    # Register hooks on each MoE layer's router gate. The gate_attr_path
    # is family-specific:
    #   'mlp.gate'                            unfused (Qwen3MoE, OLMoE, DeepSeek-V2)
    #   'block_sparse_moe.gate'               Mixtral / Phi-MoE
    #   'block_sparse_moe.router.layer'       GraniteMoE (fused)
    # The family adapter passes the right path; default is the unfused
    # layout for backwards compat with the morning's qwen3-coder forge.
    registered = 0
    for li in range(num_layers):
        layer = model.model.layers[li]
        gate = _resolve_attr_path(layer, gate_attr_path)
        if gate is None:
            _log(f"  WARN: layer {li} has no {gate_attr_path}")
            continue
        h = gate.register_forward_hook(make_hook(li))
        hooks.append(h)
        registered += 1
    _log(f"  hooks registered on {registered}/{num_layers} layers (path={gate_attr_path})")
    if registered == 0:
        # Loud failure — no silent substitute path. The MoE layout doesn't
        # match what this script knows how to hook; the right answer is to
        # pass the right gate_attr_path from the family adapter, not to
        # silently produce empty counts.
        for h in hooks:
            h.remove()
        raise RuntimeError(
            f"no router gates found on {type(model).__name__} at path "
            f"{gate_attr_path!r}. Family-specific gate paths:\n"
            f"  unfused (Qwen3MoE/OLMoE/DeepSeek-V2): 'mlp.gate'\n"
            f"  Mixtral / Phi-MoE                     : 'block_sparse_moe.gate'\n"
            f"  GraniteMoE fused                      : 'block_sparse_moe.router.layer'\n"
            f"Pass gate_attr_path=... from the family adapter."
        )

    _log(f"running {len(examples)} calibration examples through base model")
    total_tokens = 0
    t0 = time.time()
    try:
        with torch.inference_mode():
            for i, text in enumerate(examples):
                enc = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                input_ids = enc["input_ids"].to(device)
                total_tokens += input_ids.shape[1]
                model(input_ids=input_ids, use_cache=False)
                if (i + 1) % 25 == 0 or i == len(examples) - 1:
                    elapsed = time.time() - t0
                    _log(f"  {i+1}/{len(examples)} examples, {total_tokens} tokens, {elapsed:.1f}s")
    finally:
        for h in hooks:
            h.remove()

    # Build output
    out_data = {
        "model": model_label,
        "calibration_corpus": corpus_label,
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

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(out_data, f, indent=2)
    _log(f"wrote {output}")

    # Quick stats: per-layer top-5 activation counts. Pick first / mid / last
    # layer dynamically so this works on any model depth (Qwen3-Coder-30B has
    # 48 layers, OLMoE has 16, Granite-3.1-3b-a800m has 32, etc.).
    _log("per-layer top expert counts (sample):")
    sample_layers = sorted({0, num_layers // 2, num_layers - 1})
    for li in sample_layers:
        if li not in counts:
            continue
        sorted_idxs = counts[li].argsort(descending=True)[:5].tolist()
        sorted_vals = counts[li][sorted_idxs].tolist()
        zeros = (counts[li] == 0).sum().item()
        _log(f"  layer {li}: top-5 experts {sorted_idxs} counts {sorted_vals}, zeros {zeros}/{num_experts}")

    return out_data


def profile_experts(
    *,
    model,
    tokenizer,
    calibration_data: str | Path,
    output: str | Path,
    max_examples: int | None = None,
    max_length: int = 2048,
    device: str = "cuda:0",
    model_label: str | None = None,
    gate_attr_path: str = "mlp.gate",
) -> dict:
    """Profile expert activation counts on an ALREADY-LOADED model.

    Used by the family-adapter set (MoEUnfusedExpertsBase.expert_activation_profile)
    when ctx.model + ctx.tokenizer are already in memory and the script
    must NOT re-load them.

    Args:
        model:               loaded HuggingFace model object (output of from_pretrained)
        tokenizer:           loaded HuggingFace tokenizer object
        calibration_data:    path to a JSONL of {'text': ...} or {'content': ...} entries
        output:              path to write the importance JSON
        max_examples:        cap on number of calibration examples (None = use all)
        max_length:          max sequence length per example (default 2048)
        device:              device string for inputs (default 'cuda:0')
        model_label:         label written into the output JSON's 'model' field
                             (defaults to type(model).__name__)

    Returns:
        The output dict (also written to `output`).

    Raises:
        ValueError: if model.config doesn't have a recognized MoE expert
                    count or num_experts_per_tok field
        RuntimeError: if no router gates can be hooked (the layout doesn't match)
    """
    calibration_data = Path(calibration_data)
    output = Path(output)

    _log(f"loading calibration corpus from {calibration_data}")
    examples = _read_calibration_corpus(calibration_data, max_examples)
    _log(f"  {len(examples)} examples")

    return _profile_inner(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        max_length=max_length,
        device=device,
        model_label=model_label or type(model).__name__,
        corpus_label=str(calibration_data),
        output=output,
        gate_attr_path=gate_attr_path,
    )


def profile_experts_from_path(
    model_path: str | Path,
    calibration_data: str | Path,
    output: str | Path,
    *,
    max_examples: int | None = None,
    max_length: int = 2048,
    device: str = "cuda:0",
) -> dict:
    """Profile expert activation counts by loading the model from disk in 8-bit.

    Used by the CLI entry point. Loads tokenizer + model from `model_path`
    using BitsAndBytesConfig 8-bit, then delegates to _profile_inner.

    Args:
        model_path:        local path or HF id of the base MoE model
        calibration_data:  path to a JSONL of {'text': ...} entries
        output:            path to write the importance JSON
        max_examples:      cap on calibration examples
        max_length:        max sequence length per example
        device:            device string for both model placement and inputs

    Returns:
        The output dict (also written to `output`).
    """
    calibration_data = Path(calibration_data)
    output = Path(output)

    _log(f"loading calibration corpus from {calibration_data}")
    examples = _read_calibration_corpus(calibration_data, max_examples)
    _log(f"  {len(examples)} examples")

    _log(f"loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    _log(f"loading base model in 8-bit on {device}")
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=bnb_cfg,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()

    return _profile_inner(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        max_length=max_length,
        device=device,
        model_label=str(model_path),
        corpus_label=str(calibration_data),
        output=output,
    )


# ── CLI wrapper ─────────────────────────────────────────────────────────────


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

    profile_experts_from_path(
        model_path=args.model,
        calibration_data=args.calibration_data,
        output=args.output,
        max_examples=args.max_examples,
        max_length=args.max_length,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
