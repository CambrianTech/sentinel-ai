"""
v1 → v1.5 pad-upscale: turn a slice-mode-defragged model into a llama.cpp-
compatible padded equivalent without retraining or re-pruning.

Background. The published v1 14B compacted model has q_proj.shape ==
[3200, 5120], which violates llama.cpp's q_proj_out == hidden_size invariant
(see VALIDATED-TENSOR-SURGERY Finding 6). The v1 cannot load in llama.cpp.
The v1's source weights have been lost (see Finding 6 deepest), so we can't
re-derive it from the original forge state.

The v1.5 path: take the v1's surviving weights and re-pad them into a
[5120, 5120] q_proj layout that satisfies the invariant, while preserving
the trained head→kv mapping so the model produces mathematically identical
output to v1.

The mapping preservation matters. v1's group_size is num_heads_v1 /
num_kv_heads_v1 = 25 / 5 = 5. The padded layout has num_heads = 40 (forced
by hidden_size / head_dim) and num_kv_heads = 5 (unchanged), so new
group_size = 40 / 5 = 8. If we naively put the surviving 25 q heads in
positions 0..24 of the new 40-head layout, head 5 would end up using
kv_group 0 (5 // 8 = 0) instead of kv_group 1 (which it was trained for).
That breaks the model.

The fix: place each kv_group's surviving heads at positions
[g*new_group_size, g*new_group_size + v1_group_size). For Qwen2.5-Coder-14B
v1, that means:
  - v1_kv_0's q heads (rows 0..639 of v1 q_proj) → padded rows 0..639,
    rows 640..1023 zero
  - v1_kv_1's q heads (v1 rows 640..1279) → padded rows 1024..1663,
    rows 1664..2047 zero
  - ... and so on for all 5 kv groups

After this transformation, head h in the padded layout uses kv_group
h // new_group_size, and h // new_group_size matches the v1 kv_group of the
original surviving head at that position. Math is preserved.

KV (k_proj/v_proj) is unchanged: the 5 surviving KV groups stay where they
are, num_kv_heads = 5.

Output:
- New safetensors model dir with q_proj/o_proj of shape [hidden_size, hidden_size]
  and config.num_attention_heads = hidden_size // head_dim
- Stage 2-style assertions on the result
- Smoke test using assert_nondegenerate_output

Use cases:
- Recovering a runnable artifact from a slice-mode-defragged model whose
  source weights are lost
- §4.1.2 deprecation-replacement artifact for the v1 14B compacted model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from harness_checks import (
    assert_explicit_head_dim,
    assert_q_proj_invariant,
    assert_o_proj_invariant,
    assert_nondegenerate_output,
)


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def pad_upscale_layer(attn, hidden_size: int, head_dim: int,
                     v1_num_heads: int, v1_num_kv_heads: int,
                     new_num_heads: int):
    """Pad-upscale one attention layer's q_proj/o_proj from v1 → v1.5 shape.

    Preserves the trained head→kv mapping.
    """
    v1_group_size = v1_num_heads // v1_num_kv_heads
    new_group_size = new_num_heads // v1_num_kv_heads
    if v1_num_heads % v1_num_kv_heads != 0:
        raise AssertionError(
            f"v1 num_heads={v1_num_heads} not divisible by v1 num_kv_heads="
            f"{v1_num_kv_heads}. Cannot reconstruct head→kv mapping."
        )
    if new_num_heads % v1_num_kv_heads != 0:
        raise AssertionError(
            f"new num_heads={new_num_heads} not divisible by v1 num_kv_heads="
            f"{v1_num_kv_heads}. Padded layout would break GQA constraint."
        )
    if new_group_size < v1_group_size:
        raise AssertionError(
            f"new group_size={new_group_size} < v1 group_size={v1_group_size}. "
            f"Cannot fit v1's surviving heads into the padded layout."
        )

    q = attn.q_proj
    o = attn.o_proj

    # Sanity: v1 q_proj must be (v1_num_heads * head_dim, hidden_size)
    expected_v1_q = (v1_num_heads * head_dim, hidden_size)
    if tuple(q.weight.shape) != expected_v1_q:
        raise AssertionError(
            f"v1 q_proj shape {tuple(q.weight.shape)} != expected {expected_v1_q}. "
            f"Either v1_num_heads/head_dim are wrong, or this layer is not "
            f"from a v1-style sliced model."
        )
    expected_v1_o = (hidden_size, v1_num_heads * head_dim)
    if tuple(o.weight.shape) != expected_v1_o:
        raise AssertionError(
            f"v1 o_proj shape {tuple(o.weight.shape)} != expected {expected_v1_o}."
        )

    device = q.weight.device
    dtype = q.weight.dtype

    new_q_w = torch.zeros(new_num_heads * head_dim, hidden_size, device=device, dtype=dtype)
    new_o_w = torch.zeros(hidden_size, new_num_heads * head_dim, device=device, dtype=dtype)
    new_q_b = None
    if q.bias is not None:
        new_q_b = torch.zeros(new_num_heads * head_dim, device=device, dtype=q.bias.dtype)

    for kv_g in range(v1_num_kv_heads):
        # Source: v1's heads for this kv group occupy [kv_g*v1_group_size,
        # (kv_g+1)*v1_group_size) of the v1 q-head index space.
        v1_head_start = kv_g * v1_group_size
        v1_head_end = (kv_g + 1) * v1_group_size  # exclusive
        v1_row_start = v1_head_start * head_dim
        v1_row_end = v1_head_end * head_dim

        # Destination: place those v1_group_size surviving heads at the
        # FRONT of this kv group's slot in the new layout. The remaining
        # (new_group_size - v1_group_size) head slots in this kv group
        # stay zero.
        new_head_start = kv_g * new_group_size
        new_head_end = new_head_start + v1_group_size  # exclusive
        new_row_start = new_head_start * head_dim
        new_row_end = new_head_end * head_dim

        # Copy q_proj rows
        new_q_w[new_row_start:new_row_end, :] = q.weight.data[v1_row_start:v1_row_end, :]
        # Copy o_proj cols
        new_o_w[:, new_row_start:new_row_end] = o.weight.data[:, v1_row_start:v1_row_end]
        # Copy q_proj bias rows if present
        if new_q_b is not None and q.bias is not None:
            new_q_b[new_row_start:new_row_end] = q.bias.data[v1_row_start:v1_row_end]

    # Replace the layer's q_proj and o_proj with new Linear modules of the
    # padded shape. We rebuild rather than mutate so the wrapped modules
    # report correct out_features/in_features attributes.
    import torch.nn as nn

    new_q = nn.Linear(hidden_size, new_num_heads * head_dim, bias=q.bias is not None,
                      dtype=dtype, device=device)
    new_q.weight = nn.Parameter(new_q_w)
    if new_q_b is not None:
        new_q.bias = nn.Parameter(new_q_b)
    attn.q_proj = new_q

    new_o = nn.Linear(new_num_heads * head_dim, hidden_size, bias=o.bias is not None,
                      dtype=dtype, device=device)
    new_o.weight = nn.Parameter(new_o_w)
    if o.bias is not None:
        # Bias on o_proj is uncommon but handle it
        new_o.bias = nn.Parameter(o.bias.data.clone())
    attn.o_proj = new_o

    # Per-attention cached attributes
    if hasattr(attn, "num_heads"):
        attn.num_heads = new_num_heads
    if hasattr(attn, "num_key_value_heads"):
        attn.num_key_value_heads = v1_num_kv_heads  # unchanged
    if hasattr(attn, "num_key_value_groups"):
        attn.num_key_value_groups = new_num_heads // v1_num_kv_heads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("v1_dir", help="Input v1 model directory (slice-mode-defragged)")
    ap.add_argument("out_dir", help="Output directory for v1.5 padded model")
    args = ap.parse_args()

    _log(f"Loading v1 from {args.v1_dir} (fp16, no bnb override)...")
    config = AutoConfig.from_pretrained(args.v1_dir)
    # The v1 must carry head_dim explicitly. The dequantized v1 should have it
    # because we copied the source's config.json which had it.
    v1_head_dim = assert_explicit_head_dim(config)
    v1_num_heads = config.num_attention_heads
    v1_num_kv = config.num_key_value_heads
    hidden_size = config.hidden_size
    _log(f"v1 config: hidden={hidden_size} num_heads={v1_num_heads} num_kv={v1_num_kv} head_dim={v1_head_dim}")
    _log(f"v1 q_proj layout: {v1_num_heads}*{v1_head_dim}={v1_num_heads*v1_head_dim} rows")

    if v1_num_heads * v1_head_dim == hidden_size:
        raise AssertionError(
            f"v1 already satisfies q_proj_out == hidden_size "
            f"({v1_num_heads}*{v1_head_dim}={hidden_size}). Nothing to upscale; "
            f"the v1 should already load in llama.cpp. If you're trying to use "
            f"this script on a non-broken model, you don't need it."
        )

    # The new num_heads is the only value that satisfies q_proj_out ==
    # hidden_size with the existing head_dim.
    if hidden_size % v1_head_dim != 0:
        raise AssertionError(
            f"hidden_size={hidden_size} not divisible by head_dim={v1_head_dim}. "
            f"Cannot construct a padded num_heads that satisfies the invariant."
        )
    new_num_heads = hidden_size // v1_head_dim
    _log(f"target padded layout: num_heads={new_num_heads}, num_kv_heads={v1_num_kv} (unchanged)")
    _log(f"v1 group_size={v1_num_heads//v1_num_kv}, new group_size={new_num_heads//v1_num_kv}")

    # Load model in fp16 explicitly. Even if the saved config has any leftover
    # quantization metadata, we want fp16 weights so we can mutate them.
    model = AutoModelForCausalLM.from_pretrained(
        args.v1_dir, torch_dtype=torch.float16, device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.v1_dir)

    a0 = model.model.layers[0].self_attn
    _log(f"v1 layer 0 shapes: q_proj={tuple(a0.q_proj.weight.shape)} o_proj={tuple(a0.o_proj.weight.shape)} k_proj={tuple(a0.k_proj.weight.shape)} v_proj={tuple(a0.v_proj.weight.shape)}")
    if a0.q_proj.bias is not None:
        _log(f"v1 layer 0 q_proj.bias: {tuple(a0.q_proj.bias.shape)}")

    # BIT-IDENTICAL GATE — capture v1 logits BEFORE mutation so we can compare
    # against v1.5 logits AFTER mutation. The pad transformation is supposed
    # to be mathematically equivalent (same surviving heads, same head→kv
    # mapping, padded positions contribute literally zero); the only honest
    # way to verify this is to run both forms on the same input and assert
    # torch.equal. If this assertion fires, the padding scrambled something
    # and the v1.5 artifact is invalid.
    _log("Capturing v1 logits for bit-identical verification...")
    test_prompt = "def fibonacci(n):"
    test_ids = tokenizer(test_prompt, return_tensors="pt").to("cuda")
    model.eval()
    with torch.no_grad():
        v1_logits = model(**test_ids).logits.detach().clone()
    _log(f"  v1 logits shape: {tuple(v1_logits.shape)}, dtype: {v1_logits.dtype}")

    _log("Pad-upscaling all attention layers...")
    n_layers = len(model.model.layers)
    for li in range(n_layers):
        attn = model.model.layers[li].self_attn
        pad_upscale_layer(attn, hidden_size, v1_head_dim, v1_num_heads, v1_num_kv, new_num_heads)
    _log(f"  upscaled {n_layers} layers")

    # Update the model's top-level config to reflect the new layout. The
    # internal compute uses new_num_heads positions, but the per-layer attn
    # cache attributes were set inside pad_upscale_layer.
    model.config.num_attention_heads = new_num_heads
    model.config.head_dim = v1_head_dim  # explicit, must survive save_pretrained

    a0_after = model.model.layers[0].self_attn
    _log(f"v1.5 layer 0 shapes: q_proj={tuple(a0_after.q_proj.weight.shape)} o_proj={tuple(a0_after.o_proj.weight.shape)} k_proj={tuple(a0_after.k_proj.weight.shape)} v_proj={tuple(a0_after.v_proj.weight.shape)}")
    _log(f"v1.5 cfg: num_heads={model.config.num_attention_heads} num_kv={model.config.num_key_value_heads} head_dim={model.config.head_dim}")

    # HARD CHECKS — shape invariants
    assert_q_proj_invariant(a0_after.q_proj.weight, hidden_size)
    assert_o_proj_invariant(a0_after.o_proj.weight, hidden_size)
    if model.config.num_attention_heads != new_num_heads:
        raise AssertionError(
            f"post-upscale num_attention_heads={model.config.num_attention_heads} "
            f"!= expected {new_num_heads}"
        )
    assert_explicit_head_dim(model.config)
    _log("  shape invariants PASSED")

    # BIT-IDENTICAL GATE — the strongest possible verification of pad-upscale
    # correctness. If this passes, we have a mathematical guarantee that v1.5
    # behavior == v1 behavior at fp16 precision. The only remaining quality
    # question after this is whether GGUF q5_K_S quantization preserves it,
    # which is the Layer 7 gate's job, not this script's.
    _log("BIT-IDENTICAL gate: re-running same input through v1.5 model...")
    with torch.no_grad():
        v15_logits = model(**test_ids).logits.detach()
    _log(f"  v1.5 logits shape: {tuple(v15_logits.shape)}, dtype: {v15_logits.dtype}")

    if v1_logits.shape != v15_logits.shape:
        raise AssertionError(
            f"v1 logits shape {tuple(v1_logits.shape)} != v1.5 logits shape "
            f"{tuple(v15_logits.shape)}. Padding changed the output dimensions, "
            f"which means num_attention_heads or some other config is wrong."
        )

    if not torch.equal(v1_logits, v15_logits):
        max_diff = (v1_logits - v15_logits).abs().max().item()
        mean_diff = (v1_logits - v15_logits).abs().mean().item()
        # Find the worst position so the failure message is actionable
        diff = (v1_logits - v15_logits).abs()
        worst_idx = diff.argmax().item()
        raise AssertionError(
            f"v1.5 padding scrambled the model: logits are NOT bit-identical "
            f"to v1. max_abs_diff={max_diff:.6e}, mean_abs_diff={mean_diff:.6e}, "
            f"worst_flat_idx={worst_idx}. The pad-upscale transformation broke "
            f"the trained head→kv mapping, or padded positions are leaking "
            f"non-zero contributions, or something else corrupted the compute. "
            f"This MUST be fixed before the GGUF export — the v1.5 wrapper is "
            f"only valid if it produces literally identical output to v1."
        )
    _log("  BIT-IDENTICAL gate PASSED — v1.5 logits match v1 logits exactly")

    # Smoke test
    _log("Smoke test generation...")
    prompt = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return"
    ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=60, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    _log(f"  generation: {text!r}")
    assert_nondegenerate_output(text, prompt)
    _log("  smoke test PASSED")

    _log(f"Saving v1.5 to {args.out_dir}...")
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.out_dir)

    # Verify the SAVED config still has head_dim. save_pretrained sometimes
    # drops fields it doesn't recognize, and a saved config without head_dim
    # would be the v1 bug recurring at the wrong layer.
    saved_cfg = AutoConfig.from_pretrained(args.out_dir)
    assert_explicit_head_dim(saved_cfg)
    if saved_cfg.num_attention_heads != new_num_heads:
        raise AssertionError(
            f"saved cfg num_attention_heads={saved_cfg.num_attention_heads} "
            f"!= {new_num_heads}; save_pretrained corrupted the config"
        )
    _log("  saved config head_dim and num_heads verified")

    # Write a small provenance file so the v1.5 artifact knows where it came from
    provenance = {
        "tool": "v1_to_v15_pad_upscale",
        "tool_version": "1.0.0",
        "source_v1_dir": str(args.v1_dir),
        "source_v1_config": {
            "num_attention_heads": v1_num_heads,
            "num_key_value_heads": v1_num_kv,
            "head_dim": v1_head_dim,
            "hidden_size": hidden_size,
            "v1_group_size": v1_num_heads // v1_num_kv,
        },
        "v1_5_config": {
            "num_attention_heads": new_num_heads,
            "num_key_value_heads": v1_num_kv,
            "head_dim": v1_head_dim,
            "hidden_size": hidden_size,
            "new_group_size": new_num_heads // v1_num_kv,
        },
        "transformation": "per-kv-group placement preserves trained head→kv mapping",
        "compute_equivalence": "the v1.5 model is mathematically equivalent to the v1 (same surviving heads, same kv mapping); the additional head positions are zero and contribute zero to the residual stream",
    }
    (out_path / "v1_to_v15_provenance.json").write_text(json.dumps(provenance, indent=2))

    _log("Done.")


if __name__ == "__main__":
    main()
