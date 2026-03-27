#!/usr/bin/env python3
"""
defrag_model.py — Structural pruning: remove dead heads, shrink tensors.

Runs AFTER forging. Reads a forged model + pruning mask, slices out the
dead attention heads, and saves a physically smaller model.

The forge trains with forward-hook masks (heads compute but output zeroed).
Defrag removes them from the tensors entirely — smaller VRAM, faster inference,
smaller GGUF.

Usage:
    python scripts/defrag_model.py output/forged/qwen3.5-4b/
    python scripts/defrag_model.py output/forged/qwen3.5-27b/ --output output/defragged/qwen3.5-27b/
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def nested_config(config):
    """Get text config from possibly nested VLM config."""
    return getattr(config, "text_config", config)


def detect_dead_heads(model, threshold=1e-6):
    """
    Detect pruned heads by finding Q-projection rows that are all zeros
    (or near-zero from the hook-based masking).

    Returns: dict[layer_idx] -> list[head_idx] of dead heads
    """
    tc = nested_config(model.config)
    num_layers = getattr(tc, "num_hidden_layers", 12)
    num_heads = getattr(tc, "num_attention_heads", 12)
    head_dim = getattr(tc, "hidden_size", 768) // num_heads

    # Find layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        print("ERROR: Cannot find model layers")
        sys.exit(1)

    dead_heads = {}
    total_dead = 0

    for li in range(num_layers):
        layer = layers[li]
        attn = getattr(layer, "self_attn", getattr(layer, "attn", None))
        if attn is None:
            continue

        q = getattr(attn, "q_proj", None)
        if q is None or not hasattr(q, "weight"):
            continue

        w = q.weight.data.float()
        layer_dead = []

        for hi in range(num_heads):
            s, e = hi * head_dim, (hi + 1) * head_dim
            if e <= w.shape[0]:
                head_norm = w[s:e].norm().item()
                if head_norm < threshold:
                    layer_dead.append(hi)

        if layer_dead:
            dead_heads[li] = layer_dead
            total_dead += len(layer_dead)

    total_heads = num_layers * num_heads
    print(f"  Dead heads: {total_dead}/{total_heads} ({total_dead/total_heads*100:.1f}%)")
    print(f"  Layers with dead heads: {len(dead_heads)}/{num_layers}")

    return dead_heads


def compute_surviving_heads(num_heads, num_kv_heads, dead_heads_in_layer):
    """
    Given dead query heads, compute surviving query and KV heads.
    For GQA: a KV head is removed only if ALL its query heads are dead.
    """
    group_size = num_heads // num_kv_heads

    surviving_q = [h for h in range(num_heads) if h not in dead_heads_in_layer]

    surviving_kv = []
    for kv_h in range(num_kv_heads):
        group_start = kv_h * group_size
        group = list(range(group_start, group_start + group_size))
        if not all(h in dead_heads_in_layer for h in group):
            surviving_kv.append(kv_h)

    return surviving_q, surviving_kv


def defrag_layer(attn, surviving_q, surviving_kv, head_dim, num_heads, num_kv_heads):
    """
    Slice attention projection weights to remove dead heads.
    Returns new weight tensors (not in-place — safe).
    """
    results = {}

    # Q projection: keep rows for surviving query heads
    q_indices = []
    for h in surviving_q:
        q_indices.extend(range(h * head_dim, (h + 1) * head_dim))

    q = getattr(attn, "q_proj", None)
    if q and hasattr(q, "weight"):
        results["q_proj"] = q.weight.data[q_indices, :]

    # K/V projections: keep rows for surviving KV heads
    kv_indices = []
    for h in surviving_kv:
        kv_indices.extend(range(h * head_dim, (h + 1) * head_dim))

    for name in ["k_proj", "v_proj"]:
        proj = getattr(attn, name, None)
        if proj and hasattr(proj, "weight"):
            results[name] = proj.weight.data[kv_indices, :]

    # O projection: keep columns for surviving query heads
    o = getattr(attn, "o_proj", None)
    if o and hasattr(o, "weight"):
        results["o_proj"] = o.weight.data[:, q_indices]

    return results


def defrag_model(model_path: Path, output_path: Path):
    """
    Load a forged model, detect dead heads, slice them out, save smaller model.
    """
    print(f"\n[1] Loading model from {model_path}/model/")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path / "model"),
        torch_dtype=torch.float16,
        device_map="cpu",  # CPU for surgery — no VRAM needed
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path / "model"))

    tc = nested_config(model.config)
    num_heads = getattr(tc, "num_attention_heads", 12)
    num_kv_heads = getattr(tc, "num_key_value_heads", num_heads)
    head_dim = getattr(tc, "hidden_size", 768) // num_heads

    # Get original param count
    orig_params = sum(p.numel() for p in model.parameters())
    orig_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9

    print(f"\n[2] Detecting dead heads...")
    dead_heads = detect_dead_heads(model)

    if not dead_heads:
        print("  No dead heads found — model was not pruned or masks not merged.")
        print("  Nothing to defrag.")
        return

    # Find layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        layers = model.transformer.h

    # Compute per-layer surviving heads
    print(f"\n[3] Defragmenting attention layers...")
    layer_stats = []

    for li in range(len(layers)):
        dead_in_layer = dead_heads.get(li, [])
        if not dead_in_layer:
            layer_stats.append({"layer": li, "surviving_q": num_heads, "surviving_kv": num_kv_heads})
            continue

        surviving_q, surviving_kv = compute_surviving_heads(
            num_heads, num_kv_heads, dead_in_layer
        )

        attn = getattr(layers[li], "self_attn", getattr(layers[li], "attn", None))
        if attn is None:
            continue

        new_weights = defrag_layer(attn, surviving_q, surviving_kv, head_dim, num_heads, num_kv_heads)

        # Apply new weights
        for name, weight in new_weights.items():
            proj = getattr(attn, name)
            # Create new linear with correct dimensions
            new_linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=weight.dtype)
            new_linear.weight.data = weight
            setattr(attn, name, new_linear)

        n_q = len(surviving_q)
        n_kv = len(surviving_kv)
        layer_stats.append({"layer": li, "surviving_q": n_q, "surviving_kv": n_kv})

        if n_q < num_heads:
            print(f"  Layer {li}: {num_heads}→{n_q} query heads, {num_kv_heads}→{n_kv} KV heads")

    # Compute new param count
    new_params = sum(p.numel() for p in model.parameters())
    new_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    saved_pct = (1 - new_params / orig_params) * 100

    print(f"\n[4] Results:")
    print(f"  Original: {orig_params/1e9:.2f}B params, {orig_size_gb:.2f}GB")
    print(f"  Defragged: {new_params/1e9:.2f}B params, {new_size_gb:.2f}GB")
    print(f"  Saved: {saved_pct:.1f}% parameters")

    # Save
    print(f"\n[5] Saving to {output_path}/")
    output_path.mkdir(parents=True, exist_ok=True)

    # Note: saving with variable head counts per layer requires custom handling
    # For now, save as-is — the model works but HF config won't reflect per-layer changes
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Save defrag metadata
    metadata = {
        "original_params": orig_params,
        "defragged_params": new_params,
        "saved_pct": round(saved_pct, 2),
        "original_size_gb": round(orig_size_gb, 3),
        "defragged_size_gb": round(new_size_gb, 3),
        "dead_heads": {str(k): v for k, v in dead_heads.items()},
        "layer_stats": layer_stats,
    }
    (output_path / "defrag_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"  Saved model + defrag_metadata.json")
    print(f"\n  Next: convert to GGUF with llama.cpp for inference")


def main():
    parser = argparse.ArgumentParser(description="Defrag a forged model — remove dead heads")
    parser.add_argument("forged_dir", help="Path to forged model output (e.g., output/forged/qwen3.5-4b/)")
    parser.add_argument("--output", help="Output directory (default: <forged_dir>/defragged/)")
    parser.add_argument("--threshold", type=float, default=1e-6,
                       help="Weight norm threshold for dead head detection")
    args = parser.parse_args()

    forged = Path(args.forged_dir)
    output = Path(args.output) if args.output else forged / "defragged"

    if not (forged / "model").exists():
        print(f"ERROR: No model found at {forged}/model/")
        print("Run forge_model.py first.")
        sys.exit(1)

    defrag_model(forged, output)


if __name__ == "__main__":
    main()
