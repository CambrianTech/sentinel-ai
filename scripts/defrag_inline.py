#!/usr/bin/env python3
"""
defrag_inline.py — In-place structural pruning for live models during training.

Unlike defrag_model.py (post-processing on saved files), this module provides
functions that defrag a live PyTorch model in-place. Used between forge cycles
to shrink the model and free VRAM for larger batch sizes.

The key challenge: HF model code caches num_heads/num_kv_heads in the attention
module. After slicing tensors, we must update these cached values or the
forward pass will crash on view() reshape.

Usage (inside forge_model.py):
    from defrag_inline import defrag_live_model

    # After LoRA merge, before next LoRA attach:
    freed_bytes = defrag_live_model(model, pruning_mask)
    print(f"Freed {freed_bytes / 1e9:.2f}GB — can increase batch size")
"""

import torch
import torch.nn as nn


def get_layers(model):
    """Find transformer layer list."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Cannot find model layers")


def nested_config(config):
    return getattr(config, "text_config", config)


def detect_dead_heads_live(model, threshold=1e-6):
    """Detect zeroed-out heads in a live model. Returns {layer_idx: [head_indices]}."""
    tc = nested_config(model.config)
    num_heads = getattr(tc, "num_attention_heads", 12)
    head_dim = getattr(tc, "hidden_size", 768) // num_heads
    layers = get_layers(model)

    dead = {}
    for li, layer in enumerate(layers):
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
            if e <= w.shape[0] and w[s:e].norm().item() < threshold:
                layer_dead.append(hi)
        if layer_dead:
            dead[li] = layer_dead

    return dead


def _make_linear(weight_data, bias_data=None):
    """Create a new Linear layer from weight tensor."""
    out_features, in_features = weight_data.shape
    linear = nn.Linear(in_features, out_features, bias=bias_data is not None,
                       dtype=weight_data.dtype, device=weight_data.device)
    linear.weight = nn.Parameter(weight_data)
    if bias_data is not None:
        linear.bias = nn.Parameter(bias_data)
    return linear


def defrag_attention_layer(attn, surviving_q_heads, surviving_kv_heads,
                           q_head_dim, kv_head_dim, o_head_dim,
                           orig_num_heads, orig_num_kv_heads):
    """
    Structurally remove dead heads from one attention layer.
    Uses actual tensor dimensions (not config estimates).

    Returns: bytes freed
    """
    bytes_before = sum(p.numel() * p.element_size() for p in attn.parameters())

    # Q indices
    q_indices = []
    for h in surviving_q_heads:
        q_indices.extend(range(h * q_head_dim, (h + 1) * q_head_dim))

    # KV indices
    kv_indices = []
    for h in surviving_kv_heads:
        kv_indices.extend(range(h * kv_head_dim, (h + 1) * kv_head_dim))

    # O indices (may differ from Q)
    o_indices = []
    for h in surviving_q_heads:
        o_indices.extend(range(h * o_head_dim, (h + 1) * o_head_dim))

    # Slice Q projection
    q = getattr(attn, "q_proj", None)
    if q and hasattr(q, "weight") and max(q_indices, default=0) < q.weight.shape[0]:
        new_q_weight = q.weight.data[q_indices, :].contiguous()
        new_q_bias = q.bias.data[q_indices].contiguous() if q.bias is not None else None
        attn.q_proj = _make_linear(new_q_weight, new_q_bias)

    # Slice K, V projections
    for name in ["k_proj", "v_proj"]:
        proj = getattr(attn, name, None)
        if proj and hasattr(proj, "weight") and max(kv_indices, default=0) < proj.weight.shape[0]:
            new_weight = proj.weight.data[kv_indices, :].contiguous()
            new_bias = proj.bias.data[kv_indices].contiguous() if proj.bias is not None else None
            setattr(attn, name, _make_linear(new_weight, new_bias))

    # Slice O projection (columns, not rows)
    o = getattr(attn, "o_proj", None)
    if o and hasattr(o, "weight") and max(o_indices, default=0) < o.weight.shape[1]:
        new_o_weight = o.weight.data[:, o_indices].contiguous()
        attn.o_proj = _make_linear(new_o_weight)

    # Update cached attributes that HF forward() uses for view() reshaping
    new_num_heads = len(surviving_q_heads)
    new_num_kv_heads = len(surviving_kv_heads)
    new_num_kv_groups = new_num_heads // max(new_num_kv_heads, 1)

    for attr, val in [
        ("num_heads", new_num_heads),
        ("num_key_value_heads", new_num_kv_heads),
        ("num_key_value_groups", new_num_kv_groups),
        # Some architectures use these names:
        ("n_head", new_num_heads),
        ("n_kv_head", new_num_kv_heads),
    ]:
        if hasattr(attn, attr):
            setattr(attn, attr, val)

    bytes_after = sum(p.numel() * p.element_size() for p in attn.parameters())
    return bytes_before - bytes_after


def defrag_live_model(model, dead_heads=None, threshold=1e-6):
    """
    Defrag a live model in-place. Detects dead heads (or uses provided mask),
    slices attention tensors, updates cached config.

    Args:
        model: live PyTorch model (on GPU is fine)
        dead_heads: optional pre-computed {layer_idx: [head_indices]}
        threshold: weight norm threshold for dead head detection

    Returns:
        total bytes freed
    """
    if dead_heads is None:
        dead_heads = detect_dead_heads_live(model, threshold)

    if not dead_heads:
        return 0

    tc = nested_config(model.config)
    num_heads = getattr(tc, "num_attention_heads", 12)
    num_kv_heads = getattr(tc, "num_key_value_heads", num_heads)
    group_size = num_heads // max(num_kv_heads, 1)

    layers = get_layers(model)
    total_freed = 0
    defragged_layers = 0

    # Get actual head dims from tensor shapes (not config — config lies for hybrid models)
    sample_attn = None
    for layer in layers:
        sample_attn = getattr(layer, "self_attn", getattr(layer, "attn", None))
        if sample_attn and hasattr(sample_attn, "q_proj"):
            break

    if sample_attn is None:
        return 0

    q_proj = getattr(sample_attn, "q_proj", None)
    k_proj = getattr(sample_attn, "k_proj", None)
    o_proj = getattr(sample_attn, "o_proj", None)

    q_head_dim = q_proj.weight.shape[0] // num_heads if q_proj else 0
    kv_head_dim = k_proj.weight.shape[0] // num_kv_heads if k_proj else 0
    o_head_dim = o_proj.weight.shape[1] // num_heads if o_proj else 0

    for li, dead_list in dead_heads.items():
        if li >= len(layers):
            continue

        # Compute surviving heads
        surviving_q = [h for h in range(num_heads) if h not in dead_list]
        surviving_kv = []
        for kv_h in range(num_kv_heads):
            group = range(kv_h * group_size, (kv_h + 1) * group_size)
            if not all(h in dead_list for h in group):
                surviving_kv.append(kv_h)

        attn = getattr(layers[li], "self_attn", getattr(layers[li], "attn", None))
        if attn is None:
            continue

        freed = defrag_attention_layer(
            attn, surviving_q, surviving_kv,
            q_head_dim, kv_head_dim, o_head_dim,
            num_heads, num_kv_heads
        )
        total_freed += freed
        defragged_layers += 1

    # Force garbage collection to actually free GPU memory
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return total_freed


def estimate_defrag_savings(model, dead_heads=None, threshold=1e-6):
    """Estimate how much VRAM defrag would free, without actually doing it."""
    if dead_heads is None:
        dead_heads = detect_dead_heads_live(model, threshold)

    tc = nested_config(model.config)
    num_heads = getattr(tc, "num_attention_heads", 12)
    num_kv_heads = getattr(tc, "num_key_value_heads", num_heads)
    head_dim = getattr(tc, "hidden_size", 768) // num_heads
    group_size = num_heads // num_kv_heads

    total_savings = 0
    for li, dead_list in dead_heads.items():
        n_dead_q = len(dead_list)
        # Count dead KV heads (full groups only)
        n_dead_kv = 0
        for kv_h in range(num_kv_heads):
            group = range(kv_h * group_size, (kv_h + 1) * group_size)
            if all(h in dead_list for h in group):
                n_dead_kv += 1

        hidden = getattr(tc, "hidden_size", 768)
        # Q savings: n_dead_q * head_dim rows removed from [heads*dim, hidden]
        q_saved = n_dead_q * head_dim * hidden * 2  # fp16
        # KV savings: n_dead_kv * head_dim rows each for K and V
        kv_saved = n_dead_kv * head_dim * hidden * 2 * 2  # K + V, fp16
        # O savings: n_dead_q * head_dim columns removed from [hidden, heads*dim]
        o_saved = hidden * n_dead_q * head_dim * 2  # fp16

        total_savings += q_saved + kv_saved + o_saved

    return total_savings, dead_heads
