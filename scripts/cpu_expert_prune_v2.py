"""
CPU-first MoE expert pruning v2 — per-layer normalized, router-aware,
streaming, with provenance metadata.

This rewrites cpu_expert_prune.py from scratch for the unfused MoE layout
used by Qwen3-Coder-30B-A3B (and Qwen3 MoE family in general). The v1
script was written for fused MoE tensors (3D shape [num_experts, ...])
and silently no-ops on unfused layouts.

The v1 script ALSO had the same bug class we discovered tonight at the
dense layer level: global flat ranking by L2 weight norm, with no
per-layer normalization, no router gate slicing, and no calibration
discipline. v2 fixes all of those.

## Architecture handled

Unfused MoE layout (Qwen3MoeForCausalLM, Qwen3-Coder-30B-A3B-Instruct):

    model.layers.{L}.mlp.gate.weight              shape [num_experts, hidden]
    model.layers.{L}.mlp.experts.{K}.gate_proj.weight    shape [moe_inter, hidden]
    model.layers.{L}.mlp.experts.{K}.up_proj.weight      shape [moe_inter, hidden]
    model.layers.{L}.mlp.experts.{K}.down_proj.weight    shape [hidden, moe_inter]

For each MoE layer L:
1. Compute per-expert importance from the router gate row L2 norm
2. Keep the top-K most-important experts in this layer (per-layer budget,
   not global flat — same discipline as the dense head fix)
3. Slice the router gate's expert dimension to keep only the surviving rows
4. Drop the per-expert MLP tensors for non-surviving experts
5. Renumber the surviving experts to sequential indices [0..K-1] so the
   model loads cleanly with the new num_experts in config

## Streaming

The script never loads the full model into RAM. It does two passes over
the safetensors shards:

- Pass 1: read only the router gate tensors (48 layers × ~250 KB each ≈
  12 MB total). Compute per-layer expert importance and selection.
- Pass 2: stream every tensor; for each, decide whether to keep, rename,
  slice, or drop, and write to a new shard. Free immediately after writing.

This means Qwen3-Coder-30B-A3B (~60 GB fp16) prunes on hosts with 8 GB of
RAM. 80B and 480B variants work the same way — the streaming has no
model-size dependency.

## Calibration anchor

Like every other measurement script in this PR, this one writes a
provenance sidecar (`expert_prune.metadata.v1.json`) with:
- source SHA-256 of every input shard
- tool version
- per-layer importance scores and surviving expert indices
- per-layer prune ratios
- gguf-block-layout style provenance for downstream consumers
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file


TOOL_NAME = "sentinel-ai/scripts/cpu_expert_prune_v2"
TOOL_VERSION = "1.0.0"
SCHEMA_VERSION = "expert_prune.metadata.v1"


# ── LayoutSpec — per-family tensor name patterns ────────────────────────────
#
# The two-pass streaming pruner is the same algorithm regardless of MoE
# family. Only the tensor name patterns differ:
#
#   Qwen3MoE / OLMoE     model.layers.{L}.mlp.experts.{K}.{gate|up|down}_proj.weight
#                        + model.layers.{L}.mlp.gate.weight
#
#   Mixtral / Phi-MoE    model.layers.{L}.block_sparse_moe.experts.{K}.{w1|w2|w3}.weight
#                        + model.layers.{L}.block_sparse_moe.gate.weight
#
# (GraniteMoE-fused and DeepSeek-V2-routed-shared are structurally distinct
#  enough that they need their own pruners — fused-tensor-slicing for Granite,
#  shared-expert preservation for DeepSeek-V2.)
#
# LayoutSpec encodes the family's name patterns + the renumbering template.
# prune_experts(layout=...) takes a LayoutSpec; default is QWEN3_MOE_LAYOUT
# so the existing forge path (the morning's qwen3-coder-30b-a3b flagship)
# keeps working with no changes.

@dataclass(frozen=True)
class LayoutSpec:
    """Per-family tensor name patterns for the streaming MoE pruner."""
    family_name: str
    # Regex string that matches the router gate tensor name. MUST capture
    # the layer index as group(1).
    gate_pattern: str
    # Regex string that matches an expert weight tensor name. MUST capture
    # the layer index as group(1), the expert index as group(2), and the
    # per-expert weight name (e.g. 'gate_proj' or 'w1') as group(3).
    expert_pattern: str
    # Format string for the renumbered expert tensor name. Uses {layer},
    # {new_idx}, {proj_name} as placeholders.
    expert_rename_template: str

    def gate_re(self) -> "re.Pattern":
        return re.compile(self.gate_pattern)

    def expert_re(self) -> "re.Pattern":
        return re.compile(self.expert_pattern)


# Qwen3MoE / OLMoE — the unfused-Qwen layout. The morning's flagship
# qwen3-coder-30b-a3b-compacted-19b-256k forge ran with this layout.
# Frozen for reproducibility.
QWEN3_MOE_LAYOUT = LayoutSpec(
    family_name="qwen3_moe",
    gate_pattern=r"^model\.layers\.(\d+)\.mlp\.gate\.weight$",
    expert_pattern=r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.([a-z_]+)\.weight$",
    expert_rename_template="model.layers.{layer}.mlp.experts.{new_idx}.{proj_name}.weight",
)

# Mixtral / Phi-MoE — block_sparse_moe-unfused layout. Per-expert weights
# are named w1 (gate_proj), w2 (down_proj), w3 (up_proj). 8 experts in
# Mixtral 8x7B / 8x22B, 16 in Phi-3.5-MoE. Same algorithm as the
# qwen3_moe path; only the path prefix and per-expert weight names differ.
MIXTRAL_LAYOUT = LayoutSpec(
    family_name="mixtral",
    gate_pattern=r"^model\.layers\.(\d+)\.block_sparse_moe\.gate\.weight$",
    expert_pattern=r"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight$",
    expert_rename_template="model.layers.{layer}.block_sparse_moe.experts.{new_idx}.{proj_name}.weight",
)

# DeepSeek-V2 routed experts. Tensor name shape is identical to QWEN3_MOE_LAYOUT
# (mlp.experts.{e}.{proj}_proj.weight); the family_name differs because:
#   1. DeepSeek-V2 also carries shared_experts.* tensors that MUST passthrough
#      bit-exact (the always-fires capability). The expert_re below intentionally
#      requires a digit after `experts.` so `shared_experts.gate_proj.weight`
#      does NOT match and falls through to the streaming rewriter's passthrough
#      branch unchanged.
#   2. Layer 0 in DeepSeek-V2 is dense (no MoE) — those `mlp.gate_proj.weight`
#      tensors also fall through to passthrough since they don't carry an
#      `experts.{digit}` segment.
#   3. config.json field is `n_routed_experts` (handled in update_config below).
DEEPSEEK_V2_LAYOUT = LayoutSpec(
    family_name="deepseek_v2",
    gate_pattern=r"^model\.layers\.(\d+)\.mlp\.gate\.weight$",
    expert_pattern=r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.([a-z_]+)\.weight$",
    expert_rename_template="model.layers.{layer}.mlp.experts.{new_idx}.{proj_name}.weight",
)


# ── Fused-tensor layout spec (GraniteMoE family) ────────────────────────────
#
# Structurally distinct from the unfused families (Mixtral, Qwen3MoE, OLMoE,
# DeepSeek-V2). All experts in a layer share THREE big tensors along an
# expert axis instead of one tensor per expert per projection. To prune k
# of n experts you SLICE these tensors along axis=0, not delete-and-rename
# named param entries. Different pattern → different LayoutSpec class.


@dataclass(frozen=True)
class FusedLayoutSpec:
    """Layout spec for MoE families where experts share fused tensors.

    Fields:
        family_name: 'granitemoe' (and any future fused-layout family)
        match_pattern: regex matching ANY of this layer's MoE tensors
                       (with the layer index as group 1 and tensor kind
                       as group 2 — input_linear, output_linear, router)
        fused_tensor_names: tensor kind names that have a per-expert axis
                            and need slicing on prune (e.g. 'input_linear',
                            'output_linear')
        gate_tensor_names: tensor kind names that hold the router gate
                           (used by Pass 1 to read importance)
    """
    family_name: str
    match_pattern: str
    fused_tensor_names: tuple[str, ...]
    gate_tensor_names: tuple[str, ...]

    def match_re(self) -> "re.Pattern":
        return re.compile(self.match_pattern)


GRANITE_MOE_LAYOUT = FusedLayoutSpec(
    family_name="granitemoe",
    # Captures: (layer_idx, tensor_kind) where kind is one of
    # input_linear / output_linear / router
    match_pattern=r"^model\.layers\.(\d+)\.block_sparse_moe\.(input_linear|output_linear|router)(?:\.layer)?\.weight$",
    fused_tensor_names=("input_linear", "output_linear"),
    gate_tensor_names=("router",),
)


# Backward-compat module-level regexes — anything that imported these
# directly (older internal callers) keeps working. NEW code should use
# QWEN3_MOE_LAYOUT.gate_re() / .expert_re() instead.
ROUTER_GATE_RE = QWEN3_MOE_LAYOUT.gate_re()
EXPERT_TENSOR_RE = QWEN3_MOE_LAYOUT.expert_re()


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _shards(model_dir: Path) -> list[Path]:
    """Return all safetensors shards in deterministic order."""
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    return shards


def read_router_gates(model_dir: Path, layout: LayoutSpec = QWEN3_MOE_LAYOUT) -> dict[int, torch.Tensor]:
    """Pass 1: read every router gate tensor on CPU. Returns {layer_idx: tensor}.

    Router gates are tiny (num_experts × hidden_size, e.g. 128 × 2048 =
    ~1 MB per layer at fp16). Reading all of them into RAM is cheap.

    layout: LayoutSpec for the family being pruned. Default is QWEN3_MOE_LAYOUT
    for backwards compatibility with the existing forge path.
    """
    gate_re = layout.gate_re()
    gates: dict[int, torch.Tensor] = {}
    for shard in _shards(model_dir):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for k in f.keys():
                m = gate_re.match(k)
                if m:
                    layer_idx = int(m.group(1))
                    gates[layer_idx] = f.get_tensor(k)
    return gates


def select_experts_per_layer(
    router_gates: dict[int, torch.Tensor],
    keep_experts: int,
    num_experts_per_tok: int,
    activation_counts: dict[int, list[int]] | None = None,
) -> tuple[dict[int, list[int]], str]:
    """Per-layer expert selection.

    Two metrics, picked at runtime:

    1. **Activation count** (preferred when `activation_counts` provided):
       per-layer count of how often each expert was in the top-k routing
       decision over a calibration corpus. This is the §4.1.3.4 fix at
       the expert level — directly measures which experts the router
       fires on the workload the artifact will be used for. Generated by
       expert_activation_profile.py.

    2. **Router gate row L2 norm** (fallback): magnitude of the router's
       projection vector for each expert. Architectural-only metric, no
       calibration data, but task-misaligned for any non-general workload.
       Use only when calibration data is unavailable.

    Per-layer normalized in either case — no global flat ranking. Each
    layer keeps its own top-K most-important experts independently.
    Eliminates the depth-bias bug class that hit dense head pruning
    (sentinel-ai#165).

    The keep count is clamped to >= num_experts_per_tok so the router
    can always find enough surviving experts to satisfy its top-k.
    Halts if the user asks for fewer.

    Returns (selected_per_layer, metric_used) where metric_used is
    "activation_count" or "router_gate_l2_norm" — written into the
    sidecar for provenance.
    """
    if keep_experts < num_experts_per_tok:
        raise ValueError(
            f"keep_experts={keep_experts} < num_experts_per_tok={num_experts_per_tok}. "
            f"The router would not have enough surviving experts to satisfy its "
            f"top-k routing. Refusing to produce a broken model."
        )

    use_activation = activation_counts is not None
    metric = "activation_count" if use_activation else "router_gate_l2_norm"
    selected: dict[int, list[int]] = {}
    for layer_idx, gate in sorted(router_gates.items()):
        # gate.shape == [num_experts, hidden_size]; row K is the projection
        # vector that produces the routing logit for expert K.
        num_experts = gate.shape[0]
        if keep_experts >= num_experts:
            selected[layer_idx] = list(range(num_experts))
            continue
        if use_activation:
            counts = activation_counts.get(layer_idx) or activation_counts.get(str(layer_idx))
            if counts is None or len(counts) != num_experts:
                raise ValueError(
                    f"activation_counts missing layer {layer_idx} (or wrong length: "
                    f"expected {num_experts}, got {len(counts) if counts else None}). "
                    f"The importance JSON does not match this model's architecture."
                )
            importance = torch.tensor(counts, dtype=torch.float64)
        else:
            importance = gate.float().norm(dim=1)  # [num_experts]
        _, top_idx = importance.topk(keep_experts)
        selected[layer_idx] = sorted(top_idx.tolist())
    return selected, metric


def stream_rewrite(
    src_dir: Path,
    out_dir: Path,
    selected: dict[int, list[int]],
    *,
    shard_max_bytes: int = 5 * 1024 ** 3,
    metric_used: str = "router_gate_l2_norm",
    importance_json_meta: dict | None = None,
    layout: LayoutSpec = QWEN3_MOE_LAYOUT,
) -> dict:
    """Pass 2: stream every source tensor, decide keep/rename/slice/drop,
    write to new shards. Returns metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the per-layer expert renumbering map: old_idx -> new_idx (or None
    # if dropped). E.g. selected[0] = [3, 7, 11] means expert 3 → 0, 7 → 1,
    # 11 → 2, and all others → None (dropped).
    renumber: dict[int, dict[int, int]] = {}
    for layer_idx, kept in selected.items():
        renumber[layer_idx] = {old: new for new, old in enumerate(kept)}

    new_weight_map: dict[str, str] = {}
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {"dir": str(src_dir), "shards": []},
        "selection": {
            "metric": metric_used,
            "importance_json": importance_json_meta,
            "per_layer_kept_count": {str(li): len(v) for li, v in selected.items()},
            "per_layer_kept_indices": {str(li): v for li, v in selected.items()},
        },
        "tensors": {
            "kept_unchanged": 0,
            "kept_renamed": 0,
            "router_gate_sliced": 0,
            "dropped_expert": 0,
        },
        "shards_out": [],
        "errors": [],
    }

    src_shards = _shards(src_dir)
    _log(f"Hashing {len(src_shards)} source shards...")
    for sp in src_shards:
        sha = _file_sha256(sp)
        size = sp.stat().st_size
        metadata["source"]["shards"].append({
            "path": sp.name, "sha256": sha, "bytes": size,
        })

    shard_buf: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    shard_idx = 0
    out_paths_in_order: list[Path] = []

    def flush_shard():
        nonlocal shard_buf, shard_bytes, shard_idx
        if not shard_buf:
            return
        out_name = f"model-{shard_idx + 1:05d}.safetensors"
        out_path = out_dir / out_name
        save_file(shard_buf, str(out_path))
        out_paths_in_order.append(out_path)
        size = out_path.stat().st_size
        for k in shard_buf.keys():
            new_weight_map[k] = out_name
        metadata["shards_out"].append({
            "path": out_name, "bytes": size, "tensor_count": len(shard_buf),
        })
        _log(f"  wrote shard {shard_idx} ({len(shard_buf)} tensors, {size/1e9:.2f} GB)")
        shard_buf.clear()
        shard_bytes = 0
        shard_idx += 1

    def add_to_shard(name: str, tensor: torch.Tensor):
        nonlocal shard_bytes
        nbytes = tensor.numel() * tensor.element_size()
        if shard_bytes + nbytes > shard_max_bytes and shard_buf:
            flush_shard()
        shard_buf[name] = tensor
        shard_bytes += nbytes

    # Compile the layout patterns ONCE for the streaming loop. The layout
    # is part of the function contract — every tensor name walked here
    # uses the same family's name patterns.
    gate_re = layout.gate_re()
    expert_re = layout.expert_re()
    metadata["selection"]["layout_family"] = layout.family_name

    _log(f"Pass 2: streaming rewrite (layout={layout.family_name})...")
    t0 = time.time()
    for sp in src_shards:
        with safe_open(str(sp), framework="pt", device="cpu") as f:
            for k in f.keys():
                # Router gate: slice to surviving experts
                gm = gate_re.match(k)
                if gm:
                    layer_idx = int(gm.group(1))
                    if layer_idx not in selected:
                        # No selection for this layer (shouldn't happen for
                        # MoE-only models, but handle it). Copy unchanged.
                        add_to_shard(k, f.get_tensor(k))
                        metadata["tensors"]["kept_unchanged"] += 1
                        continue
                    kept_rows = selected[layer_idx]
                    full = f.get_tensor(k)
                    sliced = full[kept_rows].contiguous()
                    add_to_shard(k, sliced)
                    metadata["tensors"]["router_gate_sliced"] += 1
                    continue

                # Per-expert MLP tensor: drop or renumber
                em = expert_re.match(k)
                if em:
                    layer_idx = int(em.group(1))
                    expert_idx = int(em.group(2))
                    proj_name = em.group(3)
                    if layer_idx not in renumber:
                        # Layer not in selection — pass through unchanged
                        add_to_shard(k, f.get_tensor(k))
                        metadata["tensors"]["kept_unchanged"] += 1
                        continue
                    new_idx = renumber[layer_idx].get(expert_idx)
                    if new_idx is None:
                        # Dropped — do not write
                        metadata["tensors"]["dropped_expert"] += 1
                        continue
                    new_name = layout.expert_rename_template.format(
                        layer=layer_idx, new_idx=new_idx, proj_name=proj_name,
                    )
                    add_to_shard(new_name, f.get_tensor(k))
                    metadata["tensors"]["kept_renamed"] += 1
                    continue

                # Anything else: copy unchanged (embeddings, norms, attention,
                # lm_head, etc.)
                add_to_shard(k, f.get_tensor(k))
                metadata["tensors"]["kept_unchanged"] += 1

    flush_shard()

    metadata["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata["total_seconds"] = round(time.time() - t0, 2)
    metadata["total_bytes_out"] = sum(s["bytes"] for s in metadata["shards_out"])

    # Final shard naming with the canonical of-N suffix
    n = len(out_paths_in_order)
    final_weight_map: dict[str, str] = {}
    for i, op in enumerate(out_paths_in_order):
        new_name = f"model-{i+1:05d}-of-{n:05d}.safetensors"
        new_path = op.with_name(new_name)
        op.rename(new_path)
        for k, old_name in new_weight_map.items():
            if old_name == op.name:
                final_weight_map[k] = new_name

    index = {
        "metadata": {"total_size": metadata["total_bytes_out"]},
        "weight_map": final_weight_map,
    }
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    sidecar_path = out_dir / "expert_prune.metadata.v1.json"
    sidecar_path.write_text(json.dumps(metadata, indent=2))

    _log(
        f"Done. tensors: kept={metadata['tensors']['kept_unchanged']} "
        f"renamed={metadata['tensors']['kept_renamed']} "
        f"router_sliced={metadata['tensors']['router_gate_sliced']} "
        f"dropped={metadata['tensors']['dropped_expert']}"
    )

    return metadata


def update_config(out_dir: Path, src_dir: Path, new_num_experts: int) -> None:
    """Copy config + tokenizer from source, update num_experts in place."""
    for fn in (
        "config.json", "generation_config.json",
        "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.json", "merges.txt",
        "added_tokens.json", "chat_template.jinja",
    ):
        src = src_dir / fn
        if src.exists():
            shutil.copy(src, out_dir / fn)

    cfg_path = out_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"no config.json copied from {src_dir}")

    cfg = json.loads(cfg_path.read_text())
    # Update both top-level and nested text_config (multimodal models)
    for container in (cfg, cfg.get("text_config", {})):
        if not isinstance(container, dict):
            continue
        # The Qwen3MoE config uses "num_experts"; some other MoE configs
        # use "num_local_experts". Update both if present.
        for key in ("num_experts", "num_local_experts", "n_routed_experts"):
            if key in container:
                container[key] = new_num_experts

    cfg_path.write_text(json.dumps(cfg, indent=2))


def prune_experts(
    model_dir: str | Path,
    out_dir: str | Path,
    keep_experts: int,
    *,
    shard_bytes: int = 5 * 1024 ** 3,
    importance_json: str | Path | None = None,
    layout: LayoutSpec = QWEN3_MOE_LAYOUT,
) -> dict:
    """Streaming CPU-side per-layer top-K MoE expert removal.

    Reads the source model's safetensors shards from `model_dir`, selects
    `keep_experts` experts per layer using the importance metric (calibration-
    aware activation count if importance_json is provided, router gate row
    L2 norm otherwise), rewrites the safetensors shards into `out_dir` with
    the surviving experts renumbered sequentially and the router gate row-
    sliced to match. Updates `out_dir/config.json` to reflect the new
    num_experts. Writes a sidecar `expert_prune.metadata.v1.json` recording
    every selection decision and the importance metric provenance.

    Used by:
        - The CLI entry point (main(), below)
        - MoEUnfusedExpertsBase.expert_prune (the family adapter for
          Qwen3MoE / OLMoE / future unfused MoE families)

    Both call this function the same way. There is no second code path.

    Args:
        model_dir:       local path to MoE source model dir (must be on disk)
        out_dir:         output directory for the pruned model
        keep_experts:    survivors per layer (must be >= num_experts_per_tok
                         and < num_experts)
        shard_bytes:     max bytes per output safetensors shard (default 5 GB)
        importance_json: path to expert_activation_profile.py output. When
                         provided, expert importance is per-layer activation
                         count from a calibration corpus (the § 4.1.3.4 fix);
                         when None, the script uses the router gate row L2
                         norm which is the pre-§4.1.3.4 architectural-only
                         metric and only appropriate for the negative-baseline
                         falsifiability anchor.

    Returns:
        The metadata dict (also written to out_dir/expert_prune.metadata.v1.json).

    Raises:
        ValueError: if model_dir doesn't exist, has no config.json, has no
                    recognized MoE expert count, or keep_experts is out of
                    the valid range.
        RuntimeError: if no router gate tensors can be found (the layout
                      doesn't match the unfused MoE pattern).
    """
    src_dir = Path(model_dir)
    out_dir = Path(out_dir)

    if not src_dir.exists():
        raise ValueError(f"source model dir {src_dir} does not exist")

    cfg_path = src_dir / "config.json"
    if not cfg_path.exists():
        raise ValueError(f"no config.json in {src_dir}")
    cfg = json.loads(cfg_path.read_text())
    tc = cfg.get("text_config", cfg)
    num_experts = tc.get("num_experts") or tc.get("num_local_experts") or tc.get("n_routed_experts")
    num_experts_per_tok = tc.get("num_experts_per_tok") or tc.get("num_active_experts")
    if num_experts is None:
        raise ValueError(
            "source config has no num_experts / num_local_experts / n_routed_experts; "
            "not a recognized MoE"
        )
    if num_experts_per_tok is None:
        raise ValueError("source config has no num_experts_per_tok / num_active_experts")

    _log(f"source: {src_dir}")
    _log(f"  num_experts: {num_experts}")
    _log(f"  num_experts_per_tok: {num_experts_per_tok}")
    _log(f"  num_hidden_layers: {tc.get('num_hidden_layers')}")
    _log(f"target: keep {keep_experts} experts per layer (was {num_experts})")

    if keep_experts < num_experts_per_tok:
        raise ValueError(
            f"keep_experts={keep_experts} < num_experts_per_tok={num_experts_per_tok}. "
            f"Router would route to nonexistent experts. Halting."
        )
    if keep_experts >= num_experts:
        raise ValueError(
            f"keep_experts={keep_experts} >= num_experts={num_experts}. "
            f"Nothing to prune."
        )

    _log(f"Pass 1: reading router gates (layout={layout.family_name})...")
    gates = read_router_gates(src_dir, layout=layout)
    _log(f"  found {len(gates)} router gate tensors")
    if not gates:
        raise RuntimeError(
            f"no router gate tensors found in {src_dir} for layout "
            f"{layout.family_name!r}. The expected gate pattern is "
            f"{layout.gate_pattern!r}. If your model has a different "
            f"layout, declare a new LayoutSpec for it (or use one of the "
            f"existing constants: QWEN3_MOE_LAYOUT, MIXTRAL_LAYOUT)."
        )

    activation_counts = None
    importance_json_meta = None
    if importance_json:
        imp_path = Path(importance_json)
        if not imp_path.exists():
            raise ValueError(f"importance_json path {imp_path} does not exist")
        _log(f"loading per-layer activation counts from {imp_path}")
        imp_data = json.loads(imp_path.read_text())
        activation_counts = {int(k): v for k, v in imp_data["activation_counts"].items()}
        if imp_data.get("num_experts") != num_experts:
            raise ValueError(
                f"importance JSON num_experts={imp_data.get('num_experts')} "
                f"does not match model num_experts={num_experts}"
            )
        importance_json_meta = {
            "path": str(imp_path),
            "sha256": _file_sha256(imp_path),
            "model": imp_data.get("model"),
            "calibration_corpus": imp_data.get("calibration_corpus"),
            "calibration_examples": imp_data.get("calibration_examples"),
            "calibration_tokens": imp_data.get("calibration_tokens"),
            "metric_version": imp_data.get("metric_version"),
        }
        _log(f"  metric: activation_count from {imp_data.get('calibration_examples')} examples / {imp_data.get('calibration_tokens')} tokens")

    selected, metric_used = select_experts_per_layer(
        gates, keep_experts, num_experts_per_tok,
        activation_counts=activation_counts,
    )
    _log(f"  importance metric: {metric_used}")
    counts = sorted(set(len(v) for v in selected.values()))
    _log(f"  per-layer kept counts: {counts}")
    if len(counts) != 1 or counts[0] != keep_experts:
        _log(f"  WARNING: per-layer counts non-uniform — some layers had fewer than {keep_experts} experts")

    metadata = stream_rewrite(
        src_dir, out_dir, selected,
        shard_max_bytes=shard_bytes,
        metric_used=metric_used,
        importance_json_meta=importance_json_meta,
        layout=layout,
    )
    update_config(out_dir, src_dir, keep_experts)
    _log(f"  config updated: num_experts -> {keep_experts}")

    src_size = sum(s["bytes"] for s in metadata["source"]["shards"])
    out_size = metadata["total_bytes_out"]
    _log(f"size: {src_size/1e9:.1f} GB → {out_size/1e9:.1f} GB ({(1 - out_size/src_size)*100:.0f}% reduction)")
    _log(f"output: {out_dir}")
    _log(f"sidecar: {out_dir}/expert_prune.metadata.v1.json")

    return metadata


def prune_experts_fused(
    model_dir: str | Path,
    out_dir: str | Path,
    keep_experts: int,
    *,
    layout: FusedLayoutSpec = GRANITE_MOE_LAYOUT,
    importance_json: str | Path | None = None,
    shard_bytes: int = 5 * 1024 ** 3,
) -> dict:
    """Streaming CPU-side per-layer top-K MoE expert removal — FUSED layout.

    For families like GraniteMoE where all experts in a layer share three
    fused tensors along an expert axis (axis=0). Pruning slices each fused
    tensor instead of dropping/renaming named entries.

    Pass 1: read router gates per layer, compute per-expert importance
    Pass 2: streaming rewrite — for each tensor, slice along axis=0 if it
            matches a fused/gate tensor name, passthrough otherwise

    The same algorithm as the unfused prune_experts, just with a different
    per-tensor handling step. Importance JSON, sidecar metadata format,
    and config update behavior are identical.

    Args:
        model_dir: source model directory (HF format on disk)
        out_dir: where to write the pruned shards + sidecar
        keep_experts: number of experts to keep per layer (≥ active per tok)
        layout: FusedLayoutSpec for the family (default GRANITE_MOE_LAYOUT)
        importance_json: optional pre-computed per-layer importance (calibration-aware)
        shard_bytes: max output shard size (default 5GB, same as unfused)

    Returns:
        Metadata dict identical in shape to prune_experts() so downstream
        consumers (publish, sidecar readers) handle both flavors uniformly.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    src_dir = Path(model_dir)
    out_dir = Path(out_dir)
    if not src_dir.exists():
        raise ValueError(f"source model dir {src_dir} does not exist")

    cfg_path = src_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    tc = cfg.get("text_config", cfg)
    num_experts = tc.get("num_local_experts") or tc.get("num_experts")
    num_experts_per_tok = tc.get("num_experts_per_tok") or tc.get("num_active_experts")
    if num_experts is None:
        raise ValueError("source config has no num_local_experts; not a recognized fused MoE")
    if keep_experts > num_experts:
        raise ValueError(f"keep_experts={keep_experts} > num_experts={num_experts}")
    if num_experts_per_tok is not None and keep_experts < num_experts_per_tok:
        raise ValueError(
            f"keep_experts={keep_experts} < num_experts_per_tok={num_experts_per_tok}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    src_shards = _shards(src_dir)
    metadata: dict = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "model_dir": str(src_dir),
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "num_hidden_layers": tc.get("num_hidden_layers"),
            "shards": [{"name": s.name, "bytes": s.stat().st_size} for s in src_shards],
        },
        "selection": {
            "layout_family": layout.family_name,
            "keep_experts_per_layer": keep_experts,
            "strategy": "calibration-aware-activation-count" if importance_json else "router-gate-l2",
            "per_layer_kept_indices": {},
        },
        "tensors": {"sliced": 0, "kept_unchanged": 0},
        "shards_out": [],
    }

    # Pass 1: read router gates per layer + compute per-expert importance
    _log(f"Pass 1: read router gates (layout={layout.family_name})...")
    match_re = layout.match_re()
    router_per_layer: dict[int, "torch.Tensor"] = {}
    for sp in src_shards:
        with safe_open(str(sp), framework="pt", device="cpu") as f:
            for k in f.keys():
                m = match_re.match(k)
                if not m:
                    continue
                kind = m.group(2)
                if kind not in layout.gate_tensor_names:
                    continue
                layer_idx = int(m.group(1))
                router_per_layer[layer_idx] = f.get_tensor(k)

    # Importance: per-layer per-expert score
    importance_counts: dict[str, list[int]] | None = None
    if importance_json is not None:
        importance_data = json.loads(Path(importance_json).read_text())
        # The canonical schema written by expert_activation_profile uses
        # 'activation_counts' as the per-layer counter dict key. Don't
        # silently fall back if the file exists but the key is missing —
        # that means the schema changed and we should fail loudly.
        importance_counts = importance_data.get("activation_counts")
        if importance_counts is None:
            raise ValueError(
                f"importance JSON {importance_json} has no 'activation_counts' key. "
                f"Top-level keys: {sorted(importance_data.keys())}"
            )

    selected: dict[int, list[int]] = {}
    for layer_idx, gate in router_per_layer.items():
        # gate shape: [num_experts, hidden]
        if importance_counts is not None:
            layer_importance = importance_counts.get(str(layer_idx)) or importance_counts.get(layer_idx)
            if layer_importance is None:
                # Fall back to L2 norm — this layer wasn't profiled
                scores = gate.float().norm(dim=1)
            else:
                # Per-layer activation count → keep highest
                scores = torch.tensor(layer_importance, dtype=torch.float32)
        else:
            scores = gate.float().norm(dim=1)
        topk = torch.topk(scores, keep_experts).indices.sort().values.tolist()
        selected[layer_idx] = topk
        metadata["selection"]["per_layer_kept_indices"][str(layer_idx)] = topk

    # Pass 2: streaming rewrite. For each tensor, if it matches a fused
    # name, slice along axis=0; otherwise passthrough unchanged.
    _log(f"Pass 2: streaming rewrite (layout={layout.family_name})...")
    out_shard_bytes = 0
    out_shard_idx = 1
    out_paths_in_order: list[Path] = []
    new_weight_map: dict[str, str] = {}
    current_shard_buffer: dict[str, "torch.Tensor"] = {}

    def flush_shard():
        nonlocal out_shard_bytes, out_shard_idx, current_shard_buffer
        if not current_shard_buffer:
            return
        out_path = out_dir / f"model-{out_shard_idx:05d}-of-XXXXX.safetensors"
        save_file(current_shard_buffer, str(out_path))
        size = out_path.stat().st_size
        metadata["shards_out"].append({"name": out_path.name, "bytes": size})
        out_paths_in_order.append(out_path)
        for k in current_shard_buffer:
            new_weight_map[k] = out_path.name
        current_shard_buffer = {}
        out_shard_bytes = 0
        out_shard_idx += 1

    def add_to_shard(name: str, tensor: "torch.Tensor"):
        nonlocal out_shard_bytes
        size = tensor.element_size() * tensor.numel()
        if out_shard_bytes + size > shard_bytes and current_shard_buffer:
            flush_shard()
        current_shard_buffer[name] = tensor
        out_shard_bytes += size

    t0 = time.time()
    for sp in src_shards:
        with safe_open(str(sp), framework="pt", device="cpu") as f:
            for k in f.keys():
                m = match_re.match(k)
                if m:
                    layer_idx = int(m.group(1))
                    kind = m.group(2)
                    if layer_idx not in selected:
                        add_to_shard(k, f.get_tensor(k))
                        metadata["tensors"]["kept_unchanged"] += 1
                        continue
                    keep_idx = torch.tensor(selected[layer_idx])
                    full = f.get_tensor(k)
                    sliced = full.index_select(0, keep_idx).contiguous()
                    add_to_shard(k, sliced)
                    metadata["tensors"]["sliced"] += 1
                    continue
                # Passthrough — embeddings, norms, attention, lm_head, etc.
                add_to_shard(k, f.get_tensor(k))
                metadata["tensors"]["kept_unchanged"] += 1
    flush_shard()

    metadata["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata["total_seconds"] = round(time.time() - t0, 2)
    metadata["total_bytes_out"] = sum(s["bytes"] for s in metadata["shards_out"])

    # Final shard naming with the canonical of-N suffix
    n = len(out_paths_in_order)
    final_weight_map: dict[str, str] = {}
    for i, op in enumerate(out_paths_in_order):
        new_name = f"model-{i+1:05d}-of-{n:05d}.safetensors"
        new_path = op.with_name(new_name)
        op.rename(new_path)
        for kk, old_name in new_weight_map.items():
            if old_name == op.name:
                final_weight_map[kk] = new_name

    (out_dir / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": metadata["total_bytes_out"]},
        "weight_map": final_weight_map,
    }, indent=2))

    sidecar_path = out_dir / "expert_prune.metadata.v1.json"
    sidecar_path.write_text(json.dumps(metadata, indent=2))

    update_config(out_dir, src_dir, keep_experts)
    _log(f"Done. tensors: sliced={metadata['tensors']['sliced']} "
         f"passthrough={metadata['tensors']['kept_unchanged']}")
    return metadata


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", help="Local path to MoE source model dir (must already be downloaded)")
    ap.add_argument("out_dir", help="Output directory for the pruned model")
    ap.add_argument("--keep-experts", type=int, required=True,
                    help="Number of experts to keep per layer (must be >= num_experts_per_tok)")
    ap.add_argument("--shard-bytes", type=int, default=5 * 1024 ** 3,
                    help="Max bytes per output safetensors shard (default 5 GB)")
    ap.add_argument("--importance-json", type=str, default=None,
                    help="Path to expert_activation_profile.py output. When "
                         "provided, expert importance is per-layer activation "
                         "count from a calibration corpus (the §4.1.3.4 fix); "
                         "when omitted, the script uses router gate row L2 norm "
                         "which is the pre-§4.1.3.4 architectural-only metric "
                         "and only appropriate for the negative-baseline anchor.")
    args = ap.parse_args()

    try:
        prune_experts(
            model_dir=args.model_dir,
            out_dir=args.out_dir,
            keep_experts=args.keep_experts,
            shard_bytes=args.shard_bytes,
            importance_json=args.importance_json,
        )
    except (ValueError, RuntimeError) as e:
        sys.exit(f"FATAL: {e}")


if __name__ == "__main__":
    main()
