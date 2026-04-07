"""
Streaming GGUF -> safetensors dequantizer (validation harness tool).

Reads each tensor from a GGUF file individually, dequantizes on GPU, writes
to a safetensors shard, frees, repeats. Never holds the full model in either
system RAM or VRAM at once.

This tool exists because:

1. The conventional `transformers.from_pretrained(..., gguf_file=...)` path
   stages all dequantized tensors in CPU memory before moving them to GPU,
   which OOMs on hosts with less system RAM than the dequantized model size.
   For a 14B model, that means hosts with <60 GB RAM cannot use it at all.
   This tool runs on hosts with 16 GB RAM and a single GPU large enough to
   hold the dequantized weights.

2. The validation harness in `tests/defrag_validation/` needs an evaluation-
   grade safetensors model recoverable from any GGUF artifact, including
   GGUFs whose source weights have been lost (the failure mode documented
   in PLASTICITY-COMPACTION §4.1.1 Failure 3).

3. The metadata sidecar emitted by this tool is a public artifact format
   (`stream_dequant.metadata.v1.json`) that downstream forge-alloy stages
   can consume as cryptographically-chained provenance for the dequantized
   artifact. The dequantization step is itself an alloy stage; this tool
   emits the inputs and outputs that stage requires.

Usage:
    python -m scripts.stream_dequant <gguf_path> <out_dir> \\
        [--config-dir <hf_dir>] \\
        [--shard-bytes <int>] \\
        [--device cuda|cpu]

The output directory will contain:
    model-NNNNN.safetensors             # one or more weight shards
    model.safetensors.index.json        # standard HF shard index
    stream_dequant.metadata.v1.json     # provenance + per-tensor metadata
    config.json, tokenizer.json, ...    # copied from --config-dir if given
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import gguf
import torch
from gguf.quants import dequantize
from safetensors.torch import save_file
from safetensors import safe_open


# ────────────────────────────────────────────────────────────────────────────
# Tool identity (recorded in the metadata sidecar)
# ────────────────────────────────────────────────────────────────────────────

TOOL_NAME = "sentinel-ai/scripts/stream_dequant"
TOOL_VERSION = "1.0.0"
SCHEMA_VERSION = "stream_dequant.metadata.v1"


# ────────────────────────────────────────────────────────────────────────────
# GGUF -> HF tensor name mapping (Qwen2 / Llama family)
#
# Verified against llama.cpp's gguf-py and the Qwen2 model card. Qwen2 has
# Q/K/V biases (Llama does not); both share the rest of the layout. If you
# add a new architecture, the only change should be in this dict.
# ────────────────────────────────────────────────────────────────────────────

_QWEN2_LIKE = {
    "token_embd.weight":   "model.embed_tokens.weight",
    "output.weight":       "lm_head.weight",
    "output_norm.weight":  "model.norm.weight",
}

_QWEN2_LIKE_PER_LAYER = {
    "attn_norm.weight":    "input_layernorm.weight",
    "attn_q.weight":       "self_attn.q_proj.weight",
    "attn_q.bias":         "self_attn.q_proj.bias",
    "attn_k.weight":       "self_attn.k_proj.weight",
    "attn_k.bias":         "self_attn.k_proj.bias",
    "attn_v.weight":       "self_attn.v_proj.weight",
    "attn_v.bias":         "self_attn.v_proj.bias",
    "attn_output.weight":  "self_attn.o_proj.weight",
    "ffn_norm.weight":     "post_attention_layernorm.weight",
    "ffn_gate.weight":     "mlp.gate_proj.weight",
    "ffn_up.weight":       "mlp.up_proj.weight",
    "ffn_down.weight":     "mlp.down_proj.weight",
}


def gguf_to_hf_name(name: str, arch: str) -> Optional[str]:
    """Map a GGUF tensor name to its HF state_dict key. Returns None if no
    mapping is known (caller should record as 'unmapped' in the sidecar)."""
    if arch in ("qwen2", "llama"):
        if name in _QWEN2_LIKE:
            return _QWEN2_LIKE[name]
        if name.startswith("blk."):
            parts = name.split(".", 2)
            if len(parts) == 3:
                _, idx, rest = parts
                if rest in _QWEN2_LIKE_PER_LAYER:
                    return f"model.layers.{idx}.{_QWEN2_LIKE_PER_LAYER[rest]}"
        return None
    return None


# ────────────────────────────────────────────────────────────────────────────
# Streaming dequant
# ────────────────────────────────────────────────────────────────────────────


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _dequant_to_torch(reader_tensor, device: str) -> torch.Tensor:
    """Dequantize one GGUF tensor to fp16 on the requested device."""
    arr = dequantize(reader_tensor.data, reader_tensor.tensor_type)
    # `dequantize` returns a correctly-shaped float32 numpy array
    return torch.from_numpy(arr.copy()).to(device=device, dtype=torch.float16)


def stream_dequant(
    gguf_path: str,
    out_dir: str,
    *,
    config_dir: Optional[str] = None,
    shard_bytes: int = 5 * 1024 ** 3,
    device: str = "cuda",
    log: bool = True,
) -> dict:
    """Stream a GGUF file into safetensors shards, emitting a v1 metadata
    sidecar. Returns the in-memory metadata dict."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if log:
        print(f"[{_ts()}] Hashing source GGUF...", flush=True)
    src_sha256 = _file_sha256(gguf_path)
    src_size = os.path.getsize(gguf_path)
    if log:
        print(f"[{_ts()}] sha256={src_sha256[:16]}... bytes={src_size}", flush=True)
        print(f"[{_ts()}] Opening GGUF: {gguf_path}", flush=True)

    reader = gguf.GGUFReader(gguf_path)
    arch = bytes(reader.fields["general.architecture"].parts[-1]).decode()
    if log:
        print(f"[{_ts()}] Architecture: {arch}, {len(reader.tensors)} tensors", flush=True)

    metadata: dict = {
        "schema_version": SCHEMA_VERSION,
        "tool": {
            "name": TOOL_NAME,
            "version": TOOL_VERSION,
        },
        "source": {
            "gguf_path": str(gguf_path),
            "gguf_sha256": src_sha256,
            "gguf_bytes": src_size,
            "architecture": arch,
            "num_tensors": len(reader.tensors),
        },
        "provenance": {
            "device": device,
            "output_dtype": "float16",
            "shard_bytes": shard_bytes,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "shards": [],
        "tensors": [],
        "unmapped": [],
        "errors": [],
    }

    # Capture the original GGUF block layout per tensor before we touch it,
    # so a downstream tool could re-quantize losslessly back to the same
    # block structure if needed. (gguf.ReaderTensor exposes its block size
    # implicitly via the tensor_type; we record the type and the original
    # GGUF shape so the layout can be reconstructed deterministically.)
    metadata["gguf_block_layout"] = [
        {
            "name": rt.name,
            "type": str(rt.tensor_type).replace("GGMLQuantizationType.", ""),
            "type_id": int(rt.tensor_type),
            "gguf_shape": [int(d) for d in rt.shape],
        }
        for rt in reader.tensors
    ]

    shard_idx = 0
    shard: dict[str, torch.Tensor] = {}
    shard_bytes_so_far = 0
    shard_to_keys: dict[str, list[str]] = {}

    def flush_shard():
        nonlocal shard, shard_bytes_so_far, shard_idx
        if not shard:
            return
        shard_path = out / f"model-{shard_idx:05d}.safetensors"
        cpu_shard = {k: v.cpu() for k, v in shard.items()}
        save_file(cpu_shard, str(shard_path))
        size = shard_path.stat().st_size
        metadata["shards"].append({
            "path": shard_path.name,
            "bytes": size,
            "tensor_count": len(shard),
        })
        shard_to_keys[shard_path.name] = list(shard.keys())
        if log:
            print(f"[{_ts()}] Wrote shard {shard_idx} ({len(shard)} tensors, {size/1e9:.2f} GB)", flush=True)
        shard.clear()
        shard_bytes_so_far = 0
        shard_idx += 1
        if device == "cuda":
            torch.cuda.empty_cache()

    t0 = time.time()
    for i, rt in enumerate(reader.tensors):
        ts0 = time.time()
        try:
            t = _dequant_to_torch(rt, device=device)
        except Exception as e:
            if log:
                print(f"[{_ts()}] FAILED {rt.name} ({rt.tensor_type}): {e}", flush=True)
            metadata["errors"].append({"gguf_name": rt.name, "error": str(e)})
            continue

        hf_name = gguf_to_hf_name(rt.name, arch)
        if hf_name is None:
            metadata["unmapped"].append(rt.name)
            if log:
                print(f"[{_ts()}] UNMAPPED {rt.name}", flush=True)
            continue

        bytes_dq = t.numel() * t.element_size()
        l2 = float(t.float().norm().item())
        dt = time.time() - ts0

        shard[hf_name] = t
        shard_bytes_so_far += bytes_dq

        metadata["tensors"].append({
            "gguf_name": rt.name,
            "hf_name": hf_name,
            "quant_type": str(rt.tensor_type).replace("GGMLQuantizationType.", ""),
            "shape": list(t.shape),
            "dtype": "float16",
            "bytes": bytes_dq,
            "l2_norm_fp16": l2,
            "dequant_seconds": round(dt, 4),
        })

        if log and (i % 25 == 0):
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(reader.tensors) - i - 1)
            print(f"[{_ts()}] {i+1}/{len(reader.tensors)} {rt.name} -> {hf_name} {list(t.shape)} eta={eta:.0f}s", flush=True)

        if shard_bytes_so_far >= shard_bytes:
            flush_shard()

    flush_shard()

    metadata["provenance"]["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata["provenance"]["total_seconds"] = round(time.time() - t0, 2)
    metadata["total_bytes"] = sum(s["bytes"] for s in metadata["shards"])

    # Build the safetensors weight_map for the standard HF shard index.
    weight_map: dict[str, str] = {}
    for shard_name, keys in shard_to_keys.items():
        for k in keys:
            weight_map[k] = shard_name
    index = {
        "metadata": {"total_size": metadata["total_bytes"]},
        "weight_map": weight_map,
    }
    with open(out / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    sidecar_path = out / "stream_dequant.metadata.v1.json"
    with open(sidecar_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if log:
        print(
            f"[{_ts()}] Done. {len(metadata['tensors'])} mapped, "
            f"{len(metadata['unmapped'])} unmapped, {len(metadata['errors'])} errors, "
            f"{len(metadata['shards'])} shards, {metadata['provenance']['total_seconds']:.0f}s",
            flush=True,
        )
        if metadata["unmapped"]:
            print(f"  unmapped: {metadata['unmapped']}", flush=True)
        if metadata["errors"]:
            print(f"  errors: {metadata['errors']}", flush=True)

    if config_dir:
        import shutil
        for fn in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
            "vocab.json",
            "merges.txt",
        ):
            src = Path(config_dir) / fn
            if src.exists():
                shutil.copy(src, out / fn)
                if log:
                    print(f"  copied {fn}", flush=True)

    return metadata


def main():
    ap = argparse.ArgumentParser(
        description="Streaming GGUF -> safetensors dequantizer (validation harness tool)",
    )
    ap.add_argument("gguf_path", help="Path to source GGUF file")
    ap.add_argument("out_dir", help="Output directory for safetensors + sidecar")
    ap.add_argument(
        "--config-dir",
        default=None,
        help="HF model dir to copy config.json/tokenizer.json from",
    )
    ap.add_argument(
        "--shard-bytes",
        type=int,
        default=5 * 1024 ** 3,
        help="Max bytes per safetensors shard (default 5 GB)",
    )
    ap.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="Device to dequantize on (default cuda)",
    )
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: --device cuda requested but torch.cuda.is_available() is False", file=sys.stderr)
        sys.exit(2)

    stream_dequant(
        gguf_path=args.gguf_path,
        out_dir=args.out_dir,
        config_dir=args.config_dir,
        shard_bytes=args.shard_bytes,
        device=args.device,
    )


if __name__ == "__main__":
    main()
