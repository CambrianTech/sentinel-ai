#!/usr/bin/env python3
"""
CPU expert pruning: reduce MoE experts to fit on consumer GPU.
Loads one shard at a time — never needs more than 4GB RAM.

Usage:
    python scripts/cpu_expert_prune.py continuum-ai/qwen3.5-35b-a3b-compacted --keep-experts 16
"""

import argparse
import json
import os
import gc
from pathlib import Path
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import snapshot_download

def analyze_experts(model_dir):
    """Scan safetensors to find expert structure."""
    experts_per_layer = defaultdict(set)
    expert_sizes = defaultdict(int)
    non_expert_size = 0
    
    for sf in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(sf), framework="pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                size = t.numel() * t.element_size()
                if ".experts." in k:
                    # Extract layer and expert index
                    parts = k.split(".")
                    layer_idx = int(parts[parts.index("layers") + 1])
                    expert_idx = int(parts[parts.index("experts") + 1])
                    experts_per_layer[layer_idx].add(expert_idx)
                    expert_sizes[(layer_idx, expert_idx)] += size
                else:
                    non_expert_size += size
    
    return dict(experts_per_layer), dict(expert_sizes), non_expert_size


def compute_expert_importance(model_dir, experts_per_layer):
    """Compute importance of each expert by weight norm."""
    importance = {}
    
    for sf in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(sf), framework="pt") as f:
            for k in f.keys():
                if ".experts." in k and ".weight" in k:
                    parts = k.split(".")
                    layer_idx = int(parts[parts.index("layers") + 1])
                    expert_idx = int(parts[parts.index("experts") + 1])
                    key = (layer_idx, expert_idx)
                    t = f.get_tensor(k).float()
                    norm = t.norm().item()
                    importance[key] = importance.get(key, 0) + norm
    
    return importance


def prune_experts(model_dir, output_dir, keep_n, experts_per_layer, importance):
    """Remove experts, save smaller model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # For each layer, keep top-N experts by importance
    keep_experts = {}
    for layer_idx, expert_set in experts_per_layer.items():
        layer_imps = [(importance.get((layer_idx, e), 0), e) for e in expert_set]
        layer_imps.sort(reverse=True)
        keep = set(e for _, e in layer_imps[:keep_n])
        removed = len(expert_set) - len(keep)
        if removed > 0:
            print(f"  Layer {layer_idx}: {len(expert_set)} → {len(keep)} experts (removed {removed})")
        keep_experts[layer_idx] = keep
    
    # Stream through shards, skip pruned experts
    total_before = 0
    total_after = 0
    shard_idx = 0
    shard = {}
    shard_sz = 0
    shard_max = 5 * 1024**3
    weight_map = {}
    
    for sf in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(sf), framework="pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                total_before += t.numel() * t.element_size()
                
                # Check if this is an expert to prune
                skip = False
                if ".experts." in k:
                    parts = k.split(".")
                    layer_idx = int(parts[parts.index("layers") + 1])
                    expert_idx = int(parts[parts.index("experts") + 1])
                    if expert_idx not in keep_experts.get(layer_idx, set()):
                        skip = True
                
                if not skip:
                    total_after += t.numel() * t.element_size()
                    sz = t.numel() * t.element_size()
                    
                    if shard_sz + sz > shard_max and shard:
                        fname = f"model-{shard_idx+1:05d}-of-TOTAL.safetensors"
                        save_file(shard, str(output_dir / fname))
                        print(f"  Shard {shard_idx}: {shard_sz/1e9:.1f}GB")
                        shard_idx += 1
                        shard = {}
                        shard_sz = 0
                        gc.collect()
                    
                    shard[k] = t
                    shard_sz += sz
                    weight_map[k] = f"model-{shard_idx+1:05d}-of-TOTAL.safetensors"
    
    if shard:
        fname = f"model-{shard_idx+1:05d}-of-TOTAL.safetensors"
        save_file(shard, str(output_dir / fname))
        print(f"  Shard {shard_idx}: {shard_sz/1e9:.1f}GB")
        shard_idx += 1
    
    # Rename with totals
    total = shard_idx
    for i in range(total):
        old = output_dir / f"model-{i+1:05d}-of-TOTAL.safetensors"
        new = output_dir / f"model-{i+1:05d}-of-{total:05d}.safetensors"
        old.rename(new)
        for k in weight_map:
            if weight_map[k] == f"model-{i+1:05d}-of-TOTAL.safetensors":
                weight_map[k] = new.name
    
    json.dump({"metadata": {"total_size": total_after}, "weight_map": weight_map},
              open(output_dir / "model.safetensors.index.json", "w"), indent=2)
    
    return total_before, total_after


def main():
    parser = argparse.ArgumentParser(description="CPU expert pruning for MoE models")
    parser.add_argument("model", help="HF model ID or local path")
    parser.add_argument("--keep-experts", type=int, default=16, help="Experts to keep per layer")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    # Download if needed
    if "/" in args.model and not os.path.exists(args.model):
        print(f"[1] Downloading {args.model}...")
        model_dir = snapshot_download(args.model)
    else:
        model_dir = args.model
    
    slug = args.model.split("/")[-1].lower()
    output_dir = args.output or f"output/pruned/{slug}-{args.keep_experts}experts"
    
    print(f"[2] Analyzing expert structure...")
    experts_per_layer, expert_sizes, non_expert_size = analyze_experts(model_dir)
    
    total_experts = sum(len(v) for v in experts_per_layer.values())
    layers_with_experts = len(experts_per_layer)
    print(f"  {layers_with_experts} MoE layers, {total_experts} total experts")
    print(f"  Non-expert size: {non_expert_size/1e9:.1f}GB")
    print(f"  Expert size: {sum(expert_sizes.values())/1e9:.1f}GB")
    
    if layers_with_experts == 0:
        print("  No MoE experts found — not an MoE model")
        return
    
    experts_per = total_experts // layers_with_experts
    print(f"  ~{experts_per} experts per layer, keeping {args.keep_experts}")
    
    print(f"\n[3] Computing expert importance...")
    importance = compute_expert_importance(model_dir, experts_per_layer)
    
    print(f"\n[4] Pruning to {args.keep_experts} experts per layer...")
    before, after = prune_experts(model_dir, output_dir, args.keep_experts, experts_per_layer, importance)
    
    saved = before - after
    print(f"\n  Before: {before/1e9:.1f}GB → After: {after/1e9:.1f}GB (saved {saved/1e9:.1f}GB)")
    
    # Copy config + tokenizer
    import shutil
    for f in ["config.json", "tokenizer.json", "tokenizer_config.json", 
              "generation_config.json", "chat_template.jinja"]:
        src = Path(model_dir) / f
        if src.exists():
            shutil.copy2(src, Path(output_dir) / f)
    
    # Update config with new expert count
    config_path = Path(output_dir) / "config.json"
    if config_path.exists():
        config = json.load(open(config_path))
        tc = config.get("text_config", config)
        if "num_local_experts" in tc:
            tc["num_local_experts"] = args.keep_experts
            json.dump(config, open(config_path, "w"), indent=2)
            print(f"  Config updated: num_local_experts={args.keep_experts}")
    
    print(f"\nDONE: {output_dir}")
    print(f"  Now forge: python scripts/forge_model.py {output_dir} --domain code --load-in-4bit")


if __name__ == "__main__":
    main()
