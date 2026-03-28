"""CPU expert pruning for fused MoE tensors. Slices expert dimension directly."""
import torch, gc, json, shutil, os, argparse
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="HF model ID or local path")
    parser.add_argument("--keep-experts", type=int, default=16)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if "/" in args.model and not os.path.exists(args.model):
        print(f"[1] Downloading {args.model}...")
        model_dir = snapshot_download(args.model)
    else:
        model_dir = args.model

    slug = args.model.split("/")[-1].lower()
    output_dir = Path(args.output or f"output/pruned/{slug}-{args.keep_experts}exp")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[2] Analyzing...")
    # Find expert tensor shapes
    expert_info = {}
    for sf in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(sf), framework="pt") as f:
            for k in f.keys():
                if "experts" in k:
                    t = f.get_tensor(k)
                    expert_info[k] = t.shape
                    if len(expert_info) <= 3:
                        print(f"  {k}: {t.shape}")
    
    if not expert_info:
        print("  No expert tensors found!")
        return

    sample_shape = list(expert_info.values())[0]
    current_experts = sample_shape[0]
    print(f"  {len(expert_info)} expert tensors, {current_experts} experts each")
    print(f"  Keeping {args.keep_experts} of {current_experts}")

    if args.keep_experts >= current_experts:
        print(f"  Already at {current_experts} experts, nothing to prune")
        return

    # Compute per-expert importance (norm across all expert tensors)
    print(f"\n[3] Computing expert importance...")
    importance = torch.zeros(current_experts)
    for sf in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(sf), framework="pt") as f:
            for k in f.keys():
                if "experts" in k:
                    t = f.get_tensor(k).float()
                    for e in range(min(current_experts, t.shape[0])):
                        importance[e] += t[e].norm().item()
    
    _, top_indices = importance.topk(args.keep_experts)
    keep = sorted(top_indices.tolist())
    print(f"  Keeping experts: {keep[:10]}{'...' if len(keep) > 10 else ''}")

    # Prune: stream through shards, slice expert tensors
    print(f"\n[4] Pruning...")
    shard, shard_sz, si, wm = {}, 0, 0, {}
    shard_max = 5 * 1024**3
    total_b, total_a = 0, 0

    for sf in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(sf), framework="pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                total_b += t.numel() * t.element_size()

                if "experts" in k and t.dim() >= 2 and t.shape[0] == current_experts:
                    t = t[keep].contiguous()

                total_a += t.numel() * t.element_size()
                sz = t.numel() * t.element_size()
                
                if shard_sz + sz > shard_max and shard:
                    save_file(shard, str(output_dir / f"model-{si+1:05d}-of-TOTAL.safetensors"))
                    print(f"  Shard {si}: {shard_sz/1e9:.1f}GB")
                    si += 1; shard = {}; shard_sz = 0; gc.collect()

                shard[k] = t; shard_sz += sz
                wm[k] = f"model-{si+1:05d}-of-TOTAL.safetensors"

    if shard:
        save_file(shard, str(output_dir / f"model-{si+1:05d}-of-TOTAL.safetensors"))
        print(f"  Shard {si}: {shard_sz/1e9:.1f}GB"); si += 1

    total = si
    for i in range(total):
        old = output_dir / f"model-{i+1:05d}-of-TOTAL.safetensors"
        new = output_dir / f"model-{i+1:05d}-of-{total:05d}.safetensors"
        old.rename(new)
        for k in wm:
            if wm[k] == f"model-{i+1:05d}-of-TOTAL.safetensors": wm[k] = new.name
    
    json.dump({"metadata": {"total_size": total_a}, "weight_map": wm},
              open(output_dir / "model.safetensors.index.json", "w"), indent=2)

    # Copy config + tokenizer, update expert count
    for f in os.listdir(model_dir):
        if not f.endswith(".safetensors") and f != "model.safetensors.index.json":
            src = Path(model_dir) / f
            if src.is_file():
                shutil.copy2(src, output_dir / f)

    config_path = output_dir / "config.json"
    if config_path.exists():
        config = json.load(open(config_path))
        tc = config.get("text_config", config)
        for key in ["num_local_experts", "num_experts"]:
            if key in tc:
                tc[key] = args.keep_experts
        json.dump(config, open(config_path, "w"), indent=2)

    saved = total_b - total_a
    print(f"\nBefore: {total_b/1e9:.1f}GB → After: {total_a/1e9:.1f}GB (saved {saved/1e9:.1f}GB)")
    print(f"DONE: {output_dir}")
    print(f"\nNext: python scripts/forge_model.py {output_dir} --domain code --load-in-4bit")

if __name__ == "__main__":
    main()
