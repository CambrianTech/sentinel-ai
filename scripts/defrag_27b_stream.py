"""Stream defrag: dequantize one layer at a time, defrag, save, free. Never OOM."""
import torch, sys, gc, json, shutil
from pathlib import Path
from safetensors.torch import save_file
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM

sys.path.insert(0, "scripts")
from forge_model import compute_head_importance, select_heads_to_prune, get_model_info

print("[1] Loading 4-bit on CUDA...")
model = AutoModelForCausalLM.from_pretrained(
    "output/forged/qwen3.5-27b/model", device_map="auto", low_cpu_mem_usage=True
)
info = get_model_info("Qwen/Qwen3.5-27B")

print("[2] Computing importance...")
importance = compute_head_importance(model, info)
heads, n_pruned = select_heads_to_prune(importance, 0.3)
print(f"  Pruning {n_pruned} heads across {len(heads)} layers")

# Get dims from one layer
tc = json.load(open("output/forged/qwen3.5-27b/model/config.json")).get("text_config", {})
num_heads = tc.get("num_attention_heads", 24)
num_kv = tc.get("num_key_value_heads", 4)
group_size = num_heads // num_kv

# Find actual dims
q_hd = kv_hd = o_hd = 0
for name, module in model.named_modules():
    if hasattr(module, "q_proj") and isinstance(module.q_proj, bnb.nn.Linear4bit):
        dq = bnb.functional.dequantize_4bit(module.q_proj.weight.data, module.q_proj.weight.quant_state, quant_type="nf4")
        q_hd = dq.shape[0] // num_heads
        dq_k = bnb.functional.dequantize_4bit(module.k_proj.weight.data, module.k_proj.weight.quant_state, quant_type="nf4")
        kv_hd = dq_k.shape[0] // num_kv
        dq_o = bnb.functional.dequantize_4bit(module.o_proj.weight.data, module.o_proj.weight.quant_state, quant_type="nf4")
        o_hd = dq_o.shape[1] // num_heads
        del dq, dq_k, dq_o
        break
print(f"  Q_hd={q_hd}, KV_hd={kv_hd}, O_hd={o_hd}")

print("[3] Stream dequant + defrag...")
dst = Path("output/forged/qwen3.5-27b/model-defragged")
dst.mkdir(parents=True, exist_ok=True)

# Copy config + tokenizer
config = json.load(open("output/forged/qwen3.5-27b/model/config.json"))
config.pop("quantization_config", None)
config["model_type"] = "qwen3_5"
json.dump(config, open(dst / "config.json", "w"), indent=2)
for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"]:
    src = Path("output/forged/qwen3.5-27b/model") / f
    if src.exists():
        shutil.copy2(src, dst / f)

shard_max = 5 * 1024**3
cur_shard = {}
cur_size = 0
shard_idx = 0
wm = {}
total_before = 0
total_after = 0

for name, module in model.named_modules():
    for pname, param in module.named_parameters(recurse=False):
        full = f"{name}.{pname}" if name else pname

        # Dequantize
        if isinstance(module, bnb.nn.Linear4bit) and pname == "weight":
            t = bnb.functional.dequantize_4bit(param.data, param.quant_state, quant_type="nf4")
            t = t.to(torch.bfloat16).cpu().contiguous()
        else:
            t = param.data.to(torch.bfloat16).cpu().contiguous()

        total_before += t.numel() * t.element_size()

        # Defrag if this is a self_attn tensor in a pruned layer
        li = None
        if "self_attn" in full and "layers." in full:
            li = int(full.split("layers.")[1].split(".")[0])

        if li is not None and li in heads:
            dead = heads[li]
            surv_kv, surv_q = [], []
            for kh in range(num_kv):
                grp = list(range(kh * group_size, (kh + 1) * group_size))
                if all(h in dead for h in grp):
                    pass
                else:
                    surv_kv.append(kh)
                    surv_q.extend(grp)

            if len(surv_q) < num_heads:
                if "q_proj.weight" in full:
                    idx = [i for h in surv_q for i in range(h * q_hd, (h + 1) * q_hd)]
                    t = t[idx, :].contiguous()
                elif "k_proj.weight" in full or "v_proj.weight" in full:
                    idx = [i for h in surv_kv for i in range(h * kv_hd, (h + 1) * kv_hd)]
                    t = t[idx, :].contiguous()
                elif "o_proj.weight" in full:
                    idx = [i for h in surv_q for i in range(h * o_hd, (h + 1) * o_hd)]
                    t = t[:, idx].contiguous()

        total_after += t.numel() * t.element_size()

        # Add to shard
        sz = t.numel() * t.element_size()
        if cur_size + sz > shard_max and cur_shard:
            fname = f"model-{shard_idx + 1:05d}-of-TOTAL.safetensors"
            save_file(cur_shard, str(dst / fname))
            print(f"  Shard {shard_idx}: {cur_size / 1e9:.1f}GB")
            shard_idx += 1
            cur_shard = {}
            cur_size = 0
            gc.collect()

        cur_shard[full] = t
        cur_size += sz
        wm[full] = f"model-{shard_idx + 1:05d}-of-TOTAL.safetensors"

if cur_shard:
    fname = f"model-{shard_idx + 1:05d}-of-TOTAL.safetensors"
    save_file(cur_shard, str(dst / fname))
    print(f"  Shard {shard_idx}: {cur_size / 1e9:.1f}GB")
    shard_idx += 1

# Rename shards
total = shard_idx
for i in range(total):
    old = dst / f"model-{i + 1:05d}-of-TOTAL.safetensors"
    new = dst / f"model-{i + 1:05d}-of-{total:05d}.safetensors"
    old.rename(new)
    for k in wm:
        if wm[k] == f"model-{i + 1:05d}-of-TOTAL.safetensors":
            wm[k] = new.name

json.dump({"metadata": {"total_size": total_after}, "weight_map": wm},
          open(dst / "model.safetensors.index.json", "w"), indent=2)

saved = total_before - total_after
print(f"\nBefore: {total_before / 1e9:.2f}GB")
print(f"After: {total_after / 1e9:.2f}GB")
print(f"Saved: {saved / 1e6:.0f}MB")
print(f"DONE: {dst}")
