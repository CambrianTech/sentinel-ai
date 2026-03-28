"""Uniform defrag on forged 27B: stream dequant + uniform GQA group removal."""
import torch, sys, gc, json, shutil
from pathlib import Path
from safetensors.torch import save_file
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "scripts")
from forge_model import compute_head_importance, get_model_info

SRC = "output/forged/qwen3.5-27b/model"
DST = Path("output/forged/qwen3.5-27b-defragged/model")

print("[1] Loading forged 27B in 4-bit on CUDA...")
model = AutoModelForCausalLM.from_pretrained(SRC, device_map="auto", low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(SRC)
info = get_model_info("Qwen/Qwen3.5-27B")

print("[2] Computing importance...")
importance = compute_head_importance(model, info)
num_heads, num_kv = info["num_heads"], info["num_kv_heads"]
group_size = num_heads // num_kv
attn_layers = [li for li in range(info["num_layers"]) if importance[li].min() < float("inf")]

# Remove 1 worst GQA group per layer (uniform)
dead_heads = {}
for li in attn_layers:
    groups = []
    for kv_h in range(num_kv):
        grp = list(range(kv_h * group_size, (kv_h + 1) * group_size))
        grp_imp = sum(importance[li, h].item() for h in grp)
        groups.append((grp_imp, kv_h, grp))
    groups.sort()
    dead_heads[li] = groups[0][2]

new_heads = num_heads - group_size
new_kv = num_kv - 1
print(f"  {len(attn_layers)} self_attn layers, {new_heads}Q/{new_kv}KV")

# Get dims from one dequantized layer
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

print("[3] Stream dequant + uniform defrag...")
DST.mkdir(parents=True, exist_ok=True)

shard, shard_sz, si, wm = {}, 0, 0, {}
shard_max = 5 * 1024**3
total_b, total_a = 0, 0

for name, module in model.named_modules():
    for pname, param in module.named_parameters(recurse=False):
        full = f"{name}.{pname}" if name else pname

        if isinstance(module, bnb.nn.Linear4bit) and pname == "weight":
            t = bnb.functional.dequantize_4bit(param.data, param.quant_state, quant_type="nf4")
            t = t.to(torch.bfloat16).cpu().contiguous()
        else:
            t = param.data.to(torch.bfloat16).cpu().contiguous()

        total_b += t.numel() * t.element_size()

        li = None
        if "self_attn" in full and "layers." in full:
            li = int(full.split("layers.")[1].split(".")[0])

        if li in dead_heads:
            dead = dead_heads[li]
            surv_q = sorted(set(range(num_heads)) - set(dead))
            surv_kv = sorted(set(range(num_kv)) - {dead[0] // group_size})

            if "q_proj.weight" in full:
                idx = [i for h in surv_q for i in range(h*q_hd, (h+1)*q_hd)]
                t = t[idx, :].contiguous()
            elif "k_proj.weight" in full or "v_proj.weight" in full:
                idx = [i for h in surv_kv for i in range(h*kv_hd, (h+1)*kv_hd)]
                t = t[idx, :].contiguous()
            elif "o_proj.weight" in full:
                idx = [i for h in surv_q for i in range(h*o_hd, (h+1)*o_hd)]
                t = t[:, idx].contiguous()

        total_a += t.numel() * t.element_size()
        sz = t.numel() * t.element_size()
        if shard_sz + sz > shard_max and shard:
            save_file(shard, str(DST / f"model-{si+1:05d}-of-TOTAL.safetensors"))
            print(f"  Shard {si}: {shard_sz/1e9:.1f}GB")
            si += 1; shard = {}; shard_sz = 0; gc.collect()
        shard[full] = t; shard_sz += sz
        wm[full] = f"model-{si+1:05d}-of-TOTAL.safetensors"

if shard:
    save_file(shard, str(DST / f"model-{si+1:05d}-of-TOTAL.safetensors"))
    print(f"  Shard {si}: {shard_sz/1e9:.1f}GB"); si += 1

total = si
for i in range(total):
    old = DST / f"model-{i+1:05d}-of-TOTAL.safetensors"
    new = DST / f"model-{i+1:05d}-of-{total:05d}.safetensors"
    old.rename(new)
    for k in wm:
        if wm[k] == f"model-{i+1:05d}-of-TOTAL.safetensors": wm[k] = new.name
json.dump({"metadata": {"total_size": total_a}, "weight_map": wm}, open(DST / "model.safetensors.index.json", "w"), indent=2)

# Config
config = json.loads(Path(SRC).joinpath("config.json").read_text())
config.pop("quantization_config", None)
config["model_type"] = "qwen3_5"
tc = config.get("text_config", config)
tc["num_attention_heads"] = new_heads
tc["num_key_value_heads"] = new_kv
json.dump(config, open(DST / "config.json", "w"), indent=2)
tokenizer.save_pretrained(str(DST))

# Copy benchmarks
bench_src = Path("output/forged/qwen3.5-27b/benchmark")
bench_dst = DST.parent / "benchmark"
if bench_src.exists():
    shutil.copytree(bench_src, bench_dst, dirs_exist_ok=True)

saved = total_b - total_a
print(f"\nBefore: {total_b/1e9:.2f}GB → After: {total_a/1e9:.2f}GB (saved {saved/1e6:.0f}MB)")
print(f"Config: {new_heads}Q/{new_kv}KV")
print(f"DONE: {DST}")
