import torch, sys, gc, json, shutil
from pathlib import Path
from safetensors.torch import save_file
sys.path.insert(0, "scripts")
from forge_model import compute_head_importance, get_model_info, get_layers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL = "Qwen/Qwen3.5-4B"
DST = Path("output/defragged-4b-clean")

print("[1] Loading base model fp16...")
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", dtype=torch.float16, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
info = get_model_info(MODEL)

print("[2] Importance...")
importance = compute_head_importance(model, info)
num_heads, num_kv = info["num_heads"], info["num_kv_heads"]
group_size = num_heads // num_kv

attn_layers = [li for li in range(info["num_layers"]) if importance[li].min() < float("inf")]
print(f"  {len(attn_layers)} self_attn layers, {num_heads} heads, {num_kv} KV, group_size {group_size}")

# Rank groups per layer, pick worst 1 uniformly
dead_heads = {}
for li in attn_layers:
    groups = []
    for kv_h in range(num_kv):
        grp = list(range(kv_h * group_size, (kv_h + 1) * group_size))
        grp_imp = sum(importance[li, h].item() for h in grp)
        groups.append((grp_imp, kv_h, grp))
    groups.sort()
    # Remove 1 worst group
    _, _, dead_grp = groups[0]
    dead_heads[li] = dead_grp

new_heads = num_heads - group_size  # 16 - 4 = 12
new_kv = num_kv - 1  # 4 - 1 = 3
print(f"  Removing 1 group/layer → {new_heads}Q/{new_kv}KV")

# Get per-projection head dims from actual tensors
layers = get_layers(model)
sample = layers[attn_layers[0]].self_attn
q_hd = sample.q_proj.weight.shape[0] // num_heads
kv_hd = sample.k_proj.weight.shape[0] // num_kv
o_hd = sample.o_proj.weight.shape[1] // num_heads
print(f"  Q_hd={q_hd}, KV_hd={kv_hd}, O_hd={o_hd}")

print("[3] Defragging + saving...")
DST.mkdir(parents=True, exist_ok=True)

shard, shard_sz, si, wm = {}, 0, 0, {}
shard_max = 5 * 1024**3
total_b, total_a = 0, 0

for name, param in model.named_parameters():
    t = param.data.to(torch.bfloat16).cpu().contiguous()
    total_b += t.numel() * t.element_size()

    li = None
    if "self_attn" in name and "layers." in name:
        li = int(name.split("layers.")[1].split(".")[0])

    if li in dead_heads:
        dead = dead_heads[li]
        surv_q = sorted(set(range(num_heads)) - set(dead))
        surv_kv_set = set()
        for kv_h in range(num_kv):
            grp = set(range(kv_h * group_size, (kv_h + 1) * group_size))
            if not grp.issubset(set(dead)):
                surv_kv_set.add(kv_h)
        surv_kv = sorted(surv_kv_set)

        if "q_proj.weight" in name:
            idx = [i for h in surv_q for i in range(h*q_hd, (h+1)*q_hd)]
            t = t[idx, :].contiguous()
        elif "k_proj.weight" in name or "v_proj.weight" in name:
            idx = [i for h in surv_kv for i in range(h*kv_hd, (h+1)*kv_hd)]
            t = t[idx, :].contiguous()
        elif "o_proj.weight" in name:
            idx = [i for h in surv_q for i in range(h*o_hd, (h+1)*o_hd)]
            t = t[:, idx].contiguous()
        elif "q_norm.weight" in name or "k_norm.weight" in name:
            pass  # norms are per head_dim not per head — keep as is

    total_a += t.numel() * t.element_size()
    sz = t.numel() * t.element_size()
    if shard_sz + sz > shard_max and shard:
        save_file(shard, str(DST / f"model-{si+1:05d}-of-TOTAL.safetensors"))
        si += 1; shard = {}; shard_sz = 0; gc.collect()
    shard[name] = t; shard_sz += sz
    wm[name] = f"model-{si+1:05d}-of-TOTAL.safetensors"

if shard:
    save_file(shard, str(DST / f"model-{si+1:05d}-of-TOTAL.safetensors")); si += 1

total = si
for i in range(total):
    old = DST / f"model-{i+1:05d}-of-TOTAL.safetensors"
    new = DST / f"model-{i+1:05d}-of-{total:05d}.safetensors"
    old.rename(new)
    for k in wm:
        if wm[k] == f"model-{i+1:05d}-of-TOTAL.safetensors": wm[k] = new.name
json.dump({"metadata": {"total_size": total_a}, "weight_map": wm}, open(DST / "model.safetensors.index.json", "w"), indent=2)

# Config
config = AutoConfig.from_pretrained(MODEL).to_dict()
tc = config.get("text_config", config)
tc["num_attention_heads"] = new_heads
tc["num_key_value_heads"] = new_kv
json.dump(config, open(DST / "config.json", "w"), indent=2)
tokenizer.save_pretrained(str(DST))

print(f"\nBefore: {total_b/1e9:.2f}GB, After: {total_a/1e9:.2f}GB, Saved: {(total_b-total_a)/1e6:.0f}MB")
print(f"Config: {new_heads}Q/{new_kv}KV")
print(f"DONE: {DST}")
