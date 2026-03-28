"""Uniform defrag on the FORGED 4B model (after LoRA training)."""
import torch, sys, gc, json, shutil
from pathlib import Path
from safetensors.torch import save_file
sys.path.insert(0, "scripts")
from forge_model import compute_head_importance, get_model_info, get_layers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

SRC = "output/forged/qwen3.5-4b/model"
DST = Path("output/forged/qwen3.5-4b-defragged/model")

print("[1] Loading forged 4B...")
model = AutoModelForCausalLM.from_pretrained(SRC, device_map="auto", dtype=torch.float16, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(SRC)
info = get_model_info("Qwen/Qwen3.5-4B")

print("[2] Importance + uniform group selection...")
importance = compute_head_importance(model, info)
num_heads, num_kv = info["num_heads"], info["num_kv_heads"]
group_size = num_heads // num_kv
attn_layers = [li for li in range(info["num_layers"]) if importance[li].min() < float("inf")]

# Per-layer: remove 1 worst GQA group
dead_heads = {}
for li in attn_layers:
    groups = []
    for kv_h in range(num_kv):
        grp = list(range(kv_h * group_size, (kv_h + 1) * group_size))
        grp_imp = sum(importance[li, h].item() for h in grp)
        groups.append((grp_imp, kv_h, grp))
    groups.sort()
    _, _, dead_grp = groups[0]
    dead_heads[li] = dead_grp

new_heads = num_heads - group_size
new_kv = num_kv - 1

# Get dims
layers = get_layers(model)
sample = layers[attn_layers[0]].self_attn
q_hd = sample.q_proj.weight.shape[0] // num_heads
kv_hd = sample.k_proj.weight.shape[0] // num_kv
o_hd = sample.o_proj.weight.shape[1] // num_heads
print(f"  {len(attn_layers)} layers, removing 1 group each → {new_heads}Q/{new_kv}KV")

print("[3] Defrag + save...")
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
        surv_kv = sorted(set(range(num_kv)) - {dead[0] // group_size})

        if "q_proj.weight" in name:
            idx = [i for h in surv_q for i in range(h*q_hd, (h+1)*q_hd)]
            t = t[idx, :].contiguous()
        elif "k_proj.weight" in name or "v_proj.weight" in name:
            idx = [i for h in surv_kv for i in range(h*kv_hd, (h+1)*kv_hd)]
            t = t[idx, :].contiguous()
        elif "o_proj.weight" in name:
            idx = [i for h in surv_q for i in range(h*o_hd, (h+1)*o_hd)]
            t = t[:, idx].contiguous()

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
config = json.loads(Path(SRC).joinpath("config.json").read_text())
tc = config.get("text_config", config)
tc["num_attention_heads"] = new_heads
tc["num_key_value_heads"] = new_kv
json.dump(config, open(DST / "config.json", "w"), indent=2)
tokenizer.save_pretrained(str(DST))

# Copy generation samples
bench_src = Path("output/forged/qwen3.5-4b/benchmark")
bench_dst = DST.parent / "benchmark"
if bench_src.exists():
    shutil.copytree(bench_src, bench_dst, dirs_exist_ok=True)

print(f"Before: {total_b/1e9:.2f}GB → After: {total_a/1e9:.2f}GB (saved {(total_b-total_a)/1e6:.0f}MB)")
print(f"Config: {new_heads}Q/{new_kv}KV")

# Quick eval
print("[4] Eval...")
del model; gc.collect(); torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(str(DST), device_map="auto", torch_dtype=torch.float16)
from datasets import load_dataset
from torch.utils.data import DataLoader
val = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train[5000:5050]")
def tok_fn(ex):
    texts = [q + "\n" + a for q, a in zip(ex["query"], ex["answer"])]
    return tokenizer(texts, truncation=True, max_length=256, padding="max_length", return_tensors="pt")
val = val.filter(lambda x: len(str(x["query"]).strip()) > 20)
val = val.map(tok_fn, batched=True, remove_columns=val.column_names)
val.set_format("torch")
device = next(model.parameters()).device
model.eval()
tl, tt = 0.0, 0
with torch.no_grad():
    for batch in DataLoader(val, batch_size=4):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out = model(input_ids=ids, attention_mask=mask, labels=ids)
        tl += out.loss.float().item() * (mask > 0).sum().item()
        tt += (mask > 0).sum().item()
ppl = torch.exp(torch.tensor(tl / tt)).item()
print(f"FORGED+DEFRAGGED: ppl={ppl:.4f} (baseline was 3.04, forged was 2.19)")
