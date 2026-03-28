"""Fix defragged model config to match actual tensor dimensions."""
import json
from safetensors import safe_open
from pathlib import Path

def fix_defrag_config(model_dir):
    model_dir = Path(model_dir)
    config = json.load(open(model_dir / "config.json"))
    tc = config.get("text_config", config)

    # Read actual dimensions from tensors
    q_shape = k_shape = None
    for sf in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(sf), framework="pt") as f:
            for k in f.keys():
                if "self_attn.q_proj.weight" in k and q_shape is None:
                    q_shape = f.get_tensor(k).shape
                if "self_attn.k_proj.weight" in k and k_shape is None:
                    k_shape = f.get_tensor(k).shape
            if q_shape and k_shape:
                break

    if not q_shape or not k_shape:
        print("No self_attn found!")
        return

    # Original dims for reference
    orig_q_per_head = 512  # head_dim * 2 for Qwen3.5 (rope + nope)
    orig_kv_per_head = 256  # head_dim

    new_num_heads = q_shape[0] // orig_q_per_head
    new_num_kv = k_shape[0] // orig_kv_per_head

    print(f"Q: {q_shape} → {new_num_heads} heads (/{orig_q_per_head})")
    print(f"K: {k_shape} → {new_num_kv} KV heads (/{orig_kv_per_head})")
    print(f"GQA ratio: {new_num_heads}/{new_num_kv} = {new_num_heads // new_num_kv}")

    tc["num_attention_heads"] = new_num_heads
    tc["num_key_value_heads"] = new_num_kv
    # hidden_size stays the same
    # head_dim stays 256
    # Everything else unchanged

    json.dump(config, open(model_dir / "config.json", "w"), indent=2)
    print(f"Config updated: {new_num_heads}Q/{new_num_kv}KV, hidden_size={tc['hidden_size']}")

fix_defrag_config("output/forged/qwen3.5-4b/model")
