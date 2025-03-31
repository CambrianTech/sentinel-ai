import torch
from models.adaptive_transformer import AdaptiveCausalLmWrapper


def load_adaptive_model_gpt(model_name, baseline_model, config, device):
    model = AdaptiveCausalLmWrapper(config, baseline_model.get_input_embeddings(), baseline_model.get_input_embeddings()).to(device)

    model.eval()

    baseline_state = baseline_model.state_dict()
    adaptive_state = model.state_dict()

    loaded, skipped = [], []

    num_heads = config.n_head if hasattr(config, "n_head") else config.num_attention_heads
    hidden_size = config.n_embd if hasattr(config, "n_embd") else config.hidden_size
    head_dim = hidden_size // num_heads

    # Copy token and position embeddings explicitly
    try:
        model.token_embeddings.weight.data.copy_(baseline_model.transformer.wte.weight.data)
        model.position_embeddings.weight.data.copy_(baseline_model.transformer.wpe.weight.data)
        loaded.extend(["token_embeddings.weight", "position_embeddings.weight"])
    except Exception as e:
        skipped.extend(["token_embeddings.weight", "position_embeddings.weight"])

    for name in adaptive_state:
        if name in ["token_embeddings.weight", "position_embeddings.weight"]:
            continue

        if name in baseline_state and adaptive_state[name].shape == baseline_state[name].shape:
            adaptive_state[name].copy_(baseline_state[name])
            loaded.append(name)

        elif ".attn.W_" in name:
            try:
                parts = name.split(".")
                layer = int(parts[1])
                proj_type = parts[3].split("_")[1]  # q/k/v/o
                head = int(parts[4])
                param = parts[5]

                if proj_type in {"q", "k", "v"}:
                    qkv_w = baseline_state[f"transformer.h.{layer}.attn.c_attn.weight"]
                    qkv_b = baseline_state[f"transformer.h.{layer}.attn.c_attn.bias"]

                    qkv_w = qkv_w.view(3, num_heads, head_dim, hidden_size)
                    qkv_b = qkv_b.view(3, num_heads, head_dim)

                    idx = {"q": 0, "k": 1, "v": 2}[proj_type]
                    if param == "weight":
                        adaptive_state[name].copy_(qkv_w[idx, head])
                    elif param == "bias":
                        adaptive_state[name].copy_(qkv_b[idx, head])
                    loaded.append(name)

                elif proj_type == "o":
                    c_proj_weight = baseline_state[f"transformer.h.{layer}.attn.c_proj.weight"]
                    c_proj_bias = baseline_state[f"transformer.h.{layer}.attn.c_proj.bias"]

                    start = head * head_dim
                    end = (head + 1) * head_dim

                    if param == "weight":
                        adaptive_state[name].copy_(c_proj_weight[:, start:end])
                    elif param == "bias":
                        adaptive_state[name].copy_(c_proj_bias[start:end])
                    loaded.append(name)

            except Exception as e:
                skipped.append(name)

        else:
            skipped.append(name)

    print(f"✅ Adaptive model initialized from {model_name} weights ({len(loaded)}/{len(adaptive_state)} parameters loaded)")
    if skipped:
        print(f"⚠️ Skipped {len(skipped)} params (shape mismatch or custom):")
        for p in skipped[:10]:
            print("   ", p)
        if len(skipped) > 10:
            print(f"   ... and {len(skipped) - 10} more.")

    return model