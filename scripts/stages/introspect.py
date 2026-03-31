"""
Model introspection — detect what a model currently IS as an alloy spec.

Given any HuggingFace model ID or local path, generates the equivalent
SourceConfigStage + detects what modifications are possible.

Usage:
    from stages.introspect import introspect_model
    spec = introspect_model("Qwen/Qwen3.5-4B")
    print(spec)  # { "source": {...}, "currentCapabilities": {...}, "possibleStages": [...] }
"""

import json
from pathlib import Path
from typing import Optional


def introspect_model(model_id: str, local_path: Optional[str] = None) -> dict:
    """Introspect a model and return its current state as an alloy-compatible spec.

    Can work in two modes:
    1. Remote (HF API) — reads config.json without downloading weights
    2. Local — reads from a local model directory

    Returns dict with:
    - source: AlloySource-compatible dict
    - currentCapabilities: what the model can do now
    - possibleStages: what stages can be applied
    - currentAlloy: a complete alloy representing the model as-is
    """
    config = _load_config(model_id, local_path)
    if not config:
        return {"error": f"Could not load config for {model_id}"}

    # Detect architecture family
    arch = _detect_architecture(config)
    text_config = config.get("text_config", config)

    # Extract current capabilities
    hidden_size = text_config.get("hidden_size", 0)
    num_layers = text_config.get("num_hidden_layers", 0)
    num_heads = text_config.get("num_attention_heads", 0)
    num_kv_heads = text_config.get("num_key_value_heads", num_heads)
    context_length = text_config.get("max_position_embeddings", 4096)
    vocab_size = text_config.get("vocab_size", 0)
    intermediate_size = text_config.get("intermediate_size", 0)

    # Detect MoE
    is_moe = "num_experts" in text_config or "num_local_experts" in text_config
    total_experts = text_config.get("num_experts", text_config.get("num_local_experts", 0))
    active_experts = text_config.get("num_experts_per_tok", text_config.get("num_selected_experts", 0))

    # Detect existing modalities
    modalities = ["text"]
    if "vision_config" in config or "visual" in config:
        modalities.append("vision")
    if "audio_config" in config:
        modalities.append("audio")

    # Detect RoPE scaling
    rope_scaling = text_config.get("rope_scaling")
    has_rope = "rope_theta" in text_config or rope_scaling is not None

    # Estimate parameter count
    params_b = _estimate_params(text_config)
    fp16_gb = params_b * 2 / 1e9
    q4_gb = params_b * 0.5 / 1e9

    # Determine possible stages
    possible_stages = _determine_possible_stages(
        arch=arch,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        is_moe=is_moe,
        total_experts=total_experts,
        has_rope=has_rope,
        context_length=context_length,
        modalities=modalities,
        hidden_size=hidden_size,
    )

    # Build current alloy representation
    source = {
        "baseModel": model_id,
        "architecture": arch,
        "isMoE": is_moe,
    }
    if is_moe:
        source["totalExperts"] = total_experts

    current_capabilities = {
        "architecture": arch,
        "parameters": f"{params_b/1e9:.1f}B",
        "fp16SizeGb": round(fp16_gb, 1),
        "q4SizeGb": round(q4_gb, 1),
        "layers": num_layers,
        "heads": num_heads,
        "kvHeads": num_kv_heads,
        "hiddenSize": hidden_size,
        "intermediateSize": intermediate_size,
        "contextLength": context_length,
        "vocabSize": vocab_size,
        "modalities": modalities,
        "isMoE": is_moe,
        "totalExperts": total_experts if is_moe else None,
        "activeExperts": active_experts if is_moe else None,
        "hasRoPE": has_rope,
        "ropeScaling": rope_scaling,
    }

    # Generate the "current alloy" — what the model IS right now
    current_alloy = {
        "name": model_id.split("/")[-1].lower(),
        "version": "1.0.0",
        "description": f"Introspected spec for {model_id}",
        "author": "introspected",
        "tags": ["introspected"],
        "source": source,
        "stages": [],
        "cycles": 1,
    }

    return {
        "source": source,
        "currentCapabilities": current_capabilities,
        "possibleStages": possible_stages,
        "currentAlloy": current_alloy,
    }


def _load_config(model_id: str, local_path: Optional[str] = None) -> Optional[dict]:
    """Load model config from local path or HuggingFace."""
    # Try local first
    if local_path:
        config_path = Path(local_path) / "config.json"
        if config_path.exists():
            return json.loads(config_path.read_text())

    # Try HF cache
    from pathlib import Path
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    slug = f"models--{model_id.replace('/', '--')}"
    cached = cache_dir / slug
    if cached.exists():
        snapshots = cached / "snapshots"
        if snapshots.exists():
            for snap in sorted(snapshots.iterdir(), reverse=True):
                config_path = snap / "config.json"
                if config_path.exists():
                    return json.loads(config_path.read_text())

    # Try HF API (no weights download)
    try:
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(model_id, "config.json")
        return json.loads(Path(config_path).read_text())
    except Exception:
        pass

    # Last resort: transformers AutoConfig
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id)
        return config.to_dict()
    except Exception:
        return None


def _detect_architecture(config: dict) -> str:
    """Detect the architecture family from config."""
    model_type = config.get("model_type", "").lower()
    arch_map = {
        "qwen2": "qwen2",
        "qwen3": "qwen3_5",
        "llama": "llama",
        "mistral": "mistral",
        "phi": "phi",
        "gemma": "gemma",
        "deepseek": "deepseek",
        "mixtral": "mistral",
        "starcoder": "starcoder",
    }
    for key, arch in arch_map.items():
        if key in model_type:
            return arch
    return model_type or "unknown"


def _estimate_params(config: dict) -> float:
    """Estimate total parameters from config."""
    h = config.get("hidden_size", 768)
    L = config.get("num_hidden_layers", 12)
    V = config.get("vocab_size", 32000)
    I = config.get("intermediate_size", h * 4)
    n_heads = config.get("num_attention_heads", 12)
    n_kv = config.get("num_key_value_heads", n_heads)

    head_dim = h // n_heads if n_heads > 0 else 64

    # Attention: Q + K + V + O projections
    attn_params = L * (h * (n_heads * head_dim) + h * (n_kv * head_dim) * 2 + (n_heads * head_dim) * h)
    # MLP: gate + up + down
    mlp_params = L * (h * I * 3)
    # Embeddings
    embed_params = V * h * 2
    # Layer norms
    norm_params = L * h * 4

    return attn_params + mlp_params + embed_params + norm_params


def _determine_possible_stages(arch: str, num_heads: int, num_kv_heads: int,
                                is_moe: bool, total_experts: int, has_rope: bool,
                                context_length: int, modalities: list,
                                hidden_size: int) -> list:
    """Determine which alloy stages can be applied to this model."""
    stages = []

    # Source config is always possible
    stages.append({
        "type": "source-config",
        "available": True,
        "reason": "Configure target capabilities",
    })

    # Prune — needs attention heads
    if num_heads > 4:
        max_prune = min(0.9, 1.0 - (4 / num_heads))
        stages.append({
            "type": "prune",
            "available": True,
            "maxLevel": round(max_prune, 2),
            "reason": f"{num_heads} heads, can prune up to {max_prune:.0%}",
        })

    # Train — always possible
    stages.append({
        "type": "train",
        "available": True,
        "reason": "LoRA fine-tuning on domain data",
    })

    # LoRA — always possible
    stages.append({
        "type": "lora",
        "available": True,
        "reason": "LoRA adapter training",
    })

    # Expert prune — only for MoE models
    if is_moe and total_experts > 1:
        stages.append({
            "type": "expert-prune",
            "available": True,
            "maxExperts": total_experts,
            "reason": f"MoE with {total_experts} experts — can reduce",
        })
    else:
        stages.append({
            "type": "expert-prune",
            "available": False,
            "reason": "Not a MoE model",
        })

    # Context extend — needs RoPE
    if has_rope:
        stages.append({
            "type": "context-extend",
            "available": True,
            "currentLength": context_length,
            "reason": f"RoPE detected, current context: {context_length}",
        })
    else:
        stages.append({
            "type": "context-extend",
            "available": False,
            "reason": "No RoPE scaling support detected",
        })

    # Modality — vision
    if "vision" not in modalities:
        stages.append({
            "type": "modality",
            "modality": "vision",
            "available": True,
            "reason": f"Text-only model, hidden_size={hidden_size} supports projection",
        })
    else:
        stages.append({
            "type": "modality",
            "modality": "vision",
            "available": False,
            "reason": "Already has vision capability",
        })

    # Modality — audio
    if "audio" not in modalities:
        stages.append({
            "type": "modality",
            "modality": "audio",
            "available": True,
            "reason": "Can add audio encoder",
        })

    # Compact — always possible after pruning
    stages.append({
        "type": "compact",
        "available": True,
        "reason": "Mixed-precision compaction based on head utilization",
    })

    # Quant — always possible
    stages.append({
        "type": "quant",
        "available": True,
        "reason": "GGUF, MLX, ONNX, safetensors",
    })

    # Eval — always possible
    stages.append({
        "type": "eval",
        "available": True,
        "reason": "Benchmark against HumanEval, MMLU, GSM8K, etc.",
    })

    # Publish — always possible
    stages.append({
        "type": "publish",
        "available": True,
        "reason": "Push to HuggingFace with alloy",
    })

    # Deploy — always possible
    stages.append({
        "type": "deploy",
        "available": True,
        "reason": "Push to grid node for serving",
    })

    return stages


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m stages.introspect <model_id>")
        sys.exit(1)

    result = introspect_model(sys.argv[1])
    print(json.dumps(result, indent=2))
