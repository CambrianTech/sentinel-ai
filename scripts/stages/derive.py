"""
Delta derivation — compare source (introspected) to target (desired) and derive stages.

The user specifies WHAT they want (target). This module figures out HOW (stages).
Only the deltas produce work. No target change = no stage = no cost.

Usage:
    from stages.derive import derive_stages
    stages = derive_stages(introspected_capabilities, target)
"""

from typing import Optional


def derive_stages(capabilities: dict, target: dict) -> list[dict]:
    """Derive alloy stages from the delta between current capabilities and target.

    Args:
        capabilities: Output of introspect_model()["currentCapabilities"]
        target: AlloyTarget dict — only set fields that should change

    Returns:
        List of alloy stage configs (ready to put in alloy["stages"])
    """
    stages = []

    # ── Context extension ────────────────────────────────────────
    target_ctx = target.get("contextLength")
    current_ctx = capabilities.get("contextLength", 4096)
    if target_ctx and target_ctx > current_ctx:
        stages.append({
            "type": "context-extend",
            "targetLength": target_ctx,
            "method": "yarn",  # Default to YaRN — best general-purpose
        })

    # ── Modality addition ────────────────────────────────────────
    target_mods = target.get("modalities")
    current_mods = set(capabilities.get("modalities", ["text"]))
    if target_mods:
        new_mods = set(target_mods) - current_mods
        for mod in sorted(new_mods):
            encoder = _recommend_encoder(mod)
            stages.append({
                "type": "modality",
                "modality": mod,
                "encoderModel": encoder["model"],
                "projectionArch": "mlp",
                "freezeBase": True,
                "freezeEncoder": True,
                "trainingSteps": 5000,
            })

    # ── Expert pruning (MoE only) ────────────────────────────────
    target_experts = target.get("experts")
    current_experts = capabilities.get("totalExperts")
    if target_experts and current_experts and target_experts < current_experts:
        stages.append({
            "type": "expert-prune",
            "keepExperts": target_experts,
            "selectionStrategy": "activation",
        })

    # ── Head pruning ─────────────────────────────────────────────
    prune_ratio = target.get("pruneRatio")
    if prune_ratio and prune_ratio > 0:
        stages.append({
            "type": "prune",
            "strategy": "entropy",
            "level": prune_ratio,
        })

    # ── Domain training ──────────────────────────────────────────
    domain = target.get("domain")
    if domain:
        stages.append({
            "type": "train",
            "domain": domain,
            "steps": 1000,
            "learningRate": "2e-4",
        })

    # ── Quantization ─────────────────────────────────────────────
    output_formats = target.get("outputFormats")
    quant_types = target.get("quantTypes", ["Q4_K_M"])
    if output_formats:
        for fmt in output_formats:
            stages.append({
                "type": "quant",
                "format": fmt,
                "quantTypes": quant_types,
                "deviceTargets": target.get("targetDevices", []),
            })

    # ── Benchmarks ───────────────────────────────────────────────
    benchmarks = target.get("benchmarks")
    if benchmarks:
        stages.append({
            "type": "eval",
            "benchmarks": [{"name": b} for b in benchmarks],
            "compareToBase": True,
        })

    # ── Publish ──────────────────────────────────────────────────
    if target.get("publish"):
        stages.append({
            "type": "publish",
            "org": "continuum-ai",
            "includeAlloy": True,
            "cardFromBenchmarks": True,
            "tags": ["forge-alloy"],
        })

    # ── Deploy ───────────────────────────────────────────────────
    deploy_to = target.get("deployTo")
    if deploy_to:
        stages.append({
            "type": "deploy",
            "target": deploy_to,
            "healthCheck": True,
            "warmup": True,
        })

    return stages


def derive_alloy(model_id: str, target: dict, capabilities: dict = None) -> dict:
    """Build a complete alloy from a model ID and target spec.

    If capabilities not provided, introspects the model first.
    """
    if capabilities is None:
        from .introspect import introspect_model
        result = introspect_model(model_id)
        capabilities = result["currentCapabilities"]
        source = result["source"]
    else:
        source = {
            "baseModel": model_id,
            "architecture": capabilities.get("architecture", "unknown"),
            "isMoE": capabilities.get("isMoE", False),
        }

    stages = derive_stages(capabilities, target)
    slug = model_id.split("/")[-1].lower()

    # Build name from target modifications
    parts = [slug]
    if target.get("domain"):
        parts.append(target["domain"])
    if target.get("modalities"):
        new_mods = set(target["modalities"]) - set(capabilities.get("modalities", ["text"]))
        parts.extend(sorted(new_mods))
    parts.append("forged")

    return {
        "name": "-".join(parts),
        "version": "1.0.0",
        "description": f"Delta forge: {', '.join(s['type'] for s in stages)}",
        "author": "continuum-ai",
        "tags": ["forge-alloy", "delta-forge"],
        "license": "apache-2.0",
        "source": source,
        "target": target,
        "stages": stages,
        "cycles": 3 if any(s["type"] in ("prune", "train") for s in stages) else 1,
    }


def estimate_duration(stages: list, model_params_b: float = 4.0, vram_gb: float = 32.0) -> dict:
    """Estimate forge duration and cost from stages.

    Returns dict with totalMinutes, perStage breakdown, and hardware requirements.
    """
    # Steps per minute varies by model size (benchmarked on RTX 5090)
    if model_params_b <= 2:
        steps_per_min = 20
    elif model_params_b <= 5:
        steps_per_min = 10
    elif model_params_b <= 10:
        steps_per_min = 5
    elif model_params_b <= 15:
        steps_per_min = 2.5
    elif model_params_b <= 30:
        steps_per_min = 1
    else:
        steps_per_min = 0.5

    breakdown = []
    total = 0

    for stage in stages:
        stype = stage["type"]
        if stype == "train":
            steps = stage.get("steps", 1000)
            mins = steps / steps_per_min
            breakdown.append({"stage": stype, "minutes": round(mins), "reason": f"{steps} steps"})
            total += mins
        elif stype == "prune":
            mins = 2  # Pruning is fast
            breakdown.append({"stage": stype, "minutes": mins, "reason": "head analysis"})
            total += mins
        elif stype == "modality":
            steps = stage.get("trainingSteps", 5000)
            mins = steps / steps_per_min
            breakdown.append({"stage": stype, "minutes": round(mins), "reason": f"projection training {steps} steps"})
            total += mins
        elif stype == "context-extend":
            mins = 1  # Config change, minimal compute
            training_steps = stage.get("trainingSteps")
            if training_steps:
                mins += training_steps / steps_per_min
            breakdown.append({"stage": stype, "minutes": round(mins), "reason": "RoPE rescaling"})
            total += mins
        elif stype == "quant":
            mins = 5  # GGUF conversion
            breakdown.append({"stage": stype, "minutes": mins, "reason": "quantization"})
            total += mins
        elif stype == "eval":
            benchmarks = stage.get("benchmarks", [])
            mins = len(benchmarks) * 10  # ~10 min per benchmark
            breakdown.append({"stage": stype, "minutes": mins, "reason": f"{len(benchmarks)} benchmarks"})
            total += mins
        else:
            breakdown.append({"stage": stype, "minutes": 1, "reason": "minimal"})
            total += 1

    return {
        "totalMinutes": round(total),
        "breakdown": breakdown,
        "modelParamsB": model_params_b,
        "assumedHardware": f"RTX 5090 {vram_gb}GB",
    }


def _recommend_encoder(modality: str) -> dict:
    """Recommend encoder model for a modality."""
    encoders = {
        "vision": {"model": "openai/clip-vit-large-patch14", "params": "0.4B"},
        "audio": {"model": "openai/whisper-large-v3", "params": "1.6B"},
        "video": {"model": "openai/clip-vit-large-patch14", "params": "0.4B"},
    }
    return encoders.get(modality, {"model": "unknown", "params": "?"})


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, sys
    if len(sys.argv) < 3:
        print("Usage: python -m stages.derive <model_id> '<target_json>'")
        print("Example: python -m stages.derive Qwen/Qwen3.5-4B '{\"domain\":\"code\",\"contextLength\":32768,\"modalities\":[\"text\",\"vision\"]}'")
        sys.exit(1)

    model_id = sys.argv[1]
    target = json.loads(sys.argv[2])
    alloy = derive_alloy(model_id, target)

    print(json.dumps(alloy, indent=2))
    print(f"\nEstimate:")
    params_b = float(alloy["source"].get("baseModel", "").split("-")[-1].replace("B", "").replace("b", "") or "4")
    est = estimate_duration(alloy["stages"], params_b)
    print(json.dumps(est, indent=2))
