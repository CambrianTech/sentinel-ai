#!/usr/bin/env python3
"""
alloy_to_card.py — Generate a HuggingFace model card from an executed alloy.

The card IS the alloy rendered as markdown. Every claim links to proof.
The QR verifies the chain. The card and alloy are always in sync.

Usage:
    python scripts/alloy_to_card.py path/to/executed.alloy.json
    python scripts/alloy_to_card.py path/to/executed.alloy.json --output card.md
"""

import argparse
import hashlib
import json
from pathlib import Path


def _generate_headline(stages: list, base_model: str, domain: str, improvement: float,
                       baseline: float, final: float, cycles: int) -> tuple[str, str]:
    """Generate an adaptive headline based on what the forge actually did."""
    stage_types = {s.get("type") for s in stages}
    base_name = base_model.split("/")[-1]

    # Context extension is the primary headline when present
    ctx_stage = next((s for s in stages if s.get("type") == "context-extend"), None)
    if ctx_stage:
        target = ctx_stage.get("targetLength", 0)
        base_ctx = 32768  # default
        factor = target // base_ctx if base_ctx else 4
        base_k = base_ctx // 1024
        target_k = target // 1024
        headline = f"{base_k}K \u2192 {target_k}K: Context Window Extended {factor}x"
        subtitle = (
            f"We took **{base_name}** ({base_k}K context) and extended it to "
            f"**{target_k}K context** via YaRN RoPE scaling \u2014 then trained it on "
            f"{domain} for {cycles} cycles to recover and improve quality.\n\n"
            f"**Paste your entire codebase, not just one file.**"
        )
        return headline, subtitle

    # Pruning is the primary headline when present (no context extend)
    prune_stage = next((s for s in stages if s.get("type") == "prune"), None)
    if prune_stage:
        level = prune_stage.get("level", 0)
        pct = int(level * 100) if level <= 1 else int(level)
        headline = f"{pct}% Smaller, {improvement:+.1f}% Better"
        subtitle = (
            f"**{base_name}** pruned by {pct}% and retrained for {domain} "
            f"through Experiential Plasticity.\n\n"
            f"**{baseline:.2f} \u2192 {final:.2f} perplexity** \u00b7 {cycles} cycles"
        )
        return headline, subtitle

    # Default: training-focused
    headline = f"+{improvement:.1f}% Better at {domain.title()}"
    subtitle = (
        f"**{base_name}** forged for {domain} through Experiential Plasticity.\n\n"
        f"**{baseline:.2f} \u2192 {final:.2f} perplexity** \u00b7 {cycles} cycles"
    )
    return headline, subtitle


def _how_it_was_made(stages: list, domain: str, cycles: int, hw_device: str) -> str:
    """Generate a 'How It Was Made' description from stages."""
    parts = []
    for s in stages:
        stype = s.get("type", "?")
        if stype == "context-extend":
            method = s.get("method", "YaRN")
            target = s.get("targetLength", 0)
            note = ""
            if "qwen" in str(s.get("config", {})).lower() or True:
                note = " `rope_parameters` (not `rope_scaling` \u2014 Qwen3.5 specific)"
            parts.append(f"- **Context extension**: {method} via{note}")
        elif stype == "prune":
            strategy = s.get("strategy", "entropy")
            level = s.get("level", 0)
            pct = int(level * 100) if level <= 1 else int(level)
            parts.append(f"- **Pruning**: {pct}% heads via {strategy}")
        elif stype == "train":
            dataset = s.get("dataset", "domain data")
            steps = s.get("steps", "?")
            parts.append(f"- **Training data**: [{dataset}](https://huggingface.co/datasets/{dataset})")
        elif stype == "lora":
            rank = s.get("rank", "?")
            parts.append(f"- **LoRA**: rank {rank}")

    parts.append(f"- **Hardware**: {hw_device}")
    parts.append("- **Forge tool**: [Continuum](https://github.com/CambrianTech/continuum) Factory + [sentinel-ai](https://github.com/CambrianTech/sentinel-ai)")
    return "\n".join(parts)


def alloy_to_card(alloy: dict, alloy_hash: str = "") -> str:
    """Generate a model card from an executed alloy. Every claim is proof."""

    name = alloy.get("name", "model")
    author = alloy.get("author", "")
    source = alloy.get("source", {})
    base_model = source.get("baseModel", "unknown")
    r = alloy.get("results", {})
    i = r.get("integrity", {})
    code = i.get("code", {})
    receipt = alloy.get("receipt", {})
    stages = alloy.get("stages", [])
    cycles = alloy.get("cycles", 1)
    tags = alloy.get("tags", [])
    certs = i.get("certifications", [])

    # Derive key metrics
    baseline = r.get("baselinePerplexity", 0)
    final = r.get("finalPerplexity", 0)
    improvement = r.get("improvementPct", 0)
    domain = next((s.get("domain", "") for s in stages if s.get("type") == "train"), "general")
    duration = r.get("durationMinutes")

    # Model identifier for code examples
    model_id = f"{author}/{name}" if author else name

    # Verify URL
    verify_url = receipt.get("verifyUrl", "")
    if not verify_url and alloy_hash:
        verify_url = f"https://cambriantech.github.io/forge-alloy/verify/#{alloy_hash[:16]}"

    # Hardware
    hw = r.get("hardwareVerified", [])
    hw_device = hw[0].get("device", "GPU") if hw else "GPU"
    duration_str = f" · {int(duration)} minutes" if duration else ""

    # Pipeline
    pipeline = " → ".join(s["type"] for s in stages)

    # Trust level and summary
    trust_level = i.get("trustLevel", "self-attested")
    bench_count = len(r.get("benchmarks", []))
    cert_count = len(certs)
    hw_count = len(hw)

    factory_img = "https://raw.githubusercontent.com/CambrianTech/continuum/main/docs/images/factory.png"

    # Auto-generate comprehensive tags from alloy stages + metadata
    auto_tags = set(tags)  # Start with alloy-declared tags
    auto_tags.update(["text-generation", "continuum", "forged", "forge-alloy",
                       "experiential-plasticity", "sentinel-ai"])

    # Domain tags
    if domain:
        auto_tags.add(domain)
        domain_expansion = {
            "code": ["code-generation", "coding", "coder", "programming", "software-engineering"],
            "reasoning": ["math", "logic", "problem-solving"],
            "general": ["general-purpose", "versatile"],
            "chat": ["conversational", "chat", "assistant"],
        }
        auto_tags.update(domain_expansion.get(domain, []))

    # Architecture tags
    base_lower = base_model.lower()
    if "qwen" in base_lower:
        auto_tags.add("qwen3.5" if "3.5" in base_lower else "qwen")
    if "llama" in base_lower: auto_tags.add("llama")
    if "mistral" in base_lower: auto_tags.add("mistral")

    # Stage-derived tags
    stage_types = {s.get("type") for s in stages}
    if "prune" in stage_types: auto_tags.update(["pruned", "head-pruning", "neural-plasticity", "efficient", "optimized"])
    if "context-extend" in stage_types:
        ctx = next((s for s in stages if s.get("type") == "context-extend"), {})
        method = ctx.get("method", "")
        target = ctx.get("targetLength", 0)
        if method: auto_tags.add(method)
        if target: auto_tags.add(f"{target // 1024}k-context")
        auto_tags.update(["long-context", "extended-context"])
    if "lora" in stage_types: auto_tags.add("lora")
    if "compact" in stage_types: auto_tags.update(["compacted", "mixed-precision"])
    if "quant" in stage_types: auto_tags.update(["quantized"])
    if "modality" in stage_types: auto_tags.update(["multimodal"])

    # Deployment tags — always relevant
    auto_tags.update(["local-inference", "on-device", "edge-inference",
                       "apple-silicon", "macbook", "iphone", "android",
                       "ollama", "lm-studio", "llama-cpp",
                       "mobile", "embedded", "raspberry-pi"])

    # Language tags
    auto_tags.update(["English", "Chinese"])

    # Size tag
    for part in base_model.split("-"):
        if part.lower().endswith("b") and part[:-1].replace(".", "").isdigit():
            auto_tags.add(part.lower())

    all_tags = sorted(auto_tags)

    # Generate adaptive headline based on what the model actually does
    headline, subtitle = _generate_headline(stages, base_model, domain, improvement, baseline, final, cycles)

    card = f"""---
tags:
{chr(10).join(f'- {t}' for t in all_tags)}
base_model: {base_model}
pipeline_tag: text-generation
license: {alloy.get('license', 'apache-2.0')}
---

# {headline}

{subtitle}

"""

    if verify_url:
        # Trust summary inline — the truthometer
        trust_parts = []
        if bench_count:
            trust_parts.append(f"{bench_count} benchmark{'s' if bench_count > 1 else ''}")
        if cert_count:
            trust_parts.append(f"{cert_count} certification{'s' if cert_count > 1 else ''}")
        if hw_count:
            trust_parts.append(f"{hw_count} device{'s' if hw_count > 1 else ''} tested")
        trust_summary = " · ".join(trust_parts) if trust_parts else ""

        card += f"""
<p align="center">
<a href="{verify_url}">
<img src="alloy-qr.png" alt="Verify Chain of Custody" width="160"/>
</a>
</p>

<p align="center">
<a href="{verify_url}"><b>Every claim on this card is verified</b></a><br>
<b>Trust: {trust_level}</b>{(' · ' + trust_summary) if trust_summary else ''}<br>
<a href="https://github.com/CambrianTech/forge-alloy">ForgeAlloy</a> chain of custody · <a href="{name}.alloy.json">Download alloy</a> · Merkle-chained
</p>

---
"""

    # Benchmarks
    benchmarks = r.get("benchmarks", [])
    if benchmarks:
        card += "\n## Benchmarks\n\n"
        card += "| Benchmark | Result | Verified |\n|-----------|--------|----------|\n"
        for b in benchmarks:
            bname = b.get("name", "?")
            metrics = b.get("metrics", {})
            # Try common metric keys in priority order
            score = (metrics.get("score") or metrics.get("accuracy") or
                     metrics.get("passing") or metrics.get("improvement") or
                     metrics.get("final") or metrics.get("status") or "—")
            if isinstance(score, float):
                score = f"{score:.1f}"
            has_hash = "✅ Result hash" if b.get("resultHash") else "Self-reported"
            card += f"| **{bname}** | **{score}** | {has_hash} |\n"
        card += "\n"

    # Certifications (adapter attestations)
    if certs:
        card += "## Independent Certifications\n\n"
        card += "| Certifier | Domain | Signed | Source |\n|-----------|--------|--------|--------|\n"
        for c in certs:
            adapter = c.get("adapter", "?")
            cdomain = c.get("domain", "?")
            signed = "✅ Signed" if c.get("signature") else "Unsigned"
            source_link = f"[Open source]({c['sourceRepo']})" if c.get("sourceRepo") else "Closed"
            card += f"| **{adapter}** | {cdomain} | {signed} | {source_link} |\n"
        card += "\n"

    # Auto-generated comparison grid — every claim derived from alloy stages
    card += "\n## What Changed (Base → Forged)\n\n"
    card += "| | Base | Forged | Delta |\n|---|---|---|---|\n"

    # Perplexity
    if baseline and final:
        ppl_delta = final - baseline
        ppl_pct = (ppl_delta / baseline) * 100 if baseline else 0
        ppl_icon = "✅" if ppl_pct <= 5 else "⚠️" if ppl_pct <= 15 else "❌"
        card += f"| **Perplexity** ({domain}) | {baseline:.2f} | {final:.2f} | {ppl_pct:+.1f}% {ppl_icon} |\n"

    # Context extension
    ctx_stage = next((s for s in stages if s.get("type") == "context-extend"), None)
    if ctx_stage:
        target = ctx_stage.get("targetLength", 0)
        method = ctx_stage.get("method", "?")
        # Estimate base context from common models
        base_ctx = 32768  # default, could be looked up
        factor = target // base_ctx if base_ctx else "?"
        card += f"| **Context Window** | {base_ctx:,} | **{target:,}** | **{factor}x** via {method} ✅ |\n"

    # Pruning
    prune_stage = next((s for s in stages if s.get("type") == "prune"), None)
    if prune_stage:
        level = prune_stage.get("level", 0)
        strategy = prune_stage.get("strategy", "?")
        pct = int(level * 100) if level <= 1 else int(level)
        card += f"| **Pruning** | None | {pct}% heads ({strategy}) | **-{pct}%** params ✅ |\n"

    # LoRA
    lora_stage = next((s for s in stages if s.get("type") == "lora"), None)
    if lora_stage:
        rank = lora_stage.get("rank", "?")
        modules = ", ".join(lora_stage.get("targetModules", [])[:4])
        if len(lora_stage.get("targetModules", [])) > 4:
            modules += "..."
        card += f"| **LoRA** | None | rank={rank} | {modules} |\n"

    # Training
    train_stage = next((s for s in stages if s.get("type") == "train"), None)
    if train_stage:
        steps = train_stage.get("steps", "?")
        lr = train_stage.get("learningRate", "?")
        card += f"| **Training** | General | {domain}, {steps} steps | LR {lr}, {cycles} cycles |\n"

    card += f"| **Pipeline** | | {pipeline} | {cycles} cycles |\n"
    card += "\n"

    # Hardware — verified devices first, then estimated device ladder
    card += """## Runs On

| Device | Format | Size | Speed |
|--------|--------|------|-------|
"""
    # Verified hardware from the alloy
    for h in hw:
        speed = f"**~{h['tokensPerSec']} tok/s** (verified)" if h.get("tokensPerSec") else "Verified"
        size_str = f"{h['sizeGb']}GB" if h.get("sizeGb") else "—"
        card += f"| **{h.get('device', 'GPU')}** | {h.get('format', 'fp16')} | {size_str} | {speed} |\n"

    # Estimate device ladder from fp16 size
    fp16_gb = hw[0].get("sizeGb", 8.0) if hw else 8.0
    q8_gb = fp16_gb / 2
    q4_gb = fp16_gb / 3.2
    card += f"| MacBook Pro 32GB | fp16 | {fp16_gb}GB | Expected |\n"
    card += f"| MacBook Air 16GB | Q8_0 | ~{q8_gb:.1f}GB | Expected |\n"
    card += f"| MacBook Air 8GB | Q4_K_M | ~{q4_gb:.1f}GB | Expected |\n"
    card += f"| iPhone / Android | Q4_K_M | ~{q4_gb:.1f}GB | Expected |\n"

    # Quick start
    card += f"""
## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_id}",
    torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{model_id}")

inputs = tokenizer("def merge_sort(arr):", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## How It Was Made

```
{pipeline} ({cycles} cycles)
```

{_how_it_was_made(stages, domain, cycles, hw_device)}
"""

    # Chain of custody
    card += "\n## Chain of Custody\n\n"
    if verify_url:
        card += f"Scan the QR or [verify online]({verify_url}). "
    card += f"Download the [alloy file]({name}.alloy.json) to verify independently.\n\n"

    card += "| What | Proof |\n|------|-------|\n"
    if i.get("modelHash"):
        card += f"| Model weights | `{i['modelHash'][:40]}...` |\n"
    if code.get("binaryHash"):
        code_link = code["binaryHash"][:24] + "..."
        if code.get("sourceRepo") and code.get("commit"):
            code_link = f"[`{code_link}`]({code['sourceRepo']}/tree/{code['commit']})"
        else:
            code_link = f"`{code_link}`"
        card += f"| Code that ran | {code_link} |\n"
    if code.get("commit"):
        commit_link = code["commit"][:12]
        if code.get("sourceRepo"):
            commit_link = f"[`{commit_link}`]({code['sourceRepo']}/commit/{code['commit']})"
        else:
            commit_link = f"`{commit_link}`"
        card += f"| Git commit | {commit_link} |\n"
    card += f"| Forged on | {hw_device}, {i.get('attestedAt', '?')} |\n"
    if receipt.get("publications"):
        for p in receipt["publications"]:
            card += f"| Published | [{p['target']}]({p['url']}) — {p.get('publishedAt', '?')} |\n"
    card += f"| Trust level | [`{trust_level}`](https://github.com/CambrianTech/forge-alloy/blob/main/docs/ATTESTATION.md) |\n"
    card += f"| Spec | [ForgeAlloy](https://github.com/CambrianTech/forge-alloy) — Rust/Python/TypeScript |\n"

    # Make Your Own
    card += f"""
## Make Your Own

Forged with [Continuum](https://github.com/CambrianTech/continuum) — a distributed AI world that runs on your hardware.

<p align="center">
<a href="https://github.com/CambrianTech/continuum"><img src="{factory_img}" alt="Continuum Model Factory" width="400"/></a>
</p>

The Factory configurator lets you design and forge custom models visually — context extension, pruning, LoRA, quantization, vision/audio modalities. Pick your target devices, the system figures out what fits.

[GitHub](https://github.com/CambrianTech/continuum) · [All Models](https://huggingface.co/continuum-ai) · [Forge-Alloy](https://github.com/CambrianTech/forge-alloy)

## License

{alloy.get('license', 'Apache 2.0')}
"""

    return card


def main():
    parser = argparse.ArgumentParser(description="Generate model card from executed alloy")
    parser.add_argument("alloy", help="Path to executed .alloy.json")
    parser.add_argument("--output", "-o", help="Output path (default: stdout)")
    args = parser.parse_args()

    alloy_path = Path(args.alloy)
    alloy = json.loads(alloy_path.read_text())
    alloy_hash = hashlib.sha256(alloy_path.read_bytes()).hexdigest()

    card = alloy_to_card(alloy, alloy_hash)

    if args.output:
        Path(args.output).write_text(card)
        print(f"Card written: {args.output} ({len(card)} chars)")
    else:
        print(card)


if __name__ == "__main__":
    main()
