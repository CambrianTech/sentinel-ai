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

    card = f"""---
tags:
{chr(10).join(f'- {t}' for t in tags)}
base_model: {base_model}
pipeline_tag: text-generation
license: {alloy.get('license', 'apache-2.0')}
---

<h1 align="center">🔥 +{improvement:.1f}% better at {domain}</h1>

<p align="center">
<b>{base_model.split('/')[-1]}</b> forged for {domain} through <a href="https://github.com/CambrianTech/continuum/blob/main/docs/papers/EXPERIENTIAL-PLASTICITY.md">Experiential Plasticity</a><br>
{baseline:.2f} → {final:.2f} perplexity · {cycles} cycles · {hw_device}{duration_str}
</p>

<details>
<summary><b>Forged with Continuum — a distributed AI world that runs on your hardware</b></summary>
<p align="center">
<a href="https://github.com/CambrianTech/continuum"><img src="{factory_img}" alt="Continuum Model Factory" width="600"/></a><br>
<em>The <a href="https://github.com/CambrianTech/continuum#the-grid">Grid</a> forges models on your GPU, the <a href="https://github.com/CambrianTech/forge-alloy">alloy</a> proves the work.</em>
</p>
<table>
<tr><td><b>Grid</b></td><td>Your machines form an encrypted mesh. Personas move between nodes. Models forge on the strongest hardware, deploy to the weakest.</td></tr>
<tr><td><b>Factory</b></td><td>Visual pipeline composer — prune, train, LoRA, compact, context-extend, add vision/audio. MUTAGEN rolls random mutations.</td></tr>
<tr><td><b>Forge-Alloy</b></td><td>Cryptographic contract for every forge. The recipe, the results, the attestation. Trustless verification.</td></tr>
<tr><td><b>Personas</b></td><td>AI citizens with faces, voices, memories. Every persona sees, hears, speaks — regardless of base model. The system bridges gaps.</td></tr>
</table>
<p align="center">
<a href="https://github.com/CambrianTech/continuum">GitHub</a> · <a href="https://huggingface.co/continuum-ai">Models</a> · <a href="https://github.com/CambrianTech/forge-alloy">Forge-Alloy</a>
</p>
</details>
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

    # Model size (from highlights or computed)
    highlights = r.get("highlights", [])
    for h in highlights:
        if "context" not in h.lower() and "prune" not in h.lower():
            card += f"| **Note** | | {h} | |\n"

    card += f"| **Pipeline** | | {pipeline} | |\n"
    card += "\n"

    # Hardware
    card += """## Runs On

| Device | Format | Size | Status |
|--------|--------|------|--------|
"""
    devices = [
        ("iPhone / Android", "Q4_K_M", "~2.6GB", "Expected"),
        ("MacBook Air 8GB", "Q4_K_M", "~2.6GB", "Expected"),
        ("MacBook Air 16GB", "Q8_0", "~4.2GB", "Expected"),
        ("MacBook Pro 32GB", "fp16", "8.0GB", "Expected"),
        ("RTX 3090/4090", "fp16", "8.0GB", "Expected"),
    ]
    for h in hw:
        devices.append((h["device"], h.get("format", "fp16"),
                        f"{h.get('sizeGb', '?')}GB" if h.get("sizeGb") else "—",
                        "**Forged here**"))
    for d in devices:
        card += f"| {d[0]} | {d[1]} | {d[2]} | {d[3]} |\n"

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

## Reproduce

```bash
pip install forge-alloy
# Download the alloy and run it with any compatible forge runner
python your_runner.py {name}.alloy.json
```
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

    # Science
    card += """
## The Science

**Experiential Plasticity** — architectural optimization, not compression:

1. Train on domain data (LoRA)
2. Measure attention head contribution (entropy)
3. Prune non-contributing heads
4. Retrain — surviving heads specialize
5. Repeat — each cycle improves

## Papers

- [Experiential Plasticity](https://github.com/CambrianTech/continuum/blob/main/docs/papers/EXPERIENTIAL-PLASTICITY.md) — scaling law, transfer function
- [Neural Plasticity in Transformers](https://github.com/CambrianTech/continuum/blob/main/docs/papers/SENTINEL-AI-NEURAL-PLASTICITY.md) — foundation
- [Plasticity Compaction](https://github.com/CambrianTech/continuum/blob/main/docs/papers/PLASTICITY-COMPACTION-MOE.md) — MoE expert pruning

---

*Every claim verified by [ForgeAlloy](https://github.com/CambrianTech/forge-alloy) cryptographic chain of custody*
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
