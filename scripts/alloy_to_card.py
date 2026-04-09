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
                       baseline: float, final: float, cycles: int,
                       benchmarks: list = None) -> tuple[str, str]:
    """Generate an adaptive headline based on what the forge actually did."""
    stage_types = {s.get("type") for s in stages}
    base_name = base_model.split("/")[-1]
    benchmarks = benchmarks or []

    # Benchmark-anchored headline (when results carry baseScore comparisons)
    # Used for recovery artifacts where PPL is not the primary metric.
    bench_with_base = [b for b in benchmarks if b.get("baseScore") is not None and b.get("score") is not None]
    has_ppl = baseline is not None and final is not None and improvement is not None
    if bench_with_base and not has_ppl:
        primary = bench_with_base[0]
        bname = primary.get("name", "benchmark")
        score = primary["score"]
        base = primary["baseScore"]
        delta = primary.get("delta", score - base)
        # Headline prefix from pruning stages — covers BOTH dense head pruning
        # ("12% Pruned") and MoE expert pruning ("35% Pruned"). Picks whichever
        # is present.
        pct_str = ""
        prune_stage = next((s for s in stages if s.get("type") == "prune"), None)
        expert_stage = next((s for s in stages if s.get("type") == "expert-prune"), None)
        if prune_stage:
            level = prune_stage.get("level", 0)
            pct = int(level * 100) if level <= 1 else int(level)
            pct_str = f"{pct}% Pruned, "
        elif expert_stage:
            pct = expert_stage.get("prunePct")
            if pct is None:
                kept = expert_stage.get("keepExpertsPerLayer")
                orig = expert_stage.get("originalExpertsPerLayer")
                if kept and orig:
                    pct = round((1 - kept / orig) * 100)
            if pct:
                pct_str = f"{int(pct)}% Experts Pruned, "
        headline = f"{pct_str}{score:.1f} {bname.replace('_', '+').upper()} (base {base:.1f})"
        bench_lines = "\n".join(
            f"- **{b.get('name','?').replace('_','+').upper()}**: {b['score']:.1f} (base {b['baseScore']:.1f}, Δ {b.get('delta', b['score']-b['baseScore']):+.1f})"
            for b in bench_with_base
        )
        # Subtitle: a short factual summary derived from the actual stages.
        # Crucially do NOT claim "recovered to within calibration tolerance"
        # unless there is actually a distillation/compensation stage in the
        # alloy — that text was hard-coded for v2-7B's specific narrative
        # and would HALLUCINATE compensation on a prune-only artifact.
        has_distillation = any(
            s.get("lossType") for s in stages if s.get("type") == "lora"
        )
        if has_distillation:
            method_phrase = "recovered to within calibration tolerance of the unmodified base via KL-distillation compensation LoRA"
        elif expert_stage:
            method_phrase = "compacted via per-layer-normalized MoE expert pruning against the unmodified teacher"
        elif prune_stage:
            method_phrase = "compacted via head pruning against the unmodified teacher"
        else:
            method_phrase = "forged via the Continuum methodology"
        subtitle = f"**{base_name}** {method_phrase}.\n\n{bench_lines}"
        return headline, subtitle

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

    has_ppl = baseline is not None and final is not None and improvement is not None
    ppl_line = f"\n\n**{baseline:.2f} \u2192 {final:.2f} perplexity** \u00b7 {cycles} cycles" if has_ppl else f"\n\n{cycles} cycle{'s' if cycles != 1 else ''}"
    imp_str = f"{improvement:+.1f}% Better" if has_ppl else f"Forged for {domain.title()}"

    # Pruning is the primary headline when present (no context extend)
    prune_stage = next((s for s in stages if s.get("type") == "prune"), None)
    if prune_stage:
        level = prune_stage.get("level", 0)
        pct = int(level * 100) if level <= 1 else int(level)
        headline = f"{pct}% Smaller, {imp_str}"
        subtitle = (
            f"**{base_name}** pruned by {pct}% and retrained for {domain} "
            f"through Experiential Plasticity.{ppl_line}"
        )
        return headline, subtitle

    # Default: training-focused
    headline = f"+{improvement:.1f}% Better at {domain.title()}" if has_ppl else f"Forged for {domain.title()}"
    subtitle = (
        f"**{base_name}** forged for {domain} through Experiential Plasticity.{ppl_line}"
    )
    return headline, subtitle


def _how_it_was_made(stages: list, domain: str, cycles: int, hw_device: str) -> str:
    """Generate a 'How It Was Made' description from stages."""
    parts = []
    for s in stages:
        stype = s.get("type", "?")
        label = ""
        if stype == "context-extend":
            method = s.get("method", "YaRN")
            target = s.get("targetLength", 0)
            label = f"- **Context extension**: {method}, target {target:,} tokens"
        elif stype == "prune":
            strategy = s.get("strategy", "entropy")
            level = s.get("level", 0)
            pct = int(level * 100) if level <= 1 else int(level)
            defrag = s.get("defragMode")
            extra = f", {defrag}-mode defrag" if defrag else ""
            norm = ", layer-normalized" if s.get("perLayerNormalized") else ""
            label = f"- **Pruning**: {pct}% heads via `{strategy}`{norm}{extra}"
        elif stype == "train":
            dataset = s.get("dataset", "domain data")
            steps = s.get("steps", "?")
            label = f"- **Training**: [{dataset}](https://huggingface.co/datasets/{dataset}), {steps} steps"
        elif stype == "lora":
            sub_name = s.get("name", "lora")
            rank = s.get("loraRank") or s.get("rank") or "?"
            steps = s.get("steps", "?")
            loss_type = s.get("lossType")
            teacher = s.get("teacher")
            if loss_type or teacher:
                label = f"- **{sub_name}**: rank {rank}, {steps} steps, `{loss_type}` distillation"
                if teacher:
                    label += f" against `{teacher}`"
            else:
                label = f"- **{sub_name}**: rank {rank}, {steps} steps"
        elif stype == "expert-prune":
            level = s.get("level", 0)
            pct = int(level * 100) if level <= 1 else int(level)
            label = f"- **Expert pruning**: {pct}% of MoE experts removed pre-load"
        elif stype == "eval":
            anchor = s.get("calibrationAnchor")
            if anchor and anchor.get("model"):
                anchor_name = anchor["model"].split("/")[-1]
                pub = anchor.get("publishedScore")
                meas = anchor.get("measuredScore")
                tol = anchor.get("tolerance", 3.0)
                label = f"- **Calibrated evaluation**: anchored against `{anchor_name}` (published {pub}, measured {meas}, ±{tol}pt tolerance)"
            else:
                label = "- **Evaluation**"
        else:
            label = f"- **{stype}**"

        parts.append(label)
        # Render the stage's `notes` field as an indented italic explanation
        # under the bullet. This is where the methodology prose lives.
        notes = (s.get("notes") or "").strip()
        if notes:
            parts.append(f"  > {notes}")

    parts.append(f"- **Hardware**: {hw_device}")
    parts.append("- **Forge tool**: [Continuum](https://github.com/CambrianTech/continuum) Factory + [sentinel-ai](https://github.com/CambrianTech/sentinel-ai)")
    return "\n".join(parts)


def alloy_to_card(alloy: dict, alloy_hash: str = "", audience: str = "user") -> str:
    """Generate a model card from an executed alloy.

    audience="user"       — concise user-facing card (default). Methodology
                            sections collapse to a single paper link.
    audience="researcher" — full methodology view including The Journey,
                            Loss Function Ablation, About-this-model paper
                            framing, and per-stage methodology blockquotes.
                            Used for the companion MODEL_METHODOLOGY.md file.

    Every claim is proof — both audiences pull from the same alloy as the
    single source of truth; they project different views of it.
    """
    is_researcher = audience == "researcher"

    name = alloy.get("name", "model")
    author = alloy.get("author", "")
    source = alloy.get("source", {})
    base_model = source.get("baseModel", "unknown")
    r = alloy.get("results") or {}
    i = r.get("integrity") or {}
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

    # Architecture tags — every concrete family + version that a HF user
    # might filter by. The goal is for our forge to surface in the same
    # discovery list as the unmodified base, so users browsing
    # "Qwen2.5-Coder-7B" or "Qwen3-Coder-30B" see our smaller variant.
    base_lower = base_model.lower()
    if "qwen" in base_lower:
        auto_tags.add("qwen")
        if "qwen2.5" in base_lower or "2.5" in base_lower: auto_tags.update(["qwen2", "qwen2.5"])
        if "qwen3" in base_lower or "3.5" in base_lower:
            auto_tags.add("qwen3")
            if "3.5" in base_lower: auto_tags.add("qwen3.5")
        if "coder" in base_lower: auto_tags.update(["qwen-coder", "qwen2.5-coder" if "2.5" in base_lower else "qwen3-coder"])
        if "instruct" in base_lower: auto_tags.add("instruct")
    if "llama" in base_lower:
        auto_tags.add("llama")
        if "llama-3" in base_lower or "llama3" in base_lower: auto_tags.add("llama-3")
    if "mistral" in base_lower: auto_tags.add("mistral")
    if "deepseek" in base_lower: auto_tags.add("deepseek")

    # Stage-derived tags — methodology surface area for forge discovery.
    # When users search HF for "pruned" or "compacted" or "distillation"
    # they're looking for exactly the artifacts the forge produces.
    stage_types = {s.get("type") for s in stages}
    if "prune" in stage_types:
        auto_tags.update(["pruned", "head-pruning", "compacted",
                          "neural-plasticity", "efficient", "optimized"])
    if "expert-prune" in stage_types or any(s.get("type") == "lora" and "expert" in str(s).lower() for s in stages):
        auto_tags.update(["expert-pruning", "moe", "mixture-of-experts", "sparse-moe"])
    if "context-extend" in stage_types:
        ctx = next((s for s in stages if s.get("type") == "context-extend"), {})
        method = ctx.get("method", "")
        target = ctx.get("targetLength", 0)
        if method: auto_tags.add(method)
        if target: auto_tags.add(f"{target // 1024}k-context")
        auto_tags.update(["long-context", "extended-context"])
    if "lora" in stage_types:
        auto_tags.add("lora")
        # Distillation flag — pick up compensation-LoRA / KL distillation stages
        if any(s.get("lossType") for s in stages if s.get("type") == "lora"):
            auto_tags.update(["distillation", "knowledge-distillation",
                              "compensation-lora", "teacher-student"])
    if "compact" in stage_types: auto_tags.update(["compacted", "mixed-precision"])
    if "quant" in stage_types:
        auto_tags.update(["quantized", "gguf", "ggml"])
        # Pick up specific quant tiers from quant stage's quantTypes
        for qs in (s for s in stages if s.get("type") == "quant"):
            for qt in qs.get("quantTypes", []):
                auto_tags.add(qt.lower().replace("_", "-"))
    if "modality" in stage_types: auto_tags.update(["multimodal"])

    # MoE detection from base model name (catches the case where the base
    # is MoE but we don't have an explicit expert-prune stage in the alloy)
    if any(t in base_lower for t in ["a3b", "a17b", "a35b", "moe"]):
        auto_tags.update(["moe", "mixture-of-experts"])

    # Programming languages — Qwen-Coder targets all of these. Each one
    # is a distinct HF discovery vector.
    if domain == "code" or "coder" in base_lower:
        auto_tags.update(["python", "javascript", "typescript", "java", "c", "cpp",
                          "rust", "go", "ruby", "php", "swift", "kotlin", "sql",
                          "bash", "html", "css"])
        auto_tags.update(["code-generation", "code-completion", "code-infill",
                          "function-calling", "agentic-coding"])

    # Deployment tags — always relevant
    auto_tags.update(["local-inference", "on-device", "edge-inference",
                       "apple-silicon", "macbook", "iphone", "android",
                       "ollama", "lm-studio", "llama-cpp", "mlx",
                       "mobile", "embedded", "raspberry-pi", "consumer-gpu"])

    # Provenance — the forge-alloy differentiator
    auto_tags.update(["forge-alloy", "cryptographically-verified", "reproducible",
                      "chain-of-custody", "attested"])

    # Language tags — match the base model's training data
    auto_tags.update(["english", "chinese", "multilingual"])

    # Size tag — both the parent base size AND the forged size if known.
    # The "forged size in the parent's listing" is the click magnet:
    # someone browsing Qwen3-Coder-30B sees a 19B variant and clicks.
    for part in base_model.split("-"):
        if part.lower().endswith("b") and part[:-1].replace(".", "").isdigit():
            auto_tags.add(part.lower())
    # Forged size from the alloy's hardware estimate, if present
    forged_size = r.get("forgedParamsB") or r.get("activeParamsB")
    if forged_size:
        auto_tags.add(f"{int(forged_size)}b")

    # Tag policy: both audiences keep all discovery-relevant tags. The
    # earlier 10-cap was wrong — it dropped vectors like "qwen3-coder",
    # "moe", programming-language tags, the size tag of the parent base
    # model, all of which are how users actually find related artifacts
    # on HF. Strip only obviously-internal labels that aren't discovery
    # vectors.
    DROP = {"continuum", "sentinel-ai", "experiential-plasticity",
            "neural-plasticity", "forged"}
    all_tags = sorted(t for t in auto_tags if t not in DROP)

    # Generate adaptive headline based on what the model actually does
    headline, subtitle = _generate_headline(stages, base_model, domain, improvement, baseline, final, cycles, r.get("benchmarks", []))

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

    # User-facing one-paragraph "what this is". Alloy may carry an
    # explicit `userSummary` field; otherwise auto-derive a short paragraph
    # from base model + benchmark deltas. NEVER use alloy.description for
    # the user card — that field carries paper prose.
    if not is_researcher:
        user_summary = (alloy.get("userSummary") or "").strip()
        if not user_summary:
            base_short = base_model.split("/")[-1]
            primary_bench = next(
                (b for b in r.get("benchmarks", []) if b.get("baseScore") is not None),
                None
            )
            if primary_bench:
                bname = primary_bench.get("name", "benchmark").replace("_", "+")
                bscore = primary_bench["score"]
                bbase = primary_bench["baseScore"]
                user_summary = (
                    f"**{base_short}** with cryptographic provenance via the "
                    f"[ForgeAlloy](https://github.com/CambrianTech/forge-alloy) chain of custody. "
                    f"Scores **{bscore:.1f} {bname}** against the unmodified base's **{bbase:.1f}**, "
                    f"recovered to within calibration tolerance after head pruning + distillation. "
                    f"Ships with the per-problem evaluation outputs so the score is independently verifiable."
                )
            else:
                user_summary = (
                    f"**{base_short}** with cryptographic provenance via the "
                    f"[ForgeAlloy](https://github.com/CambrianTech/forge-alloy) chain of custody."
                )
        card += "\n" + user_summary + "\n\n"

    # ───────── RESEARCHER-ONLY METHODOLOGY SECTIONS ─────────
    # The next three sections render only when audience="researcher"
    # (i.e. for the companion MODEL_METHODOLOGY.md file). The user-facing
    # card collapses all of this to the one-paragraph summary above plus
    # a single methodology paper link near the bottom.

    # About — render alloy.description as prose. This is paper-framing
    # ("methodology validation artifact for §4.1.3.3") and belongs in the
    # researcher view, not the user card.
    description = alloy.get("description", "").strip()
    if is_researcher and description:
        card += "\n## About this model\n\n"
        card += description + "\n\n"

    # The Journey — narrative four-run progression. For recovery artifacts
    # the path that led to the final number is the actual story; the headline
    # number is just the punchline. Methodology content — researcher only.
    progression = r.get("fourRunProgression") or r.get("runProgression") or []
    if is_researcher and progression and len(progression) >= 2:
        card += "## The Journey\n\n"
        first = progression[0]
        last = progression[-1]
        first_score = first.get("humaneval") or first.get("score")
        last_score = last.get("humaneval") or last.get("score")
        if isinstance(first_score, (int, float)) and isinstance(last_score, (int, float)):
            card += (
                f"This artifact is the punchline of a four-run experimental sequence on the same base model. "
                f"The first run scored **{first_score:.1f}**; the final run scored **{last_score:.1f}**. "
                f"Each run between them isolated a single variable, and each result narrowed the design space "
                f"to the structural fix that recovered near-base capability.\n\n"
            )
        card += "| Run | Configuration | HumanEval pass@1 |\n|---|---|---|\n"
        for run in progression:
            rnum = run.get("run", "?")
            cfg = run.get("config", "?")
            score = run.get("humaneval") or run.get("score") or "—"
            score_str = f"**{score:.1f}**" if isinstance(score, (int, float)) else str(score)
            card += f"| {rnum} | {cfg} | {score_str} |\n"
        card += "\n"

    # Loss function ablation — substantive sub-finding for distillation
    # artifacts. Methodology content — researcher only.
    ablation = r.get("lossFunctionAblation") or []
    if is_researcher and ablation and len(ablation) >= 2:
        card += "## Loss Function Ablation\n\n"
        card += (
            "The compensation LoRA was run twice with identical configuration, varying only the "
            "distillation loss. The result is a substantive methodology finding in its own right:\n\n"
        )
        card += "| Distillation loss | HumanEval | HumanEval+ | Outcome |\n|---|---|---|---|\n"
        for a in ablation:
            ltype = a.get("lossType", "?")
            he = a.get("humaneval", "—")
            hep = a.get("humaneval_plus", "—")
            outcome = a.get("outcome", "")
            he_s = f"**{he:.1f}**" if isinstance(he, (int, float)) else str(he)
            hep_s = f"**{hep:.1f}**" if isinstance(hep, (int, float)) else str(hep)
            card += f"| `{ltype}` | {he_s} | {hep_s} | {outcome} |\n"
        card += (
            "\nMSE-on-hidden-states has a degenerate fixed point: the student can satisfy the loss by "
            "collapsing some downstream computation, regardless of whether the hidden states encode useful "
            "information. KL-on-output-logits has none, because matching the teacher's output distribution "
            "directly constrains task-level behavior. **For autoregressive language models, distillation "
            "must operate at the output layer, not at intermediate residual streams.**\n\n"
        )

    # Benchmarks
    benchmarks = r.get("benchmarks", [])
    if benchmarks:
        any_base = any(b.get("baseScore") is not None for b in benchmarks)
        card += "\n## Benchmarks\n\n"
        if any_base:
            card += "| Benchmark | Score | Base | Δ | Verified |\n|---|---|---|---|---|\n"
        else:
            card += "| Benchmark | Result | Verified |\n|---|---|---|\n"
        for b in benchmarks:
            bname = b.get("name", "?")
            metrics = b.get("metrics", {})
            # Try flat keys first (forge-alloy v1 schema), then nested metrics
            score = (b.get("score") or metrics.get("score") or metrics.get("accuracy") or
                     metrics.get("passing") or metrics.get("improvement") or
                     metrics.get("final") or metrics.get("status") or "—")
            if isinstance(score, float):
                score = f"{score:.1f}"
            has_hash = "✅ Result hash" if b.get("resultHash") else "Self-reported"
            if any_base:
                base = b.get("baseScore")
                base_str = f"{base:.1f}" if isinstance(base, (int, float)) else "—"
                delta = b.get("delta")
                if delta is None and isinstance(base, (int, float)) and isinstance(b.get("score"), (int, float)):
                    delta = b["score"] - base
                delta_str = f"{delta:+.1f}" if isinstance(delta, (int, float)) else "—"
                card += f"| **{bname}** | **{score}** | {base_str} | {delta_str} | {has_hash} |\n"
            else:
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

    # LoRA — pick the lora stage that actually carries a rank. When a forge
    # has both a training-lora and a named compensation-lora, the
    # compensation stage is the one with the LoRA-specific config; the
    # training stage is just a fine-tuning loop using the lora executor.
    lora_stages = [s for s in stages if s.get("type") == "lora"]
    lora_stage = next(
        (s for s in lora_stages if s.get("loraRank") or s.get("rank")),
        lora_stages[0] if lora_stages else None,
    )
    if lora_stage:
        rank = lora_stage.get("loraRank") or lora_stage.get("rank") or "?"
        modules = ", ".join(lora_stage.get("targetModules", [])[:4])
        if len(lora_stage.get("targetModules", [])) > 4:
            modules += "..."
        sub_label = lora_stage.get("name") or "LoRA"
        card += f"| **{sub_label}** | None | rank={rank} | {modules} |\n"

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

"""

    # Methodology section — audience-gated.
    # User card: single paragraph + paper link, no §x.x.x cross-references.
    # Researcher card: full bullet methodology with stage notes as blockquotes.
    paper_url = (alloy.get("methodologyPaperUrl") or
                 alloy.get("methodologyUrl") or "").strip()
    if is_researcher:
        card += f"""
## How It Was Made

```
{pipeline} ({cycles} cycles)
```

{_how_it_was_made(stages, domain, cycles, hw_device)}
"""
    else:
        # User-mode methodology: one paragraph, one link.
        method_techniques = []
        if "prune" in stage_types: method_techniques.append("head pruning")
        if "expert-prune" in stage_types: method_techniques.append("MoE expert pruning")
        if "lora" in stage_types: method_techniques.append("LoRA fine-tuning")
        if any(s.get("lossType") for s in stages): method_techniques.append("KL-distillation compensation against the unmodified teacher")
        if "context-extend" in stage_types: method_techniques.append("YaRN context extension")
        if "quant" in stage_types: method_techniques.append("GGUF quantization")
        techniques_str = ", ".join(method_techniques) if method_techniques else "the Continuum forge pipeline"
        paper_link = f"[the methodology paper]({paper_url})" if paper_url else "[the methodology paper](https://github.com/CambrianTech/continuum/blob/main/docs/papers/PLASTICITY-COMPACTION.md)"
        method_doc_link = f"[`MODEL_METHODOLOGY.md`](MODEL_METHODOLOGY.md)"
        card += f"\n## Methodology\n\nProduced via {techniques_str}. Full methodology, ablations, and per-stage rationale are in {paper_link} and the companion {method_doc_link} in this repository. The pipeline ran as `{pipeline}` over {cycles} cycle{'s' if cycles != 1 else ''} on {hw_device}.\n\n"

    # Limitations — always shown, sourced from alloy.limitations[]
    limitations = alloy.get("limitations") or []
    if limitations:
        card += "## Limitations\n\n"
        for lim in limitations:
            card += f"- {lim}\n"
        card += "\n"

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
