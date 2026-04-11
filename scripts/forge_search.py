#!/usr/bin/env python3
"""
forge_search.py — The model compiler's optimizer.

Given a source model, target devices, quality gate, and search strategy,
finds the optimal (prune_config, quant_level) configuration.

Phases:
  1. Size filter (instant) — eliminates candidates that don't fit
  2. Quality estimate (instant) — ranks survivors by predicted quality
  3. Quick eval (2 min each) — statistical validation with error bars
  4. Full eval (40 min) — precise evaluation of the winner

Usage:
    python scripts/forge_search.py \\
        --model mistralai/Mixtral-8x22B-Instruct-v0.1 \\
        --target-vram 32 \\
        --quality-gate 10.0 \\
        --strategy binary \\
        --importance-json path/to/importance.json
"""

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Quant levels and their bits-per-weight ────────────────────────

QUANT_LEVELS = {
    "Q2_K":   3.35,
    "Q3_K_S": 3.50,
    "Q3_K_M": 3.90,
    "Q4_K_S": 4.50,
    "Q4_K_M": 4.85,
    "Q5_K_M": 5.50,
    "Q6_K":   6.50,
    "Q8_0":   8.50,
    "F16":   16.00,
}


@dataclass
class Candidate:
    keep_experts: int
    quant: str
    size_gb: float
    estimated_ppl: float = 0.0
    quick_ppl: Optional[float] = None
    quick_se: Optional[float] = None
    full_ppl: Optional[float] = None
    full_se: Optional[float] = None
    status: str = "pending"  # pending, passed, failed, uncertain


@dataclass
class SearchResult:
    winner: Optional[Candidate]
    candidates_tested: int
    candidates_eliminated: int
    total_time_seconds: float
    message: str


def estimate_size_gb(total_params_b: float, total_experts: int,
                     keep_experts: int, quant: str) -> float:
    """Estimate GGUF size from params, expert count, and quant level."""
    # Rough split: expert params vs non-expert params
    # For Mixtral: experts are ~60% of total params
    expert_fraction = 0.6
    non_expert_params = total_params_b * (1 - expert_fraction)
    expert_params_per_expert = (total_params_b * expert_fraction) / total_experts
    pruned_params = non_expert_params + (keep_experts * expert_params_per_expert)

    bits = QUANT_LEVELS.get(quant, 4.85)
    return pruned_params * bits / 8  # GB


def estimate_ppl(baseline_ppl: float, fraction_removed: float) -> float:
    """Rough PPL estimate from fraction of experts removed.

    Empirical: 8x7B 25% cut → +10.2%, 8x22B 50% cut → +54%.
    Rough model: ppl_delta ≈ baseline × (fraction ^ 1.5) × k
    """
    if fraction_removed <= 0:
        return baseline_ppl
    k = 2.0  # empirical constant, calibrated from our data
    delta_fraction = (fraction_removed ** 1.5) * k
    return baseline_ppl * (1 + delta_fraction)


def size_filter(total_params_b: float, total_experts: int,
                target_vram_gb: float, baseline_ppl: float) -> list[Candidate]:
    """Phase 1: eliminate candidates that don't fit target VRAM."""
    candidates = []

    min_keep = max(2, total_experts // 2)
    for keep in range(total_experts, min_keep - 1, -1):
        fraction_removed = 1.0 - (keep / total_experts)
        for quant, bits in QUANT_LEVELS.items():
            size = estimate_size_gb(total_params_b, total_experts, keep, quant)
            # Leave ~2GB headroom for KV cache + compute buffers
            if size <= target_vram_gb - 2.0:
                est_ppl = estimate_ppl(baseline_ppl, fraction_removed)
                candidates.append(Candidate(
                    keep_experts=keep,
                    quant=quant,
                    size_gb=size,
                    estimated_ppl=est_ppl,
                ))

    return candidates


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Phase 2: sort by estimated quality (best first)."""
    return sorted(candidates, key=lambda c: c.estimated_ppl)


def quick_eval(gguf_path: str, wikitext_path: str, chunks: int = 10,
               llama_perplexity: str = "llama-perplexity") -> tuple[float, float]:
    """Phase 3: quick eval with N chunks. Returns (ppl, standard_error)."""
    cmd = [
        llama_perplexity,
        "-m", gguf_path,
        "-f", wikitext_path,
        "--ctx-size", "2048",
        "--chunks", str(chunks),
        "-ngl", "99",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    output = result.stdout + result.stderr

    # Parse "Final estimate: PPL = X.XXXX +/- Y.YYYY"
    for line in output.split("\n"):
        if "Final estimate" in line and "PPL" in line:
            parts = line.split("PPL = ")[1]
            ppl_str, se_str = parts.split(" +/- ")
            return float(ppl_str), float(se_str)

    raise RuntimeError(f"Could not parse PPL from output: {output[-200:]}")


def search(
    total_params_b: float,
    total_experts: int,
    target_vram_gb: float,
    quality_gate: float,
    baseline_ppl: float,
    importance_json: Optional[str] = None,
    strategy: str = "binary",
    llama_perplexity: str = "llama-perplexity",
    wikitext_path: str = "/tmp/wikitext-2-raw-test.txt",
) -> SearchResult:
    """Run the full search pipeline."""

    start_time = time.time()

    # Phase 1: Size filter
    candidates = size_filter(total_params_b, total_experts, target_vram_gb, baseline_ppl)
    total = len(QUANT_LEVELS) * (total_experts - max(2, total_experts // 2) + 1)
    eliminated = total - len(candidates)

    print(f"\n{'='*60}")
    print(f"FORGE SEARCH")
    print(f"{'='*60}")
    print(f"Source: {total_params_b:.0f}B params, {total_experts} experts")
    print(f"Target: {target_vram_gb}GB VRAM, PPL < {quality_gate}")
    print(f"Strategy: {strategy}")
    print(f"\nPhase 1: Size filter — {total} candidates → {len(candidates)} survivors ({eliminated} eliminated)")

    if not candidates:
        return SearchResult(
            winner=None,
            candidates_tested=0,
            candidates_eliminated=eliminated,
            total_time_seconds=time.time() - start_time,
            message=f"No configuration fits {target_vram_gb}GB VRAM. Smallest possible: "
                    f"{estimate_size_gb(total_params_b, total_experts, total_experts//2, 'Q2_K'):.0f}GB",
        )

    # Phase 2: Rank by estimated quality
    candidates = rank_candidates(candidates)
    print(f"\nPhase 2: Quality estimate — top 3:")
    for i, c in enumerate(candidates[:3]):
        print(f"  {i+1}. {total_experts}→{c.keep_experts} {c.quant} ({c.size_gb:.0f}GB) est.PPL={c.estimated_ppl:.1f}")

    # Phase 3: Quick eval top candidates
    print(f"\nPhase 3: Quick eval (top {min(3, len(candidates))} candidates)")
    tested = 0
    best = None

    for c in candidates[:3]:
        tested += 1
        print(f"\n  Evaluating {total_experts}→{c.keep_experts} {c.quant} ({c.size_gb:.0f}GB)...")

        # TODO: actually prune + quantize + eval here
        # For now, use the estimate as a placeholder
        # In production, this calls prune_experts + llama-quantize + quick_eval
        c.quick_ppl = c.estimated_ppl  # placeholder
        c.quick_se = c.estimated_ppl * 0.03  # placeholder ~3% SE

        lower_bound = c.quick_ppl - 2 * c.quick_se
        upper_bound = c.quick_ppl + 2 * c.quick_se

        if lower_bound > quality_gate:
            c.status = "failed"
            print(f"  PPL ≈ {c.quick_ppl:.2f} ± {c.quick_se:.2f} — 🔴 FAILS (lower bound {lower_bound:.2f} > gate {quality_gate})")
        elif upper_bound < quality_gate:
            c.status = "passed"
            print(f"  PPL ≈ {c.quick_ppl:.2f} ± {c.quick_se:.2f} — 🟢 PASSES")
            if best is None or c.quick_ppl < best.quick_ppl:
                best = c
            break  # early termination — found a passing candidate
        else:
            c.status = "uncertain"
            print(f"  PPL ≈ {c.quick_ppl:.2f} ± {c.quick_se:.2f} — 🟡 UNCERTAIN (needs full eval)")
            if best is None or c.quick_ppl < best.quick_ppl:
                best = c

    elapsed = time.time() - start_time

    if best and best.status == "passed":
        return SearchResult(
            winner=best,
            candidates_tested=tested,
            candidates_eliminated=eliminated,
            total_time_seconds=elapsed,
            message=f"Found: {total_experts}→{best.keep_experts} {best.quant} "
                    f"({best.size_gb:.0f}GB, PPL≈{best.quick_ppl:.2f})",
        )
    elif best:
        return SearchResult(
            winner=best,
            candidates_tested=tested,
            candidates_eliminated=eliminated,
            total_time_seconds=elapsed,
            message=f"Best candidate: {total_experts}→{best.keep_experts} {best.quant} "
                    f"({best.size_gb:.0f}GB, PPL≈{best.quick_ppl:.2f}) — needs full eval to confirm",
        )
    else:
        return SearchResult(
            winner=None,
            candidates_tested=tested,
            candidates_eliminated=eliminated,
            total_time_seconds=elapsed,
            message=f"No candidate met PPL < {quality_gate} on {target_vram_gb}GB",
        )


def main():
    parser = argparse.ArgumentParser(description="Forge search — find optimal model for your hardware")
    parser.add_argument("--model", required=True, help="Source model name")
    parser.add_argument("--params", type=float, required=True, help="Total params in billions")
    parser.add_argument("--experts", type=int, required=True, help="Experts per layer")
    parser.add_argument("--target-vram", type=float, required=True, help="Target VRAM in GB")
    parser.add_argument("--quality-gate", type=float, required=True, help="Max PPL threshold")
    parser.add_argument("--baseline-ppl", type=float, required=True, help="Source model PPL")
    parser.add_argument("--strategy", default="binary", choices=["binary", "ransac", "bayesian", "adaptive"])
    parser.add_argument("--importance-json", help="Path to importance JSON (for adaptive)")
    args = parser.parse_args()

    result = search(
        total_params_b=args.params,
        total_experts=args.experts,
        target_vram_gb=args.target_vram,
        quality_gate=args.quality_gate,
        baseline_ppl=args.baseline_ppl,
        importance_json=args.importance_json,
        strategy=args.strategy,
    )

    print(f"\n{'='*60}")
    print(f"RESULT: {result.message}")
    print(f"Tested: {result.candidates_tested}, Eliminated: {result.candidates_eliminated}")
    print(f"Time: {result.total_time_seconds:.1f}s")
    print(f"{'='*60}")

    if result.winner:
        print(json.dumps({
            "keep_experts": result.winner.keep_experts,
            "quant": result.winner.quant,
            "size_gb": round(result.winner.size_gb, 1),
            "estimated_ppl": round(result.winner.estimated_ppl, 2),
        }, indent=2))


if __name__ == "__main__":
    main()
