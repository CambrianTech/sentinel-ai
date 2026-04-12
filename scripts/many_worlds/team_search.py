"""team_search.py — Find the optimal model team for Many-Worlds substrate training.

Before training a substrate, MEASURE which model pair has the most
complementary knowledge on the target benchmark. The pair that disagrees
the most has the most opportunity for substrate transfer.

This is the Many-Worlds equivalent of the activation profile in pruning:
one tells you which experts to keep, the other tells you which models
to combine.

Usage:
    python -m many_worlds.team_search \
        --candidates Qwen/Qwen3-1.7B,Qwen/Qwen3-4B,microsoft/phi-2,microsoft/phi-3-mini-4k-instruct \
        --benchmark gsm8k \
        --num-problems 50 \
        --output team_search_results.json

Or as a module:
    from many_worlds.team_search import search_team
    best_pair, matrix = search_team(candidates, benchmark, num_problems)
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelScore:
    """One model's results on the benchmark."""
    name: str
    correct: list[bool]
    score: float
    eval_time: float


@dataclass
class PairDivergence:
    """Divergence analysis for a model pair."""
    model_a: str
    model_b: str
    score_a: int
    score_b: int
    both_right: int
    a_only: int
    b_only: int
    both_wrong: int

    @property
    def complementary(self) -> int:
        """Problems where ONE model knows and the other doesn't."""
        return self.a_only + self.b_only

    @property
    def combined_potential(self) -> int:
        """Problems the PAIR could solve if substrate transfers perfectly."""
        return self.both_right + self.a_only + self.b_only

    @property
    def diversity_score(self) -> float:
        """0-1 score: how different are the models' knowledge?
        1.0 = perfectly complementary (no overlap in correct answers)
        0.0 = identical knowledge (same answers right and wrong)
        """
        total_correct = self.score_a + self.score_b
        if total_correct == 0:
            return 0.0
        return self.complementary / total_correct


def load_benchmark(name: str, num_problems: int):
    """Load benchmark problems. Returns list of (prompt, ground_truth) tuples."""
    if name == "gsm8k":
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split=f"test[:{num_problems}]")
        problems = []
        for row in ds:
            prompt = f"Question: {row['question']}\nAnswer:"
            gt = row["answer"].split("####")[-1].strip()
            problems.append({"prompt": prompt, "gt": gt, "question": row["question"]})
        return problems
    else:
        raise ValueError(f"Unknown benchmark: {name}. Supported: gsm8k")


def check_answer(generated: str, ground_truth: str) -> bool:
    """Check if the generated text contains the ground truth answer."""
    return ground_truth in generated


def evaluate_model(
    model_name: str,
    problems: list[dict],
    device: str = "cuda",
    max_new_tokens: int = 200,
) -> ModelScore:
    """Run one model on all problems and return results."""
    print(f"\n  Evaluating: {model_name}")
    start = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.pad_token or tok.eos_token

    correct = []
    for i, prob in enumerate(problems):
        inputs = tok(prob["prompt"], return_tensors="pt",
                     truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        correct.append(check_answer(gen, prob["gt"]))

        if (i + 1) % 10 == 0:
            pct = sum(correct) / (i + 1) * 100
            print(f"    [{i+1}/{len(problems)}] {sum(correct)}/{i+1} ({pct:.0f}%)")

    elapsed = time.time() - start
    score = sum(correct) / len(correct)
    print(f"    Final: {sum(correct)}/{len(correct)} = {score*100:.0f}% ({elapsed:.0f}s)")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return ModelScore(name=model_name, correct=correct, score=score, eval_time=elapsed)


def compute_divergence(a: ModelScore, b: ModelScore) -> PairDivergence:
    """Compute divergence between two models' results."""
    both_right = sum(1 for x, y in zip(a.correct, b.correct) if x and y)
    a_only = sum(1 for x, y in zip(a.correct, b.correct) if x and not y)
    b_only = sum(1 for x, y in zip(a.correct, b.correct) if not x and y)
    both_wrong = sum(1 for x, y in zip(a.correct, b.correct) if not x and not y)

    return PairDivergence(
        model_a=a.name, model_b=b.name,
        score_a=sum(a.correct), score_b=sum(b.correct),
        both_right=both_right, a_only=a_only,
        b_only=b_only, both_wrong=both_wrong,
    )


def search_team(
    candidates: list[str],
    benchmark: str = "gsm8k",
    num_problems: int = 50,
    device: str = "cuda",
) -> tuple[PairDivergence, list[PairDivergence]]:
    """Search for the optimal model pair.

    Returns:
        (best_pair, all_pairs) — the pair with highest complementary count,
        and the full divergence matrix for analysis.
    """
    problems = load_benchmark(benchmark, num_problems)
    print(f"Benchmark: {benchmark} ({len(problems)} problems)")
    print(f"Candidates: {len(candidates)} models")

    # Evaluate each model
    scores = {}
    for name in candidates:
        try:
            scores[name] = evaluate_model(name, problems, device)
        except Exception as e:
            print(f"  SKIP {name}: {e}")

    # Compute all pairwise divergences
    names = list(scores.keys())
    all_pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            div = compute_divergence(scores[a], scores[b])
            all_pairs.append(div)

    # Sort by complementary count (descending)
    all_pairs.sort(key=lambda d: d.complementary, reverse=True)

    # Print matrix
    print(f"\n{'='*70}")
    print(f"DIVERGENCE MATRIX — {benchmark} ({num_problems} problems)")
    print(f"{'='*70}")

    for d in all_pairs:
        short_a = d.model_a.split("/")[-1][:18]
        short_b = d.model_b.split("/")[-1][:18]
        print(f"  {short_a:18s} + {short_b:18s} | "
              f"A={d.score_a:2d} B={d.score_b:2d} | "
              f"comp={d.complementary:2d} combined={d.combined_potential:2d} "
              f"diversity={d.diversity_score:.2f}")

    best = all_pairs[0]
    print(f"\nBEST PAIR: {best.model_a.split('/')[-1]} + {best.model_b.split('/')[-1]}")
    print(f"  Complementary: {best.complementary} problems")
    print(f"  Combined potential: {best.combined_potential}/{num_problems}")
    print(f"  Diversity score: {best.diversity_score:.2f}")

    return best, all_pairs


def main():
    parser = argparse.ArgumentParser(description="Many-Worlds team search")
    parser.add_argument("--candidates", required=True, help="Comma-separated model names")
    parser.add_argument("--benchmark", default="gsm8k", help="Benchmark name")
    parser.add_argument("--num-problems", type=int, default=50)
    parser.add_argument("--output", default="team_search_results.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    candidates = [c.strip() for c in args.candidates.split(",")]

    best, all_pairs = search_team(
        candidates, args.benchmark, args.num_problems, args.device)

    # Save results
    output = {
        "benchmark": args.benchmark,
        "num_problems": args.num_problems,
        "best_pair": {
            "model_a": best.model_a,
            "model_b": best.model_b,
            "complementary": best.complementary,
            "combined_potential": best.combined_potential,
            "diversity_score": best.diversity_score,
        },
        "all_pairs": [
            {
                "model_a": d.model_a, "model_b": d.model_b,
                "score_a": d.score_a, "score_b": d.score_b,
                "both_right": d.both_right, "a_only": d.a_only,
                "b_only": d.b_only, "both_wrong": d.both_wrong,
                "complementary": d.complementary,
                "combined_potential": d.combined_potential,
                "diversity_score": d.diversity_score,
            }
            for d in all_pairs
        ],
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
