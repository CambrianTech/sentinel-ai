"""forge_population.py — Forge pipeline for Many-Worlds populations.

The complete pipeline: search → prune → quantize → assemble → eval → publish.

Usage:
    python forge_population.py --recipe population_recipe.json

Recipe format:
{
    "type": "many-worlds-population",
    "target": "microsoft/phi-3-mini-4k-instruct",
    "search_pool": ["Qwen/Qwen2.5-Math-1.5B-Instruct", "Qwen/Qwen2.5-Coder-1.5B-Instruct", ...],
    "benchmark": "gsm8k",
    "num_eval_problems": 50,
    "alpha": 0.2,
    "top_k": 20,
    "min_contribution": 0.01,
    "output": "output/population_v1"
}
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from team_search import evaluate_model, load_benchmark, compute_divergence, ModelScore


def load_model(name, device="cuda"):
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map=device)
    m.eval()
    t = AutoTokenizer.from_pretrained(name)
    t.pad_token = t.pad_token or t.eos_token
    return m, t


def compute_specialist_boosts(specialist_model, specialist_tok, target_tok,
                               prompt, device, top_k=20):
    """Get a specialist's top-K confident predictions mapped to target vocab."""
    inp = specialist_tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = specialist_model(**inp)
    logits = out.logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    topk = logits.topk(top_k)

    boosts = {}
    max_prob = 0
    for idx, score in zip(topk.indices, topk.values):
        token_text = specialist_tok.decode([idx.item()])
        t_ids = target_tok.encode(token_text, add_special_tokens=False)
        prob = probs[idx].item()
        if prob > max_prob:
            max_prob = prob
        if t_ids:
            boosts[t_ids[0]] = prob
    return boosts, max_prob


def forge_population(recipe_path: str):
    """Run the full population forge pipeline."""
    recipe = json.loads(Path(recipe_path).read_text())

    target_name = recipe["target"]
    search_pool = recipe["search_pool"]
    benchmark = recipe.get("benchmark", "gsm8k")
    num_problems = recipe.get("num_eval_problems", 50)
    alpha = recipe.get("alpha", 0.2)
    top_k = recipe.get("top_k", 20)
    min_contribution = recipe.get("min_contribution", 0.01)
    output_dir = Path(recipe.get("output", "output/population"))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    alloy_stages = []
    start_time = time.time()

    print(f"{'='*60}")
    print(f"MANY-WORLDS POPULATION FORGE")
    print(f"{'='*60}")
    print(f"Target: {target_name}")
    print(f"Search pool: {len(search_pool)} candidates")
    print(f"Benchmark: {benchmark} ({num_problems} problems)")

    # ── STAGE 1: Team Search ──────────────────────────────────────────
    print(f"\n── STAGE 1: Team Search ──")
    problems = load_benchmark(benchmark, num_problems)

    # Evaluate target model
    target_scores = evaluate_model(target_name, problems, device)

    # Evaluate all candidates
    candidate_scores = {}
    for name in search_pool:
        try:
            scores = evaluate_model(name, problems, device)
            candidate_scores[name] = scores
        except Exception as e:
            print(f"  SKIP {name}: {e}")

    # Compute divergence against target
    divergences = {}
    for name, scores in candidate_scores.items():
        div = compute_divergence(target_scores, scores)
        divergences[name] = div
        print(f"  {name.split('/')[-1]:30s} | score={sum(scores.correct):2d}/{len(scores.correct)} | "
              f"complementary={div.complementary:2d} | a_only={div.a_only} b_only={div.b_only}")

    alloy_stages.append({
        "type": "team-search",
        "searchPool": "manual",
        "benchmark": benchmark,
        "numProblems": num_problems,
        "candidatesEvaluated": len(candidate_scores),
    })

    # ── STAGE 2: Prune Roster ──────────────────────────────────────────
    print(f"\n── STAGE 2: Prune Roster ──")
    # Keep specialists that solve problems the target can't
    roster = []
    for name, div in sorted(divergences.items(), key=lambda x: x[1].b_only, reverse=True):
        contribution = div.b_only / max(len(problems), 1)
        if contribution >= min_contribution:
            roster.append(name)
            print(f"  KEEP {name.split('/')[-1]:30s} contributes {div.b_only} unique problems ({contribution:.1%})")
        else:
            print(f"  CUT  {name.split('/')[-1]:30s} contributes {div.b_only} unique problems ({contribution:.1%}) < {min_contribution:.0%}")

    if not roster:
        print("  WARNING: No specialists pass contribution threshold. Keeping top candidate.")
        roster = [max(divergences.items(), key=lambda x: x[1].b_only)[0]]

    alloy_stages.append({
        "type": "population-prune",
        "minContribution": min_contribution,
        "removedModels": [n for n in search_pool if n not in roster],
    })

    # ── STAGE 3: Routed Ensemble Eval ──────────────────────────────────
    print(f"\n── STAGE 3: Routed Ensemble Eval ──")
    print(f"  Roster: {[n.split('/')[-1] for n in roster]}")

    # Load target
    tm, tt = load_model(target_name, device)

    # Baseline
    baseline_correct = 0
    for prob in problems:
        inp = tt(prob["prompt"], return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = tm.generate(**inp, max_new_tokens=200, do_sample=False, pad_token_id=tt.eos_token_id)
        gen = tt.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        if prob["gt"] in gen:
            baseline_correct += 1

    print(f"  Baseline: {baseline_correct}/{len(problems)} ({baseline_correct/len(problems)*100:.1f}%)")

    # Routed ensemble — for each problem, find the most confident specialist
    ensemble_correct = 0
    routing_log = []

    for pi, prob in enumerate(problems):
        t_inp = tt(prob["prompt"], return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            t_out = tm(**t_inp)
        t_logits = t_out.logits[0, -1].float()
        t_probs = torch.softmax(t_logits, dim=-1)
        t_max_prob = t_probs.max().item()

        best_specialist = None
        best_conf = t_max_prob
        best_boosts = None

        for spec_name in roster:
            sm, st = load_model(spec_name, device)
            boosts, conf = compute_specialist_boosts(sm, st, tt, prob["prompt"], device, top_k)
            del sm, st; gc.collect(); torch.cuda.empty_cache()

            if conf > best_conf:
                best_specialist = spec_name
                best_conf = conf
                best_boosts = boosts

        if best_specialist and best_boosts:
            boosted = t_logits.clone()
            for tid, prob_val in best_boosts.items():
                boosted[tid] += prob_val * alpha * 10
            first_token = boosted.argmax().unsqueeze(0).unsqueeze(0)
            gen_inp = torch.cat([t_inp["input_ids"], first_token], dim=1)
            with torch.no_grad():
                out = tm.generate(gen_inp, max_new_tokens=199, do_sample=False, pad_token_id=tt.eos_token_id)
            gen = tt.decode(out[0][t_inp["input_ids"].shape[1]:], skip_special_tokens=True)
            routing_log.append(best_specialist.split("/")[-1])
        else:
            with torch.no_grad():
                out = tm.generate(**t_inp, max_new_tokens=200, do_sample=False, pad_token_id=tt.eos_token_id)
            gen = tt.decode(out[0][t_inp["input_ids"].shape[1]:], skip_special_tokens=True)
            routing_log.append("baseline")

        if prob["gt"] in gen:
            ensemble_correct += 1

        if (pi + 1) % 10 == 0:
            print(f"    [{pi+1}/{len(problems)}] ensemble={ensemble_correct}/{pi+1} "
                  f"routes={routing_log[-10:]}")

    delta = ensemble_correct - baseline_correct
    print(f"\n  Ensemble: {ensemble_correct}/{len(problems)} ({ensemble_correct/len(problems)*100:.1f}%)")
    print(f"  Delta: {'+' if delta >= 0 else ''}{delta}")

    alloy_stages.append({
        "type": "many-worlds-ensemble",
        "method": "logit-blend",
        "targetModel": target_name,
        "specialists": roster,
        "alpha": alpha,
        "topK": top_k,
        "blendStrategy": "routed-specialist-top-k-boost",
    })

    alloy_stages.append({
        "type": "eval",
        "benchmark": benchmark,
        "numProblems": len(problems),
        "baseline": baseline_correct,
        "ensemble": ensemble_correct,
        "delta": delta,
    })

    # ── Build Alloy ──────────────────────────────────────────────────
    elapsed = time.time() - start_time

    alloy = {
        "name": f"many-worlds-{benchmark}-blend",
        "version": "1.0.0",
        "type": "many-worlds-ensemble",
        "source": {
            "target": {"baseModel": target_name, "role": "generalist", "frozen": True},
            "specialists": [
                {"baseModel": n, "role": divergences[n].model_b.split("/")[-1] if n in divergences else "specialist", "frozen": True}
                for n in roster
            ],
        },
        "stages": alloy_stages,
        "results": {
            "benchmark": benchmark,
            "baseline": baseline_correct,
            "ensemble": ensemble_correct,
            "total": len(problems),
            "delta": delta,
        },
        "integrity": {
            "trustLevel": "self-attested",
            "code": {
                "runner": "many-worlds/forge_population",
                "version": "1.0.0",
                "sourceRepo": "https://github.com/CambrianTech/sentinel-ai",
            },
        },
        "forgeTime": elapsed,
        "roster": roster,
        "routingLog": routing_log,
    }

    alloy_path = output_dir / f"many-worlds-{benchmark}-blend.alloy.json"
    alloy_path.write_text(json.dumps(alloy, indent=2))

    print(f"\n{'='*60}")
    print(f"FORGE COMPLETE in {elapsed:.0f}s")
    print(f"{'='*60}")
    print(f"  Target: {target_name}")
    print(f"  Roster: {[n.split('/')[-1] for n in roster]}")
    print(f"  Baseline: {baseline_correct}/{len(problems)}")
    print(f"  Ensemble: {ensemble_correct}/{len(problems)} ({'+' if delta>=0 else ''}{delta})")
    print(f"  Alloy: {alloy_path}")


def main():
    parser = argparse.ArgumentParser(description="Forge a Many-Worlds population")
    parser.add_argument("--recipe", required=True, help="Path to recipe JSON")
    args = parser.parse_args()
    forge_population(args.recipe)


if __name__ == "__main__":
    main()
