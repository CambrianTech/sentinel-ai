"""eval_humaneval.py — Many-Worlds HumanEval+ thesis test.

Runs each condition as a separate phase to avoid OOM on 64GB systems.
Each phase loads only the models it needs, generates completions,
saves them to disk, then frees everything before the next phase.

Usage:
    python eval_humaneval.py --substrate /path/to/v2/ --limit 20 --output results.json
"""

from __future__ import annotations
import argparse, gc, json, subprocess, sys, tempfile, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evalplus.data import get_human_eval_plus


def load_model(name, device="cuda"):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def free_models():
    gc.collect()
    torch.cuda.empty_cache()


def generate_completion(model, tok, prompt, max_new=512):
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(gen, skip_special_tokens=True)
    # Stop at first non-indented line after body
    lines = text.split("\n")
    result = []
    for line in lines:
        if result and line.strip() and not line.startswith(" ") and not line.startswith("\t"):
            break
        result.append(line)
    return "\n".join(result)


def check_correctness(task_id, completion, problem):
    solution = problem["prompt"] + completion
    entry = problem["entry_point"]
    test_code = solution + "\n\n" + problem["test"] + f"\ncheck({entry})\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        f.flush()
        try:
            r = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=10)
            return r.returncode == 0
        except:
            return False


def run_single_model(model_name, task_ids, problems, device):
    """Run one model on all problems, return pass dict."""
    model, tok = load_model(model_name, device)
    results = {}
    for i, tid in enumerate(task_ids):
        comp = generate_completion(model, tok, problems[tid]["prompt"])
        ok = check_correctness(tid, comp, problems[tid])
        results[tid] = ok
        if (i+1) % 5 == 0:
            passed = sum(results.values())
            print(f"    [{i+1}/{len(task_ids)}] {passed}/{i+1} ({passed/(i+1)*100:.0f}%)")
    del model, tok
    free_models()
    return results


def run_substrate(source_name, target_name, substrate_dir, task_ids, problems, device, use_random=False):
    """Run substrate-coordinated generation via cross-attention."""
    sys.path.insert(0, str(Path(__file__).parent))
    from substrate import SubstrateVectorSpace
    from project_read import AdapterPair
    from cross_attention import SubstrateCrossAttention, SubstrateCrossAttentionHook

    source_model, source_tok = load_model(source_name, device)
    target_model, target_tok = load_model(target_name, device)

    substrate = SubstrateVectorSpace.load(str(substrate_dir / "substrate.pt"), device=device)
    source_safe = source_name.replace("/", "_")
    target_safe = target_name.replace("/", "_")
    adapter_source = AdapterPair.load(str(substrate_dir / f"adapter_{source_safe}.pt"), device=device)
    adapter_target = AdapterPair.load(str(substrate_dir / f"adapter_{target_safe}.pt"), device=device)

    # Load cross-attention block
    ca_path = substrate_dir / f"cross_attn_{target_safe}.pt"
    meta = json.loads((substrate_dir / "training_metadata.json").read_text())
    ca = SubstrateCrossAttention(
        target_hidden_dim=meta["hidden_dims"][target_name],
        substrate_dim=meta["substrate_dim"],
    ).to(device)
    if ca_path.exists():
        ca.load_state_dict(torch.load(ca_path, map_location=device, weights_only=True))
    ca.eval()

    # Install cross-attention hook
    hook = SubstrateCrossAttentionHook(ca, target_model, adapter_target.config.layer_idx)

    results = {}
    for i, tid in enumerate(task_ids):
        prompt = problems[tid]["prompt"]

        # Get source hidden states → project into substrate
        src_inputs = source_tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
        src_inputs = {k: v.to(device) for k, v in src_inputs.items()}
        with torch.no_grad():
            src_out = source_model(**src_inputs, output_hidden_states=True)
        src_hidden = src_out.hidden_states[adapter_source.config.layer_idx].float()
        mu, _ = adapter_source.project(src_hidden)
        if use_random:
            mu = torch.randn_like(mu)

        # Set substrate field for cross-attention hook
        hook.set_substrate_field(mu.detach())

        # Generate with cross-attention active
        comp = generate_completion(target_model, target_tok, prompt)

        ok = check_correctness(tid, comp, problems[tid])
        results[tid] = ok

        if (i+1) % 5 == 0:
            passed = sum(results.values())
            print(f"    [{i+1}/{len(task_ids)}] {passed}/{i+1} ({passed/(i+1)*100:.0f}%)")

    hook.remove()
    del source_model, source_tok, target_model, target_tok
    free_models()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="humaneval_results.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    substrate_dir = Path(args.substrate)
    meta = json.loads((substrate_dir / "training_metadata.json").read_text())
    source_name, target_name = meta["models"]
    compare_name = "Qwen/Qwen3-4B"

    problems = get_human_eval_plus()
    task_ids = sorted(problems.keys())
    if args.limit > 0:
        task_ids = task_ids[:args.limit]

    print(f"{'='*60}")
    print(f"MANY-WORLDS HUMANEVAL+ (pass@1)")
    print(f"{'='*60}")
    print(f"Problems: {len(task_ids)}")
    print(f"Source: {source_name} | Target: {target_name} | Compare: {compare_name}")

    # A: Source alone
    print(f"\n  A. {source_name}")
    results_a = run_single_model(source_name, task_ids, problems, args.device)

    # B: Target alone
    print(f"\n  B. {target_name}")
    results_b = run_single_model(target_name, task_ids, problems, args.device)

    # C: Substrate transfer
    print(f"\n  C. Substrate ({source_name} → {target_name})")
    results_c = run_substrate(source_name, target_name, substrate_dir, task_ids, problems, args.device)

    # D: Comparable single model
    print(f"\n  D. {compare_name}")
    results_d = run_single_model(compare_name, task_ids, problems, args.device)

    # E: Random substrate (negative control)
    print(f"\n  E. Random substrate (control)")
    results_e = run_substrate(source_name, target_name, substrate_dir, task_ids, problems, args.device, use_random=True)

    # Summary
    def score(r): return sum(r.values()), len(r)
    sa, ta = score(results_a); sb, tb = score(results_b)
    sc, tc = score(results_c); sd, td = score(results_d); se, te = score(results_e)

    print(f"\n{'='*60}")
    print(f"RESULTS — HumanEval+ pass@1 ({len(task_ids)} problems)")
    print(f"{'='*60}")
    print(f"  A. {source_name:35} {sa:3}/{ta:3} = {sa/ta*100:5.1f}%")
    print(f"  B. {target_name:35} {sb:3}/{tb:3} = {sb/tb*100:5.1f}%")
    print(f"  C. Substrate transfer              {sc:3}/{tc:3} = {sc/tc*100:5.1f}%")
    print(f"  D. {compare_name:35} {sd:3}/{td:3} = {sd/td*100:5.1f}%")
    print(f"  E. Random substrate (control)      {se:3}/{te:3} = {se/te*100:5.1f}%")
    print()
    print(f"  C > max(A,B)?  {'YES' if sc/tc > max(sa/ta, sb/tb) else 'NO':3}  (substrate beats both alone)")
    print(f"  C > E?         {'YES' if sc/tc > se/te else 'NO':3}  (trained > random)")
    print(f"  C ~ D?         {'YES' if abs(sc/tc - sd/td) < 0.1 else 'NO':3}  (competitive with {compare_name})")

    # Per-problem analysis: find problems where C passes but A+B fail
    discoveries = []
    for tid in task_ids:
        if results_c[tid] and not results_a[tid] and not results_b[tid]:
            discoveries.append(tid)
    if discoveries:
        print(f"\n  DISCOVERIES ({len(discoveries)} problems solved ONLY by substrate):")
        for tid in discoveries:
            print(f"    {tid}: {problems[tid]['entry_point']}")

    output = {
        "meta": meta, "benchmark": "HumanEval+", "num_problems": len(task_ids),
        "pass_at_1": {"A": sa/ta, "B": sb/tb, "C": sc/tc, "D": sd/td, "E": se/te},
        "raw": {tid: {"A": results_a[tid], "B": results_b[tid], "C": results_c[tid],
                       "D": results_d[tid], "E": results_e[tid]} for tid in task_ids},
        "discoveries": discoveries,
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
