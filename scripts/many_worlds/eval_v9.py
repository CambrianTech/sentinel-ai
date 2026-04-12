"""eval_v9.py — HumanEval+ eval for soft-prompt Many-Worlds.

Both models frozen. Substrate field → soft tokens → prepended to prompt.
No hooks, no LoRA, no perturbation. Pure knowledge transfer through
learned context tokens.
"""

import argparse, json, sys, gc, tempfile, subprocess
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from evalplus.data import get_human_eval_plus

sys.path.insert(0, str(Path(__file__).parent))
from substrate import SubstrateVectorSpace
from project_read import AdapterPair
from train_v9 import SubstrateToSoftPrompt


def load_model(name, device="cuda"):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def generate(model, tok, prompt, max_new=512):
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    g = out[0][inp["input_ids"].shape[1]:]
    text = tok.decode(g, skip_special_tokens=True)
    lines = text.split("\n")
    res = []
    for line in lines:
        if res and line.strip() and not line.startswith(" ") and not line.startswith("\t"):
            break
        res.append(line)
    return "\n".join(res)


def generate_with_soft_prompt(source_model, source_tok, target_model, target_tok,
                               source_adapter, soft_prompt_converter, embed_layer,
                               prompt, num_soft_tokens, device="cuda", max_new=512):
    """Generate with substrate soft tokens prepended."""
    # Source → substrate → pooled
    src_inp = source_tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        src_out = source_model(**src_inp, output_hidden_states=True)
    src_hidden = src_out.hidden_states[-1].float()
    mu, _ = source_adapter.project(src_hidden)
    mu_pooled = mu.mean(dim=1)  # (1, substrate_dim)

    # Soft tokens
    soft_tokens = soft_prompt_converter(mu_pooled)  # (1, num_tokens, embed_dim)

    # Target embeddings
    tgt_inp = target_tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        real_embeds = embed_layer(tgt_inp["input_ids"])

    # Combine: soft tokens + real tokens
    combined = torch.cat([soft_tokens.to(real_embeds.dtype), real_embeds], dim=1)
    soft_mask = torch.ones(1, num_soft_tokens, device=device, dtype=tgt_inp["attention_mask"].dtype)
    combined_mask = torch.cat([soft_mask, tgt_inp["attention_mask"]], dim=1)

    # Generate
    with torch.no_grad():
        out = target_model.generate(
            inputs_embeds=combined,
            attention_mask=combined_mask,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=target_tok.eos_token_id,
        )

    # Decode — skip soft token positions in output
    gen = out[0][combined.shape[1]:]
    text = target_tok.decode(gen, skip_special_tokens=True)
    lines = text.split("\n")
    res = []
    for line in lines:
        if res and line.strip() and not line.startswith(" ") and not line.startswith("\t"):
            break
        res.append(line)
    return "\n".join(res)


def check(tid, comp, prob):
    code = prob["prompt"] + comp + "\n\n" + prob["test"] + "\ncheck(" + prob["entry_point"] + ")\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code); f.flush()
        try:
            r = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=10)
            return r.returncode == 0
        except:
            return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate-dir", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default="eval_v9_results.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    sdir = Path(args.substrate_dir)
    meta = json.loads((sdir / "training_metadata.json").read_text())
    sn, tn = meta["models"]
    compare_name = "Qwen/Qwen3-4B"

    problems = get_human_eval_plus()
    tids = sorted(problems.keys())[:args.limit]

    print(f"{'='*60}")
    print(f"MANY-WORLDS v9 HUMANEVAL+ (soft prompt, both frozen)")
    print(f"{'='*60}")
    print(f"Problems: {len(tids)}")
    print(f"Source: {sn} | Target: {tn} | Compare: {compare_name}")

    # Phase A: Source alone
    print(f"\n  A. {sn}")
    sm, st = load_model(sn, device)
    ra = {}
    for i, tid in enumerate(tids):
        ok = check(tid, generate(sm, st, problems[tid]["prompt"]), problems[tid])
        ra[tid] = ok
        if (i+1) % 5 == 0: print(f"    [{i+1}/{len(tids)}] {sum(ra.values())}/{i+1}")
    del sm, st; gc.collect(); torch.cuda.empty_cache()

    # Phase B: Target alone
    print(f"\n  B. {tn}")
    tm, tt = load_model(tn, device)
    rb = {}
    for i, tid in enumerate(tids):
        ok = check(tid, generate(tm, tt, problems[tid]["prompt"]), problems[tid])
        rb[tid] = ok
        if (i+1) % 5 == 0: print(f"    [{i+1}/{len(tids)}] {sum(rb.values())}/{i+1}")
    del tm, tt; gc.collect(); torch.cuda.empty_cache()

    # Phase C: Substrate soft prompt
    print(f"\n  C. Substrate soft prompt ({sn} → {tn})")
    sm, st = load_model(sn, device)
    tm, tt = load_model(tn, device)
    sa = AdapterPair.load(str(sdir / f"adapter_{sn.replace('/', '_')}.pt"), device=device)
    sp = SubstrateToSoftPrompt(meta["substrate_dim"], meta["hidden_dims"][tn], meta["num_soft_tokens"]).to(device)
    sp.load_state_dict(torch.load(sdir / "soft_prompt.pt", map_location=device, weights_only=True))
    sp.eval()

    # Find embed layer
    embed_layer = None
    for name, mod in tm.named_modules():
        if isinstance(mod, nn.Embedding) and mod.weight.shape[0] > 1000:
            embed_layer = mod; break

    rc = {}
    for i, tid in enumerate(tids):
        comp = generate_with_soft_prompt(
            sm, st, tm, tt, sa, sp, embed_layer,
            problems[tid]["prompt"], meta["num_soft_tokens"], device)
        ok = check(tid, comp, problems[tid])
        rc[tid] = ok
        if (i+1) % 5 == 0: print(f"    [{i+1}/{len(tids)}] {sum(rc.values())}/{i+1}")
    del sm, st, tm, tt; gc.collect(); torch.cuda.empty_cache()

    # Phase D: Compare model
    print(f"\n  D. {compare_name}")
    cm, ct = load_model(compare_name, device)
    rd = {}
    for i, tid in enumerate(tids):
        ok = check(tid, generate(cm, ct, problems[tid]["prompt"]), problems[tid])
        rd[tid] = ok
        if (i+1) % 5 == 0: print(f"    [{i+1}/{len(tids)}] {sum(rd.values())}/{i+1}")
    del cm, ct; gc.collect(); torch.cuda.empty_cache()

    # Results
    sa_v = sum(ra.values()); sb_v = sum(rb.values()); sc_v = sum(rc.values()); sd_v = sum(rd.values())
    n = len(tids)
    print(f"\n{'='*60}")
    print(f"RESULTS — HumanEval+ pass@1 ({n} problems)")
    print(f"{'='*60}")
    print(f"  A. {sn:35} {sa_v:3}/{n:3} = {sa_v/n*100:5.1f}%")
    print(f"  B. {tn:35} {sb_v:3}/{n:3} = {sb_v/n*100:5.1f}%")
    print(f"  C. Substrate soft prompt           {sc_v:3}/{n:3} = {sc_v/n*100:5.1f}%")
    print(f"  D. {compare_name:35} {sd_v:3}/{n:3} = {sd_v/n*100:5.1f}%")
    print()
    print(f"  C > max(A,B)?  {'YES' if sc_v > max(sa_v, sb_v) else 'NO'}")

    # Discoveries
    discoveries = [tid for tid in tids if rc[tid] and not ra[tid] and not rb[tid]]
    if discoveries:
        print(f"\n  DISCOVERIES ({len(discoveries)} solved ONLY by substrate):")
        for tid in discoveries:
            print(f"    {tid}: {problems[tid]['entry_point']}")

    output = {
        "version": "v9", "architecture": "soft_prompt",
        "num_problems": n,
        "pass_at_1": {"A": sa_v/n, "B": sb_v/n, "C": sc_v/n, "D": sd_v/n},
        "discoveries": discoveries,
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
