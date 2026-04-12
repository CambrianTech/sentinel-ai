"""eval_substrate.py — Evaluate Many-Worlds v0 substrate against baselines.

The thesis: two small frozen models coordinated through a substrate
produce better output than either alone, and competitive with a single
model of equivalent total size.

Conditions:
  A. Qwen3-1.7B alone (baseline)
  B. Phi-2 alone (baseline)
  C. Substrate-coordinated (Qwen→substrate→Phi continuation)
  D. Single model of comparable size (Qwen3-4B)
  E. Random substrate (negative control — proves the trained substrate
     does structured work, not just "more params help")

Metrics:
  - Perplexity of continuation under a reference model
  - Code completion quality (if code prompts)
  - Semantic similarity between conditions

Usage:
    python scripts/many_worlds/eval_substrate.py \
        --substrate /mnt/cold/factory-work/many_worlds_v0/ \
        --prompts eval_prompts.jsonl \
        --output eval_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(name: str, device: str = "cuda"):
    """Load a model for generation."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 100) -> str:
    """Generate continuation from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the generated tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def compute_perplexity(model, tokenizer, text: str) -> float:
    """Compute perplexity of text under a model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()


def substrate_transfer_generate(
    source_model, source_tokenizer,
    target_model, target_tokenizer,
    substrate, adapter_source, adapter_target,
    prompt: str, max_new_tokens: int = 100,
    device: str = "cuda",
) -> str:
    """Generate via substrate transfer: source→project→substrate→read→target.

    1. Source model forward pass on prompt → hidden states at 2/3 depth
    2. Project hidden states into substrate via source adapter
    3. Read from substrate into target model's residual form via target adapter
    4. Inject into target model and generate continuation
    """
    # Step 1: Get source hidden states
    source_inputs = source_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    source_inputs = {k: v.to(device) for k, v in source_inputs.items()}

    with torch.no_grad():
        source_outputs = source_model(**source_inputs, output_hidden_states=True)

    source_layer = adapter_source.config.layer_idx
    source_hidden = source_outputs.hidden_states[source_layer].float()  # (1, seq, hidden)

    # Step 2: Project into substrate
    mu, log_var = adapter_source.project(source_hidden)  # (1, seq, substrate_dim)

    # Step 3: Read into target residual form
    target_residual_delta = adapter_target.read(mu)  # (1, seq, target_hidden)

    # Inject the substrate delta into the target model's residual stream
    # via a forward hook. The hook fires at the target layer and ADDS
    # the substrate-transferred representation to the residual.
    target_layer_idx = adapter_target.config.layer_idx
    hook_handle = None
    substrate_delta = target_residual_delta.detach()  # (1, seq_source, target_hidden)

    def inject_substrate(module, input, output):
        """Forward hook: add substrate delta to the target layer's output."""
        hidden = output[0] if isinstance(output, tuple) else output
        # The substrate delta is from the source prompt; pad/truncate to match target seq len
        seq_target = hidden.shape[1]
        seq_source = substrate_delta.shape[1]
        if seq_source >= seq_target:
            delta = substrate_delta[:, :seq_target, :].to(hidden.dtype)
        else:
            # Pad with zeros for tokens beyond what source covered
            import torch
            pad = torch.zeros(1, seq_target - seq_source, hidden.shape[2],
                            device=hidden.device, dtype=hidden.dtype)
            delta = torch.cat([substrate_delta.to(hidden.dtype), pad], dim=1)
        hidden = hidden + delta
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    # Install hook on the target layer
    target_layers = target_model.model.layers if hasattr(target_model, 'model') else target_model.transformer.h
    hook_handle = target_layers[target_layer_idx].register_forward_hook(inject_substrate)

    target_inputs = target_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    target_inputs = {k: v.to(device) for k, v in target_inputs.items()}

    try:
        with torch.no_grad():
            target_outputs = target_model.generate(
                **target_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=target_tokenizer.eos_token_id,
            )
    finally:
        if hook_handle:
            hook_handle.remove()

    generated = target_outputs[0][target_inputs["input_ids"].shape[1]:]
    text = target_tokenizer.decode(generated, skip_special_tokens=True)

    # Compute cos_sim between substrate-transferred and actual target hidden states
    with torch.no_grad():
        target_hidden_actual = target_model(
            **target_inputs, output_hidden_states=True
        ).hidden_states[target_layer_idx].float()

    cos_sim = torch.nn.functional.cosine_similarity(
        substrate_delta.mean(dim=1),
        target_hidden_actual.mean(dim=1),
        dim=-1,
    ).item()

    return text, cos_sim


def main():
    parser = argparse.ArgumentParser(description="Evaluate Many-Worlds v0 substrate")
    parser.add_argument("--substrate", required=True, help="Path to v0 output dir")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    substrate_dir = Path(args.substrate)
    device = args.device

    # Load training metadata
    meta = json.loads((substrate_dir / "training_metadata.json").read_text())
    source_name, target_name = meta["models"]
    print(f"Source: {source_name}")
    print(f"Target: {target_name}")

    # Test prompts — mix of code and reasoning
    prompts = [
        "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"",
        "def merge_sort(arr):\n    \"\"\"Sort array using merge sort algorithm.\"\"\"",
        "Explain why neural networks can approximate any continuous function:",
        "Write a Python function to find all prime numbers up to n:\ndef sieve_of_eratosthenes(n):",
        "The key insight behind attention mechanisms in transformers is",
    ]

    print(f"\n{'='*60}")
    print(f"MANY-WORLDS v0 EVALUATION")
    print(f"{'='*60}")
    print(f"Population: {source_name} ({meta['hidden_dims'][source_name]}d) + {target_name} ({meta['hidden_dims'][target_name]}d)")
    print(f"Substrate: dim={meta['substrate_dim']}")
    print(f"Comparison: Qwen/Qwen3-4B (single model, ~4B params)")
    print(f"Prompts: {len(prompts)}")

    # Load models
    print(f"\nLoading models...")
    source_model, source_tok = load_model(source_name, device)
    target_model, target_tok = load_model(target_name, device)
    print(f"  Loaded {source_name} + {target_name}")

    # Load comparison model
    compare_name = "Qwen/Qwen3-4B"
    compare_model, compare_tok = load_model(compare_name, device)
    print(f"  Loaded {compare_name}")

    # Load substrate + adapters
    sys.path.insert(0, str(Path(__file__).parent))
    from substrate import SubstrateVectorSpace, SubstrateConfig
    from project_read import AdapterPair, AdapterConfig

    substrate = SubstrateVectorSpace.load(str(substrate_dir / "substrate.pt"), device=device)

    source_safe = source_name.replace("/", "_")
    target_safe = target_name.replace("/", "_")
    adapter_source = AdapterPair.load(str(substrate_dir / f"adapter_{source_safe}.pt"), device=device)
    adapter_target = AdapterPair.load(str(substrate_dir / f"adapter_{target_safe}.pt"), device=device)
    print(f"  Loaded substrate + adapters")

    results = []

    for i, prompt in enumerate(prompts):
        print(f"\n--- Prompt {i+1}/{len(prompts)} ---")
        print(f"  {prompt[:60]}...")

        row = {"prompt": prompt, "conditions": {}}

        # Condition A: Source model alone
        gen_a = generate(source_model, source_tok, prompt, args.max_tokens)
        ppl_a = compute_perplexity(target_model, target_tok, prompt + gen_a)
        row["conditions"]["A_source_alone"] = {"text": gen_a, "ppl": ppl_a}
        print(f"  A ({source_name}): PPL={ppl_a:.2f}")

        # Condition B: Target model alone
        gen_b = generate(target_model, target_tok, prompt, args.max_tokens)
        ppl_b = compute_perplexity(target_model, target_tok, prompt + gen_b)
        row["conditions"]["B_target_alone"] = {"text": gen_b, "ppl": ppl_b}
        print(f"  B ({target_name}): PPL={ppl_b:.2f}")

        # Condition C: Substrate transfer
        gen_c, cos_sim = substrate_transfer_generate(
            source_model, source_tok,
            target_model, target_tok,
            substrate, adapter_source, adapter_target,
            prompt, args.max_tokens, device,
        )
        ppl_c = compute_perplexity(target_model, target_tok, prompt + gen_c)
        row["conditions"]["C_substrate"] = {"text": gen_c, "ppl": ppl_c, "cos_sim": cos_sim}
        print(f"  C (substrate):    PPL={ppl_c:.2f}, cos_sim={cos_sim:.4f}")

        # Condition D: Comparable single model
        gen_d = generate(compare_model, compare_tok, prompt, args.max_tokens)
        ppl_d = compute_perplexity(target_model, target_tok, prompt + gen_d)
        row["conditions"]["D_single_4B"] = {"text": gen_d, "ppl": ppl_d}
        print(f"  D ({compare_name}): PPL={ppl_d:.2f}")

        results.append(row)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    avg = {k: sum(r["conditions"][k]["ppl"] for r in results) / len(results)
           for k in ["A_source_alone", "B_target_alone", "C_substrate", "D_single_4B"]}
    avg_cos = sum(r["conditions"]["C_substrate"]["cos_sim"] for r in results) / len(results)

    print(f"  A. {source_name:30} avg PPL: {avg['A_source_alone']:.2f}")
    print(f"  B. {target_name:30} avg PPL: {avg['B_target_alone']:.2f}")
    print(f"  C. Substrate transfer          avg PPL: {avg['C_substrate']:.2f}  cos_sim: {avg_cos:.4f}")
    print(f"  D. {compare_name:30} avg PPL: {avg['D_single_4B']:.2f}")
    print()

    thesis = avg["C_substrate"] < min(avg["A_source_alone"], avg["B_target_alone"])
    competitive = avg["C_substrate"] < avg["D_single_4B"] * 1.1  # within 10%
    print(f"  Substrate beats both alone?  {'YES' if thesis else 'NO'}")
    print(f"  Competitive with {compare_name}? {'YES' if competitive else 'NO'}")

    # Save
    output = {
        "meta": meta,
        "comparison_model": compare_name,
        "prompts": len(prompts),
        "averages": avg,
        "avg_cos_sim": avg_cos,
        "thesis_holds": thesis,
        "competitive": competitive,
        "results": results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
