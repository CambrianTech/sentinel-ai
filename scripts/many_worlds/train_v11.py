"""train_v11.py — Many-Worlds with Q-Former bridge.

Fixes from v1-v10:
1. Q-Former replaces linear projection — each query extracts a DIFFERENT
   aspect of the source's knowledge (not 16 copies of the same info)
2. Per-token substrate field, NOT pooled — positional structure preserved
3. Middle layer extraction (2/3 depth, not final) — semantic, not vocab-specific
4. LayerNorm + small-gain init on output — natural magnitude control
5. Both models frozen — diversity preserved

Architecture:
    Source (frozen) layer L → Adapter → substrate field (seq, 256)
         ↓ K, V
    Q-Former (16 learned queries, 2 layers of cross-attn + self-attn)
         ↓
    soft tokens (16, target_embed_dim) → [prepend] → Target (frozen) → NTP loss
"""

import argparse, json, sys, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from substrate import SubstrateVectorSpace, SubstrateConfig
from project_read import AdapterPair, AdapterConfig
from qformer import SubstrateQFormer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--target", default="microsoft/phi-2")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--substrate-dim", type=int, default=256)
    parser.add_argument("--num-queries", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", default="output/many_worlds_v11")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    corpus = []
    with open(args.corpus) as f:
        for line in f:
            item = json.loads(line)
            text = item.get("text", item.get("content", ""))
            if text.strip():
                corpus.append(text)

    print(f"{'='*60}")
    print(f"MANY-WORLDS v11 — Q-Former Bridge")
    print(f"{'='*60}")
    print(f"Source: {args.source} (frozen)")
    print(f"Target: {args.target} (frozen)")
    print(f"Substrate: dim={args.substrate_dim}")
    print(f"Q-Former: {args.num_queries} queries, 2 layers")
    print(f"Steps: {args.steps}, LR: {args.lr}")
    print(f"Corpus: {len(corpus)} examples")

    # Load source (frozen)
    print(f"\nLoading {args.source}...")
    source_model = AutoModelForCausalLM.from_pretrained(
        args.source, torch_dtype=torch.bfloat16, device_map=device)
    source_model.eval()
    for p in source_model.parameters():
        p.requires_grad = False
    source_tok = AutoTokenizer.from_pretrained(args.source)
    source_tok.pad_token = source_tok.pad_token or source_tok.eos_token
    src_dim = source_model.config.hidden_size
    src_layers = source_model.config.num_hidden_layers
    src_extract = int(src_layers * 2 / 3)  # middle-ish layer, not final

    # Load target (frozen)
    print(f"Loading {args.target}...")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, device_map=device)
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad = False
    target_tok = AutoTokenizer.from_pretrained(args.target)
    target_tok.pad_token = target_tok.pad_token or target_tok.eos_token
    tgt_dim = target_model.config.hidden_size

    # Find target embedding layer
    embed_layer = None
    for name, mod in target_model.named_modules():
        if isinstance(mod, nn.Embedding) and mod.weight.shape[0] > 1000:
            embed_layer = mod
            print(f"  Embed: {name} ({mod.weight.shape})")
            break

    # Get target embedding norm for reference
    with torch.no_grad():
        sample = target_tok("hello world", return_tensors="pt").to(device)
        sample_embeds = embed_layer(sample["input_ids"])
        tgt_embed_norm = sample_embeds.norm(dim=-1).mean().item()
    print(f"  Target embed norm: {tgt_embed_norm:.2f}")

    # Substrate
    substrate = SubstrateVectorSpace(
        SubstrateConfig(dimensionality=args.substrate_dim, num_bases=128), device=device)

    # Source adapter — extracts at 2/3 depth, per-token (no pooling)
    src_adapter = AdapterPair(
        AdapterConfig(
            residual_hidden_size=src_dim,
            substrate_dim=args.substrate_dim,
            lora_rank=args.substrate_dim,
            layer_idx=src_extract,
        ),
        args.source, device=device,
    )
    print(f"  Source adapter: layer {src_extract}, rank {args.substrate_dim}")

    # Q-Former bridge — target_scale set from measured embedding norm
    qformer = SubstrateQFormer(
        substrate_dim=args.substrate_dim,
        target_embed_dim=tgt_dim,
        num_queries=args.num_queries,
        num_heads=4,
        num_layers=2,
    ).to(device)
    qformer.target_scale.fill_(tgt_embed_norm)
    print(f"  Q-Former target_scale: {tgt_embed_norm:.2f}")

    qf_params = sum(p.numel() for p in qformer.parameters())
    ad_params = sum(p.numel() for p in src_adapter.parameters())
    sub_params = sum(p.numel() for p in substrate.parameters())
    total = qf_params + ad_params + sub_params
    print(f"  Q-Former: {qf_params:,} params")
    print(f"  Source adapter: {ad_params:,} params")
    print(f"  Substrate: {sub_params:,} params")
    print(f"  Total trainable: {total:,} ({total/1e6:.1f}M)")

    # Optimizer — only Q-Former + adapter + substrate
    all_params = (
        list(qformer.parameters())
        + list(src_adapter.parameters())
        + list(substrate.parameters())
    )
    optimizer = AdamW(all_params, lr=args.lr, weight_decay=0.01)
    warmup_steps = min(args.steps // 10, 200)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=args.steps - warmup_steps, eta_min=args.lr * 0.01)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

    print(f"\nTraining...")
    start = time.time()
    losses = []
    best_loss = float("inf")
    best_step = 0

    for step in range(args.steps):
        text = corpus[step % len(corpus)]

        # Source forward (frozen) → hidden states at extraction layer
        src_inputs = source_tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            src_out = source_model(**src_inputs, output_hidden_states=True)
        src_hidden = src_out.hidden_states[src_extract].float()  # (1, seq, src_dim)

        # Project into substrate — PER-TOKEN, not pooled
        mu, _ = src_adapter.project(src_hidden)  # (1, seq, substrate_dim)

        # Q-Former: learned queries cross-attend to substrate field
        soft_tokens = qformer(mu)  # (1, num_queries, tgt_dim)

        # Get target embeddings
        tgt_inputs = target_tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            real_embeds = embed_layer(tgt_inputs["input_ids"])  # (1, seq, tgt_dim)

        # Verify magnitude (should be close to tgt_embed_norm without explicit normalization)
        if step == 0:
            soft_norm = soft_tokens.norm(dim=-1).mean().item()
            print(f"  Step 0 — soft token norm: {soft_norm:.2f}, target embed norm: {tgt_embed_norm:.2f}, ratio: {soft_norm/tgt_embed_norm:.2f}x")

        # Prepend soft tokens to real embeddings
        combined = torch.cat([soft_tokens.to(real_embeds.dtype), real_embeds], dim=1)
        soft_mask = torch.ones(1, args.num_queries, device=device, dtype=tgt_inputs["attention_mask"].dtype)
        combined_mask = torch.cat([soft_mask, tgt_inputs["attention_mask"]], dim=1)

        # Labels: -100 for soft token positions (don't predict them)
        prefix_labels = torch.full((1, args.num_queries), -100, device=device, dtype=torch.long)
        real_labels = tgt_inputs["input_ids"].clone()
        combined_labels = torch.cat([prefix_labels, real_labels], dim=1)

        # Target forward with soft prompt
        outputs = target_model(
            inputs_embeds=combined,
            attention_mask=combined_mask,
            labels=combined_labels,
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        losses.append(loss_val)
        avg = sum(losses[-50:]) / min(50, len(losses))
        if avg < best_loss:
            best_loss = avg
            best_step = step

        if step % 100 == 0 or step == args.steps - 1:
            elapsed = time.time() - start
            lr_now = scheduler.get_last_lr()[0]
            print(f"  step {step:5d}/{args.steps} | loss={loss_val:.4f} | avg50={avg:.4f} | best={best_loss:.4f}@{best_step} | lr={lr_now:.2e} | {elapsed:.0f}s")

    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.0f}s")

    # Save
    substrate.save(str(out / "substrate.pt"))
    src_adapter.save(str(out / f"adapter_{args.source.replace('/', '_')}.pt"))
    torch.save(qformer.state_dict(), out / "qformer.pt")

    meta = {
        "version": "v11",
        "architecture": "qformer_soft_prompt",
        "models": [args.source, args.target],
        "substrate_dim": args.substrate_dim,
        "num_queries": args.num_queries,
        "source_extract_layer": src_extract,
        "steps": args.steps,
        "learning_rate": args.lr,
        "corpus_size": len(corpus),
        "final_loss": losses[-1] if losses else None,
        "best_loss": best_loss,
        "best_step": best_step,
        "training_time_seconds": elapsed,
        "total_trainable": total,
        "qformer_params": qf_params,
        "adapter_params": ad_params,
        "substrate_params": sub_params,
        "target_embed_norm": tgt_embed_norm,
        "hidden_dims": {args.source: src_dim, args.target: tgt_dim},
        "losses_history": losses[-100:],
    }
    (out / "training_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
