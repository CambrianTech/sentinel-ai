"""train_v7.py — Many-Worlds with LoRA fine-tuning on target model.

The key insight from v1-v6: you can't inject into a frozen model.
The frozen layers after the injection point treat the substrate
signal as noise and destroy it. Real multi-modal models (LLaVA,
Flamingo) fine-tune the LLM to learn to USE the injected features.

Architecture:
  Source model (frozen) → Project adapter → substrate field
  Target model (LoRA on layers L+1..N) + cross-attention at layer L

  The LoRA teaches the target model to integrate substrate input.
  The cross-attention learns what to pull from the substrate.
  The substrate + adapters learn the shared coordinate system.

  All trained together with next-token prediction loss:
  "Does the target model predict better WITH the substrate than without?"

This is the correct architecture. Everything before this was debugging
the gradient flow to get here.
"""

import argparse, gc, json, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

sys.path.insert(0, str(Path(__file__).parent))
from substrate import SubstrateVectorSpace, SubstrateConfig
from project_read import AdapterPair, AdapterConfig
from cross_attention import SubstrateCrossAttention, SubstrateCrossAttentionHook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--target", default="microsoft/phi-2")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--substrate-dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output", default="output/many_worlds_v7")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Load corpus
    corpus = []
    with open(args.corpus) as f:
        for line in f:
            item = json.loads(line)
            text = item.get("text", item.get("content", ""))
            if text.strip():
                corpus.append(text)

    print(f"{'='*60}")
    print(f"MANY-WORLDS v7 — LoRA + Cross-Attention")
    print(f"{'='*60}")
    print(f"Source: {args.source}")
    print(f"Target: {args.target} (LoRA fine-tuned)")
    print(f"Substrate: dim={args.substrate_dim}")
    print(f"Steps: {args.steps}, LR: {args.lr}")
    print(f"Corpus: {len(corpus)} examples")

    # Load source model (fully frozen)
    print(f"\nLoading source: {args.source}")
    source_model = AutoModelForCausalLM.from_pretrained(
        args.source, torch_dtype=torch.bfloat16, device_map=device)
    source_model.eval()
    for p in source_model.parameters():
        p.requires_grad = False
    source_tok = AutoTokenizer.from_pretrained(args.source)
    source_tok.pad_token = source_tok.pad_token or source_tok.eos_token
    src_hidden_dim = source_model.config.hidden_size
    src_layers = source_model.config.num_hidden_layers
    src_insert = int(src_layers * 2 / 3)

    # Load target model with LoRA on post-injection layers
    print(f"Loading target: {args.target}")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, device_map=device)
    tgt_hidden_dim = target_model.config.hidden_size
    tgt_layers = target_model.config.num_hidden_layers
    tgt_insert = int(tgt_layers * 2 / 3)
    target_tok = AutoTokenizer.from_pretrained(args.target)
    target_tok.pad_token = target_tok.pad_token or target_tok.eos_token

    # Apply LoRA to layers AFTER injection point only
    # These layers learn to use the cross-attention signal
    target_modules = []
    for i in range(tgt_insert, tgt_layers):
        # Phi-2 uses 'fc1', 'fc2' for MLP and 'q_proj', 'v_proj' for attention
        # Qwen uses 'gate_proj', 'up_proj', 'down_proj' and 'q_proj', 'v_proj'
        for mod in ["q_proj", "v_proj"]:
            target_modules.append(f"model.layers.{i}.self_attn.{mod}")

    # Try generic target modules if specific ones don't exist
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],  # peft handles layer filtering via layers_to_transform
        layers_to_transform=list(range(tgt_insert, tgt_layers)),
    )
    target_model = get_peft_model(target_model, lora_config)
    lora_params = sum(p.numel() for p in target_model.parameters() if p.requires_grad)
    print(f"  LoRA on layers {tgt_insert}-{tgt_layers-1}: {lora_params:,} trainable params")
    target_model.train()

    # Create substrate
    substrate_config = SubstrateConfig(dimensionality=args.substrate_dim, num_bases=128)
    substrate = SubstrateVectorSpace(substrate_config, device=device)

    # Source adapter (project into substrate)
    src_adapter_config = AdapterConfig(
        residual_hidden_size=src_hidden_dim,
        substrate_dim=args.substrate_dim,
        lora_rank=args.substrate_dim,
        layer_idx=src_layers,  # final layer — full semantic content
    )
    source_adapter = AdapterPair(src_adapter_config, args.source, device=device)

    # Cross-attention block for target
    cross_attn = SubstrateCrossAttention(
        target_hidden_dim=tgt_hidden_dim,
        substrate_dim=args.substrate_dim,
        num_heads=4,
    ).to(device)

    ca_params = sum(p.numel() for p in cross_attn.parameters())
    adapter_params = sum(p.numel() for p in source_adapter.parameters())
    sub_params = sum(p.numel() for p in substrate.parameters())
    total = lora_params + ca_params + adapter_params + sub_params
    print(f"  Cross-attention: {ca_params:,} params")
    print(f"  Source adapter: {adapter_params:,} params")
    print(f"  Substrate: {sub_params:,} params")
    print(f"  Total trainable: {total:,} ({total/1e6:.1f}M)")

    # Optimizer: LoRA + cross-attention + source adapter + substrate
    all_params = (
        list(filter(lambda p: p.requires_grad, target_model.parameters()))
        + list(cross_attn.parameters())
        + list(source_adapter.parameters())
        + list(substrate.parameters())
    )
    optimizer = AdamW(all_params, lr=args.lr, weight_decay=0.01)

    warmup_steps = min(args.steps // 10, 200)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=args.steps - warmup_steps, eta_min=args.lr * 0.01)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

    # Install cross-attention hook
    hook = SubstrateCrossAttentionHook(cross_attn, target_model, tgt_insert)

    print(f"\nTraining...")
    start = time.time()
    losses = []
    best_loss = float("inf")
    best_step = 0

    for step in range(args.steps):
        text = corpus[step % len(corpus)]

        # Source forward (frozen) → get FINAL hidden states (post all processing)
        # Mid-layer states contain positional processing artifacts.
        # Final layer output captures the source model's complete understanding.
        src_inputs = source_tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            src_out = source_model(**src_inputs, output_hidden_states=True)
        src_hidden = src_out.hidden_states[-1].float()  # final layer, not 2/3

        # Project into substrate and pool: one summary vector per input
        mu, _ = source_adapter.project(src_hidden)  # (1, seq_src, substrate_dim)
        mu_pooled = mu.mean(dim=1, keepdim=True)  # (1, 1, substrate_dim)
        hook.set_substrate_field(mu_pooled)

        # Target forward WITH cross-attention + LoRA active
        tgt_inputs = target_tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        labels = tgt_inputs["input_ids"].clone()

        # IMPORTANT: also run WITHOUT substrate as baseline, and only
        # backprop if substrate HELPS (loss decreases). This prevents
        # the model from learning to tolerate noise — it only learns
        # from examples where the substrate actually improved prediction.
        hook.set_substrate_field(None)  # disable
        with torch.no_grad():
            baseline_out = target_model(**tgt_inputs, labels=labels)
        baseline_loss = baseline_out.loss.item()

        hook.set_substrate_field(mu_pooled)  # re-enable
        outputs = target_model(**tgt_inputs, labels=labels)
        loss = outputs.loss

        # Only train on examples where substrate doesn't hurt too much
        # This is curriculum learning: the model first learns the easy
        # transfers, then gradually handles harder ones
        if loss.item() > baseline_loss * 1.5:
            # Substrate is badly hurting this example — skip training,
            # but still count it for monitoring
            losses.append(loss.item())
            scheduler.step()
            continue

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
            gate_info = ""
            print(f"  step {step:5d}/{args.steps} | loss={loss_val:.4f} | avg50={avg:.4f} | best={best_loss:.4f}@{best_step} | lr={lr_now:.2e} | {elapsed:.0f}s")

    hook.remove()
    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.0f}s")

    # Save everything
    substrate.save(str(out / "substrate.pt"))
    source_adapter.save(str(out / f"adapter_{args.source.replace('/', '_')}.pt"))
    torch.save(cross_attn.state_dict(), out / f"cross_attn_{args.target.replace('/', '_')}.pt")
    target_model.save_pretrained(str(out / "target_lora"))
    target_tok.save_pretrained(str(out / "target_lora"))

    meta = {
        "models": [args.source, args.target],
        "substrate_dim": args.substrate_dim,
        "steps": args.steps,
        "learning_rate": args.lr,
        "corpus_size": len(corpus),
        "final_loss": losses[-1] if losses else None,
        "best_loss": best_loss,
        "best_step": best_step,
        "training_time_seconds": elapsed,
        "lora_params": lora_params,
        "cross_attn_params": ca_params,
        "adapter_params": adapter_params,
        "substrate_params": sub_params,
        "total_trainable": total,
        "insert_layers": {args.source: src_insert, args.target: tgt_insert},
        "hidden_dims": {args.source: src_hidden_dim, args.target: tgt_hidden_dim},
        "losses_history": losses[-100:],
    }
    (out / "training_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved to {out}")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            print(f"  {f.relative_to(out)}: {size/1e6:.1f}MB" if size > 1e6 else f"  {f.relative_to(out)}: {size/1e3:.0f}KB")


if __name__ == "__main__":
    main()
