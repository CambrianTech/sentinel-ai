"""train_v9.py — Many-Worlds via soft prompt injection.

Key insight from v1-v8: injecting into a frozen model's residual stream
disrupts it regardless of scale or mechanism. The model wasn't trained
to receive foreign input at intermediate layers.

New approach: convert the substrate field into a SOFT PROMPT — learned
embedding tokens prepended to the target model's input. The target
model processes them through its NORMAL forward pass. No hooks, no
perturbation, no disruption. The substrate information enters through
the front door, not a side window.

Architecture:
  Source model (frozen) → Project adapter → substrate μ (pooled)
  substrate μ → learned linear → k soft tokens (target embed dim)
  [soft_tokens] + [real_tokens] → Target model → NTP loss

The soft tokens act as a prefix that primes the target model's
computation with the source model's knowledge. The target processes
them natively through all layers, so there's no distribution shift.
"""

import argparse, json, sys, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

sys.path.insert(0, str(Path(__file__).parent))
from substrate import SubstrateVectorSpace, SubstrateConfig
from project_read import AdapterPair, AdapterConfig


class SubstrateToSoftPrompt(nn.Module):
    """Convert a substrate field vector into soft prompt tokens.

    Maps a single substrate vector (substrate_dim) into k soft tokens,
    each of target_embed_dim. These tokens are prepended to the target
    model's input embeddings before the forward pass.
    """
    def __init__(self, substrate_dim: int, target_embed_dim: int, num_tokens: int = 8):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Linear(substrate_dim, target_embed_dim * num_tokens)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.zeros_(self.proj.bias)

    def forward(self, substrate_field: torch.Tensor) -> torch.Tensor:
        """
        Args:
            substrate_field: (batch, substrate_dim)
        Returns:
            soft_tokens: (batch, num_tokens, target_embed_dim)
        """
        B = substrate_field.shape[0]
        flat = self.proj(substrate_field)  # (B, num_tokens * embed_dim)
        return flat.view(B, self.num_tokens, -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--target", default="microsoft/phi-2")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--substrate-dim", type=int, default=256)
    parser.add_argument("--num-soft-tokens", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="output/many_worlds_v9")
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
    print(f"MANY-WORLDS v9 — Soft Prompt Injection")
    print(f"{'='*60}")
    print(f"Source: {args.source} (frozen)")
    print(f"Target: {args.target} (LoRA)")
    print(f"Substrate: dim={args.substrate_dim}")
    print(f"Soft tokens: {args.num_soft_tokens}")
    print(f"Corpus: {len(corpus)} examples")

    # Load source
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

    # Load target with LoRA
    print(f"Loading {args.target}...")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, device_map=device)
    tgt_dim = target_model.config.hidden_size
    tgt_layers = target_model.config.num_hidden_layers
    target_tok = AutoTokenizer.from_pretrained(args.target)
    target_tok.pad_token = target_tok.pad_token or target_tok.eos_token

    # NO LoRA — target model stays fully frozen.
    # If soft prompt + substrate can improve predictions without
    # changing the target model at all, that's the purest proof
    # of cross-model knowledge transfer. The diversity is preserved.
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad = False
    lora_params = 0
    print(f"  Target: FULLY FROZEN (no LoRA)")

    # Substrate + source adapter
    substrate = SubstrateVectorSpace(
        SubstrateConfig(dimensionality=args.substrate_dim, num_bases=128), device=device)
    src_adapter = AdapterPair(
        AdapterConfig(residual_hidden_size=src_dim, substrate_dim=args.substrate_dim,
                      lora_rank=args.substrate_dim, layer_idx=src_layers),
        args.source, device=device)

    # Soft prompt converter
    soft_prompt = SubstrateToSoftPrompt(
        args.substrate_dim, tgt_dim, args.num_soft_tokens).to(device)

    sp_params = sum(p.numel() for p in soft_prompt.parameters())
    ad_params = sum(p.numel() for p in src_adapter.parameters())
    sub_params = sum(p.numel() for p in substrate.parameters())
    total = lora_params + sp_params + ad_params + sub_params
    print(f"  Soft prompt: {sp_params:,} params")
    print(f"  Source adapter: {ad_params:,} params")
    print(f"  Substrate: {sub_params:,} params")
    print(f"  Total: {total:,} ({total/1e6:.1f}M)")

    # Get target model's embedding layer for prepending soft tokens
    embed_layer = None
    for name, mod in target_model.named_modules():
        if isinstance(mod, nn.Embedding) and mod.weight.shape[0] > 1000:  # vocab-sized embedding
            embed_layer = mod
            print(f"  Embed layer: {name} ({type(mod).__name__}, {mod.weight.shape})")
            break
    if embed_layer is None:
        raise RuntimeError(f"Can't find embedding layer in {type(target_model)}")

    # Only train: soft prompt converter + source adapter + substrate
    # Target model is FROZEN — no params from it
    all_params = (
        list(soft_prompt.parameters())
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

        # Source → substrate → pooled vector
        src_inputs = source_tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            src_out = source_model(**src_inputs, output_hidden_states=True)
        src_hidden = src_out.hidden_states[-1].float()  # final layer
        mu, _ = src_adapter.project(src_hidden)  # (1, seq, substrate_dim)
        mu_pooled = mu.mean(dim=1)  # (1, substrate_dim)

        # Convert to soft prompt tokens
        soft_tokens = soft_prompt(mu_pooled)  # (1, num_tokens, tgt_dim)

        # Get target embeddings for the real tokens
        tgt_inputs = target_tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            real_embeds = embed_layer(tgt_inputs["input_ids"])  # (1, seq, tgt_dim)

        # Normalize soft tokens to match real embedding magnitude.
        # Without this, soft tokens dominate attention (2000× larger)
        # and the model ignores the actual input entirely.
        with torch.no_grad():
            target_norm = real_embeds.norm(dim=-1).mean().item()
        soft_norm = soft_tokens.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        soft_tokens = soft_tokens * (target_norm / soft_norm)

        # Prepend soft tokens to real embeddings
        combined_embeds = torch.cat([soft_tokens.to(real_embeds.dtype), real_embeds], dim=1)

        # Build attention mask for soft tokens + real tokens
        soft_mask = torch.ones(1, args.num_soft_tokens, device=device, dtype=tgt_inputs["attention_mask"].dtype)
        combined_mask = torch.cat([soft_mask, tgt_inputs["attention_mask"]], dim=1)

        # Labels: -100 for soft token positions (don't compute loss on prefix)
        prefix_labels = torch.full((1, args.num_soft_tokens), -100, device=device, dtype=torch.long)
        real_labels = tgt_inputs["input_ids"].clone()
        combined_labels = torch.cat([prefix_labels, real_labels], dim=1)

        # Forward with embeddings (bypass tokenizer/embedding layer)
        outputs = target_model(
            inputs_embeds=combined_embeds,
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
    torch.save(soft_prompt.state_dict(), out / "soft_prompt.pt")
    target_model.save_pretrained(str(out / "target_lora"))
    target_tok.save_pretrained(str(out / "target_lora"))

    meta = {
        "version": "v9",
        "architecture": "soft_prompt",
        "models": [args.source, args.target],
        "substrate_dim": args.substrate_dim,
        "num_soft_tokens": args.num_soft_tokens,
        "steps": args.steps,
        "learning_rate": args.lr,
        "corpus_size": len(corpus),
        "final_loss": losses[-1] if losses else None,
        "best_loss": best_loss,
        "best_step": best_step,
        "training_time_seconds": elapsed,
        "total_trainable": total,
        "losses_history": losses[-100:],
    }
    (out / "training_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
